import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class EmbedFC(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, learned_ordinal: bool = False) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.GELU() if not learned_ordinal else nn.Identity(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(-1, self.in_dim))


class VesselConditionEncoder(nn.Module):
    def __init__(self, out_dim: int, token_grid_size: int):
        super().__init__()
        self.token_grid_size = token_grid_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, out_dim, 3, 1, 1),
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
        )

    def forward(self, vessel_mask: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(vessel_mask)
        feat = F.adaptive_avg_pool2d(feat, (self.token_grid_size, self.token_grid_size))
        return feat.flatten(2).transpose(1, 2)


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        cross_out, _ = self.cross_attn(self.norm_q(x), self.norm_kv(cond_tokens), self.norm_kv(cond_tokens))
        return x + self.gate(c).unsqueeze(1) * cross_out


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_vessel_condition=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.use_vessel_condition = use_vessel_condition
        if use_vessel_condition:
            self.cross_attn = CrossAttentionBlock(hidden_size, num_heads)

    def forward(self, x, c, vessel_tokens=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_vessel_condition and vessel_tokens is not None:
            x = self.cross_attn(x, vessel_tokens, c)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(
        self,
        input_size=256,
        patch_size=2,
        in_channels=3,
        hidden_size=512,
        depth=8,
        num_heads=4,
        mlp_ratio=4.0,
        learn_sigma=True,
        ordinal_input=True,
        learned_ordinal_input=True,
        use_structure=True,
        use_vessel_condition=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.ordinal_input = ordinal_input
        self.use_structure = use_structure
        self.learned_ordinal_input = learned_ordinal_input
        self.use_vessel_condition = use_vessel_condition
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if learned_ordinal_input:
            self.v = nn.Embedding(5, 1)
        if ordinal_input or learned_ordinal_input:
            self.y_embedder = EmbedFC(1, hidden_size, learned_ordinal=learned_ordinal_input)
        else:
            self.y_embedder = EmbedFC(5, hidden_size)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_vessel_condition=use_vessel_condition)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        if self.use_vessel_condition:
            token_grid_size = input_size // patch_size
            self.vessel_encoder = VesselConditionEncoder(hidden_size, token_grid_size)
        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            if self.use_vessel_condition:
                nn.init.constant_(block.cross_attn.gate[-1].weight, 0)
                nn.init.constant_(block.cross_attn.gate[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(
        self,
        x,
        labels,
        t,
        label_mask=None,
        s=None,
        structure_mask=None,
        vessel_mask=None,
        vessel_cond_mask=None,
    ):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        if self.ordinal_input:
            y_input = labels.float()
        elif self.learned_ordinal_input:
            y_input = labels.float()
        else:
            y_input = nn.functional.one_hot(labels.long(), num_classes=5).float()
        y = self.y_embedder(y_input)
        if label_mask is not None:
            y = y * (1 - label_mask.to(torch.float32).view(-1, 1))
        #c = self.cond_proj(torch.cat([t, y], dim=1))
        c= t + y

        vessel_tokens = None
        if self.use_vessel_condition and vessel_mask is not None:
            vessel_mask = vessel_mask.to(dtype=x.dtype)
            if vessel_cond_mask is not None:
                vessel_mask = vessel_mask * (1 - vessel_cond_mask.to(torch.float32).view(-1, 1, 1, 1))
            vessel_tokens = self.vessel_encoder(vessel_mask)

        for block in self.blocks:
            x = block(x, c, vessel_tokens=vessel_tokens)
        x = self.final_layer(x, c)
        return self.unpatchify(x)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)
