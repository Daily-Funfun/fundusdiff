import copy
import math
import os

import lightning.pytorch as pl
import numpy as np
import torch
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance

from .criterion import get_criterion
from .data import get_dataloader
from .metrics import get_metrics
from .models import VAE, get_model
from .optimizers import get_optimizer, get_scheduler


def denormalize(
    img,
    mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
    std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242],
):
    mean = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, -1, 1, 1)
    return img * std + mean


class ModelEMA:
    def __init__(self, model, decay=0.9999, update_after_step=100):
        self.decay = decay
        self.update_after_step = update_after_step
        self.num_updates = 0
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def to(self, device):
        self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        if self.num_updates <= self.update_after_step:
            self.ema_model.load_state_dict(model.state_dict())
            return

        model_state = model.state_dict()
        for key, value in self.ema_model.state_dict().items():
            source = model_state[key].detach()
            if not torch.is_floating_point(value):
                value.copy_(source)
            else:
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)


class NoiseScheduler(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        scheduler_cfg = cfg.noise_scheduler
        self.timesteps = int(scheduler_cfg.num_timesteps)
        self.schedule_type = str(
            getattr(scheduler_cfg, "schedule_type", getattr(scheduler_cfg, "name", "linear"))
        ).lower()

        if self.schedule_type == "cosine":
            betas = self._cosine_beta_schedule(
                self.timesteps,
                s=float(getattr(scheduler_cfg, "cosine_s", 0.008)),
            )
        elif self.schedule_type == "linear":
            betas = torch.linspace(
                float(scheduler_cfg.beta_start),
                float(scheduler_cfg.beta_end),
                self.timesteps,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported noise schedule: {self.schedule_type}")

        betas = betas.clamp(1e-8, 0.999)
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hats", alpha_hats)

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        return 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    def q_sample(self, x_start, t, noise):
        sqrt_alpha_hat = self.alpha_hats[t] ** 0.5
        sqrt_one_minus_alpha_hat = (1 - self.alpha_hats[t]) ** 0.5
        return sqrt_alpha_hat[:, None, None, None] * x_start + sqrt_one_minus_alpha_hat[:, None, None, None] * noise


class LightningWrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.val_loader = get_dataloader(cfg, mode="val") #4.23
        # self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)

        self.noise_scheduler = NoiseScheduler(cfg)
        self.T = self.noise_scheduler.timesteps
        self.model = get_model(cfg)
        self.v = []
        self.criterion = get_criterion(cfg)
        self.vae = self._load_pretrained_vae(cfg)

        ema_cfg = cfg.training.get("ema", {})
        self.ema = None
        if bool(ema_cfg.get("enabled", False)):
            self.ema = ModelEMA(
                self.model,
                decay=float(ema_cfg.get("decay", 0.9999)),
                update_after_step=int(ema_cfg.get("update_after_step", 100)),
            )

        self.latent_resolution = cfg.data.image_resolution // 4
        self.latent_channels = 4
        self.image_resolution = cfg.data.image_resolution
        self.optimizer_name = cfg.optimizer.name
        self.optimizer_kwargs = cfg.optimizer.kwargs
        self.scheduler_name = cfg.scheduler.name
        self.scheduler_kwargs = cfg.scheduler.kwargs
        self.modes = ["train", "val"]
        self.losses = {mode: {"loss": []} for mode in self.modes}
        self.metrics = torch.nn.ModuleList([get_metrics(cfg) for mode in self.modes])
        self.mode_to_metrics = {
            mode: metric for mode, metric in zip(self.modes, self.metrics)
        }

    def _load_pretrained_vae(self, cfg):
        vae_checkpoint_path = cfg.vae.checkpoint
        if not os.path.exists(vae_checkpoint_path):
            raise FileNotFoundError(f"VAE checkpoint not found at {vae_checkpoint_path}")

        vae = VAE(in_channels=3, latent_channels=4, ch_mult=[1, 2, 4])
        checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
        vae.load_state_dict(checkpoint)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        return vae

    def configure_optimizers(self):
        param_groups = []
        base_weight_decay = float(self.optimizer_kwargs.get("weight_decay", 0.0))

        if self.cfg.model.learned_ordinal_input and hasattr(self.model, "v"):
            v_params = [self.model.v.weight]
            other_params = [param for name, param in self.model.named_parameters() if name != "v.weight"]
            param_groups.append(
                {"params": other_params, "lr": self.optimizer_kwargs.get("lr"), "weight_decay": base_weight_decay}
            )
            param_groups.append(
                {"params": v_params, "lr": self.optimizer_kwargs.get("v_lr", 1e-3), "weight_decay": 0.0}
            )
        else:
            param_groups.append(
                {
                    "params": list(self.model.parameters()),
                    "lr": self.optimizer_kwargs.get("lr"),
                    "weight_decay": base_weight_decay,
                }
            )

        optimizer_class = get_optimizer(self.optimizer_name)
        optimizer_kwargs_clean = {
            k: v for k, v in self.optimizer_kwargs.items() if k not in ["lr", "v_lr", "weight_decay"]
        }
        optimizer = optimizer_class(param_groups, **optimizer_kwargs_clean)

        if self.scheduler_name is None:
            return optimizer

        scheduler_class = get_scheduler(self.scheduler_name)
        scheduler = scheduler_class(optimizer, **self.scheduler_kwargs)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        if self.ema is not None:
            self.ema.to(self.device)

    def on_train_epoch_start(self):
        self.__reset_metrics("train")

    def on_validation_epoch_start(self):
        self.__reset_metrics("val")

    def __reset_metrics(self, mode):
        self.losses[mode] = {"loss": []}
        self.mode_to_metrics[mode].reset()

    def training_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, "val")

    def __step(self, batch, batch_idx, mode):
        if self.cfg.model.use_vessel_condition:
            imgs, labels, iq, vessel_mask = batch
            vessel_mask = vessel_mask.to(imgs.device)
        else:
            imgs, labels, iq = batch
            vessel_mask = None

        with torch.no_grad():
            mu, log_var = self.vae.encoder(imgs)
            latents = mu

        if self.cfg.training.criterion == "ODloss":
            t = torch.randint(0, self.noise_scheduler.timesteps, (1,), device=imgs.device).expand(imgs.size(0))
        else:
            t = torch.randint(0, self.noise_scheduler.timesteps, (latents.size(0),), device=latents.device)

        noise = torch.randn_like(latents)
        x_t = self.noise_scheduler.q_sample(latents, t, noise)
        label_mask = torch.bernoulli(torch.zeros_like(labels) + 0.1)
        structure_mask = None
        vessel_cond_mask = torch.bernoulli(torch.zeros_like(labels) + 0.1) if vessel_mask is not None else None

        n, c, h, w = latents.shape
        assert c == self.latent_channels, f"Expected {self.latent_channels} latent channels, got {c}"
        assert h == w == self.latent_resolution, f"Expected {self.latent_resolution}x{self.latent_resolution} latent resolution, got {h}x{w}"

        noise_pred = self.model(
            x_t,
            labels,
            t,
            label_mask,
            None,
            structure_mask,
            vessel_mask=vessel_mask,
            vessel_cond_mask=vessel_cond_mask,
        )

        if self.model.learn_sigma:
            noise_pred, _ = noise_pred.chunk(2, dim=1)

        if self.cfg.training.criterion == "ODloss":
            loss = self.criterion(noise_pred, noise, labels, t, ordinal=True)
        else:
            loss = self.criterion(noise_pred, noise)

        if self.cfg.model.learned_ordinal_input:
            loss += 0.0001 * (1 / ((self.model.v.weight) ** 2 + 1e-8)).mean()

        if torch.isnan(loss.cpu()):
            raise Exception("loss is Nan.")

        loss_value = loss.detach()
        self.losses[mode]["loss"].append(loss_value.cpu().item())
        self.log(
            f"{mode} loss step",
            loss_value,
            on_step=True,
            on_epoch=False,
            sync_dist=(mode == "val"),
            batch_size=imgs.size(0),
        )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.cfg.model.learned_ordinal_input:
            self.v.append(self.model.v.weight.detach().cpu().clone())
        if self.ema is not None:
            self.ema.update(self.model)

    def on_train_epoch_end(self):
        self.__log_metrics("train")
        log_dir = getattr(self.logger, "log_dir", self.cfg.training.out_dir)
        os.makedirs(log_dir, exist_ok=True)

        if self.cfg.model.learned_ordinal_input and self.v:
            torch.save(torch.stack(self.v, dim=0).cpu(), os.path.join(log_dir, "v.pt"))

        if self.ema is not None:
            torch.save(self.ema.ema_model.state_dict(), os.path.join(log_dir, "ema_last.pt"))

    def on_validation_epoch_end(self):
        self.__log_metrics("val")

    def __log_metrics(self, mode):
        metrics_out = self.mode_to_metrics[mode].get_out_dict()
        logs = {f"{mode} {key}": val for key, val in metrics_out.items()}
        logs[f"{mode} loss"] = np.mean(self.losses[mode]["loss"]) if self.losses[mode]["loss"] else 0.0
        logs["step"] = float(self.current_epoch)
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def generate_samples(self, num_samples, device=None, ddim_steps=50, eta=0.0):
        x_i = torch.randn((num_samples, 4, 64, 64), device=device)
        labels = torch.zeros((num_samples,), dtype=torch.long).to(device)
        context_mask = torch.ones_like(torch.arange(num_samples), dtype=torch.float).to(device)
        sample_model = self.ema.ema_model if self.ema is not None else self.model
        T = self.noise_scheduler.timesteps - 1

        with torch.no_grad():
            ddim_timesteps = torch.linspace(0, T, 50, device=device).long()
            for i in reversed(range(0, 50)):
                t = ddim_timesteps[i]
                t_prev = ddim_timesteps[i - 1] if i > 0 else 0
                t_tensor = torch.full((len(labels),), t, device=device)
                predicted_noise = sample_model(
                    x_i,
                    labels,
                    t_tensor,
                    context_mask,
                    None,
                    None,
                    vessel_mask=None,
                    vessel_cond_mask=None,
                )
                predicted_noise = predicted_noise[:, :4, :, :]
                alpha_hat_t = self.noise_scheduler.alpha_hats[t]
                alpha_hat_t_prev = self.noise_scheduler.alpha_hats[t_prev] if t_prev > 0 else torch.tensor(1.0).to(device)
                x_0_hat = (x_i - (1 - alpha_hat_t).sqrt() * predicted_noise) / alpha_hat_t.sqrt()
                x_i = alpha_hat_t_prev.sqrt() * x_0_hat + (1 - alpha_hat_t_prev).sqrt() * predicted_noise

        with torch.no_grad():
            return self.vae.decoder(x_i)

    @torch.no_grad()
    def evaluate_fid_and_save_best(self):
        return
