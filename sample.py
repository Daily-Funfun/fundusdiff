import os
from omegaconf import OmegaConf

import torch

from ori.models import get_model
from ori.models.autoencoder import VAE
from ori.training import NoiseScheduler


def denormalize(img,  mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
                      std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242]):
    mean = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, -1, 1, 1)
    return img * std + mean


class Sampler:
  def __init__(self, exp_path, guide_w):
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      self.cfg = OmegaConf.load(exp_path + "/hparams.yaml").cfg

      self.model = get_model(self.cfg)
      checkpoint_dir = os.path.join(exp_path, "checkpoints")
      checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

      self.vae = VAE(in_channels=3, latent_channels=4, ch_mult=[1, 2, 4])
      self.vae.load_state_dict(torch.load(self.cfg.vae.checkpoint, map_location=self.device))
      self.vae.to(self.device)
      self.vae.eval()

      if not checkpoint_files:
          raise FileNotFoundError(f"No .pt checkpoint found in {checkpoint_dir}")
      checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
      self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)
      self.model.to(self.device)
      self.model.eval()

      self.output_dir = exp_path + f"/samples_gw{guide_w}"
      if not os.path.exists(self.output_dir):
          os.makedirs(self.output_dir)

  def sample(self, labels, iqs, guide_w, sample_method, batch_size, vessel_masks=None):
    all_samples = []

    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i:i+batch_size]
        batch_iqs = iqs[i:i+batch_size]
        batch_vessel_masks = vessel_masks[i:i+batch_size] if vessel_masks is not None else None
        batch_samples = self._sample_batch(batch_labels, batch_iqs, guide_w, sample_method, vessel_masks=batch_vessel_masks)
        all_samples.append(batch_samples)
        
        torch.cuda.empty_cache()
    
    return torch.cat(all_samples, dim=0)

  def _sample_batch(self, labels, iqs, guide_w, sample_method="ddpm", vessel_masks=None):
    x_i = torch.randn((len(labels), 4, self.cfg.data.image_resolution // 4, self.cfg.data.image_resolution // 4), device=self.device)
    labels = labels.to(self.device)
    labels_in = labels.repeat(2)

    context_mask = torch.zeros_like(labels, dtype=torch.float).to(self.device)
    context_mask = context_mask.repeat(2)
    context_mask[labels.size(0):] = 1

    structure = None
    structure_mask = None
    vessel_cond_mask = torch.ones_like(context_mask, dtype=torch.float).to(self.device)
    vessel_masks_in = None
    if vessel_masks is not None:
      vessel_masks = vessel_masks.to(self.device)
      vessel_masks_in = vessel_masks.repeat(2, 1, 1, 1)
      vessel_cond_mask[:labels.size(0)] = 0

    NS = NoiseScheduler(self.cfg)
    NS.to(self.device) 
    T = NS.timesteps-1

    start_w = 0.5

    with torch.no_grad():
      if sample_method == "ddpm":
        for t in reversed(range(T)):
          t_tensor = torch.full((len(labels),), t, device=self.device)
          t_in = t_tensor.repeat(2)

          x_in = x_i.repeat(2, 1, 1, 1)

          predicted_noise = self.model(
              x_in,
              labels_in,
              t_in,
              context_mask,
              structure,
              structure_mask,
              vessel_mask=vessel_masks_in,
              vessel_cond_mask=vessel_cond_mask,
          )

          if self.cfg.model.name == "DiT":
             predicted_noise = predicted_noise[:, :4, :, :]
          eps_cond, eps_uncond = predicted_noise.chunk(2, dim=0)

          eps_guided = eps_uncond + guide_w * (eps_cond - eps_uncond)

          alpha_t = NS.alphas[t]
          alpha_hat_t = NS.alpha_hats[t]
          beta_t = NS.betas[t]

          x_i = (1/alpha_t.sqrt()) * (x_i - beta_t/(1-alpha_hat_t).sqrt() * eps_guided)

          if t > 0:
            temp = 1
            noise = torch.randn_like(x_i) * temp
            x_i += beta_t.sqrt() * noise
      elif sample_method == "ddim":
        ddim_steps = 100
        ddim_timesteps = torch.linspace(0, T, ddim_steps, device=self.device).long()
        for i in reversed(range(len(ddim_timesteps))):
          t = ddim_timesteps[i]
          t_prev = ddim_timesteps[i - 1] if i > 0 else 0

          t_tensor = torch.full((len(labels),), t, device=self.device)
          t_in = t_tensor.repeat(2)

          x_in = x_i.repeat(2, 1, 1, 1)

          predicted_noise = self.model(
              x_in,
              labels_in,
              t_in,
              context_mask,
              structure,
              structure_mask,
              vessel_mask=vessel_masks_in,
              vessel_cond_mask=vessel_cond_mask,
          )
          if self.cfg.model.name == "DiT":
             predicted_noise = predicted_noise[:, :4, :, :]

          eps_cond, eps_uncond = predicted_noise.chunk(2, dim=0)

          progress = t/T
  
          guide_w_t = start_w + progress * (guide_w - start_w)
          eps_guided = eps_uncond + guide_w_t * (eps_cond - eps_uncond)

          alpha_hat_t = NS.alpha_hats[t]
          alpha_hat_t_prev = NS.alpha_hats[t_prev] if t_prev > 0 else torch.tensor(1.0).to(self.device)

          x_0_hat = (x_i - (1 - alpha_hat_t).sqrt() * eps_guided) / alpha_hat_t.sqrt()
          x_i = alpha_hat_t_prev.sqrt() * x_0_hat + (1 - alpha_hat_t_prev).sqrt() * eps_guided
    
    with torch.no_grad():
      x_i = self.vae.decoder(x_i)

    x_i = denormalize(x_i)
    return x_i