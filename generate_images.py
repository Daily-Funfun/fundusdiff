import os
import argparse
from tqdm import tqdm
import torch
from torchvision.utils import save_image

from sample import Sampler
from types import SimpleNamespace

cfg = SimpleNamespace(data=SimpleNamespace(image_resolution=256, batch_size=64, num_workers=4, name='EP', diffusion_transform=False, use_imbalanced_sampler=False, encoder_transform=False, n_val = 10),
                      model=SimpleNamespace(name="DiT"),
                      seed=42
                      )


def denormalize(img,  mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
                      std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242]):
    mean = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, -1, 1, 1)
    return img * std + mean

def save_images_by_class(loader, save_dir, num_images_per_class, mean, std):
    os.makedirs(save_dir, exist_ok=True)
    class_counts = {}

    for imgs, labels, _ in loader:
        if mean is not None and std is not None:
            imgs = denormalize(imgs, mean, std)

        for img, label in zip(imgs, labels):
            label = int(label.item())
            class_dir = os.path.join(save_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)

            if label not in class_counts:
                class_counts[label] = 0

            if class_counts[label] >= num_images_per_class:
                continue

            save_path = os.path.join(class_dir, f"{class_counts[label]}.png")
            save_image(img, save_path)
            class_counts[label] += 1

        if all(v >= num_images_per_class for v in class_counts.values()):
            break

def save_generated_images_by_class(samples, labels, save_dir):
    """
    samples: Tensor [N, C, H, W] in [0,1]
    labels: Tensor [N]
    """
    os.makedirs(save_dir, exist_ok=True)
    class_counts = {}

    for img, label in zip(samples, labels):
        label = label.item()
        class_dir = os.path.join(save_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        if label not in class_counts:
            # count existing images in folder to avoid overwriting
            class_counts[label] = len(os.listdir(class_dir))

        save_path = os.path.join(class_dir, f"{class_counts[label]}.png")
        save_image(img, save_path)
        class_counts[label] += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="basestruct")
    parser.add_argument("--num_images_per_class", type=int, default=100)
    parser.add_argument("--out_dir", default="out/")
    parser.add_argument("--guide_w", type=float, default=4.0)
    args = parser.parse_args()
    guide_w = args.guide_w

    exp_path = f"model_weights/diffusion/{args.model}"
    save_path = exp_path + f"/samples_gw{args.guide_w}"

    sampler = Sampler(exp_path, guide_w=args.guide_w)

    semantic_levels = [0, 1, 2, 3, 4] 
    batch_size = 6
    sample_method = "ddim"

    dr_levels = [0, 1, 2, 3, 4] 
    
    assert len(dr_levels) == len(semantic_levels)

    gen_save_dir = f"{args.out_dir}/generated_{args.model}_gw{guide_w}"
    if len(dr_levels) > 20:
        gen_save_dir = f"{args.out_dir}/generated_{args.model}_gw{guide_w}_interpol"
    os.makedirs(gen_save_dir, exist_ok=True)

    for idx, cls_value in enumerate(dr_levels):
        cond_labels = torch.tensor(
            [cls_value] * args.num_images_per_class, dtype=torch.float
        )
        class_labels = torch.tensor(
            [semantic_levels[idx]] * args.num_images_per_class, dtype=torch.float
        )

        iqs = torch.full((args.num_images_per_class,), 0.8)

        pbar = tqdm(total=args.num_images_per_class, desc=f"Class {dr_levels[idx]}")

        for i in range(0, args.num_images_per_class, batch_size):
            batch_cond = cond_labels[i:i+batch_size]
            batch_class = class_labels[i:i+batch_size]
            batch_iqs = iqs[i:i+batch_size]

            batch_samples = sampler.sample(
                batch_cond,
                batch_iqs,
                guide_w=guide_w,
                sample_method=sample_method,
                batch_size=batch_size
            )

            save_generated_images_by_class(
                batch_samples,
                batch_class,
                gen_save_dir
            )

            pbar.update(len(batch_samples))

            del batch_samples
            torch.cuda.empty_cache()

    print(f"Generated images saved to {gen_save_dir}")