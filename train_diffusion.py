import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import OmegaConf

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except ImportError as exc:
    raise ImportError(
        "swanlab is required for logging. Please install it with `pip install swanlab`."
    ) from exc

from ori.config import get_config
from ori.data import get_dataloader
from ori.training import LightningWrapper
from ori.utils import LitProgressBar, TorchScriptModelCheckpoint

PATH_TO_DEFAULT_CFG = "configs/diffusion.yaml"


def _should_use_ddp(devices):
    if isinstance(devices, int):
        return devices > 1
    if isinstance(devices, (list, tuple)):
        return len(devices) > 1
    return False


def main(cfg):
    cfg = OmegaConf.create(cfg)
    module = LightningWrapper(cfg)
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    try:
        os.mkdir(cfg.training.out_dir)
    except:
        pass

    swanlab_cfg = cfg.training.get("swanlab", {})
    logger = SwanLabLogger(
        project=swanlab_cfg.get("project", "OrdinalDiffusion"),
        workspace=swanlab_cfg.get("workspace"),
        experiment_name=swanlab_cfg.get("name", "diffusion"),
        description=swanlab_cfg.get("description"),
        tags=swanlab_cfg.get("tags"),
        mode=swanlab_cfg.get("mode"),
        save_dir=cfg.training.out_dir,
        logdir=cfg.training.out_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=(
            pl.strategies.DDPStrategy(find_unused_parameters=True)
            if _should_use_ddp(cfg.devices)
            else "auto"
        ),
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=[
            TorchScriptModelCheckpoint(
                save_top_k=cfg.training.checkpoints.save_top_k,
                monitor=cfg.training.checkpoints.monitor,
                mode=cfg.training.checkpoints.mode,
                filename=cfg.training.checkpoints.filename,
            ),
            *([
                EarlyStopping(
                    monitor=cfg.training.early_stopping.monitor,
                    mode=cfg.training.early_stopping.mode,
                    patience=int(cfg.training.early_stopping.patience),
                    min_delta=float(cfg.training.early_stopping.min_delta),
                )
            ] if bool(cfg.training.early_stopping.enabled) else []),
        ],
        default_root_dir=cfg.training.out_dir,
        log_every_n_steps=1,
        val_check_interval=None,
        check_val_every_n_epoch=1,
        precision=cfg.training.precision,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        enable_progress_bar=False
    )
    trainer.fit(
        module,
        train_dataloaders=get_dataloader(cfg, mode="train"),
        val_dataloaders=get_dataloader(cfg, mode="val")
                )


if __name__ == "__main__":
    cfg = get_config(PATH_TO_DEFAULT_CFG)
    main(cfg)
