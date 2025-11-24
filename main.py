import os
import uuid
from datetime import timedelta
from pathlib import Path

import fire
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from thor.core.serialization import read_yaml
from thor.core.task_registry import TASKS
from thor.utils.constants import WANDB_ENTITY, WANDB_PROJECT


def train(cfg):
    gpus, nodes, job_id = get_lumi_config()
    cfg = read_yaml(cfg)

    trainer_cfg = cfg["training"]["trainer"]

    random_seed = cfg.get("random_seed")
    if random_seed:
        pl.seed_everything(random_seed, workers=True)

    # GPU
    if gpus and nodes:
        cfg["training"]["trainer"]["gpus"] = gpus * nodes
        cfg["training"]["trainer"]["nodes"] = nodes
    else:
        gpus = trainer_cfg.get("gpus", 1)
        nodes = trainer_cfg.get("num_nodes", 1)

    strategy = trainer_cfg.get("strategy", "auto")
    unused_params = trainer_cfg.get("unused_params", False)

    task = TASKS.build(cfg)

    num_gpu = gpus * nodes if isinstance(gpus, int) else len(gpus)
    if gpus == -1 or num_gpu >= 1:
        if strategy == "ddp":
            strategy = DDPStrategy(
                find_unused_parameters=unused_params,
                timeout=timedelta(milliseconds=5000000),
            )

    print("Strategy", strategy)

    # Precision
    mixed_precision = cfg.get("mixed_precision", False)
    if mixed_precision:
        precision = "bf16-mixed"
    else:
        precision = "32-true"

    # Logging
    save_dir = trainer_cfg.get("save_dir", "results/default")
    exp_name = cfg.get("exp_name", f"exp_{str(uuid.uuid1())[:8]}")
    log_name = exp_name
    version = cfg.get("version", 0)
    ckpt_dir = Path(save_dir) / exp_name / f"version_{version}" / "ckpt"
    wandb_entity = cfg.get("w&b_entity", WANDB_ENTITY)
    wandb_project = cfg.get("w&b_project", WANDB_PROJECT)
    logger = WandbLogger(
        name=f"{log_name}_v{version}", save_dir=save_dir, project=wandb_project, entity=wandb_entity, id=job_id
    )

    # Callbacks - Checkpointing and early stop
    save_top_k = trainer_cfg.get("save_top_k", -1)
    monitor_metric = trainer_cfg.get("monitor_metric", "train_loss_epoch")
    monitor_mode = trainer_cfg.get("monitor_mode", "min")
    every_n_epochs = trainer_cfg.get("every_n_epochs", 10)
    patience = trainer_cfg.get("patience", 10)

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=save_top_k,
        verbose=True,
        monitor=monitor_metric,
        mode=monitor_mode,
        every_n_epochs=every_n_epochs,
        save_last=True,
    )
    earlystop_cb = EarlyStopping(monitor=monitor_metric, patience=patience, verbose=True, mode=monitor_mode)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer config
    gradient_clip_val = trainer_cfg.get("gradient_clip_val", 0)
    limit_train_batches = trainer_cfg.get("limit_train_batches", 1.0)
    enable_model_summary = trainer_cfg.get("enable_model_summary", False)
    max_epochs = trainer_cfg.get("max_epochs", 100)
    accumulated_batches = trainer_cfg.get("accumulated_batches", 1)
    resume_checkpoint = trainer_cfg.get("resume_ckpt", None)

    trainer = Trainer(
        accelerator="gpu",
        precision=precision,
        devices=gpus,
        num_nodes=nodes,
        strategy=strategy,
        logger=logger,
        callbacks=[
            ckpt_cb,
            earlystop_cb,
            lr_monitor,
        ],
        gradient_clip_val=gradient_clip_val,
        limit_train_batches=limit_train_batches,
        enable_model_summary=enable_model_summary,
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulated_batches,
        limit_val_batches=0.0 if "pretrain" in cfg["task"].lower() else 1.0,
        log_every_n_steps=5,
        detect_anomaly=False,
        use_distributed_sampler="iterable"
        not in cfg["dataset"]["name"].lower(),  # Turn off when we use iterable dataset
    )

    if resume_checkpoint:
        trainer.fit(task, ckpt_path=resume_checkpoint)
    else:
        trainer.fit(task)


def test(cfg):
    gpus, nodes, job_id = get_lumi_config()
    cfg = read_yaml(cfg)
    trainer_cfg = cfg["training"]["trainer"]

    random_seed = cfg.get("random_seed")
    if random_seed:
        pl.seed_everything(random_seed)

    # GPU
    if gpus and nodes:
        cfg["training"]["trainer"]["gpus"] = gpus * nodes
        cfg["training"]["trainer"]["nodes"] = nodes
    else:
        gpus = trainer_cfg.get("gpus", 1)
        nodes = trainer_cfg.get("num_nodes", 1)

    # GPU
    strategy = trainer_cfg.get("strategy", "auto")
    unused_params = trainer_cfg.get("unused_params", False)

    # get task
    task = TASKS.build(cfg)

    num_gpu = gpus * nodes if isinstance(gpus, int) else len(gpus)
    if gpus == -1 or num_gpu > 1:
        if strategy == "ddp":
            strategy = DDPStrategy(find_unused_parameters=unused_params)

    print("strategy", strategy)

    # Precision
    mixed_precision = cfg.get("mixed_precision", False)
    if mixed_precision:
        precision = "bf16-mixed"
    else:
        precision = "32-true"

    # Logging
    save_dir = trainer_cfg.get("save_dir", "results/default")
    exp_name = cfg.get("exp_name", f"exp_{str(uuid.uuid1())[:8]}")
    version = cfg.get("version", 0)
    wandb_entity = cfg.get("w&b_entity", WANDB_ENTITY)
    wandb_project = cfg.get("w&b_project", WANDB_PROJECT)

    logger = WandbLogger(name=f"{exp_name}_v{version}", save_dir=save_dir, project=wandb_project, entity=wandb_entity)

    # Trainer config
    gradient_clip_val = trainer_cfg.get("gradient_clip_val", 0)
    limit_train_batches = trainer_cfg.get("limit_train_batches", 1.0)
    enable_model_summary = trainer_cfg.get("enable_model_summary", False)
    max_epochs = trainer_cfg.get("max_epochs", 100)
    accumulated_batches = trainer_cfg.get("accumulated_batches", 1)
    test_checkpoint = trainer_cfg.get("test_ckpt", None)

    trainer = Trainer(
        accelerator="gpu",
        precision=precision,
        devices=gpus,
        num_nodes=nodes,
        strategy=strategy,
        logger=logger,
        gradient_clip_val=gradient_clip_val,
        limit_train_batches=limit_train_batches,
        enable_model_summary=enable_model_summary,
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulated_batches,
        log_every_n_steps=5,
    )

    trainer.test(task, ckpt_path=test_checkpoint)


def get_lumi_config():
    gpus = int(os.environ.get("GPUS", 0))
    nodes = int(os.environ.get("NODES", 1))
    job_id = os.environ.get("SLURM_JOB_ID", None)
    print("NODES requested on LUMI:", nodes)
    print("GPUS requested on each node on LUMI:", gpus)
    print("Slurm job id:", job_id)
    return gpus, nodes, job_id


if __name__ == "__main__":
    fire.Fire()
