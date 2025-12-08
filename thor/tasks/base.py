import math

import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader

from thor.core.dataset_registry import DATASETS
from thor.core.model_registry import MODELS
from thor.core.task_registry import TASKS


@TASKS.register()
class BasePretrainTask(pl.LightningModule):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        self.training_cfg = self.hparams["training"]
        self.batch_size = self.training_cfg["batch_size"]

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_nb):
        raise NotImplementedError

    def validation_step(self, batch, batch_nb):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer_name = self.training_cfg.get("optimizer", "Adam")
        optimizer_cfg = self.training_cfg.get("optimizer_cfg", {"lr": 0.001})

        if "lr" not in optimizer_cfg:
            msg = "You must provide learning rate in optimizer cfg"
            raise KeyError(msg)

        if optimizer_name == "Adam":
            optimizer_class = optim.Adam
        elif optimizer_name == "SGD":
            optimizer_class = optim.SGD
        elif optimizer_name == "AdamW":
            optimizer_class = optim.AdamW
        else:
            msg = f"{optimizer_name} is not supported, add it to configure_optimizers in base lightning class."
            raise ValueError(msg)

        optimizer = optimizer_class(self.parameters(), **optimizer_cfg)

        return optimizer

    def train_dataloader(self):
        dataset = DATASETS.build(self.hparams["dataset"], split="train")
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        data_loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.training_cfg["train_loader_worker"],
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return data_loader

    def val_dataloader(self):
        dataset = DATASETS.build(self.hparams["dataset"], split="eval")
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        data_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.training_cfg["eval_loader_worker"],
            collate_fn=collate_fn,
        )
        return data_loader

    def test_dataloader(self):
        dataset = DATASETS.build(self.hparams["dataset"], split="test")
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        data_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.training_cfg["eval_loader_worker"],
            collate_fn=collate_fn,
        )
        return data_loader


@TASKS.register()
class BaseDownstreamTask(BasePretrainTask):
    def __init__(self, params):
        super().__init__(params=params)

        models = MODELS.build(self.hparams["models"])
        self.encoder = models["encoder"]
        self.head = models["head"]
        gpus = params["training"]["trainer"]["gpus"]
        self.num_gpu = gpus if isinstance(gpus, int) else len(gpus)
        self.epoch_progress = 1

    def train_dataloader(self):
        """overwrite to calculate epoch progress"""
        data_loader = super().train_dataloader()
        self.num_steps_per_train_epoch = math.ceil(len(data_loader) / self.num_gpu)
        return data_loader

    def test_dataloader(self):
        """overwrite to calculate epoch progress"""
        data_loader = super().test_dataloader()
        self.num_steps_per_train_epoch = math.ceil(len(data_loader) / self.num_gpu)
        return data_loader

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = super().configure_optimizers()
        self.scheduler_name = self.training_cfg.get("scheduler", None)
        scheduler_cfg = self.training_cfg.get("scheduler_cfg", {})

        # Only ReduceLROnPlateau operates on epoch interval
        if self.scheduler_name == "Plateau":
            monitor_metric = scheduler_cfg.pop("monitor_metric", "Val/epoch_avg_loss")
            scheduler_class = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_cfg)
            return [optimizer], [{"scheduler": scheduler_class, "monitor": monitor_metric}]
        elif self.scheduler_name == "MultiStep":
            scheduler_class = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_cfg)
            return [optimizer], [{"scheduler": scheduler_class, "interval": "epoch"}]
        # Manually schedule for Cosine and Polynomial scheduler
        elif self.scheduler_name in [None, "Cosine", "Poly"]:
            return optimizer
        else:
            msg = f"{self.scheduler_name} is not supported, add it to configure_optimizers in BaseDownstreamTask."
            raise ValueError(msg)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Overwrite with warmup epoch and manual lr decay"""

        self.epoch_progress = self.current_epoch + min((batch_idx + 1) / self.num_steps_per_train_epoch, 1)
        initial_lr = self.training_cfg["optimizer_cfg"]["lr"]
        warm_up_epoch = self.training_cfg.get("warm_up_epoch", 0)
        max_epochs = self.training_cfg["trainer"]["max_epochs"]

        if self.scheduler_name in ["Cosine", "Poly"] or self.epoch_progress <= warm_up_epoch:
            if self.epoch_progress <= warm_up_epoch:
                lr = initial_lr * self.epoch_progress / warm_up_epoch
            elif self.scheduler_name == "Cosine":
                lr = (
                    initial_lr
                    * 0.5
                    * (1.0 + math.cos(math.pi * (self.epoch_progress - warm_up_epoch) / (max_epochs - warm_up_epoch)))
                )
            else:
                power = self.training_cfg.get("scheduler_cfg", {}).get("power", 0.5)
                lr = initial_lr * (1.0 - (self.epoch_progress - warm_up_epoch) / (max_epochs - warm_up_epoch)) ** power
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        optimizer.step(closure=optimizer_closure)
