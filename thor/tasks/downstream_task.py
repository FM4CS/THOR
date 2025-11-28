from typing import Any

import torch
import torchmetrics
import torchmetrics.classification as tmc
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch import nn

from thor.core.task_registry import TASKS
from thor.tasks.base import BaseDownstreamTask
from thor.utils.constants import FINETUNING, PROBING, TRAIN_METHODS
from thor.utils.helper import safe_copy


@TASKS.register()
class MulticlassClassification(BaseDownstreamTask):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        # mixup config
        self.mixup_fn = None
        mix_cfg = self.training_cfg.get("mix_cfg", {})
        if mix_cfg.get("mixup", 0) > 0 or mix_cfg.get("cutmix", 0) > 0 or mix_cfg.get("cutmix_minmax", None):
            self.mixup_fn = Mixup(**mix_cfg)
        if self.mixup_fn:
            # smoothing is handled with mixup label transform
            self.mixup_criterion = SoftTargetCrossEntropy()

        # Set to a large number if num_classes pnot found
        num_classes = getattr(self.head, "num_classes", 10000)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.eval_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.training_method = params.get("training_method", PROBING)
        self.feature_opt = params.get("feature_opt", "flatten")
        assert self.training_method in TRAIN_METHODS

    def forward(self, x: torch.Tensor):
        if self.training_method == PROBING:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        if self.feature_opt == "flatten":
            features = features.flatten(1)
        elif self.feature_opt == "cls_token":
            features = features[:, 0]
        elif self.feature_opt == "pooled":
            features = features.mean(dim=1)
        x = self.head(features)
        return x

    def training_step(self, batch, batch_nb):
        if self.training_method == PROBING:
            self.encoder.eval()
        else:
            self.encoder.train()

        x, y = batch

        if self.mixup_fn:
            x, y = self.mixup_fn(x, y)
            logit = self.forward(x)
            loss = self.mixup_criterion(logit, y)
        else:
            logit = self.forward(x)
            loss = self.criterion(logit, y)

        pred = logit.argmax(1)
        acc = self.train_acc(pred, y)

        self.log("Train/loss", loss, prog_bar=False)
        self.log("Train/acc", acc, prog_bar=True)
        self.log("Train/epoch_avg_loss", loss, prog_bar=False, sync_dist=True, on_epoch=True, on_step=False)
        self.log("Train/epoch_avg_acc", acc, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        ret = {"loss": loss, "accuracy": acc}
        return ret

    def on_train_epoch_end(self):
        self.train_acc.reset()
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_nb):
        self.encoder.eval()
        with torch.no_grad():
            x, y = batch
            logit = self.forward(x)
            loss = self.criterion(logit, y)
            pred = logit.argmax(1)
            acc = self.eval_acc(pred, y)

        self.log("Val/loss", loss, prog_bar=False, sync_dist=True)
        self.log("Val/acc", acc, prog_bar=True, sync_dist=True)
        self.log("Val/epoch_avg_loss", loss, prog_bar=False, sync_dist=True, on_epoch=True, on_step=False)
        self.log("Val/epoch_avg_acc", acc, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        ret = {"loss": loss, "accuracy": acc}
        return ret

    def on_validation_epoch_end(self):
        self.eval_acc.reset()
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_nb):
        self.encoder.eval()
        with torch.no_grad():
            x, y = batch
            logit = self.forward(x)
            loss = self.criterion(logit, y)
            pred = logit.argmax(1)
            acc = self.test_acc(pred, y)

        self.log("Test/loss", loss, prog_bar=False, sync_dist=True)
        self.log("Test/acc", acc, prog_bar=True, sync_dist=True)
        self.log("Test/epoch_avg_loss", loss, prog_bar=False, sync_dist=True, on_epoch=True, on_step=False)
        self.log("Test/epoch_avg_acc", acc, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        ret = {"loss": loss, "accuracy": acc}
        return ret

    def on_test_epoch_end(self):
        self.test_acc.reset()
        return super().on_test_epoch_end()


@TASKS.register()
class MultiLabelClassification(BaseDownstreamTask):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        num_labels = getattr(self.head, "num_classes", 1)
        self.criterion = nn.MultiLabelSoftMarginLoss()

        avg_method = params.get("avg_method", "macro")
        self.mAP_map = {
            "train": tmc.MultilabelAveragePrecision(num_labels, avg_method),
            "val": tmc.MultilabelAveragePrecision(num_labels, avg_method),
            "test": tmc.MultilabelAveragePrecision(num_labels, avg_method),
        }
        self.training_method = params.get("training_method", PROBING)
        self.feature_opt = params.get("feature_opt", "flatten")
        assert self.training_method in TRAIN_METHODS

    def forward(self, x: torch.Tensor):
        with torch.set_grad_enabled(self.training_method == FINETUNING):
            features = self.encoder(x)

        if self.feature_opt == "flatten":
            features = features.flatten(1)
        elif self.feature_opt == "cls_token":
            features = features[:, 0]
        elif self.feature_opt == "pooled":
            features = features.mean(dim=1)

        x = self.head(features)
        return x

    def _shared_step(self, batch, split):
        x, y = batch
        logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = torch.sigmoid(logit)
        self.mAP_map[split].update(pred, y)
        self.log(f"{split.capitalize()}/loss", loss, prog_bar=False, sync_dist=(split != "train"))
        if split == "train":
            self.log(
                f"{split.capitalize()}/epoch_avg_loss",
                loss,
                prog_bar=False,
                sync_dist=False,  # (split != "train"),
                on_epoch=True,
                on_step=False,
            )
        ret = {"loss": loss}
        return ret

    def on_train_epoch_start(self) -> None:
        if self.training_method == PROBING:
            self.encoder.eval()
        else:
            self.encoder.train()

    def training_step(self, batch, batch_nb):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_nb):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_nb):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.log("Train/epoch_avg_mAP", self.mAP_map["train"].compute(), logger=True, sync_dist=True)
        super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        self.log("Val/epoch_avg_mAP", self.mAP_map["val"].compute(), logger=True, sync_dist=True)
        super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        self.log("Test/epoch_avg_mAP", self.mAP_map["test"].compute(), logger=True, sync_dist=True)
        super().on_test_epoch_end()
