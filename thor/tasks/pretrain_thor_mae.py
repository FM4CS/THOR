import logging
import math

import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import nn
from torch.utils.data import DataLoader

from thor.core.dataset_registry import DATASETS
from thor.core.model_registry import MODELS
from thor.core.task_registry import TASKS
from thor.data.thor_dataset_base import BandDataBatch
from thor.models.thor_mae import ThorMAEOutput
from thor.models.uniformity_loss import uniformity_loss
from thor.tasks.base import BasePretrainTask
from thor.tasks.optimizer import LARS
from thor.utils.helper import safe_copy

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@rank_zero_only
def rank_zero_print(*args, **kwargs):
    print(*args, **kwargs)


@TASKS.register()
class PretrainThorMAE(BasePretrainTask):
    def __init__(self, params):
        # linearly rescale lr before passing to save final parameters
        gpus = params["training"]["trainer"]["gpus"]
        self.num_gpu = gpus if isinstance(gpus, int) else len(gpus)
        self.batch_size = params["training"]["batch_size"]
        lr = params["training"]["optimizer_cfg"]["lr"]
        params["training"]["optimizer_cfg"]["lr"] = lr * (self.batch_size * self.num_gpu) / 256

        super().__init__(params=params)

        nodes = params["training"]["trainer"].get("nodes", 1)
        self.gpus_per_node = self.num_gpu // nodes

        self.hparams["models"]["mae"]["input_params"]["batch_size"] = self.batch_size
        self.hparams["models"]["mae"]["input_params"]["world_size"] = self.num_gpu
        self.hparams["models"]["mae"]["input_params"]["gpus_per_node"] = self.gpus_per_node

        models = MODELS.build(self.hparams["models"])
        self.mae = models["mae"]

        self.max_epochs = self.hparams["training"]["trainer"]["max_epochs"]
        self.mask_ratio = self.hparams["training"]["mae"]["mask_ratio"]
        self.spectral_mask_ratio = self.hparams["training"]["mae"].get("spectral_mask_ratio", self.mask_ratio)
        self.log_img_per_steps = self.hparams["training"]["mae"]["log_img_per_steps"]
        self.log_land_cover_pred = self.hparams["training"].get("log_land_cover_pred", False)
        self.channels = self.mae.channels
        self.recon_lambda = self.hparams["training"].get("recon_lambda", 1.0)
        self.uniformity_loss = self.hparams["training"].get("uniformity_loss", False)
        self.uniformity_lambda = self.hparams["training"].get("uniformity_lambda", 0.01)

        self.era5_lambda = self.hparams["training"].get("era5_lambda", 0.02)
        self.contrastive_lambda = self.hparams["training"].get("contrastive_lambda", 0.1)
        self.fft_lambda = self.hparams["training"].get("fft_lambda", None)
        if self.fft_lambda is None:
            self.fft_lambda = self.hparams["models"]["mae"]["input_params"].get("fft_alpha", 0.01)
        self.land_cover_lambdas = self.hparams["training"].get("land_cover_lambdas", None)
        self.latlon_lambda = self.hparams["training"].get("latlon_lambda", 0.01)
        self.month_lambda = self.hparams["training"].get("month_lambda", 0.01)
        self.orbit_direction_lambda = self.hparams["training"].get("orbit_direction_lambda", 0.01)
        self.incidence_angle_lambda = self.hparams["training"].get("incidence_angle_lambda", 0.01)

        self.epoch_progress = 1
        self.batch_counter = 0
        self.log_counter = 0
        self.log_start = True
        self.token_counter = 0
        self.prepare_data_per_node = False  # We have a shared file system, so no need to prepare data per node
        self.strict_loading = False

    def prepare_data(self):
        """
        Builds a sampling index for the dataset, which is used to sample patches
        """

        dataset_cfg = self.hparams["dataset"]["train_kwargs"]
        if dataset_cfg.get("skip_build_index", False):
            logger.info("Skipping thor dataset index building as per configuration.")
            return

        from thor.data.utils import build_index

        logger.info("Building thor dataset index")
        build_index(
            dataset_cfg,
            num_workers=self.hparams["training"]["train_loader_worker"] * self.gpus_per_node,
        )

    def forward(self, data: BandDataBatch, return_cls_feats=False, return_channel_params=False) -> ThorMAEOutput:
        return self.mae(
            data.band_imgs,
            data.metadata,
            self.mask_ratio,
            self.spectral_mask_ratio,
            era5_land_labels=data.era5_land_data,
            return_cls_feats=return_cls_feats,
            return_channel_params=return_channel_params,
        )

    def training_step(self, batch: BandDataBatch, batch_nb) -> dict:
        if self.log_start:
            self.log_start = False

            if self.mae.dist_group is None and self.mae.use_contrastive_loss and self.mae.use_local_dist:
                global_rank = self.global_rank
                node_rank = global_rank // self.gpus_per_node

                local_node_ranks = list(range(node_rank * self.gpus_per_node, (node_rank + 1) * self.gpus_per_node))

                self.mae.dist_group = dist.new_group(
                    ranks=local_node_ranks,
                    use_local_synchronization=True,
                )

                logger.info(
                    f"global rank {global_rank} node_rank {node_rank}, local node ranks: {local_node_ranks}, set up dist group: {self.mae.dist_group}"
                )

        self.epoch_progress = self.current_epoch + batch_nb / self.num_steps_per_train_epoch

        x = batch.band_imgs
        metadata = batch.metadata

        land_cover_data = {}
        for task in self.mae.land_cover_tasks:
            if task in x:
                land_cover_data[task] = x[task].clone()

        model_output = self.forward(
            batch,
            return_cls_feats=self.uniformity_loss,
            return_channel_params=True,
        )
        era5_land_loss = model_output.loss_era5_land
        pred = model_output.pred
        mask = model_output.mask

        number_of_tokens = sum(p.shape[0] * p.shape[1] for p in pred.values())
        self.token_counter += number_of_tokens

        # NOTE: If we are using flexivit/multilooking, we need updated channel params, i.e., nunber of patches for each channel
        channel_params = model_output.channel_params if model_output.channel_params else self.channels

        if self.uniformity_loss:
            reg_loss = uniformity_loss(model_output.cls_feats)
        else:
            reg_loss = torch.zeros_like(model_output.loss_recon)

        loss = (
            self.recon_lambda * model_output.loss_recon
            + self.fft_lambda * model_output.loss_fft
            + self.uniformity_lambda * reg_loss
            + self.era5_lambda * (era5_land_loss["total"] if isinstance(era5_land_loss, dict) else era5_land_loss)
            + self.contrastive_lambda * model_output.loss_contrastive
            + self.latlon_lambda * model_output.loss_latlon
            + self.month_lambda * model_output.loss_month
            + self.orbit_direction_lambda * model_output.loss_orbit_direction
            + self.incidence_angle_lambda * model_output.loss_incidence_angle
        )
        for land_cover_task in model_output.loss_land_cover_tasks:
            loss += self.land_cover_lambdas[land_cover_task] * model_output.loss_land_cover_tasks[land_cover_task]

        self.batch_counter += 1
        if self.global_rank == 0:
            # log images
            if self.batch_counter >= self.log_img_per_steps:
                self.batch_counter = 0
                x_log = safe_copy(x)
                pred_log = safe_copy(pred)
                mask_log = safe_copy(mask)

                # iterate only pred keys due to the token budget in flexivit during training
                for k in pred_log.keys():
                    input_band = x_log[k]
                    prediction = pred_log[k]
                    image_mask = mask_log[k]
                    num_patch = channel_params[k]["num_patch"]
                    gsd = channel_params[k]["GSD"]
                    p = metadata.ground_cover // gsd // num_patch
                    b = input_band.shape[0]
                    num_patch = input_band.shape[-1] // p

                    prediction = self.mae.unpatchify(prediction, p)
                    image_keep = nn.functional.conv_transpose2d(
                        (image_mask == 0).float().reshape(b, 1, num_patch, num_patch),
                        torch.ones(1, 1, p, p, device=image_mask.device),
                        stride=p,
                    )
                    image_mask = nn.functional.conv_transpose2d(
                        (image_mask == 1).float().reshape(b, 1, num_patch, num_patch),
                        torch.ones(1, 1, p, p, device=image_mask.device),
                        stride=p,
                    )
                    input_band = input_band[0]
                    input_band = (input_band - input_band.min()) / (input_band.max() - input_band.min())
                    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
                    masked_input = input_band * image_keep[0]
                    prediction = prediction[0] * image_mask[0] + input_band * image_keep[0]

                    # scale to 0-255
                    input_band = (input_band * 255).clamp(0, 255).byte()
                    masked_input = (masked_input * 255).clamp(0, 255).byte()
                    prediction = (prediction * 255).clamp(0, 255).byte()

                    self.logger.log_image(
                        key=f"Sample/{k.replace(':', '_')}",
                        images=[input_band, masked_input, prediction],
                        caption=[f"Input, gc: {metadata.ground_cover}m", "Masked Input", "Prediction"],
                    )

                if self.log_land_cover_pred:
                    available_groups = self.mae.get_available_groups(mask_log)
                    for task in model_output.land_cover_pred.keys():
                        if (
                            task not in land_cover_data
                            or task not in model_output.land_cover_pred
                            or task not in model_output.land_cover_mask
                        ):
                            logger.debug(f"Something wrong, {task} not in land_cover_data")
                            continue
                        self.log_land_cover(
                            task=task,
                            lc_gt_log=land_cover_data[task].detach().clone().cpu(),
                            lc_pred_log=safe_copy(model_output.land_cover_pred[task]),
                            lc_mask_log=safe_copy(model_output.land_cover_mask[task]),
                            ground_cover=metadata.ground_cover,
                            channel_params=channel_params,
                            available_groups=available_groups,
                        )

        self.log(
            "Debug/num_tokens",
            self.token_counter,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=False,
            sync_dist=True,
        )
        self.log("Debug/epoch_progress", self.epoch_progress, logger=True, prog_bar=False, rank_zero_only=True)
        self.log("Train/R0_loss", loss, logger=True, prog_bar=False, rank_zero_only=True)
        self.log("Train/R0_recon_loss", model_output.loss_recon, logger=True, prog_bar=False, rank_zero_only=True)
        self.log("Train/R0_reg_loss", reg_loss, logger=True, prog_bar=False, rank_zero_only=True)
        self.log("Train/R0_fft_loss", model_output.loss_fft, logger=True, prog_bar=False, rank_zero_only=True)
        self.log(
            "Train/R0_tau",
            model_output.tau if model_output.tau is not None else 0.0,
            logger=True,
            prog_bar=False,
            rank_zero_only=True,
        )

        self.log("Train/R0_latlon_loss", model_output.loss_latlon, logger=True, prog_bar=False, rank_zero_only=True)
        self.log("Train/R0_month_loss", model_output.loss_month, logger=True, prog_bar=False, rank_zero_only=True)
        self.log(
            "Train/R0_orbit_direction_loss",
            model_output.loss_orbit_direction,
            logger=True,
            prog_bar=False,
            rank_zero_only=True,
        )
        self.log(
            "Train/R0_incidence_angle_loss",
            model_output.loss_incidence_angle,
            logger=True,
            prog_bar=False,
            rank_zero_only=True,
        )

        if isinstance(era5_land_loss, dict):
            for k, v in era5_land_loss.items():
                if k == "total":
                    continue
                self.log(f"Train_R0_era5_land_loss/{k}", v, logger=True, prog_bar=False, rank_zero_only=True)
            self.log(
                "Train/R0_era5_land_loss", era5_land_loss["total"], logger=True, prog_bar=False, rank_zero_only=True
            )
        else:
            self.log("Train/R0_era5_land_loss", era5_land_loss, logger=True, prog_bar=False, rank_zero_only=True)

        self.log(
            "Train/R0_contrastive_loss", model_output.loss_contrastive, logger=True, prog_bar=False, rank_zero_only=True
        )

        self.log(
            "train_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=False,
            sync_dist=True,
        )

        self.log(
            "train_loss_recon_epoch",
            model_output.loss_recon,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=False,
            sync_dist=True,
        )

        self.log(
            "train_loss_fft_epoch",
            model_output.loss_fft,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=False,
            sync_dist=True,
        )
        self.log(
            "train_loss_latlon_epoch",
            model_output.loss_latlon,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=False,
            sync_dist=True,
        )

        self.log(
            "train_loss_month_epoch",
            model_output.loss_month,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=False,
            sync_dist=True,
        )

        self.log(
            "train_loss_contrastive_epoch",
            model_output.loss_contrastive,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=False,
            sync_dist=True,
        )

        # self.log(
        #     "train_reg_loss_epoch",
        #     reg_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        #     rank_zero_only=False,
        #     sync_dist=True,
        # )

        self.log(
            "train_loss_era5_land_epoch",
            era5_land_loss["total"] if isinstance(era5_land_loss, dict) else era5_land_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=False,
            sync_dist=True,
        )
        if isinstance(era5_land_loss, dict):
            for k, v in era5_land_loss.items():
                if k == "total":
                    continue
                self.log(
                    f"Train_era5_land_loss/{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    rank_zero_only=False,
                    sync_dist=True,
                )

        self.log(
            "train_loss_orbit_direction",
            model_output.loss_orbit_direction,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=False,
            sync_dist=True,
        )

        self.log(
            "train_loss_incidence_angle",
            model_output.loss_incidence_angle,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=False,
            sync_dist=True,
        )

        self.log(
            "train_loss_land_cover",
            sum(task_loss for task_loss in model_output.loss_land_cover_tasks.values()),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=False,
            sync_dist=True,
        )

        for task in model_output.loss_land_cover_tasks:
            self.log(
                f"Train/R0_land_cover_loss_{task}",
                model_output.loss_land_cover_tasks[task],
                prog_bar=False,
                logger=True,
                rank_zero_only=True,
            )

        if model_output.loss_land_cover_tasks_group is not None:
            for task in model_output.loss_land_cover_tasks_group:
                if isinstance(model_output.loss_land_cover_tasks_group[task], dict):
                    for k, v in model_output.loss_land_cover_tasks_group[task].items():
                        self.log(
                            f"Train_R0_land_cover_loss/{task}/{k}",
                            v,
                            prog_bar=False,
                            logger=True,
                            rank_zero_only=True,
                        )
                        self.log(
                            f"Train_land_cover_loss/{task}/{k}",
                            v,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            logger=True,
                            rank_zero_only=False,
                            sync_dist=True,
                        )

        return {"loss": loss}

    def log_land_cover(self, task, lc_gt_log, lc_pred_log, lc_mask_log, ground_cover, channel_params, available_groups):
        for group_name, lc_pred in lc_pred_log.items():
            if group_name not in available_groups:
                logger.warning(f"group {group_name} not in available groups")
                continue
            if group_name not in lc_mask_log:
                logger.warning(f"group {group_name} not in land cover mask log")
                continue
            product_band = available_groups[group_name][0]
            image_mask = lc_mask_log[group_name]

            task_gsd = int(ground_cover / lc_gt_log.shape[-1])  # Assuming ground_cover is the GSD in meters

            if task_gsd != self.mae.land_cover_tasks[task]["gsd"]:
                logger.warning(
                    f"task GSD {task_gsd} does not match expected GSD {self.mae.land_cover_tasks[task]['gsd']} for task {task}"
                )

            task_num_classes = self.mae.land_cover_tasks[task]["num_classes"]
            task_group_num_classes = self.mae.land_cover_tasks[task]["groups"][group_name]["valid_classes"]

            num_patch = channel_params[product_band]["num_patch"]
            p = ground_cover // task_gsd // num_patch
            b = lc_gt_log.shape[0]
            num_patch = lc_gt_log.shape[-1] // p

            lc_pred = self.mae.unpatchify(lc_pred, p, c=task_group_num_classes)

            image_keep = nn.functional.conv_transpose2d(
                (image_mask == 0).float().reshape(b, 1, num_patch, num_patch),
                torch.ones(1, 1, p, p, device=image_mask.device),
                stride=p,
            )
            image_mask = nn.functional.conv_transpose2d(
                (image_mask == 1).float().reshape(b, 1, num_patch, num_patch),
                torch.ones(1, 1, p, p, device=image_mask.device),
                stride=p,
            )
            if self.mae.land_cover_tasks[task]["loss_type"] == "cross_entropy":
                lc_pred = torch.argmax(lc_pred, dim=1, keepdim=True)
                lc_pred_shape = lc_pred.shape
                lc_pred = self.mae.invert_map_targets(
                    lc_pred.flatten().long(),
                    self.mae.land_cover_mappings[task][group_name],
                    num_classes=task_num_classes,
                    ignore_classes=self.mae.land_cover_tasks[task]["groups"][group_name]["ignore"],
                )
                lc_pred = lc_pred.reshape(lc_pred_shape[0], 1, lc_pred_shape[2], lc_pred_shape[3])

                lc_gt_band = lc_gt_log[0][0].int()
                full_lc_pred = lc_pred[0][0].int()

                lc_pred = (lc_pred[0][0] * image_mask[0][0] + lc_gt_band * image_keep[0][0]).int()

                lc_colors = getattr(self.trainer.train_dataloader.dataset, f"{task}_COLORS", None)
                if lc_colors is None:
                    lc_colors = np.zeros((task_num_classes, 3), dtype=np.uint8)
                    lc_colors[:, 0] = np.linspace(0, 255, task_num_classes)
                    lc_colors[:, 1] = np.linspace(0, 255, task_num_classes)
                    lc_colors[:, 2] = np.linspace(0, 255, task_num_classes)
                lc_colors = torch.tensor(lc_colors).int()

                H, W = lc_gt_band.shape

                lc_gt_band = lc_gt_band.reshape(H, W, 1).repeat(1, 1, 3)
                full_lc_pred = full_lc_pred.reshape(H, W, 1).repeat(1, 1, 3)
                lc_pred = lc_pred.reshape(H, W, 1).repeat(1, 1, 3)

                for i in range(task_num_classes):
                    lc_gt_band[(lc_gt_band[:, :, 0] == i).reshape(H, W), :] = lc_colors[i]
                    full_lc_pred[(full_lc_pred[:, :, 0] == i).reshape(H, W), :] = lc_colors[i]
                    lc_pred[(lc_pred[:, :, 0] == i).reshape(H, W), :] = lc_colors[i]

                self.logger.log_image(
                    key=f"Landcover/{task}/{group_name}",
                    images=[
                        lc_gt_band.numpy().astype("uint8"),
                        full_lc_pred.numpy().astype("uint8"),
                        lc_pred.numpy().astype("uint8"),
                    ],
                    caption=[f"GT, gc: {ground_cover}m", "Full Pred", "GT + Pred"],
                )
            elif self.mae.land_cover_tasks[task]["loss_type"] == "mse":
                H, W = lc_gt_log[0][0].shape[0], lc_gt_log[0][0].shape[1]
                if lc_gt_log.shape[1] == 1:
                    lc_gt_band = lc_gt_log[0][0]

                    lc_pred = lc_pred[0][0]
                elif lc_gt_log.shape[1] == 2:
                    lc_gt_band = torch.zeros((H, W, 3), device=lc_gt_log.device, dtype=lc_gt_log.dtype)
                    lc_gt_band[:, :, 0] = lc_gt_log[0][0]
                    lc_gt_band[:, :, 1] = lc_gt_log[0][1]
                    lc_gt_band[:, :, 2] = lc_gt_log[0][0] + lc_gt_log[0][1]

                    lc_pred_c = torch.zeros((H, W, 3), device=lc_pred.device, dtype=lc_pred.dtype)
                    lc_pred_c[:, :, 0] = lc_pred[0][0]
                    lc_pred_c[:, :, 1] = lc_pred[0][1]
                    lc_pred_c[:, :, 2] = lc_pred[0][0] + lc_pred[0][1]

                    lc_pred = lc_pred_c

                lc_gt_band = ((lc_gt_band - lc_gt_band.min()) / (lc_gt_band.max() - lc_gt_band.min()) * 255).to(
                    torch.uint8
                )

                # min max normalize the land cover band
                lc_pred = ((lc_pred - lc_pred.min()) / (lc_pred.max() - lc_pred.min()) * 255).to(torch.uint8)

                full_lc_pred = lc_pred.clone()

                lc_pred_mixed = lc_pred * image_mask[0][0][:, :, None] + lc_gt_band * image_keep[0][0][:, :, None]

                self.logger.log_image(
                    key=f"Landcover/{task}/{group_name}",
                    images=[
                        lc_gt_band.numpy().squeeze(),
                        full_lc_pred.numpy().squeeze(),
                        lc_pred_mixed.numpy().squeeze(),
                    ],
                    caption=[f"GT, gc: {ground_cover}m", "Full Pred", "GT + Pred"],
                )

    def validation_step(self, batch: BandDataBatch, batch_nb) -> dict:
        """overwrite to skip for now"""
        pass

    def test_step(self, batch: BandDataBatch, batch_nb) -> dict:
        x = batch
        loss, pred, mask = self.forward(x)

        self.batch_counter += 1
        if self.global_rank == 0:
            # log images
            if self.batch_counter >= self.log_img_per_steps:
                self.log_counter += 1
                self.batch_counter = 0
                for (k, input_band), (_, prediction), (_, image_mask) in zip(
                    x.items(), pred.items(), mask.items(), strict=False
                ):
                    prediction = pred[k]
                    image_mask = mask[k]
                    num_patch = self.channels[k]["num_patch"]
                    gsd = self.channels[k]["GSD"]
                    p = self.mae.ground_cover // gsd // num_patch
                    b = input_band.shape[0]
                    num_patch = input_band.shape[-1] // p

                    prediction = self.mae.unpatchify(prediction, p)
                    image_keep = nn.functional.conv_transpose2d(
                        (image_mask == 0).float().reshape(b, 1, num_patch, num_patch),
                        torch.ones(1, 1, p, p, device=image_mask.device),
                        stride=p,
                    )
                    image_mask = nn.functional.conv_transpose2d(
                        (image_mask == 1).float().reshape(b, 1, num_patch, num_patch),
                        torch.ones(1, 1, p, p, device=image_mask.device),
                        stride=p,
                    )
                    input_band = input_band[0][0]
                    input_band[input_band.isnan()] = 0
                    input_band = (input_band - input_band.min()) / (input_band.max() - input_band.min())
                    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
                    masked_input = input_band * image_keep[0][0]
                    prediction = prediction[0][0] * image_mask[0][0] + input_band * image_keep[0][0]

                    # scale to 0-255
                    input_band = (input_band * 255).clamp(0, 255).byte()
                    masked_input = (masked_input * 255).clamp(0, 255).byte()
                    prediction = (prediction * 255).clamp(0, 255).byte()

                    self.logger.log_image(
                        key=f"Sample/{k.replace(':', '_')}",
                        step=self.log_counter * self.log_img_per_steps,
                        images=[input_band, masked_input, prediction],
                        caption=["Input", "Masked Input", "Prediction"],
                    )

        self.log("Test/R0_loss", loss, logger=True, prog_bar=False, rank_zero_only=True)

        self.log(
            "test_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=False,
            sync_dist=True,
        )

        return {"loss": loss}

    def on_validation_epoch_end(self, outputs):
        """overwirte to skip for now"""
        pass

    def on_train_epoch_start(self):
        if hasattr(self.trainer.train_dataloader.dataset, "set_epoch"):
            rank_zero_print(f"PL start setting epoch to: {self.trainer.current_epoch}")
            self.trainer.train_dataloader.dataset.set_epoch(self.trainer.current_epoch)

    def train_dataloader(self):
        """overwrite since we need to drop last"""

        self.hparams["dataset"]["train_kwargs"]["random_seed"] = self.hparams["random_seed"]
        self.hparams["dataset"]["train_kwargs"]["batch_size"] = self.batch_size

        dataloader_batch_size = None
        dataloader_drop_last = False
        if "iterable" in self.hparams["dataset"]["name"].lower():
            self.hparams["dataset"]["train_kwargs"]["batch_size"] = self.batch_size
            self.hparams["dataset"]["train_kwargs"]["global_rank"] = self.global_rank
            self.hparams["dataset"]["train_kwargs"]["world_size"] = self.num_gpu
            self.hparams["dataset"]["train_kwargs"]["current_epoch"] = self.trainer.current_epoch
            # Turn off automatic batching, we do it in the dataset
            dataloader_shuffle = False
        else:
            dataloader_shuffle = True

        dataset = DATASETS.build(self.hparams["dataset"], split="train")
        data_loader = DataLoader(
            dataset,
            shuffle=dataloader_shuffle,
            batch_size=dataloader_batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=self.training_cfg["train_loader_worker"],
            pin_memory=True,
            persistent_workers=False,
            timeout=4 * 640,  # seconds
            drop_last=dataloader_drop_last,
            prefetch_factor=2,
        )
        if "iterable" in self.hparams["dataset"]["name"].lower():
            # Iterable dataset already divides length by world size
            self.num_steps_per_train_epoch = len(data_loader)
        else:
            self.num_steps_per_train_epoch = len(data_loader) // self.num_gpu
        print(f"num steps per epoch: {self.num_steps_per_train_epoch}")
        return data_loader

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        """overwrite since we need to drop last"""
        dataset = DATASETS.build(self.hparams["dataset"], split="test")
        data_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=self.training_cfg["eval_loader_worker"],
        )
        self.num_steps_per_train_epoch = len(data_loader) // self.num_gpu
        return data_loader

    def configure_optimizers(self):
        optimizer_name = self.training_cfg.get("optimizer", "LARS")
        optimizer_cfg = self.training_cfg.get("optimizer_cfg", {"lr": 0.001})

        if "lr" not in optimizer_cfg:
            msg = "You must provide learning rate in optimizer cfg"
            raise KeyError(msg)

        if optimizer_name == "AdamW":
            optimizer_class = torch.optim.AdamW
        elif optimizer_name == "LARS":
            optimizer_class = LARS
        else:
            msg = f"{optimizer_name} is not supported, add it to configure_optimizers in base lightning class."
            raise ValueError(msg)

        optimizer = optimizer_class(self.parameters(), **optimizer_cfg)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):  # ,
        # on_tpu=False, using_native_amp=False, using_lbfgs=False):
        """Decays the learning rate with half-cycle cosine after warmup"""
        initial_lr = self.hparams["training"]["optimizer_cfg"]["lr"]
        warm_up_epoch = self.hparams["training"]["warmup_epoch"]
        if self.epoch_progress < warm_up_epoch:
            lr = initial_lr * self.epoch_progress / warm_up_epoch
        else:
            lr = (
                initial_lr
                * 0.5
                * (1.0 + math.cos(math.pi * (self.epoch_progress - warm_up_epoch) / (self.max_epochs - warm_up_epoch)))
            )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step(closure=optimizer_closure)
