import fnmatch
import logging
import math
from collections import Counter, OrderedDict
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from thor.core.model_registry import MODELS
from thor.data.thor_dataset_base import MetaData
from thor.models.contrastive_loss_func_lc import LandCoverGroupContrastLoss
from thor.models.contrastive_loss_func_lc_v2 import LandCoverGroupContrastLossV2
from thor.models.thor_vit import Block, ThorViTEncoder, alibi_cls_token_pad, get_alibi_thor, get_slopes
from thor.utils.patch_embed import (
    FlexiConvTransDecoder,
    FlexiLinDecoder,
    FlexiPosEmbed,
    FlexiPosResEmbed,
    get_flexivit_grid_sizes,
    resize_abs_pos_embed,
)
from thor.utils.pos_embed import get_1d_sincos_pos_embed_from_grid_torch

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ThorMAEOutput(NamedTuple):
    loss_recon: torch.FloatTensor
    loss_fft: torch.FloatTensor
    loss_contrastive: torch.FloatTensor
    loss_era5_land: torch.FloatTensor | dict[str, torch.FloatTensor]
    pred: dict[str, torch.FloatTensor]
    mask: dict[str, torch.LongTensor]
    loss_latlon: torch.FloatTensor | None = None
    loss_month: torch.FloatTensor | None = None
    loss_orbit_direction: torch.FloatTensor | None = None
    loss_incidence_angle: torch.FloatTensor | None = None
    loss_group: dict[str, torch.FloatTensor] | None = None
    loss_contrastive_group: dict[str, torch.FloatTensor] | None = None
    loss_land_cover_tasks: dict[str, torch.FloatTensor] | None = None
    loss_land_cover_tasks_group: dict[str, dict[str, torch.FloatTensor]] | None = None
    land_cover_pred: dict[str, dict[str, torch.FloatTensor]] | None = None
    land_cover_mask: dict[str, dict[str, torch.LongTensor]] | None = None
    cls_feats: torch.FloatTensor | None = None
    channel_params: dict[str, dict[str, int]] | None = None
    tau: float | None = None


# Copied from https://github.com/antofuller/CROMA/blob/59505a6bcadbf36ba20767270154bf9f3067c5e7/pretrain_croma.py#L343
def apply_mask_to_alibi(
    alibi, ids_keep_queries, ids_keep_keys, batch_size, orig_seq_len, masked_seq_len, attention_heads
):
    ids_keep_matrix = (
        rearrange(ids_keep_queries, "b i -> b i 1") + rearrange(ids_keep_keys, "b i -> b 1 i") * orig_seq_len
    )
    ids_keep_long_sequence = rearrange(ids_keep_matrix, "b i j -> b (i j)")
    alibi_long_sequence = rearrange(alibi.repeat(batch_size, 1, 1, 1), "b n i j -> b (i j) n")
    alibi_masked = torch.gather(
        alibi_long_sequence, dim=1, index=ids_keep_long_sequence.unsqueeze(-1).repeat(1, 1, attention_heads)
    )
    return rearrange(alibi_masked, "b (i j) n -> b n i j", i=masked_seq_len, j=masked_seq_len)


class ThorMAE(ThorViTEncoder):
    def __init__(
        self,
        input_params: dict[str, Any],
        # ----Encoder specific
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        embed_band: bool = True,
        band_embed_dim: int = 128,
        pad_band_embed_null: bool = False,
        embed_prod: bool = True,
        prod_embed_dim: int = 128,
        pad_prod_embed_null: bool = False,
        embed_patch_size: bool = False,
        # ----Decoder specific
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        decoder_band_embed_dim: int = 64,
        decoder_prod_embed_dim: int = 64,
        # ----Common
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        log_patch_size: bool = False,
        norm_pix_loss=False,
    ) -> None:
        # =======================================================================
        # MAE encoder specific from parent class
        super().__init__(
            input_params,
            embed_dim,
            depth,
            num_heads,
            embed_prod,
            prod_embed_dim,
            pad_prod_embed_null,
            embed_band,
            band_embed_dim,
            pad_band_embed_null,
            embed_patch_size,
            mlp_ratio,
            qkv_bias,
            norm_layer,
            log_patch_size,
        )

        # =======================================================================
        # MAE defcoder specific
        # Initialize decoder projection
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # Initialize mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # Initialize decoder embedding
        self.decoder_num_heads = decoder_num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_prod_embed_dim = decoder_prod_embed_dim if embed_prod else 0
        self.decoder_band_embed_dim = decoder_band_embed_dim if embed_band else 0
        self.decoder_pos_embed_dim = decoder_embed_dim - self.decoder_prod_embed_dim - self.decoder_band_embed_dim
        self.decoder_patch_size_embed_dim = self.decoder_pos_embed_dim if embed_patch_size else 0

        self.masking_schema = input_params.pop("masking_schema", {"spatial": "random_same_ratio", "spectral": "all"})
        self.era5_land_products = input_params.get("era5_land_products", [])
        self.era5_return_dict = input_params.get("era5_return_dict", False)
        self.use_fft_loss = input_params.pop("use_fft_loss", False)
        self.use_contrastive_loss = input_params.get("use_contrastive_loss", False)
        self.contrast_min_num_patches = input_params.get("contrast_min_num_patches", 8)
        self.max_contrastive_groups = input_params.get("max_contrastive_groups", 4)
        self.min_process_ratio = input_params.get("min_process_ratio", 0.5)
        self.use_local_dist = input_params.get("use_local_dist", False)
        self.predict_month = input_params.pop("predict_month", True)
        self.predict_location = input_params.pop("predict_location", True)
        self.predict_orbit_direction = input_params.pop("predict_orbit_direction", True)
        self.predict_incidence_angle = input_params.pop("predict_incidence_angle", True)

        self.log_per_group = input_params.get("log_per_group", False)

        self.decoder_pos_type = input_params.get("decoder_pos_type", "absolute_res")
        assert self.decoder_pos_type in ["alibi", "pooled", "interpolate", "absolute_res"], (
            f"encoder_pos_type {self.decoder_pos_type} is not supported."
        )
        self.flexivit_decoder_type = input_params.get("flexivit_decoder_type", "conv")
        assert self.flexivit_decoder_type in ["linear", "conv"], "Invalid flexivit decoder type"

        self.gpus_per_node = input_params.get("gpus_per_node", None)
        if self.gpus_per_node is None:
            self.gpus_per_node = input_params.get("world_size", 1)

        self.dist_group = None

        self.decoder_pos_embed, self.decoder_band_embed, self.decoder_prod_embed, *_ = self.initialize_embedding(
            self.decoder_pos_embed_dim,
            self.decoder_band_embed_dim,
            self.decoder_prod_embed_dim,
            encoder=False,
        )

        # Initialize decoder transformer blocks
        self.decoder_blocks = nn.ModuleList(
            [
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        if self.flexivit_decoder_type == "linear":
            flexivit_decoder = FlexiLinDecoder
        elif self.flexivit_decoder_type == "conv":
            flexivit_decoder = FlexiConvTransDecoder

        self.flexivit_decode_pred = flexivit_decoder(
            ground_covers=self.ground_covers,
            channels=self.channels,
            groups=self.groups,
            decoder_embed_dim=decoder_embed_dim,
            patch_size_seqs=self.ind_patch_embed.patch_size_seqs,
            pi_inverse=False,
        )

        if self.decoder_pos_type in {"interpolate", "absolute_res"}:
            _reference_grid_size, grid_sizes, ground_cover_lookup = get_flexivit_grid_sizes(
                self.ground_covers,
                self.channels,
                self.ind_patch_embed.patch_size_seqs,
                self.flexivit_ref_patch_size,
            )

            if self.decoder_pos_type == "interpolate":
                self.decoder_ref_pos_embed = FlexiPosEmbed(
                    grid_size=self.flexivit_ref_grid_size,
                    pos_embed_dim=self.decoder_pos_embed_dim,
                    grid_sizes=grid_sizes,
                    interpolation="bilinear",
                )
            elif self.decoder_pos_type == "absolute_res":
                self.decoder_ref_pos_embed = FlexiPosResEmbed(
                    pos_embed_dim=self.decoder_pos_embed_dim,
                    ground_cover_lookup=ground_cover_lookup,
                    channels=self.channels,
                    interpolate=False,
                    interpolation="bilinear",
                )

        elif self.decoder_pos_type == "alibi":
            self.register_buffer("decoder_slopes", torch.tensor(get_slopes(decoder_num_heads)))

        land_cover_tasks = input_params.get("land_cover_tasks", {})
        self.land_cover_tasks = {}
        self.land_cover_decoders = {}
        self.land_cover_mappings = {}
        for task_name, task_params in land_cover_tasks.items():
            land_cover_pred = {}
            expanded_groups = {}
            assert task_params["loss_on_patches"] in [
                "masked",
                "unmasked",
                "all",
            ], f"Invalid loss_on_patches {task_params['loss_on_patches']}"
            self.land_cover_mappings[task_name] = {}

            for group_key, group_params in task_params["groups"].items():
                for group_name, group_members in self.groups.items():
                    if all(fnmatch.fnmatch(group_member, group_key) for group_member in group_members):
                        if "ignore" not in group_params:
                            group_params["ignore"] = []
                        mapping, valid_classes = self._create_class_mapping(
                            task_params["num_classes"], group_params["ignore"], self.device
                        )
                        self.land_cover_mappings[task_name][group_name] = mapping
                        group_params["valid_classes"] = len(valid_classes)

                        expanded_groups[group_name] = group_params
                        kernel_size = group_params["patch_size"]
                        land_cover_pred[group_name] = nn.ConvTranspose2d(
                            decoder_embed_dim, group_params["valid_classes"], kernel_size, stride=kernel_size
                        )

            task_params["groups"] = expanded_groups
            self.land_cover_tasks[task_name] = task_params
            self.land_cover_decoders[task_name] = nn.ModuleDict(land_cover_pred)
        self.land_cover_decoders = nn.ModuleDict(self.land_cover_decoders)

        logger.info(f"Land cover tasks: {self.land_cover_tasks}")
        logger.info(f"Land cover decoders: {self.land_cover_decoders}")

        if len(self.era5_land_products) > 0:
            self.era5_land_gsd = 11132.0  # 11132 is the GSD of ERA5 land daily
            # self.era5_land_patch_size = math.ceil(self.ground_cover / 11132.0)  # 11132 is the GSD of ERA5 land daily
            self.era5_land_patch_size = 2
            self.era5_land_head = nn.ConvTranspose2d(
                self.embed_dim,
                len(self.era5_land_products),
                kernel_size=self.era5_land_patch_size,
                stride=self.era5_land_patch_size,
            )

        if self.predict_month:
            self.month_head = nn.Linear(embed_dim, 2)

        if self.predict_location:
            self.location_head = nn.Linear(embed_dim, 4)

        if self.predict_orbit_direction:
            self.orbit_direction_head = nn.Linear(embed_dim, 2)

        if self.predict_incidence_angle:
            self.incidence_angle_head = nn.Linear(embed_dim, 1)

        self.contrast_type = None
        if self.use_contrastive_loss:
            cont_params = input_params["contrastive_loss"]
            self.contrast_type = cont_params.pop("type", "land_cover_v2")
            assert self.contrast_type in ["land_cover", "land_cover_v2"], (
                f"Invalid contrastive loss type {self.contrast_type}"
            )
            if self.contrast_type == "land_cover":
                self.contrast_loss = LandCoverGroupContrastLoss(
                    groups=self.groups,
                    num_samples=cont_params["num_samples"],
                    tau=cont_params["tau"],
                    projection_input=self.embed_dim,
                    projection_output=self.embed_dim,
                    eps=cont_params["eps"],
                    smoothing=cont_params["smoothing"],
                    modulate=cont_params["modulate"],
                    norm_type=cont_params["norm_type"],
                    pixel_ratio=cont_params["pixel_ratio"],
                )
            elif self.contrast_type == "land_cover_v2":
                self.contrast_loss = LandCoverGroupContrastLossV2(
                    groups=self.groups,
                    num_samples=cont_params["num_samples"],
                    batch_size=input_params.get("batch_size", 1),
                    world_size=input_params.get("world_size", 1) if not self.use_local_dist else self.gpus_per_node,
                    tau=cont_params["tau"],
                    projection_input=self.embed_dim,
                    projection_output=self.embed_dim,
                    eps=cont_params["eps"],
                    norm_type=cont_params["norm_type"],
                    pixel_ratio=cont_params["pixel_ratio"],
                )

        self.norm_pix_loss = norm_pix_loss
        # =======================================================================
        # need to run this again (it has been run once in parent class initialization in super call)
        #  to make sure all the parameters are initialized properly
        self.initialize_weight()

    def initialize_weight(self) -> None:
        if self.cls_token_type == "token":
            torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _create_class_mapping(
        self, num_classes: int, ignore_classes: list[int], device: torch.device
    ) -> tuple[torch.Tensor, list[int]]:
        """Create mapping from original classes to valid classes"""
        valid_classes = [i for i in range(num_classes) if i not in ignore_classes]

        # Create forward mapping: original_class -> new_class_index
        mapping = torch.full((num_classes,), -1, dtype=torch.long, device=device)
        for new_idx, orig_idx in enumerate(valid_classes):
            mapping[orig_idx] = new_idx

        return mapping, valid_classes

    def get_decoder_auxilliary_embed(
        self,
        original_input: dict[str, torch.Tensor],
        available_groups: dict[str, list[str]],
        channel_params: dict[str, dict[str, int]],
    ) -> torch.Tensor:
        auxilliary_embed = []
        for group_name, group_member in available_groups.items():
            product_band = group_member[0]
            product, _band = product_band.split(":")

            if self.decoder_pos_type == "pooled":
                group_pos_embed = self.decoder_pos_embed[str(self.channels[product_band]["GSD"])]

                new_size = channel_params[product_band]["num_patch"]
                group_pos_embed = resize_abs_pos_embed(
                    group_pos_embed[None, :, :], new_size=new_size, num_prefix_tokens=0
                )[0]

            elif self.decoder_pos_type == "interpolate":
                new_size = channel_params[product_band]["num_patch"]
                group_pos_embed = self.decoder_ref_pos_embed(new_size)

            elif self.decoder_pos_type == "absolute_res":
                grid_size = channel_params[product_band]["num_patch"]
                patch_size = channel_params[product_band]["patch_size"]
                gsd = channel_params[product_band]["GSD"]
                group_pos_embed = self.decoder_ref_pos_embed(
                    grid_size=grid_size, patch_size=patch_size, gsd=gsd, device=self.device
                )

            elif self.decoder_pos_type == "alibi":
                _x = next(iter(original_input.values()))
                if self.embed_patch_size:
                    patch_sizes = (
                        torch.ones((channel_params[product_band]["num_patch"] ** 2), device=_x.device, dtype=_x.dtype)
                        * channel_params[product_band]["patch_size"]
                        * channel_params[product_band]["GSD"]
                        / (self.flexivit_ref_patch_size * self.min_gsd)
                    )

                    group_pos_embed = get_1d_sincos_pos_embed_from_grid_torch(
                        self.decoder_patch_size_embed_dim, pos=patch_sizes
                    ).to(_x.dtype)
                else:
                    group_pos_embed = torch.zeros(
                        (
                            channel_params[product_band]["num_patch"] ** 2,
                            self.decoder_pos_embed_dim,
                        ),
                        device=_x.device,
                        dtype=_x.dtype,
                    )

            auxilliary_embed.append(group_pos_embed)
            if self.embed_band:
                # Expand to sequence length and concatenate
                auxilliary_embed[-1] = torch.cat(
                    (
                        auxilliary_embed[-1],
                        self.decoder_band_embed[group_name].expand(auxilliary_embed[-1].shape[0], -1),
                    ),
                    dim=-1,
                )
            if self.embed_prod:
                # Expand to sequence length and concatenate
                auxilliary_embed[-1] = torch.cat(
                    (
                        auxilliary_embed[-1],
                        self.decoder_prod_embed[product].expand(auxilliary_embed[-1].shape[0], -1),
                    ),
                    dim=-1,
                )

            data = original_input[product_band]
            band_ground_cover = int(data.shape[-1] * channel_params[product_band]["GSD"])
            if band_ground_cover not in self.ground_covers:
                msg = (
                    f"Input ground cover for {product_band} is {band_ground_cover}x{band_ground_cover}, "
                    f"which does match grid of {channel_params[product_band]['num_patch']}x{channel_params[product_band]['num_patch']}"
                    f"with patch size {channel_params[product_band]['patch_size']} and GSD {channel_params[product_band]['GSD']}"
                    f" patches for defined ground covers {self.ground_covers}."
                )
                raise ValueError(msg)

        auxilliary_embed = torch.cat(auxilliary_embed, dim=0)
        return auxilliary_embed

    def random_masking_fixed(
        self,
        x: torch.Tensor,
        intervals: list[tuple[int, int, bool, bool, int]],
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adapt from SatMAE: https://github.com/sustainlab-group/SatMAE

        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise. Modified to allow
        for per-band random masking to ensure that mask_ratio is consistent
        across bands with a different number of patches.
        x: [B, T, D], sequence
        """
        B, T, D = x.shape  # batch, sequence length, dim

        tot_len_keep = 0
        len_keeps = []
        for interval in intervals:
            start, stop, needs_masking, _ = interval
            size = stop - start
            if size == 1:
                len_keep = (torch.rand(1, device=x.device) > mask_ratio).int().item()
            else:
                # round up
                len_keep = int(size * (1 - mask_ratio) + 0.5)

            len_keeps.append(len_keep)

            if needs_masking:
                tot_len_keep += len_keep

        all_ids_restore, all_ids_keep = [], []

        idx = 0
        restore_ids_offset = 0
        masked_restore_ids_offset = tot_len_keep
        for i, interval in enumerate(intervals):
            start, stop, needs_masking, _ = interval
            size = stop - start
            len_keep = len_keeps[i]

            if needs_masking:
                noise = torch.rand(B, size, device=x.device)  # noise in [0, 1]
                # sort noise for each sample
                ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
                ids_restore = torch.argsort(ids_shuffle, dim=1)
                # keep the first subset
                ids_keep = ids_shuffle[:, :len_keep]
            else:
                ids_restore = torch.arange(size, device=x.device).repeat(B, 1)
                ids_keep = torch.arange(size, device=x.device).repeat(B, 1)

            # Update the index to match real position in original tensor

            # we will have (T-g_i)*(1-r) number of tokens after our current group
            # we need to shift the indices by this amount
            masked_restore_ids_offset -= len_keep
            ids_restore[ids_restore >= len_keep] += masked_restore_ids_offset
            # the ids will be shifted by the number of tokens we have kept so far
            ids_restore += restore_ids_offset
            ids_keep += idx

            all_ids_restore.append(ids_restore)
            all_ids_keep.append(ids_keep)

            idx += size
            restore_ids_offset += len_keep

        ids_restore = torch.cat(all_ids_restore, dim=1)
        ids_keep = torch.cat(all_ids_keep, dim=1)
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones((B, T), dtype=torch.bool, device=x.device)
        mask[:, :tot_len_keep] = False

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def random_masking_old(
        self, x: torch.Tensor, intervals: list[tuple[int, int, bool, bool, int]], mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adapt from SatMAE: https://github.com/sustainlab-group/SatMAE

        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise. Modified to allow
        for per-band random masking to ensure that mask_ratio is consistent
        across bands with a different number of patches.
        x: [B, T, D], sequence
        """
        B, _T, D = x.shape  # batch, sequence length, dim

        all_ids_restore, all_ids_keep, all_masks = [], [], []
        for interval in intervals:
            start, stop, needs_masking, _ = interval
            size = stop - start
            if size == 1:
                len_keep = (torch.rand(1, device=x.device) > mask_ratio).int().item()
            else:
                # round up
                len_keep = int(size * (1 - mask_ratio) + 0.5)

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, size], device=x.device)
            if needs_masking:
                noise = torch.rand(B, size, device=x.device)  # noise in [0, 1]
                # sort noise for each sample
                ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
                ids_restore = torch.argsort(ids_shuffle, dim=1)
                # keep the first subset
                ids_keep = ids_shuffle[:, :len_keep]
                mask[:, :len_keep] = 0
            else:
                ids_restore = torch.arange(size, device=x.device).repeat(B, 1)
                ids_keep = torch.arange(size, device=x.device).repeat(B, 1)
                mask[:, :] = 0

            # Update the index to match real position in original tensor
            ids_restore += start
            ids_keep += start
            all_ids_restore.append(ids_restore)
            all_ids_keep.append(ids_keep)
            all_masks.append(mask)

        ids_restore = torch.cat(all_ids_restore, dim=1)
        ids_keep = torch.cat(all_ids_keep, dim=1)
        mask = torch.cat(all_masks, dim=1)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def consistent_masking(
        self, x: torch.Tensor, intervals: list[tuple[int, int, bool, int]], mask_ratio: float, type: str = "random"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _T, D = x.shape

        all_ids_restore = [None] * len(intervals)
        all_ids_keep = [None] * len(intervals)
        all_masks = [None] * len(intervals)
        base_num_patches = None
        base_keep_ids = []

        # Sort by larget GSD and apply spatial masks to those first
        idxs = np.argsort(np.array(intervals)[:, 3])
        for _idx_num, idx in enumerate(idxs):
            start, stop, needs_masking, num_patches = intervals[idx]
            size = stop - start
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, size], device=x.device)
            if needs_masking:
                if base_num_patches is None:
                    base_num_patches = num_patches
                    if type == "random":
                        # Setup the initial masking on the last number of patches
                        noise = torch.rand(B, size, device=x.device)  # noise in [0, 1]
                        # sort noise for each sample
                        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
                        ids_restore = torch.argsort(ids_shuffle, dim=1)
                        # keep the first subset
                        len_keep = int(size * (1 - mask_ratio))
                        ids_keep = ids_shuffle[:, :len_keep]
                        mask[:, :len_keep] = 0
                    elif type == "structured":
                        # Obtain a border around the image, make it randomly sized across batches
                        if num_patches > 2:
                            border = max(
                                1,
                                min(num_patches // 2 - 1, torch.normal(num_patches // 4, 1, size=(1, 1)).int().item()),
                            )
                            all_idxs = torch.arange(size, device=x.device).reshape(num_patches, num_patches)
                            ids_keep = all_idxs[border:-border, border:-border].flatten().repeat(B, 1)
                            ids_restore = torch.arange(size, device=x.device).repeat(B, 1)

                            row_num, col_num = ids_keep.shape
                            idx0 = torch.arange(row_num).reshape(-1, 1).repeat(1, col_num).flatten()
                            idx1 = ids_keep.flatten()
                            mask[idx0, idx1] = 0

                    # Copy this base set of indices to all other bands
                    base_keep_ids = ids_keep.clone()
                else:
                    # Convert each of the base_keep_ids into ids for the current entry
                    ratio = int(num_patches / base_num_patches)
                    ids_keep = base_keep_ids.repeat(int(ratio * ratio), 1, 1).permute(1, 2, 0)
                    for idx_x in range(ratio):
                        for idx_y in range(ratio):
                            entry_idx = idx_x * ratio + idx_y
                            row_scale = torch.div(ids_keep[:, :, entry_idx], base_num_patches, rounding_mode="floor")
                            col_scale = ids_keep[:, :, entry_idx] % base_num_patches
                            ids_keep[:, :, entry_idx] = (
                                row_scale * ratio * num_patches + col_scale * ratio + idx_x * num_patches + idx_y
                            )
                    ids_keep = ids_keep.reshape(ids_keep.shape[0], -1)
                    ids_restore = torch.arange(size, device=x.device).repeat(B, 1)

                    # Convert ids_keep into indexes for mask
                    row_num, col_num = ids_keep.shape
                    idx0 = torch.arange(row_num).reshape(-1, 1).repeat(1, col_num).flatten()
                    idx1 = ids_keep.flatten()
                    mask[idx0, idx1] = 0
            else:
                ids_keep = torch.arange(size, device=x.device).repeat(B, 1)
                ids_restore = torch.arange(size, device=x.device).repeat(B, 1)
                mask[:, :] = 0

            # Update the index to match real position in original tensor
            ids_restore += start
            ids_keep += start
            all_ids_restore[idx] = ids_restore
            all_ids_keep[idx] = ids_keep
            all_masks[idx] = mask

        ids_restore = torch.cat(all_ids_restore, dim=1)
        ids_keep = torch.cat(all_ids_keep, dim=1)
        mask = torch.cat(all_masks, dim=1)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def full_masking(self, x, intervals):
        B, T, D = x.shape
        ids_restore = torch.arange(T, device=x.device).repeat(B, 1)
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([B, T], device=x.device)
        for interval in intervals:
            start, stop, needs_masking, _ = interval
            if needs_masking:
                mask[:, start:stop] = 1
        mask_idxs = (mask[0, :] == 0).nonzero().repeat(B, 1, D)
        x_masked = torch.gather(x, dim=1, index=mask_idxs)
        return x_masked, mask, ids_restore

    def mixed_masking(self, x, intervals, mask_ratio):
        B, _T, D = x.shape  # batch, sequence length, dim

        all_ids_restore, all_ids_keep, all_masks = [], [], []

        # Compute total number of patches
        # num_patches = sum([stop - start for start, stop, needs_masking, _ in intervals if not needs_masking])

        for interval in intervals:
            start, stop, needs_masking, _ = interval
            size = stop - start

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, size], device=x.device)
            if needs_masking:
                ids_restore = torch.arange(size, device=x.device).repeat(B, 1)
                ids_keep = None
                mask[:, :] = 1
            else:
                len_keep = int(size * (1 - mask_ratio))
                noise = torch.rand(B, size, device=x.device)  # noise in [0, 1]
                # sort noise for each sample
                ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
                ids_restore = torch.argsort(ids_shuffle, dim=1)
                # keep the first subset
                ids_keep = ids_shuffle[:, :len_keep]
                mask[:, :len_keep] = 0

            # Update the index to match real position in original tensor
            ids_restore += start
            all_ids_restore.append(ids_restore)
            if ids_keep is not None:
                ids_keep += start
                all_ids_keep.append(ids_keep)
            all_masks.append(mask)

        ids_restore = torch.cat(all_ids_restore, dim=1)
        ids_keep = torch.cat(all_ids_keep, dim=1)
        mask = torch.cat(all_masks, dim=1)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def inference_masking(self, x, band_metadata, mask_ratio):
        # Only supports batch size of 1, might be possible with B>1 some torch-foo

        B, _T, D = x.shape  # batch, sequence length, dim
        assert B == 1, "Inference masking only supports batch size of 1"
        assert not self.training, "Inference masking should only be used during evaluation"

        all_ids_restore, all_ids_keep, all_masks = [], [], []

        # Compute total number of patches
        # num_patches = sum([
        #     stop - start for start, stop, needs_masking, _ in band_metadata.values() if not needs_masking
        # ])
        for band_name, interval in band_metadata.items():
            start, stop, needs_masking, _ = interval
            size = stop - start

            ids_keep = None
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, size], device=x.device)
            if needs_masking:
                ids_restore = torch.arange(size, device=x.device).repeat(B, 1)
                # ids_keep = None
                # mask[:, :] = 1
            else:
                nans = x[:, start:stop].isnan().any(dim=-1)
                len_keep = (~nans).sum().item()
                if nans.any():
                    logger.info(f"Eval: masking {nans.sum()} nan patches in group {band_name} out of {size} patches")

                ids_shuffle = torch.argsort(nans.int(), dim=1)  # ascend: small is keep, large is remove
                ids_restore = torch.argsort(ids_shuffle, dim=1)

                if len_keep > 1:
                    ids_keep = ids_shuffle[:, :len_keep]
                    mask[:, :len_keep] = 0

            # Update the index to match real position in original tensor
            ids_restore += start
            all_ids_restore.append(ids_restore)
            if ids_keep is not None:
                ids_keep += start
                all_ids_keep.append(ids_keep)
            all_masks.append(mask)

        ids_restore = torch.cat(all_ids_restore, dim=1)
        ids_keep = torch.cat(all_ids_keep, dim=1)
        mask = torch.cat(all_masks, dim=1)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def compute_band_metadata(
        self, group_embed: dict[str, torch.Tensor], spectral_mask_ratio: float
    ) -> OrderedDict[str, OrderedDict[str, tuple[int, int, bool, int]]]:
        """Compute per-product, per-band metadata - range of patch embed indices
        for a given band, whether or not spatial mask should be applied to a given
        band, and number of patches. This is computed every batch to support
        multi-dataset training schemes.
        Ex.
        metadata = {
            "S2": {
                "Red": (12, 34, True, 4),
                "Green": (34, 56, True, 4)
            },
            "NAIP": {
                "Red": (12, 34, False, 8),
                "Green": (34, 56, False, 8)
            }
        }
        """
        masked_groups = []
        if self.masking_schema:
            all_groups = list(group_embed.keys())
            # Specify the groups to mask by index
            if isinstance(self.masking_schema["spectral"], list):
                masked_groups.extend([all_groups[idx] for idx in self.masking_schema["spectral"]])
            # Specify predefined spectral masking pattern
            elif isinstance(self.masking_schema["spectral"], str):
                if self.masking_schema["spectral"] == "random_groups":
                    masked_groups = np.random.choice(
                        all_groups, size=int(len(all_groups) * spectral_mask_ratio), replace=False
                    )
                elif self.masking_schema["spectral"] == "one_group":
                    masked_groups = np.random.choice(all_groups)
                elif self.masking_schema["spectral"] == "all_but_one_group":
                    masked_groups = np.random.choice(all_groups, size=len(all_groups) - 1, replace=False)
                elif self.masking_schema["spectral"] == "all":
                    masked_groups.extend(all_groups)
                else:
                    msg = "Invalid spectral masking scheme provided"
                    raise Exception(msg)

        running_idx = 0
        metadata = OrderedDict()
        for (
            group,
            data,
        ) in group_embed.items():
            num_patches = data.shape[1]
            incr = running_idx + num_patches
            metadata[group] = (
                running_idx,
                incr,
                group in masked_groups,
                num_patches**0.5,
            )
            running_idx = incr

        return metadata

    def mask(
        self,
        x: torch.Tensor,
        band_metadata: OrderedDict[str, OrderedDict[str, tuple[int, int, bool, int]]],
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.masking_schema is None:
            # This is fine
            return self.random_masking_old(x, [(0, x.shape[1], True, False, 0)], mask_ratio)

        interval_list = list(band_metadata.values())
        if self.masking_schema["spatial"] == "random":
            # This is fine
            return self.random_masking_old(x, [(0, x.shape[1], True, False, 0)], mask_ratio)
        elif self.masking_schema["spatial"] == "random_same_ratio":
            # This is fine
            return self.random_masking_fixed(x, interval_list, mask_ratio)
        elif self.masking_schema["spatial"] == "random_consistent":
            return self.consistent_masking(x, interval_list, mask_ratio, type="random")
        elif self.masking_schema["spatial"] == "structured":
            return self.consistent_masking(x, interval_list, mask_ratio, type="structured")
        elif self.masking_schema["spatial"] == "all":
            return self.full_masking(x, interval_list)
        elif self.masking_schema["spatial"] == "mixed":
            return self.mixed_masking(x, interval_list, mask_ratio)
        elif self.masking_schema["spatial"] == "inference":
            return self.inference_masking(x, band_metadata, mask_ratio)
        else:
            msg = "Invalid spatial masking scheme"
            raise Exception(msg)

    def forward_encoder(
        self,
        x: dict[str, torch.Tensor],
        metadata: dict[str, dict[str, int]],
        ground_cover: int | None,
        mask_ratio: float,
        spectral_mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, list[str]], dict[str, dict[str, int]]]:
        # overwrite to add random masking
        # B: Batch size, H: Height, W: Width, p: patch size
        # D: Embedding dimension, N_n: Number of patches for each product_band
        # T: Sequence length (number of patches * number of spectral bands)
        # r: mask ratio
        # x = {'product:band': (B, 1, H, W), ...}

        # extract patches
        # Patch projection{'product:band': (B, N_n, D), ...}
        if self.training and self.use_flexivit:
            # Use random flexivit patch sizes
            # NOTE: some products/groups might dissapear when using token budget
            patch_embed = self.ind_patch_embed(
                x,
                patch_sizes=None,
                token_budget=self.token_budget,
                device=self.device,
            )
        else:
            patch_sizes = self.get_patch_sizes(
                method=self.select_patch_strategy,
                x=x,
                patch_sizes=self.ind_patch_embed.patch_size_seqs,
                ground_cover=ground_cover,
            )

            # Use fixed defined patch sizes
            patch_embed = self.ind_patch_embed(x, patch_sizes=patch_sizes, device=self.device)

        # find available groups in input
        available_groups = self.get_available_groups(patch_embed)  # {'group0': [product_band, ...], ...}

        channel_params = self.get_channel_params(patch_embed, metadata, ground_cover=ground_cover)

        group_embed = self.aggregate_by_group(patch_embed, available_groups)  # {'group0': (B, N_n, D), ...}

        if len(group_embed) == 0:
            msg = (
                "No available groups found in input data for encoder forward pass."
                f"Available groups: {available_groups}, channel params: {channel_params}"
            )
            raise ValueError(msg)

        # add additional embedding # {'group0': (N_n, D), ...}
        auxilliary_embed = self.get_encoder_auxilliary_embed(
            x,
            available_groups,
            channel_params,
        )

        # compute metadata before concat bands
        band_metadata = self.compute_band_metadata(group_embed, spectral_mask_ratio)

        # Determine the bands + products you want to apply masking to
        x = torch.cat(
            [group_embed[group_name].add_(auxilliary_embed[group_name]) for group_name in group_embed.keys()], dim=1
        )  # (B, T, D)

        orig_seq_len = x.shape[1]

        # Apply mask
        x, mask, ids_restore, ids_keep = self.mask(x, band_metadata, mask_ratio)

        # append cls token
        if self.cls_token_type == "token":
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # (N, (1-r)*T + 1, D)

        if self.encoder_pos_type == "alibi":
            alibi = get_alibi_thor(
                channel_params,
                available_groups,
                slopes=self.encoder_slopes,
                offset=self.num_prefix_tokens,
                device=x.device,
            )
            alibi = apply_mask_to_alibi(
                alibi=alibi,
                ids_keep_queries=ids_keep,
                ids_keep_keys=ids_keep,
                batch_size=x.shape[0],
                orig_seq_len=orig_seq_len,
                masked_seq_len=x.shape[1] - self.num_prefix_tokens,
                attention_heads=self.num_heads,
            )  # .contiguous()
            if self.cls_token_type == "token":
                alibi = alibi_cls_token_pad(alibi)
        else:
            alibi = None

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, alibi)
        x = self.norm(x)

        if self.cls_token_type == "token":
            # remove cls token
            cls_token = x[:, 0, :]
            # x = x[:, 1:, :]  # (N, T, dD)
        elif self.cls_token_type == "pooled":
            cls_token = x.mean(dim=1)
        else:
            cls_token = None

        if cls_token is not None:
            cls_token = self.norm(cls_token)

        return x, mask, ids_restore, cls_token, available_groups, channel_params

    def forward_decoder(
        self,
        laten: torch.Tensor,
        ids_restore: torch.Tensor,
        mask: torch.Tensor,
        original_input: dict[str, torch.Tensor],
        land_cover_data: dict[str, torch.Tensor],
        available_groups: dict[str, list[str]],
        channel_params: dict[str, dict[str, int]],
        ground_cover: int,
        return_cls_token: bool = False,
    ) -> (
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]
        | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]
    ):
        # dD: decoder embedding dimension
        # project to decoder dimension
        x = self.decoder_embed(laten)  # (B, (1-r)*T + num_prefix_tokens, dD)

        # append mask tokens to sequence
        num_mask_tokens = ids_restore.shape[1] - x.shape[1] + self.num_prefix_tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], num_mask_tokens, 1)  # expand to (B, T*r, dD)
        x_ = torch.cat([x[:, self.num_prefix_tokens :, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, : self.num_prefix_tokens, :], x_], dim=1)  # append cls token, if existing  (B, 1 + T, D)

        # Get auxilliary embedding [(B, T, dD)]
        auxilliary_embed = self.get_decoder_auxilliary_embed(
            original_input,
            available_groups,
            channel_params,
        )
        x[:, self.num_prefix_tokens :, :].add_(auxilliary_embed)  # add additional embedding

        if self.decoder_pos_type == "alibi":
            alibi = get_alibi_thor(
                channel_params,
                available_groups,
                slopes=self.decoder_slopes,
                offset=self.num_prefix_tokens,
                device=x.device,
            )
            if self.cls_token_type == "token":
                alibi = alibi_cls_token_pad(alibi)
            alibi = alibi.expand(x.shape[0], -1, -1, -1)  # .contiguous()
        else:
            alibi = None

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, alibi)
        x = self.decoder_norm(x)

        if self.cls_token_type == "token":
            # remove cls token
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]  # (N, T, dD)
        elif self.cls_token_type == "pooled" and return_cls_token:
            cls_token = x.mean(dim=1)
        else:
            cls_token = None

        if cls_token is not None:
            cls_token = self.decoder_norm(cls_token)

        start_idx = 0
        out = {}
        land_cover_out = {}
        land_cover_scale = {}
        scale = {}
        mask_out = {}
        # Important that we iterate through this in the same order we encoded
        # We iterate through all possible groups, but only use the ones that are available
        # This is to make sure that the order of the bands is consistent and that the shape of the output is correct
        for group_name, full_group_members in self.groups.items():  # available_groups.items():
            if group_name not in available_groups:
                continue
            product_band = available_groups[group_name][0]
            # band_ground_cover = int(original_input[product_band].shape[-1] * channel_params[product_band]["GSD"])
            num_patches = channel_params[product_band]["num_patch"] ** 2
            # (B * N_n , dD)
            x_ = x[:, start_idx : start_idx + num_patches, :].reshape(-1, x.shape[-1])
            # {(B, N_n, len(groups),  p**2), ...}
            new_patch_size = channel_params[product_band]["patch_size"]
            decoded_group_out, group_scale = self.flexivit_decode_pred(
                x_,
                group_name,
                (new_patch_size, new_patch_size),
                return_pinv=True,
            )
            for task in self.land_cover_decoders:
                if task not in land_cover_data:
                    continue
                if group_name in self.land_cover_decoders[task]:
                    if task not in land_cover_out:
                        land_cover_out[task] = {}
                    if task not in land_cover_scale:
                        land_cover_scale[task] = {}

                    land_cover_gsd = int(ground_cover / land_cover_data[task].shape[-1])
                    if land_cover_gsd != self.land_cover_tasks[task]["gsd"] and not ("dem" in task or "INC" in task):
                        logger.warning(
                            f"Ground cover {ground_cover} for task {task} does not match "
                            f"land cover gsd {land_cover_gsd} for task {task}, expected "
                            f"{self.land_cover_tasks[task]['gsd']}"
                        )
                        continue
                    new_patch_size = int(ground_cover / land_cover_gsd / channel_params[product_band]["num_patch"])
                    if new_patch_size * land_cover_gsd * channel_params[product_band]["num_patch"] != ground_cover:
                        logger.debug(
                            f"patch size {new_patch_size}, with {channel_params[product_band]['num_patch']} patches not compatible with ground cover {ground_cover} for task {task}"
                        )
                        continue
                    elif new_patch_size > self.land_cover_tasks[task]["max_pred_patch_size"]:
                        continue
                    elif new_patch_size < self.flexivit_ref_patch_size:
                        continue
                    old_patch_size = self.land_cover_tasks[task]["groups"][group_name]["patch_size"]
                    lc_out, lc_task_scale = self.flexivit_decode_pred.functional(
                        x_,
                        self.land_cover_decoders[task][group_name],
                        (old_patch_size, old_patch_size),
                        (new_patch_size, new_patch_size),
                        return_pinv=True,
                    )
                    land_cover_out[task][group_name] = lc_out.reshape(x.shape[0], num_patches, -1)
                    land_cover_scale[task][group_name] = lc_task_scale

            decoded_group_out = decoded_group_out.reshape(x.shape[0], num_patches, len(full_group_members), -1)
            for idx, product_band in enumerate(full_group_members):
                if product_band not in available_groups[group_name]:
                    continue
                out[product_band] = decoded_group_out[:, :, idx, :]
                scale[product_band] = group_scale
                # same mask for all bands in the same group
                mask_out[product_band] = mask[:, start_idx : start_idx + num_patches]
            start_idx += num_patches

        if start_idx != x.shape[1]:
            msg = f"Expected {x.shape[1]} patches, we have only processed {start_idx} patches"
            raise ValueError(msg)

        if return_cls_token:
            return out, scale, mask_out, land_cover_out, land_cover_scale, cls_token
        return out, scale, mask_out, land_cover_out, land_cover_scale

    def find_global_available_groups(
        self, local_grups: list[str], device, min_process_ratio: float = 1.0, max_groups: int = 4
    ) -> list[str]:
        """Find available groups across all devices.

        Args:
            local_grups: List of locally available groups
            device: Device to put tensors on
            min_process_ratio: Minimum ratio of processes that need to have a group (default: 1.0)
                             e.g., 0.5 means at least half of processes need the group
            max_groups: Maximum number of groups to select (default: 4)
        """
        world_size = dist.get_world_size(self.dist_group)
        min_process_count = math.ceil(world_size * min_process_ratio)

        # [group0, group1, ...] -> [0, 1, ...]
        group_idx = [int(group.split("group")[-1]) for group in local_grups]

        # pad to the same length across all devices [0, 1, ...] -> [0, 1, ..., -1, -1, ...]
        group_idx = group_idx + [-1] * (len(self.groups) - len(group_idx))
        group_idx = torch.tensor(group_idx, dtype=torch.int, device=device)
        all_group_idx = [torch.ones_like(group_idx) * -1 for _ in range(world_size)]
        dist.all_gather(all_group_idx, group_idx, group=self.dist_group)

        all_group_idx = torch.cat(all_group_idx, dim=0)
        all_group_idx, counts = all_group_idx.unique(return_counts=True)
        # find groups that meet the minimum process count requirement
        intersection = all_group_idx[torch.where(counts.ge(min_process_count))]
        # [0, 1, ...] -> [group0, group1, ...]
        groups = [f"group{idx}" for idx in sorted(intersection.tolist()) if idx != -1]

        if len(groups) > max_groups:
            if dist.get_rank(self.dist_group) == 0:
                logger.info(f"Found {len(groups)} groups, limiting to {max_groups} groups")
                random_selection = torch.randperm(len(groups), device=device, dtype=torch.long)[:max_groups]
                dist.broadcast(random_selection, 0, self.dist_group)
            else:
                random_selection = torch.zeros(max_groups, device=device, dtype=torch.long)
                dist.broadcast(random_selection, 0, self.dist_group)
            groups = [groups[idx] for idx in random_selection.tolist()]

        return groups

    def group_latent(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
        mask: torch.Tensor,
        mask_ratio: float,
        channel_params: dict[str, dict[str, int]],
        available_groups: dict[str, list[str]],
        land_cover_data: dict[str, torch.Tensor] | None = None,
        ground_cover: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, dict[str, int] | None, dict[str, int] | None]:
        """Group sequence of encoder tokens by product groups.

        I.e. laten: (B, 1 + T, D) -> {'group0': (B, N_n, D), ...}

        Note that the encoder tokens are already masked and shuffled,
        so we need to reconstruct the full tensor before selecting the groups.

        Args:
            x: Encoder tokens
            ids_restore: Indices to restore the original order
            mask: Mask indicating which tokens are masked
            mask_ratio: Ratio of tokens that are masked
            channel_params: Channel parameters
            available_groups: Available groups
            land_cover_data: Dictionary mapping land cover task names to ground truth tensors
            ground_cover: Ground cover size in meters

        Returns:
            tuple: (grouped_latent, land_cover_patch_hist, lc_classes, patch_sizes)
                - grouped_latent: Dictionary mapping from group names to latent representations (unmasked) {"group0": (B, N_n, D), ...}
                - land_cover_patch_hist: Dictionary mapping from group names to land cover patch histograms
                  if land cover data is available {"group0": (B, N_n, num_classes), ...}, will be None if not using land cover contrastive loss
                - lc_classes: Dictionary mapping from group names to number of land cover classes {"group0": num_classes, ...}, will be None if not using land cover contrastive loss
                - patch_sizes: Dictionary mapping from group names to patch sizes {"group0": patch_size, ...}, will be None if not using land cover contrastive loss
        """

        global_available_groups = list(available_groups.keys())

        # Filter available groups based on num patches
        global_available_groups = [
            g_group
            for g_group in global_available_groups
            if (product_band := next(iter(self.groups[g_group]))) in channel_params
            and channel_params[product_band]["num_patch"] ** 2 * (1 - mask_ratio) >= self.contrast_min_num_patches
        ]

        # find available groups across all devices
        if dist.is_available() and dist.is_initialized():
            global_available_groups = self.find_global_available_groups(
                global_available_groups,
                x.device,
                min_process_ratio=self.min_process_ratio,
                max_groups=self.max_contrastive_groups,
            )

        # Process land cover data to determine dominant land cover for each patch if provided
        land_cover_patch_hist = None
        lc_classes = None
        patch_sizes = None
        if hasattr(self, "land_cover_tasks") and self.contrast_type in ["land_cover", "land_cover_v2"]:
            land_cover_patch_hist = {}
            lc_classes = {}
            patch_sizes = {}

            # Process each land cover task
            # Important that we iterate through this in the same order for all devices
            for group_name in global_available_groups:
                for task_name in self.land_cover_tasks:
                    if task_name == "SCL":
                        # Skip SCL task to not confuse contrastive loss with groups
                        # I.e, on device 0 we use world cover and on device 1 we use SCL...
                        continue
                    elif "dem" in task_name or "INC" in task_name:
                        # Skip DEM tasks for now, as they are not supported
                        continue
                    if group_name not in self.land_cover_tasks[task_name]["groups"]:
                        # Skip if group is not in land cover task
                        continue

                    # Skip if land cover data is not available for this task or group is missing
                    if task_name not in land_cover_data or group_name not in available_groups:
                        # Create empty tensor for dominant land cover
                        land_cover_patch_hist[group_name] = torch.zeros(
                            (x.shape[0], 0, self.land_cover_tasks[task_name]["num_classes"]),
                            device=x.device,
                            dtype=torch.long,
                        )
                        lc_classes[group_name] = self.land_cover_tasks[task_name]["num_classes"]
                        patch_sizes[group_name] = 1
                        continue

                    task_data = land_cover_data[task_name]

                    product_band = available_groups[group_name][0]

                    # Calculate patch size for this task and group
                    task_gsd = self.land_cover_tasks[task_name]["gsd"]
                    patch_size = int(ground_cover / task_gsd / channel_params[product_band]["num_patch"])

                    # Check if patch size is compatible with ground cover
                    if patch_size * task_gsd * channel_params[product_band]["num_patch"] != ground_cover:
                        land_cover_patch_hist[group_name] = torch.zeros(
                            (x.shape[0], 0, self.land_cover_tasks[task_name]["num_classes"]),
                            device=x.device,
                            dtype=torch.long,
                        )
                        lc_classes[group_name] = self.land_cover_tasks[task_name]["num_classes"]
                        patch_sizes[group_name] = 1
                        logger.debug(
                            f"patch size {patch_size} not compatible with ground cover {ground_cover} for task {task_name}"
                        )
                        continue

                    # Patchify the land cover data
                    patched_data = self.patchify(task_data, patch_size, 1)  # (B, N_n, p**2)

                    # Convert to one-hot
                    patched_data = F.one_hot(
                        patched_data.long(), num_classes=self.land_cover_tasks[task_name]["num_classes"]
                    )

                    for ignore_val in self.land_cover_tasks[task_name]["groups"][group_name]["ignore"]:
                        patched_data[:, :, :, ignore_val] = 0

                    # Sum land cover counts across patches
                    patched_data_sum = patched_data.sum(dim=-2)

                    # Apply the mask to only keep unmasked patches
                    group_mask = ~mask[product_band].bool()
                    land_cover_patch_hist[group_name] = patched_data_sum[group_mask].reshape(
                        x.shape[0], -1, self.land_cover_tasks[task_name]["num_classes"]
                    )
                    if land_cover_patch_hist[group_name].shape[1] == 1:
                        land_cover_patch_hist[group_name] = torch.zeros(
                            (x.shape[0], 0, self.land_cover_tasks[task_name]["num_classes"]),
                            device=x.device,
                            dtype=torch.long,
                        )
                    lc_classes[group_name] = self.land_cover_tasks[task_name]["num_classes"]
                    patch_sizes[group_name] = patch_size

        # append zero tokens to sequence
        zero_token = torch.zeros((1, x.shape[-1]), device=x.device, dtype=x.dtype)
        zero_tokens = zero_token.repeat(
            x.shape[0], ids_restore.shape[1] + self.num_prefix_tokens - x.shape[1], 1
        )  # expand to (B, T*r, dD)
        x_ = torch.cat([x[:, self.num_prefix_tokens :, :], zero_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        start_idx = 0
        grouped_latent = {}
        # Important that we iterate through this in the same order we encoded
        # We iterate through all possible groups, but only use the ones that are available
        # This is to make sure that the order of the bands is consistent and that the shape of the output is correct
        for group_name in self.groups:
            if group_name in available_groups:
                product_band = available_groups[group_name][0]
                num_patches = channel_params[product_band]["num_patch"] ** 2
                # (B * N_n , dD)
            if group_name in global_available_groups and group_name in available_groups:
                group_mask = ~mask[product_band].bool()
                group_x = x[:, start_idx : start_idx + num_patches, :]
                grouped_latent[group_name] = group_x[group_mask].reshape(x.shape[0], -1, x.shape[-1])
            elif group_name in global_available_groups and group_name not in available_groups:
                grouped_latent[group_name] = torch.zeros((x.shape[0], 0, x.shape[-1]), device=x.device, dtype=x.dtype)

            if group_name in available_groups:
                start_idx += num_patches

        if start_idx != x.shape[1]:
            msg = f"Expected {x.shape[1]} patches, we have only processed {start_idx} patches"
            raise ValueError(msg)

        return grouped_latent, land_cover_patch_hist, lc_classes, patch_sizes

    def reconstruction_loss(
        self,
        inpudict: dict[str, torch.Tensor],
        pred_dict: dict[str, torch.Tensor],
        scale_dict: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor],
        available_groups: dict[str, list[str]],
        channel_params: dict[str, dict[str, int]],
    ):
        """
        inpudict: {'product:band': (B, 1, H, W), ...}
        pred_dict: {'product:band': (B, N_n, p**2), ...}
        mask: {'product:band': (B, N_n), ...}

        Vanilla implementation of reconstruction loss, we reconstruct every original spectral bands,
        could make change later to something else.
        """
        total_loss = 0
        total_fft_loss = 0
        num_remove = 0
        if self.log_per_group:
            total_loss_per_group = Counter()
            num_remove_per_group = Counter()
        # NOTE this can be subset of input dict
        for product_band in pred_dict:  # noqa: PLC0206
            spectral_img = inpudict[product_band]
            pred = pred_dict[product_band]
            mask = mask_dict[product_band]
            patch_size = channel_params[product_band]["patch_size"]
            product_group = self.group_lookup[product_band]
            batch_size = pred.shape[0]

            target = self.patchify(spectral_img, patch_size, 1)  # [B, N_n, p**2]

            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5

            scale = scale_dict[product_band].float()

            err = pred.float() - target.float()

            # Scale the error with the pseudo-inverse of the interpolation matrix
            err = torch.einsum("BND,DE->BNE", err, scale)

            loss = err**2
            loss = loss.mean(dim=-1)  # [B, N_n], mean loss per patch

            if self.use_fft_loss:
                target_fft = torch.fft.fft2(target.reshape(batch_size, -1, patch_size, patch_size).float()).reshape(
                    batch_size, -1, patch_size**2
                )
                pred_fft = torch.fft.fft2(pred.reshape(batch_size, -1, patch_size, patch_size).float()).reshape(
                    batch_size, -1, patch_size**2
                )

                target_mag = torch.abs(target_fft)
                pred_mag = torch.abs(pred_fft)

                loss_fft = torch.abs(target_mag - pred_mag)  # L1 loss
                loss_fft = loss_fft.mean(dim=-1)  # [B, N_n], mean loss per patch

                total_fft_loss += (loss_fft * mask).sum()

            total_loss += (loss * mask).sum()
            num_remove += mask.sum()  # mean loss on removed patches
            if self.log_per_group:
                total_loss_per_group[product_group] += (loss * mask).sum().item()
                num_remove_per_group[product_group] += mask.sum().item()

        if not self.use_fft_loss:
            total_fft_loss = torch.zeros_like(total_loss)

        # Average loss
        if self.log_per_group:
            # find available groups in input
            available_groups = self.get_available_groups(inpudict)  # {'group0': [product_band, ...], ...}
            per_group = {}
            for group_name in self.groups.keys():
                if group_name in available_groups:
                    nr = num_remove_per_group[group_name]
                    if nr == 0:
                        continue
                    per_group[group_name] = total_loss_per_group[group_name] / nr
                # else:
                #     per_group[group_name] = 0.0
        else:
            per_group = None

        if num_remove > 0:
            total_loss = total_loss / num_remove
            total_fft_loss = total_fft_loss / num_remove
        return total_loss, total_fft_loss, per_group

    def map_targets(
        self, targets, mapping: torch.IntTensor, num_classes: int, ignore_classes: list[int]
    ) -> torch.Tensor:
        """Map original class indices to valid class indices"""
        device = targets.device
        if mapping.device != device:
            mapping = mapping.to(device)

        # Map targets, keeping -1 for ignored classes
        mapped_targets = mapping[targets.clamp(0, num_classes - 1)]

        # Set invalid/ignored classes to -1
        invalid_mask = targets < 0
        for ignore_cls in ignore_classes:
            invalid_mask |= targets == ignore_cls
        mapped_targets[invalid_mask] = -1

        return mapped_targets

    def invert_map_targets(
        self, targets, mapping: torch.IntTensor, num_classes: int, ignore_classes: list[int]
    ) -> torch.Tensor:
        """Map valid class indices back to original class indices"""
        device = targets.device
        if mapping.device != device:
            mapping = mapping.to(device)

        inv_mapping = torch.full((num_classes,), -1, dtype=torch.int, device=device)
        for orig_cls in range(num_classes):
            mapped_cls = mapping[orig_cls]
            if mapped_cls >= 0 and mapped_cls < num_classes:
                inv_mapping[mapped_cls] = orig_cls

        # Map targets, keeping -1 for ignored classes
        inv_mapped_targets = inv_mapping[targets.clamp(0, num_classes - 1)]

        # Set invalid/ignored classes to 0
        invalid_mask = targets < 0
        for ignore_cls in ignore_classes:
            invalid_mask |= inv_mapped_targets == ignore_cls
        inv_mapped_targets[invalid_mask] = 0

        return inv_mapped_targets

    def land_cover_mask_loss(
        self,
        land_cover_data: torch.Tensor,
        pred_dict: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor],
        scale_dict: dict[str, torch.Tensor],
        land_cover_mappings: dict[str, torch.IntTensor],
        channel_params: dict[str, dict[str, int]],
        ground_cover: int,
        available_groups: dict[str, list[str]],
        lc_params: dict[str, Any],
    ):
        """
        land_cover data: (B, 1, H, W)
        pred_dict: {'group{idx}': (B, N_n, C*p**2), ...}
        mask: {'product:band': (B, N_n), ...}

        """

        total_loss = 0
        total_loss_per_group = Counter()
        num_remove = 0
        num_remove_per_group = Counter()
        lc_mask = {}
        for group_name, group_members in available_groups.items():
            if group_name not in pred_dict:
                continue
            product_band = group_members[0]
            lc_group_params = lc_params["groups"][group_name]

            lc_gsd = int(ground_cover / land_cover_data.shape[-1])
            patch_size = int(ground_cover // float(lc_gsd) // channel_params[product_band]["num_patch"])
            if patch_size * lc_gsd * channel_params[product_band]["num_patch"] != ground_cover:
                logger.warning(
                    "patch size does not match ground cover"
                    f"Patch size {patch_size} does not match ground cover {ground_cover} and GSD {lc_gsd}, num patch {channel_params[product_band]['num_patch']}"
                )
                continue

            pred = pred_dict[group_name]

            B, N, D = pred.shape

            pred = pred.reshape(B * N, lc_group_params["valid_classes"], D // lc_group_params["valid_classes"])

            if lc_params["loss_on_patches"] == "unmasked":
                mask = torch.logical_not(mask_dict[product_band])
            elif lc_params["loss_on_patches"] == "masked":
                mask = mask_dict[product_band]
            elif lc_params["loss_on_patches"] == "all":
                mask = torch.ones_like(mask_dict[product_band], dtype=torch.bool)
            else:
                msg = f"Unknown land cover patch loss method: {lc_params['loss_on_patches']}"
                raise ValueError(msg)

            if lc_params["loss_type"] == "cross_entropy":
                pred = pred.reshape(B * N, lc_group_params["valid_classes"], patch_size, patch_size)

                target = self.patchify(land_cover_data, patch_size, 1)  # [B, N_n, p**2]
                target = self.map_targets(
                    target.flatten().long(),
                    land_cover_mappings[group_name],
                    lc_params["num_classes"],
                    lc_group_params["ignore"],
                )  # [B * N_n * p**2]
                target = target.reshape(B * N, patch_size, patch_size)

                # Count ignore values in each patch
                count_neg_ones = (target == -1).sum(dim=(-2, -1))  # Shape: (B*N,)
                sum_not_ignore = (patch_size**2 - count_neg_ones).reshape(B, N)

                loss = F.cross_entropy(
                    pred.float(),
                    target,
                    reduction="none",
                    ignore_index=-1,
                    label_smoothing=lc_params["label_smoothing"],
                ).reshape(B, N, -1)

                scale = scale_dict[group_name].float()
                ratio = scale.shape[1] / scale.shape[0]
                loss = torch.einsum("BND,DE->BNE", loss, scale)
                loss = loss.sum(dim=-1) / (ratio * sum_not_ignore).clamp(min=1)  # [B, N_n]

            elif lc_params["loss_type"] == "mse":
                target = self.patchify(land_cover_data, patch_size, lc_params["num_classes"])  # [B, N_n, C*p**2]

                target_nan_mask = target.isnan()
                target_nan_mask = target_nan_mask.sum(dim=-1) > 0
                mask = mask & ~target_nan_mask

                target = target.reshape(B * N, lc_params["num_classes"], patch_size**2)

                pred = torch.where(mask.reshape(B * N)[:, None, None], pred, 0)
                target = torch.where(mask.reshape(B * N)[:, None, None], target, 0)
                err = pred.float() - target.float()

                err = err.reshape(B, N, lc_params["num_classes"], -1)  # [B, N_n, C*p**2]

                scale = scale_dict[group_name].float()
                err = torch.einsum("BNCD,DE->BNCE", err, scale)
                err = err.reshape(B, N, -1)
                loss = err**2
                loss = loss.mean(dim=-1)  # [B, N_n], mean loss per patch

            total_loss += (loss * mask).sum()
            num_remove += mask.sum()  # mean loss on unmasked patches
            if self.log_per_group:
                total_loss_per_group[group_name] += (loss * mask).sum().item()
                num_remove_per_group[group_name] += mask.sum().item()

            lc_mask[group_name] = mask

        # Average loss
        if self.log_per_group:
            per_group = {}
            for group_name in available_groups:
                if group_name not in pred_dict:
                    continue
                nr = num_remove_per_group[group_name]
                if nr == 0:
                    continue
                per_group[group_name] = total_loss_per_group[group_name] / nr
        else:
            per_group = None

        if num_remove > 0:
            total_loss = total_loss / num_remove
        return total_loss, per_group, lc_mask

    @staticmethod
    def cyclic_encoding(x: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Cyclic encoding for x tensor.
        Args:
            x: Input tensor of shape (B, C, ...)
            scale: Scale for cyclic encoding.
        Returns:
            Tensor of shape (B, 2*C, ...) with cyclic encoding applied.
        """
        v = x / scale * 2 * math.pi  # Should be in the range [0, 2*pi]
        return torch.concat([torch.sin(v), torch.cos(v)], dim=1)

    def era5_loss(
        self, era5_pred: torch.Tensor, era5_labels: dict[str, torch.Tensor], era5_patch_size: int
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        era5_labels_stacked = torch.stack([era5_labels[band] for band in era5_labels], dim=1)
        era5_nan_mask_stacked = era5_labels_stacked.isnan()
        era5_targets = self.patchify(era5_labels_stacked, era5_patch_size, era5_labels_stacked.shape[1]).squeeze(1)
        era5_nan_mask = self.patchify(era5_nan_mask_stacked, era5_patch_size, era5_nan_mask_stacked.shape[1]).squeeze(1)
        era5_pred = self.patchify(era5_pred, era5_patch_size, era5_pred.shape[1]).squeeze(1)

        era5_pred = torch.where(era5_nan_mask, 0, era5_pred)
        era5_targets = torch.where(era5_nan_mask, 0, era5_targets)
        loss = (era5_pred.float() - era5_targets.float()) ** 2
        loss = torch.clamp(loss, 0, 10)

        if self.era5_return_dict:
            loss_per_prod = torch.zeros(era5_labels_stacked.shape[1], dtype=loss.dtype, device=loss.device)

            prod_indices = torch.arange(era5_labels_stacked.shape[1], device=loss.device)[None, ..., None, None].expand(
                era5_labels_stacked.shape
            )
            prod_indices = self.patchify(prod_indices, era5_patch_size, prod_indices.shape[1]).squeeze(1)

            loss_per_prod.scatter_reduce_(0, prod_indices.flatten(), loss.flatten(), reduce="sum")

            # Count non-NaN in PATCHIFIED data, not original
            non_nan_mask_patchified = ~era5_nan_mask  # This is already patchified
            num_non_nan_per_prod = torch.zeros(era5_labels_stacked.shape[1], dtype=torch.long, device=loss.device)

            # Count non-NaN per product using the same indexing scheme
            ones = torch.ones_like(non_nan_mask_patchified, dtype=torch.long)
            ones = torch.where(non_nan_mask_patchified, ones, 0)
            num_non_nan_per_prod.scatter_reduce_(0, prod_indices.flatten(), ones.flatten(), reduce="sum")

            num_non_nan_per_prod = torch.clamp(num_non_nan_per_prod, min=1)

            loss_per_prod = loss_per_prod / num_non_nan_per_prod

            loss_per_prod_dict = {}
            for idx, band in enumerate(era5_labels):
                loss_per_prod_dict[band] = loss_per_prod[idx]

            loss_total = loss.sum()
            non_nan_count = (~era5_nan_mask).sum()
            non_nan_count = torch.clamp(non_nan_count, min=1)

            loss_per_prod_dict["total"] = loss_total / non_nan_count

            return loss_per_prod_dict

        else:
            loss = loss.sum()
            non_nan_count = (~era5_nan_mask).sum()
            non_nan_count = torch.clamp(non_nan_count, min=1)
            return loss / non_nan_count

    def forward(
        self,
        x: dict[str, torch.Tensor],
        metadata: MetaData | dict[str, Any],
        mask_ratio: float,
        spectral_mask_ratio: float,
        era5_land_labels: dict[str, torch.Tensor] | None = None,
        return_cls_feats=False,
        return_channel_params=False,
    ) -> ThorMAEOutput:
        land_cover_data = {}
        for land_cover_task in self.land_cover_tasks:
            if land_cover_task in x:
                land_cover_data[land_cover_task] = x.pop(land_cover_task)

        if isinstance(metadata, MetaData):
            ground_cover = metadata.ground_cover
            month = metadata.month
            center_coords = metadata.center_coords
            s1_orbit_direction = metadata.s1_orbit_direction
            s1_incidence_angles = metadata.s1_incidence_angles
            metadata = metadata.channel_params
        else:
            ground_cover = metadata.pop("ground_cover", None)
            metadata = metadata["channel_params"]
            month = metadata.pop("month", None)
            center_coords = metadata.pop("center_coords", None)
            s1_orbit_direction = metadata.pop("s1_orbit_direction", None)
            s1_incidence_angles = metadata.pop("s1_incidence_angles", None)
        latent, mask_latent, ids_restore, cls_token, available_groups, channel_params = self.forward_encoder(
            x,
            metadata=metadata,
            ground_cover=ground_cover,
            mask_ratio=mask_ratio,
            spectral_mask_ratio=spectral_mask_ratio,
        )

        pred, scale, mask, land_cover_pred, land_cover_scale = self.forward_decoder(
            latent,
            ids_restore,
            mask_latent,
            x,
            land_cover_data=land_cover_data,
            available_groups=available_groups,
            channel_params=channel_params,
            ground_cover=ground_cover,
            return_cls_token=False,
        )
        loss, loss_fft, loss_group = self.reconstruction_loss(
            x,
            pred,
            scale,
            mask,
            channel_params=channel_params,
            available_groups=available_groups,
        )
        land_cover_losses = {}
        land_cover_group_losses = {}
        land_cover_masks = {}
        for land_cover_task in land_cover_pred:
            if land_cover_task in land_cover_data and len(land_cover_pred[land_cover_task]) > 0:
                lc_loss, lc_group_loss, lc_mask = self.land_cover_mask_loss(
                    land_cover_data=land_cover_data[land_cover_task],
                    pred_dict=land_cover_pred[land_cover_task],
                    mask_dict=mask,
                    scale_dict=land_cover_scale[land_cover_task],
                    land_cover_mappings=self.land_cover_mappings[land_cover_task],
                    channel_params=channel_params,
                    ground_cover=ground_cover,
                    available_groups=available_groups,
                    lc_params=self.land_cover_tasks[land_cover_task],
                )
                land_cover_losses[land_cover_task] = lc_loss
                land_cover_group_losses[land_cover_task] = lc_group_loss
                land_cover_masks[land_cover_task] = lc_mask

            else:
                land_cover_losses[land_cover_task] = torch.zeros_like(loss)
        if self.use_contrastive_loss:
            grouped_latent, land_cover_patch_hist, lc_classes, group_patch_sizes = self.group_latent(
                latent,
                ids_restore,
                mask,
                mask_ratio,
                channel_params=channel_params,
                available_groups=available_groups,
                land_cover_data=land_cover_data,
                ground_cover=ground_cover,
            )
            if self.contrast_type == "land_cover":
                per_group_contrastive_loss, total_contrastive_loss, _ = self.contrast_loss(
                    grouped_latent,
                    partitioning=land_cover_patch_hist,
                    lc_classes=lc_classes,
                    patch_sizes=group_patch_sizes,
                    latlon=center_coords,
                    month=month,
                    dist_group=self.dist_group,
                )
            elif self.contrast_type == "land_cover_v2":
                per_group_contrastive_loss, total_contrastive_loss, tau, _ = self.contrast_loss(
                    grouped_latent,
                    partitioning=land_cover_patch_hist,
                    lc_classes=lc_classes,
                    patch_sizes=group_patch_sizes,
                    dist_group=self.dist_group,
                )
        else:
            total_contrastive_loss = torch.zeros_like(loss)
            per_group_contrastive_loss = {}

        if center_coords is not None:
            latlon_pred = self.location_head(cls_token)
            latlon_labels = self.cyclic_encoding(center_coords, scale=180).to(dtype=latlon_pred.dtype)
            latlon_loss = F.mse_loss(latlon_pred.float(), latlon_labels.float(), reduction="mean")
        else:
            latlon_loss = torch.zeros_like(loss)

        if month is not None:
            month_pred = self.month_head(cls_token)
            month_labels = self.cyclic_encoding(month, scale=12).to(dtype=month_pred.dtype)
            month_loss = F.mse_loss(month_pred.float(), month_labels.float(), reduction="mean")
        else:
            month_loss = torch.zeros_like(loss)

        if s1_orbit_direction is not None:
            orbit_pred = self.orbit_direction_head(cls_token)
            orbit_loss = F.cross_entropy(
                orbit_pred.float(),
                s1_orbit_direction.long().squeeze(-1),
                reduction="mean",
            )
        else:
            orbit_loss = torch.zeros_like(loss)

        if s1_incidence_angles is not None:
            incidence_pred = self.incidence_angle_head(cls_token)
            incidence_loss = F.mse_loss(
                incidence_pred.float(), (s1_incidence_angles.float() - 34.0) / 8.0, reduction="mean"
            )
            incidence_loss = torch.clamp(incidence_loss, 0, 10)
        else:
            incidence_loss = torch.zeros_like(loss)

        if era5_land_labels is not None:
            era5_land_shape = next(iter(era5_land_labels.values())).shape

            if era5_land_shape[-1] != self.era5_land_patch_size:
                era5_land_pred = self.flexivit_decode_pred.functional(
                    cls_token,
                    self.era5_land_head,
                    (self.era5_land_patch_size, self.era5_land_patch_size),
                    (era5_land_shape[-2], era5_land_shape[-1]),
                )
            else:
                era5_land_pred = self.era5_land_head(cls_token.view(cls_token.shape[0], cls_token.shape[-1], 1, 1))
            loss_era5_land = self.era5_loss(era5_land_pred, era5_land_labels, era5_land_shape[-1])
        else:
            loss_era5_land = torch.zeros_like(loss)

        return ThorMAEOutput(
            loss_recon=loss,
            loss_fft=loss_fft,
            loss_contrastive=total_contrastive_loss,
            loss_era5_land=loss_era5_land,
            pred=pred,
            mask=mask,
            cls_feats=cls_token if return_cls_feats else None,
            loss_group=loss_group,
            loss_latlon=latlon_loss,
            loss_month=month_loss,
            loss_orbit_direction=orbit_loss,
            loss_incidence_angle=incidence_loss,
            loss_land_cover_tasks=land_cover_losses,
            land_cover_pred=land_cover_pred,
            land_cover_mask=land_cover_masks,
            loss_land_cover_tasks_group=land_cover_group_losses if self.log_per_group else None,
            loss_contrastive_group=per_group_contrastive_loss if self.log_per_group else None,
            channel_params=channel_params if return_channel_params else None,
            tau=tau if self.contrast_type == "land_cover_v2" else None,
        )

    def patchify(self, imgs, p, c=1):
        """
        imgs: (B, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (B, L, C*patch_size**2)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x, p, c=1):
        """
        x: (B, L, C*patch_size**2)
        p: Patch embed patch size
        c: Num channels
        imgs: (B, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, c, p, p))
        x = torch.einsum("nhwcpq->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


@MODELS.register()
def thor_mae_tiny_alibi_patch_size_embed_v1(input_params, **kwargs):
    model_kwargs = dict(
        input_params=input_params,
        embed_dim=192,
        depth=12,
        num_heads=3,
        embed_band=True,
        band_embed_dim=32,
        embed_prod=False,
        prod_embed_dim=0,
        embed_patch_size=True,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=8,
        decoder_band_embed_dim=128,
        decoder_prod_embed_dim=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return ThorMAE(**model_kwargs)


@MODELS.register()
def thor_mae_small_alibi_patch_size_embed_v1(input_params, **kwargs):
    model_kwargs = dict(
        input_params=input_params,
        embed_dim=384,
        depth=12,
        num_heads=6,
        embed_band=True,
        band_embed_dim=64,
        embed_prod=False,
        prod_embed_dim=0,
        embed_patch_size=True,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_band_embed_dim=128,
        decoder_prod_embed_dim=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return ThorMAE(**model_kwargs)


@MODELS.register()
def thor_mae_base_alibi(input_params, **kwargs):
    model_kwargs = dict(
        input_params=input_params,
        embed_dim=768,
        depth=12,
        num_heads=12,
        embed_band=True,
        band_embed_dim=768,
        embed_prod=False,
        prod_embed_dim=0,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_band_embed_dim=512,
        decoder_prod_embed_dim=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return ThorMAE(**model_kwargs)


@MODELS.register()
def thor_mae_base_alibi_patch_size_embed(input_params, **kwargs):
    model_kwargs = dict(
        input_params=input_params,
        embed_dim=768,
        depth=12,
        num_heads=12,
        embed_band=True,
        band_embed_dim=384,
        embed_prod=False,
        prod_embed_dim=0,
        embed_patch_size=True,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_band_embed_dim=256,
        decoder_prod_embed_dim=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return ThorMAE(**model_kwargs)


@MODELS.register()
def thor_mae_base_alibi_patch_size_embed_v1(input_params, **kwargs):
    model_kwargs = dict(
        input_params=input_params,
        embed_dim=768,
        depth=12,
        num_heads=12,
        embed_band=True,
        band_embed_dim=128,
        embed_prod=False,
        prod_embed_dim=0,
        embed_patch_size=True,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_band_embed_dim=128,
        decoder_prod_embed_dim=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return ThorMAE(**model_kwargs)


@MODELS.register()
def thor_mae_large_alibi_patch_size_embed_v1(input_params, **kwargs):
    model_kwargs = dict(
        input_params=input_params,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        embed_band=True,
        band_embed_dim=256,
        embed_prod=False,
        prod_embed_dim=0,
        embed_patch_size=True,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_band_embed_dim=128,
        decoder_prod_embed_dim=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return ThorMAE(**model_kwargs)
