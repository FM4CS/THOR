import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import torch
import torch.nn.functional as F
from timm.models._features import feature_take_indices
from timm.models.vision_transformer import Block
from torch import nn

from thor.core.model_registry import MODELS
from thor.utils.patch_embed import (
    FlexiPosEmbed,
    IndFlexiPatchEmbed,
    get_flexivit_grid_sizes,
    resize_abs_pos_embed,
)
from thor.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_1d_sincos_pos_embed_from_grid_torch,
    get_2d_sincos_pos_embed,
)

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompactAlibiSpec:
    q_x: torch.Tensor
    k_x: torch.Tensor
    q_y: torch.Tensor
    k_y: torch.Tensor
    slopes: torch.Tensor
    num_prefix_tokens: int = 0


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


@torch.compile
def get_alibi_points_thor(
    metadata: dict[str, dict[str, int]],
    available_groups: dict[str, list[str]],
    ground_cover: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build normalized 2D coordinates for tokens in the same order used by encoder concatenation.
    Args:
        metadata: metadata of the input data
        available_groups: available groups of the input data
        ground_cover: ground cover of the input data, used to normalize the bias to the same scale, default is None
        (useful for example if we want to use the same bias for different ground covers), default is to use the max patch size
        device: device to use for computation
    Returns:
        points: tensor of shape (num_patches, 2), where num_patches = sum(num_patches of all groups)

    """

    all_points = []
    max_patch_gsd_size = 0
    for _group_name, group_members in available_groups.items():
        first_member = group_members[0]
        product_gsd = metadata[first_member]["GSD"]
        product_num_patch = metadata[first_member]["num_patch"]
        product_patch_size = metadata[first_member]["patch_size"]
        max_patch_gsd_size = max(max_patch_gsd_size, product_patch_size * product_gsd)

        line_of_points = torch.arange(0, product_num_patch, dtype=dtype, device=device)
        line_of_points *= product_patch_size
        line_of_points += product_patch_size / 2
        line_of_points *= product_gsd

        points = torch.cartesian_prod(line_of_points, line_of_points)
        all_points.append(points)

    points = torch.cat(all_points, dim=0)

    # Either normalize by max patch gsd size or by ground cover
    if ground_cover is not None:
        points = points / ground_cover
    else:
        points = points / max_patch_gsd_size

    return points


@torch.compile
def get_alibi_thor(
    metadata: dict[str, dict[str, int]],
    available_groups: dict[str, list[str]],
    slopes: torch.Tensor,
    offset: float | int = 0.0,
    ground_cover: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    2D Alibi implementation using euclidean distance between patches
    Args:
        metadata: metadata of the input data
        available_groups: available groups of the input data
        slopes: slopes of the attention heads, used to scale the distance, should be a tensor of shape (num_heads,)
        offset: offset to add to the distance, useful if we want to for example add a class token ( with distance 0)
        ground_cover: ground cover of the input data, used to normalize the bias to the same scale, default is None
        (useful for example if we want to use the same bias for different ground covers), default is to use the max patch size
        device: device to use for computation
    Returns:
        distances: alibi tensor of shape (batch_size, num_heads, num_patches, num_patches), where num_patches = sum(num_patches of all groups)

    """

    points = get_alibi_points_thor(
        metadata,
        available_groups,
        ground_cover=ground_cover,
        device=device,
        dtype=dtype,
    )
    num_patches = points.shape[0]
    attention_heads = slopes.shape[0]
    slopes = slopes.unsqueeze(1).unsqueeze(2)
    distances = torch.cdist(points, points)
    distances += float(offset)
    distances = distances.unsqueeze(0)
    distances = distances * slopes * -1
    distances = distances.view(-1, attention_heads, num_patches, num_patches)
    return distances


def get_compact_alibi_thor(
    metadata: dict[str, dict[str, int]],
    available_groups: dict[str, list[str]],
    slopes: torch.Tensor,
    num_prefix_tokens: int = 0,
    ground_cover: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> CompactAlibiSpec:
    points = get_alibi_points_thor(
        metadata,
        available_groups,
        ground_cover=ground_cover,
        device=device,
        dtype=dtype,
    )
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return CompactAlibiSpec(
        q_x=x_coords,
        k_x=x_coords,
        q_y=y_coords,
        k_y=y_coords,
        slopes=slopes.to(device=device),
        num_prefix_tokens=int(num_prefix_tokens),
    )


def compact_alibi_to_dense(alibi: CompactAlibiSpec, batch_size: int | None = None) -> torch.Tensor:
    q_x = alibi.q_x.unsqueeze(1)
    k_x = alibi.k_x.unsqueeze(0)
    q_y = alibi.q_y.unsqueeze(1)
    k_y = alibi.k_y.unsqueeze(0)

    dx = q_x - k_x
    dy = q_y - k_y
    distances = torch.sqrt(dx * dx + dy * dy)
    dense_alibi = -alibi.slopes[:, None, None] * (distances + float(alibi.num_prefix_tokens))
    dense_alibi = dense_alibi.unsqueeze(0)

    if alibi.num_prefix_tokens:
        dense_alibi = F.pad(
            dense_alibi,
            (
                alibi.num_prefix_tokens,
                0,
                alibi.num_prefix_tokens,
                0,
            ),
            mode="constant",
            value=0.0,
        )

    if batch_size is not None:
        dense_alibi = dense_alibi.expand(batch_size, -1, -1, -1)

    return dense_alibi


@torch.compile
def alibi_cls_token_pad(alibi: torch.Tensor) -> torch.Tensor:
    """
    Pad the alibi tensor to include a cls token (distance 0).
    Args:
        alibi: alibi tensor of shape (batch_size, num_heads, num_patches, num_patches)
    Returns:
        alibi: padded alibi tensor of shape (batch_size, num_heads, num_patches + 1, num_patches + 1), zero padded at the beginning
    """
    return F.pad(
        alibi,
        (
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
        ),
        mode="constant",
        value=0.0,
    )


class ThorPosPooler(nn.Module):
    def __init__(self, pos_embed_dim, ref_patch_size, rel_patch_size, overlap_factor):
        super().__init__()

        # add fuzzy_pe_pooler to the model, to allow for random noise to be added to the position embedding on the fly
        self.fuzzy_pe_pooler = nn.AvgPool2d(
            rel_patch_size * overlap_factor,
            stride=rel_patch_size,
            padding=(overlap_factor // 2) * rel_patch_size,
        )
        self.pos_embed_dim = pos_embed_dim
        self.ref_patch_size = ref_patch_size

    def forward(self, ref_pos_embed):
        ref_pos_embed_grid = ref_pos_embed.transpose(1, 0).reshape(
            1, self.pos_embed_dim, self.ref_patch_size, self.ref_patch_size
        )
        pos_embed = self.fuzzy_pe_pooler(ref_pos_embed_grid).squeeze(0).reshape(self.pos_embed_dim, -1).transpose(1, 0)
        return pos_embed


class ThorViTEncoder(nn.Module):
    def __init__(
        self,
        input_params: dict[str, Any],
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        embed_prod: bool = True,  # Product embedding for different product
        prod_embed_dim: int = 128,
        pad_prod_embed_null: bool = False,
        embed_band: bool = True,  # Band embedding for different bands
        band_embed_dim: int = 128,
        pad_band_embed_null: bool = False,
        embed_patch_size: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        log_patch_size: bool = False,
    ) -> None:
        super().__init__()

        # Parse input params
        self.ground_covers: list[int] = input_params.pop("ground_covers")
        self.aggr_type = input_params.pop("aggr_type", "subsetmean")
        self.cls_token_type = input_params.pop("cls_token_type", "pooled")
        assert self.cls_token_type in [
            "pooled",
            "token",
        ], f"cls_token_type {self.cls_token_type} is not supported, only `pooled` and `token` are supported."

        self.use_superposition_encoding = input_params.pop("use_superposition_encoding", False)
        patch_size_seqs = input_params.pop("flexivit_patch_size_seqs", None)
        self.flexivit_ref_patch_size = input_params.pop("flexivit_ref_patch_size", 4)
        if patch_size_seqs is None:
            patch_size_seqs = [self.flexivit_ref_patch_size]
        self.patch_size_seqs = sorted(patch_size_seqs)
        self.flexivit_ref_grid_size = input_params.pop("flexivit_ref_grid_size", 14)
        self.use_flexivit = input_params.pop("use_flexivit", True)
        self.token_budget = input_params.pop("flexivit_token_budget", 1296)
        self.select_patch_strategy = input_params.pop("select_patch_strategy", "min")
        # Only for evaluation min, max, equal-min, equal-max

        self.encoder_pos_type = input_params.pop("encoder_pos_type", "alibi")
        assert self.encoder_pos_type in [
            "alibi",
            "pooled",
            "interpolate",
        ], f"encoder_pos_type {self.encoder_pos_type} is not supported."
        channels = input_params.pop("channels")
        # Backwards compat channels
        self.channels = {}
        self.channel_rename_map = {}
        for channel, params in channels.items():
            if "num_patch" in params and "patch_size" not in params:
                if len(self.ground_covers) > 1:
                    msg = "num_patch is ambiguous supported for multiple ground covers."
                    raise ValueError(msg)
                params["patch_size"] = self.ground_covers[0] // params["num_patch"] // params["GSD"]
            if "patch_size" in params and "num_patch" not in params and not self.use_flexivit:
                if len(self.ground_covers) > 1:
                    msg = "patch_size is ambiguous supported for multiple ground covers."
                    raise ValueError(msg)
                min_patch_size_seq = min(patch_size_seqs)
                patch_size = min(params["patch_size"], min_patch_size_seq)

                params["num_patch"] = self.ground_covers[0] // patch_size // params["GSD"]

            rename_name = params.pop("patch_embed_name", channel)
            if rename_name in self.channel_rename_map:
                msg = f"Duplicate patch embed name {rename_name} found."
                raise ValueError(msg)
            self.channel_rename_map[channel] = rename_name
            self.channels[channel] = params

        self.min_gsd = min([params["GSD"] for params in self.channels.values()])
        self.groups = self.validate_group(
            input_params.pop("groups", None)
        )  # {'group0':[product_band, ...], 'group1': ..., ...}

        self.ind_patch_embed = IndFlexiPatchEmbed(
            self.ground_covers,
            self.channels,
            self.groups,
            self.channel_rename_map,
            min_patch_size=self.flexivit_ref_patch_size,
            embed_dim=embed_dim,
            patch_size_seqs=patch_size_seqs,
        )
        self.use_fuzzy_encoding = input_params.pop("use_fuzzy_encoding", False)
        self.group_lookup = {
            group_member: group_name
            for group_name, group_members in self.groups.items()
            for group_member in group_members
        }

        if self.use_fuzzy_encoding:
            gsd_patch_size = {params["GSD"]: params["patch_size"] for params in self.channels.values()}
            gsd_patch_size = sorted(gsd_patch_size.items(), key=lambda x: x[0])
            ref_patch_size = gsd_patch_size[0][-1]
            ref_gsd = gsd_patch_size[0][0]
            self.ref_patch_grid_size = self.ground_covers[0] // ref_patch_size // ref_gsd

            IJ = torch.meshgrid(
                torch.arange(self.ref_patch_grid_size),
                torch.arange(self.ref_patch_grid_size),
                indexing="ij",
            )
            fuzzy_grid = torch.stack(IJ, dim=-1).float() + 0.5
            self.fuzzy_grid = nn.Parameter(fuzzy_grid, requires_grad=False)

        # Initialize embedding
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.embed_prod = embed_prod
        self.prod_embed_dim = prod_embed_dim if embed_prod else 0
        self.pad_prod_embed_null = pad_prod_embed_null
        self.embed_band = embed_band
        self.band_embed_dim = band_embed_dim if embed_band else 0
        self.pad_band_embed_null = pad_band_embed_null
        self.pos_embed_dim = embed_dim - prod_embed_dim - band_embed_dim
        self.embed_patch_size = embed_patch_size
        self.log_patch_size = log_patch_size
        self.patch_size_embed_dim = self.pos_embed_dim if embed_patch_size else 0

        if self.encoder_pos_type == "interpolate":
            _reference_grid_size, grid_sizes, _ground_cover_lookup = get_flexivit_grid_sizes(
                self.ground_covers,
                self.channels,
                self.ind_patch_embed.patch_size_seqs,
                self.flexivit_ref_patch_size,
            )
            self.ref_pos_embed = FlexiPosEmbed(
                grid_size=self.flexivit_ref_grid_size,
                pos_embed_dim=self.pos_embed_dim,
                grid_sizes=sorted(grid_sizes),
            )
        elif self.encoder_pos_type == "alibi":
            self.register_buffer("encoder_slopes", torch.tensor(get_slopes(num_heads)))

        # NOTE: band_embed is now spectral group embed, not changing the name to avoid unexpected break
        self.init_embeds()

        # Initialize cls token
        if self.cls_token_type == "token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_prefix_tokens = 1
        else:
            self.cls_token = None
            self.num_prefix_tokens = 0

        # Initialize transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Register device as a buffer
        self.register_buffer("_device_dummy", torch.zeros(1), persistent=False)

        self.initialize_weights()

    @property
    def device(self):
        return self._device_dummy.device

    def init_embeds(self, pos_only=False):
        ret = self.initialize_embedding(
            self.pos_embed_dim,
            self.band_embed_dim,
            self.prod_embed_dim,
        )
        if self.use_fuzzy_encoding:
            pos_embed, band_embed, prod_embed, pos_embed_poolers = ret
            self.pos_embed_poolers = pos_embed_poolers
        else:
            pos_embed, band_embed, prod_embed = ret
        self.pos_embed = pos_embed

        if not pos_only:
            self.band_embed = band_embed
            self.prod_embed = prod_embed

    def validate_group(self, groups):
        valid_groups = {}
        if groups is None:
            for group_idx, prodcut_band in enumerate(self.channels.keys()):
                valid_groups[f"group{group_idx}"] = [prodcut_band]
            return valid_groups
        found_bands = []
        for group_idx, group in enumerate(groups):
            group_product = None
            group_gsd = None
            group_patch_size = None
            valid_groups[f"group{group_idx}"] = []
            for product_band in group:
                product, _ = product_band.split(":")
                group_product = product if group_product is None else group_product
                group_gsd = self.channels[product_band]["GSD"] if group_gsd is None else group_gsd
                group_patch_size = (
                    self.channels[product_band]["patch_size"] if group_patch_size is None else group_patch_size
                )
                if self.channels[product_band]["GSD"] != group_gsd:
                    msg = f"GSD {self.channels[product_band]['GSD']} in group {group} does not match with GSD {group_gsd} in the same group."
                    raise ValueError(msg)
                if self.channels[product_band]["patch_size"] != group_patch_size:
                    msg = f"Patch size {self.channels[product_band]['patch_size']} in group {group} does not match with patch size {group_patch_size} in the same group."
                    raise ValueError(msg)
                valid_groups[f"group{group_idx}"].append(product_band)
                found_bands.append(product_band)

        # Log compact group summary
        for gname, gmembers in valid_groups.items():
            bands_str = ", ".join(m.split(":")[1] for m in gmembers)
            product = gmembers[0].split(":")[0]
            gsd = self.channels[gmembers[0]]["GSD"]
            ps = self.channels[gmembers[0]]["patch_size"]
            logger.info(f"  {gname}: {product} [{bands_str}] (GSD={gsd}, patch_size={ps})")
        logger.info(f"Total: {len(found_bands)} bands in {len(valid_groups)} groups")

        # Remove bands from channels that are not present in the groups
        keys = list(self.channels.keys())
        remove_bands = set(keys) - set(found_bands)
        for band in remove_bands:
            del self.channels[band]
        if len(remove_bands) > 0:
            logger.warning(f"Removed bands {remove_bands} from channels that are not present in the groups.")

        return valid_groups

    def initialize_embedding(
        self, pos_embed_dim: int, band_embed_dim: int, prod_embed_dim: int, encoder: bool = True
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Take dimension as input for MAE implementation where encoder decoder dimension can be different
        pos_embed = {}
        if self.encoder_pos_type in ["interpolate", "pooled"]:
            # initialize pos_embed
            gsd_patch_size = {params["GSD"]: params["patch_size"] for params in self.channels.values()}
            gsd_patch_size = sorted(gsd_patch_size.items(), key=lambda x: x[0])
            ref_gsd = gsd_patch_size[0][0]
            ref_patch_size = gsd_patch_size[0][-1]

            ref_patch_grid_size = self.ground_covers[-1] // ref_patch_size // ref_gsd
            ref_pos_embed = torch.from_numpy(
                get_2d_sincos_pos_embed(pos_embed_dim, ref_patch_grid_size, cls_token=False)
            ).float()
            #
            # NOTE: this functions is called from the decoder as well, we need to make sure we do not try to overwrite the decoder pos embed as well
            # Check if pos_embed is already initialized
            if (
                encoder
                and hasattr(self, "pos_embed")
                and self.pos_embed[str(gsd_patch_size[0][0])].shape == ref_pos_embed.shape
            ):
                ref_pos_embed = self.pos_embed[str(gsd_patch_size[0][0])]

            if self.use_fuzzy_encoding:
                pos_embed_poolers = {}
                pos_embed_poolers[str(gsd_patch_size[0][0])] = nn.Identity()

            overlap_factor = 1
            pos_embed[str(gsd_patch_size[0][0])] = ref_pos_embed
            for gsd, patch_size in gsd_patch_size[1:]:
                num_patch = self.ground_covers[-1] // patch_size // gsd
                assert self.ground_covers[-1] % (patch_size * gsd) == 0, (
                    f"Patch size {patch_size} * GSD {gsd} does not divide ground cover {self.ground_covers[-1]}"
                )
                rel_patch_grid_size = int(ref_patch_grid_size // num_patch)
                # This look very long and complicated, it just average over the nearby patch using convolution
                if self.use_superposition_encoding:
                    sup_pe_pooler = nn.AvgPool2d(
                        rel_patch_grid_size * overlap_factor,
                        stride=rel_patch_grid_size,
                        padding=(overlap_factor // 2) * rel_patch_grid_size,
                    )
                    ref_pos_embed_grid = ref_pos_embed.transpose(1, 0).reshape(
                        1, pos_embed_dim, ref_patch_grid_size, ref_patch_grid_size
                    )
                    pos_embed[str(gsd)] = (
                        sup_pe_pooler(ref_pos_embed_grid).squeeze(0).reshape(pos_embed_dim, -1).transpose(1, 0)
                    )
                    if self.use_fuzzy_encoding:
                        # add pooling module, to allow for averaging over the reference pos embed with noise
                        pos_embed_poolers[str(gsd)] = ThorPosPooler(
                            pos_embed_dim,
                            ref_patch_grid_size,
                            rel_patch_grid_size,
                            overlap_factor,
                        )
                else:
                    pos_embed[str(gsd)] = torch.from_numpy(
                        get_2d_sincos_pos_embed(pos_embed_dim, num_patch, cls_token=False)
                    ).float()

        band_embed = {}
        if self.embed_band:
            # unique_groups = defaultdict(list)
            # for idx, (spectral_group, group_bands) in enumerate(self.groups.items()):
            #    unique_group = ','.join(sorted([band.split(':')[1] for band in group_bands]))
            #    unique_groups[unique_group].append(spectral_group)
            # embed = get_1d_sincos_pos_embed_from_grid(band_embed_dim, torch.arange(len(unique_groups)).numpy())
            # for idx, spectral_groups in enumerate(unique_groups.values()):
            #    for spectral_group in spectral_groups:
            #        if self.pad_band_embed_null:
            #            band_embed[spectral_group] = torch.zeros(embed[idx].shape).float()
            #        else:
            #            band_embed[spectral_group] = torch.from_numpy(embed[idx]).float()
            embed = get_1d_sincos_pos_embed_from_grid(band_embed_dim, torch.arange(len(self.groups)).cpu().numpy())
            for i, spectral_group in enumerate(self.groups.keys()):
                band_embed[spectral_group] = torch.from_numpy(embed[i]).float()

        prod_embed = {}
        if self.embed_prod:
            unique_prod = {prduct_band.split(":")[0]: None for prduct_band in self.channels.keys()}
            embed = get_1d_sincos_pos_embed_from_grid(prod_embed_dim, torch.arange(len(unique_prod)).cpu().numpy())
            for i, prod in enumerate(unique_prod.keys()):
                if self.pad_prod_embed_null:
                    prod_embed[prod] = torch.zeros_like(torch.from_numpy(embed[i])).float()
                else:
                    prod_embed[prod] = torch.from_numpy(embed[i]).float()

        pos_embed = nn.ParameterDict(pos_embed).requires_grad_(False)
        band_embed = nn.ParameterDict(band_embed).requires_grad_(False)
        prod_embed = nn.ParameterDict(prod_embed).requires_grad_(False)

        if self.use_fuzzy_encoding:
            pos_embed_poolers = nn.ModuleDict(pos_embed_poolers).requires_grad_(False)
            return pos_embed, band_embed, prod_embed, pos_embed_poolers

        return pos_embed, band_embed, prod_embed

    def fuzzy_pos_embed(self, pos_embed):
        """
        Add randomly sampled noise to positional embeddings (ref ViTAR paper)
        """

        sampled_points = self.fuzzy_grid + torch.rand_like(self.fuzzy_grid) - 0.5
        sampled_points -= self.ref_patch_grid_size / 2
        sampled_points /= self.ref_patch_grid_size / 2

        pos_embed_sampled = F.grid_sample(
            pos_embed.permute(1, 0).reshape(
                -1,
                self.pos_embed_dim,
                self.ref_patch_grid_size,
                self.ref_patch_grid_size,
            ),
            sampled_points.unsqueeze(0),
            mode="bicubic",
            align_corners=True,
        )

        return pos_embed_sampled.reshape(self.pos_embed_dim, -1).permute(1, 0)

    def initialize_weights(self) -> None:
        if self.cls_token_type == "token":
            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_available_groups(self, original_input: dict[str, torch.Tensor]) -> dict[str, list[str]]:
        available_groups: dict[str, list[str]] = {}
        for group_name, group_member in self.groups.items():
            for member in group_member:
                if member in original_input:
                    if group_name not in available_groups:
                        available_groups[group_name] = [member]
                    else:
                        available_groups[group_name].append(member)
        return available_groups

    def get_channel_params(
        self,
        patch_embed: dict[str, torch.Tensor],
        metadata: dict[str, dict[str, int]] | None = None,
        ground_cover: int | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get GSD, num_patch, patch_size for each channel in the input data.
        Args:
            patch_embed: patch embeddings of the input data {'product_band': (B, N, C)}
            metadata: metadata of the input data
        """

        channel_params = {
            product_band: {"GSD": self.channels[product_band]["GSD"]} for product_band in patch_embed.keys()
        }

        # Override with default if available
        if ground_cover is None and len(self.ground_covers) == 1:
            ground_cover = self.ground_covers[0]
        elif ground_cover is None and len(self.ground_covers) > 1:
            msg = "Ground cover is not defined in metadata and multiple ground covers are defined."
            raise ValueError(msg)

        assert isinstance(ground_cover, int), (
            f"Ground cover {ground_cover} is not defined as int, please provide a valid ground cover."
        )

        for product_band in patch_embed:
            # Override with metadata if available
            if metadata is not None and product_band in metadata and "GSD" in metadata[product_band]:
                channel_params[product_band]["GSD"] = metadata[product_band]["GSD"]
            num_patch = int(patch_embed[product_band].shape[1] ** 0.5)
            patch_size = ground_cover // num_patch // channel_params[product_band]["GSD"]
            channel_params[product_band]["num_patch"] = num_patch
            channel_params[product_band]["patch_size"] = patch_size
            assert patch_size * channel_params[product_band]["GSD"] * num_patch == ground_cover, (
                f"Patch size {patch_size} * GSD {channel_params[product_band]['GSD']} * num_patch {num_patch} does not match ground cover {ground_cover}"
            )

        return channel_params

    def get_encoder_auxilliary_embed(
        self,
        original_input: dict[str, torch.Tensor],
        available_groups: dict[str, list[str]],
        channel_params: dict[str, dict[str, int]],
    ) -> dict[str, torch.Tensor]:
        auxilliary_embed = {}
        if self.use_fuzzy_encoding:
            # get ref pos embed

            gsd_patch_size = {
                params["GSD"]: params["patch_size"] for params in channel_params.values() if isinstance(params, dict)
            }
            gsd_patch_size = sorted(gsd_patch_size.items(), key=lambda x: x[0])
            ref_pos_embed = self.pos_embed[str(gsd_patch_size[0][0])]
            sampled_ref_pos_embed = self.fuzzy_pos_embed(ref_pos_embed)
        for group_name, group_member in available_groups.items():
            product_band = group_member[0]
            product, _band = product_band.split(":")
            if self.use_fuzzy_encoding:
                group_pos_embed = self.pos_embed_poolers[str(channel_params[product_band]["GSD"])](
                    sampled_ref_pos_embed
                )
            elif self.encoder_pos_type == "pooled":
                group_pos_embed = self.pos_embed[str(channel_params[product_band]["GSD"])]

                new_size = channel_params[product_band]["num_patch"]
                group_pos_embed = resize_abs_pos_embed(
                    group_pos_embed[None, :, :], new_size=new_size, num_prefix_tokens=0
                )[0]

            elif self.encoder_pos_type == "interpolate":
                new_size = channel_params[product_band]["num_patch"]
                group_pos_embed = self.ref_pos_embed(new_size)

            elif self.encoder_pos_type == "alibi":
                _x = next(iter(original_input.values()))

                if self.embed_patch_size:
                    patch_sizes = (
                        torch.ones((channel_params[product_band]["num_patch"] ** 2), device=_x.device, dtype=_x.dtype)
                        * channel_params[product_band]["patch_size"]
                        * channel_params[product_band]["GSD"]
                        / (self.flexivit_ref_patch_size * self.min_gsd)
                    )
                    if self.log_patch_size:
                        patch_sizes = torch.log(patch_sizes)

                    group_pos_embed = get_1d_sincos_pos_embed_from_grid_torch(
                        self.patch_size_embed_dim, pos=patch_sizes
                    ).to(_x.dtype)
                else:
                    group_pos_embed = torch.zeros(
                        (
                            channel_params[product_band]["num_patch"] ** 2,
                            self.pos_embed_dim,
                        ),
                        device=_x.device,
                        dtype=_x.dtype,
                    )

            auxilliary_embed[group_name] = group_pos_embed

            if self.embed_band:
                auxilliary_embed[group_name] = torch.cat(
                    (
                        auxilliary_embed[group_name],
                        self.band_embed[group_name].expand(auxilliary_embed[group_name].shape[0], -1),
                    ),
                    dim=-1,
                )
            if self.embed_prod:
                auxilliary_embed[group_name] = torch.cat(
                    (
                        auxilliary_embed[group_name],
                        self.prod_embed[product].expand(auxilliary_embed[group_name].shape[0], -1),
                    ),
                    dim=-1,
                )

            data = original_input[product_band]
            band_ground_cover = int(data.shape[-1] * channel_params[product_band]["GSD"])
            if band_ground_cover not in self.ground_covers:
                msg = (
                    f"Input ground cover for {product_band} is {band_ground_cover}x{band_ground_cover}, (image shape:{data.shape[-2:]}) "
                    f"which does match grid of {channel_params[product_band]['num_patch']}x{channel_params[product_band]['num_patch']}"
                    f"with patch size {channel_params[product_band]['patch_size']} and GSD {channel_params[product_band]['GSD']}."
                    f" patches for defined ground covers {self.ground_covers}."
                )
                raise ValueError(msg)

        return auxilliary_embed

    def aggregate_by_group(
        self,
        patch_embed: dict[str, torch.Tensor],
        available_groups: dict[str, list[str]],
    ) -> dict[str, torch.Tensor]:
        group_embed = {}
        for group_name, group_member in available_groups.items():
            if self.aggr_type == "mean":
                group_embed[group_name] = torch.stack(
                    [patch_embed[product_band] for product_band in group_member], -1
                ).mean(-1)
            elif self.aggr_type == "sum":
                group_embed[group_name] = torch.stack(
                    [patch_embed[product_band] for product_band in group_member], -1
                ).sum(-1)
            elif self.aggr_type == "nanmean":
                to_stack = [
                    patch_embed[product_band]
                    for product_band in group_member
                    if not patch_embed[product_band].isnan().any().item()
                ]
                if len(to_stack) == 0:
                    to_stack = [patch_embed[product_band] for product_band in group_member]  # This will forward NaNs
                group_embed[group_name] = torch.stack(to_stack, -1).mean(-1)
            elif self.aggr_type == "nansum":
                to_stack = [
                    patch_embed[product_band]
                    for product_band in group_member
                    if not patch_embed[product_band].isnan().any().item()
                ]
                if len(to_stack) == 0:
                    to_stack = [patch_embed[product_band] for product_band in group_member]  # This will forward NaNs
                group_embed[group_name] = torch.stack(to_stack, -1).sum(-1)
            elif self.aggr_type == "subsetmean":
                to_stack = [patch_embed[product_band] for product_band in group_member if product_band in patch_embed]
                if len(to_stack) > 0:
                    group_embed[group_name] = torch.stack(to_stack, -1).mean(-1)
            elif self.aggr_type == "subsetsum":
                to_stack = [patch_embed[product_band] for product_band in group_member if product_band in patch_embed]
                if len(to_stack) > 0:
                    group_embed[group_name] = torch.stack(to_stack, -1).sum(-1)

        return group_embed

    def _build_encoder_alibi(
        self,
        available_groups: dict[str, list[str]],
        channel_params: dict[str, dict[str, int]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | CompactAlibiSpec | None:
        if self.encoder_pos_type != "alibi":
            return None

        from thor.models.patch_timm import use_flex_attn

        if not self.training and not torch.jit.is_scripting() and use_flex_attn():
            return get_compact_alibi_thor(
                channel_params,
                available_groups,
                slopes=self.encoder_slopes,
                num_prefix_tokens=self.num_prefix_tokens,
                device=device,
                dtype=dtype,
            )

        alibi = get_alibi_thor(
            channel_params,
            available_groups,
            slopes=self.encoder_slopes,
            offset=self.num_prefix_tokens,
            device=device,
            dtype=dtype,
        )
        if self.cls_token_type == "token":
            alibi = alibi_cls_token_pad(alibi)
        return alibi.expand(batch_size, -1, -1, -1)

    def forward_encoder(
        self,
        x: dict[str, torch.Tensor],
        metadata: dict[str, dict[str, dict[str, int]]] | None = None,
        ground_cover: int | None = None,
        return_channel_params: bool = False,
    ) -> torch.Tensor:
        # B: Batch size, H: Height, W: Width
        # D: Embedding dimension, N_n: Number of patches for each product_band
        # T: Sequence length (number of patches * number of spectral bands)

        # x = {'product:band': (B, 1, H, W), ...}
        # extract patches
        if self.training and self.use_flexivit:
            # NOTE: some products/groups might dissapear when using token budget
            # Drawing patch sizes randomly
            patch_embed = self.ind_patch_embed(x, token_budget=self.token_budget, device=self.device)
            # {'product:band': (B, N_n, D), ...}
        else:
            # Not drawing patch sizes randomly
            patch_sizes = self.get_patch_sizes(
                method=self.select_patch_strategy,
                x=x,
                patch_sizes=self.ind_patch_embed.patch_size_seqs,
                ground_cover=ground_cover,
            )

            patch_embed = self.ind_patch_embed(x, patch_sizes=patch_sizes, device=self.device)
            # {'product:band': (B, N_n, D), ...}

        # find available groups in input
        available_groups = self.get_available_groups(patch_embed)  # {'group0': [product_band, ...], ...}

        channel_params = self.get_channel_params(patch_embed, metadata, ground_cover)

        group_embed = self.aggregate_by_group(patch_embed, available_groups)  # {'group0': (B, N_n, D), ...}

        # add additional embedding  {'group0': (N_n, D), ...}
        auxilliary_embed = self.get_encoder_auxilliary_embed(x, available_groups, channel_params)

        x = torch.cat(
            [group_embed[group_name].add_(auxilliary_embed[group_name]) for group_name in group_embed.keys()],
            dim=1,
        )  # (B, T, D)

        # append cls token
        if self.cls_token_type == "token":
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # (N, T + 1, D)

        alibi = self._build_encoder_alibi(
            available_groups,
            channel_params,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, alibi)
        x = self.norm(x)

        if return_channel_params:
            return x, channel_params
        return x

    def get_patch_sizes(
        self,
        method: Literal["min", "max", "equal-min", "equal-max"],
        x: dict[str, torch.Tensor],
        patch_sizes: dict[str, Sequence[int]],
        ground_cover: int | None = None,
    ) -> dict[str, int]:
        """Get patch sizes for each product band based on the method."""
        if method == "min":
            return {p: min(p_sizes) for p, p_sizes in patch_sizes.items()}
        elif method == "max":
            return {p: max(p_sizes) for p, p_sizes in patch_sizes.items()}
        elif method in {"equal-min", "equal-max"}:
            select_largest = method == "equal-max"
            # Find patch sizes that result in equal number of patches across bands
            # Formula: num_patches = ground_cover // GSD // patch_size

            # Calculate all possible num_patches for each band
            band_num_patches = {}
            for p, p_sizes in patch_sizes.items():
                if p not in x:
                    continue
                if ground_cover is None:
                    input_size = x[p].shape[-1]
                    band_num_patches[p] = {patch_size: input_size // patch_size for patch_size in p_sizes}
                else:
                    gsd = self.channels[p]["GSD"]
                    band_num_patches[p] = {patch_size: ground_cover // gsd // patch_size for patch_size in p_sizes}

            # Find common num_patches values across all bands
            all_num_patches = [set(np_dict.values()) for np_dict in band_num_patches.values()]
            common_num_patches = set.intersection(*all_num_patches) if all_num_patches else set()

            if not common_num_patches:
                msg = (
                    f"No common number of patches found across bands with ground_cover={ground_cover}. "
                    f"Band num_patches: {band_num_patches}"
                )
                raise ValueError(msg)

            # Select the largest common num_patches (finest resolution) or smallest (coarsest) based on method
            target_num_patches = max(common_num_patches) if not select_largest else min(common_num_patches)

            # Select patch size for each band that gives the target num_patches
            result = {}
            for p in patch_sizes.keys():
                if p not in x:
                    continue
                for patch_size, num_patches in band_num_patches[p].items():
                    if num_patches == target_num_patches:
                        result[p] = patch_size
                        break

            return result
        else:
            msg = f"Unknown method {method} for getting patch sizes."
            raise ValueError(msg)

    def forward_intermediates(
        self,
        x: dict[str, torch.Tensor],
        metadata: dict[str, dict[str, int]] | None = None,
        ground_cover: int | None = None,
        indices: int | list[int] | tuple[int] | None = None,
        norm: bool = False,
        stop_early: bool = False,
        intermediates_only: bool = False,
        return_channel_params: bool = False,
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward features that returns intermediates.

        Args:
            x: Input dictionary containg bands.
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            intermediates_only: Only return intermediate features
        Returns:

        """

        # B: Batch size, H: Height, W: Width
        # D: Embedding dimension, N_n: Number of patches for each product_band
        # T: Sequence length (number of patches * number of spectral bands)

        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # x = {'product:band': (B, 1, H, W), ...}
        # extract patches
        # NOTE: some products/groups might dissapear when using token budget
        if self.training and self.use_flexivit:
            # Drawing patch sizes randomly
            patch_embed = self.ind_patch_embed(
                x,
                token_budget=None,  # self.token_budget,
                device=self.device,
            )
            # {'product:band': (B, N_n, D), ...}
        else:
            # Not drawing patch sizes randomly

            patch_sizes = self.get_patch_sizes(
                method=self.select_patch_strategy,
                x=x,
                patch_sizes=self.ind_patch_embed.patch_size_seqs,
                ground_cover=ground_cover,
            )

            patch_embed = self.ind_patch_embed(x=x, patch_sizes=patch_sizes, device=self.device)
            # {'product:band': (B, N_n, D), ...}

        # find available groups in input
        available_groups = self.get_available_groups(patch_embed)  # {'group0': [product_band, ...], ...}

        channel_params = self.get_channel_params(patch_embed, metadata, ground_cover)

        group_embed = self.aggregate_by_group(patch_embed, available_groups)  # {'group0': (B, N_n, D), ...}

        # add additional embedding # {'group0': (N_n, D), ...}
        auxilliary_embed = self.get_encoder_auxilliary_embed(x, available_groups, channel_params)

        # Concatenate tokens (B, T, D)
        x = torch.cat(
            [group_embed[group_name].add_(auxilliary_embed[group_name]) for group_name in group_embed.keys()],
            dim=1,
        )

        # append cls token
        if self.cls_token_type == "token":
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # (N, T + 1, D)

        alibi = self._build_encoder_alibi(
            available_groups,
            channel_params,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[: max_index + 1]

        # apply Transformer blocks
        for i, blk in enumerate(blocks):
            x = blk(x, alibi)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            # prefix_tokens = [y[:, 0 : self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens :] for y in intermediates]

        if intermediates_only and return_channel_params:
            return intermediates, channel_params

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        if return_channel_params:
            return x, intermediates, channel_params

        return x, intermediates

    def forward(
        self,
        x: dict[str, torch.Tensor],
        metadata: dict[str, dict[str, Any]] | None = None,
        ground_cover: int | None = None,
        return_channel_params: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, dict[str, int]]]:
        x = self.forward_encoder(x, metadata, ground_cover, return_channel_params)
        return x


@MODELS.register()
def thor_vit_tiny_encoder_alibi_patch_size_embed_v1(input_params, **kwargs):
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
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return ThorViTEncoder(**model_kwargs)


@MODELS.register()
def thor_vit_small_encoder_alibi_patch_size_embed_v1(input_params, **kwargs):
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
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return ThorViTEncoder(**model_kwargs)


@MODELS.register()
def thor_vit_base_encoder_alibi(input_params, **kwargs):
    model_kwargs = dict(
        input_params=input_params,
        embed_dim=768,
        depth=12,
        num_heads=12,
        embed_band=True,
        band_embed_dim=768,
        embed_prod=False,
        prod_embed_dim=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return ThorViTEncoder(**model_kwargs)


@MODELS.register()
def thor_vit_base_encoder_alibi_patch_size_embed(input_params, **kwargs):
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
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return ThorViTEncoder(**model_kwargs)


@MODELS.register()
def thor_vit_base_encoder_alibi_patch_size_embed_v1(input_params, **kwargs):
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
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return ThorViTEncoder(**model_kwargs)


@MODELS.register()
def thor_vit_large_encoder_alibi_patch_size_embed_v1(input_params, **kwargs):
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
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return ThorViTEncoder(**model_kwargs)
