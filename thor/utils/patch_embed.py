# MiT License, from https://github.com/bwconrad/flexivit/blob/main/flexivit_pytorch/patch_embed.py

import logging
import math
from collections.abc import Iterable, Sequence
from itertools import repeat
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn, vmap

from thor.utils.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_with_resolution

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@torch.compile
def random_partition_sizes_compiled(
    total_length: int, min_sizes: torch.Tensor, max_sizes: torch.Tensor, device: torch.device | None = None
) -> torch.Tensor:
    """
    Generate random sizes for partitioning a range starting at 0.

    Args:
        total_length: Total length of the range to partition
        min_sizes: Tensor of minimum sizes for each partition
        max_sizes: Tensor of maximum sizes for each partition

    Returns:
        Tensor of partition sizes
    """
    n_parts = min_sizes.size(0)
    assert max_sizes.size(0) == n_parts, "min_sizes and max_sizes must have the same length"

    min_sum = torch.sum(min_sizes)

    # Validate if constraints can be satisfied
    if min_sum > total_length:
        msg = f"Sum of minimum sizes ({min_sum}) exceeds total range ({total_length})"
        raise RuntimeError(msg)

    # Initialize result tensor
    sizes = torch.zeros(n_parts, dtype=torch.int64, device=device)

    remaining_length = total_length
    for i in range(n_parts):
        is_last_part = i == n_parts - 1
        min_size = min_sizes[i]
        max_size = max_sizes[i]

        # Calculate remaining minimum space needed for future partitions
        remaining_min_space = 0
        if not is_last_part:
            remaining_min_space = torch.sum(min_sizes[i + 1 :])

        # Calculate valid range for this partition size
        available_space = remaining_length - remaining_min_space

        # Determine the bounds for this partition size
        actual_min_size = min(min_size, available_space)
        actual_max_size = min(max_size, available_space)

        # Generate random size within constraints
        partition_size = 0
        if actual_min_size >= actual_max_size:
            # If constraints are tight, use the minimum
            partition_size = actual_min_size
        else:
            # Use the torch random module for TorchScript compatibility
            partition_size = torch.randint(
                actual_min_size,
                actual_max_size + 1,  # +1 because torch.randint upper bound is exclusive
                (1,),
            )

        # Special case for the last partition
        if is_last_part:
            partition_size = remaining_length

        # Store partition size
        sizes[i] = partition_size

        # Update remaining length
        remaining_length -= partition_size

    return sizes


def get_flexivit_grid_sizes(
    ground_covers: Sequence[int],
    channels: dict[str, dict[str, int]],
    patch_size_seqs: dict[str, Sequence[int]],
    flexivit_ref_patch_size: int = 32,
) -> tuple[int, list[int]]:
    gsd_sorted = [params["GSD"] for params in channels.values()]
    gsd_sorted = sorted(gsd_sorted)

    ref_grid_size = max(ground_covers) // flexivit_ref_patch_size // gsd_sorted[0]
    grid_sizes = []
    ground_cover_lookup = {}

    for ground_cover in ground_covers:
        if ground_cover not in ground_cover_lookup:
            ground_cover_lookup[ground_cover] = {}
        for product_band, patch_sizes in patch_size_seqs.items():
            product_band_gsd = channels[product_band]["GSD"]
            for patch_size in patch_sizes:
                grid_size = ground_cover // patch_size // product_band_gsd
                if grid_size * product_band_gsd * patch_size != ground_cover:
                    # Skip patch sizes that don't add up for the given ground cover
                    logger.debug(
                        f"Not a perfect grid size for {product_band} with GSD {product_band_gsd} and patch size {patch_size},"
                        f" got grid size {grid_size} which results in {grid_size * product_band_gsd * patch_size} instead of {ground_cover}"
                    )
                    continue
                if grid_size < 2 or grid_size > 32:
                    logger.debug(f"Grid size {grid_size} is not in the range of 2-32, skipping this grid size.")
                    continue
                if grid_size not in grid_sizes:
                    grid_sizes.append(grid_size)
                if grid_size not in ground_cover_lookup[ground_cover]:
                    ground_cover_lookup[ground_cover][grid_size] = []
                ground_cover_lookup[ground_cover][grid_size].append(
                    {
                        "product_band": product_band,
                        "patch_size": patch_size,
                        "gsd": product_band_gsd,
                    }
                )

    # Sort grid sizes
    grid_sizes = sorted(grid_sizes)
    return ref_grid_size, grid_sizes, ground_cover_lookup


def to_2tuple(x: Any) -> tuple:
    if isinstance(x, Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


def resize_abs_pos_embed(
    pos_embed: torch.Tensor,
    new_size: int | tuple[int, int],
    old_size: int | tuple[int, int] | None = None,
    num_prefix_tokens: int = 1,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> torch.Tensor:
    """Resize absolute position embeddings to a target resolution via interpolation

    Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py

    Args:
        pos_embed: Position embeddings tensor of size [b, n, d]
        new_size: Target [height, width] of embedding
        old_size: Original [height, width] of embedding
        num_prefix_tokens: Number of non-spatial prefix tokens (eg. cls)
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [b, n', d]
    """

    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]

    if not old_size:
        old_size = int(math.sqrt(pos_embed.shape[1] - num_prefix_tokens))  # type:ignore
    old_size = to_2tuple(old_size)

    # Return if no resize necessary
    if new_size == old_size:
        return pos_embed

    if num_prefix_tokens:
        posemb_prefix, pos_embed = (
            pos_embed[:, :num_prefix_tokens],
            pos_embed[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, pos_embed = None, pos_embed

    # Interpolate position embedding
    pos_embed = pos_embed.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(pos_embed, size=new_size, mode=interpolation, antialias=antialias)
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    # Add back extra prefix tokens
    if posemb_prefix is not None:
        pos_embed = torch.cat([posemb_prefix, pos_embed], dim=1)

    return pos_embed


class FlexiPosEmbed(nn.Module):
    def __init__(
        self,
        grid_size: int,
        pos_embed_dim: int = 768,
        grid_sizes: Sequence[int] | None = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """2D resizable pos embedding, caching interpolated pos_embeds for different grid sizes

        Args:
            grid_size: Size of pos_embed buffer
            pos_embed_dim: positional embedding dimension size
            grid_sizes: List of grid sizes to cache interpolated pos_embeds for
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """
        super().__init__()

        grid_size = to_2tuple(grid_size)

        self.grid_size = grid_size
        self.pos_embed_dim = pos_embed_dim
        logger.info(f"flexible pos embed reference grid size: {grid_size}")
        self.ref_pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(self.pos_embed_dim, grid_size[0], cls_token=False)
        ).float()

        self.ref_pos_embed = nn.Parameter(self.ref_pos_embed, requires_grad=False)

        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias

        self.grid_sizes = grid_sizes

        if self.grid_sizes is not None:
            self.pos_embeds = nn.ParameterDict(self._cache_pos_embeds()).requires_grad_(False)
        else:
            self.pos_embeds = {}

    def _cache_pos_embeds(self) -> dict[int, torch.Tensor]:
        pos_embeds = {}
        for grid_size in self.grid_sizes:
            grid_size = to_2tuple(grid_size)
            if grid_size == self.grid_size:
                continue
            pos_embeds[str(grid_size)] = resize_abs_pos_embed(
                self.ref_pos_embed[None, :, :],
                new_size=grid_size,
                old_size=self.grid_size,
                num_prefix_tokens=0,
                interpolation=self.interpolation,
                antialias=self.antialias,
            ).squeeze(0)
        return pos_embeds

    def forward(self, grid_size) -> torch.Tensor:
        grid_size = to_2tuple(grid_size)
        if self.grid_size == grid_size:
            return self.ref_pos_embed

        if str(grid_size) not in self.pos_embeds:
            logger.info(f"interpolating new pos_embed for grid_size: {grid_size}, from {self.grid_size}")
            self.pos_embeds[str(grid_size)] = (
                resize_abs_pos_embed(
                    self.ref_pos_embed[None, :, :],
                    new_size=grid_size,
                    old_size=self.grid_size,
                    num_prefix_tokens=0,
                    interpolation=self.interpolation,
                    antialias=self.antialias,
                )
                .squeeze(0)
                .to(self.ref_pos_embed.device)
            )

        return self.pos_embeds[str(grid_size)]


class FlexiPosResEmbed(nn.Module):
    def __init__(
        self,
        pos_embed_dim: int = 768,
        ground_cover_lookup: dict[int, list[int]] | None = None,
        channels: Sequence[int] | None = None,
        interpolate: bool = False,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """2D resizable pos embedding, caching interpolated pos_embeds for different grid sizes

        Args:
            grid_size: Size of pos_embed buffer
            pos_embed_dim: positional embedding dimension size
            ground_cover_lookup: Dict mapping ground covers to grid sizes to patch size and gsd params
            channels: Dict mapping product bands to their channel info
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """
        super().__init__()

        self.pos_embed_dim = pos_embed_dim

        # Flexi specific attributes
        self.interpolate = interpolate
        self.interpolation = interpolation
        self.antialias = antialias

        self.channels = channels

        self.ground_cover_lookup = ground_cover_lookup

        self.base_grid_sizes = None
        if self.ground_cover_lookup is not None and not self.interpolate:
            self.pos_embeds = nn.ParameterDict(self._cache_pos_embeds()).requires_grad_(False)
        elif self.ground_cover_lookup is not None and self.interpolate:
            self.pos_embeds, self.base_grid_sizes = self._cache_pos_embeds_interpolate()
            self.pos_embeds = nn.ParameterDict(self.pos_embeds).requires_grad_(False)
        else:
            self.pos_embeds = {}

    def _cache_pos_embeds(self) -> dict[int, torch.Tensor]:
        pos_embeds = {}
        for ground_cover, grid_size_lookup in self.ground_cover_lookup.items():
            if ground_cover not in pos_embeds:
                pos_embeds[str(ground_cover)] = {}
            for grid_size, params in grid_size_lookup.items():
                grid_size = to_2tuple(grid_size)

                for p in params:
                    if grid_size not in pos_embeds[str(ground_cover)]:
                        patch_size_gsd = p["patch_size"] * p["gsd"]
                        pos_embeds[str(ground_cover)][str(grid_size)] = get_2d_sincos_pos_embed_with_resolution(
                            self.pos_embed_dim,
                            grid_size=grid_size[0],
                            res=torch.tensor([patch_size_gsd]),
                            cls_token=False,
                            centered=True,
                        ).squeeze(0)
        return pos_embeds

    def _cache_pos_embeds_interpolate(self) -> dict[int, torch.Tensor]:
        pos_embeds = {}
        base_grid_sizes = {}
        for ground_cover, grid_size_lookup in self.ground_cover_lookup.items():
            if ground_cover not in pos_embeds:
                pos_embeds[ground_cover] = {}

            for grid_size, params in grid_size_lookup.items():
                grid_size = to_2tuple(grid_size)
                for p in params:
                    patch_size_gsd = p["patch_size"] * ["gsd"]
                    if p["patch_size"] != self.channels[p["product_band"]]["patch_size"]:
                        continue
                    base_grid_sizes[str(ground_cover)] = grid_size
                    pos_embeds[str(ground_cover)][str(grid_size)] = get_2d_sincos_pos_embed_with_resolution(
                        self.pos_embed_dim,
                        grid_size=grid_size[0],
                        res=torch.tensor([patch_size_gsd]),
                        cls_token=False,
                        centered=True,
                    ).squeeze(0)

        for ground_cover, grid_size_lookup in self.ground_cover_lookup.items():
            if ground_cover not in pos_embeds:
                pos_embeds[ground_cover] = {}

            for grid_size, params in grid_size_lookup.items():
                grid_size = to_2tuple(grid_size)
                for p in params:
                    patch_size_gsd = p["patch_size"] * p["gsd"]

                    base_grid_size = base_grid_sizes[str(ground_cover)]
                    if grid_size == base_grid_size:
                        continue
                    pos_embeds[str(ground_cover)][str(grid_size)] = resize_abs_pos_embed(
                        pos_embeds[str(ground_cover)][str(base_grid_size)][None, :, :],
                        new_size=grid_size,
                        old_size=base_grid_size,
                        num_prefix_tokens=0,
                        interpolation=self.interpolation,
                        antialias=self.antialias,
                    ).squeeze(0)

        return pos_embeds, base_grid_sizes

    def forward(self, grid_size, patch_size, gsd, device) -> torch.Tensor:
        patch_size_gsd = patch_size * gsd
        ground_cover = str(grid_size * patch_size_gsd)
        grid_size = to_2tuple(grid_size)

        if ground_cover not in self.pos_embeds:
            self.pos_embeds[ground_cover] = {}

        if str(grid_size) not in self.pos_embeds[ground_cover]:
            self.pos_embeds[ground_cover][str(grid_size)] = get_2d_sincos_pos_embed_with_resolution(
                self.pos_embed_dim,
                grid_size=grid_size[0],
                res=torch.tensor([patch_size_gsd], device=device),
                cls_token=False,
                centered=True,
                device=device,
            ).squeeze(0)

        return self.pos_embeds[ground_cover][str(grid_size)].to(device)


def pi_resize_patch_embed(
    patch_embed: Tensor,
    new_patch_size: tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """Resample patch embedding weights to a target resolution via pseudo-inverse
    resizing.

    Based on:
        https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py
        https://arxiv.org/abs/2212.08013

    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [d, c h', w']
    """
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4D tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (height, width)"

    old_patch_size = tuple(patch_embed.shape[2:])

    # Return original kernel if no resize is necessary
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x: Tensor, shape: tuple[int, int]):
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=interpolation,
            antialias=antialias,
        )
        return x_resized[0, 0, ...]

    def calculate_pinv(old_shape: tuple[int, int], new_shape: tuple[int, int]):
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    # Calculate pseudo-inverse of resize matrix
    resize_matrix_pinv = calculate_pinv(old_patch_size, new_patch_size)
    resize_matrix_pinv = resize_matrix_pinv.to(patch_embed.device)

    def resample_patch_embed(patch_embed: Tensor):
        h, w = new_patch_size
        resampled_kernel = resize_matrix_pinv @ patch_embed.reshape(-1)
        return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

    v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

    return v_resample_patch_embed(patch_embed)


def interpolate_resize_patch_embed(
    patch_embed: Tensor,
    new_patch_size: tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """Resample patch embedding weights to a target resolution via interpolation

    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [d, c h', w']
    """
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4D tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (height, width)"

    patch_embed = F.interpolate(patch_embed, new_patch_size, mode=interpolation, antialias=antialias)

    return patch_embed


# More or less original from https://github.com/bwconrad/flexivit/blob/main/flexivit_pytorch/patch_embed.py
class FlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int] = 32,
        grid_size: int | tuple[int, int] = 7,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module | None = None,
        flatten: bool = True,
        bias: bool = True,
        patch_size_seq: Sequence[int] = (8, 10, 12, 15, 16, 20, 24, 30, 40, 48),
        patch_size_probs: Sequence[float] | None = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """2D image to patch embedding w/ flexible patch sizes
        Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24

        Args:
            patch_size: Base patch size. i.e the size of the parameter buffer
            grid_size: Size of pos_embed buffer
            in_chans: Number of input image channels
            embed_dim: Network embedding dimension size
            norm_layer: Optional normalization layer
            flatten: Whether to flatten the spatial dimensions of the output
            bias: Whether to use bias in convolution
            patch_size_seq: List of patch sizes to randomly sample from
            patch_size_probs: Optional list of probabilities to sample corresponding
                patch_size_seq elements. If None, then uniform distribution is used
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """
        super().__init__()

        self.patch_size = to_2tuple(patch_size)
        self.grid_size = to_2tuple(grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias

        self.patch_size_seq = patch_size_seq

        if self.patch_size_seq:
            if not patch_size_probs:
                n = len(self.patch_size_seq)
                self.patch_size_probs = [1.0 / n] * n
            else:
                self.patch_size_probs = [p / sum(patch_size_probs) for p in patch_size_probs]
        else:
            self.patch_size_probs = []

        # Pre-calculate pinvs
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        """Pre-calculate all pinv matrices"""
        pinvs = {}
        for ps in self.patch_size_seq:
            ps = to_2tuple(ps)
            pinvs[ps] = self._calculate_pinv(self.patch_size, ps)
        return pinvs

    def _resize(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(self, old_shape: tuple[int, int], new_shape: tuple[int, int]) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(self, patch_embed: Tensor, new_patch_size: tuple[int, int]):
        """Resize patch_embed to target resolution via pseudo-inverse resizing"""
        # Return original kernel if no resize is necessary
        if self.patch_size == new_patch_size:
            return patch_embed

        # Calculate pseudo-inverse of resize matrix
        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(self.patch_size, new_patch_size)
        pinv = self.pinvs[new_patch_size]
        pinv = pinv.to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor):
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

        return v_resample_patch_embed(patch_embed)

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
        return_patch_size: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[int, int]]:
        if not patch_size and not self.training:
            # During evaluation use base patch size if not specified
            patch_size = self.patch_size
        elif not patch_size:
            # During training choose uniformly at random if not specified
            assert self.patch_size_seq, (
                "No patch size specified during forward and no patch_size_seq given to FlexiPatchEmbed"
            )
            patch_size = np.random.choice(self.patch_size_seq, p=self.patch_size_probs)

        patch_size = to_2tuple(patch_size)

        # Resize conv weights
        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)

        # Apply conv with resized weights
        x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)

        if return_patch_size:
            return x, patch_size

        return x


class FlexiBase(nn.Module):
    def __init__(
        self,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """Base class for flexible thor patch embedding modules and MAE decoders"""
        super().__init__()

        self.interpolation = interpolation
        self.antialias = antialias

    def _resize(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(self, old_shape: tuple[int, int], new_shape: tuple[int, int]) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(self, patch_embed: Tensor, patch_size: tuple[int, int], new_patch_size: tuple[int, int]):
        """Resize patch_embed to target resolution via pseudo-inverse resizing"""
        # Return original kernel if no resize is necessary
        if patch_size == new_patch_size:
            return patch_embed

        # Calculate pseudo-inverse of resize matrix
        if patch_size not in self.pinvs or new_patch_size not in self.pinvs[patch_size]:
            if patch_size not in self.pinvs:
                self.pinvs[patch_size] = {}
            self.pinvs[patch_size][new_patch_size] = self._calculate_pinv(patch_size, new_patch_size)
        pinv = self.pinvs[patch_size][new_patch_size]
        pinv = pinv.to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor):
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

        return v_resample_patch_embed(patch_embed)

    def get_pinv(self, patch_size: tuple[int, int], new_patch_size: tuple[int, int], device):
        # Return original kernel if no resize is necessary
        if patch_size == new_patch_size:
            return torch.eye(np.prod(patch_size)).to(device)

        # Calculate pseudo-inverse of resize matrix
        if patch_size not in self.pinvs or new_patch_size not in self.pinvs[patch_size]:
            logger.info(f"getting and caching pinv for {patch_size} -> {new_patch_size}")
            if patch_size not in self.pinvs:
                self.pinvs[patch_size] = {}
            self.pinvs[patch_size][new_patch_size] = self._calculate_pinv(patch_size, new_patch_size)
        pinv = self.pinvs[patch_size][new_patch_size]
        pinv = pinv.to(device)
        return pinv


class IndFlexiPatchEmbed(FlexiBase):
    def __init__(
        self,
        ground_covers: list[int],
        channels: dict[str, dict[str, int]],
        groups: dict[str, list],
        channel_rename_map: dict[str, str] | None = None,
        min_patch_size: int = 6,
        embed_dim: int = 768,
        norm_layer: nn.Module | None = None,
        flatten: bool = True,
        bias: bool = True,
        patch_size_seqs: dict[str, Sequence[int]] | Sequence[int] = (8, 10, 12, 15, 16, 20, 24, 30, 40, 48),
        patch_size_probs: dict[str, Sequence[float]] | Sequence[float] | None = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """2D image to patch embedding w/ flexible patch sizes, for multiple product bands
        Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24

        Args:
            ground_cover: Ground cover size in meters
            channels: Dictionary of product bands and their parameters (GSD, num_patch)
            groups: Dictionary of channel groups and their members
            channel_rename_map: Dictionary of channel names to rename, to i.e., use same parameters for different bands
            min_patch_size: Minimum possible patch size to consider
            embed_dim: Network embedding dimension size
            norm_layer: Optional normalization layer
            flatten: Whether to flatten the spatial dimensions of the output
            bias: Whether to use bias in convolution
            patch_size_seqs: Dict of List of patch sizes for each band or list of patch sizes for all bands to
                randomly sample from, unvalidated patch sizes are dropped
            patch_size_probs: Optional Dict of list of probabilities for each band or list of probabilities for all
                bands to sample corresponding patch_size_seqs elements. If None, then uniform distribution is used
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """
        super().__init__(
            interpolation=interpolation,
            antialias=antialias,
        )

        self.flatten = flatten
        self.channels = channels
        self.groups = groups
        self.channel_rename_map = channel_rename_map
        self.min_patch_size = min_patch_size
        self.embed_dim = embed_dim
        self.patch_sizes = {}
        proj_dict = {}
        for product_band, params in channels.items():
            kernel_size = params["patch_size"]
            self.patch_sizes[product_band] = to_2tuple(kernel_size)
            if channel_rename_map and product_band in channel_rename_map:
                product_band = channel_rename_map[product_band]
            if product_band in proj_dict:
                logger.info(f"Product band {product_band} already added, skipping")
                continue
            proj_dict[product_band] = nn.Conv2d(1, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.patch_embed = nn.ModuleDict(proj_dict)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        if not isinstance(patch_size_seqs, dict):
            patch_size_seqs = dict.fromkeys(channels, patch_size_seqs)

        # filter valid patch size seqs
        missing_product_bands = []
        for product_band in patch_size_seqs.keys():
            _patch_size_seq = []

            for patch_size in patch_size_seqs[product_band]:
                product_gsd = channels[product_band]["GSD"]
                for ground_cover in ground_covers:
                    product_num_patch = ground_cover // patch_size // product_gsd
                    if patch_size * product_num_patch * product_gsd != ground_cover:
                        # Skip patch sizes that don't add up for the given ground cover
                        continue
                    if patch_size not in _patch_size_seq:
                        _patch_size_seq.append(patch_size)

            if len(_patch_size_seq) == 0:
                msg = (
                    f"No valid patch sizes for {product_band} for ground cover {ground_covers} and GSD {product_gsd}"
                    f" with patch size seq {patch_size_seqs[product_band]}"
                )
                logger.debug(msg)
                missing_product_bands.append(product_band)
                continue

            logger.info(f"product_band: {product_band}, _patch_size_seq: {_patch_size_seq}\n")
            patch_size_seqs[product_band] = sorted(_patch_size_seq)
        if missing_product_bands:
            logger.info(f"Missing product bands due to no valid patch sizes: {missing_product_bands}")

        self.patch_size_seqs = patch_size_seqs

        self.patch_size_probs = {}
        if self.patch_size_seqs:
            if not patch_size_probs:
                for product_band in self.patch_size_seqs.keys():
                    n = len(self.patch_size_seqs[product_band])
                    patch_size_probs = [1.0 / n] * n
                    self.patch_size_probs[product_band] = patch_size_probs
            elif isinstance(patch_size_probs, dict):
                self.patch_size_probs = {
                    product_band: [p / sum(patch_size_probs[product_band]) for p in patch_size_probs[product_band]]
                    for product_band in channels
                }
            else:
                patch_size_probs = [p / sum(patch_size_probs) for p in patch_size_probs]
                self.patch_size_probs = dict.fromkeys(channels, patch_size_probs)

        # Pre-calculate pinvs
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict[tuple[int, int], dict[tuple[int, int], Tensor]]:
        """Pre-calculate all pinv matrices"""
        pinvs = {}
        for product_band, patch_size in self.patch_sizes.items():
            if patch_size not in pinvs:
                pinvs[patch_size] = {}

            for ps in self.patch_size_seqs[product_band]:
                ps = to_2tuple(ps)
                if ps not in pinvs[patch_size] and ps != patch_size:
                    pinvs[patch_size][ps] = self._calculate_pinv(patch_size, ps)
        return pinvs

    def min_patch_size_check(self, img_size: int, token_budget: int | None) -> int:
        """Get the minimum viable patch size for a given image size and token budget"""
        if token_budget is None:
            # Return min_patch_size if no token budget (infinte budget)
            return self.min_patch_size
        if token_budget <= 0:
            # Token budget is used up
            return 1000000
        max_number_of_tokens_sqrt = int(token_budget**0.5)
        return math.ceil(img_size / max_number_of_tokens_sqrt)

    def get_available_groups(self, original_input: dict[str, torch.Tensor]) -> dict[str, list[str]]:
        available_groups = {}
        for group_name, group_member in self.groups.items():
            for member in group_member:
                if member in original_input:
                    if group_name not in available_groups:
                        available_groups[group_name] = [member]
                    else:
                        available_groups[group_name].append(member)
        return available_groups

    def random_patch_sizes(
        self,
        x: dict[str, Tensor],
        num_tokens: int,
        device: torch.device | None = None,
    ) -> dict[str, tuple[int, int]]:
        """Randomly sample patch sizes for each product band, optionally respecting a given token budget.

        The patch sizes will be the same for all product bands in a group.

        """
        patch_sizes: dict[str, tuple[int, int]] = {}
        # Find which groups are available in the input
        available_groups = self.get_available_groups(x)
        group_names = list(available_groups.keys())
        # Randomly shuffle the groups to avoid bias when using token budget
        group_idx = torch.randperm(len(available_groups))

        # Get lower and upper bounds for number of tokens possible per group
        lower_bounds = torch.zeros(len(group_names), dtype=torch.int64, device=device)
        upper_bounds = torch.zeros(len(group_names), dtype=torch.int64, device=device) * num_tokens

        for idx in group_idx:
            _group_name = group_names[idx]
            _group_members = available_groups[_group_name]
            first_product_band = _group_members[0]

            product_shape = x[first_product_band].shape[-1]
            # get largest and smallest patch size for the group
            patch_size_seq = self.patch_size_seqs[first_product_band]
            min_patch_size = min(patch_size_seq)
            max_patch_size = max(patch_size_seq)
            # get max and min number of tokens for the group

            min_num_tokens = max(2, math.floor(product_shape / max_patch_size)) ** 2
            max_num_tokens = min(32, math.ceil(product_shape / min_patch_size)) ** 2
            lower_bounds[idx] = min_num_tokens
            upper_bounds[idx] = max_num_tokens
            # logger.debug(
            #     f"product_band: {first_product_band}, min_num_tokens: {min_num_tokens}, max_num_tokens: {max_num_tokens}"
            # )

        partition_sizes = random_partition_sizes_compiled(
            num_tokens,
            lower_bounds,
            upper_bounds,
            device=device,
        )

        used_budget = 0
        for idx in group_idx:
            _group_name = group_names[idx]
            _group_members = available_groups[_group_name]
            _first_group_member: str | None = None
            # budget = partitions[idx, 1] - partitions[idx, 0]
            budget = partition_sizes[idx]
            for product_band in _group_members:
                if _first_group_member is not None:
                    if _first_group_member == "no_valid":
                        # If the first group member is not valid, skip this group
                        continue

                    # If group member, use the same patch size as the first group member
                    patch_sizes[product_band] = patch_sizes[_first_group_member]
                    continue

                product_shape = x[product_band].shape[-1]

                # Minimum patch size for the given image size and token budget
                patch_size_seq = self.patch_size_seqs[product_band]
                for patch_size in patch_size_seq:
                    used_tokens = int((product_shape // patch_size) ** 2)
                    if used_tokens <= budget and product_shape % patch_size == 0:
                        patch_sizes[product_band] = (patch_size, patch_size)
                        _first_group_member = product_band
                        # calculate number of tokens for the group
                        used_budget += used_tokens
                        # logger.debug(
                        #     f"product_band: {product_band}, patch_size: {patch_size}, budget: {budget}, used tokens: {used_tokens}"
                        # )
                        break

                if _first_group_member is None:
                    # If no valid patch size is found, skip this group
                    _first_group_member = "no_valid"
                    continue

        logger.debug(f"used budget: {used_budget}, remaining budget: {num_tokens - used_budget}\n")
        return patch_sizes

    def forward(
        self,
        x: dict[str, Tensor],
        patch_sizes: dict[str, tuple[int, int]] | None = None,
        token_budget: int | None = 1792,
        return_patch_size: bool = False,
        device: torch.device | None = None,
    ) -> Tensor | tuple[Tensor, tuple[int, int]]:
        if patch_sizes is None and not self.training:
            # During evaluation use base patch sizes if not specified
            patch_sizes = self.patch_sizes

        elif patch_sizes is None:
            # During training choose random patch sizes
            if self.training:
                assert token_budget is not None, (
                    "No token budget specified during forward and no patch_sizes given to FlexiPatchEmbed"
                )
            else:
                token_budget = 1000000
            patch_sizes = self.random_patch_sizes(
                x,
                num_tokens=int(token_budget),
                device=device,
            )

        patch_embed_dict = {}
        patch_size_dict = {}
        for product_band, data in x.items():
            if product_band not in patch_sizes:
                logger.debug(f"Skipping product band: {product_band}\n")
                continue

            patch_size = patch_sizes[product_band]
            patch_size = to_2tuple(patch_size)

            patch_embed_name = self.channel_rename_map[product_band] if self.channel_rename_map else product_band

            # Resize conv weights
            if patch_size == self.patch_sizes[product_band]:
                weight = self.patch_embed[patch_embed_name].weight
            else:
                weight = self.resize_patch_embed(
                    self.patch_embed[patch_embed_name].weight, self.patch_sizes[product_band], patch_size
                )

            # Apply conv with resized weights
            data = F.conv2d(data, weight, bias=self.patch_embed[patch_embed_name].bias, stride=patch_size)

            if self.flatten:
                data = data.flatten(2).transpose(1, 2)  # BCHW -> BNC

            data = self.norm(data)
            patch_embed_dict[product_band] = data

            if return_patch_size:
                patch_size_dict[product_band] = patch_size

        if return_patch_size:
            return patch_embed_dict, patch_size_dict
        return patch_embed_dict


class FlexiLinDecoder(FlexiBase):
    def __init__(
        self,
        ground_covers: int,
        channels: dict[str, dict[str, int]],
        groups: dict[str, list],
        decoder_embed_dim: int = 768,
        patch_size_seqs: dict[str, Sequence[int]] | None = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
        pi_inverse: bool = False,
    ) -> None:
        """Flexible linear decoder for multiple product bands, maps embeddings to image patches

        Args:
            ground_cover: Ground cover size in meters
            channels: Dictionary of product bands and their parameters (GSD, num_patch)
            groups: Dictionary of groups and their members
            decoder_embed_dim: Network embedding dimension size
            patch_size_seqs: Dict of list of patch sizes to randomly sample from
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
            pi_inverse: Whether to use pseudo-inverse resizing or normal interpolation
        """
        super().__init__(
            interpolation=interpolation,
            antialias=antialias,
        )

        self.ground_covers = ground_covers
        self.channels = channels
        self.groups = groups
        self.decoder_embed_dim = decoder_embed_dim
        self.pi_inverse = pi_inverse

        self.patch_size_seqs = patch_size_seqs

        self.patch_sizes: dict[str, tuple[int, int]] = {}
        # compute data embed size without addtional embeding
        self.decode_pred = {}
        for group_name, group_members in self.groups.items():
            product_band = group_members[0]
            params = self.channels[product_band]
            kernel_size = params["patch_size"]
            self.decode_pred[group_name] = nn.Linear(decoder_embed_dim, len(group_members) * kernel_size**2)
            self.patch_sizes[group_name] = to_2tuple(kernel_size)
        self.decode_pred = nn.ModuleDict(self.decode_pred)

        # Pre-calculate pinvs
        if patch_size_seqs is not None:
            logger.info("Caching lin decoder pinvs")
            self.pinvs = self._cache_pinvs()
        else:
            self.pinvs = {}

    def _cache_pinvs(self) -> dict:
        """Pre-calculate all pinv matrices"""
        pinvs = {}
        for group_name, patch_size in self.patch_sizes.items():
            product_band = self.groups[group_name][0]
            if patch_size not in pinvs:
                pinvs[patch_size] = {}

            for ps in self.patch_size_seqs[product_band]:
                ps = to_2tuple(ps)
                if ps not in pinvs[patch_size] and ps != patch_size:
                    logger.info(f"Caching pinv for group: {group_name}, patch_size: {patch_size}, new_patch_size: {ps}")
                    pinvs[patch_size][ps] = self._calculate_pinv(patch_size, ps)
        return pinvs

    def get_pinv(self, patch_size: tuple[int, int], new_patch_size: tuple[int, int], device):
        # Return original kernel if no resize is necessary
        if patch_size == new_patch_size:
            return torch.eye(np.prod(patch_size)).to(device)

        # Calculate pseudo-inverse of resize matrix
        if patch_size not in self.pinvs or new_patch_size not in self.pinvs[patch_size]:
            logger.info(f"getting and caching pinv for {patch_size} -> {new_patch_size}")
            if patch_size not in self.pinvs:
                self.pinvs[patch_size] = {}
            self.pinvs[patch_size][new_patch_size] = self._calculate_pinv(patch_size, new_patch_size)
        pinv = self.pinvs[patch_size][new_patch_size]
        pinv = pinv.to(device)
        return pinv

    def unshuffle_weight(self, w, c=1):
        """
        w: (c*p*p, d)
        c: Num channels
        ret : (d, c, p, p)
        """
        d = w.shape[1]
        num_patches = w.shape[0] // c
        p = int(num_patches**0.5)

        # assert p**2 == num_patches
        # assert w.shape[0] == c * p * p

        ret = rearrange(w, "(c p q) d -> d c p q", c=c, d=d, p=p, q=p)
        return ret

    def shuffle_weight(self, w, c=1):
        """
        w: (d, c, p, p)
        c: Num channels
        ret : (c*p*p, d)
        """

        # d, c, p, p = w.shape
        # assert c == c
        # num_patches = p**2

        ret = rearrange(w, "d c p q -> (c p q) d")
        return ret

    def forward(
        self,
        x: Tensor,
        group_name: str,
        new_patch_size: int | tuple[int, int],
        return_pinv: bool = False,
    ) -> Tensor:
        new_patch_size = to_2tuple(new_patch_size)
        old_patch_size = self.patch_sizes[group_name]
        num_group_members = len(self.groups[group_name])

        # Resize linear weights
        weight = self.decode_pred[group_name].weight
        bias = self.decode_pred[group_name].bias
        if new_patch_size != old_patch_size:
            bias = bias.unsqueeze(1)
            bias = self.unshuffle_weight(bias, c=num_group_members)
            weight = self.unshuffle_weight(weight, c=num_group_members)

            if self.pi_inverse:
                bias = self.resize_patch_embed(bias, old_patch_size, new_patch_size)
                weight = self.resize_patch_embed(weight, old_patch_size, new_patch_size)
            else:
                bias = interpolate_resize_patch_embed(bias, new_patch_size, self.interpolation, self.antialias)
                weight = interpolate_resize_patch_embed(weight, new_patch_size, self.interpolation, self.antialias)

            weight = self.shuffle_weight(weight, c=num_group_members)
            bias = self.shuffle_weight(bias, c=num_group_members)
            bias = bias.squeeze(1)

        # Apply conv with resized weights
        data = F.linear(x, weight, bias)

        if return_pinv:
            return data, self.get_pinv(old_patch_size, new_patch_size, x.device)
        return data


class FlexiConvTransDecoder(FlexiBase):
    def __init__(
        self,
        ground_covers: list[int],
        channels: dict[str, dict[str, int]],
        groups: dict[str, list],
        decoder_embed_dim: int = 768,
        patch_size_seqs: dict[str, Sequence[int]] | None = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
        pi_inverse: bool = False,
    ) -> None:
        """Flexible tranpose conv decoder for multiple product bands, maps embeddings to image patches

        Args:
            ground_cover: Ground cover size in meters
            channels: Dictionary of product bands and their parameters (GSD, num_patch)
            groups: Dictionary of groups and their members
            decoder_embed_dim: Network embedding dimension size
            patch_size_seqs: Dict of list of patch sizes to randomly sample from
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
            pi_inverse: Whether to use pseudo-inverse resizing or normal interpolation
        """
        super().__init__(
            interpolation=interpolation,
            antialias=antialias,
        )

        self.ground_covers = ground_covers
        self.channels = channels
        self.groups = groups
        self.decoder_embed_dim = decoder_embed_dim
        self.pi_inverse = pi_inverse

        self.patch_size_seqs = patch_size_seqs

        self.patch_sizes = {}
        # compute data embed size without addtional embeding
        self.decode_pred = {}
        for group_name, group_members in self.groups.items():
            product_band = group_members[0]
            params = self.channels[product_band]
            kernel_size = params["patch_size"]
            self.decode_pred[group_name] = nn.ConvTranspose2d(
                decoder_embed_dim, len(group_members), kernel_size, stride=kernel_size
            )
            self.patch_sizes[group_name] = to_2tuple(kernel_size)
        self.decode_pred = nn.ModuleDict(self.decode_pred)

        # Pre-calculate pinvs
        if patch_size_seqs is not None:
            self.pinvs = self._cache_pinvs()
        else:
            self.pinvs = {}

    def _cache_pinvs(self) -> dict:
        """Pre-calculate all pinv matrices"""
        pinvs = {}
        for group_name, patch_size in self.patch_sizes.items():
            product_band = self.groups[group_name][0]
            if patch_size not in pinvs:
                pinvs[patch_size] = {}

            for ps in self.patch_size_seqs[product_band]:
                ps = to_2tuple(ps)

                if ps not in pinvs[patch_size] and ps != patch_size:
                    logger.info(f"Caching pinv for patch_size: {patch_size}, new_patch_size: {ps}")
                    pinvs[patch_size][ps] = self._calculate_pinv(patch_size, ps)
        return pinvs

    def forward(
        self,
        x: Tensor,
        group_name: str,
        new_patch_size: int | tuple[int, int],
        return_pinv: bool = False,
    ) -> Tensor:
        new_patch_size = to_2tuple(new_patch_size)

        BN, D = x.shape
        x = x.view(BN, D, 1, 1)

        bias = self.decode_pred[group_name].bias
        weight = self.decode_pred[group_name].weight

        old_patch_size = self.patch_sizes[group_name]

        # Resize conv weights
        if new_patch_size != old_patch_size:
            if self.pi_inverse:
                weight = self.resize_patch_embed(weight, old_patch_size, new_patch_size)
            else:
                weight = interpolate_resize_patch_embed(weight, new_patch_size, self.interpolation, self.antialias)

        # # Apply conv transpose with resized weights
        data = F.conv_transpose2d(x, weight, bias=bias, stride=new_patch_size)
        # data: [B*N, C, H, W]

        if return_pinv:
            return data, self.get_pinv(old_patch_size, new_patch_size, x.device)
        return data

    def functional(
        self,
        x: Tensor,
        module: nn.ConvTranspose2d,
        old_patch_size: int | tuple[int, int],
        new_patch_size: int | tuple[int, int],
        pi_inverse: bool = False,
        return_pinv: bool = False,
    ) -> Tensor:
        old_patch_size = to_2tuple(old_patch_size)
        new_patch_size = to_2tuple(new_patch_size)

        BN, D = x.shape
        x = x.view(BN, D, 1, 1)

        bias = module.bias
        weight = module.weight

        # Resize conv weights
        if new_patch_size != old_patch_size:
            if pi_inverse:
                weight = self.resize_patch_embed(weight, old_patch_size, new_patch_size)
            else:
                weight = interpolate_resize_patch_embed(weight, new_patch_size, self.interpolation, self.antialias)

        # # Apply conv transpose with resized weights
        data = F.conv_transpose2d(x, weight, bias=bias, stride=new_patch_size)
        # data: [B*N, C, H, W]

        if return_pinv:
            return data, self.get_pinv(old_patch_size, new_patch_size, x.device)
        return data
