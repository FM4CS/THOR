"""Tests for the core ThorViTEncoder functionality."""

from functools import partial

import pytest
import torch
from torch import nn

from thor.models.thor_vit import (
    ThorViTEncoder,
    alibi_cls_token_pad,
    get_alibi_thor,
    get_slopes,
)

# ---------------------------------------------------------------------------
# Constants (ground_cover=960m, patch_size=16px)
# ---------------------------------------------------------------------------
GROUND_COVER = 960

# S2 10m: image = 960/10 = 96px; patches = (96/16)^2 = 36
IMG_10M = 96
NUM_PATCHES_10M = (IMG_10M // 16) ** 2  # 36

# S2 20m: image = 960/20 = 48px; patches = (48/16)^2 = 9
IMG_20M = 48
NUM_PATCHES_20M = (IMG_20M // 16) ** 2  # 9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_input_params(**overrides) -> dict:
    """Return a fresh input_params dict (ThorViTEncoder.__init__ mutates it via pop)."""
    params: dict = {
        "ground_covers": [GROUND_COVER],
        "channels": {"S2:Red": {"GSD": 10, "patch_size": 16}},
        # Use a single valid patch size that matches the conv kernel, so no FlexiViT
        # resizing is needed and output shapes are fully predictable.
        "flexivit_patch_size_seqs": [16],
        "encoder_pos_type": "alibi",
        "cls_token_type": "pooled",
        "aggr_type": "subsetmean",
    }
    params.update(overrides)
    return params


def tiny_encoder(input_params: dict, **extra) -> ThorViTEncoder:
    """Build a minimal ThorViTEncoder (embed_dim=64, depth=2) for fast tests."""
    kwargs = dict(
        embed_dim=64,
        depth=2,
        num_heads=4,
        embed_band=True,
        band_embed_dim=16,
        embed_prod=False,
        prod_embed_dim=0,
        embed_patch_size=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    kwargs.update(extra)
    return ThorViTEncoder(input_params=input_params, **kwargs)


def s2_red_input(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {"S2:Red": torch.randn(batch_size, 1, IMG_10M, IMG_10M)}


# ---------------------------------------------------------------------------
# get_slopes
# ---------------------------------------------------------------------------


def test_get_slopes_length_power_of_2():
    assert len(get_slopes(8)) == 8


def test_get_slopes_length_non_power_of_2():
    assert len(get_slopes(12)) == 12


def test_get_slopes_all_positive():
    assert all(s > 0 for s in get_slopes(12))


def test_get_slopes_strictly_decreasing_power_of_2():
    slopes = get_slopes(8)
    assert all(slopes[i] > slopes[i + 1] for i in range(len(slopes) - 1))


# ---------------------------------------------------------------------------
# get_alibi_thor
# ---------------------------------------------------------------------------


def _make_alibi_metadata(num_patch: int = 6) -> tuple:
    metadata = {"S2:Red": {"GSD": 10, "num_patch": num_patch, "patch_size": 16}}
    available_groups = {"group0": ["S2:Red"]}
    slopes = torch.tensor(get_slopes(4))
    return metadata, available_groups, slopes


def test_get_alibi_thor_output_shape():
    metadata, groups, slopes = _make_alibi_metadata(num_patch=6)
    alibi = get_alibi_thor(metadata, groups, slopes)
    assert alibi.shape == (1, 4, 36, 36)  # 6*6=36 patches


def test_get_alibi_thor_diagonal_is_zero():
    """Self-attention bias: distance from patch to itself should be 0."""
    metadata, groups, slopes = _make_alibi_metadata(num_patch=4)
    alibi = get_alibi_thor(metadata, groups, slopes, offset=0)
    for h in range(4):
        assert torch.allclose(alibi[0, h].diagonal(), torch.zeros(16))


def test_get_alibi_thor_non_positive():
    """All ALiBi biases should be <= 0 (negative distance penalty)."""
    metadata, groups, slopes = _make_alibi_metadata(num_patch=4)
    alibi = get_alibi_thor(metadata, groups, slopes, offset=0)
    assert (alibi <= 0).all()


def test_get_alibi_thor_two_groups():
    """Multi-group: total patches = group0 + group1."""
    metadata = {
        "S2:Red": {"GSD": 10, "num_patch": 6, "patch_size": 16},
        "S2:RE1": {"GSD": 20, "num_patch": 3, "patch_size": 16},
    }
    available_groups = {"group0": ["S2:Red"], "group1": ["S2:RE1"]}
    slopes = torch.tensor(get_slopes(4))
    alibi = get_alibi_thor(metadata, available_groups, slopes)
    total = 36 + 9  # 45
    assert alibi.shape == (1, 4, total, total)


# ---------------------------------------------------------------------------
# alibi_cls_token_pad
# ---------------------------------------------------------------------------


def test_alibi_cls_token_pad_shape():
    alibi = torch.zeros(1, 4, 36, 36)
    padded = alibi_cls_token_pad(alibi)
    assert padded.shape == (1, 4, 37, 37)


def test_alibi_cls_token_pad_first_row_col_zero():
    alibi = torch.ones(2, 4, 36, 36)
    padded = alibi_cls_token_pad(alibi)
    assert (padded[:, :, 0, :] == 0).all()
    assert (padded[:, :, :, 0] == 0).all()


def test_alibi_cls_token_pad_interior_unchanged():
    alibi = torch.ones(1, 4, 36, 36) * 5.0
    padded = alibi_cls_token_pad(alibi)
    assert (padded[:, :, 1:, 1:] == 5.0).all()


# ---------------------------------------------------------------------------
# ThorViTEncoder initialisation
# ---------------------------------------------------------------------------


def test_init_single_channel_pooled():
    model = tiny_encoder(make_input_params())
    assert model.embed_dim == 64
    assert len(model.blocks) == 2
    assert model.cls_token is None
    assert model.num_prefix_tokens == 0


def test_init_cls_token():
    model = tiny_encoder(make_input_params(cls_token_type="token"))
    assert model.cls_token is not None
    assert model.num_prefix_tokens == 1


def test_init_accepts_global_int_patch_size_seq():
    model = tiny_encoder(make_input_params(flexivit_patch_size_seqs=16))
    assert model.ind_patch_embed.patch_size_seqs["S2:Red"] == [16]


def test_init_accepts_per_band_patch_size_seq_dict():
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:RE1": {"GSD": 20, "patch_size": 16},
    }
    patch_size_seqs = {
        "S2:Red": 8,
        "S2:RE1": [8, 16],
    }
    model = tiny_encoder(
        make_input_params(
            channels=channels,
            groups=[["S2:Red"], ["S2:RE1"]],
            flexivit_patch_size_seqs=patch_size_seqs,
        )
    )

    assert model.ind_patch_embed.patch_size_seqs["S2:Red"] == [8]
    assert model.ind_patch_embed.patch_size_seqs["S2:RE1"] == [8, 16]


def test_init_warns_per_band_patch_size_seq_dict_missing_channel(caplog):
    """Missing channels fall back to default patch size and emit a warning."""
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:RE1": {"GSD": 20, "patch_size": 16},
    }
    import logging

    with caplog.at_level(logging.WARNING):
        model = tiny_encoder(
            make_input_params(
                channels=channels,
                groups=[["S2:Red"], ["S2:RE1"]],
                flexivit_patch_size_seqs={"S2:Red": [8, 16]},
            )
        )
    assert any("Missing flexivit_patch_size_seqs" in r.message for r in caplog.records)
    # S2:RE1 was missing from the dict — it should have fallen back to the default patch size
    assert "S2:RE1" in model.ind_patch_embed.patch_size_seqs


def test_init_two_groups():
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:Green": {"GSD": 10, "patch_size": 16},
        "S2:RE1": {"GSD": 20, "patch_size": 16},
    }
    groups = [["S2:Red", "S2:Green"], ["S2:RE1"]]
    model = tiny_encoder(make_input_params(channels=channels, groups=groups))
    assert len(model.groups) == 2


def test_init_validate_group_raises_on_gsd_mismatch():
    """Channels with different GSD must not be put in the same group."""
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:RE1": {"GSD": 20, "patch_size": 16},
    }
    groups = [["S2:Red", "S2:RE1"]]
    with pytest.raises(ValueError, match="GSD"):
        tiny_encoder(make_input_params(channels=channels, groups=groups))


def test_init_non_flexivit_uses_per_band_min_patch_size_seq_for_num_patch():
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:RE1": {"GSD": 20, "patch_size": 16},
    }
    model = tiny_encoder(
        make_input_params(
            channels=channels,
            groups=[["S2:Red"], ["S2:RE1"]],
            use_flexivit=False,
            flexivit_patch_size_seqs={
                "S2:Red": [8, 16],
                "S2:RE1": 16,
            },
        )
    )

    assert model.channels["S2:Red"]["num_patch"] == GROUND_COVER // 10 // 8
    assert model.channels["S2:RE1"]["num_patch"] == GROUND_COVER // 20 // 16


# ---------------------------------------------------------------------------
# get_available_groups
# ---------------------------------------------------------------------------


def test_get_available_groups_all_present():
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:Green": {"GSD": 10, "patch_size": 16},
    }
    model = tiny_encoder(make_input_params(channels=channels, groups=[["S2:Red"], ["S2:Green"]]))
    x = {
        "S2:Red": torch.randn(1, 1, IMG_10M, IMG_10M),
        "S2:Green": torch.randn(1, 1, IMG_10M, IMG_10M),
    }
    avail = model.get_available_groups(x)
    assert set(avail.keys()) == {"group0", "group1"}


def test_get_available_groups_partial_input():
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:Green": {"GSD": 10, "patch_size": 16},
    }
    model = tiny_encoder(make_input_params(channels=channels, groups=[["S2:Red"], ["S2:Green"]]))
    x = {"S2:Red": torch.randn(1, 1, IMG_10M, IMG_10M)}  # only one group
    avail = model.get_available_groups(x)
    assert "group0" in avail
    assert "group1" not in avail


# ---------------------------------------------------------------------------
# aggregate_by_group
# ---------------------------------------------------------------------------


def test_aggregate_by_group_mean():
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:Green": {"GSD": 10, "patch_size": 16},
    }
    model = tiny_encoder(make_input_params(channels=channels, groups=[["S2:Red", "S2:Green"]], aggr_type="mean"))
    B, N, D = 2, NUM_PATCHES_10M, 64
    patch_embed = {
        "S2:Red": torch.ones(B, N, D),
        "S2:Green": torch.ones(B, N, D) * 3.0,
    }
    result = model.aggregate_by_group(patch_embed, {"group0": ["S2:Red", "S2:Green"]})
    assert torch.allclose(result["group0"], torch.ones(B, N, D) * 2.0)


def test_aggregate_by_group_subsetmean_missing_band():
    """subsetmean with one of two bands absent should use only the present band."""
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:Green": {"GSD": 10, "patch_size": 16},
    }
    model = tiny_encoder(make_input_params(channels=channels, groups=[["S2:Red", "S2:Green"]], aggr_type="subsetmean"))
    B, N, D = 2, NUM_PATCHES_10M, 64
    patch_embed = {"S2:Red": torch.ones(B, N, D) * 7.0}  # S2:Green absent
    result = model.aggregate_by_group(patch_embed, {"group0": ["S2:Red", "S2:Green"]})
    assert torch.allclose(result["group0"], torch.ones(B, N, D) * 7.0)


def test_aggregate_by_group_sum():
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:Green": {"GSD": 10, "patch_size": 16},
    }
    model = tiny_encoder(make_input_params(channels=channels, groups=[["S2:Red", "S2:Green"]], aggr_type="sum"))
    B, N, D = 1, NUM_PATCHES_10M, 64
    patch_embed = {
        "S2:Red": torch.ones(B, N, D) * 2.0,
        "S2:Green": torch.ones(B, N, D) * 3.0,
    }
    result = model.aggregate_by_group(patch_embed, {"group0": ["S2:Red", "S2:Green"]})
    assert torch.allclose(result["group0"], torch.ones(B, N, D) * 5.0)


# ---------------------------------------------------------------------------
# get_patch_sizes
# ---------------------------------------------------------------------------


def test_get_patch_sizes_min():
    model = tiny_encoder(make_input_params(flexivit_patch_size_seqs=[8, 16]))
    x = {"S2:Red": torch.randn(1, 1, IMG_10M, IMG_10M)}
    sizes = model.get_patch_sizes("min", x, model.ind_patch_embed.patch_size_seqs)
    assert sizes["S2:Red"] == 8


def test_get_patch_sizes_max():
    model = tiny_encoder(make_input_params(flexivit_patch_size_seqs=[8, 16]))
    x = {"S2:Red": torch.randn(1, 1, IMG_10M, IMG_10M)}
    sizes = model.get_patch_sizes("max", x, model.ind_patch_embed.patch_size_seqs)
    assert sizes["S2:Red"] == 16


def test_get_patch_sizes_equal_min_max():
    """equal-min and equal-max select different target num_patches when multiple common values exist."""
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:RE1": {"GSD": 20, "patch_size": 16},
    }
    # GSD=10 with [4,8,16]: num_patches → {4:24, 8:12, 16:6}
    # GSD=20 with [4,8,16]: num_patches → {4:12, 8:6, 16:3}
    # Common num_patches: {6, 12}
    # equal-min → target=12 (most patches) → GSD=10 uses patch_size=8, GSD=20 uses patch_size=4
    # equal-max → target=6 (fewest patches) → GSD=10 uses patch_size=16, GSD=20 uses patch_size=8
    model = tiny_encoder(
        make_input_params(channels=channels, groups=[["S2:Red"], ["S2:RE1"]], flexivit_patch_size_seqs=[4, 8, 16])
    )
    x = {
        "S2:Red": torch.randn(1, 1, IMG_10M, IMG_10M),
        "S2:RE1": torch.randn(1, 1, IMG_20M, IMG_20M),
    }
    patch_sizes = model.ind_patch_embed.patch_size_seqs

    eq_min = model.get_patch_sizes("equal-min", x, patch_sizes)
    eq_max = model.get_patch_sizes("equal-max", x, patch_sizes)

    # equal-min → finest common resolution (most patches)
    assert eq_min["S2:Red"] == 8
    assert eq_min["S2:RE1"] == 4

    # equal-max → coarsest common resolution (fewest patches)
    assert eq_max["S2:Red"] == 16
    assert eq_max["S2:RE1"] == 8


def test_get_patch_sizes_unknown_method_raises():
    model = tiny_encoder(make_input_params())
    x = {"S2:Red": torch.randn(1, 1, IMG_10M, IMG_10M)}
    with pytest.raises(ValueError, match="Unknown method"):
        model.get_patch_sizes("bad-method", x, model.ind_patch_embed.patch_size_seqs)


# ---------------------------------------------------------------------------
# forward_encoder
# ---------------------------------------------------------------------------


def test_forward_encoder_output_shape_single_channel():
    model = tiny_encoder(make_input_params())
    model.eval()
    out = model.forward_encoder(s2_red_input(batch_size=2))
    assert out.shape == (2, NUM_PATCHES_10M, 64)


def test_forward_encoder_cls_token_adds_one_token():
    model = tiny_encoder(make_input_params(cls_token_type="token"))
    model.eval()
    out = model.forward_encoder(s2_red_input(batch_size=2))
    assert out.shape == (2, NUM_PATCHES_10M + 1, 64)


def test_forward_encoder_multi_group_shape():
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:Green": {"GSD": 10, "patch_size": 16},
        "S2:RE1": {"GSD": 20, "patch_size": 16},
    }
    groups = [["S2:Red", "S2:Green"], ["S2:RE1"]]
    model = tiny_encoder(make_input_params(channels=channels, groups=groups))
    model.eval()
    x = {
        "S2:Red": torch.randn(2, 1, IMG_10M, IMG_10M),
        "S2:Green": torch.randn(2, 1, IMG_10M, IMG_10M),
        "S2:RE1": torch.randn(2, 1, IMG_20M, IMG_20M),
    }
    out = model.forward_encoder(x)
    assert out.shape == (2, NUM_PATCHES_10M + NUM_PATCHES_20M, 64)


def test_forward_encoder_partial_groups():
    """Encoder should still run when only a subset of defined groups is provided."""
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:RE1": {"GSD": 20, "patch_size": 16},
    }
    model = tiny_encoder(make_input_params(channels=channels, groups=[["S2:Red"], ["S2:RE1"]]))
    model.eval()
    # Only pass the 10m group
    out = model.forward_encoder({"S2:Red": torch.randn(1, 1, IMG_10M, IMG_10M)})
    assert out.shape == (1, NUM_PATCHES_10M, 64)


def test_forward_encoder_return_channel_params():
    model = tiny_encoder(make_input_params())
    model.eval()
    out, ch_params = model.forward_encoder(s2_red_input(1), return_channel_params=True)
    assert isinstance(ch_params, dict)
    assert "S2:Red" in ch_params
    assert ch_params["S2:Red"]["GSD"] == 10
    assert ch_params["S2:Red"]["num_patch"] == 6  # 96 / 16 = 6
    assert ch_params["S2:Red"]["patch_size"] == 16


def test_forward_encoder_no_nans():
    model = tiny_encoder(make_input_params())
    model.eval()
    out = model.forward_encoder(s2_red_input(1))
    assert not torch.isnan(out).any()


def test_forward_encoder_batch_size_one_and_four():
    model = tiny_encoder(make_input_params())
    model.eval()
    for bs in (1, 4):
        out = model.forward_encoder(s2_red_input(bs))
        assert out.shape[0] == bs


# ---------------------------------------------------------------------------
# forward_intermediates
# ---------------------------------------------------------------------------


def test_forward_intermediates_default_returns_all_blocks():
    model = tiny_encoder(make_input_params())  # depth=2
    model.eval()
    final, intermediates = model.forward_intermediates(s2_red_input(1))
    assert len(intermediates) == 2
    assert final.shape == (1, NUM_PATCHES_10M, 64)


def test_forward_intermediates_selected_index():
    model = tiny_encoder(make_input_params())
    model.eval()
    _, intermediates = model.forward_intermediates(s2_red_input(1), indices=[0])
    assert len(intermediates) == 1
    assert intermediates[0].shape == (1, NUM_PATCHES_10M, 64)


def test_forward_intermediates_only():
    model = tiny_encoder(make_input_params())
    model.eval()
    result = model.forward_intermediates(s2_red_input(1), intermediates_only=True)
    assert isinstance(result, list)
    assert len(result) == 2


def test_forward_intermediates_return_channel_params():
    model = tiny_encoder(make_input_params())
    model.eval()
    intermediates, ch_params = model.forward_intermediates(
        s2_red_input(1), intermediates_only=True, return_channel_params=True
    )
    assert isinstance(intermediates, list)
    assert "S2:Red" in ch_params


def test_forward_intermediates_with_norm():
    model = tiny_encoder(make_input_params())
    model.eval()
    _, intermediates_normed = model.forward_intermediates(s2_red_input(1), norm=True)
    _, intermediates_raw = model.forward_intermediates(s2_red_input(1), norm=False)
    # Normed and raw intermediates should differ (norm layer is not identity)
    assert not torch.allclose(intermediates_normed[0], intermediates_raw[0])


# ---------------------------------------------------------------------------
# forward (top-level)
# ---------------------------------------------------------------------------


def test_forward_delegates_to_encoder():
    model = tiny_encoder(make_input_params())
    model.eval()
    out = model(s2_red_input(2))
    assert out.shape == (2, NUM_PATCHES_10M, 64)


def test_forward_return_channel_params():
    model = tiny_encoder(make_input_params())
    model.eval()
    out, ch_params = model(s2_red_input(1), return_channel_params=True)
    assert isinstance(ch_params, dict)
    assert out.shape == (1, NUM_PATCHES_10M, 64)
