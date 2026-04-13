"""Tests for thor.utils.patch_embed — FlexiViT patch embedding components.

Inspired by the official flexivit_pytorch tests, adapted for the THOR
multi-channel (IndFlexiPatchEmbed) and position-embedding helpers.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from thor.utils.patch_embed import (
    FlexiPatchEmbed,
    FlexiPosEmbed,
    IndFlexiPatchEmbed,
    pi_resize_patch_embed,
    resize_abs_pos_embed,
)

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

EMBED_DIM = 64
GROUND_COVER = 960  # metres


def _make_ind_embed(
    channels=None,
    groups=None,
    patch_size_seqs=None,
    embed_dim=EMBED_DIM,
    **kwargs,
) -> IndFlexiPatchEmbed:
    if channels is None:
        channels = {"S2:Red": {"GSD": 10, "patch_size": 16}}
    if groups is None:
        groups = {"group0": ["S2:Red"]}
    if patch_size_seqs is None:
        patch_size_seqs = [16]
    channel_rename_map = {k: k for k in channels}
    return IndFlexiPatchEmbed(
        ground_covers=[GROUND_COVER],
        channels=channels,
        groups=groups,
        channel_rename_map=channel_rename_map,
        embed_dim=embed_dim,
        patch_size_seqs=patch_size_seqs,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# pi_resize_patch_embed
# ---------------------------------------------------------------------------


def _pi_resize_invariance(old_shape, new_shape, n_patches=1):
    """
    Core invariance check: resizing both the image patch and the conv weight
    (via PI resize) should produce the same embedding as the original.
    NOTE: Only holds for n_patches=1 because bicubic interpolation is non-local.
    """
    patch_size = old_shape[2:]
    new_patch_size = new_shape[2:]

    patches = torch.randn(n_patches, *old_shape[1:])
    w_orig = torch.randn(*old_shape)
    orig_out = F.conv2d(patches, w_orig, stride=patch_size, padding="valid")

    patches_resized = F.interpolate(patches, new_patch_size, mode="bicubic", antialias=True)
    w_resized = pi_resize_patch_embed(w_orig, new_patch_size, interpolation="bicubic", antialias=True)
    assert w_resized.shape == new_shape

    new_out = F.conv2d(patches_resized, w_resized, stride=new_patch_size, padding="valid")
    assert orig_out.shape == new_out.shape

    np.testing.assert_allclose(
        orig_out.detach().numpy(),
        new_out.detach().numpy(),
        rtol=1e-1,
        atol=1e-4,
    )


def test_pi_resize_output_shape():
    w = torch.randn(64, 1, 16, 16)
    resized = pi_resize_patch_embed(w, (8, 8))
    assert resized.shape == (64, 1, 8, 8)


def test_pi_resize_same_size_noop():
    w = torch.randn(64, 1, 16, 16)
    resized = pi_resize_patch_embed(w, (16, 16))
    assert resized is w  # should return original tensor unchanged


def test_pi_resize_dtype_preserved():
    w = torch.randn(32, 3, 8, 8)
    resized = pi_resize_patch_embed(w, (12, 12))
    assert resized.dtype == w.dtype


def test_pi_resize_upsampling_invariance_square():
    """PI resize invariance for common square upsampling pairs (single-patch)."""
    d = 64
    for old_s in [8, 10, 12]:
        for new_s in [old_s, old_s + 4, 16]:
            if new_s <= old_s:
                continue
            _pi_resize_invariance((d, 1, old_s, old_s), (d, 1, new_s, new_s))


def test_pi_resize_upsampling_invariance_multichannel():
    for c in [1, 3]:
        _pi_resize_invariance((64, c, 8, 8), (64, c, 12, 12))


def test_pi_resize_downsampling_shape_and_type():
    """Downsampling does not guarantee value equality; just check shape/type."""
    for t in [4, 5, 6, 7]:
        for c in [1, 3]:
            w = torch.randn(64, c, 8, 8)
            out = pi_resize_patch_embed(w, (t, t))
            assert out.shape == (64, c, t, t)
            assert out.dtype == w.dtype


def test_pi_resize_rejects_wrong_ndim():
    with pytest.raises(AssertionError):
        pi_resize_patch_embed(torch.randn(64, 16, 16), (8, 8))  # 3D, not 4D


def test_pi_resize_rejects_wrong_new_patch_size():
    with pytest.raises(AssertionError):
        pi_resize_patch_embed(torch.randn(64, 1, 16, 16), (8, 8, 8))  # 3-tuple


# ---------------------------------------------------------------------------
# resize_abs_pos_embed
# ---------------------------------------------------------------------------


def test_resize_abs_pos_embed_same_size_noop():
    pos = torch.randn(1, 36, 128)  # (B, 6*6, D)
    out = resize_abs_pos_embed(pos, new_size=6, num_prefix_tokens=0)
    assert torch.equal(out, pos)


def test_resize_abs_pos_embed_different_size():
    pos = torch.randn(1, 36, 128)  # 6x6 grid
    out = resize_abs_pos_embed(pos, new_size=4, num_prefix_tokens=0)
    assert out.shape == (1, 16, 128)  # 4x4 grid


def test_resize_abs_pos_embed_preserves_prefix_tokens():
    # 1 cls token + 36 spatial = 37 tokens
    pos = torch.randn(1, 37, 128)
    out = resize_abs_pos_embed(pos, new_size=4, num_prefix_tokens=1)
    # cls token preserved, spatial resized 6x6 → 4x4
    assert out.shape == (1, 1 + 16, 128)
    # cls token should be unchanged
    assert torch.equal(out[:, :1, :], pos[:, :1, :])


def test_resize_abs_pos_embed_upscale():
    pos = torch.randn(1, 9, 64)  # 3x3 grid
    out = resize_abs_pos_embed(pos, new_size=6, num_prefix_tokens=0)
    assert out.shape == (1, 36, 64)  # 6x6 grid


# ---------------------------------------------------------------------------
# FlexiPosEmbed
# ---------------------------------------------------------------------------


def test_flexi_pos_embed_base_size_returns_ref():
    pe = FlexiPosEmbed(grid_size=6, pos_embed_dim=64)
    out = pe(6)
    assert out.shape == (36, 64)  # 6*6 tokens


def test_flexi_pos_embed_cached_size():
    pe = FlexiPosEmbed(grid_size=6, pos_embed_dim=64, grid_sizes=[4, 8])
    out = pe(4)
    assert out.shape == (16, 64)  # 4*4


def test_flexi_pos_embed_uncached_interpolates():
    pe = FlexiPosEmbed(grid_size=6, pos_embed_dim=64)
    out = pe(3)
    assert out.shape == (9, 64)  # 3*3, interpolated on-the-fly


def test_flexi_pos_embed_caches_after_first_call():
    pe = FlexiPosEmbed(grid_size=6, pos_embed_dim=64)
    pe(3)
    assert str((3, 3)) in pe.pos_embeds


def test_flexi_pos_embed_no_nans():
    pe = FlexiPosEmbed(grid_size=7, pos_embed_dim=128, grid_sizes=[4, 7, 10])
    for size in [4, 7, 10, 5]:
        out = pe(size)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# FlexiPatchEmbed (single-channel, THOR reimplementation)
# ---------------------------------------------------------------------------


def test_flexi_patch_embed_forward_base_patch_size():
    pe = FlexiPatchEmbed(patch_size=16, grid_size=6, in_chans=1, embed_dim=64)
    pe.eval()
    x = torch.randn(2, 1, 96, 96)
    out, ps = pe(x, return_patch_size=True)
    assert ps == (16, 16)
    assert out.shape == (2, 36, 64)


def test_flexi_patch_embed_forward_explicit_patch_size():
    pe = FlexiPatchEmbed(patch_size=16, grid_size=12, in_chans=1, embed_dim=64)
    x = torch.randn(2, 1, 96, 96)
    for patch_size in [8, 16]:
        out, ps = pe(x, patch_size=patch_size, return_patch_size=True)
        expected_tokens = (96 // patch_size) ** 2
        assert out.shape == (2, expected_tokens, 64)
        assert ps == (patch_size, patch_size)


def test_flexi_patch_embed_eval_uses_base_patch_size():
    pe = FlexiPatchEmbed(patch_size=16, in_chans=1, embed_dim=64)
    pe.eval()
    x = torch.randn(2, 1, 96, 96)
    _, ps = pe(x, return_patch_size=True)
    assert ps == (16, 16)


def test_flexi_patch_embed_no_nans():
    pe = FlexiPatchEmbed(patch_size=16, in_chans=1, embed_dim=64)
    pe.eval()
    x = torch.randn(4, 1, 96, 96)
    out = pe(x)
    assert not torch.isnan(out).any()


def test_flexi_patch_embed_resize_invariance():
    """PI-resized weights on an upsampled single patch ≈ original weight on original patch.

    The invariance only holds for upsampling (new kernel ≥ base kernel).
    """
    # Base kernel is 8; we upsample the kernel to 16 via PI resize.
    pe = FlexiPatchEmbed(patch_size=8, grid_size=1, in_chans=1, embed_dim=64, patch_size_seq=[8, 16])
    pe.eval()

    # Single 8×8 image → 1 token with base kernel
    x_orig = torch.randn(1, 1, 8, 8)
    orig_out = pe(x_orig, patch_size=8)

    # Upsample image to 16×16; PI-resized kernel (8→16) should reproduce the same embedding
    x_big = F.interpolate(x_orig, (16, 16), mode="bicubic", antialias=True)
    big_out = pe(x_big, patch_size=16)

    np.testing.assert_allclose(
        orig_out.detach().numpy(),
        big_out.detach().numpy(),
        rtol=1e-1,
        atol=1e-4,
    )


# ---------------------------------------------------------------------------
# IndFlexiPatchEmbed — THOR multi-channel embedding
# ---------------------------------------------------------------------------


def test_ind_flexi_patch_embed_forward_eval_shape():
    embed = _make_ind_embed()
    embed.eval()
    x = {"S2:Red": torch.randn(2, 1, 96, 96)}
    out = embed(x)
    assert "S2:Red" in out
    assert out["S2:Red"].shape == (2, 36, EMBED_DIM)  # 6*6=36 patches


def test_ind_flexi_patch_embed_forward_explicit_patch_sizes():
    embed = _make_ind_embed(patch_size_seqs=[8, 16])
    embed.eval()
    x = {"S2:Red": torch.randn(2, 1, 96, 96)}
    for ps, expected_n in [(8, 144), (16, 36)]:
        out, ps_dict = embed(x, patch_sizes={"S2:Red": ps}, return_patch_size=True)
        assert out["S2:Red"].shape == (2, expected_n, EMBED_DIM)
        assert ps_dict["S2:Red"] == (ps, ps)


def test_ind_flexi_patch_embed_forward_no_nans():
    embed = _make_ind_embed()
    embed.eval()
    x = {"S2:Red": torch.randn(2, 1, 96, 96)}
    out = embed(x)
    assert not torch.isnan(out["S2:Red"]).any()


def test_ind_flexi_patch_embed_forward_multi_channel():
    channels = {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:Green": {"GSD": 10, "patch_size": 16},
        "S2:RE1": {"GSD": 20, "patch_size": 16},
    }
    groups = {
        "group0": ["S2:Red", "S2:Green"],
        "group1": ["S2:RE1"],
    }
    embed = _make_ind_embed(channels=channels, groups=groups)
    embed.eval()
    x = {
        "S2:Red": torch.randn(2, 1, 96, 96),
        "S2:Green": torch.randn(2, 1, 96, 96),
        "S2:RE1": torch.randn(2, 1, 48, 48),
    }
    out = embed(x)
    assert out["S2:Red"].shape == (2, 36, EMBED_DIM)  # 6*6
    assert out["S2:Green"].shape == (2, 36, EMBED_DIM)
    assert out["S2:RE1"].shape == (2, 9, EMBED_DIM)  # 3*3


def test_ind_flexi_patch_embed_skips_bands_not_in_patch_sizes():
    """Bands not present in patch_sizes dict should be skipped."""
    embed = _make_ind_embed()
    embed.eval()
    x = {"S2:Red": torch.randn(2, 1, 96, 96)}
    # Pass empty patch_sizes — no bands should be embedded
    out = embed(x, patch_sizes={})
    assert len(out) == 0


def test_ind_flexi_patch_embed_train_token_budget_constrains_tokens():
    """Training with a token budget should not exceed the budget per group."""
    embed = _make_ind_embed(patch_size_seqs=[8, 16])
    embed.train()

    x = {"S2:Red": torch.randn(1, 1, 96, 96)}

    # budget=50: only patch_size=16 fits (36 tokens) since patch_size=8 → 144 > 50
    out = embed(x, token_budget=50)
    assert out["S2:Red"].shape[1] == 36

    # budget=200: patch_size=8 fits (144 tokens <= 200)
    out = embed(x, token_budget=200)
    assert out["S2:Red"].shape[1] == 144


def test_ind_flexi_patch_embed_channel_rename_shares_weights():
    """Two bands sharing patch_embed_name should use the same conv weights."""
    channels = {
        "S1:IW-VV": {"GSD": 10, "patch_size": 16},
        "S1:EW-VV": {"GSD": 10, "patch_size": 16},
    }
    groups = {"group0": ["S1:IW-VV", "S1:EW-VV"]}
    rename = {"S1:IW-VV": "S1:VV", "S1:EW-VV": "S1:VV"}  # both → same key
    embed = IndFlexiPatchEmbed(
        ground_covers=[GROUND_COVER],
        channels=channels,
        groups=groups,
        channel_rename_map=rename,
        embed_dim=EMBED_DIM,
        patch_size_seqs=[16],
    )
    # Only one unique conv kernel should exist
    assert len(embed.patch_embed) == 1
    assert "S1:VV" in embed.patch_embed


def test_ind_flexi_patch_embed_resize_invariance():
    """PI-resized weights on an upsampled single patch ≈ base weights on original patch.

    The invariance only holds for upsampling (new kernel ≥ base kernel).
    Base kernel is 16; we upsample to 32.  patch_size=32 is valid for
    ground_cover=960, GSD=10 because 960 // 32 // 10 = 3 exactly.
    """
    channels = {"S2:Red": {"GSD": 10, "patch_size": 16}}
    groups = {"group0": ["S2:Red"]}
    rename = {"S2:Red": "S2:Red"}
    embed = IndFlexiPatchEmbed(
        ground_covers=[GROUND_COVER],
        channels=channels,
        groups=groups,
        channel_rename_map=rename,
        embed_dim=EMBED_DIM,
        patch_size_seqs=[16, 32],
        bias=False,  # bias-free makes the invariance exact
    )
    embed.eval()

    # Single 16×16 image → 1 token with base kernel (patch_size=16)
    x_16 = torch.randn(1, 1, 16, 16)
    out_16 = embed({"S2:Red": x_16}, patch_sizes={"S2:Red": 16})

    # Upsample to 32×32; PI-resized kernel (16→32) should reproduce the same embedding
    x_32 = F.interpolate(x_16, (32, 32), mode="bicubic", antialias=True)
    out_32 = embed({"S2:Red": x_32}, patch_sizes={"S2:Red": 32})

    np.testing.assert_allclose(
        out_16["S2:Red"].detach().numpy(),
        out_32["S2:Red"].detach().numpy(),
        rtol=1e-1,
        atol=1e-4,
    )


def test_ind_flexi_patch_embed_valid_patch_size_filtering():
    """Patch sizes that don't evenly divide the image for any ground cover are dropped."""
    # GSD=10, ground_cover=960: valid patch sizes where 960 / (ps * 10) is integer
    # patch_size=7 → 960 / 70 ≈ 13.7 (not integer) → should be filtered out
    channels = {"S2:Red": {"GSD": 10, "patch_size": 16}}
    groups = {"group0": ["S2:Red"]}
    embed = _make_ind_embed(channels=channels, groups=groups, patch_size_seqs=[7, 16])
    assert 7 not in embed.patch_size_seqs.get("S2:Red", [])
    assert 16 in embed.patch_size_seqs["S2:Red"]


def test_ind_flexi_patch_embed_min_patch_size_check_no_budget():
    embed = _make_ind_embed()
    # No budget → returns min_patch_size
    result = embed.min_patch_size_check(img_size=96, token_budget=None)
    assert result == embed.min_patch_size


def test_ind_flexi_patch_embed_min_patch_size_check_with_budget():
    embed = _make_ind_embed()
    # budget=36 → sqrt=6 → ceil(96/6)=16
    result = embed.min_patch_size_check(img_size=96, token_budget=36)
    assert result == 16


def test_ind_flexi_patch_embed_min_patch_size_check_zero_budget():
    embed = _make_ind_embed()
    result = embed.min_patch_size_check(img_size=96, token_budget=0)
    assert result == 1000000  # exhausted budget sentinel
