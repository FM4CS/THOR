from __future__ import annotations

import warnings
from copy import deepcopy

import pytest
import torch
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock

from thor.models import patch_timm
from thor.models.thor_vit import (
    CompactAlibiSpec,
    ThorViTEncoder,
    alibi_cls_token_pad,
    compact_alibi_to_dense,
    get_alibi_thor,
    get_compact_alibi_thor,
)


def make_input_params(cls_token_type: str = "pooled") -> dict:
    return {
        "ground_covers": [32],
        "channels": {
            "A:Band": {
                "GSD": 1,
                "patch_size": 8,
            },
            "B:Band": {
                "GSD": 2,
                "patch_size": 4,
            },
        },
        "groups": [["A:Band"], ["B:Band"]],
        "encoder_pos_type": "alibi",
        "cls_token_type": cls_token_type,
        "use_flexivit": False,
    }


def make_model(cls_token_type: str = "pooled", device: str | torch.device = "cpu") -> ThorViTEncoder:
    torch.manual_seed(0)
    model = ThorViTEncoder(
        deepcopy(make_input_params(cls_token_type=cls_token_type)),
        embed_dim=32,
        depth=2,
        num_heads=4,
        embed_prod=False,
        prod_embed_dim=0,
        embed_band=False,
        band_embed_dim=0,
    )
    return model.to(device=device, dtype=torch.float32)


def make_inputs(batch_size: int = 2, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "A:Band": torch.randn(batch_size, 1, 32, 32, device=device),
        "B:Band": torch.randn(batch_size, 1, 16, 16, device=device),
    }


def prepare_encoder_alibi_inputs(
    model: ThorViTEncoder,
    x: dict[str, torch.Tensor],
    ground_cover: int = 32,
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    patch_sizes = model.get_patch_sizes(
        method=model.select_patch_strategy,
        x=x,
        patch_sizes=model.ind_patch_embed.patch_size_seqs,
        ground_cover=ground_cover,
    )
    patch_embed = model.ind_patch_embed(x=x, patch_sizes=patch_sizes, device=model.device)
    available_groups = model.get_available_groups(patch_embed)
    channel_params = model.get_channel_params(patch_embed, metadata=None, ground_cover=ground_cover)
    return available_groups, channel_params


def can_run_flex_attention(device: torch.device) -> bool:
    try:
        from torch.nn.attention.flex_attention import flex_attention
    except ImportError:
        return False

    q = torch.randn(1, 4, 8, 8, device=device)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="flex_attention called without torch.compile\\(\\)",
                category=UserWarning,
            )
            flex_attention(q, q, q)
    except Exception:
        return False
    return True


def patch_attention_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    patch_timm.clear_flex_attention_impl_cache()
    monkeypatch.setattr(TimmBlock, "forward", patch_timm._alibi_block_forward)
    if mode == "dense":
        monkeypatch.setattr(TimmAttention, "forward", patch_timm._alibi_attn_forward)
        monkeypatch.setattr(patch_timm, "use_flex_attn", lambda: False)
        return

    from torch.nn.attention.flex_attention import flex_attention

    monkeypatch.setattr(TimmAttention, "forward", patch_timm._alibi_attn_flex_forward)
    monkeypatch.setattr(patch_timm, "use_flex_attn", lambda: True)
    monkeypatch.setattr(patch_timm, "flex_attention", flex_attention, raising=False)


def test_get_flex_attention_impl_compiles_once(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled_impl = object()
    compile_calls: list[tuple[object, dict]] = []
    raw_impl = object()

    def fake_compile(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return compiled_impl

    patch_timm.clear_flex_attention_impl_cache()
    monkeypatch.setattr(patch_timm, "use_flex_attn", lambda: True)
    monkeypatch.setattr(patch_timm, "COMPILE_FLEX_ATTENTION", True)
    monkeypatch.setattr(patch_timm, "FLEX_ATTENTION_COMPILE_MODE", None)
    monkeypatch.setattr(patch_timm, "_flex_attention_plain", raw_impl, raising=False)
    monkeypatch.setattr(patch_timm.torch, "compile", fake_compile)

    assert patch_timm.get_flex_attention_impl("plain") is compiled_impl
    assert patch_timm.get_flex_attention_impl("plain") is compiled_impl
    assert compile_calls == [(raw_impl, {"dynamic": True})]

    patch_timm.clear_flex_attention_impl_cache()


def test_compact_alibi_matches_dense_reconstruction() -> None:
    metadata = {
        "group_a": {"GSD": 1, "num_patch": 4, "patch_size": 8},
        "group_b": {"GSD": 2, "num_patch": 4, "patch_size": 4},
    }
    available_groups = {"group0": ["group_a"], "group1": ["group_b"]}
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125])

    dense = get_alibi_thor(metadata, available_groups, slopes=slopes, offset=1, dtype=torch.float32)
    dense = alibi_cls_token_pad(dense)

    compact = get_compact_alibi_thor(
        metadata,
        available_groups,
        slopes=slopes,
        num_prefix_tokens=1,
        dtype=torch.float32,
    )
    rebuilt = compact_alibi_to_dense(compact)

    assert torch.allclose(rebuilt, dense)


def test_eval_flex_prepares_compact_alibi(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model(cls_token_type="token")
    model.eval()
    x = make_inputs(batch_size=2)
    available_groups, channel_params = prepare_encoder_alibi_inputs(model, x)

    monkeypatch.setattr(patch_timm, "use_flex_attn", lambda: True)
    alibi = model._build_encoder_alibi(
        available_groups,
        channel_params,
        batch_size=2,
        device=model.device,
        dtype=torch.float32,
    )

    assert isinstance(alibi, CompactAlibiSpec)
    assert alibi.q_x.shape == alibi.k_x.shape
    assert alibi.q_y.shape == alibi.k_y.shape
    expected_num_tokens = sum(params["num_patch"] ** 2 for params in channel_params.values())
    assert alibi.q_x.numel() == expected_num_tokens
    assert alibi.num_prefix_tokens == 1


def test_training_keeps_dense_alibi_with_flex_available(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model(cls_token_type="token")
    model.train()
    x = make_inputs(batch_size=2)
    available_groups, channel_params = prepare_encoder_alibi_inputs(model, x)

    monkeypatch.setattr(patch_timm, "use_flex_attn", lambda: True)
    alibi = model._build_encoder_alibi(
        available_groups,
        channel_params,
        batch_size=2,
        device=model.device,
        dtype=torch.float32,
    )

    assert isinstance(alibi, torch.Tensor)
    expected_num_tokens = sum(params["num_patch"] ** 2 for params in channel_params.values()) + model.num_prefix_tokens
    assert alibi.shape == (2, model.num_heads, expected_num_tokens, expected_num_tokens)


@pytest.mark.parametrize("cls_token_type", ["pooled", "token"])
def test_eval_outputs_match_between_dense_and_compact_alibi(
    monkeypatch: pytest.MonkeyPatch,
    cls_token_type: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not can_run_flex_attention(device):
        pytest.skip("flex_attention is not runnable on this device")

    model = make_model(cls_token_type=cls_token_type, device=device)
    model.eval()
    x = make_inputs(batch_size=2, device=device)

    patch_attention_mode(monkeypatch, mode="dense")
    with torch.no_grad():
        dense_output = model(x, ground_cover=32)

    patch_attention_mode(monkeypatch, mode="flex")
    with torch.no_grad():
        compact_output = model(x, ground_cover=32)

    assert torch.allclose(compact_output, dense_output, atol=1e-4, rtol=1e-4)


def test_large_compact_alibi_smoke_does_not_materialize_dense_tensor() -> None:
    metadata = {
        "group_a": {"GSD": 1, "num_patch": 128, "patch_size": 4},
    }
    available_groups = {"group0": ["group_a"]}
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125])

    compact = get_compact_alibi_thor(
        metadata,
        available_groups,
        slopes=slopes,
        num_prefix_tokens=1,
        dtype=torch.float32,
    )

    assert isinstance(compact, CompactAlibiSpec)
    assert compact.q_x.numel() == 128 * 128
    assert compact.slopes.shape == slopes.shape
