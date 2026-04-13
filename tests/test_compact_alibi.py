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


def make_model(
    cls_token_type: str = "pooled",
    device: str | torch.device = "cpu",
    *,
    embed_dim: int = 32,
    num_heads: int = 4,
) -> ThorViTEncoder:
    torch.manual_seed(0)
    model = ThorViTEncoder(
        deepcopy(make_input_params(cls_token_type=cls_token_type)),
        embed_dim=embed_dim,
        depth=2,
        num_heads=num_heads,
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
    monkeypatch.setattr(patch_timm, "flex_attention", raw_impl, raising=False)
    monkeypatch.setattr(patch_timm.torch, "compile", fake_compile)

    assert patch_timm.get_flex_attention_impl() is compiled_impl
    assert patch_timm.get_flex_attention_impl() is compiled_impl
    assert compile_calls == [(raw_impl, {"dynamic": True})]

    patch_timm.clear_flex_attention_impl_cache()


def test_get_flex_attention_impl_keeps_small_cuda_heads_uncompiled(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_calls: list[tuple[object, dict]] = []
    raw_impl = object()

    class FakeTensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.device = torch.device("cuda")
            self.shape = shape

    def fake_compile(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return object()

    patch_timm.clear_flex_attention_impl_cache()
    monkeypatch.setattr(patch_timm, "use_flex_attn", lambda: True)
    monkeypatch.setattr(patch_timm, "COMPILE_FLEX_ATTENTION", True)
    monkeypatch.setattr(patch_timm, "flex_attention", raw_impl, raising=False)
    monkeypatch.setattr(patch_timm.torch, "compile", fake_compile)

    q = FakeTensor((2, 4, 17, 8))
    v = FakeTensor((2, 4, 17, 8))

    assert patch_timm.get_flex_attention_impl(q=q, v=v) is raw_impl
    assert compile_calls == []

    patch_timm.clear_flex_attention_impl_cache()


def test_make_compact_alibi_score_mod_clamps_prefix_indices() -> None:
    class IndexCheckingTensor:
        def __init__(self, values: torch.Tensor) -> None:
            self.values = values

        def __getitem__(self, idx: torch.Tensor | int) -> torch.Tensor:
            idx_value = int(idx.item()) if isinstance(idx, torch.Tensor) else int(idx)
            assert idx_value >= 0
            return self.values[idx_value]

    score_mod = patch_timm._make_compact_alibi_score_mod(
        IndexCheckingTensor(torch.tensor([0.0, 1.0])),
        IndexCheckingTensor(torch.tensor([0.0, 1.0])),
        IndexCheckingTensor(torch.tensor([0.0, 1.0])),
        IndexCheckingTensor(torch.tensor([0.0, 1.0])),
        torch.tensor([1.0]),
        num_prefix_tokens=1,
    )

    score = torch.tensor(3.0)
    out = score_mod(score, torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0))
    assert torch.equal(out, score)


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


def test_compact_alibi_reuses_coordinate_storage() -> None:
    metadata = {
        "group_a": {"GSD": 1, "num_patch": 4, "patch_size": 8},
    }
    available_groups = {"group0": ["group_a"]}
    compact = get_compact_alibi_thor(
        metadata,
        available_groups,
        slopes=torch.tensor([1.0, 0.5]),
        num_prefix_tokens=1,
        dtype=torch.float32,
    )

    assert compact.q_x.data_ptr() == compact.k_x.data_ptr()
    assert compact.q_y.data_ptr() == compact.k_y.data_ptr()


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


def test_flex_forward_uses_sdpa_for_dense_alibi(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model(cls_token_type="token")
    attn = model.blocks[0].attn
    x = torch.randn(2, 17, model.embed_dim)
    dense_alibi = torch.randn(2, attn.num_heads, x.shape[1], x.shape[1])
    calls: list[dict[str, object]] = []

    def fake_sdpa(q, k, v, attn_mask=None, dropout_p=0.0):
        calls.append({"attn_mask": attn_mask, "dropout_p": dropout_p, "shape": tuple(q.shape)})
        return torch.zeros_like(q)

    monkeypatch.setattr(patch_timm.F, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setattr(
        patch_timm,
        "get_flex_attention_impl",
        lambda *args, **kwargs: pytest.fail("dense ALiBi should not use flex_attention"),
    )

    out = patch_timm._alibi_attn_flex_forward(attn, x, dense_alibi)

    assert out.shape == x.shape
    assert calls and calls[0]["attn_mask"] is dense_alibi


def test_flex_forward_uses_sdpa_when_alibi_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model()
    attn = model.blocks[0].attn
    x = torch.randn(2, 16, model.embed_dim)
    calls: list[dict[str, object]] = []

    def fake_sdpa(q, k, v, attn_mask=None, dropout_p=0.0):
        calls.append({"attn_mask": attn_mask, "dropout_p": dropout_p, "shape": tuple(q.shape)})
        return torch.zeros_like(q)

    monkeypatch.setattr(patch_timm.F, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setattr(
        patch_timm,
        "get_flex_attention_impl",
        lambda *args, **kwargs: pytest.fail("no-ALiBi path should not use flex_attention"),
    )

    out = patch_timm._alibi_attn_flex_forward(attn, x, None)

    assert out.shape == x.shape
    assert calls and calls[0]["attn_mask"] is None


def test_flex_forward_uses_compact_score_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model(cls_token_type="token")
    attn = model.blocks[0].attn
    x = torch.randn(2, 17, model.embed_dim)
    compact = CompactAlibiSpec(
        q_x=torch.linspace(0.0, 1.0, steps=16),
        k_x=torch.linspace(0.0, 1.0, steps=16),
        q_y=torch.linspace(0.0, 1.0, steps=16),
        k_y=torch.linspace(0.0, 1.0, steps=16),
        slopes=torch.ones(attn.num_heads),
        num_prefix_tokens=1,
    )
    captured: dict[str, object] = {}

    def fake_flex(q, k, v, score_mod=None):
        captured["score_mod"] = score_mod
        return torch.zeros_like(q)

    monkeypatch.setattr(
        patch_timm.F,
        "scaled_dot_product_attention",
        lambda *args, **kwargs: pytest.fail("compact ALiBi should use flex_attention"),
    )
    monkeypatch.setattr(patch_timm, "get_flex_attention_impl", lambda *args, **kwargs: fake_flex)

    out = patch_timm._alibi_attn_flex_forward(attn, x, compact)

    assert out.shape == x.shape
    assert callable(captured["score_mod"])


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


@pytest.mark.parametrize("cls_token_type", ["pooled", "token"])
def test_eval_outputs_match_between_dense_and_compact_alibi_compiled_cuda(
    monkeypatch: pytest.MonkeyPatch,
    cls_token_type: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for compiled flex_attention parity")

    device = torch.device("cuda")
    if not can_run_flex_attention(device):
        pytest.skip("flex_attention is not runnable on this device")

    model = make_model(cls_token_type=cls_token_type, device=device, embed_dim=64, num_heads=4)
    assert model.blocks[0].attn.head_dim >= 16
    model.eval()
    x = make_inputs(batch_size=2, device=device)

    monkeypatch.setattr(patch_timm, "COMPILE_FLEX_ATTENTION", True)

    patch_attention_mode(monkeypatch, mode="dense")
    with torch.no_grad():
        dense_output = model(x, ground_cover=32)

    patch_attention_mode(monkeypatch, mode="flex")
    with torch.no_grad():
        compact_output = model(x, ground_cover=32)

    assert patch_timm._COMPILED_FLEX_ATTENTION is not None
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
