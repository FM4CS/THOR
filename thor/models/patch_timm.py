import logging
import os
from typing import Any

import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


ddp_flex_available = torch.torch_version.Version(torch.__version__) >= torch.torch_version.Version("2.6")
USE_FLEX_ATTENTION = os.environ.get("USE_FLEX_ATTENTION", "0") == "1"
COMPILE_FLEX_ATTENTION = os.environ.get("COMPILE_FLEX_ATTENTION", "1") == "1"
FLEX_ATTENTION_COMPILE_MODE = os.environ.get("FLEX_ATTENTION_COMPILE_MODE")


def use_flex_attn() -> bool:
    return ddp_flex_available and USE_FLEX_ATTENTION


if use_flex_attn():
    from torch.nn.attention.flex_attention import flex_attention


_COMPILED_FLEX_ATTENTION: Any | None = None


def _get_flex_attention_compile_kwargs() -> dict[str, Any]:
    compile_kwargs: dict[str, Any] = {"dynamic": True}
    if FLEX_ATTENTION_COMPILE_MODE:
        compile_kwargs["mode"] = FLEX_ATTENTION_COMPILE_MODE
    return compile_kwargs


def _supports_compiled_flex_attention_inputs(q: torch.Tensor, v: torch.Tensor) -> bool:
    if q.device.type != "cuda":
        return False
    return q.shape[-1] >= 16 and v.shape[-1] >= 16


def clear_flex_attention_impl_cache() -> None:
    global _COMPILED_FLEX_ATTENTION
    _COMPILED_FLEX_ATTENTION = None


def get_flex_attention_impl(q: torch.Tensor | None = None, v: torch.Tensor | None = None):
    if not use_flex_attn():
        msg = "Flex attention is not enabled."
        raise RuntimeError(msg)

    if not COMPILE_FLEX_ATTENTION:
        return flex_attention

    if q is not None and v is not None and not _supports_compiled_flex_attention_inputs(q, v):
        return flex_attention

    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        logger.info("Compiling flex_attention for fused attention kernels")
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, **_get_flex_attention_compile_kwargs())
    return _COMPILED_FLEX_ATTENTION


def _make_compact_alibi_score_mod(
    q_x: torch.Tensor,
    k_x: torch.Tensor,
    q_y: torch.Tensor,
    k_y: torch.Tensor,
    slopes: torch.Tensor,
    num_prefix_tokens: int,
):
    def apply_alibi(score, b, h, q_idx, kv_idx):
        prefix_mask = (q_idx < num_prefix_tokens) | (kv_idx < num_prefix_tokens)
        q_patch_idx = torch.clamp(q_idx - num_prefix_tokens, min=0)
        kv_patch_idx = torch.clamp(kv_idx - num_prefix_tokens, min=0)

        dx = q_x[q_patch_idx] - k_x[kv_patch_idx]
        dy = q_y[q_patch_idx] - k_y[kv_patch_idx]
        distance = torch.sqrt(dx * dx + dy * dy)
        bias = -slopes[h] * (distance + num_prefix_tokens)
        return torch.where(prefix_mask, score, score + bias)

    return apply_alibi


def _is_compact_alibi(alibi: Any | None) -> bool:
    return alibi is not None and all(
        hasattr(alibi, attr) for attr in ("q_x", "k_x", "q_y", "k_y", "slopes", "num_prefix_tokens")
    )


def _is_head_dim_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _alibi_attn_forward(self, x: torch.Tensor, alibi: Any | None = None) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if _is_compact_alibi(alibi):
        msg = "Compact ALiBi requires flex attention and is not supported by the dense attention fallback."
        raise TypeError(msg)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=alibi,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if alibi is not None:
            attn = attn + alibi
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _alibi_attn_flex_forward(self, x: torch.Tensor, alibi: Any | None = None) -> torch.Tensor:
    if not _is_head_dim_power_of_two(self.head_dim):
        msg = f"head_dim {self.head_dim} is not a power of 2, required for flex attention"
        raise ValueError(msg)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if _is_compact_alibi(alibi):
        score_mod = _make_compact_alibi_score_mod(
            alibi.q_x,
            alibi.k_x,
            alibi.q_y,
            alibi.k_y,
            alibi.slopes,
            alibi.num_prefix_tokens,
        )
        x = get_flex_attention_impl(q, v)(q, k, v, score_mod=score_mod)
    else:
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=alibi,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _alibi_block_forward(self, x: torch.Tensor, attn_mask: Any | None = None) -> torch.Tensor:
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


def enable_alibi_for_timm():
    if getattr(enable_alibi_for_timm, "_done", False):
        return
    logger.info("Patching timm Attention and Block to use Alibi...")
    if use_flex_attn():
        logger.info("Using flex attention for timm Attention")
        TimmAttention.forward = _alibi_attn_flex_forward
    else:
        logger.info("Using normal attention for timm Attention")
        TimmAttention.forward = _alibi_attn_forward
    TimmBlock.forward = _alibi_block_forward
    enable_alibi_for_timm._done = True
