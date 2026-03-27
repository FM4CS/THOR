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


_FLEX_ATTENTION_IMPL_CACHE: dict[str, Any] = {}


def _get_flex_attention_compile_kwargs() -> dict[str, Any]:
    compile_kwargs: dict[str, Any] = {"dynamic": True}
    if FLEX_ATTENTION_COMPILE_MODE:
        compile_kwargs["mode"] = FLEX_ATTENTION_COMPILE_MODE
    return compile_kwargs


def _flex_attention_plain(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return flex_attention(q, k, v)


def _flex_attention_dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alibi: torch.Tensor) -> torch.Tensor:
    def apply_alibi(score, b, h, q_idx, kv_idx):
        return score + alibi[b, h, q_idx, kv_idx]

    return flex_attention(
        q,
        k,
        v,
        score_mod=apply_alibi,
    )


def _flex_attention_compact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_x: torch.Tensor,
    k_x: torch.Tensor,
    q_y: torch.Tensor,
    k_y: torch.Tensor,
    slopes: torch.Tensor,
    num_prefix_tokens: int,
) -> torch.Tensor:
    def apply_alibi(score, b, h, q_idx, kv_idx):
        prefix_mask = (q_idx < num_prefix_tokens) | (kv_idx < num_prefix_tokens)
        q_patch_idx = q_idx - num_prefix_tokens
        kv_patch_idx = kv_idx - num_prefix_tokens

        dx = q_x[q_patch_idx] - k_x[kv_patch_idx]
        dy = q_y[q_patch_idx] - k_y[kv_patch_idx]
        distance = torch.sqrt(dx * dx + dy * dy)
        bias = -slopes[h] * (distance + num_prefix_tokens)
        return torch.where(prefix_mask, score, score + bias)

    return flex_attention(
        q,
        k,
        v,
        score_mod=apply_alibi,
    )


def clear_flex_attention_impl_cache() -> None:
    _FLEX_ATTENTION_IMPL_CACHE.clear()


def get_flex_attention_impl(kind: str = "plain"):
    if not use_flex_attn():
        msg = "Flex attention is not enabled."
        raise RuntimeError(msg)

    impl_lookup = {
        "plain": _flex_attention_plain,
        "dense": _flex_attention_dense,
        "compact": _flex_attention_compact,
    }
    if kind not in impl_lookup:
        msg = f"Unknown flex attention implementation kind: {kind}"
        raise ValueError(msg)

    impl = impl_lookup[kind]
    if not COMPILE_FLEX_ATTENTION:
        return impl

    cached_impl = _FLEX_ATTENTION_IMPL_CACHE.get(kind)
    if cached_impl is None:
        logger.info("Compiling flex_attention for fused attention kernels (%s)", kind)
        cached_impl = torch.compile(impl, **_get_flex_attention_compile_kwargs())
        _FLEX_ATTENTION_IMPL_CACHE[kind] = cached_impl
    return cached_impl


def _is_compact_alibi(alibi: Any | None) -> bool:
    return alibi is not None and all(
        hasattr(alibi, attr) for attr in ("q_x", "k_x", "q_y", "k_y", "slopes", "num_prefix_tokens")
    )


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
    def is_power_of_two(n):
        if n <= 0:
            return False
        return (n & (n - 1)) == 0

    if not is_power_of_two(self.head_dim):
        msg = f"head_dim {self.head_dim} is not a power of 2, please use a power of 2 for the head_dim for flexi attention to work"
        raise ValueError(msg)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if _is_compact_alibi(alibi):
        x = get_flex_attention_impl("compact")(
            q,
            k,
            v,
            alibi.q_x,
            alibi.k_x,
            alibi.q_y,
            alibi.k_y,
            alibi.slopes,
            alibi.num_prefix_tokens,
        )
    elif alibi is None:
        x = get_flex_attention_impl("plain")(q, k, v)
    else:
        x = get_flex_attention_impl("dense")(q, k, v, alibi)

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
