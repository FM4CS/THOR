import logging
import os

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


def use_flex_attn() -> bool:
    return ddp_flex_available and USE_FLEX_ATTENTION


if use_flex_attn():
    from torch.nn.attention.flex_attention import flex_attention


def _alibi_attn_forward(self, x: torch.Tensor, alibi: torch.Tensor | None = None) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

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


def _alibi_attn_flex_forward(self, x: torch.Tensor, alibi: torch.Tensor | None = None) -> torch.Tensor:
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

    # B, Hq, L, E
    def apply_alibi(score, b, h, q_idx, kv_idx):
        if alibi is None:
            return score
        bias = alibi[b, h, q_idx, kv_idx]
        return score + bias

    x = flex_attention(
        q,
        k,
        v,
        score_mod=apply_alibi,
    )

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _alibi_block_forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
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
