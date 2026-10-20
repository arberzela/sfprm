"""Transformer layers of the EqR port.

Ported from upstream EqR, models/layers.py (github.com/locuslab/EqR @ 134c754, Apache-2.0,
Copyright (c) 2026 Benhao Huang, Zhengyang Geng, Zico Kolter).

One deviation: Attention always calls F.scaled_dot_product_attention. Upstream uses flash_attn_func
(FlashAttention-3, else -2) on CUDA when head_dim % 8 == 0 and SDPA otherwise. flash-attn is not
installed in this repo's environment. The math is the same (softmax(q k^T / sqrt(head_dim)) v,
non-causal); only the kernel differs, so bf16 rounding differs. On CPU upstream also takes the
SDPA path.
"""

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_
from models.layers import (  # noqa: F401  (re-exported for models.equilibrium_reasoning.eqr)
    CastedEmbedding,
    RotaryEmbedding,
    _find_multiple,
    apply_rotary_pos_emb,
    rms_norm,
    rotate_half,
)


CosSin = Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    None,
]


def apply_rotary_pos_emb_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = q.dtype
    q, k = q.to(cos_h.dtype), k.to(cos_h.dtype)
    d = q.size(-1) // 2
    q_h, q_w = q[..., :d], q[..., d:]
    k_h, k_w = k[..., :d], k[..., d:]
    cos_h, sin_h = cos_h[None, :, None, None, :], sin_h[None, :, None, None, :]
    cos_w, sin_w = cos_w[None, None, :, None, :], sin_w[None, None, :, None, :]
    q_h, k_h = q_h * cos_h + rotate_half(q_h) * sin_h, k_h * cos_h + rotate_half(k_h) * sin_h
    q_w, k_w = q_w * cos_w + rotate_half(q_w) * sin_w, k_w * cos_w + rotate_half(k_w) * sin_w
    return torch.cat([q_h, q_w], dim=-1).to(dtype), torch.cat([k_h, k_w], dim=-1).to(dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features**0.5)))
        self.bias = nn.Parameter(torch.zeros((out_features,))) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, self.weight.to(input.dtype), bias=bias)


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim: int, max_height: int, max_width: int, base: float, device=None) -> None:
        super().__init__()
        dim_half = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_half, 2, dtype=torch.float32, device=device) / dim_half))
        freqs_h = torch.outer(torch.arange(max_height, dtype=torch.float32, device=device), inv_freq)
        freqs_w = torch.outer(torch.arange(max_width, dtype=torch.float32, device=device), inv_freq)
        emb_h, emb_w = torch.cat((freqs_h, freqs_h), dim=-1), torch.cat((freqs_w, freqs_w), dim=-1)
        self.cos_cached_height = nn.Buffer(emb_h.cos(), persistent=False)
        self.sin_cached_height = nn.Buffer(emb_h.sin(), persistent=False)
        self.cos_cached_width = nn.Buffer(emb_w.cos(), persistent=False)
        self.sin_cached_width = nn.Buffer(emb_w.sin(), persistent=False)

    def forward(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return (self.cos_cached_height, self.cos_cached_width), (self.sin_cached_height, self.sin_cached_width)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, num_key_value_heads: int, causal: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.qkv_proj = CastedLinear(hidden_size, (num_heads + 2 * num_key_value_heads) * head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=False)

    def _apply_rope(self, cos_sin: CosSin, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if cos_sin is None:
            return query, key
        cos, sin = cos_sin
        if isinstance(cos, tuple) and isinstance(sin, tuple):
            cos_h, cos_w = cos
            sin_h, sin_w = sin
            b, n, heads, d = query.shape
            h, w = int(cos_h.shape[0]), int(cos_w.shape[0])
            if h * w != n:
                raise ValueError(f"2D RoPE needs {h * w} tokens, got seq_len={n}.")
            q, k = query.view(b, h, w, heads, d), key.view(b, h, w, heads, d)
            q, k = apply_rotary_pos_emb_2d(q, k, cos_h, sin_h, cos_w, sin_w)
            return q.view(b, n, heads, d), k.view(b, n, heads, d)
        return apply_rotary_pos_emb(query, key, cos, sin)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        b, n, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(b, n, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        q = qkv[:, :, : self.num_heads]
        k = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        v = qkv[:, :, self.num_heads + self.num_key_value_heads :]
        q, k = self._apply_rope(cos_sin, q, k)
        # Upstream: flash_attn_func(q, k, v, causal) on CUDA when head_dim % 8 == 0, else this SDPA
        # call. Always SDPA here (no flash-attn in this environment); see the module docstring.
        y = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), is_causal=self.causal
        ).permute(0, 2, 1, 3)
        return self.o_proj(y.reshape(b, n, self.output_size))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float) -> None:
        super().__init__()
        intermediate = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, intermediate * 2, bias=False)
        self.down_proj = CastedLinear(intermediate, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
