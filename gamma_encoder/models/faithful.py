"""Faithful BaRISTA encoder block: RoPE + RMSNorm + SwiGLU MLP.

Mirrors ``barista/models/transformer.py``:
- ``RMSNorm``: variance in fp32, eps=1e-6, learned scale.
- ``RotaryEmbedding`` + ``apply_rotary_pos_emb``: sliced-halves convention
  (LLaMA/HF), base=10000, applied to Q and K only, broadcast across heads.
- ``RotarySelfAttention``: vanilla MHA with RoPE on Q/K.
- ``GatedTransformerMLP``: SwiGLU, mlp_ratio=4.
- ``FaithfulEncoderLayer``: pre-norm, post-attn residual + dropout, mlp
  residual (no extra dropout on the residual path itself; mlp has internal
  dropouts).
- ``FaithfulTransformerEncoder``: stack of N layers + final RMSNorm.

Inputs/outputs are ``(B, S, d_model)``, matching ``SmallTransformerEncoder``.
Position ids are taken as ``arange(S)`` by default; callers that want
"all channels at the same temporal patch share a position id" should pass
explicit ``position_ids`` (BaRISTA tokenizer convention).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d_hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (self.weight * x32).to(in_dtype)


# ---------------------------------------------------------------------------
# RoPE — sliced-halves (LLaMA/HF) convention
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    def __init__(self, d_head: int, base: float = 10000.0, max_seq_len: int = 4096) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(f"d_head must be even for RoPE, got {d_head}")
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)        # (S, d/2)
        emb = torch.cat((freqs, freqs), dim=-1)                  # (S, d)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device, dtype):
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.size(-1)
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q, k: (B, S, H, D). cos/sin: (S, D). Broadcasts across batch and heads."""
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Rotary self-attention
# ---------------------------------------------------------------------------


class RotarySelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 max_seq_len: int = 4096) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.d_head, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head)
        cos, sin = self.rope(S, x.device, x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # (B, H, S, D)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------


class GatedTransformerMLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        d_ff = mlp_ratio * d_model
        self.gate_proj = nn.Linear(d_model, d_ff, bias=True)
        self.up_proj = nn.Linear(d_model, d_ff, bias=True)
        self.down_proj = nn.Linear(d_ff, d_model, bias=True)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        gated = self.drop1(gated)
        return self.drop2(self.down_proj(gated))


# ---------------------------------------------------------------------------
# Encoder layer + stack
# ---------------------------------------------------------------------------


class FaithfulEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.0, norm_eps: float = 1e-6,
                 max_seq_len: int = 4096) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.attn = RotarySelfAttention(d_model, n_heads, dropout=dropout,
                                         max_seq_len=max_seq_len)
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.mlp = GatedTransformerMLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_drop(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class FaithfulTransformerEncoder(nn.Module):
    """BaRISTA-faithful transformer encoder. (B, S, d_model) -> (B, S, d_model)."""

    def __init__(
        self,
        d_model: int = 32,
        n_layers: int = 6,
        n_heads: int = 2,
        mlp_ratio: int = 4,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            FaithfulEncoderLayer(
                d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio,
                dropout=dropout, norm_eps=norm_eps, max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, S, d), got {tuple(x.shape)}")
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
