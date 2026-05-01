"""Small transformer encoder for the overfit-sweep harness.

Stock ``nn.TransformerEncoderLayer`` blocks with sinusoidal positional
embeddings on the (channel * patch) sequence axis. BaRISTA uses RoPE +
RMSNorm + GatedMLP; we substitute stock components here so the pipeline
comes alive without a from-scratch attention port. Equivalent capacity
on tiny data; the swap to a faithful BaRISTA stack is a follow-up.

Input/output: (B, S, d_model) where S = n_channels * n_patches.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding."""
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (max_len, d_model)


class SmallTransformerEncoder(nn.Module):
    """Lightweight transformer encoder (defaults: d=32, 6 layers, 2 heads)."""

    def __init__(
        self,
        d_model: int = 32,
        n_layers: int = 6,
        n_heads: int = 2,
        ff_mult: int = 4,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.register_buffer("pe", _sinusoidal_pe(max_seq_len, d_model), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, d_model) -> (B, S, d_model)."""
        if x.dim() != 3:
            raise ValueError(f"expected (B, S, d), got {tuple(x.shape)}")
        S = x.size(1)
        if S > self.pe.size(0):
            raise ValueError(f"seq len {S} exceeds PE capacity {self.pe.size(0)}")
        x = x + self.pe[:S].unsqueeze(0)
        return self.encoder(x)
