"""Dilated-CNN tokenizer (BaRISTA-style, small variant).

Stack of 1D dilated convolutions consuming a single-channel patch of
length L, followed by a **learned** linear projection from the full
conv-stack output to ``d_model``. Dilations grow exponentially
(1, 2, 4, ..., 2^(n_blocks-1)) so the receptive field reaches a
substantial fraction of the patch.

The time-to-hidden head is a single ``Linear(hidden*L, d_model)``,
matching the spirit of BaRISTA's ``temporal_pooler`` MLP: every
(conv-channel, time-step) coefficient contributes a learned weight to
every output unit. Earlier versions of this file used
``AdaptiveAvgPool1d``; that is a free low-pass on the conv output and
directly works against the gamma-fidelity thesis, so we don't use it.

This is a faithful-spirit reimplementation, NOT a port of BaRISTA's
``TSEncoder2D`` — we keep it small and 1D for clarity. The architectural
properties that matter for the gamma-fidelity experiments are preserved:
repeated conv smoothing of high-frequency content within the conv stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gamma_encoder.tokenizers.base import Tokenizer


class _DilatedBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class DilatedCNNTokenizer(Tokenizer):
    """Per-(channel, patch) tokenizer: (B, C, n, L) -> (B, C, n, d_model)."""

    def __init__(
        self,
        d_model: int,
        patch_samples: int,
        n_blocks: int = 5,
        hidden: int = 16,
        kernel: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.patch_samples = int(patch_samples)
        self.n_blocks = int(n_blocks)

        blocks = []
        in_ch = 1
        for i in range(n_blocks):
            blocks.append(_DilatedBlock(in_ch, hidden, kernel, dilation=2 ** i))
            in_ch = hidden
        self.blocks = nn.Sequential(*blocks)
        # Learned projection from the full (hidden, L) conv-stack output to
        # one d_model-sized token. No averaging — keeps high-frequency
        # information available to the head instead of low-passing it away.
        self.head = nn.Linear(hidden * self.patch_samples, d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 4:
            raise ValueError(f"expected (B, C, n, L), got {tuple(patches.shape)}")
        B, C, n, L = patches.shape
        if L != self.patch_samples:
            raise ValueError(f"patch length {L} != configured {self.patch_samples}")
        x = patches.reshape(B * C * n, 1, L)
        x = self.blocks(x)              # (B*C*n, hidden, L)
        x = x.flatten(start_dim=1)      # (B*C*n, hidden * L)
        x = self.head(x)                # (B*C*n, d_model)
        return x.reshape(B, C, n, self.d_model)
