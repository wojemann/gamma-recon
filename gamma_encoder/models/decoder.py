"""Per-token linear decoder: latent (d_model) -> raw waveform patch (L)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearPatchDecoder(nn.Module):
    """Linear projection from d_model to patch_samples, applied per token.

    Used to reconstruct raw waveform patches from per-(channel, patch)
    latents. The downstream caller reshapes (B, C, n, L) into (B, C, T)
    via :func:`gamma_encoder.tokenizers.base.unpatchify`.
    """

    def __init__(self, d_model: int, patch_samples: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, patch_samples)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, C, n, d_model) -> (B, C, n, patch_samples)."""
        return self.proj(tokens)
