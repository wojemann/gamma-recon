"""EEGM2-style composite loss: time L1 + alpha * spectral L1.

From Yu et al. 2025 (EEGM2). Direct EEG-SSL precedent for combining a
time-domain robust regression term with a spectral term. Default
alpha = 0.5 matches the paper's recipe.

Spectral term is L1 on rFFT magnitudes (not log). Pairs cleanly with
the time-domain L1 term because they're at the same scale.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gamma_encoder.losses.base import ReconstructionLoss


class EEGM2Loss(ReconstructionLoss):
    """time L1 + alpha * spectral magnitude L1."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        time_term = F.l1_loss(pred, target)
        T = pred.shape[-1]
        pred_mag = torch.fft.rfft(pred, n=T, dim=-1).abs()
        target_mag = torch.fft.rfft(target, n=T, dim=-1).abs()
        spec_term = (pred_mag - target_mag).abs().mean()
        return time_term + self.alpha * spec_term
