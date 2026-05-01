"""Robust-regression baseline losses (MAE, Huber).

These exist as sanity-check baselines: any improvement over MSE in the
ablation matrix should come from spectrum awareness (Whitened MSE,
multi-res STFT, etc.), not just from down-weighting outliers. If MAE or
Huber alone closes most of the gamma-NMSE gap, that reframes the story.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gamma_encoder.losses.base import ReconstructionLoss


class MAELoss(ReconstructionLoss):
    """L1 / mean-absolute-error on the waveform."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        return F.l1_loss(pred, target)


class HuberLoss(ReconstructionLoss):
    """Huber loss (smoothL1) — quadratic near zero, linear for large residuals."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = float(delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        return F.huber_loss(pred, target, delta=self.delta)
