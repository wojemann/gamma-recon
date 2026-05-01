"""MSE reconstruction loss (BaRISTA baseline)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gamma_encoder.losses.base import ReconstructionLoss


class MSELoss(ReconstructionLoss):
    """Plain mean-squared error on the waveform."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        return F.mse_loss(pred, target)
