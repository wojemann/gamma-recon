"""Content-aware reweighted L1 (BrainBERT recipe).

L1 with per-sample weights proportional to ``|target|``. Upweights
high-amplitude regions, which is exactly where gamma bursts live in a
1/f-dominated signal — those bursts ride on top of the slow envelope
but show up as locally-large absolute deviations once the slow envelope
is z-scored away. So this is a simple way to focus the loss on the
parts of the signal we care about reconstructing faithfully.

Weights are normalized per-batch so the loss scale is invariant to the
overall amplitude of the target.
"""

from __future__ import annotations

import torch

from gamma_encoder.losses.base import ReconstructionLoss


class ContentAwareL1Loss(ReconstructionLoss):
    """Weighted L1 with weight proportional to |target|.

    Parameters
    ----------
    weight_floor : float
        Added to the weight before normalization, so quiet regions are
        not totally ignored. Defaults to 0.1 of the mean target
        magnitude (a value of 0.1 in normalized units).
    """

    def __init__(self, weight_floor: float = 0.1) -> None:
        super().__init__()
        self.weight_floor = float(weight_floor)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        mag = target.abs()
        # Normalize magnitudes by their own mean so weight_floor is on the
        # same scale regardless of input amplitude. Add the floor and
        # renormalize so the weights average to 1 (i.e. when pred=target,
        # contribution per sample is comparable to plain L1 on that batch).
        mean_mag = mag.mean().clamp_min(1e-8)
        w = mag / mean_mag + self.weight_floor
        w = w / w.mean().clamp_min(1e-8)
        return (w * (pred - target).abs()).mean()
