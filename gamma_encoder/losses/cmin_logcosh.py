"""Circular-minimum log-cosh loss (BrainState / Johnson et al. 2024).

Time-shift-invariant variant of log-cosh. For a target y and prediction
y_hat both of length T along the last axis, we compute log-cosh at
every circular roll of y_hat by k samples (k=0..T-1) and return the
minimum over k. This makes the loss insensitive to a global temporal
offset between prediction and target — useful for asymmetric
encoder-decoder forecasting setups where the model's reconstruction
may be slightly shifted in time.

We compute the per-sample log-cosh-at-roll-k via the FFT
cross-correlation trick: for each k, the mean-log-cosh is roughly
proportional to a function of the cross-correlation, but log-cosh
itself is nonlinear so we compute it explicitly via a single
``torch.roll`` per k. With T circular shifts that's O(T^2) per sample,
which is fine at L=512 (the patch size) and still tractable at the
3-sec-segment scale (T=6144) for one forward pass — but expensive in
training. We expose ``max_shift`` to limit the search radius.

The shift axis is the **last** axis (time). Roll is circular.
"""

from __future__ import annotations

from typing import Optional

import torch

from gamma_encoder.losses.base import ReconstructionLoss


class CMinLogCoshLoss(ReconstructionLoss):
    """Minimum-over-circular-shifts of mean log-cosh residual.

    Parameters
    ----------
    max_shift : Optional[int]
        Search shifts in [-max_shift, +max_shift]. ``None`` searches
        all T shifts (full circular minimum, expensive).
    """

    def __init__(self, max_shift: Optional[int] = 32) -> None:
        super().__init__()
        self.max_shift = max_shift

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        T = pred.shape[-1]
        if self.max_shift is None:
            shifts = range(0, T)
        else:
            r = min(int(self.max_shift), T - 1)
            shifts = list(range(-r, r + 1))

        best: Optional[torch.Tensor] = None
        for k in shifts:
            rolled = torch.roll(pred, shifts=k, dims=-1)
            diff = rolled - target
            # log(cosh(x)) is numerically stable as |x| + softplus(-2|x|) - log(2).
            ax = diff.abs()
            val = (ax + torch.nn.functional.softplus(-2.0 * ax) - torch.log(torch.tensor(2.0, device=ax.device, dtype=ax.dtype))).mean()
            if best is None or val < best:
                best = val
        assert best is not None
        return best
