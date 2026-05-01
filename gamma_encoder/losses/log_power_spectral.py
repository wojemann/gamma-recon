"""Log-power spectral L1 loss.

Robust variant of the 1/f-aware family: L1 distance between log
power spectra, computed via rFFT along the time axis. Equivalent to
Whitened MSE up to log-vs-divide and L1-vs-L2, but tends to be more
numerically forgiving across the wide dynamic range of a 1/f signal.

This is the simplest dynamic-range-tame loss: take log, then L1.
"""

from __future__ import annotations

import torch

from gamma_encoder.losses.base import ReconstructionLoss


class LogPowerSpectralL1Loss(ReconstructionLoss):
    """L1 between log power spectra of pred and target.

    log(|FFT(x)|^2 + eps), L1 averaged over (batch, channels, freqs).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        T = pred.shape[-1]
        pred_p = torch.fft.rfft(pred, n=T, dim=-1).abs() ** 2
        target_p = torch.fft.rfft(target, n=T, dim=-1).abs() ** 2
        return (torch.log(pred_p + self.eps) - torch.log(target_p + self.eps)).abs().mean()
