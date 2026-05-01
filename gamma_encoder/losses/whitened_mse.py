"""Whitened MSE loss.

The principled spectrum-aware loss for the 1/f problem. Procedure:
   1. rFFT pred and target along the time axis.
   2. Divide both spectra by sqrt(PSD), where PSD is estimated once
      from training data (or, in the no-prior mode, from the target
      batch itself).
   3. MSE in the whitened complex spectrum.

Whitening makes every frequency bin contribute equal expected power,
so MSE in this domain weights gamma errors as heavily as low-frequency
errors. Closest waveform analog of Merk et al. 2025's 1/f-scaled
spectrogram MAE.

Two modes:
- ``psd`` is provided at construction: fixed normalizer (the
  paper-ready setup; estimate once from training distribution).
- ``psd`` is None: the loss estimates PSD from ``target`` on each
  call. Cheap, parameter-free, but introduces a batch-dependent
  rescaling. Useful for unit tests and laptop debugging.
"""

from __future__ import annotations

from typing import Optional

import torch

from gamma_encoder.losses.base import ReconstructionLoss


class WhitenedMSELoss(ReconstructionLoss):
    """MSE in spectrally-whitened domain.

    Parameters
    ----------
    psd : Optional[torch.Tensor]
        Power spectral density of shape (n_freqs,) where
        n_freqs == T // 2 + 1, matching the rFFT of length T. If None,
        PSD is estimated from the target on each forward call.
    eps : float
        Floor on PSD before sqrt to avoid division by zero in empty
        bins (e.g., the DC bin if target is zero-mean).
    """

    def __init__(self, psd: Optional[torch.Tensor] = None, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)
        if psd is not None:
            self.register_buffer("psd", psd.clone().detach().to(torch.float32))
        else:
            self.psd = None  # type: ignore[assignment]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        T = pred.shape[-1]
        pred_f = torch.fft.rfft(pred, n=T, dim=-1)
        target_f = torch.fft.rfft(target, n=T, dim=-1)

        if self.psd is not None:
            psd = self.psd
            if psd.shape[0] != pred_f.shape[-1]:
                raise ValueError(
                    f"psd length {psd.shape[0]} != n_freqs {pred_f.shape[-1]}"
                )
        else:
            # Mean power across batch/channels per frequency bin.
            psd = (target_f.abs() ** 2).mean(dim=tuple(range(target_f.dim() - 1)))

        scale = torch.rsqrt(psd + self.eps)  # 1/sqrt(psd)
        diff = (pred_f - target_f) * scale
        # |z|^2 = re^2 + im^2; mean over all axes for a scalar.
        return (diff.real ** 2 + diff.imag ** 2).mean()
