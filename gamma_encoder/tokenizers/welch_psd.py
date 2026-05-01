"""Welch-PSD tokenizer (Merk et al. 2025-style).

Per-patch Welch power spectral density, log-transformed, then linearly
projected to ``d_model``. Welch's method estimates PSD as the average
of windowed periodograms across overlapping segments — a smoother,
lower-variance spectrum estimate than a single STFT frame.

Compared to ``STFTMagnitudeTokenizer``:
- Time axis is collapsed (averaged), not preserved as frames. The token
  has no time-within-patch resolution, only frequency.
- Power (magnitude squared), not magnitude.
- Log-transformed, which compresses the wide dynamic range of a 1/f
  spectrum and is the convention in Merk's model.

Implementation: STFT -> |.|^2 -> mean over time frames -> log(.+eps) ->
Linear -> d_model. We use ``torch.stft`` for batching simplicity rather
than calling out to ``scipy.signal.welch``; the math is the same up to
window-energy normalization, which is absorbed by the learned linear
projection downstream.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gamma_encoder.tokenizers.base import Tokenizer


def _welch_logpsd_per_patch(
    patches: torch.Tensor,
    n_fft: int,
    hop: int,
    win: int,
    eps: float,
) -> torch.Tensor:
    """Welch log-PSD over per-patch time axis.

    patches: (B, C, n, L) -> (B, C, n, n_freqs).
    """
    B, C, n, L = patches.shape
    flat = patches.reshape(-1, L)
    window = torch.hann_window(win, device=patches.device, dtype=patches.dtype)
    spec = torch.stft(
        flat,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    power = spec.real ** 2 + spec.imag ** 2  # (B*C*n, n_freqs, n_frames)
    psd = power.mean(dim=-1)                  # average periodogram across frames
    log_psd = torch.log(psd + eps)
    n_freqs = log_psd.shape[-1]
    return log_psd.reshape(B, C, n, n_freqs)


class WelchPSDTokenizer(Tokenizer):
    """Per-patch Welch log-PSD -> Linear -> d_model."""

    def __init__(
        self,
        d_model: int,
        patch_samples: int,
        n_fft: int = 64,
        hop_length: int = 16,
        win_length: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.patch_samples = int(patch_samples)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.eps = float(eps)

        with torch.no_grad():
            probe = torch.zeros(1, 1, 1, self.patch_samples)
            psd = _welch_logpsd_per_patch(
                probe, self.n_fft, self.hop_length, self.win_length, self.eps
            )
            n_freqs = psd.shape[-1]
        self._n_freqs = n_freqs
        self.proj = nn.Linear(n_freqs, self.d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 4:
            raise ValueError(f"expected (B, C, n, L), got {tuple(patches.shape)}")
        if patches.shape[-1] != self.patch_samples:
            raise ValueError(
                f"patch length {patches.shape[-1]} != configured {self.patch_samples}"
            )
        log_psd = _welch_logpsd_per_patch(
            patches, self.n_fft, self.hop_length, self.win_length, self.eps
        )
        return self.proj(log_psd)
