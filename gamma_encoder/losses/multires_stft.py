"""Multi-resolution STFT magnitude L1 loss.

Audio-domain SOTA for high-fidelity waveform reconstruction (Parallel
WaveGAN, HiFi-GAN, UnivNet, DAC). Sum of L1 distances on STFT
magnitudes across multiple (window, hop, n_fft) configurations. Each
scale captures a different time/frequency resolution; together they
constrain the model to match spectral structure across the band.

Configurations default to (n_fft, hop, win) triples scaled for our
2048 Hz iEEG signal at 250 ms patches:
    (64, 16, 64)
    (256, 64, 256)
    (1024, 256, 1024)

Last config requires the input length to be at least 1024 samples; the
3-sec segment (T=6144) and 250 ms patch (L=512) handle the smaller
configs but the largest config requires the segment-level input. We
clamp configs to those that fit the input length so the loss is safe
to apply on short patches as well.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from gamma_encoder.losses.base import ReconstructionLoss


# (n_fft, hop_length, win_length)
DEFAULT_STFT_CONFIGS: List[Tuple[int, int, int]] = [
    (64, 16, 64),
    (256, 64, 256),
    (1024, 256, 1024),
]


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    """Compute |STFT(x)| over the last dim, batching the leading dims.

    Returns (..., n_freqs, n_frames).
    """
    leading = x.shape[:-1]
    T = x.shape[-1]
    flat = x.reshape(-1, T)
    window = torch.hann_window(win, device=x.device, dtype=x.dtype)
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
    mag = spec.abs()
    return mag.reshape(*leading, mag.shape[-2], mag.shape[-1])


class MultiResolutionSTFTLoss(ReconstructionLoss):
    """Sum of STFT-magnitude L1 across multiple resolutions.

    Configs whose ``win_length`` exceeds the input length T are skipped
    silently. If all configs are skipped, raises a ValueError on first
    call.
    """

    def __init__(self, configs: List[Tuple[int, int, int]] | None = None) -> None:
        super().__init__()
        self.configs = configs if configs is not None else DEFAULT_STFT_CONFIGS

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        T = pred.shape[-1]
        usable = [c for c in self.configs if c[2] <= T]
        if not usable:
            raise ValueError(
                f"no STFT configs fit input length T={T}; configs={self.configs}"
            )
        total = pred.new_zeros(())
        for n_fft, hop, win in usable:
            m_pred = _stft_mag(pred, n_fft, hop, win)
            m_target = _stft_mag(target, n_fft, hop, win)
            total = total + (m_pred - m_target).abs().mean()
        return total / len(usable)
