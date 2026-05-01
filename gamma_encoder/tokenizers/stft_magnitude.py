"""STFT-magnitude tokenizer (BrainBERT-style).

Compute |STFT(patch)| over each patch's time axis, flatten the
(n_freqs, n_frames) grid, and project to d_model with a learned
``Linear``. Phase is discarded — that's the deliberate property of
this tokenizer; a phase-preserving counterpart lives in
``complex_stft.py``.

STFT settings default to a window short enough to give multiple frames
per patch even at L=512. Defaults: ``n_fft=64``, ``hop=16``,
``win=64`` -> at L=512, ``n_frames=L//hop+1 = 33`` and
``n_freqs=n_fft//2+1 = 33``. These are knobs we can tune; current
defaults err on the side of high time resolution, low frequency
resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gamma_encoder.tokenizers.base import Tokenizer


def _stft_mag_per_patch(
    patches: torch.Tensor, n_fft: int, hop: int, win: int
) -> torch.Tensor:
    """STFT magnitude over per-patch time axis.

    patches: (B, C, n, L) -> (B, C, n, n_freqs, n_frames).
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
    mag = spec.abs()
    n_freqs, n_frames = mag.shape[-2], mag.shape[-1]
    return mag.reshape(B, C, n, n_freqs, n_frames)


class STFTMagnitudeTokenizer(Tokenizer):
    """Per-patch STFT magnitude -> Linear -> d_model."""

    def __init__(
        self,
        d_model: int,
        patch_samples: int,
        n_fft: int = 64,
        hop_length: int = 16,
        win_length: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.patch_samples = int(patch_samples)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)

        with torch.no_grad():
            probe = torch.zeros(1, 1, 1, self.patch_samples)
            spec = _stft_mag_per_patch(probe, self.n_fft, self.hop_length, self.win_length)
            n_freqs, n_frames = spec.shape[-2], spec.shape[-1]
        self._n_freqs = n_freqs
        self._n_frames = n_frames
        self.proj = nn.Linear(n_freqs * n_frames, self.d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 4:
            raise ValueError(f"expected (B, C, n, L), got {tuple(patches.shape)}")
        if patches.shape[-1] != self.patch_samples:
            raise ValueError(
                f"patch length {patches.shape[-1]} != configured {self.patch_samples}"
            )
        mag = _stft_mag_per_patch(patches, self.n_fft, self.hop_length, self.win_length)
        flat = mag.flatten(start_dim=-2)
        return self.proj(flat)
