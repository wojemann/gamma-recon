"""Complex-STFT tokenizer (phase-preserving counterpart to STFT magnitude).

Compute STFT(patch), stack real and imaginary parts along the frequency
axis (so n_freqs becomes 2*n_freqs), flatten with the time-frame axis,
and project to d_model. Preserves phase information that the magnitude
variant discards. Pairs with ``stft_magnitude.py`` to test "does phase
matter at the tokenizer step?".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gamma_encoder.tokenizers.base import Tokenizer


def _stft_complex_per_patch(
    patches: torch.Tensor, n_fft: int, hop: int, win: int
) -> torch.Tensor:
    """Complex STFT, real+imag stacked. -> (B, C, n, 2*n_freqs, n_frames)."""
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
    real = spec.real
    imag = spec.imag
    stacked = torch.cat([real, imag], dim=-2)
    n_feat, n_frames = stacked.shape[-2], stacked.shape[-1]
    return stacked.reshape(B, C, n, n_feat, n_frames)


class ComplexSTFTTokenizer(Tokenizer):
    """Per-patch complex STFT (real+imag stacked) -> Linear -> d_model."""

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
            spec = _stft_complex_per_patch(probe, self.n_fft, self.hop_length, self.win_length)
            n_feat, n_frames = spec.shape[-2], spec.shape[-1]
        self._n_feat = n_feat
        self._n_frames = n_frames
        self.proj = nn.Linear(n_feat * n_frames, self.d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 4:
            raise ValueError(f"expected (B, C, n, L), got {tuple(patches.shape)}")
        if patches.shape[-1] != self.patch_samples:
            raise ValueError(
                f"patch length {patches.shape[-1]} != configured {self.patch_samples}"
            )
        spec = _stft_complex_per_patch(patches, self.n_fft, self.hop_length, self.win_length)
        flat = spec.flatten(start_dim=-2)
        return self.proj(flat)
