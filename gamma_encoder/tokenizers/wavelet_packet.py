"""Wavelet-packet-decomposition tokenizer (Hi-WaveTST-style).

Apply a depth-D wavelet packet decomposition to each patch, producing
2^D subbands of length L/2^D. Concatenate the subband coefficients and
project to d_model with a learned ``Linear``. Phase is preserved
(wavelet coefficients are real-valued but encode both magnitude and
sign), and the time/frequency resolution adapts across scales.

Implementation: torch-native db4 (Daubechies-4 length-8) low/high-pass
filters applied recursively via stride-2 conv1d. Each level halves the
length and doubles the number of subbands; the WPD packet decomposition
applies BOTH filters at every node, not just the lowpass branch (which
would be the standard DWT). At depth D, we get 2^D subbands.

We don't depend on `pywt` — the four filter coefficients per filter
are short and the recursive conv is a few dozen lines. This keeps the
tokenizer self-contained and trivially batched on GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gamma_encoder.tokenizers.base import Tokenizer


# Daubechies-4 ("db4") filter coefficients, length 8. Lowpass = scaling
# filter h; highpass = wavelet filter g, related by g[k] = (-1)^k h[N-1-k].
_DB4_LO = [
    -0.010597401784997278,
    0.032883011666982945,
    0.030841381835986965,
    -0.18703481171888114,
    -0.027983769416983849,
    0.63088076792959036,
    0.71484657055254153,
    0.23037781330885523,
]


def _db4_filters(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (lowpass, highpass) analysis filters as (1, 1, 8) tensors."""
    h = torch.tensor(_DB4_LO, dtype=dtype, device=device)
    n = h.shape[0]
    g = torch.empty_like(h)
    for k in range(n):
        g[k] = ((-1) ** k) * h[n - 1 - k]
    return h.view(1, 1, n), g.view(1, 1, n)


def _wpd_step(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """One WPD level: each (..., L) input -> (..., 2*N, L/2) output.

    Applies symmetric reflection padding so output length is exactly L/2
    when L is even.
    """
    *lead, L = x.shape
    flat = x.reshape(-1, 1, L)
    pad = lo.shape[-1] - 1  # filter length - 1
    # Symmetric reflect padding split before/after.
    p_l = pad // 2
    p_r = pad - p_l
    flat_p = F.pad(flat, (p_l, p_r), mode="reflect")
    a = F.conv1d(flat_p, lo, stride=2)
    d = F.conv1d(flat_p, hi, stride=2)
    # Trim to exactly L//2 along the last axis (symmetric reflect can
    # over-shoot by 1 depending on parity).
    out_len = L // 2
    a = a[..., :out_len]
    d = d[..., :out_len]
    out = torch.stack([a, d], dim=1)  # (B, 2, 1, L/2)
    out = out.reshape(*lead, 2, out_len)
    return out


def _wpd(x: torch.Tensor, depth: int) -> torch.Tensor:
    """Full WPD at given depth.

    Input shape: (..., L). Output shape: (..., 2^depth, L / 2^depth).
    """
    lo, hi = _db4_filters(x.dtype, x.device)
    if x.shape[-1] % (2 ** depth) != 0:
        raise ValueError(
            f"input length {x.shape[-1]} not divisible by 2^depth = {2**depth}"
        )
    cur = x
    for _ in range(depth):
        # cur shape: (..., S, L_cur). Apply WPD step on last axis, getting
        # (..., S, 2, L_cur/2), then merge S and 2 axes -> (..., 2*S, L_cur/2).
        if cur.dim() == x.dim():
            # First iteration: insert subband axis.
            cur = cur.unsqueeze(-2)  # (..., 1, L)
        *lead, S, L_cur = cur.shape
        flat = cur.reshape(-1, L_cur)
        step = _wpd_step(flat, lo, hi)  # (B, 2, L_cur/2)
        cur = step.reshape(*lead, S * 2, L_cur // 2)
    return cur


class WaveletPacketTokenizer(Tokenizer):
    """Per-patch db4 wavelet packet decomposition + Linear -> d_model.

    Parameters
    ----------
    depth : int
        Decomposition depth. Produces ``2^depth`` subbands, each of
        length ``patch_samples / 2^depth``. Total feature dim equals
        ``patch_samples`` (orthonormal-ish).
    """

    def __init__(self, d_model: int, patch_samples: int, depth: int = 4) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.patch_samples = int(patch_samples)
        self.depth = int(depth)
        if self.patch_samples % (2 ** self.depth) != 0:
            raise ValueError(
                f"patch_samples={self.patch_samples} not divisible by 2^depth={2**self.depth}"
            )
        # Total coeffs == patch_samples (decomposition preserves length).
        self.proj = nn.Linear(self.patch_samples, self.d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 4:
            raise ValueError(f"expected (B, C, n, L), got {tuple(patches.shape)}")
        if patches.shape[-1] != self.patch_samples:
            raise ValueError(
                f"patch length {patches.shape[-1]} != configured {self.patch_samples}"
            )
        coeffs = _wpd(patches, depth=self.depth)
        flat = coeffs.flatten(start_dim=-2)
        return self.proj(flat)
