"""Cache a stable PSD estimate for WhitenedMSELoss.

Reads ``results/overfit_batch.pt`` (16 segments x 8 channels x 6144
samples), computes a per-frequency-bin power spectral density by
averaging |rFFT|^2 across all segments and channels, and saves the
result to ``results/whitened_mse_psd.pt`` as a 1D float32 tensor of
length ``T // 2 + 1``.

The averaging gives 128 independent estimates per bin, which is enough
to keep the resulting whitened loss O(1) at random init (compared to
the 1e9-scale we got from on-the-fly per-batch PSD estimation).

For the eventual real-data run, this should be recomputed on a much
larger Subject 2 sample (e.g. all of session 1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, default=Path("results/overfit_batch.pt"))
    p.add_argument("--out", type=Path, default=Path("results/whitened_mse_psd.pt"))
    args = p.parse_args()

    payload = torch.load(args.batch, map_location="cpu", weights_only=False)
    segments: torch.Tensor = payload["segments"].float()  # (B, C, T)
    B, C, T = segments.shape
    print(f"loaded {tuple(segments.shape)} from {args.batch}")

    spec = torch.fft.rfft(segments, n=T, dim=-1)          # (B, C, n_freqs)
    power = spec.real ** 2 + spec.imag ** 2
    psd = power.mean(dim=(0, 1))                          # (n_freqs,)

    # Floor at median*1e-4 to keep 1/sqrt dynamic range bounded; raw
    # near-zero bins (DC for zero-mean signals, sometimes Nyquist) would
    # otherwise drive the whitened error to ~1e10 even though the
    # *signal* in those bins is effectively zero.
    floor = psd.median() * 1e-4
    raw_min = psd.min().item()
    psd = torch.clamp(psd, min=float(floor))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(psd, args.out)
    print(f"saved PSD shape={tuple(psd.shape)} "
          f"raw_min={raw_min:.3e} floor={float(floor):.3e} "
          f"median={psd.median().item():.3e} max={psd.max().item():.3e} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
