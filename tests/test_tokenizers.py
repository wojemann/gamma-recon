"""Tests for new tokenizers.

Each tokenizer gets:
  1. shape contract;
  2. patch_samples mismatch raises;
  3. gradient flow;
  4. (where applicable) frequency-discrimination check.

Spectral tokenizers (STFT magnitude, complex STFT, wavelet packet) must
distinguish a 10 Hz tone from a 130 Hz tone — that's the whole point of
having frequency structure baked into the tokenizer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gamma_encoder.tokenizers.complex_stft import ComplexSTFTTokenizer
from gamma_encoder.tokenizers.linear import LinearTokenizer
from gamma_encoder.tokenizers.stft_magnitude import STFTMagnitudeTokenizer
from gamma_encoder.tokenizers.wavelet_packet import WaveletPacketTokenizer
from gamma_encoder.tokenizers.welch_psd import WelchPSDTokenizer


FS = 2048.0
PATCH = 512


def _tone(freq_hz: float, n: int = PATCH, fs: float = FS) -> torch.Tensor:
    """Single-channel, single-patch tone tensor of shape (1, 1, 1, n)."""
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    return torch.from_numpy(x).reshape(1, 1, 1, n)


def _shape_contract(tok, d_model: int):
    x = torch.randn(2, 3, 4, tok.patch_samples)
    out = tok(x)
    assert out.shape == (2, 3, 4, d_model)


def _grad_flows(tok):
    x = torch.randn(1, 2, 2, tok.patch_samples, requires_grad=True)
    out = tok(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def _patch_mismatch_raises(tok):
    with pytest.raises(ValueError):
        tok(torch.randn(1, 1, 1, tok.patch_samples + 1))


def _freq_discriminates(tok, threshold: float = 0.01):
    """Two pure tones at very different frequencies should yield distinct tokens."""
    torch.manual_seed(0)
    low = _tone(10.0)   # ~delta-theta band
    high = _tone(130.0) # high-gamma band
    out_low = tok(low).flatten()
    out_high = tok(high).flatten()
    dist = (out_low - out_high).norm().item() / max(out_low.norm().item(), 1e-8)
    assert dist > threshold, (
        f"{type(tok).__name__} failed frequency discrimination: relative dist {dist:.4f}"
    )


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------


def test_linear_shape():
    torch.manual_seed(0)
    _shape_contract(LinearTokenizer(d_model=16, patch_samples=PATCH), d_model=16)


def test_linear_patch_mismatch_raises():
    _patch_mismatch_raises(LinearTokenizer(d_model=16, patch_samples=PATCH))


def test_linear_grad_flows():
    _grad_flows(LinearTokenizer(d_model=16, patch_samples=PATCH))


# ---------------------------------------------------------------------------
# STFT magnitude
# ---------------------------------------------------------------------------


def test_stft_magnitude_shape():
    torch.manual_seed(0)
    _shape_contract(STFTMagnitudeTokenizer(d_model=16, patch_samples=PATCH), d_model=16)


def test_stft_magnitude_patch_mismatch_raises():
    _patch_mismatch_raises(STFTMagnitudeTokenizer(d_model=16, patch_samples=PATCH))


def test_stft_magnitude_grad_flows():
    _grad_flows(STFTMagnitudeTokenizer(d_model=16, patch_samples=PATCH))


def test_stft_magnitude_frequency_discriminates():
    torch.manual_seed(0)
    tok = STFTMagnitudeTokenizer(d_model=16, patch_samples=PATCH)
    _freq_discriminates(tok)


# ---------------------------------------------------------------------------
# Complex STFT
# ---------------------------------------------------------------------------


def test_complex_stft_shape():
    torch.manual_seed(0)
    _shape_contract(ComplexSTFTTokenizer(d_model=16, patch_samples=PATCH), d_model=16)


def test_complex_stft_patch_mismatch_raises():
    _patch_mismatch_raises(ComplexSTFTTokenizer(d_model=16, patch_samples=PATCH))


def test_complex_stft_grad_flows():
    _grad_flows(ComplexSTFTTokenizer(d_model=16, patch_samples=PATCH))


def test_complex_stft_frequency_discriminates():
    torch.manual_seed(0)
    tok = ComplexSTFTTokenizer(d_model=16, patch_samples=PATCH)
    _freq_discriminates(tok)


# ---------------------------------------------------------------------------
# Wavelet packet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_wavelet_packet_shape(depth):
    torch.manual_seed(0)
    tok = WaveletPacketTokenizer(d_model=16, patch_samples=PATCH, depth=depth)
    _shape_contract(tok, d_model=16)


def test_wavelet_packet_patch_mismatch_raises():
    _patch_mismatch_raises(WaveletPacketTokenizer(d_model=16, patch_samples=PATCH, depth=4))


def test_wavelet_packet_indivisible_raises():
    with pytest.raises(ValueError):
        WaveletPacketTokenizer(d_model=16, patch_samples=100, depth=4)  # 100 / 16 not integer


def test_wavelet_packet_grad_flows():
    _grad_flows(WaveletPacketTokenizer(d_model=16, patch_samples=PATCH, depth=4))


def test_wavelet_packet_frequency_discriminates():
    torch.manual_seed(0)
    tok = WaveletPacketTokenizer(d_model=16, patch_samples=PATCH, depth=4)
    _freq_discriminates(tok)


# ---------------------------------------------------------------------------
# Welch PSD
# ---------------------------------------------------------------------------


def test_welch_psd_shape():
    torch.manual_seed(0)
    _shape_contract(WelchPSDTokenizer(d_model=16, patch_samples=PATCH), d_model=16)


def test_welch_psd_patch_mismatch_raises():
    _patch_mismatch_raises(WelchPSDTokenizer(d_model=16, patch_samples=PATCH))


def test_welch_psd_grad_flows():
    _grad_flows(WelchPSDTokenizer(d_model=16, patch_samples=PATCH))


def test_welch_psd_frequency_discriminates():
    torch.manual_seed(0)
    tok = WelchPSDTokenizer(d_model=16, patch_samples=PATCH)
    _freq_discriminates(tok)


def test_welch_psd_time_invariant():
    """Welch averages over time frames, so a circularly-shifted version of
    the same tone should produce nearly-identical tokens."""
    torch.manual_seed(0)
    tok = WelchPSDTokenizer(d_model=16, patch_samples=PATCH)
    x = _tone(80.0)
    x_shifted = torch.roll(x, shifts=37, dims=-1)
    out_x = tok(x).flatten()
    out_s = tok(x_shifted).flatten()
    rel = (out_x - out_s).norm().item() / max(out_x.norm().item(), 1e-8)
    # Edge effects from STFT center=True padding mean it's not exact, but
    # should be much smaller than the 10-vs-130 Hz frequency-discrim gap.
    assert rel < 0.1, f"Welch PSD changed too much under time shift: rel dist {rel:.4f}"
