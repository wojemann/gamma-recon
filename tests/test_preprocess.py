"""Tests for gamma_encoder.data.preprocess.

Synthetic-signal first (per CLAUDE.md). Real BrainTreebank data is
exercised in a separate integration test once the loader is wired.
"""

from __future__ import annotations

import numpy as np
import pytest

from gamma_encoder.data.preprocess import (
    DEFAULT_FS_HZ,
    apply_laplacian_reref,
    build_laplacian_neighbors,
    notch_filter,
    parse_electrode_name,
    segment_signal,
    zscore_segment,
)


# ---------------------------------------------------------------------------
# Notch filter
# ---------------------------------------------------------------------------


def _power_at(sig: np.ndarray, fs: float, f: float, bw: float = 1.0) -> float:
    """Total power within +/- bw Hz of f, via rfft on the last axis."""
    spec = np.fft.rfft(sig, axis=-1)
    freqs = np.fft.rfftfreq(sig.shape[-1], d=1.0 / fs)
    mask = (freqs >= f - bw) & (freqs <= f + bw)
    return float(np.sum(np.abs(spec[..., mask]) ** 2))


def test_notch_attenuates_powerline_tone():
    """Inject a 60 Hz tone in pink noise; notch filter should suppress it."""
    rng = np.random.default_rng(0)
    fs = DEFAULT_FS_HZ
    n_samples = int(fs * 4)
    t = np.arange(n_samples) / fs
    base = rng.standard_normal(n_samples)
    tone = 5.0 * np.sin(2 * np.pi * 60.0 * t)
    sig = (base + tone)[None, :]  # (1, n_samples)

    p_before = _power_at(sig, fs, 60.0)
    filtered = notch_filter(sig, fs=fs)
    p_after = _power_at(filtered, fs, 60.0)

    # Expect at least 50x reduction at 60 Hz. (Q=30 IIR notch via lfilter
    # delivers ~90x in practice; allow some headroom for transient
    # behavior at the start of the signal.)
    assert p_after < 0.02 * p_before, (
        f"notch failed to suppress 60Hz: before={p_before:.2e}, after={p_after:.2e}"
    )


def test_notch_preserves_off_band_content():
    """Power at 75 Hz (well off any notch, far from harmonics) should be largely untouched."""
    rng = np.random.default_rng(1)
    fs = DEFAULT_FS_HZ
    n_samples = int(fs * 4)
    t = np.arange(n_samples) / fs
    sig = (rng.standard_normal(n_samples) + 3.0 * np.sin(2 * np.pi * 75.0 * t))[None, :]

    p_before = _power_at(sig, fs, 75.0, bw=0.5)
    p_after = _power_at(notch_filter(sig, fs=fs), fs, 75.0, bw=0.5)

    # Allow some loss from the wing of the 60/120 notch but require >70% preserved.
    assert p_after > 0.7 * p_before, (
        f"75Hz attenuated too much: before={p_before:.2e}, after={p_after:.2e}"
    )


def test_notch_rejects_freq_above_nyquist():
    sig = np.zeros((1, 1024))
    with pytest.raises(ValueError):
        notch_filter(sig, fs=2048.0, freqs=(2050.0,))


# ---------------------------------------------------------------------------
# Electrode name parsing + neighbor building
# ---------------------------------------------------------------------------


def test_parse_electrode_name_basic():
    assert parse_electrode_name("LT3a1") == ("LT3a", 1)
    assert parse_electrode_name("LT3a12") == ("LT3a", 12)
    assert parse_electrode_name("RT2aA1") == ("RT2aA", 1)
    assert parse_electrode_name("DC10") == ("DC", 10)


def test_parse_electrode_name_rejects_no_digits():
    with pytest.raises(ValueError):
        parse_electrode_name("REF")


def test_parse_electrode_name_strips_markers():
    """BaRISTA convention: ``*``, ``#``, ``_`` are noise markers and must
    be stripped before parsing so corrupted/marked names parse to the
    same (stem, num) as their clean counterparts.
    """
    assert parse_electrode_name("RT2aA1#") == ("RT2aA", 1)
    assert parse_electrode_name("LT3a*2") == ("LT3a", 2)
    # Underscores are stripped, but digits embedded in the stem (e.g.
    # the ``3`` in ``LT3a``) are part of the stem itself.
    assert parse_electrode_name("LT3_a_3") == ("LT3a", 3)
    assert parse_electrode_name("DC10*#_") == ("DC", 10)


def test_build_laplacian_neighbors_handles_marked_names():
    """Mixing marked and unmarked names along the same lead must still
    produce the right neighbor pairs after sanitization."""
    elecs = ["LT3a1", "LT3a2#", "LT3a3", "LT3a4*", "LT3a5"]
    nbrs = build_laplacian_neighbors(elecs)
    # All three middle contacts should be eligible; markers must not
    # split the lead into two stems.
    assert set(nbrs.keys()) == {"LT3a2#", "LT3a3", "LT3a4*"}
    assert nbrs["LT3a3"] == ("LT3a2#", "LT3a4*")


def test_build_laplacian_neighbors_simple_lead():
    """A clean 5-contact lead. Only contacts 2,3,4 should have both neighbors."""
    elecs = ["LT3a1", "LT3a2", "LT3a3", "LT3a4", "LT3a5"]
    nbrs = build_laplacian_neighbors(elecs)
    assert set(nbrs.keys()) == {"LT3a2", "LT3a3", "LT3a4"}
    assert nbrs["LT3a2"] == ("LT3a1", "LT3a3")
    assert nbrs["LT3a3"] == ("LT3a2", "LT3a4")
    assert nbrs["LT3a4"] == ("LT3a3", "LT3a5")


def test_build_laplacian_neighbors_excludes_corrupted():
    """A corrupted neighbor should knock out the dependent center channel."""
    elecs = ["LT3a1", "LT3a2", "LT3a3", "LT3a4", "LT3a5"]
    # LT3a3 corrupted -> LT3a2 and LT3a4 lose a neighbor; LT3a3 itself is also out.
    nbrs = build_laplacian_neighbors(elecs, excluded={"LT3a3"})
    assert "LT3a3" not in nbrs
    assert "LT3a2" not in nbrs
    assert "LT3a4" not in nbrs


def test_build_laplacian_neighbors_separate_stems_dont_mix():
    """Channels with different stems are independent leads."""
    elecs = ["LT3a1", "LT3a2", "LT3a3", "RT1c1", "RT1c2", "RT1c3"]
    nbrs = build_laplacian_neighbors(elecs)
    assert nbrs["LT3a2"] == ("LT3a1", "LT3a3")
    assert nbrs["RT1c2"] == ("RT1c1", "RT1c3")
    # No cross-stem confusion.
    assert "LT3a1" not in nbrs and "RT1c1" not in nbrs


# ---------------------------------------------------------------------------
# Laplacian reref math
# ---------------------------------------------------------------------------


def test_laplacian_removes_common_mode():
    """Common-mode noise across the lead should be removed by Laplacian.

    Setup: 3 contacts on one lead share an additive common-mode signal,
    plus per-channel independent noise. After reref, the common mode
    cancels (mean of two equals the center for a constant), leaving
    only differences in the per-channel components.
    """
    rng = np.random.default_rng(42)
    n = 2048
    common = 5.0 * np.sin(2 * np.pi * 7.0 * np.arange(n) / 2048)
    x1 = rng.standard_normal(n) * 0.1
    x2 = rng.standard_normal(n) * 0.1
    x3 = rng.standard_normal(n) * 0.1
    data = np.stack([common + x1, common + x2, common + x3])
    elecs = ["LT3a1", "LT3a2", "LT3a3"]
    nbrs = build_laplacian_neighbors(elecs)
    reref, kept = apply_laplacian_reref(data, elecs, nbrs)

    assert kept == ["LT3a2"]
    # x2 - 0.5*(x1 + x3) should not contain 7Hz oscillation.
    spec = np.fft.rfft(reref[0])
    freqs = np.fft.rfftfreq(n, d=1.0 / 2048)
    bin_7 = np.argmin(np.abs(freqs - 7.0))
    common_amp = np.abs(np.fft.rfft(common))[bin_7]
    reref_amp = np.abs(spec)[bin_7]
    assert reref_amp < 0.05 * common_amp, (
        f"common-mode not removed: {reref_amp:.2e} vs {common_amp:.2e}"
    )


def test_laplacian_reref_shape_and_dtype():
    rng = np.random.default_rng(0)
    elecs = ["LT3a1", "LT3a2", "LT3a3", "LT3a4"]
    data = rng.standard_normal((4, 100)).astype(np.float64)
    nbrs = build_laplacian_neighbors(elecs)
    reref, kept = apply_laplacian_reref(data, elecs, nbrs)
    assert reref.shape == (2, 100)
    assert kept == ["LT3a2", "LT3a3"]
    assert reref.dtype == data.dtype


def test_laplacian_reref_rejects_mismatched_names():
    data = np.zeros((3, 100))
    with pytest.raises(ValueError):
        apply_laplacian_reref(data, ["LT3a1", "LT3a2"], {})


# ---------------------------------------------------------------------------
# Segmentation + z-score
# ---------------------------------------------------------------------------


def test_segment_signal_nonoverlapping():
    data = np.arange(20).reshape(2, 10)  # 2 chans, 10 samples
    seg = segment_signal(data, segment_samples=4)
    # n_seg = 1 + (10-4)//4 = 2
    assert seg.shape == (2, 2, 4)
    np.testing.assert_array_equal(seg[0, 0], np.arange(4))
    np.testing.assert_array_equal(seg[1, 0], np.arange(4) + 4)


def test_segment_signal_too_short_returns_empty():
    data = np.zeros((3, 10))
    seg = segment_signal(data, segment_samples=20)
    assert seg.shape == (0, 3, 20)


def test_zscore_segment_per_row():
    rng = np.random.default_rng(0)
    seg = rng.standard_normal((4, 8, 2048)) * 5.0 + 2.0
    z = zscore_segment(seg)
    assert z.shape == seg.shape
    np.testing.assert_allclose(z.mean(axis=-1), 0.0, atol=1e-8)
    np.testing.assert_allclose(z.std(axis=-1), 1.0, atol=1e-6)


def test_zscore_segment_handles_flat_channel():
    """A flat (zero-variance) channel should not produce NaN."""
    seg = np.zeros((1, 1, 100))
    z = zscore_segment(seg)
    assert not np.any(np.isnan(z))
    assert np.allclose(z, 0.0)
