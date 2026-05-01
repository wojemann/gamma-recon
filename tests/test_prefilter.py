"""Tests for the pre-filtering evaluation path.

The key claim being tested: pre-filtering on the long signal then slicing
gives substantially better band-resolved NMSE on short segments than
filtering each short segment independently.
"""

from __future__ import annotations

import numpy as np
import pytest

from gamma_eval.evaluator import ReconstructionEvaluator
from gamma_eval.metrics.prefilter import (
    PreFilteredSignals,
    evaluate_prefiltered_segment,
    prefilter_signal,
)
from gamma_eval.metrics.reconstruction import (
    DEFAULT_BANDS,
    band_resolved_nmse,
)
from gamma_eval.synthetic.signals import SignalConfig, generate_signal


FS = 2048.0


class TestPreFilteredSignals:
    def test_construction_has_expected_shape(self):
        signal = generate_signal(
            SignalConfig(n_samples=int(10 * FS), n_channels=4, fs=FS, seed=0)
        )
        pre = prefilter_signal(signal, FS)
        assert pre.signal.shape == signal.shape
        for name in DEFAULT_BANDS:
            assert name in pre.band_filtered
            assert pre.band_filtered[name].shape == signal.shape

    def test_slice_segment_is_a_view(self):
        signal = generate_signal(
            SignalConfig(n_samples=int(10 * FS), n_channels=2, fs=FS, seed=1)
        )
        pre = prefilter_signal(signal, FS)
        sliced = pre.slice_segment(0, 1024)
        # Modifying the slice should reflect in the parent (i.e., it's a view).
        # We don't actually want to verify mutation, but we want to verify
        # the shape is right and the content matches.
        assert sliced.signal.shape == (2, 1024)
        np.testing.assert_array_equal(sliced.signal, pre.signal[..., :1024])

    def test_custom_bands_propagated(self):
        signal = generate_signal(
            SignalConfig(n_samples=int(5 * FS), n_channels=2, fs=FS, seed=2)
        )
        custom = {"my_band": (10.0, 30.0)}
        pre = prefilter_signal(signal, FS, bands=custom)
        assert set(pre.band_filtered.keys()) == {"my_band"}
        assert pre.bands == custom


class TestEvaluatePrefilteredSegment:
    def test_perfect_reconstruction_is_zero_in_every_band(self):
        signal = generate_signal(
            SignalConfig(n_samples=int(10 * FS), n_channels=4, fs=FS, seed=10)
        )
        pre = prefilter_signal(signal, FS)
        # Perfect reconstruction: same signal, same pre-filter.
        report = evaluate_prefiltered_segment(pre, pre)
        for band, vals in report.band_nmse.items():
            np.testing.assert_allclose(
                vals, 0.0, atol=1e-8, err_msg=f"Band {band} not zero"
            )
        np.testing.assert_allclose(report.nmse_total, 0.0, atol=1e-8)

    def test_slicing_with_start_stop(self):
        signal = generate_signal(
            SignalConfig(n_samples=int(10 * FS), n_channels=4, fs=FS, seed=11)
        )
        pre = prefilter_signal(signal, FS)
        seg_len = int(3 * FS)
        report = evaluate_prefiltered_segment(pre, pre, start=0, stop=seg_len)
        assert report.nmse_total.shape == (4,)

    def test_band_mismatch_raises(self):
        signal = generate_signal(
            SignalConfig(n_samples=int(5 * FS), n_channels=2, fs=FS, seed=12)
        )
        pre_a = prefilter_signal(signal, FS, bands={"a": (1.0, 8.0)})
        pre_b = prefilter_signal(signal, FS, bands={"b": (1.0, 8.0)})
        with pytest.raises(ValueError, match="same band definitions"):
            evaluate_prefiltered_segment(pre_a, pre_b)

    def test_shape_mismatch_raises(self):
        s1 = generate_signal(
            SignalConfig(n_samples=int(5 * FS), n_channels=2, fs=FS, seed=13)
        )
        s2 = generate_signal(
            SignalConfig(n_samples=int(5 * FS), n_channels=3, fs=FS, seed=14)
        )
        pre1 = prefilter_signal(s1, FS)
        pre2 = prefilter_signal(s2, FS)
        with pytest.raises(ValueError, match="Shape mismatch"):
            evaluate_prefiltered_segment(pre1, pre2)

    def test_unknown_gamma_band_name_raises(self):
        signal = generate_signal(
            SignalConfig(n_samples=int(5 * FS), n_channels=2, fs=FS, seed=15)
        )
        pre = prefilter_signal(signal, FS)
        with pytest.raises(ValueError, match="not in pre-filtered bands"):
            evaluate_prefiltered_segment(pre, pre, gamma_band_name="nonexistent")


class TestPrefilterReducesEdgeArtifacts:
    """The whole point of pre-filtering: edge artifacts on short segments.

    If we filter the long signal once then slice into 3-second segments,
    delta-theta NMSE on a perfect-reconstruction case should be near zero.
    If we filter each 3-second segment independently, delta-theta NMSE on
    the same case will also be zero (since identical inputs filter to
    identical outputs), but that's not the right test — the test is whether
    delta-theta NMSE is meaningful on a near-perfect-but-imperfect
    reconstruction.

    The realistic scenario: prediction is the true signal plus a small
    amount of broadband noise. With per-segment filtering, the filter
    transient on each segment dominates the delta-theta NMSE. With
    pre-filtering, the transient is localized to the start/end of the
    SESSION, and the middle segments give clean delta-theta NMSE.
    """

    def test_prefilter_gives_better_delta_theta_on_short_segments(self):
        rng = np.random.default_rng(20)
        # Long signal: 30 seconds.
        signal = generate_signal(
            SignalConfig(
                n_samples=int(30 * FS), n_channels=4, fs=FS,
                aperiodic_exponent=1.5, seed=20,
            )
        )
        # Add a small amount of broadband noise to simulate imperfect prediction.
        noise = rng.standard_normal(signal.shape) * 0.1 * signal.std()
        prediction = signal + noise

        # Pre-filter both, then evaluate per 3-second segment.
        pre_true = prefilter_signal(signal, FS)
        pre_pred = prefilter_signal(prediction, FS)
        seg_len = int(3 * FS)
        # Use middle segments to avoid the very edges.
        n_segments = signal.shape[1] // seg_len
        prefilter_dt_nmse = []
        per_segment_dt_nmse = []
        for s in range(2, n_segments - 2):
            start = s * seg_len
            stop = start + seg_len
            # Pre-filtered path.
            report_pre = evaluate_prefiltered_segment(
                pre_true, pre_pred, start=start, stop=stop
            )
            prefilter_dt_nmse.append(np.mean(report_pre.band_nmse["delta_theta"]))
            # Per-segment path: filter the slice in isolation.
            results = band_resolved_nmse(
                signal[:, start:stop], prediction[:, start:stop], FS
            )
            per_segment_dt_nmse.append(np.mean(results["delta_theta"]))

        prefilter_mean = np.mean(prefilter_dt_nmse)
        per_segment_mean = np.mean(per_segment_dt_nmse)

        # The pre-filtered delta-theta NMSE should be substantially smaller
        # than the per-segment one. On a near-perfect reconstruction with
        # 0.1*std noise, true delta-theta NMSE is bounded above by
        # ~(0.1)^2 / (delta-theta variance fraction), which is small.
        # Per-segment filtering inflates this dramatically.
        assert prefilter_mean < per_segment_mean, (
            f"Expected pre-filtered NMSE ({prefilter_mean:.3f}) to be lower "
            f"than per-segment ({per_segment_mean:.3f})"
        )
        # Stronger claim: pre-filtered should be at least 2x better.
        assert prefilter_mean * 2 < per_segment_mean, (
            f"Pre-filter improvement too small: "
            f"{per_segment_mean:.3f} / {prefilter_mean:.3f} = "
            f"{per_segment_mean / prefilter_mean:.2f}x"
        )


class TestEvaluatorPrefilteredMode:
    def test_accumulate_segment_works(self):
        signal = generate_signal(
            SignalConfig(n_samples=int(10 * FS), n_channels=4, fs=FS, seed=30)
        )
        pre = prefilter_signal(signal, FS)
        evaluator = ReconstructionEvaluator(fs=FS)
        seg_len = int(3 * FS)
        for s in range(3):
            evaluator.accumulate_segment(pre, pre, s * seg_len, (s + 1) * seg_len)
        summary = evaluator.summarize()
        assert summary["n_segments"] == 3
        assert summary["nmse_total_mean"] < 1e-6

    def test_accumulate_segment_missing_band_raises(self):
        signal = generate_signal(
            SignalConfig(n_samples=int(5 * FS), n_channels=2, fs=FS, seed=31)
        )
        pre = prefilter_signal(signal, FS, bands={"only_band": (1.0, 8.0)})
        # Evaluator wants the default bands; pre-filtered only has one.
        evaluator = ReconstructionEvaluator(fs=FS)
        with pytest.raises(ValueError, match="missing required bands"):
            evaluator.accumulate_segment(pre, pre, 0, 1024)

    def test_mixed_modes_produce_combined_summary(self):
        signal = generate_signal(
            SignalConfig(n_samples=int(10 * FS), n_channels=4, fs=FS, seed=32)
        )
        pre = prefilter_signal(signal, FS)
        evaluator = ReconstructionEvaluator(fs=FS)
        # Pre-filtered segment.
        evaluator.accumulate_segment(pre, pre, 0, int(3 * FS))
        # Raw segment.
        seg = signal[:, : int(3 * FS)]
        evaluator.accumulate(seg, seg)
        summary = evaluator.summarize()
        assert summary["n_segments"] == 2
