"""Tests for ReconstructionEvaluator."""

from __future__ import annotations

import numpy as np
import pytest

from gamma_eval.evaluator import ReconstructionEvaluator
from gamma_eval.synthetic.signals import (
    SignalConfig,
    generate_signal,
    smooth_high_frequencies,
)


FS = 2048.0


class TestEvaluator:
    def test_single_batch_evaluation(self):
        signal = generate_signal(
            SignalConfig(
                n_samples=4096, n_channels=4, fs=FS,
                bursts=[(80.0, 5.0, 2.0)], seed=0,
            )
        )
        evaluator = ReconstructionEvaluator(fs=FS)
        metrics = evaluator.evaluate_batch(signal, signal)
        assert metrics["nmse_total_mean"] < 1e-6
        assert metrics["nmse_high_gamma_mean"] < 1e-6
        # Connectivity not in per-segment summary anymore.
        assert "connectivity_pearson_r" not in metrics

    def test_evaluate_batch_doesnt_pollute_state(self):
        signal = generate_signal(
            SignalConfig(n_samples=4096, n_channels=2, fs=FS, seed=1)
        )
        evaluator = ReconstructionEvaluator(fs=FS)
        evaluator.accumulate(signal, signal)
        n_before = len(evaluator._reports)
        evaluator.evaluate_batch(signal, signal)
        assert len(evaluator._reports) == n_before

    def test_accumulate_and_summarize(self):
        evaluator = ReconstructionEvaluator(fs=FS)
        for seed in range(5):
            signal = generate_signal(
                SignalConfig(n_samples=4096, n_channels=4, fs=FS, seed=seed)
            )
            evaluator.accumulate(signal, signal)
        summary = evaluator.summarize()
        assert summary["n_segments"] == 5
        assert summary["nmse_total_mean"] < 1e-6

    def test_reset(self):
        signal = generate_signal(
            SignalConfig(n_samples=4096, n_channels=2, fs=FS, seed=2)
        )
        evaluator = ReconstructionEvaluator(fs=FS)
        evaluator.accumulate(signal, signal)
        assert len(evaluator._reports) == 1
        evaluator.reset()
        assert len(evaluator._reports) == 0

    def test_3d_input_treated_as_batched(self):
        signal = np.random.RandomState(0).randn(3, 4, 2048)
        evaluator = ReconstructionEvaluator(fs=FS)
        evaluator.accumulate(signal, signal)
        assert len(evaluator._reports) == 3

    def test_2d_input_treated_as_single_segment(self):
        signal = np.random.RandomState(0).randn(4, 2048)
        evaluator = ReconstructionEvaluator(fs=FS)
        evaluator.accumulate(signal, signal)
        assert len(evaluator._reports) == 1

    def test_shape_mismatch_raises(self):
        evaluator = ReconstructionEvaluator(fs=FS)
        with pytest.raises(ValueError, match="Shape mismatch"):
            evaluator.accumulate(np.zeros((4, 1024)), np.zeros((4, 2048)))

    def test_invalid_dim_raises(self):
        evaluator = ReconstructionEvaluator(fs=FS)
        bad = np.zeros((4, 4, 4, 1024))
        with pytest.raises(ValueError, match="Expected 2D"):
            evaluator.accumulate(bad, bad)

    def test_prefix_in_summary(self):
        signal = generate_signal(
            SignalConfig(n_samples=4096, n_channels=2, fs=FS, seed=3)
        )
        evaluator = ReconstructionEvaluator(fs=FS)
        evaluator.accumulate(signal, signal)
        summary = evaluator.summarize(prefix="val/")
        assert "val/nmse_total_mean" in summary
        assert "val/n_segments" in summary

    def test_per_channel_summary_shape(self):
        signal = generate_signal(
            SignalConfig(n_samples=4096, n_channels=4, fs=FS, seed=4)
        )
        evaluator = ReconstructionEvaluator(fs=FS)
        evaluator.accumulate(signal, signal)
        evaluator.accumulate(signal, signal)
        per_chan = evaluator.per_channel_summary()
        assert per_chan["nmse_total"].shape == (8,)

    def test_smoothed_signal_pattern(self):
        signal = generate_signal(
            SignalConfig(
                n_samples=2**14, n_channels=8, fs=FS,
                bursts=[(80.0, 4.0, 2.0)], seed=10,
            )
        )
        smoothed = smooth_high_frequencies(signal, FS, cutoff_hz=30.0)
        evaluator = ReconstructionEvaluator(fs=FS)
        metrics = evaluator.evaluate_batch(signal, smoothed)
        assert metrics["nmse_delta_theta_mean"] < 0.1
        assert metrics["nmse_high_gamma_mean"] > 0.5

    def test_connectivity_method_separate(self):
        # Long signal so connectivity is meaningful.
        signal = generate_signal(
            SignalConfig(
                n_samples=int(60 * FS), n_channels=4, fs=FS,
                bursts=[(80.0, 5.0, 2.0)], seed=20,
            )
        )
        evaluator = ReconstructionEvaluator(fs=FS)
        # accumulate isn't required for compute_connectivity.
        result = evaluator.compute_connectivity(signal, signal)
        assert result["connectivity_mae"] == 0.0
        assert result["connectivity_pearson_r"] > 0.99

    def test_connectivity_method_with_prefix(self):
        signal = generate_signal(
            SignalConfig(n_samples=int(60 * FS), n_channels=4, fs=FS, seed=21)
        )
        evaluator = ReconstructionEvaluator(fs=FS)
        result = evaluator.compute_connectivity(signal, signal, prefix="val/")
        assert "val/connectivity_pearson_r" in result


class TestNumpyTorchInterop:
    def test_torch_like_object_is_converted(self):
        class FakeTensor:
            def __init__(self, arr):
                self._arr = arr
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self._arr

        signal = generate_signal(
            SignalConfig(n_samples=2048, n_channels=2, fs=FS, seed=20)
        )
        evaluator = ReconstructionEvaluator(fs=FS)
        metrics = evaluator.evaluate_batch(FakeTensor(signal), FakeTensor(signal))
        assert metrics["nmse_total_mean"] < 1e-6
