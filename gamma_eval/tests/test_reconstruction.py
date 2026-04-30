"""Tests for the reconstruction evaluation harness."""

from __future__ import annotations

import numpy as np
import pytest

from gamma_eval.metrics.reconstruction import (
    DEFAULT_BANDS,
    band_resolved_nmse,
    bandpass_filter,
    compute_connectivity_metrics,
    envelope_correlation_matrix,
    evaluate_reconstruction,
    log_spectral_distance,
    nmse,
    power_spectrum,
    r_squared,
)
from gamma_eval.synthetic.signals import (
    SignalConfig,
    generate_signal,
    smooth_high_frequencies,
)


FS = 2048.0


class TestNMSE:
    def test_perfect_reconstruction_gives_zero(self):
        signal = generate_signal(
            SignalConfig(n_samples=4096, n_channels=4, fs=FS, seed=0)
        )
        np.testing.assert_allclose(nmse(signal, signal), 0.0, atol=1e-10)

    def test_predicting_mean_gives_one(self):
        signal = generate_signal(
            SignalConfig(n_samples=4096, n_channels=4, fs=FS, seed=1)
        )
        prediction = np.broadcast_to(
            signal.mean(axis=-1, keepdims=True), signal.shape
        ).copy()
        np.testing.assert_allclose(nmse(signal, prediction), 1.0, atol=1e-10)

    def test_constant_signal_returns_nan(self):
        true = np.ones((2, 100))
        predicted = np.zeros((2, 100))
        assert np.all(np.isnan(nmse(true, predicted)))

    def test_worse_than_mean_gives_above_one(self):
        signal = generate_signal(
            SignalConfig(n_samples=4096, n_channels=2, fs=FS, seed=2)
        )
        np.testing.assert_allclose(nmse(signal, -signal), 4.0, atol=1e-10)

    def test_r_squared_complement_of_nmse(self):
        signal = generate_signal(
            SignalConfig(n_samples=4096, n_channels=2, fs=FS, seed=3)
        )
        rng = np.random.default_rng(99)
        prediction = signal + rng.standard_normal(signal.shape) * 0.5
        np.testing.assert_allclose(
            r_squared(signal, prediction),
            1.0 - nmse(signal, prediction),
            atol=1e-10,
        )


class TestBandResolvedNMSE:
    def test_smoothed_signal_fails_high_gamma(self):
        signal = generate_signal(
            SignalConfig(
                n_samples=8192, n_channels=4, fs=FS,
                aperiodic_exponent=1.5,
                bursts=[(80.0, 5.0, 2.0)], seed=10,
            )
        )
        smoothed = smooth_high_frequencies(
            signal, FS, cutoff_hz=30.0, attenuation_db=20.0
        )
        results = band_resolved_nmse(signal, smoothed, FS)
        assert np.nanmean(results["delta_theta"]) < 0.1
        assert np.nanmean(results["high_gamma"]) > 0.5

    def test_bands_have_expected_keys(self):
        signal = generate_signal(
            SignalConfig(n_samples=4096, n_channels=2, fs=FS, seed=11)
        )
        results = band_resolved_nmse(signal, signal, FS)
        assert set(results.keys()) == set(DEFAULT_BANDS.keys())

    def test_custom_bands(self):
        signal = generate_signal(
            SignalConfig(n_samples=4096, n_channels=2, fs=FS, seed=12)
        )
        results = band_resolved_nmse(
            signal, signal, FS, bands={"my_band": (10.0, 20.0)}
        )
        assert set(results.keys()) == {"my_band"}

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            band_resolved_nmse(np.zeros((2, 100)), np.zeros((2, 200)), FS)

    def test_perfect_reconstruction_zero_in_every_band(self):
        signal = generate_signal(
            SignalConfig(n_samples=8192, n_channels=2, fs=FS, seed=13)
        )
        results = band_resolved_nmse(signal, signal, FS)
        for band_name, vals in results.items():
            np.testing.assert_allclose(
                vals, 0.0, atol=1e-8, err_msg=f"Band {band_name} not zero"
            )

    def test_invalid_band_raises(self):
        with pytest.raises(ValueError, match="must exceed"):
            band_resolved_nmse(
                np.zeros((2, 1024)), np.zeros((2, 1024)), FS,
                bands={"bad": (50.0, 30.0)},
            )

    def test_band_above_nyquist_raises(self):
        with pytest.raises(ValueError, match="Nyquist"):
            band_resolved_nmse(
                np.zeros((2, 1024)), np.zeros((2, 1024)),
                fs=400.0, bands={"bad": (250.0, 300.0)},
            )


class TestSpectralMetrics:
    def test_pink_noise_has_correct_slope(self):
        signal = generate_signal(
            SignalConfig(
                n_samples=2**16, n_channels=1, fs=FS,
                aperiodic_exponent=1.5, seed=20,
            )
        )
        freqs, psd = power_spectrum(signal[0], FS, nperseg=2048)
        mask = (freqs >= 5.0) & (freqs <= 200.0)
        slope, _ = np.polyfit(np.log10(freqs[mask]), np.log10(psd[mask]), 1)
        np.testing.assert_allclose(slope, -1.5, atol=0.1)

    def test_log_spectral_distance_zero_for_identical(self):
        signal = generate_signal(
            SignalConfig(n_samples=4096, n_channels=2, fs=FS, seed=21)
        )
        np.testing.assert_allclose(
            log_spectral_distance(signal, signal, FS), 0.0, atol=1e-10
        )

    def test_log_spectral_distance_band_restriction(self):
        signal = generate_signal(
            SignalConfig(
                n_samples=8192, n_channels=2, fs=FS,
                bursts=[(80.0, 5.0, 2.0)], seed=22,
            )
        )
        smoothed = smooth_high_frequencies(signal, FS, cutoff_hz=30.0)
        full_dist = log_spectral_distance(signal, smoothed, FS)
        low_dist = log_spectral_distance(signal, smoothed, FS, band=(1.0, 25.0))
        gamma_dist = log_spectral_distance(
            signal, smoothed, FS, band=(50.0, 200.0)
        )
        assert np.all(low_dist < full_dist)
        assert np.all(gamma_dist > full_dist)


class TestConnectivity:
    def test_envelope_correlation_self_is_identity(self):
        signal = generate_signal(
            SignalConfig(n_samples=2**14, n_channels=4, fs=FS, seed=30)
        )
        conn = envelope_correlation_matrix(signal, FS)
        np.testing.assert_allclose(np.diag(conn), 1.0, atol=1e-10)

    def test_correlated_channels_have_high_envelope_correlation(self):
        base = generate_signal(
            SignalConfig(
                n_samples=2**14, n_channels=1, fs=FS,
                bursts=[(80.0, 5.0, 3.0)], seed=31,
            )
        )
        n_0 = generate_signal(
            SignalConfig(n_samples=2**14, n_channels=1, fs=FS, seed=32)
        )
        chan_0 = base[0] + 0.5 * n_0[0]
        n_1 = generate_signal(
            SignalConfig(n_samples=2**14, n_channels=1, fs=FS, seed=33)
        )
        chan_1 = base[0] + 0.5 * n_1[0]
        chan_2 = generate_signal(
            SignalConfig(
                n_samples=2**14, n_channels=1, fs=FS,
                bursts=[(140.0, 5.0, 3.0)], seed=34,
            )
        )[0]
        signal = np.stack([chan_0, chan_1, chan_2], axis=0)
        conn = envelope_correlation_matrix(signal, FS)
        assert conn[0, 1] > conn[0, 2]
        assert conn[0, 1] > 0.3

    def test_connectivity_metrics_perfect_match(self):
        signal = generate_signal(
            SignalConfig(
                n_samples=int(60 * FS), n_channels=4, fs=FS,
                bursts=[(80.0, 5.0, 2.0)], seed=40,
            )
        )
        result = compute_connectivity_metrics(signal, signal, FS)
        np.testing.assert_allclose(result["mae"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["frobenius_distance"], 0.0, atol=1e-10)
        assert result["pearson_r"] > 0.99

    def test_connectivity_short_signal_warns(self):
        signal = generate_signal(
            SignalConfig(n_samples=2**13, n_channels=4, fs=FS, seed=41)
        )
        with pytest.warns(UserWarning, match="noisy below"):
            compute_connectivity_metrics(signal, signal, FS)

    def test_connectivity_two_channels_returns_nan_pearson(self):
        signal = generate_signal(
            SignalConfig(n_samples=int(60 * FS), n_channels=2, fs=FS, seed=42)
        )
        result = compute_connectivity_metrics(signal, signal, FS)
        assert np.isnan(result["pearson_r"])
        assert result["mae"] == 0.0

    def test_connectivity_degraded_when_gamma_replaced_by_noise(self):
        rng = np.random.default_rng(43)
        n_samples = int(90 * FS)
        base = generate_signal(
            SignalConfig(
                n_samples=n_samples, n_channels=1, fs=FS,
                bursts=[(80.0, 5.0, 3.0)], seed=43,
            )
        )[0]
        n_channels = 6
        signal = np.zeros((n_channels, n_samples))
        for i in range(n_channels):
            cluster = i // 2
            shifted = np.roll(base, cluster * 100)
            own_noise = generate_signal(
                SignalConfig(n_samples=n_samples, n_channels=1, fs=FS,
                             seed=100 + i)
            )[0]
            signal[i] = shifted + 0.5 * own_noise

        low_freq = bandpass_filter(signal, FS, 1.0, 30.0)
        bad_gamma = rng.standard_normal(signal.shape) * np.std(
            bandpass_filter(signal, FS, 30.0, 200.0)
        )
        bad_prediction = low_freq + bandpass_filter(bad_gamma, FS, 30.0, 200.0)

        result = compute_connectivity_metrics(signal, bad_prediction, FS)
        assert result["pearson_r"] < 0.9


class TestEvaluateReconstruction:
    def test_full_pipeline_runs(self):
        signal = generate_signal(
            SignalConfig(
                n_samples=2**14, n_channels=4, fs=FS,
                bursts=[(80.0, 5.0, 2.0)], seed=50,
            )
        )
        smoothed = smooth_high_frequencies(signal, FS, cutoff_hz=30.0)
        report = evaluate_reconstruction(signal, smoothed, FS)
        assert set(report.band_nmse.keys()) == set(DEFAULT_BANDS.keys())
        assert not hasattr(report, "connectivity")
        summary = report.summary()
        for k, v in summary.items():
            assert isinstance(v, float)
        assert (
            np.nanmean(report.band_nmse["high_gamma"])
            > np.nanmean(report.band_nmse["delta_theta"])
        )

    def test_single_channel_works(self):
        signal = generate_signal(
            SignalConfig(n_samples=4096, n_channels=1, fs=FS, seed=51)
        )
        report = evaluate_reconstruction(signal, signal, FS)
        assert report.nmse_total.shape == (1,)
