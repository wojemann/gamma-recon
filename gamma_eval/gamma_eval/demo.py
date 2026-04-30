"""End-to-end demo: BaRISTA-style high-gamma failure on synthetic data.

Three "models" evaluated via the recommended pre-filtered path:
  1. Oracle (perfect).
  2. Smoothed (mimics MSE-trained model that smooths gamma).
  3. Low-freq + uncorrelated noise (mimics model that gives up on gamma).

The pre-filtered path filters the long signal once per session, then slices
into 3-second segments for per-segment metric computation. This avoids
filter edge artifacts that contaminate per-segment band-resolved NMSE on
short segments.
"""

from __future__ import annotations

import numpy as np

from gamma_eval.evaluator import ReconstructionEvaluator
from gamma_eval.metrics.prefilter import prefilter_signal
from gamma_eval.metrics.reconstruction import bandpass_filter
from gamma_eval.synthetic.signals import (
    SignalConfig,
    generate_signal,
    smooth_high_frequencies,
)


FS = 2048.0


def make_signal(seed: int = 0) -> np.ndarray:
    """16 channels, 60s, 1/f^1.5 + bursts at multiple bands."""
    return generate_signal(
        SignalConfig(
            n_samples=int(60 * FS),
            n_channels=16,
            fs=FS,
            aperiodic_exponent=1.5,
            bursts=[
                (10.0, 1.5, 0.5),
                (25.0, 1.0, 1.0),
                (80.0, 3.0, 2.0),
                (130.0, 2.0, 1.5),
            ],
            seed=seed,
        )
    )


def make_predictions(true_signal: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    sigma = float(np.std(true_signal))

    perfect = true_signal.copy()
    smoothed = smooth_high_frequencies(
        true_signal, FS, cutoff_hz=30.0, attenuation_db=15.0
    )
    low_freq = bandpass_filter(true_signal, FS, 1.0, 30.0)
    uncorrelated_high = rng.standard_normal(true_signal.shape) * 0.3 * sigma
    high_band = bandpass_filter(uncorrelated_high, FS, 30.0, 200.0)
    noisy = low_freq + high_band

    return {"oracle": perfect, "smoothed": smoothed, "low_freq_only": noisy}


def format_row(name: str, summary: dict[str, float]) -> str:
    keys = [
        "nmse_total_mean",
        "nmse_delta_theta_mean",
        "nmse_alpha_beta_mean",
        "nmse_low_gamma_mean",
        "nmse_high_gamma_mean",
        "connectivity_pearson_r",
    ]
    cells = [f"{name:<16}"]
    for k in keys:
        v = summary.get(k, np.nan)
        cells.append(f"{v:>10.3f}")
    return " | ".join(cells)


def main():
    print(f"Generating 60s synthetic signal (16 channels @ {FS} Hz)...")
    true_signal = make_signal(seed=0)
    print(f"Shape: {true_signal.shape}, std: {true_signal.std():.3f}\n")

    predictions = make_predictions(true_signal)

    seg_len = int(3 * FS)
    n_segments = true_signal.shape[1] // seg_len

    header = ["model", "nmse_total", "delta_theta", "alpha_beta",
              "low_gamma", "high_gamma", "conn_r"]
    print(" | ".join(f"{k:>10}" if k != "model" else f"{k:<16}" for k in header))
    print("-" * 100)

    # Pre-filter the true signal once.
    pre_true = prefilter_signal(true_signal, FS)

    for name, predicted in predictions.items():
        # Pre-filter the prediction once.
        pre_pred = prefilter_signal(predicted, FS)

        evaluator = ReconstructionEvaluator(fs=FS)
        # Accumulate per-segment metrics by slicing the pre-filtered signals.
        for s in range(n_segments):
            evaluator.accumulate_segment(
                pre_true, pre_pred, s * seg_len, (s + 1) * seg_len
            )
        summary = evaluator.summarize()
        # Connectivity on the full pooled signal.
        summary.update(evaluator.compute_connectivity(true_signal, predicted))
        print(format_row(name, summary))

    print()
    print("Expected:")
    print("  oracle:       all NMSE ~0, conn_r ~1.")
    print("  smoothed:     delta-theta near 0 (now that pre-filtering removes")
    print("                the per-segment edge artifact), NMSE rises with")
    print("                frequency, high-gamma > 0.5, conn_r ~1.")
    print("  low_freq_only: low-freq near 0, high-gamma ~1, conn_r near 0.")


if __name__ == "__main__":
    main()
