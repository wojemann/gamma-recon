"""Model-agnostic evaluation interface for training loops.

Two evaluation modes:

1. **Raw segments (slow path)**: pass each (true, pred) segment to
   `accumulate(...)`. The bandpass filter runs per-segment, which means
   filter edge artifacts span the entire segment for low-frequency bands
   on short (~3s) segments. Use only for quick checks.

2. **Pre-filtered (recommended)**: filter the long session signal once via
   `prefilter_signal`, then pass `(start, stop)` indices for each segment
   to `accumulate_segment(...)`. Filter transients are localized to the
   start/end of the long signal, not every segment boundary.

Connectivity is computed once at the end on pooled long signals via
`compute_connectivity(...)`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from gamma_eval.metrics.prefilter import (
    PreFilteredReport,
    PreFilteredSignals,
    evaluate_prefiltered_segment,
)
from gamma_eval.metrics.reconstruction import (
    DEFAULT_BANDS,
    ReconstructionReport,
    compute_connectivity_metrics,
    evaluate_reconstruction,
)


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy without importing torch."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if hasattr(x, "__array__"):
        return np.asarray(x)
    raise TypeError(f"Cannot convert {type(x)} to numpy array")


# Both report types share the same field names, so we type them as a Union
# for the accumulator.
_AnyReport = ReconstructionReport | PreFilteredReport


class ReconstructionEvaluator:
    """Stateful evaluator. Accumulates per-segment metrics across batches.

    Supports two modes — raw-segment (`accumulate`) and pre-filtered
    (`accumulate_segment`). You can mix both within a single evaluator,
    though typically you'll pick one based on whether you have access to
    the long session signal.

    Connectivity is intentionally separate — call `compute_connectivity()`
    on pooled signals at the end of evaluation.
    """

    def __init__(
        self,
        fs: float,
        bands: dict[str, tuple[float, float]] | None = None,
        gamma_band: tuple[float, float] = (50.0, 200.0),
        gamma_band_name: str = "high_gamma",
    ):
        """
        Args:
            fs: Sampling rate (Hz).
            bands: Bands for band-resolved NMSE. Defaults to DEFAULT_BANDS.
            gamma_band: (low, high) used for the gamma log-spec-distance
                metric AND for connectivity. Used in raw-segment mode.
            gamma_band_name: Key into `bands` for the gamma log-spec-distance
                metric in pre-filtered mode. Should refer to the same band
                as `gamma_band` for consistency.
        """
        self.fs = fs
        self.bands = bands if bands is not None else DEFAULT_BANDS
        self.gamma_band = gamma_band
        self.gamma_band_name = gamma_band_name
        self._reports: list[_AnyReport] = []

    def reset(self):
        """Clear accumulated reports."""
        self._reports = []

    @staticmethod
    def _validate_shapes(
        true: np.ndarray, predicted: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coerce to (n_segments, n_channels, n_samples)."""
        if true.shape != predicted.shape:
            raise ValueError(
                f"Shape mismatch: true {true.shape} vs predicted {predicted.shape}"
            )
        if true.ndim == 2:
            true = true[np.newaxis, ...]
            predicted = predicted[np.newaxis, ...]
        elif true.ndim != 3:
            raise ValueError(
                f"Expected 2D (channels, samples) or 3D (segments, channels, "
                f"samples), got shape {true.shape}"
            )
        return true, predicted

    def accumulate(self, true: Any, predicted: Any) -> None:
        """Raw-segment mode: pass (true, pred) tensors per segment.

        Filtering happens per-segment, which is incorrect for low-frequency
        bands on short segments (filter transients span the segment). For
        accurate band-resolved NMSE on real data, use `accumulate_segment`
        with pre-filtered signals instead.

        Args:
            true, predicted: numpy arrays or torch tensors of shape
                (n_channels, n_samples) or
                (n_segments, n_channels, n_samples).
        """
        true_np = _to_numpy(true)
        pred_np = _to_numpy(predicted)
        true_np, pred_np = self._validate_shapes(true_np, pred_np)

        for i in range(true_np.shape[0]):
            report = evaluate_reconstruction(
                true_np[i],
                pred_np[i],
                fs=self.fs,
                bands=self.bands,
                gamma_band=self.gamma_band,
            )
            self._reports.append(report)

    def accumulate_segment(
        self,
        pre_true: PreFilteredSignals,
        pre_pred: PreFilteredSignals,
        start: int,
        stop: int,
    ) -> None:
        """Pre-filtered mode: slice the pre-filtered arrays at [start:stop].

        Args:
            pre_true, pre_pred: PreFilteredSignals for the SAME long session.
                Must already be filtered with bands matching this evaluator's
                `self.bands` (or a superset).
            start, stop: Sample indices defining this segment.
        """
        # Sanity: bands available in pre-filtered must include all bands the
        # evaluator was configured with.
        missing = set(self.bands) - set(pre_true.bands)
        if missing:
            raise ValueError(
                f"Pre-filtered signal missing required bands: {sorted(missing)}"
            )
        report = evaluate_prefiltered_segment(
            pre_true,
            pre_pred,
            start=start,
            stop=stop,
            gamma_band_name=self.gamma_band_name,
        )
        self._reports.append(report)

    def evaluate_batch(self, true: Any, predicted: Any) -> dict[str, float]:
        """One-shot raw-segment evaluation. Doesn't modify self's state."""
        scratch = ReconstructionEvaluator(
            fs=self.fs,
            bands=self.bands,
            gamma_band=self.gamma_band,
            gamma_band_name=self.gamma_band_name,
        )
        scratch.accumulate(true, predicted)
        return scratch.summarize()

    def summarize(self, prefix: str = "") -> dict[str, float]:
        """Aggregate per-segment metrics. Connectivity NOT included."""
        if not self._reports:
            return {f"{prefix}n_segments": 0}

        all_nmse = np.concatenate([r.nmse_total for r in self._reports])
        all_log_spec = np.concatenate(
            [r.log_spectral_distance for r in self._reports]
        )
        all_log_spec_gamma = np.concatenate(
            [r.log_spectral_distance_gamma for r in self._reports]
        )

        out = {
            f"{prefix}nmse_total_mean": float(np.nanmean(all_nmse)),
            f"{prefix}log_spec_dist_mean": float(np.nanmean(all_log_spec)),
            f"{prefix}log_spec_dist_gamma_mean": float(
                np.nanmean(all_log_spec_gamma)
            ),
            f"{prefix}n_segments": len(self._reports),
        }
        for band in self.bands:
            band_vals = np.concatenate(
                [r.band_nmse[band] for r in self._reports]
            )
            out[f"{prefix}nmse_{band}_mean"] = float(np.nanmean(band_vals))
        return out

    def compute_connectivity(
        self,
        true_pool: Any,
        predicted_pool: Any,
        prefix: str = "",
        min_duration_sec: float = 60.0,
    ) -> dict[str, float]:
        """Connectivity on pooled long signals. Call at end of eval."""
        true_np = _to_numpy(true_pool)
        pred_np = _to_numpy(predicted_pool)
        result = compute_connectivity_metrics(
            true_np,
            pred_np,
            fs=self.fs,
            band=self.gamma_band,
            min_duration_sec=min_duration_sec,
        )
        return {f"{prefix}connectivity_{k}": v for k, v in result.items()}

    def per_channel_summary(self, prefix: str = "") -> dict[str, np.ndarray]:
        """Per-channel metric arrays for plotting."""
        if not self._reports:
            return {}
        out = {
            f"{prefix}nmse_total": np.concatenate(
                [r.nmse_total for r in self._reports]
            ),
            f"{prefix}log_spec_dist": np.concatenate(
                [r.log_spectral_distance for r in self._reports]
            ),
            f"{prefix}log_spec_dist_gamma": np.concatenate(
                [r.log_spectral_distance_gamma for r in self._reports]
            ),
        }
        for band in self.bands:
            out[f"{prefix}nmse_{band}"] = np.concatenate(
                [r.band_nmse[band] for r in self._reports]
            )
        return out


def evaluate_model_on_loader(
    model_predict_fn,
    data_loader: Iterable,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    gamma_band: tuple[float, float] = (50.0, 200.0),
    pool_for_connectivity: bool = True,
) -> dict[str, float]:
    """Run a model over a loader, return aggregated per-segment metrics.

    Uses the raw-segment (slow) path. For real evaluation you'll usually
    want a custom loop that pre-filters per session — see the docstring of
    `gamma_eval.metrics.prefilter` for the recommended pattern.
    """
    evaluator = ReconstructionEvaluator(
        fs=fs, bands=bands, gamma_band=gamma_band
    )
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    for batch in data_loader:
        true, predicted = model_predict_fn(batch)
        evaluator.accumulate(true, predicted)
        if pool_for_connectivity:
            t = _to_numpy(true)
            p = _to_numpy(predicted)
            if t.ndim == 3:
                t = np.concatenate(list(t), axis=-1)
                p = np.concatenate(list(p), axis=-1)
            pooled_true.append(t)
            pooled_pred.append(p)

    metrics = evaluator.summarize()
    if pool_for_connectivity and pooled_true:
        true_pool = np.concatenate(pooled_true, axis=-1)
        pred_pool = np.concatenate(pooled_pred, axis=-1)
        metrics.update(evaluator.compute_connectivity(true_pool, pred_pool))
    return metrics
