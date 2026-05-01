"""Pre-filtering pipeline: filter long signals once, slice per segment.

The naive approach (calling `bandpass_filter` inside `band_resolved_nmse` on
each 3-second segment) has a serious problem: filter transients at low
frequencies span the entire segment. A 4th-order Butterworth at 1 Hz has
~3 sec transients at each edge after `filtfilt` — there's no clean middle
of a 3-second segment.

The fix: filter the *long* signal (one whole session) once, then slice the
filtered array into segments. The transient is localized to the first/last
few seconds of the session, not every segment boundary.

Usage:

    # Once per session, on the long signal:
    pre_true = prefilter_signal(true_session, fs=2048.0)
    pre_pred = prefilter_signal(predicted_session, fs=2048.0)

    # In the evaluation loop, slicing per segment:
    for start, stop in segment_indices:
        report = evaluate_prefiltered_segment(pre_true, pre_pred, start, stop)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gamma_eval.metrics.reconstruction import (
    DEFAULT_BANDS,
    _validate_bands,
    bandpass_filter,
    log_spectral_distance,
    nmse,
)


@dataclass
class PreFilteredSignals:
    """Long signal + per-band filtered versions, for cheap segment-wise access.

    Attributes:
        signal: Original (n_channels, n_samples) signal.
        band_filtered: Dict mapping band name to filtered (n_channels, n_samples)
            array. Same shape as `signal`.
        fs: Sampling rate (Hz). Stored so consumers can compute time-aligned
            segment indices.
        bands: Dict of band name -> (low_hz, high_hz). Stored so consumers can
            verify which bands are available.
    """

    signal: np.ndarray
    band_filtered: dict[str, np.ndarray]
    fs: float
    bands: dict[str, tuple[float, float]]

    def slice_segment(
        self, start: int, stop: int
    ) -> "PreFilteredSignals":
        """Return a view containing only samples [start:stop].

        Cheap: numpy slicing creates views, not copies. Useful when passing
        a single segment to a metric function that expects a PreFilteredSignals
        but you only want one chunk.
        """
        return PreFilteredSignals(
            signal=self.signal[..., start:stop],
            band_filtered={
                name: arr[..., start:stop] for name, arr in self.band_filtered.items()
            },
            fs=self.fs,
            bands=self.bands,
        )

    def __len__(self) -> int:
        return self.signal.shape[-1]


def prefilter_signal(
    signal: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    order: int = 4,
) -> PreFilteredSignals:
    """Bandpass-filter `signal` into each band; return a PreFilteredSignals.

    Filter is applied to the FULL signal (along the last axis), so transient
    edge artifacts only appear at the start/end of the entire input — not at
    arbitrary segmentation boundaries.

    Args:
        signal: (..., n_samples) array. The long signal you'll later slice.
        fs: Sampling rate (Hz).
        bands: Bands to compute. Defaults to DEFAULT_BANDS.
        order: Butterworth order (per direction; 2x effective with filtfilt).

    Returns:
        PreFilteredSignals with the original signal and one filtered array
        per band.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    _validate_bands(bands, fs)

    band_filtered = {}
    for name, (low, high) in bands.items():
        band_filtered[name] = bandpass_filter(signal, fs, low, high, order=order)

    return PreFilteredSignals(
        signal=signal, band_filtered=band_filtered, fs=fs, bands=bands
    )


@dataclass
class PreFilteredReport:
    """Per-segment metrics computed from pre-filtered signals.

    Equivalent to ReconstructionReport but produced via the prefilter
    pipeline. Same fields, intended to be a drop-in replacement.
    """

    nmse_total: np.ndarray
    band_nmse: dict[str, np.ndarray]
    log_spectral_distance: np.ndarray
    log_spectral_distance_gamma: np.ndarray

    def summary(self) -> dict[str, float]:
        out = {
            "nmse_total_mean": float(np.nanmean(self.nmse_total)),
            "log_spec_dist_mean": float(np.nanmean(self.log_spectral_distance)),
            "log_spec_dist_gamma_mean": float(
                np.nanmean(self.log_spectral_distance_gamma)
            ),
        }
        for band, vals in self.band_nmse.items():
            out[f"nmse_{band}_mean"] = float(np.nanmean(vals))
        return out


def evaluate_prefiltered_segment(
    pre_true: PreFilteredSignals,
    pre_pred: PreFilteredSignals,
    start: int | None = None,
    stop: int | None = None,
    gamma_band_name: str = "high_gamma",
) -> PreFilteredReport:
    """Compute per-segment metrics by slicing pre-filtered arrays.

    No filtering happens here — that was done upstream in `prefilter_signal`.
    This function just slices `[start:stop]` from each pre-filtered band
    array and computes NMSE on the slice. Cheap.

    Args:
        pre_true, pre_pred: PreFilteredSignals from the same long session.
            Must have matching bands and matching shapes.
        start, stop: Sample indices defining the segment. If both None, the
            full pre-filtered signal is used (i.e., evaluate the whole
            session as one segment).
        gamma_band_name: Which band to use for the gamma log-spec-distance
            metric. Must be a key of pre_true.bands. Defaults to "high_gamma".

    Returns:
        PreFilteredReport with per-channel arrays.
    """
    if pre_true.signal.shape != pre_pred.signal.shape:
        raise ValueError(
            f"Shape mismatch: true {pre_true.signal.shape} vs pred {pre_pred.signal.shape}"
        )
    if pre_true.bands != pre_pred.bands:
        raise ValueError(
            "Pre-filtered signals must have the same band definitions"
        )
    if pre_true.fs != pre_pred.fs:
        raise ValueError(
            f"Sampling rate mismatch: {pre_true.fs} vs {pre_pred.fs}"
        )
    if gamma_band_name not in pre_true.bands:
        raise ValueError(
            f"gamma_band_name '{gamma_band_name}' not in pre-filtered bands "
            f"{list(pre_true.bands.keys())}"
        )

    # Slice. None,None = full signal.
    if start is None and stop is None:
        true_slice = pre_true.signal
        pred_slice = pre_pred.signal
        true_bands = pre_true.band_filtered
        pred_bands = pre_pred.band_filtered
    else:
        sl = slice(start, stop)
        true_slice = pre_true.signal[..., sl]
        pred_slice = pre_pred.signal[..., sl]
        true_bands = {k: v[..., sl] for k, v in pre_true.band_filtered.items()}
        pred_bands = {k: v[..., sl] for k, v in pre_pred.band_filtered.items()}

    if true_slice.ndim != 2:
        raise ValueError(
            f"Expected 2D (n_channels, n_samples) slice, got {true_slice.shape}"
        )

    # Per-band NMSE: just slice and compute, no re-filtering.
    band_nmse = {
        name: nmse(true_bands[name], pred_bands[name])
        for name in pre_true.bands
    }

    # Log-spectral distances are PSD-based; they handle their own windowing
    # internally via Welch. Compute on the (unfiltered) slice. These are
    # less affected by filter edge effects since Welch averages over
    # segments and uses Hann windows.
    log_spec = log_spectral_distance(true_slice, pred_slice, pre_true.fs)
    gamma_low, gamma_high = pre_true.bands[gamma_band_name]
    log_spec_gamma = log_spectral_distance(
        true_slice, pred_slice, pre_true.fs, band=(gamma_low, gamma_high)
    )

    return PreFilteredReport(
        nmse_total=nmse(true_slice, pred_slice),
        band_nmse=band_nmse,
        log_spectral_distance=log_spec,
        log_spectral_distance_gamma=log_spec_gamma,
    )
