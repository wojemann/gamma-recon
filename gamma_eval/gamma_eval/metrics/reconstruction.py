"""Reconstruction quality metrics for neural signal models.

Design principles:
  - All functions are stateless and operate on numpy arrays.
  - Convention: signals have shape (..., n_samples) along the time axis.
  - Multi-channel signals are (n_channels, n_samples) or
    (n_segments, n_channels, n_samples).
  - We avoid assumptions about the model — these work on any (true, predicted)
    pair regardless of how `predicted` was generated.
  - We return both per-electrode and aggregate values where it makes sense, so
    the caller can decide what to plot.

The metric definitions match BaRISTA Table 13 conventions where applicable so
results are comparable to the published numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal


# ----------------------------------------------------------------------------
# Band definitions
# ----------------------------------------------------------------------------

# Standard neuroscience bands. The "high gamma" band here is 50-200 Hz per
# the user's preference (rather than the narrower 70-150 Hz definition some
# papers use). 50-200 captures the "broadband gamma" / high-frequency
# activity range that's coupled to BOLD and to cognitive task performance.
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta_theta": (1.0, 8.0),
    "alpha_beta": (8.0, 30.0),
    "low_gamma": (30.0, 50.0),
    "high_gamma": (50.0, 200.0),
}


# ----------------------------------------------------------------------------
# Filtering utilities
# ----------------------------------------------------------------------------


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Zero-phase (filtfilt) is important here because we don't want phase
    distortion to confound reconstruction quality comparisons.

    Note on edge effects: filtfilt produces transient artifacts of length
    roughly 3 * order / low_hz seconds at each edge. For a 4th-order filter
    at 1 Hz, that's ~12 seconds — which exceeds typical 3-second segments.
    For low-frequency band metrics on short segments, expect inflated NMSE
    from edge artifacts. Mitigations: pad with surrounding context before
    filtering, or restrict the lowest band cutoff to ~4 Hz on 3-second data.

    Args:
        signal: (..., n_samples) array.
        fs: Sampling rate.
        low_hz, high_hz: Passband edges in Hz.
        order: Filter order (per direction; effective order is 2x with filtfilt).

    Returns:
        Filtered signal, same shape as input.
    """
    nyq = fs / 2.0
    if low_hz <= 0:
        raise ValueError(f"low_hz must be > 0, got {low_hz}")
    if high_hz <= low_hz:
        raise ValueError(
            f"high_hz ({high_hz}) must be greater than low_hz ({low_hz})"
        )
    if low_hz >= nyq:
        raise ValueError(
            f"low_hz ({low_hz}) must be below Nyquist ({nyq})"
        )
    if high_hz >= nyq:
        # Highpass fallback when band extends to Nyquist. Warn so a typo
        # (e.g. high_hz=200 with fs=400) doesn't silently change behavior.
        import warnings
        warnings.warn(
            f"high_hz={high_hz} >= Nyquist={nyq}; using highpass instead "
            f"of bandpass.",
            stacklevel=2,
        )
        sos = scipy_signal.butter(
            order, low_hz / nyq, btype="highpass", output="sos"
        )
    else:
        sos = scipy_signal.butter(
            order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos"
        )
    return scipy_signal.sosfiltfilt(sos, signal, axis=-1)


def _validate_bands(
    bands: dict[str, tuple[float, float]], fs: float
) -> None:
    """Check that all bands are well-formed and within Nyquist."""
    nyq = fs / 2.0
    for name, edges in bands.items():
        if len(edges) != 2:
            raise ValueError(f"Band '{name}': expected (low, high), got {edges}")
        low, high = edges
        if low <= 0:
            raise ValueError(f"Band '{name}': low={low} must be > 0")
        if high <= low:
            raise ValueError(f"Band '{name}': high={high} must exceed low={low}")
        if low >= nyq:
            raise ValueError(
                f"Band '{name}': low={low} >= Nyquist={nyq} for fs={fs}"
            )


# ----------------------------------------------------------------------------
# Core reconstruction metrics
# ----------------------------------------------------------------------------


def nmse(
    true: np.ndarray, predicted: np.ndarray, axis: int = -1
) -> np.ndarray:
    """Normalized mean squared error.

    NMSE = MSE(true, predicted) / Var(true)

    NMSE = 1.0 means the model is no better than predicting the mean.
    NMSE > 1.0 means the model is *worse* than predicting the mean.
    NMSE = 0.0 means perfect reconstruction.

    This is the metric BaRISTA reports in their Table 13. Their high-frequency
    NMSE > 1.0 across all configurations is the smoking gun that motivates
    this whole project.

    Args:
        true: Ground truth signal.
        predicted: Reconstructed signal.
        axis: Axis along which to compute the variance and MSE. Default -1
            (the time axis).

    Returns:
        NMSE per channel/segment (whatever dimensions remain after reducing
        the time axis).
    """
    mse_val = np.mean((true - predicted) ** 2, axis=axis)
    var_val = np.var(true, axis=axis)
    # Avoid divide-by-zero if a channel is constant. A constant true signal
    # is degenerate; return NaN so the caller can handle it.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(var_val > 0, mse_val / var_val, np.nan)
    return result


def r_squared(
    true: np.ndarray, predicted: np.ndarray, axis: int = -1
) -> np.ndarray:
    """Coefficient of determination.

    R^2 = 1 - NMSE.

    R^2 = 1.0: perfect reconstruction.
    R^2 = 0.0: model no better than mean.
    R^2 < 0.0: model worse than mean.

    Reported alongside NMSE for readability — some readers prefer R^2.
    """
    return 1.0 - nmse(true, predicted, axis=axis)


def band_resolved_nmse(
    true: np.ndarray,
    predicted: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    axis: int = -1,
) -> dict[str, np.ndarray]:
    """Compute NMSE within each frequency band.

    For each band, we bandpass-filter both true and predicted signals to the
    band, then compute NMSE. This is the band-by-band breakdown that lets us
    see *where* in frequency a model fails — the central diagnostic for this
    project.

    Args:
        true, predicted: (..., n_samples) arrays. Must be the same shape.
        fs: Sampling rate.
        bands: Dict of band name -> (low_hz, high_hz). Defaults to
            DEFAULT_BANDS.
        axis: Time axis.

    Returns:
        Dict mapping band name to NMSE array (shape determined by axis
        argument and input shape).
    """
    if true.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: true {true.shape} vs predicted {predicted.shape}"
        )
    if bands is None:
        bands = DEFAULT_BANDS
    _validate_bands(bands, fs)

    results = {}
    for band_name, (low, high) in bands.items():
        true_band = bandpass_filter(true, fs, low, high)
        pred_band = bandpass_filter(predicted, fs, low, high)
        results[band_name] = nmse(true_band, pred_band, axis=axis)
    return results


# ----------------------------------------------------------------------------
# Spectral metrics
# ----------------------------------------------------------------------------


def power_spectrum(
    signal: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch's PSD estimate.

    Returns:
        (frequencies, power) tuple. Power has the input shape with the time
        axis replaced by frequency.
    """
    if nperseg is None:
        # Default: ~1 second of data per segment, capped at signal length.
        nperseg = min(int(fs), signal.shape[axis])
    freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg, axis=axis)
    return freqs, psd


def log_spectral_distance(
    true: np.ndarray,
    predicted: np.ndarray,
    fs: float,
    band: tuple[float, float] | None = None,
    axis: int = -1,
) -> np.ndarray:
    """Mean absolute distance in log-power between true and predicted.

    Useful for asking "does the predicted signal have the right *spectral
    shape*, even if the time-domain match is imperfect?" This catches a
    failure mode where a model gets the spectrum right but the phases wrong.

    Args:
        true, predicted: signals.
        fs: Sampling rate.
        band: Optional (low, high) Hz range to restrict the comparison to.
            If None, use the full frequency range.
        axis: Time axis.

    Returns:
        Mean |log10(P_true) - log10(P_pred)| over frequencies in the band.
    """
    freqs, psd_true = power_spectrum(true, fs, axis=axis)
    _, psd_pred = power_spectrum(predicted, fs, axis=axis)

    # Eps is a tiny fraction of the per-channel median power, so it's
    # negligible for normal values but prevents log(0) on dead bins.
    # Using true's median (true is the reference signal).
    median_power = np.median(psd_true, axis=-1, keepdims=True)
    eps = 1e-10 * np.maximum(median_power, np.finfo(np.float64).tiny)
    log_true = np.log10(psd_true + eps)
    log_pred = np.log10(psd_pred + eps)

    diff = np.abs(log_true - log_pred)

    if band is not None:
        mask = (freqs >= band[0]) & (freqs <= band[1])
        diff = diff[..., mask]

    return np.mean(diff, axis=-1)


# ----------------------------------------------------------------------------
# Connectivity / network metrics
# ----------------------------------------------------------------------------
#
# DESIGN NOTE: connectivity is computed on POOLED long signals, not on
# per-segment chunks. High-gamma envelope correlations are noisy on short
# windows (3 seconds is far too short for stable connectivity estimates).
# The intended workflow is:
#   1. Run model over full validation set, collect (true, pred) per segment.
#   2. Concatenate all segments into one long signal per channel.
#   3. Call compute_connectivity_metrics(true_pool, pred_pool, fs).
# Connectivity is therefore NOT part of ReconstructionReport (which is
# per-segment) and NOT computed inside ReconstructionEvaluator.accumulate().


def gamma_envelope(
    signal: np.ndarray,
    fs: float,
    band: tuple[float, float] = (50.0, 200.0),
) -> np.ndarray:
    """Bandpass + Hilbert magnitude. Standard high-gamma envelope.

    Args:
        signal: (..., n_samples) array.
        fs: Sampling rate.
        band: (low_hz, high_hz) for the band of interest.

    Returns:
        Envelope, same shape as input.
    """
    filtered = bandpass_filter(signal, fs, band[0], band[1])
    analytic = scipy_signal.hilbert(filtered, axis=-1)
    return np.abs(analytic)


def envelope_correlation_matrix(
    signal: np.ndarray,
    fs: float,
    band: tuple[float, float] = (50.0, 200.0),
    log_envelope: bool = True,
) -> np.ndarray:
    """Pairwise envelope correlations across channels (iEEG analogue of
    fMRI functional connectivity; Das & Menon 2022).

    Args:
        signal: (n_channels, n_samples).
        fs: Sampling rate.
        band: Frequency band for the envelope.
        log_envelope: If True, log-transform the envelope before correlating.
            Envelope distributions are log-normal-ish; log makes Pearson
            more meaningful. Pearson is shift-invariant so the log offset
            doesn't bias the correlation.

    Returns:
        (n_channels, n_channels) correlation matrix.
    """
    if signal.ndim != 2:
        raise ValueError(
            f"Expected (n_channels, n_samples), got shape {signal.shape}"
        )
    env = gamma_envelope(signal, fs, band)
    if log_envelope:
        # Tiny additive eps to avoid log(0) on dead/saturated channels.
        # Pearson is shift-invariant, so the constant offset doesn't matter.
        env = np.log(env + 1e-12)
    return np.corrcoef(env)


def compute_connectivity_metrics(
    true_signal: np.ndarray,
    predicted_signal: np.ndarray,
    fs: float,
    band: tuple[float, float] = (50.0, 200.0),
    min_duration_sec: float = 60.0,
) -> dict[str, float]:
    """Compare connectivity matrices from true and predicted POOLED signals.

    Call this ONCE on long pooled signals (whole validation set
    concatenated), not on individual segments. Per-segment connectivity is
    too noisy to be meaningful.

    Args:
        true_signal, predicted_signal: (n_channels, n_samples). Should be
            the full pooled validation signal, not a single segment.
        fs: Sampling rate.
        band: Band for the gamma envelope.
        min_duration_sec: Warn if signal is shorter than this; below ~60s
            envelope correlation matrices are dominated by noise.

    Returns:
        Dict with 'pearson_r' (off-diagonal correlation), 'mae', and
        'frobenius_distance'. Returns NaN for pearson_r if n_channels < 3
        (insufficient off-diagonal pairs to compute correlation).
    """
    if true_signal.shape != predicted_signal.shape:
        raise ValueError("Signal shapes must match")
    if true_signal.ndim != 2:
        raise ValueError(
            f"Expected (n_channels, n_samples), got {true_signal.shape}"
        )

    n_channels, n_samples = true_signal.shape
    duration = n_samples / fs
    if duration < min_duration_sec:
        import warnings
        warnings.warn(
            f"Connectivity computed on {duration:.1f}s of signal; "
            f"results are likely noisy below {min_duration_sec}s. "
            f"Pool more segments before calling.",
            stacklevel=2,
        )

    true_conn = envelope_correlation_matrix(true_signal, fs, band)
    pred_conn = envelope_correlation_matrix(predicted_signal, fs, band)

    iu = np.triu_indices(n_channels, k=1)
    true_off = true_conn[iu]
    pred_off = pred_conn[iu]

    mae = float(np.mean(np.abs(true_off - pred_off)))
    frob = float(np.linalg.norm(true_conn - pred_conn, ord="fro"))

    # Pearson over off-diagonal entries needs >= 2 pairs (n_channels >= 3).
    # With 2 channels there's only one pair and correlation is undefined.
    if n_channels < 3:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(true_off, pred_off)[0, 1])

    return {
        "pearson_r": pearson_r,
        "mae": mae,
        "frobenius_distance": frob,
    }


# Kept under old name for any external callers; new code should use
# compute_connectivity_metrics.
connectivity_similarity = compute_connectivity_metrics


# ----------------------------------------------------------------------------
# Aggregate report
# ----------------------------------------------------------------------------


@dataclass
class ReconstructionReport:
    """Bundle of per-segment metrics for a single (true, predicted) pair.

    Per-segment only. Connectivity is NOT included here because per-segment
    connectivity matrices are too noisy to be useful (3-second windows are
    far too short for stable envelope correlations). Use
    `compute_connectivity_metrics` separately on pooled signals.

    Contract: assumes both `true` and `predicted` are z-scored at segment
    or channel level (typical preprocessing). NMSE values across configs
    are only comparable if z-scoring is consistent.
    """

    nmse_total: np.ndarray  # per-channel
    band_nmse: dict[str, np.ndarray]  # band -> per-channel NMSE
    log_spectral_distance: np.ndarray  # per-channel, full band
    log_spectral_distance_gamma: np.ndarray  # per-channel, gamma band only

    def summary(self) -> dict[str, float]:
        """Compact dict suitable for logging to W&B / printing."""
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


def evaluate_reconstruction(
    true: np.ndarray,
    predicted: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    gamma_band: tuple[float, float] = (50.0, 200.0),
) -> ReconstructionReport:
    """Per-segment evaluation. Connectivity must be computed separately
    on pooled signals via `compute_connectivity_metrics`.

    Args:
        true: (n_channels, n_samples) array.
        predicted: (n_channels, n_samples) array, same shape.
        fs: Sampling rate.
        bands: Frequency bands for band-resolved NMSE. Defaults to
            DEFAULT_BANDS.
        gamma_band: Band for the log-spectral-distance gamma metric.

    Returns:
        ReconstructionReport with per-channel metrics.
    """
    if true.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {true.shape} vs {predicted.shape}")
    if true.ndim != 2:
        raise ValueError(
            "Expected 2D (n_channels, n_samples) input. For batched data, "
            "iterate over the batch dimension."
        )

    return ReconstructionReport(
        nmse_total=nmse(true, predicted),
        band_nmse=band_resolved_nmse(true, predicted, fs, bands),
        log_spectral_distance=log_spectral_distance(true, predicted, fs),
        log_spectral_distance_gamma=log_spectral_distance(
            true, predicted, fs, band=gamma_band
        ),
    )
