"""Preprocessing for BrainTreebank sEEG, BaRISTA-matched.

Pipeline (raw h5 → model-ready segments):
    notch filter  →  Laplacian rereference  →  segment  →  z-score

The notch + reref steps are channel-aware and operate on the long
session-level signal. Segmenting and z-scoring are applied per
3-second window.

Channel naming convention (BrainTreebank): ``<stem><N>`` where stem is
an alphabetic prefix and N is an integer suffix (e.g. ``LT3a1``,
``RT2aA12``). Laplacian reref subtracts the mean of the two same-stem
neighbors at index N±1.

All array operations use numpy. Functions are pure: no I/O, no global
state.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.signal


# US power line fundamental + harmonics that BaRISTA notches out.
DEFAULT_NOTCH_FREQS_HZ: Tuple[float, ...] = (60.0, 120.0, 180.0, 240.0, 300.0, 360.0)
DEFAULT_NOTCH_Q: float = 30.0
DEFAULT_FS_HZ: float = 2048.0


# ---------------------------------------------------------------------------
# Notch filter
# ---------------------------------------------------------------------------


def notch_filter(
    data: np.ndarray,
    fs: float = DEFAULT_FS_HZ,
    freqs: Sequence[float] = DEFAULT_NOTCH_FREQS_HZ,
    q: float = DEFAULT_NOTCH_Q,
) -> np.ndarray:
    """Apply IIR notch filters at each frequency along the last axis.

    Matches BaRISTA: ``scipy.signal.iirnotch`` design, applied with
    causal ``lfilter`` (NOT zero-phase ``filtfilt``).

    Parameters
    ----------
    data : np.ndarray
        Shape (..., n_samples). Time on the last axis.
    fs : float
        Sampling rate in Hz.
    freqs : sequence of float
        Frequencies (Hz) at which to place notches.
    q : float
        Quality factor. Higher Q => narrower notch.

    Returns
    -------
    np.ndarray
        Same shape as ``data``.
    """
    out = np.asarray(data, dtype=np.float64)
    nyq = fs / 2.0
    for f in freqs:
        if f <= 0 or f >= nyq:
            raise ValueError(f"notch freq {f} Hz out of (0, nyquist={nyq}) Hz")
        w0 = f / nyq
        b, a = scipy.signal.iirnotch(w0, q)
        out = scipy.signal.lfilter(b, a, out, axis=-1)
    return out


# ---------------------------------------------------------------------------
# Channel name parsing + Laplacian neighbor selection
# ---------------------------------------------------------------------------


_STEM_RE = re.compile(r"^(?P<stem>.+?)(?P<num>\d+)$")
# Markers BaRISTA strips from electrode names before parsing
# (see ``_elec_name_strip`` and ``stem_electrode_name`` in
# barista/data/braintreebank_data_helpers.py). Stripping here makes
# parse_electrode_name idempotent on raw or pre-cleaned inputs and
# eliminates a silent-drop failure mode in build_laplacian_neighbors.
_NAME_MARKERS = ("*", "#", "_")


def parse_electrode_name(name: str) -> Tuple[str, int]:
    """Split an electrode name into (alphabetic stem, integer suffix).

    Strips ``*``, ``#``, ``_`` markers before parsing — matches BaRISTA's
    convention so corrupted/marked channels parse to the same stem as
    their clean counterparts.

    Examples
    --------
    >>> parse_electrode_name("LT3a1")
    ('LT3a', 1)
    >>> parse_electrode_name("RT2aA12")
    ('RT2aA', 12)
    >>> parse_electrode_name("DC10")
    ('DC', 10)
    >>> parse_electrode_name("RT2aA1#")
    ('RT2aA', 1)
    """
    cleaned = name
    for marker in _NAME_MARKERS:
        cleaned = cleaned.replace(marker, "")
    m = _STEM_RE.match(cleaned)
    if m is None:
        raise ValueError(f"could not parse electrode name {name!r}")
    return m.group("stem"), int(m.group("num"))


def build_laplacian_neighbors(
    all_electrodes: Sequence[str],
    excluded: Iterable[str] = (),
) -> Dict[str, Tuple[str, str]]:
    """For each electrode, return its two same-stem neighbors at ±1.

    A channel is included in the output only if BOTH neighbors exist in
    ``all_electrodes`` AND none of (self, either neighbor) are in the
    ``excluded`` set. ``excluded`` typically contains the corrupted
    channels for the subject.

    Parameters
    ----------
    all_electrodes : sequence of str
        Every electrode present in the recording (in any order).
    excluded : iterable of str
        Names to drop (corrupted channels, ref/control chans, etc.).

    Returns
    -------
    dict[str, tuple[str, str]]
        Mapping target electrode -> (neighbor_minus, neighbor_plus).
        Channels not eligible for Laplacian reref are absent from the
        result.
    """
    excluded_set = set(excluded)
    parsed: Dict[Tuple[str, int], str] = {}
    for name in all_electrodes:
        try:
            stem, num = parse_electrode_name(name)
        except ValueError:
            # Non-conforming names (none expected, but be safe) get skipped.
            continue
        parsed[(stem, num)] = name

    neighbors: Dict[str, Tuple[str, str]] = {}
    for (stem, num), name in parsed.items():
        if name in excluded_set:
            continue
        nm = parsed.get((stem, num - 1))
        np_ = parsed.get((stem, num + 1))
        if nm is None or np_ is None:
            continue
        if nm in excluded_set or np_ in excluded_set:
            continue
        neighbors[name] = (nm, np_)
    return neighbors


def apply_laplacian_reref(
    data: np.ndarray,
    electrode_names: Sequence[str],
    neighbors: Dict[str, Tuple[str, str]],
) -> Tuple[np.ndarray, List[str]]:
    """Subtract the mean of each channel's two neighbors from the channel.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples). Rows correspond to
        ``electrode_names`` in order.
    electrode_names : sequence of str
        Channel names matching rows of ``data``.
    neighbors : dict
        Output of :func:`build_laplacian_neighbors`. Only channels in
        this dict are kept.

    Returns
    -------
    reref : np.ndarray
        Shape (n_kept, n_samples). Rereferenced signal.
    kept_names : list of str
        Names of the channels in ``reref``, in row order.
    """
    if data.ndim != 2:
        raise ValueError(f"expected (n_channels, n_samples), got shape {data.shape}")
    if data.shape[0] != len(electrode_names):
        raise ValueError(
            f"data has {data.shape[0]} channels but {len(electrode_names)} names"
        )

    name_to_idx = {n: i for i, n in enumerate(electrode_names)}
    kept_names = [n for n in electrode_names if n in neighbors]
    out = np.empty((len(kept_names), data.shape[1]), dtype=data.dtype)
    for row, name in enumerate(kept_names):
        nm, np_ = neighbors[name]
        i = name_to_idx[name]
        i_m = name_to_idx[nm]
        i_p = name_to_idx[np_]
        ref = 0.5 * (data[i_m] + data[i_p])
        out[row] = data[i] - ref
    return out, kept_names


# ---------------------------------------------------------------------------
# Segmentation + z-scoring
# ---------------------------------------------------------------------------


def segment_signal(
    data: np.ndarray,
    segment_samples: int,
    step_samples: int | None = None,
) -> np.ndarray:
    """Slice a long signal into non-overlapping (or strided) segments.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples).
    segment_samples : int
        Length of each segment in samples (e.g. 6144 for 3 s @ 2048 Hz).
    step_samples : int, optional
        Stride between segment starts. Defaults to ``segment_samples``
        (non-overlapping).

    Returns
    -------
    np.ndarray
        Shape (n_segments, n_channels, segment_samples).
    """
    if data.ndim != 2:
        raise ValueError(f"expected (n_channels, n_samples), got shape {data.shape}")
    if step_samples is None:
        step_samples = segment_samples
    n_chan, n_samp = data.shape
    if n_samp < segment_samples:
        return np.empty((0, n_chan, segment_samples), dtype=data.dtype)
    n_seg = 1 + (n_samp - segment_samples) // step_samples
    out = np.empty((n_seg, n_chan, segment_samples), dtype=data.dtype)
    for k in range(n_seg):
        s = k * step_samples
        out[k] = data[:, s : s + segment_samples]
    return out


def zscore_segment(segments: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score each (segment, channel) row independently.

    Parameters
    ----------
    segments : np.ndarray
        Shape (..., n_samples). Last axis is time.
    eps : float
        Added to std to avoid div-by-zero on flat channels.

    Returns
    -------
    np.ndarray
        Same shape as input. Per-row mean ~ 0, std ~ 1.
    """
    mean = segments.mean(axis=-1, keepdims=True)
    std = segments.std(axis=-1, keepdims=True)
    return (segments - mean) / (std + eps)
