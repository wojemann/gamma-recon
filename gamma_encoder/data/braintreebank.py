"""BrainTreebank trial loader for the gamma_encoder framework.

Wraps the on-disk BrainTreebank layout (HDF5 per trial + JSON labels +
CSV localization) in a class that yields preprocessed, segmented,
z-scored 3-second windows ready to feed the model.

Pipeline applied to each trial:

    h5 raw  ->  drop corrupted + atlas-missing chans
            ->  build Laplacian neighbor map
            ->  notch filter (whole-trial)
            ->  Laplacian rereference
            ->  segment into 3 s windows
            ->  z-score per (segment, channel)

Channel identity threading: BaRISTA's ``electrode_labels.json`` decorates
some names with marker characters (``*``, ``#``, ``_``) that don't appear
in the atlas. We sanitize before joining (matching their convention).

Region identity: each kept channel gets an integer region ID via a
project-stable mapping (``REGION_VOCAB`` constructed lazily, persisted
to disk by the caller if needed) over the Destrieux atlas column.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from gamma_encoder.data.preprocess import (
    DEFAULT_FS_HZ,
    apply_laplacian_reref,
    build_laplacian_neighbors,
    notch_filter,
    segment_signal,
    zscore_segment,
)


# ---------------------------------------------------------------------------
# Name sanitization
# ---------------------------------------------------------------------------


def sanitize_electrode_name(name: str) -> str:
    """Strip BaRISTA marker characters from an electrode label.

    The BrainTreebank ``electrode_labels.json`` files include ``*`` and
    ``#`` markers (and occasional ``_``) that are NOT present in the
    localization CSVs. BaRISTA's loader strips these before joining.
    See ``barista/data/braintreebank_data_helpers.py::_elec_name_strip``.
    """
    return name.replace("*", "").replace("#", "").replace("_", "")


# ---------------------------------------------------------------------------
# Region vocabulary
# ---------------------------------------------------------------------------


def build_region_vocab(region_names: Sequence[str]) -> Dict[str, int]:
    """Stable Destrieux-name -> integer mapping.

    Sorted lexicographically so the mapping is deterministic across runs
    given the same input set. The empty string / NaN gets reserved id 0
    (treated as "unknown") if present.
    """
    unique = sorted(set(region_names))
    vocab: Dict[str, int] = {}
    if "" in unique:
        vocab[""] = 0
        unique = [r for r in unique if r != ""]
    next_id = 1 if "" in vocab else 0
    for r in unique:
        vocab[r] = next_id
        next_id += 1
    return vocab


# ---------------------------------------------------------------------------
# Trial loader
# ---------------------------------------------------------------------------


@dataclass
class TrialSegments:
    """Output of :meth:`BrainTreeTrial.load_segments`."""

    segments: np.ndarray       # (n_segments, n_channels, segment_samples)
    region_ids: np.ndarray     # (n_channels,) int64
    channel_names: List[str]   # sanitized; length n_channels
    fs: float
    segment_samples: int


class BrainTreeTrial:
    """One BrainTreebank trial, ready for preprocessing.

    Construction is cheap — it only reads the small JSON / CSV metadata
    files and builds the neighbor + atlas maps. The h5 is opened only
    when ``load_segments`` is called.

    Parameters
    ----------
    subject_id : int
        Subject number (e.g. 2 for sub_2).
    trial_id : int
        Trial number (e.g. 0 for trial000).
    data_root : Path
        Root of the BrainTreebank dataset on disk
        (e.g. ``/Users/wojemann/local_data/BrainTree``).
    fs : float
        Sampling rate in Hz. Defaults to 2048 (BrainTreebank native).
    segment_seconds : float
        Length of each output segment.
    region_vocab : dict, optional
        Pre-built Destrieux-region -> int mapping. If None, a vocab is
        built from this trial's regions only (use the cross-trial helper
        in calling code if you need a stable shared vocab).
    """

    def __init__(
        self,
        subject_id: int,
        trial_id: int,
        data_root: Path,
        fs: float = DEFAULT_FS_HZ,
        segment_seconds: float = 3.0,
        region_vocab: Optional[Dict[str, int]] = None,
    ) -> None:
        self.subject_id = int(subject_id)
        self.trial_id = int(trial_id)
        self.data_root = Path(data_root)
        self.fs = float(fs)
        self.segment_samples = int(round(self.fs * segment_seconds))
        self._explicit_vocab = region_vocab is not None

        # Resolve paths.
        self.h5_path = self.data_root / f"sub_{self.subject_id}_trial{self.trial_id:03d}.h5"
        labels_path = (
            self.data_root
            / "electrode_labels"
            / f"sub_{self.subject_id}"
            / "electrode_labels.json"
        )
        atlas_path = (
            self.data_root
            / "localization"
            / f"sub_{self.subject_id}"
            / "depth-wm.csv"
        )
        corrupted_path = self.data_root / "corrupted_elec.json"

        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        if not labels_path.exists():
            raise FileNotFoundError(labels_path)
        if not atlas_path.exists():
            raise FileNotFoundError(atlas_path)
        if not corrupted_path.exists():
            raise FileNotFoundError(corrupted_path)

        # Load raw labels (one entry per h5 electrode_N) and sanitize.
        with open(labels_path) as f:
            raw_labels: List[str] = json.load(f)
        self._raw_labels = raw_labels
        self.electrode_names = [sanitize_electrode_name(n) for n in raw_labels]

        # Load corrupted set (also sanitized for consistency).
        with open(corrupted_path) as f:
            corrupted_all = json.load(f)
        sub_key = f"sub_{self.subject_id}"
        corrupted_raw = corrupted_all.get(sub_key, [])
        self.corrupted = {sanitize_electrode_name(n) for n in corrupted_raw}

        # Load atlas; build name -> Destrieux dict.
        atlas_df = pd.read_csv(atlas_path)
        atlas_df["Electrode"] = atlas_df["Electrode"].map(sanitize_electrode_name)
        self.region_by_name: Dict[str, str] = dict(
            zip(atlas_df["Electrode"], atlas_df["Destrieux"].fillna(""))
        )

        # Build the keep-set: present in h5, not corrupted, atlas-resolvable,
        # and Laplacian-eligible (both ±1 stem-neighbors present and clean).
        excluded = set(self.corrupted)
        # Also exclude any chan that has no atlas entry: we need a region id.
        for n in self.electrode_names:
            if n not in self.region_by_name:
                excluded.add(n)
        self.neighbors = build_laplacian_neighbors(self.electrode_names, excluded=excluded)
        self.kept_names = [n for n in self.electrode_names if n in self.neighbors]

        # Region ids.
        if region_vocab is None:
            kept_regions = [self.region_by_name[n] for n in self.kept_names]
            self.region_vocab = build_region_vocab(kept_regions)
        else:
            self.region_vocab = dict(region_vocab)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Cheap summary: counts only, no h5 read."""
        return {
            "subject_id": self.subject_id,
            "trial_id": self.trial_id,
            "n_raw_chans": len(self.electrode_names),
            "n_corrupted": len(self.corrupted),
            "n_kept": len(self.kept_names),
            "n_unique_regions": len(set(self.region_vocab.values())),
            "h5_path": str(self.h5_path),
        }

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _read_raw_kept(
        self,
        max_seconds: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Read every h5 channel needed for Laplacian reref of kept_names.

        We need not just kept_names but also their neighbors (which may
        themselves be kept or not). This loads the union into a single
        (n_loaded, n_samples) array and returns it alongside the row
        names.
        """
        needed: set[str] = set(self.kept_names)
        for k, (nm, np_) in self.neighbors.items():
            needed.add(nm)
            needed.add(np_)

        # Map sanitized name -> h5 electrode index (electrode_N).
        # raw_labels and electrode_names are aligned by index.
        idx_by_name: Dict[str, int] = {}
        for i, n in enumerate(self.electrode_names):
            # First occurrence wins; sanitization can in principle alias
            # but in practice names are unique post-strip on sub_2.
            idx_by_name.setdefault(n, i)

        load_names = sorted(needed, key=lambda n: idx_by_name[n])
        with h5py.File(self.h5_path, "r") as f:
            grp = f["data"]
            sample_count = grp[f"electrode_{idx_by_name[load_names[0]]}"].shape[0]
            if max_seconds is not None:
                stop = min(sample_count, int(round(max_seconds * self.fs)))
            else:
                stop = sample_count
            out = np.empty((len(load_names), stop), dtype=np.float64)
            for row, name in enumerate(load_names):
                ds = grp[f"electrode_{idx_by_name[name]}"]
                out[row] = ds[:stop]
        return out, load_names

    def load_segments(
        self,
        max_seconds: Optional[float] = None,
        notch: bool = True,
    ) -> TrialSegments:
        """Run the full preprocessing pipeline and return segments.

        Parameters
        ----------
        max_seconds : float, optional
            If given, only the first ``max_seconds`` of the trial are
            loaded and processed. Useful for overfit-batch caching.
        notch : bool
            Apply the BaRISTA notch filter chain. Defaults to True.

        Returns
        -------
        TrialSegments
        """
        raw, load_names = self._read_raw_kept(max_seconds=max_seconds)

        if notch:
            raw = notch_filter(raw, fs=self.fs)

        reref, kept_names = apply_laplacian_reref(raw, load_names, self.neighbors)
        # kept_names should match self.kept_names but reorder to enforce
        # the canonical order.
        order = [kept_names.index(n) for n in self.kept_names]
        reref = reref[order]

        segments = segment_signal(reref, segment_samples=self.segment_samples)
        segments = zscore_segment(segments)

        region_ids = np.array(
            [self.region_vocab[self.region_by_name[n]] for n in self.kept_names],
            dtype=np.int64,
        )

        return TrialSegments(
            segments=segments,
            region_ids=region_ids,
            channel_names=list(self.kept_names),
            fs=self.fs,
            segment_samples=self.segment_samples,
        )
