from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import signal


BAND_DEFS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta": (15.0, 30.0),
    "gamma": (30.0, 200.0),
    "broadband": (1.0, 200.0),
}


@dataclass
class TrialData:
    trial_id: int
    h5_path: Path
    windows: pd.DataFrame
    fs: float


def parse_trial_id(path: Path) -> int:
    match = re.search(r"trial(\d{3})", path.name)
    if not match:
        raise ValueError(f"Unable to parse trial id from {path}")
    return int(match.group(1))


def list_trial_h5_paths(root: Path, subject: str) -> List[Path]:
    paths = sorted(root.glob(f"{subject}_trial*.h5"))
    if not paths:
        raise FileNotFoundError(f"No H5 trials found for {subject} in {root}")
    return paths


def load_temporal_channel_info(root: Path, subject: str) -> pd.DataFrame:
    loc = pd.read_csv(root / "localization" / "elec_coords_full.csv")
    loc_sub = loc[
        (loc["Subject"] == subject)
        & (loc["Region"].astype(str).str.contains("temporal", case=False, na=False))
    ].copy()

    with open(root / "electrode_labels" / subject / "electrode_labels.json", "r", encoding="utf-8") as f:
        electrode_labels = json.load(f)
    with open(root / "corrupted_elec.json", "r", encoding="utf-8") as f:
        corrupted = set(json.load(f).get(subject, []))

    label_to_idx = {label: i for i, label in enumerate(electrode_labels)}
    loc_sub = loc_sub[loc_sub["Electrode"].isin(label_to_idx)].copy()
    loc_sub = loc_sub[~loc_sub["Electrode"].isin(corrupted)].copy()
    loc_sub["dataset_name"] = loc_sub["Electrode"].map(lambda e: f"electrode_{label_to_idx[e]}")
    loc_sub = loc_sub.drop_duplicates("dataset_name")
    return loc_sub[["dataset_name", "Electrode", "Region"]].reset_index(drop=True)


def build_trial_windows(
    root: Path,
    subject: str,
    trial_id: int,
    control_sec: float,
    onset_start_sec: float,
    onset_end_sec: float,
) -> TrialData:
    h5_path = root / f"{subject}_trial{trial_id:03d}.h5"
    meta_path = root / "subject_metadata" / f"{subject}_trial{trial_id:03d}_metadata.json"
    timings_path = root / "subject_timings" / f"{subject}_trial{trial_id:03d}_timings.csv"

    with open(meta_path, "r", encoding="utf-8") as f:
        movie_name = json.load(f)["filename"]
    transcript_path = root / "transcripts" / movie_name / "features.csv"

    transcript = pd.read_csv(transcript_path)
    transcript["start"] = pd.to_numeric(transcript["start"], errors="coerce")
    transcript["end"] = pd.to_numeric(transcript["end"], errors="coerce")
    transcript["is_onset"] = pd.to_numeric(transcript["is_onset"], errors="coerce").fillna(0)
    transcript["sentence_idx"] = pd.to_numeric(transcript["sentence_idx"], errors="coerce")

    onsets = (
        transcript[transcript["is_onset"] == 1.0]
        .sort_values(["sentence_idx", "start"])
        .drop_duplicates(subset=["sentence_idx"], keep="first")
        .dropna(subset=["start", "sentence_idx"])
        .reset_index(drop=True)
    )
    sentence_bounds = (
        transcript.groupby("sentence_idx", as_index=False)
        .agg(sentence_end=("end", "max"))
        .sort_values("sentence_idx")
        .reset_index(drop=True)
    )
    sentence_bounds["prev_sentence_end"] = sentence_bounds["sentence_end"].shift(1)
    onsets = onsets.merge(
        sentence_bounds[["sentence_idx", "prev_sentence_end"]],
        on="sentence_idx",
        how="left",
    )

    timings = pd.read_csv(timings_path)
    triggers = timings[timings["type"] == "trigger"].copy()
    triggers["movie_time"] = pd.to_numeric(triggers["movie_time"], errors="coerce")
    triggers["start_time"] = pd.to_numeric(triggers["start_time"], errors="coerce")
    triggers["index"] = pd.to_numeric(triggers["index"], errors="coerce")
    triggers = triggers.dropna(subset=["movie_time", "start_time", "index"]).sort_values("movie_time")

    movie_time = triggers["movie_time"].to_numpy()
    record_time = triggers["start_time"].to_numpy()
    sample_index = triggers["index"].to_numpy()
    fs = (sample_index[-1] - sample_index[0]) / (record_time[-1] - record_time[0])

    def movie_to_sample(t: float) -> float:
        return float(np.interp(t, movie_time, sample_index))

    with h5py.File(h5_path, "r") as h5f:
        n_samples = h5f["data"]["electrode_0"].shape[0]

    rows: List[dict] = []
    for _, r in onsets.iterrows():
        onset_movie = float(r["start"])
        if onset_movie < movie_time[0] or onset_movie > movie_time[-1]:
            continue
        onset_samp = movie_to_sample(onset_movie)
        ctrl_start = int(round(onset_samp - control_sec * fs))
        ctrl_end = int(round(onset_samp))
        on_start = int(round(onset_samp + onset_start_sec * fs))
        on_end = int(round(onset_samp + onset_end_sec * fs))
        if ctrl_start < 0 or on_end > n_samples:
            continue

        prev_end_movie = r["prev_sentence_end"]
        if pd.notna(prev_end_movie):
            prev_end_samp = movie_to_sample(float(prev_end_movie))
            if ctrl_start < prev_end_samp:
                continue

        rows.append(
            {
                "trial_id": trial_id,
                "sentence_idx": int(r["sentence_idx"]),
                "onset_movie_time": onset_movie,
                "ctrl_start": ctrl_start,
                "ctrl_end": ctrl_end,
                "on_start": on_start,
                "on_end": on_end,
            }
        )

    return TrialData(trial_id=trial_id, h5_path=h5_path, windows=pd.DataFrame(rows), fs=fs)


def notch_only_full_trace(x: np.ndarray, fs: float, notch_freqs: Tuple[float, ...], notch_q: float) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    for nf in notch_freqs:
        b, a = signal.iirnotch(w0=nf, Q=notch_q, fs=fs)
        y = signal.filtfilt(b, a, y)
    return y


def bandpass_full_trace(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)

