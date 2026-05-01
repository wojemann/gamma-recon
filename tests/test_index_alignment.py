"""Round-trip + bounds checks for the committed segment_indices parquets.

Confirms the time->sample mapping that built the parquets is invertible:
``t2s(features.start[source_word_idx]) == center_sample`` for every
positive across all 7 trials x 4 main tasks. Also checks fixed window
length, center-offset, and that windows fall inside the trial's valid
range. Skips when the BrainTreebank data tree isn't mounted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SEG_ROOT = REPO_ROOT / "segment_indices" / "sub_02"
BT_ROOT = Path("/Users/wojemann/local_data/BrainTree")

SUBJECT_ID = 2
SEGMENT_SAMPLES = 6144
HALF_SEG = SEGMENT_SAMPLES // 2
TRIAL_IDS = list(range(7))
TASKS = ("sentence_onset", "speech_nonspeech", "volume", "optical_flow")

pytestmark = pytest.mark.skipif(
    not BT_ROOT.exists() or not SEG_ROOT.exists(),
    reason="Real BrainTreebank tree + segment_indices parquets required",
)


def _load_t2s(trial_id: int):
    p = BT_ROOT / "subject_timings" / f"sub_{SUBJECT_ID}_trial{trial_id:03d}_timings.csv"
    df = pd.read_csv(p)
    movie_time = df["movie_time"].to_numpy(dtype=np.float64)
    sample_idx = df["index"].to_numpy(dtype=np.float64)
    order = np.argsort(movie_time)
    movie_time = movie_time[order]
    sample_idx = sample_idx[order]

    def f(t):
        t = np.asarray(t, dtype=np.float64)
        return np.rint(np.interp(t, movie_time, sample_idx)).astype(np.int64)

    beg = int(df[df["type"] == "beginning"].iloc[0]["index"])
    end = int(df[df["type"] == "end"].iloc[0]["index"])
    return f, (beg, end)


def _load_transcript(trial_id: int) -> pd.DataFrame:
    meta_p = BT_ROOT / "subject_metadata" / f"sub_{SUBJECT_ID}_trial{trial_id:03d}_metadata.json"
    with open(meta_p) as f:
        movie = json.load(f)["filename"]
    return pd.read_csv(
        BT_ROOT / "transcripts" / movie / "features.csv", low_memory=False,
    )


@pytest.mark.parametrize("trial_id", TRIAL_IDS)
@pytest.mark.parametrize("task", TASKS)
def test_window_length_and_center_offset(trial_id, task):
    path = SEG_ROOT / f"trial{trial_id:03d}__{task}.parquet"
    df = pd.read_parquet(path)
    assert (df["end_sample"] - df["start_sample"] == SEGMENT_SAMPLES).all()
    pos = df[df["source_word_idx"] >= 0]
    assert (pos["center_sample"] - pos["start_sample"] == HALF_SEG).all()


@pytest.mark.parametrize("trial_id", TRIAL_IDS)
@pytest.mark.parametrize("task", TASKS)
def test_windows_within_valid_range(trial_id, task):
    _, (valid_start, valid_end) = _load_t2s(trial_id)
    path = SEG_ROOT / f"trial{trial_id:03d}__{task}.parquet"
    df = pd.read_parquet(path)
    assert (df["start_sample"] >= valid_start).all()
    assert (df["end_sample"] <= valid_end).all()


@pytest.mark.parametrize("trial_id", TRIAL_IDS)
@pytest.mark.parametrize("task", TASKS)
def test_center_sample_round_trip(trial_id, task):
    """center_sample must equal t2s(features.start[source_word_idx]) exactly."""
    t2s, _ = _load_t2s(trial_id)
    transcript = _load_transcript(trial_id)
    path = SEG_ROOT / f"trial{trial_id:03d}__{task}.parquet"
    df = pd.read_parquet(path)
    pos = df[df["source_word_idx"] >= 0]
    if len(pos) == 0:
        pytest.skip(f"no positives in {path.name}")
    word_starts = transcript.loc[pos["source_word_idx"].to_numpy(), "start"].to_numpy(dtype=np.float64)
    recomputed = t2s(word_starts)
    diff = recomputed - pos["center_sample"].to_numpy()
    assert int(np.abs(diff).max()) == 0, f"max|diff|={int(np.abs(diff).max())}"


def test_pretrain_parquet_well_formed():
    """Smoke for pretrain (covered separately since it has no positives)."""
    for trial_id in TRIAL_IDS:
        path = SEG_ROOT / f"trial{trial_id:03d}__pretrain.parquet"
        df = pd.read_parquet(path)
        assert (df["end_sample"] - df["start_sample"] == SEGMENT_SAMPLES).all()
        # Pretrain rows are unlabeled tiles; source_word_idx should be -1.
        assert (df["source_word_idx"] == -1).all()
