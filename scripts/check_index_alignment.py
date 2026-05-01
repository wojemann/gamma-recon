"""Sanity-check segment indices against the source transcripts.

Round-trip verifies the time -> sample mapping that built the parquets:

  1. For each sentence_onset / speech_nonspeech / volume / optical_flow row
     with ``source_word_idx >= 0``, re-run trigger interpolation on the
     transcript's ``start`` time and check it equals ``center_sample``.
  2. Verify ``end_sample - start_sample == 6144`` and
     ``center_sample - start_sample == 3072`` for all positives.
  3. Verify all ``[start_sample, end_sample)`` ranges fall inside the
     timing-CSV-defined valid window for the trial.
  4. For a few sentence_onset positives, list neighboring word starts
     in time order — eyeball whether the surrounding word density is
     consistent with a sentence boundary (sparser before, denser after).

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u -m scripts.check_index_alignment --profile local
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs.config import get_config


SUBJECT_ID = 2
SEGMENT_SAMPLES = 6144
HALF_SEG = SEGMENT_SAMPLES // 2


def load_t2s(braintree_root: Path, trial_id: int):
    p = (
        braintree_root / "subject_timings"
        / f"sub_{SUBJECT_ID}_trial{trial_id:03d}_timings.csv"
    )
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


def load_transcript(braintree_root: Path, trial_id: int) -> pd.DataFrame:
    meta_p = (
        braintree_root / "subject_metadata"
        / f"sub_{SUBJECT_ID}_trial{trial_id:03d}_metadata.json"
    )
    with open(meta_p) as f:
        movie = json.load(f)["filename"]
    return pd.read_csv(
        braintree_root / "transcripts" / movie / "features.csv",
        low_memory=False,
    )


def check_trial(
    braintree_root: Path,
    seg_root: Path,
    trial_id: int,
    n_sample_rows: int = 5,
) -> dict:
    print(f"\n--- trial{trial_id:03d} ---")
    t2s, (valid_start, valid_end) = load_t2s(braintree_root, trial_id)
    transcript = load_transcript(braintree_root, trial_id)

    out: dict = {"valid_window": [valid_start, valid_end]}
    for task in ("sentence_onset", "speech_nonspeech", "volume", "optical_flow"):
        path = seg_root / "sub_02" / f"trial{trial_id:03d}__{task}.parquet"
        df = pd.read_parquet(path)

        # All windows in valid range, fixed length.
        assert (df["end_sample"] - df["start_sample"] == SEGMENT_SAMPLES).all()
        in_range = ((df["start_sample"] >= valid_start)
                    & (df["end_sample"] <= valid_end)).all()
        assert in_range, f"{task}: row outside valid window"

        pos = df[df["source_word_idx"] >= 0]
        # center - start == 3072 for every positive
        assert (pos["center_sample"] - pos["start_sample"] == HALF_SEG).all(), \
            f"{task}: center/start offset mismatch"

        # Round-trip every positive's center_sample.
        widx = pos["source_word_idx"].to_numpy()
        word_starts = transcript.loc[widx, "start"].to_numpy(dtype=np.float64)
        recomputed = t2s(word_starts)
        diff = recomputed - pos["center_sample"].to_numpy()
        max_abs = int(np.abs(diff).max()) if len(diff) else 0
        assert max_abs == 0, f"{task}: round-trip diff up to {max_abs} samples"

        n_pos = len(pos)
        n_neg = (df["label"] == 0).sum()
        n_total = len(df)
        print(f"  {task:20s}  total={n_total:>4d}  pos={n_pos:>4d}  neg={n_neg:>4d}  "
              f"round-trip max|diff|={max_abs}")
        out[task] = {"total": int(n_total), "n_pos": int(n_pos), "n_neg": int(n_neg)}

    # Eyeball: a sentence-onset positive should be preceded by a word-density
    # gap (or the trial start) and followed by a relatively dense word region.
    sent_path = seg_root / "sub_02" / f"trial{trial_id:03d}__sentence_onset.parquet"
    df = pd.read_parquet(sent_path)
    pos = df[df["source_word_idx"] >= 0].sort_values("center_sample").reset_index(drop=True)
    fs = 2048.0
    print(f"\n  spot-check {min(n_sample_rows, len(pos))} sentence_onset positives:")
    print(f"  {'word_idx':>8s}  {'t_word_s':>10s}  {'gap_before_s':>14s}  {'gap_after_s':>14s}")
    for i in range(min(n_sample_rows, len(pos))):
        widx = int(pos.iloc[i]["source_word_idx"])
        t_word = float(transcript.loc[widx, "start"])
        if widx > 0:
            prev = float(transcript.loc[widx - 1, "end"])
            gap_b = t_word - prev
        else:
            gap_b = float("nan")
        if widx + 1 < len(transcript):
            nxt = float(transcript.loc[widx + 1, "start"])
            gap_a = nxt - t_word
        else:
            gap_a = float("nan")
        print(f"  {widx:>8d}  {t_word:>10.3f}  {gap_b:>14.3f}  {gap_a:>14.3f}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="local", choices=["local", "server"])
    args = parser.parse_args()

    cfg = get_config(args.profile)
    bt_root = Path(cfg.braintree_data_root)
    seg_root = Path(__file__).resolve().parents[1] / "segment_indices"

    summary: dict = {}
    for trial_id in range(7):
        summary[f"trial{trial_id:03d}"] = check_trial(bt_root, seg_root, trial_id)

    print("\nAll checks passed: bounds, fixed length, center alignment, round-trip mapping.")


if __name__ == "__main__":
    main()
