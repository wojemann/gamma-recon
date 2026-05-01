"""Pre-compute segment-index tables for sub_2 (BrainTreebank).

Generates lookup tables of (start_sample, end_sample) ranges into the
sub_2 trial h5 files for two evaluation schemes:

  1. ``pretrain``: non-overlapping 3-s windows tiling the valid recording
     window (between the timing CSV's ``beginning`` and ``end`` markers).
     Random 80/10/10 train/valid/test split with seed 42. The same tiling
     doubles as the channel-reconstruction candidate pool.

  2. ``main`` downstream (Appendix A protocol). For four binary tasks:
     - sentence_onset       (positives = is_onset==1)
     - speech_nonspeech     (positives = every word)
     - volume               (top vs. bottom quartile of `rms`)
     - optical_flow         (top vs. bottom quartile of `max_global_magnitude`)
     Positives are 3-s windows centered on the relevant word/onset. Negatives
     are 3-s windows that contain *no* word interval (BaRISTA's stricter
     definition; PopT only required the center 1 s clean). Non-overlap is
     enforced greedily, kept-positives win over candidate-negatives, classes
     are balanced (subsample the larger), then split 80/10/10 random.

Output: one parquet per (trial, task) under ``segment_indices/sub_02/`` plus
``segment_indices/manifest.json`` summarising counts and seeds.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -m scripts.build_segment_indices --profile local
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from configs.config import get_config


FS = 2048
SEGMENT_S = 3.0
SEGMENT_SAMPLES = int(SEGMENT_S * FS)  # 6144
HALF_SEG = SEGMENT_SAMPLES // 2  # 3072
SUBJECT_ID = 2
TRIAL_IDS = list(range(7))  # sub_2 has trials 000..006
SEED_PRETRAIN = 42
SEED_MAIN = 42

# CLAUDE.md session-role mapping for sub_2 (BaRISTA Appendix Table 5).
SESSION_ROLE = {
    0: "pretrain", 1: "pretrain", 2: "pretrain", 3: "pretrain", 4: "pretrain",
    5: "downstream_val", 6: "downstream_test",
}

OPTICAL_FLOW_COL = "max_global_magnitude"


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_timings(braintree_root: Path, trial_id: int) -> pd.DataFrame:
    p = braintree_root / "subject_timings" / f"sub_{SUBJECT_ID}_trial{trial_id:03d}_timings.csv"
    return pd.read_csv(p)


def load_metadata(braintree_root: Path, trial_id: int) -> dict:
    p = braintree_root / "subject_metadata" / f"sub_{SUBJECT_ID}_trial{trial_id:03d}_metadata.json"
    with open(p) as f:
        return json.load(f)


def load_transcript(braintree_root: Path, movie_filename: str) -> pd.DataFrame:
    p = braintree_root / "transcripts" / movie_filename / "features.csv"
    df = pd.read_csv(p, low_memory=False)
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "row_idx"})
    return df


def valid_window(timings: pd.DataFrame) -> Tuple[int, int]:
    beg = timings[timings["type"] == "beginning"].iloc[0]
    end = timings[timings["type"] == "end"].iloc[0]
    return int(beg["index"]), int(end["index"])


def time_to_sample_fn(timings: pd.DataFrame):
    """Return f(t_seconds) -> h5 sample index, via interpolation against triggers.

    Uses every row of the timings table (beginning, all triggers, end) since
    `index` is monotonic in `movie_time` and the trigger-density (~12 Hz) is
    more than fine enough to absorb clock drift below sample resolution.
    """
    movie_time = timings["movie_time"].to_numpy(dtype=np.float64)
    sample_idx = timings["index"].to_numpy(dtype=np.float64)
    order = np.argsort(movie_time)
    movie_time = movie_time[order]
    sample_idx = sample_idx[order]

    def f(t):
        t = np.asarray(t, dtype=np.float64)
        out = np.full(t.shape, -1, dtype=np.int64)
        ok = np.isfinite(t)
        if ok.any():
            interp = np.rint(np.interp(t[ok], movie_time, sample_idx))
            out[ok] = interp.astype(np.int64)
        return out

    return f


# -----------------------------------------------------------------------------
# Window construction primitives
# -----------------------------------------------------------------------------

def pretrain_tiling(start_valid: int, end_valid: int) -> np.ndarray:
    """Non-overlapping 3-s window starts in [start_valid, end_valid - SEGMENT_SAMPLES]."""
    starts = np.arange(start_valid, end_valid - SEGMENT_SAMPLES + 1, SEGMENT_SAMPLES, dtype=np.int64)
    return starts


def centered_window(center_sample: np.ndarray, valid: Tuple[int, int]) -> np.ndarray:
    """3-s window starts centered on `center_sample`, masked to fit in valid window."""
    start = center_sample - HALF_SEG
    end = start + SEGMENT_SAMPLES
    keep = (start >= valid[0]) & (end <= valid[1])
    return start.astype(np.int64), keep


def windows_with_no_speech(
    starts: np.ndarray,
    word_intervals: np.ndarray,
) -> np.ndarray:
    """Return the boolean mask of `starts` whose [start, start+L) contains NO word interval.

    `word_intervals` is shape (N, 2) of (word_start_sample, word_end_sample), exclusive end.
    Definition (BaRISTA Appendix A, stricter than PopT): the *entire* 3-s window must be
    speech-free.

    Implementation: a window [a, b) is speech-free iff there is no word interval
    [ws, we) with ws < b and we > a. Use sorted-search on word ends.
    """
    if word_intervals.size == 0:
        return np.ones(len(starts), dtype=bool)

    word_intervals = word_intervals[word_intervals[:, 0].argsort()]
    ws = word_intervals[:, 0]
    we = word_intervals[:, 1]

    a = starts
    b = starts + SEGMENT_SAMPLES

    # First word with ws >= b: that's outside the window. The candidate
    # words to check are those with index < idx_first_after_b.
    idx_first_after_b = np.searchsorted(ws, b, side="left")
    # Of words with ws < b, the window has speech iff any has we > a.
    # Equivalently: max(we[: idx_first_after_b]) > a.
    we_running_max = np.maximum.accumulate(we)
    # Append a leading -inf so we can index by idx_first_after_b - 1 safely.
    safe_max = np.concatenate([[np.iinfo(np.int64).min], we_running_max])
    max_we_before_b = safe_max[idx_first_after_b]  # idx 0 -> -inf
    has_speech = max_we_before_b > a
    return ~has_speech


def greedy_non_overlap(starts: np.ndarray) -> np.ndarray:
    """Greedy chronological keep-first non-overlap.

    Returns the indices into `starts` (already-sorted) that are kept.
    Two windows overlap iff their start gap is < SEGMENT_SAMPLES.
    """
    if len(starts) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(starts, kind="stable")
    sorted_starts = starts[order]
    kept = [0]
    last_end = sorted_starts[0] + SEGMENT_SAMPLES
    for i in range(1, len(sorted_starts)):
        if sorted_starts[i] >= last_end:
            kept.append(i)
            last_end = sorted_starts[i] + SEGMENT_SAMPLES
    return order[np.array(kept, dtype=np.int64)]


def drop_overlapping_with(starts: np.ndarray, blockers: np.ndarray) -> np.ndarray:
    """Boolean mask over `starts` keeping only those that don't overlap any window in `blockers`."""
    if len(blockers) == 0 or len(starts) == 0:
        return np.ones(len(starts), dtype=bool)
    blockers = np.sort(blockers)
    a = starts
    b = starts + SEGMENT_SAMPLES
    # Find the first blocker with start >= b (out of the window). Candidate
    # blockers are everything before that. Among them, the window is OK iff
    # max(blocker_start + L) <= a, i.e., max blocker_end <= a.
    idx_first_after_b = np.searchsorted(blockers, b, side="left")
    # max blocker end seen so far
    blocker_ends = blockers + SEGMENT_SAMPLES
    running_max = np.maximum.accumulate(blocker_ends)
    safe_max = np.concatenate([[np.iinfo(np.int64).min], running_max])
    max_end_before_b = safe_max[idx_first_after_b]
    overlaps = max_end_before_b > a
    return ~overlaps


# -----------------------------------------------------------------------------
# Split helpers
# -----------------------------------------------------------------------------

def random_split_80_10_10(n: int, seed: int) -> np.ndarray:
    """Return a length-n array of strings ('train'/'valid'/'test')."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * 0.8))
    n_valid = int(round(n * 0.1))
    # n_test gets the remainder so counts always sum to n.
    splits = np.empty(n, dtype=object)
    splits[perm[:n_train]] = "train"
    splits[perm[n_train:n_train + n_valid]] = "valid"
    splits[perm[n_train + n_valid:]] = "test"
    return splits


def class_balance_indices(n_pos: int, n_neg: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample the larger class uniformly; return kept index arrays into the per-class lists."""
    rng = np.random.default_rng(seed)
    k = min(n_pos, n_neg)
    if n_pos > k:
        pos_keep = np.sort(rng.choice(n_pos, size=k, replace=False))
    else:
        pos_keep = np.arange(n_pos)
    if n_neg > k:
        neg_keep = np.sort(rng.choice(n_neg, size=k, replace=False))
    else:
        neg_keep = np.arange(n_neg)
    return pos_keep, neg_keep


# -----------------------------------------------------------------------------
# Frame builders
# -----------------------------------------------------------------------------

BASE_COLUMNS = [
    "session_id", "subject_id", "trial_id", "movie", "task", "session_role",
    "split", "split_seed",
    "start_sample", "end_sample",
    "label", "center_sample", "source_word_idx", "notes",
]


def empty_rows() -> dict:
    return {c: [] for c in BASE_COLUMNS}


def append_row(rows, *, trial_id, movie, task, session_role, split, split_seed,
               start_sample, label, center_sample, source_word_idx, notes):
    rows["session_id"].append(f"sub_02_trial_{trial_id:03d}")
    rows["subject_id"].append(SUBJECT_ID)
    rows["trial_id"].append(trial_id)
    rows["movie"].append(movie)
    rows["task"].append(task)
    rows["session_role"].append(session_role)
    rows["split"].append(split)
    rows["split_seed"].append(split_seed)
    rows["start_sample"].append(int(start_sample))
    rows["end_sample"].append(int(start_sample) + SEGMENT_SAMPLES)
    rows["label"].append(int(label))
    rows["center_sample"].append(int(center_sample))
    rows["source_word_idx"].append(int(source_word_idx))
    rows["notes"].append(notes)


def to_frame(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=BASE_COLUMNS)
    int_cols = ["subject_id", "trial_id", "split_seed", "start_sample", "end_sample",
                "label", "center_sample", "source_word_idx"]
    for c in int_cols:
        df[c] = df[c].astype(np.int64)
    return df


# -----------------------------------------------------------------------------
# Per-trial generators
# -----------------------------------------------------------------------------

def build_pretrain_frame(trial_id: int, movie: str, valid: Tuple[int, int]) -> pd.DataFrame:
    starts = pretrain_tiling(valid[0], valid[1])
    splits = random_split_80_10_10(len(starts), seed=SEED_PRETRAIN)
    role = SESSION_ROLE[trial_id]
    rows = empty_rows()
    for s, sp in zip(starts, splits):
        append_row(rows, trial_id=trial_id, movie=movie, task="pretrain",
                   session_role=role, split=sp, split_seed=SEED_PRETRAIN,
                   start_sample=int(s), label=-1, center_sample=-1,
                   source_word_idx=-1, notes="pretrain_tile")
    return to_frame(rows)


def _build_main_task(
    *,
    trial_id: int,
    movie: str,
    task: str,
    valid: Tuple[int, int],
    pos_centers: np.ndarray,
    pos_word_idx: np.ndarray,
    pos_label_value: np.ndarray,  # vector of 1s for sent_onset/speech, top/bottom-quartile labels for vol/of
    pos_notes: np.ndarray,        # per-positive note string
    word_intervals: np.ndarray,
) -> Tuple[pd.DataFrame, dict]:
    """Shared core for all four main-eval tasks.

    For volume / optical_flow, callers pre-filter `pos_centers` to top+bottom quartile
    candidates and pass `pos_label_value` as the 0/1 array.
    """
    role = SESSION_ROLE[trial_id]

    # ---- positive candidates (centered windows, in-bounds) ----
    if len(pos_centers) > 0:
        pos_starts_all, in_bounds = centered_window(pos_centers, valid)
        pos_starts = pos_starts_all[in_bounds]
        pos_word_idx_kept = pos_word_idx[in_bounds]
        pos_label_kept = pos_label_value[in_bounds]
        pos_notes_kept = pos_notes[in_bounds]
        pos_centers_kept = pos_centers[in_bounds]
    else:
        pos_starts = np.empty(0, dtype=np.int64)
        pos_word_idx_kept = np.empty(0, dtype=np.int64)
        pos_label_kept = np.empty(0, dtype=np.int64)
        pos_notes_kept = np.empty(0, dtype=object)
        pos_centers_kept = np.empty(0, dtype=np.int64)

    # ---- greedy non-overlap among positives (chronological keep-first) ----
    pos_kept_idx = greedy_non_overlap(pos_starts)
    pos_starts_g = pos_starts[pos_kept_idx]
    pos_word_idx_g = pos_word_idx_kept[pos_kept_idx]
    pos_label_g = pos_label_kept[pos_kept_idx]
    pos_notes_g = pos_notes_kept[pos_kept_idx]
    pos_centers_g = pos_centers_kept[pos_kept_idx]

    # ---- negative candidates: pretrain tiling AND no-speech ----
    neg_candidates = pretrain_tiling(valid[0], valid[1])
    no_speech_mask = windows_with_no_speech(neg_candidates, word_intervals)
    neg_candidates = neg_candidates[no_speech_mask]
    # Greedy non-overlap is automatic for the pretrain tiling, but keep the
    # call so the contract is explicit.
    neg_kept_idx = greedy_non_overlap(neg_candidates)
    neg_starts_g = neg_candidates[neg_kept_idx]

    # ---- cross-prune negatives that overlap any kept positive ----
    if len(pos_starts_g) > 0:
        keep_neg_mask = drop_overlapping_with(neg_starts_g, pos_starts_g)
        neg_starts_g = neg_starts_g[keep_neg_mask]

    # ---- class balance ----
    n_pos = len(pos_starts_g)
    n_neg = len(neg_starts_g)
    pos_keep_idx, neg_keep_idx = class_balance_indices(n_pos, n_neg, seed=SEED_MAIN)
    pos_starts_b = pos_starts_g[pos_keep_idx]
    pos_word_idx_b = pos_word_idx_g[pos_keep_idx]
    pos_label_b = pos_label_g[pos_keep_idx]
    pos_notes_b = pos_notes_g[pos_keep_idx]
    pos_centers_b = pos_centers_g[pos_keep_idx]
    neg_starts_b = neg_starts_g[neg_keep_idx]

    # ---- 80/10/10 random split over the balanced set (positives + negatives mixed) ----
    n_total = len(pos_starts_b) + len(neg_starts_b)
    splits = random_split_80_10_10(n_total, seed=SEED_MAIN)

    rows = empty_rows()
    # positives first
    for i in range(len(pos_starts_b)):
        append_row(rows, trial_id=trial_id, movie=movie, task=task,
                   session_role=role, split=splits[i], split_seed=SEED_MAIN,
                   start_sample=int(pos_starts_b[i]),
                   label=int(pos_label_b[i]),
                   center_sample=int(pos_centers_b[i]),
                   source_word_idx=int(pos_word_idx_b[i]),
                   notes=str(pos_notes_b[i]))
    # then negatives
    offset = len(pos_starts_b)
    for j in range(len(neg_starts_b)):
        append_row(rows, trial_id=trial_id, movie=movie, task=task,
                   session_role=role, split=splits[offset + j], split_seed=SEED_MAIN,
                   start_sample=int(neg_starts_b[j]),
                   label=0,
                   center_sample=-1, source_word_idx=-1,
                   notes="no_speech_negative")

    df = to_frame(rows)
    counts = {
        "n_pos_candidates_in_bounds": int(len(pos_starts)),
        "n_pos_after_overlap_prune": int(n_pos),
        "n_neg_after_overlap_prune": int(n_neg),
        "n_pos_after_balance": int(len(pos_starts_b)),
        "n_neg_after_balance": int(len(neg_starts_b)),
        "split_train": int((df["split"] == "train").sum()),
        "split_valid": int((df["split"] == "valid").sum()),
        "split_test": int((df["split"] == "test").sum()),
        "total": int(len(df)),
    }
    return df, counts


def build_sentence_onset(trial_id, movie, valid, transcript, t2s):
    onset = transcript[transcript["is_onset"] == 1.0]
    centers = t2s(onset["start"].to_numpy())
    word_idx = onset.index.to_numpy(dtype=np.int64)
    labels = np.ones(len(centers), dtype=np.int64)
    notes = np.array(["sentence_onset_pos"] * len(centers), dtype=object)
    word_intervals = np.stack(
        [t2s(transcript["start"].to_numpy()), t2s(transcript["end"].to_numpy())],
        axis=1,
    )
    return _build_main_task(
        trial_id=trial_id, movie=movie, task="sentence_onset", valid=valid,
        pos_centers=centers, pos_word_idx=word_idx,
        pos_label_value=labels, pos_notes=notes,
        word_intervals=word_intervals,
    )


def build_speech_nonspeech(trial_id, movie, valid, transcript, t2s):
    centers = t2s(transcript["start"].to_numpy())
    word_idx = transcript.index.to_numpy(dtype=np.int64)
    labels = np.ones(len(centers), dtype=np.int64)
    notes = np.array(["word_pos"] * len(centers), dtype=object)
    word_intervals = np.stack(
        [centers, t2s(transcript["end"].to_numpy())],
        axis=1,
    )
    return _build_main_task(
        trial_id=trial_id, movie=movie, task="speech_nonspeech", valid=valid,
        pos_centers=centers, pos_word_idx=word_idx,
        pos_label_value=labels, pos_notes=notes,
        word_intervals=word_intervals,
    )


def _quartile_labels(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mask_keep, label) where label is 1 for top quartile, 0 for bottom, mask drops middle two."""
    finite = np.isfinite(values)
    valid_values = values[finite]
    if len(valid_values) < 4:
        return np.zeros(len(values), dtype=bool), np.zeros(len(values), dtype=np.int64)
    q1, q3 = np.quantile(valid_values, [0.25, 0.75])
    label = np.full(len(values), -1, dtype=np.int64)
    label[finite & (values <= q1)] = 0
    label[finite & (values >= q3)] = 1
    keep = label != -1
    return keep, label


def build_quartile_task(trial_id, movie, valid, transcript, t2s, *, task: str, value_col: str):
    centers_all = t2s(transcript["start"].to_numpy())
    values = transcript[value_col].to_numpy(dtype=np.float64)
    keep, labels = _quartile_labels(values)
    centers = centers_all[keep]
    word_idx = transcript.index.to_numpy(dtype=np.int64)[keep]
    label_kept = labels[keep]
    notes = np.where(label_kept == 1,
                     f"{task}_top_quartile_pos",
                     f"{task}_bottom_quartile_pos").astype(object)
    word_intervals = np.stack(
        [centers_all, t2s(transcript["end"].to_numpy())],
        axis=1,
    )
    df, counts = _build_main_task(
        trial_id=trial_id, movie=movie, task=task, valid=valid,
        pos_centers=centers, pos_word_idx=word_idx,
        pos_label_value=label_kept, pos_notes=notes,
        word_intervals=word_intervals,
    )
    counts["n_words_total"] = int(len(values))
    counts["n_words_top_or_bottom_quartile"] = int(keep.sum())
    return df, counts


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="local", choices=["local", "server"])
    parser.add_argument("--out-dir", default=None,
                        help="Override output directory (default: <repo>/segment_indices)")
    args = parser.parse_args()

    cfg = get_config(args.profile)
    bt_root = Path(cfg.braintree_data_root)

    repo_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_dir) if args.out_dir else repo_root / "segment_indices"
    sub_out = out_root / "sub_02"
    sub_out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "subject_id": SUBJECT_ID,
        "fs": FS,
        "segment_samples": SEGMENT_SAMPLES,
        "seeds": {"pretrain_split": SEED_PRETRAIN, "main_split": SEED_MAIN},
        "session_role_map": SESSION_ROLE,
        "schemes": ["pretrain", "main"],
        "tasks_main": ["sentence_onset", "speech_nonspeech", "volume", "optical_flow"],
        "optical_flow_column": OPTICAL_FLOW_COL,
        "trials": {},
    }

    main_task_builders = [
        ("sentence_onset", build_sentence_onset),
        ("speech_nonspeech", build_speech_nonspeech),
        ("volume", lambda *a, **k: build_quartile_task(*a, task="volume", value_col="rms", **k)),
        ("optical_flow", lambda *a, **k: build_quartile_task(*a, task="optical_flow", value_col=OPTICAL_FLOW_COL, **k)),
    ]

    table6_targets = {
        "sentence_onset":    {"train": 1036, "valid": 129, "test": 129},
        "speech_nonspeech":  {"train": 1470, "valid": 183, "test": 183},
        "channel_recon":     {"train": 3385, "valid": 422, "test": 422},
    }

    for trial_id in TRIAL_IDS:
        meta = load_metadata(bt_root, trial_id)
        movie_filename = meta["filename"]
        timings = load_timings(bt_root, trial_id)
        valid = valid_window(timings)
        t2s = time_to_sample_fn(timings)
        transcript = load_transcript(bt_root, movie_filename)

        trial_entry = {
            "movie": movie_filename,
            "movie_title": meta.get("title"),
            "session_role": SESSION_ROLE[trial_id],
            "valid_window": [int(valid[0]), int(valid[1])],
            "valid_duration_s": float((valid[1] - valid[0]) / FS),
            "n_words": int(len(transcript)),
            "n_sentence_onsets": int((transcript["is_onset"] == 1.0).sum()),
            "counts": {},
        }

        # ---- pretrain tiling (also serves as channel_recon candidate pool) ----
        df_pre = build_pretrain_frame(trial_id, movie_filename, valid)
        pre_path = sub_out / f"trial{trial_id:03d}__pretrain.parquet"
        df_pre.to_parquet(pre_path, index=False)
        trial_entry["counts"]["pretrain"] = {
            "total": int(len(df_pre)),
            "train": int((df_pre["split"] == "train").sum()),
            "valid": int((df_pre["split"] == "valid").sum()),
            "test": int((df_pre["split"] == "test").sum()),
        }

        # ---- four main-eval tasks ----
        for task_name, builder in main_task_builders:
            df_t, counts = builder(trial_id, movie_filename, valid, transcript, t2s)
            out_path = sub_out / f"trial{trial_id:03d}__{task_name}.parquet"
            df_t.to_parquet(out_path, index=False)
            trial_entry["counts"][task_name] = counts

        manifest["trials"][f"trial{trial_id:03d}"] = trial_entry
        print(
            f"[trial{trial_id:03d}] {movie_filename:35s} "
            f"role={SESSION_ROLE[trial_id]:>16s}  "
            f"pretrain={trial_entry['counts']['pretrain']['total']:>5d}  "
            f"sent_onset={trial_entry['counts']['sentence_onset']['total']:>5d}  "
            f"speech={trial_entry['counts']['speech_nonspeech']['total']:>5d}  "
            f"volume={trial_entry['counts']['volume']['total']:>5d}  "
            f"of={trial_entry['counts']['optical_flow']['total']:>5d}"
        )

    # ---- Table 6 sanity check ----
    # BaRISTA Table 6 reports per-subject train/valid/test for the *test session(s)*.
    # CLAUDE.md flags trial006 as the downstream-test session for sub_2.
    test_trial = "trial006"
    t6 = manifest["trials"][test_trial]["counts"]
    sanity = {}
    sanity["sentence_onset"] = {
        "expected_table6": table6_targets["sentence_onset"],
        "got_trial006":   {k: t6["sentence_onset"][f"split_{k}"] for k in ("train", "valid", "test")},
    }
    sanity["speech_nonspeech"] = {
        "expected_table6": table6_targets["speech_nonspeech"],
        "got_trial006":   {k: t6["speech_nonspeech"][f"split_{k}"] for k in ("train", "valid", "test")},
    }
    sanity["channel_recon"] = {
        "expected_table6": table6_targets["channel_recon"],
        "got_trial006_pretrain": {
            "train": t6["pretrain"]["train"],
            "valid": t6["pretrain"]["valid"],
            "test":  t6["pretrain"]["test"],
        },
    }
    manifest["table6_sanity"] = sanity

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=int)

    print("\n--- Table 6 (sub_2) sanity check ---")
    for task in ("sentence_onset", "speech_nonspeech"):
        e = sanity[task]["expected_table6"]
        g = sanity[task]["got_trial006"]
        print(f"  {task:18s}  expected {e['train']:>5d}/{e['valid']:>4d}/{e['test']:>4d}  "
              f"got {g['train']:>5d}/{g['valid']:>4d}/{g['test']:>4d}")
    e = sanity["channel_recon"]["expected_table6"]
    g = sanity["channel_recon"]["got_trial006_pretrain"]
    print(f"  {'channel_recon':18s}  expected {e['train']:>5d}/{e['valid']:>4d}/{e['test']:>4d}  "
          f"got {g['train']:>5d}/{g['valid']:>4d}/{g['test']:>4d}  (= trial006 pretrain tiling)")
    print(f"\nWrote {len(list(sub_out.glob('*.parquet')))} parquets to {sub_out}")
    print(f"Manifest at {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
