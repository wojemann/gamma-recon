from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm.auto import tqdm

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parents[1]
for _p in (_SCRIPTS, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from brt_utils import (
    BAND_DEFS,
    bandpass_full_trace,
    build_trial_windows,
    list_trial_h5_paths,
    load_temporal_channel_info,
    notch_only_full_trace,
    parse_trial_id,
)
from configs.config import get_config


def expand_windows(window_df: pd.DataFrame) -> pd.DataFrame:
    onset = window_df[["trial_id", "sentence_idx", "onset_movie_time", "on_start", "on_end"]].copy()
    onset = onset.rename(columns={"on_start": "start", "on_end": "end"})
    onset["label"] = "onset"
    onset["y"] = 1

    ctrl = window_df[["trial_id", "sentence_idx", "onset_movie_time", "ctrl_start", "ctrl_end"]].copy()
    ctrl = ctrl.rename(columns={"ctrl_start": "start", "ctrl_end": "end"})
    ctrl["label"] = "control"
    ctrl["y"] = 0
    return pd.concat([onset, ctrl], ignore_index=True)


def split_windows_by_sentence(windows_df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sentence_ids = windows_df["sentence_idx"].drop_duplicates().to_numpy()
    tr_ids, te_ids = train_test_split(sentence_ids, test_size=test_size, random_state=seed, shuffle=True)
    train_w = windows_df[windows_df["sentence_idx"].isin(tr_ids)].reset_index(drop=True)
    test_w = windows_df[windows_df["sentence_idx"].isin(te_ids)].reset_index(drop=True)
    return train_w, test_w


def select_channels_train_only(
    root: Path,
    subject: str,
    trial_id: int,
    train_windows: pd.DataFrame,
    channel_info: pd.DataFrame,
    fs: float,
    notch_freqs: tuple[float, ...],
    notch_q: float,
    t_threshold: float,
    fallback_top_k: int,
) -> tuple[List[str], pd.DataFrame]:
    h5_path = root / f"{subject}_trial{trial_id:03d}.h5"
    channel_names = channel_info["dataset_name"].tolist()
    region_map = dict(zip(channel_info["dataset_name"], channel_info["Region"]))
    rows = []

    with h5py.File(h5_path, "r") as h5f:
        group = h5f["data"]
        for ds_name in tqdm(channel_names, desc=f"Trial {trial_id:03d} selection channels", leave=False):
            raw = np.asarray(group[ds_name][:], dtype=float)
            xn = notch_only_full_trace(raw, fs=fs, notch_freqs=notch_freqs, notch_q=notch_q)
            ctrl_vals = []
            onset_vals = []
            for _, w in train_windows.iterrows():
                ctrl = np.abs(xn[int(w["ctrl_start"]):int(w["ctrl_end"])])
                onset = np.abs(xn[int(w["on_start"]):int(w["on_end"])])
                if ctrl.size and onset.size:
                    ctrl_vals.append(float(np.mean(ctrl)))
                    onset_vals.append(float(np.mean(onset)))
            if len(ctrl_vals) < 5 or len(onset_vals) < 5:
                continue

            t_stat, p_val = ttest_ind(onset_vals, ctrl_vals, equal_var=False, nan_policy="omit")
            if not np.isfinite(t_stat):
                continue
            rows.append(
                {
                    "dataset_name": ds_name,
                    "region": region_map.get(ds_name, "Unknown"),
                    "control_mean_train": float(np.mean(ctrl_vals)),
                    "onset_mean_train": float(np.mean(onset_vals)),
                    "t_stat": float(t_stat),
                    "abs_t_stat": float(abs(t_stat)),
                    "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
                }
            )

    stats = pd.DataFrame(rows).sort_values("abs_t_stat", ascending=False).reset_index(drop=True)
    selected = stats.loc[stats["abs_t_stat"] > t_threshold, "dataset_name"].tolist()
    if not selected:
        selected = stats["dataset_name"].head(fallback_top_k).tolist()
    return selected, stats


def build_features_for_split(
    root: Path,
    subject: str,
    trial_id: int,
    windows_df: pd.DataFrame,
    selected_channels: List[str],
    fs: float,
    notch_freqs: tuple[float, ...],
    notch_q: float,
) -> pd.DataFrame:
    samples = expand_windows(windows_df)
    for band in BAND_DEFS:
        samples[f"{band}_power"] = 0.0
    samples["n_channels"] = 0

    h5_path = root / f"{subject}_trial{trial_id:03d}.h5"
    sw_idx = samples.index.to_numpy()
    sw_ranges = samples[["start", "end"]].to_numpy(dtype=int)

    with h5py.File(h5_path, "r") as h5f:
        group = h5f["data"]
        for ds_name in tqdm(selected_channels, desc=f"Trial {trial_id:03d} feature channels", leave=False):
            raw = np.asarray(group[ds_name][:], dtype=float)
            xn = notch_only_full_trace(raw, fs=fs, notch_freqs=notch_freqs, notch_q=notch_q)
            filtered = {band: bandpass_full_trace(xn, fs, lo, hi) for band, (lo, hi) in BAND_DEFS.items()}
            for i, (s0, s1) in zip(sw_idx, sw_ranges):
                if s1 <= s0:
                    continue
                for band in BAND_DEFS:
                    seg = filtered[band][s0:s1]
                    if seg.size:
                        samples.at[i, f"{band}_power"] += float(np.mean(seg ** 2))
                samples.at[i, "n_channels"] += 1

    valid = samples["n_channels"] > 0
    for band in BAND_DEFS:
        samples.loc[valid, f"{band}_power"] = samples.loc[valid, f"{band}_power"] / samples.loc[valid, "n_channels"]
    return samples


def fit_predict_models(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int):
    feature_cols = [f"{b}_power" for b in BAND_DEFS]
    model_defs = [{"model_name": f"logreg_{col}", "cols": [col]} for col in feature_cols]
    model_defs.append({"model_name": "logreg_all_features", "cols": feature_cols})

    pred_rows = []
    metric_rows = []
    for m in model_defs:
        cols = m["cols"]
        xtr = train_df[cols].to_numpy()
        ytr = train_df["y"].to_numpy()
        xte = test_df[cols].to_numpy()
        yte = test_df["y"].to_numpy()

        pipe = Pipeline([("scaler", RobustScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
        pipe.fit(xtr, ytr)

        for split_name, x, y, df in [("train", xtr, ytr, train_df), ("test", xte, yte, test_df)]:
            proba = pipe.predict_proba(x)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metric_rows.append(
                {
                    "model_name": m["model_name"],
                    "split": split_name,
                    "auc": float(roc_auc_score(y, proba)),
                    "accuracy": float(accuracy_score(y, pred)),
                }
            )
            for i, p, yhat in zip(df.index.to_numpy(), proba, pred):
                pred_rows.append(
                    {
                        "sample_uid": str(df.at[i, "sample_uid"]),
                        "model_name": m["model_name"],
                        "split": split_name,
                        "proba": float(p),
                        "pred": int(yhat),
                    }
                )
    return pd.DataFrame(pred_rows), pd.DataFrame(metric_rows)


def main():
    parser = argparse.ArgumentParser(description="BrainTree per-trial 70/30 train-test analysis")
    parser.add_argument("--path-profile", type=str, choices=("local", "server"), default="local")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--subject", type=str, default="sub_2")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--t-threshold", type=float, default=2.0)
    parser.add_argument("--fallback-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    cfg = get_config(args.path_profile)
    if args.root is None:
        args.root = cfg.brain_tree_root
    if args.output_dir is None:
        args.output_dir = _SCRIPTS / "outputs" / "per_trial"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trial_channels").mkdir(parents=True, exist_ok=True)

    trial_paths = list_trial_h5_paths(args.root, args.subject)
    trial_ids = [parse_trial_id(p) for p in trial_paths]
    channel_info = load_temporal_channel_info(args.root, args.subject)

    trial_data = {}
    for tid in tqdm(trial_ids, desc="Building trial windows"):
        trial_data[tid] = build_trial_windows(
            root=args.root,
            subject=args.subject,
            trial_id=tid,
            control_sec=0.5,
            onset_start_sec=0.2,
            onset_end_sec=0.7,
        )

    all_features = []
    all_predictions = []
    all_metrics = []
    all_selected = []

    for trial_id in tqdm(trial_ids, desc="Per-trial analyses"):
        windows = trial_data[trial_id].windows.copy()
        fs = trial_data[trial_id].fs
        train_windows, test_windows = split_windows_by_sentence(windows, test_size=args.test_size, seed=args.seed)

        selected, stats = select_channels_train_only(
            root=args.root,
            subject=args.subject,
            trial_id=trial_id,
            train_windows=train_windows,
            channel_info=channel_info,
            fs=fs,
            notch_freqs=(60.0, 120.0, 180.0),
            notch_q=30.0,
            t_threshold=args.t_threshold,
            fallback_top_k=args.fallback_top_k,
        )

        stats["trial_id"] = trial_id
        stats["is_selected"] = stats["dataset_name"].isin(selected)
        stats.to_csv(args.output_dir / "trial_channels" / f"channels_trial{trial_id:03d}.csv", index=False)

        for rank, ds in enumerate(selected, start=1):
            row = stats.loc[stats["dataset_name"] == ds].iloc[0].to_dict()
            row.update({"trial_id": trial_id, "rank": rank})
            all_selected.append(row)

        train_features = build_features_for_split(
            root=args.root,
            subject=args.subject,
            trial_id=trial_id,
            windows_df=train_windows,
            selected_channels=selected,
            fs=fs,
            notch_freqs=(60.0, 120.0, 180.0),
            notch_q=30.0,
        )
        train_features["split"] = "train"
        train_features["trial_id"] = trial_id
        train_features["y"] = (train_features["label"] == "onset").astype(int)
        train_features["sample_uid"] = train_features.apply(lambda r: f"{trial_id}_train_{int(r.name)}", axis=1)

        test_features = build_features_for_split(
            root=args.root,
            subject=args.subject,
            trial_id=trial_id,
            windows_df=test_windows,
            selected_channels=selected,
            fs=fs,
            notch_freqs=(60.0, 120.0, 180.0),
            notch_q=30.0,
        )
        test_features["split"] = "test"
        test_features["trial_id"] = trial_id
        test_features["y"] = (test_features["label"] == "onset").astype(int)
        test_features["sample_uid"] = test_features.apply(lambda r: f"{trial_id}_test_{int(r.name)}", axis=1)

        trial_features = pd.concat([train_features, test_features], ignore_index=True)
        trial_features["fold_id"] = f"trial_{trial_id:03d}"

        preds, metrics = fit_predict_models(train_features, test_features, seed=args.seed)
        preds["trial_id"] = trial_id
        preds["fold_id"] = f"trial_{trial_id:03d}"
        preds = preds.merge(trial_features, on="sample_uid", how="left")
        metrics["trial_id"] = trial_id
        metrics["fold_id"] = f"trial_{trial_id:03d}"

        all_features.append(trial_features)
        all_predictions.append(preds)
        all_metrics.append(metrics)

    features_df = pd.concat(all_features, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    selected_df = pd.DataFrame(all_selected)

    features_df.to_csv(args.output_dir / "samples_features.csv", index=False)
    predictions_df.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    selected_df.to_csv(args.output_dir / "selected_channels.csv", index=False)

    config = {
        "subject": args.subject,
        "trial_ids": trial_ids,
        "paradigm": "per_trial_train_test",
        "test_size": args.test_size,
        "t_threshold": args.t_threshold,
        "bands": BAND_DEFS,
        "window_definition_ms": {"control": [-500, 0], "onset": [200, 700]},
    }
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Saved outputs:")
    print(f"- {args.output_dir / 'samples_features.csv'}")
    print(f"- {args.output_dir / 'predictions.csv'}")
    print(f"- {args.output_dir / 'metrics.csv'}")
    print(f"- {args.output_dir / 'selected_channels.csv'}")


if __name__ == "__main__":
    main()

