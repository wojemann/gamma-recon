from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
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
    TrialData,
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


def precompute_selection_means_per_trial(
    root: Path,
    subject: str,
    trial_ids: List[int],
    trial_data: Dict[int, TrialData],
    channel_info: pd.DataFrame,
    fs_by_trial: Dict[int, float],
    notch_freqs: tuple[float, ...],
    notch_q: float,
) -> tuple[Dict[int, Dict[str, List[float]]], Dict[int, Dict[str, List[float]]]]:
    """
    For each trial and channel, mean |signal| in control vs onset windows (after notch).
    Used to build fold-specific train pools without re-reading HDF5 on every holdout fold.
    """
    channel_names = channel_info["dataset_name"].tolist()
    ctrl_by_trial: Dict[int, Dict[str, List[float]]] = {
        tid: {ch: [] for ch in channel_names} for tid in trial_ids
    }
    onset_by_trial: Dict[int, Dict[str, List[float]]] = {
        tid: {ch: [] for ch in channel_names} for tid in trial_ids
    }

    for trial_id in tqdm(trial_ids, desc="Precompute selection (once per trial)"):
        h5_path = root / f"{subject}_trial{trial_id:03d}.h5"
        fs = fs_by_trial[trial_id]
        tw = trial_data[trial_id].windows
        with h5py.File(h5_path, "r") as h5f:
            group = h5f["data"]
            for ds_name in tqdm(channel_names, desc=f"Trial {trial_id:03d} channels", leave=False):
                raw = np.asarray(group[ds_name][:], dtype=float)
                xn = notch_only_full_trace(raw, fs=fs, notch_freqs=notch_freqs, notch_q=notch_q)
                for _, w in tw.iterrows():
                    c = np.abs(xn[int(w["ctrl_start"]) : int(w["ctrl_end"])])
                    o = np.abs(xn[int(w["on_start"]) : int(w["on_end"])])
                    if c.size and o.size:
                        ctrl_by_trial[trial_id][ds_name].append(float(np.mean(c)))
                        onset_by_trial[trial_id][ds_name].append(float(np.mean(o)))

    return ctrl_by_trial, onset_by_trial


def select_channels_from_precomputed(
    train_trials: List[int],
    channel_info: pd.DataFrame,
    ctrl_by_trial: Dict[int, Dict[str, List[float]]],
    onset_by_trial: Dict[int, Dict[str, List[float]]],
    t_threshold: float,
    fallback_top_k: int,
) -> tuple[List[str], pd.DataFrame]:
    """Pool precomputed per-trial window means for train_trials and run t-tests (same logic as before)."""
    channel_names = channel_info["dataset_name"].tolist()
    region_map = dict(zip(channel_info["dataset_name"], channel_info["Region"]))
    ctrl_vals: Dict[str, List[float]] = {ch: [] for ch in channel_names}
    onset_vals: Dict[str, List[float]] = {ch: [] for ch in channel_names}

    for trial_id in train_trials:
        for ch in channel_names:
            ctrl_vals[ch].extend(ctrl_by_trial[trial_id][ch])
            onset_vals[ch].extend(onset_by_trial[trial_id][ch])

    rows = []
    for ds_name in tqdm(channel_names, desc="Selection channel stats", leave=False):
        cvals = np.asarray(ctrl_vals[ds_name], dtype=float)
        ovals = np.asarray(onset_vals[ds_name], dtype=float)
        if cvals.size < 5 or ovals.size < 5:
            continue
        t_stat, p_val = ttest_ind(ovals, cvals, equal_var=False, nan_policy="omit")
        if not np.isfinite(t_stat):
            continue
        rows.append(
            {
                "dataset_name": ds_name,
                "region": region_map.get(ds_name, "Unknown"),
                "control_mean_train": float(np.mean(cvals)),
                "onset_mean_train": float(np.mean(ovals)),
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
    windows_df: pd.DataFrame,
    selected_channels: List[str],
    fs_by_trial: Dict[int, float],
    notch_freqs: tuple[float, ...],
    notch_q: float,
) -> pd.DataFrame:
    samples = expand_windows(windows_df)
    for band in BAND_DEFS:
        samples[f"{band}_power"] = 0.0
    samples["n_channels"] = 0

    for trial_id, sw in tqdm(dict(tuple(samples.groupby("trial_id"))).items(), desc="Feature trials", leave=False):
        h5_path = root / f"{subject}_trial{trial_id:03d}.h5"
        fs = fs_by_trial[trial_id]
        sw_idx = sw.index.to_numpy()
        sw_ranges = samples.loc[sw_idx, ["start", "end"]].to_numpy(dtype=int)

        with h5py.File(h5_path, "r") as h5f:
            group = h5f["data"]
            for ds_name in tqdm(selected_channels, desc=f"Trial {trial_id:03d} selected", leave=False):
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
    for m in tqdm(model_defs, desc="Model fits", leave=False):
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
    parser = argparse.ArgumentParser(description="BrainTree hold-one-trial-out analysis")
    parser.add_argument("--path-profile", type=str, choices=("local", "server"), default="local")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--subject", type=str, default="sub_2")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--t-threshold", type=float, default=2.0)
    parser.add_argument("--fallback-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    cfg = get_config(args.path_profile)
    if args.root is None:
        args.root = cfg.brain_tree_root
    if args.output_dir is None:
        args.output_dir = _SCRIPTS / "outputs" / "holdout"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fold_channels").mkdir(parents=True, exist_ok=True)

    trial_paths = list_trial_h5_paths(args.root, args.subject)
    trial_ids = [parse_trial_id(p) for p in trial_paths]
    channel_info = load_temporal_channel_info(args.root, args.subject)

    trial_data = {}
    for tid in tqdm(trial_ids, desc="Building trial windows"):
        td = build_trial_windows(
            root=args.root,
            subject=args.subject,
            trial_id=tid,
            control_sec=0.5,
            onset_start_sec=0.2,
            onset_end_sec=0.7,
        )
        trial_data[tid] = td

    fs_by_trial = {tid: td.fs for tid, td in trial_data.items()}
    ctrl_by_trial, onset_by_trial = precompute_selection_means_per_trial(
        root=args.root,
        subject=args.subject,
        trial_ids=trial_ids,
        trial_data=trial_data,
        channel_info=channel_info,
        fs_by_trial=fs_by_trial,
        notch_freqs=(60.0, 120.0, 180.0),
        notch_q=30.0,
    )

    all_features = []
    all_predictions = []
    all_metrics = []
    all_selected = []

    for heldout in tqdm(trial_ids, desc="Holdout folds", leave=True):
        train_trials = [t for t in trial_ids if t != heldout]
        train_windows = pd.concat([trial_data[t].windows for t in train_trials], ignore_index=True)
        test_windows = trial_data[heldout].windows.copy()

        selected, stats = select_channels_from_precomputed(
            train_trials=train_trials,
            channel_info=channel_info,
            ctrl_by_trial=ctrl_by_trial,
            onset_by_trial=onset_by_trial,
            t_threshold=args.t_threshold,
            fallback_top_k=args.fallback_top_k,
        )

        stats["heldout_trial"] = heldout
        stats["is_selected"] = stats["dataset_name"].isin(selected)
        stats.to_csv(args.output_dir / "fold_channels" / f"channels_fold_trial{heldout:03d}.csv", index=False)

        for rank, ds in enumerate(selected, start=1):
            row = stats.loc[stats["dataset_name"] == ds].iloc[0].to_dict()
            row.update({"heldout_trial": heldout, "rank": rank})
            all_selected.append(row)

        train_features = build_features_for_split(
            root=args.root,
            subject=args.subject,
            windows_df=train_windows,
            selected_channels=selected,
            fs_by_trial=fs_by_trial,
            notch_freqs=(60.0, 120.0, 180.0),
            notch_q=30.0,
        )
        train_features["split"] = "train"
        train_features["heldout_trial"] = heldout
        train_features["y"] = (train_features["label"] == "onset").astype(int)
        train_features["sample_uid"] = train_features.apply(lambda r: f"{heldout}_train_{int(r.name)}", axis=1)

        test_features = build_features_for_split(
            root=args.root,
            subject=args.subject,
            windows_df=test_windows,
            selected_channels=selected,
            fs_by_trial=fs_by_trial,
            notch_freqs=(60.0, 120.0, 180.0),
            notch_q=30.0,
        )
        test_features["split"] = "test"
        test_features["heldout_trial"] = heldout
        test_features["y"] = (test_features["label"] == "onset").astype(int)
        test_features["sample_uid"] = test_features.apply(lambda r: f"{heldout}_test_{int(r.name)}", axis=1)

        fold_features = pd.concat([train_features, test_features], ignore_index=True)
        fold_features["fold_id"] = f"holdout_trial_{heldout:03d}"

        preds, metrics = fit_predict_models(train_features, test_features, seed=args.seed)
        preds["heldout_trial"] = heldout
        preds["fold_id"] = f"holdout_trial_{heldout:03d}"
        preds = preds.merge(fold_features, on="sample_uid", how="left")
        metrics["heldout_trial"] = heldout
        metrics["fold_id"] = f"holdout_trial_{heldout:03d}"

        all_features.append(fold_features)
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
        "paradigm": "hold_one_trial_out",
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
