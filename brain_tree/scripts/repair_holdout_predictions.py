"""
Repair hold-one-trial-out predictions.csv when it was merged badly (duplicate _x/_y
columns or wrong trial_id/y). Does not rerun feature extraction — realigns each
(model, fold, split) block to samples_features.csv row order using sample_index as
positional index inside that block — matching how fit_predict walks train_df then test_df.

Usage:
  cd scripts && conda run -n gamma-env python repair_holdout_predictions.py --output-dir outputs/holdout
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from brt_utils import BAND_DEFS


def model_order():
    return [f"logreg_{b}_power" for b in BAND_DEFS] + ["logreg_all_features"]


def _split_col(preds: pd.DataFrame) -> str:
    if "split_x" in preds.columns:
        return "split_x"
    if "split" in preds.columns:
        return "split"
    raise ValueError("No split column")


def _heldout_col(preds: pd.DataFrame) -> str:
    if "heldout_trial_x" in preds.columns:
        return "heldout_trial_x"
    if "heldout_trial" in preds.columns:
        return "heldout_trial"
    raise ValueError("No heldout_trial column")


def repair_predictions(preds: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    sc, hc = _split_col(preds), _heldout_col(preds)
    pred_groups = preds.groupby([hc, sc, "model_name"], sort=False)
    truth_cols = [
        "trial_id",
        "sentence_idx",
        "onset_movie_time",
        "start",
        "end",
        "label",
        "y",
        "delta_power",
        "theta_power",
        "alpha_power",
        "beta_power",
        "gamma_power",
        "broadband_power",
        "n_channels",
        "fold_id",
    ]
    out_rows = []
    for h in sorted(feats["heldout_trial"].unique()):
        ff = feats[feats["heldout_trial"] == h]
        for model_name in model_order():
            for split in ("train", "test"):
                f_block = ff[ff["split"] == split].reset_index(drop=True)
                try:
                    p_block = pred_groups.get_group((h, split, model_name)).sort_values("sample_index")
                except KeyError:
                    raise ValueError(f"Missing pred block fold={h} model={model_name} split={split}") from None
                p_block = p_block.reset_index(drop=True)
                if len(f_block) != len(p_block):
                    raise ValueError(
                        f"Length mismatch fold={h} model={model_name} split={split}: "
                        f"features={len(f_block)} preds={len(p_block)}"
                    )
                uid_prefix = f"{h}_{split}_{model_name}"
                for i in range(len(f_block)):
                    tr = f_block.iloc[i]
                    pr = p_block.iloc[i]
                    uid = pr["sample_uid"] if "sample_uid" in preds.columns and pd.notna(pr.get("sample_uid")) else f"{uid_prefix}_{int(pr['sample_index'])}"
                    row = {c: tr[c] for c in truth_cols if c in tr.index}
                    row.update(
                        {
                            "sample_uid": str(uid),
                            "sample_index": int(pr["sample_index"]),
                            "model_name": model_name,
                            "split": split,
                            "heldout_trial": int(h),
                            "proba": float(pr["proba"]),
                            "pred": int(pr["pred"]),
                        }
                    )
                    out_rows.append(row)
    return pd.DataFrame(out_rows)


def recompute_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in model_order():
        for split in ("train", "test"):
            for h in sorted(preds["heldout_trial"].unique()):
                sub = preds[
                    (preds["heldout_trial"] == h)
                    & (preds["split"] == split)
                    & (preds["model_name"] == model_name)
                ]
                if len(sub) < 2:
                    auc = float("nan")
                else:
                    try:
                        auc = float(roc_auc_score(sub["y"].to_numpy(), sub["proba"].to_numpy()))
                    except ValueError:
                        auc = float("nan")
                acc = float(accuracy_score(sub["y"].to_numpy(), sub["pred"].to_numpy()))
                rows.append(
                    {
                        "model_name": model_name,
                        "split": split,
                        "heldout_trial": int(h),
                        "fold_id": sub["fold_id"].iloc[0] if len(sub) else "",
                        "auc": auc,
                        "accuracy": acc,
                    }
                )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Repair holdout predictions CSV from samples_features")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="Validate alignment only")
    ap.add_argument("--force", action="store_true", help="Rebuild even if split_x absent (already-repaired)")
    args = ap.parse_args()
    out_dir = args.output_dir
    pred_path = out_dir / "predictions.csv"
    feat_path = out_dir / "samples_features.csv"

    preds_raw = pd.read_csv(pred_path)
    if "split_x" not in preds_raw.columns and not args.force:
        print("predictions.csv has no merge-artifact columns (split_x). Nothing to repair. Use --force to rebuild from sample_index.")
        return
    feats = pd.read_csv(feat_path)

    if "sample_index" not in preds_raw.columns:
        raise ValueError(
            "predictions.csv lacks sample_index — cannot positional-repair safely. "
            "Re-run run_holdout_analysis.py with merge on sample_uid, or regenerate."
        )

    fixed = repair_predictions(preds_raw, feats)

    bad = preds_raw.duplicated(subset=[_heldout_col(preds_raw), _split_col(preds_raw), "model_name", "sample_index"]).sum()
    if bad:
        raise ValueError(f"Duplicate ({bad}) broken merge rows — repair logic may be wrong.")

    if args.dry_run:
        print("Dry run OK:", len(fixed), "rows; sample:")
        print(fixed.head(2))
        return

    backup = out_dir / "predictions_pre_repair.csv"
    shutil.copy(pred_path, backup)
    print(f"Backed up original to {backup}")

    fixed.to_csv(pred_path, index=False)
    print(f"Wrote repaired {pred_path}")

    metrics = recompute_metrics(fixed)
    metrics_path = out_dir / "metrics.csv"
    shutil.copy(metrics_path, out_dir / "metrics_pre_repair.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"Wrote recomputed metrics to {metrics_path}")


if __name__ == "__main__":
    main()
