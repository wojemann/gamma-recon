from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm.auto import tqdm

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parents[1]
for _p in (_SCRIPTS, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from brt_utils import BAND_DEFS  # noqa: E402
from configs.config import get_config  # noqa: E402


def normalize_prediction_columns(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds.copy()
    if "split" not in out.columns:
        if "split_x" in out.columns:
            out["split"] = out["split_x"]
        elif "split_y" in out.columns:
            out["split"] = out["split_y"]
    if "heldout_trial" not in out.columns:
        if "heldout_trial_x" in out.columns:
            out["heldout_trial"] = out["heldout_trial_x"]
        elif "heldout_trial_y" in out.columns:
            out["heldout_trial"] = out["heldout_trial_y"]
    if "trial_id" not in out.columns:
        if "trial_id_x" in out.columns:
            out["trial_id"] = out["trial_id_x"]
        elif "trial_id_y" in out.columns:
            out["trial_id"] = out["trial_id_y"]
    if "fold_id" not in out.columns:
        if "fold_id_x" in out.columns:
            out["fold_id"] = out["fold_id_x"]
        elif "fold_id_y" in out.columns:
            out["fold_id"] = out["fold_id_y"]
    return out


def fold_column_for_plots(df: pd.DataFrame) -> str:
    """Hold-one-trial-out uses fold_id / heldout_trial; per-trial uses fold_id / trial_id."""
    if "fold_id" in df.columns and df["fold_id"].notna().any():
        return "fold_id"
    if "heldout_trial" in df.columns and df["heldout_trial"].notna().any():
        return "heldout_trial"
    if "trial_id" in df.columns and df["trial_id"].notna().any():
        return "trial_id"
    raise ValueError("No fold column found (expected fold_id, heldout_trial, or trial_id)")


def roc_plot_title(df: pd.DataFrame, fc: str) -> str:
    if fc == "fold_id" and len(df):
        s = df["fold_id"].dropna().astype(str)
        if s.str.startswith("holdout_trial").any():
            return "Hold-one-trial-out ROC (Test Set)"
        if s.str.startswith("trial_").any():
            return "Per-trial train/test ROC (Test Set)"
    if fc == "heldout_trial":
        return "Hold-one-trial-out ROC (Test Set)"
    return "ROC (Test Set)"


def format_fold_legend_label(fc: str, fold) -> str:
    if fc == "fold_id":
        return str(fold)
    try:
        return f"Trial {int(fold):03d}"
    except (TypeError, ValueError):
        return str(fold)


plt.rcParams['image.cmap'] = 'magma'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['legend.frameon'] = False



def plot_roc(preds: pd.DataFrame, out_path: Path):
    d = preds[(preds["split"] == "test") & (preds["model_name"] == "logreg_all_features")].copy()
    fc = fold_column_for_plots(d)
    folds = sorted(d[fc].dropna().unique(), key=lambda x: (str(type(x)), str(x)))
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("deep", n_colors=max(len(folds), 1))
    for color, fold in tqdm(list(zip(palette, folds)), desc="ROC folds", leave=False):
        df = d[d[fc] == fold]
        if df["y"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(df["y"], df["proba"])
        fold_auc = auc(fpr, tpr)
        fold_label = format_fold_legend_label(fc, fold)
        ax.plot(fpr, tpr, color=color, alpha=0.6, lw=1.8, label=f"{fold_label} (AUC={fold_auc:.3f})")
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    if not tprs:
        ax.text(0.5, 0.5, "No valid ROC folds (single class in y)", ha="center", va="center", transform=ax.transAxes)
    else:
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, color="black", lw=3, label=f"Mean ROC (AUC={mean_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#666666", lw=1)
    ax.set_title(roc_plot_title(d, fc))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if tprs:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_roc_mean_by_frequency_band(preds: pd.DataFrame, out_path: Path, split: str = "test"):
    """Mean ROC across holdout folds — one curve per narrow-band logistic (single band)."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    d = preds[(preds["split"] == split)].copy()
    fc = fold_column_for_plots(d)
    folds = sorted(d[fc].dropna().unique(), key=lambda x: (str(type(x)), str(x)))
    band_models = [f"logreg_{band}_power" for band in BAND_DEFS]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    mean_fpr = np.linspace(0.0, 1.0, 200)

    cmap = sns.color_palette("muted", n_colors=len(band_models))
    for bm, color in zip(band_models, cmap):
        if bm not in d["model_name"].values:
            continue
        fold_aucs = []
        tprs = []
        for fold in folds:
            df = d[(d[fc] == fold) & (d["model_name"] == bm)]
            if df["y"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(df["y"], df["proba"])
            fold_aucs.append(auc(fpr, tpr))
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        band_key = bm.replace("logreg_", "").replace("_power", "")
        if not tprs:
            continue
        tpr_arr = np.vstack(tprs)
        mean_tpr = np.mean(tpr_arr, axis=0)
        tpr_lo = np.clip(np.percentile(tpr_arr, 2.5, axis=0), 0.0, 1.0)
        tpr_hi = np.clip(np.percentile(tpr_arr, 97.5, axis=0), 0.0, 1.0)
        mean_auc = float(np.nanmean(fold_aucs)) if fold_aucs else np.nan
        ax.fill_between(mean_fpr, tpr_lo, tpr_hi, color=color, alpha=0.22, linewidth=0)
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2.6, alpha=0.95, label=f"{band_key}: mean AUC={mean_auc:.3f}")

    ax.plot([0, 1], [0, 1], "--", color="#666666", lw=1)
    ax.set_title(f"Mean ROC by frequency band — 95% CI across folds ({split})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    sns.despine(ax=ax)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_confusion(preds: pd.DataFrame, out_path: Path):
    d = preds[(preds["split"] == "test") & (preds["model_name"] == "logreg_all_features")].copy()
    cm = confusion_matrix(d["y"], d["pred"])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="gray_r")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black", fontsize=12)
    ax.set_xticks([0, 1], ["Control", "Onset"])
    ax.set_yticks([0, 1], ["Control", "Onset"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Combined Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_violin(features: pd.DataFrame, out_path: Path):
    d = features[features["split"] == "test"].copy()
    feature_cols = [c for c in d.columns if c.endswith("_power")]
    n = len(feature_cols)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.5 * n_rows), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    palette = {"control": "#6b7280", "onset": "#c2410c"}

    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        sub = d[["label", feature]].rename(columns={feature: "value"})
        sns.boxplot(
            data=sub,
            x="label",
            y="value",
            hue="label",
            linewidth=1,
            palette=palette,
            dodge=False,
            showfliers=False,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_title(feature.replace("_power", ""))
        ax.set_xlabel("")
        ax.set_ylabel("Power (log scale)")
        if ax.legend_ is not None:
            ax.legend_.remove()
        sns.despine(ax=ax)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Line2D([0], [0], color=palette["control"], lw=6),
        plt.Line2D([0], [0], color=palette["onset"], lw=6),
    ]
    fig.legend(handles, ["control", "onset"], loc="upper right", frameon=False)
    fig.suptitle("Test-Set Feature Distributions (independent y-axis per band)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot hold-out analysis outputs")
    parser.add_argument("--path-profile", type=str, choices=("local", "server"), default="local")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--fig-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = get_config(args.path_profile)
    if args.results_dir is None:
        args.results_dir = _SCRIPTS / "outputs" / "holdout"
    if args.fig_dir is None:
        args.fig_dir = args.results_dir / "figures"

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    # set_style()

    preds = pd.read_csv(args.results_dir / "predictions.csv")
    preds = normalize_prediction_columns(preds)
    features = pd.read_csv(args.results_dir / "samples_features.csv")

    for plot_name, fn, out in tqdm(
        [
            ("ROC", plot_roc, args.fig_dir / "roc_holdout_test.png"),
            (
                "ROC bands (mean folds, test)",
                lambda p, o: plot_roc_mean_by_frequency_band(p, o, "test"),
                args.fig_dir / "roc_bands_mean_fold_test.png",
            ),
            (
                "ROC bands (mean folds, train)",
                lambda p, o: plot_roc_mean_by_frequency_band(p, o, "train"),
                args.fig_dir / "roc_bands_mean_fold_train.png",
            ),
            ("Confusion", plot_confusion, args.fig_dir / "confusion_matrix_test_gray.png"),
            ("Violin", plot_violin, args.fig_dir / "feature_violins_test_log.png"),
        ],
        desc="Generating plots",
    ):
        if plot_name == "Violin":
            fn(features, out)
        else:
            fn(preds, out)

    print("Saved figures:")
    print(f"- {args.fig_dir / 'roc_holdout_test.png'}")
    print(f"- {args.fig_dir / 'roc_bands_mean_fold_test.png'}")
    print(f"- {args.fig_dir / 'roc_bands_mean_fold_train.png'}")
    print(f"- {args.fig_dir / 'confusion_matrix_test_gray.png'}")
    print(f"- {args.fig_dir / 'feature_violins_test_log.png'}")


if __name__ == "__main__":
    main()
