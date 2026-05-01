"""Band-resolved NMSE for every saved sweep checkpoint.

Loads each ``results/overfit_runs/<run>/model.pt``, reconstructs the
cached overfit batch, and computes band-resolved NMSE via
:mod:`gamma_eval` (delta-theta, alpha-beta, low-gamma, high-gamma).

Outputs:
- ``results/overfit_runs/band_eval.csv`` — wide table, one row per
  run, columns include final train loss + per-band NMSE means.
- ``results/overfit_runs/band_eval_loss_axis.png`` —
  (loss × band) heatmap, dilated_cnn tokenizer.
- ``results/overfit_runs/band_eval_tokenizer_axis.png`` —
  (tokenizer × band) heatmap, mse loss.
- ``results/overfit_runs/band_eval_linear_ar_axis.png`` —
  (loss × band) heatmap, linear_ar model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gamma_encoder.models.full_model import GammaEncoderConfig, GammaEncoderModel
from gamma_encoder.models.linear_ar import LinearARModel
from gamma_encoder.training.overfit import build_tokenizer
from gamma_eval.evaluator import ReconstructionEvaluator
from gamma_eval.metrics.reconstruction import DEFAULT_BANDS


def _load_model(ckpt_path: Path, segments: torch.Tensor, region_ids: torch.Tensor):
    """Reconstruct the model from a saved checkpoint."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_type = blob.get("model_type", "transformer")
    cfg_dict = blob["config"]
    if model_type == "transformer":
        _, C, T = segments.shape
        patch_samples = cfg_dict["patch_samples"]
        cfg = GammaEncoderConfig(
            d_model=cfg_dict["d_model"],
            n_layers=cfg_dict["n_layers"],
            n_heads=cfg_dict["n_heads"],
            patch_samples=patch_samples,
            num_regions=max(int(region_ids.max().item()) + 1, 64),
            max_seq_len=C * (T // patch_samples) + 16,
        )
        tokenizer = build_tokenizer(blob["tokenizer"], cfg)
        model = GammaEncoderModel(tokenizer, cfg)
    elif model_type == "linear_ar":
        model = LinearARModel(num_channels=segments.shape[1], order=int(cfg_dict.get("ar_order") or 3))
    else:
        raise ValueError(f"unknown model_type {model_type}")
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


_EVAL_MASK_SEED = 12345


def _sample_eval_region_mask(B: int, region_ids: torch.Tensor, k_regions: int) -> torch.Tensor:
    """Deterministic eval region-mask: same seed across all configs for fair comparison."""
    C = region_ids.shape[0]
    unique = torch.unique(region_ids)
    n_unique = unique.numel()
    g = torch.Generator().manual_seed(_EVAL_MASK_SEED)
    mask = torch.zeros(B, C, dtype=torch.bool)
    for b in range(B):
        perm = torch.randperm(n_unique, generator=g)[:k_regions]
        chosen = unique[perm]
        in_chosen = (region_ids.unsqueeze(0) == chosen.unsqueeze(1)).any(dim=0)
        mask[b] = in_chosen
    return mask


def _eval_run(run_dir: Path, segments: torch.Tensor, region_ids: torch.Tensor, fs: float) -> dict:
    ckpt = run_dir / "model.pt"
    summary_path = run_dir / "summary.json"
    if not ckpt.exists() or not summary_path.exists():
        return {}
    with open(summary_path) as f:
        summary = json.load(f)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg_dict = blob["config"]
    mask_k = int(cfg_dict.get("mask_n_regions") or 0)
    model = _load_model(ckpt, segments, region_ids)

    B, C, T = segments.shape
    if mask_k > 0:
        mask = _sample_eval_region_mask(B, region_ids, mask_k)
        with torch.no_grad():
            recon = model(segments, region_ids, mask_channels=mask)
        # Score on masked channels only. Region-mask uses same regions
        # for every batch element (same region_ids), so each row has the
        # same number of masked channels — pack as (B, n_masked_C, T).
        sel = mask  # bool (B, C)
        n_masked_c = int(mask[0].sum().item())
        true_packed = segments[sel].view(B, n_masked_c, T)
        pred_packed = recon[sel].view(B, n_masked_c, T)
    else:
        with torch.no_grad():
            recon = model(segments, region_ids)
        true_packed = segments
        pred_packed = recon
    true_np = true_packed.numpy()
    pred_np = pred_packed.numpy()
    evaluator = ReconstructionEvaluator(fs=fs)
    evaluator.accumulate(true_np, pred_np)
    metrics = evaluator.summarize()
    return {
        "run": run_dir.name,
        "final_loss": summary["final_loss"],
        "initial_loss": summary["initial_loss"],
        "mask_n_regions": mask_k,
        **{k: v for k, v in metrics.items() if k.startswith("nmse_")},
        "log_spec_dist_gamma_mean": metrics.get("log_spec_dist_gamma_mean"),
    }


def _heatmap(df: pd.DataFrame, row_col: str, row_order: list[str], title: str, out_path: Path) -> None:
    bands = list(DEFAULT_BANDS.keys())
    cols = [f"nmse_{b}_mean" for b in bands]
    sub = df.set_index(row_col).loc[[r for r in row_order if r in df[row_col].values], cols]
    if sub.empty:
        print(f"  [skip heatmap] no matching rows for {row_col}")
        return
    fig, ax = plt.subplots(figsize=(1.0 + 1.0 * len(bands), 0.4 + 0.4 * len(sub)))
    im = ax.imshow(sub.values, aspect="auto", cmap="magma_r", vmin=0, vmax=2.0)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands, rotation=30, ha="right")
    ax.set_yticks(range(len(sub.index)))
    ax.set_yticklabels(sub.index)
    ax.set_title(title)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 1.0 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="NMSE (clipped at 2.0)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, default=Path("results/overfit_batch.pt"))
    p.add_argument("--runs-root", type=Path, default=Path("results/overfit_runs"))
    p.add_argument("--out-csv", type=Path, default=Path("results/overfit_runs/band_eval.csv"))
    args = p.parse_args()

    payload = torch.load(args.batch, map_location="cpu", weights_only=False)
    segments = payload["segments"].float()
    region_ids = payload["region_ids"].long()
    fs = float(payload["fs"])

    rows = []
    for run_dir in sorted(args.runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "model.pt").exists():
            continue
        print(f"eval {run_dir.name} ...")
        row = _eval_run(run_dir, segments, region_ids, fs)
        if row:
            rows.append(row)
            band_str = "  ".join(
                f"{b}={row[f'nmse_{b}_mean']:.3f}" for b in DEFAULT_BANDS
            )
            print(f"  final_loss={row['final_loss']:.4f}  {band_str}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}\n")

    # Add structured columns for splitting.
    def _split(name: str) -> tuple[str, str]:
        if "__" in name:
            loss, model = name.split("__", 1)
            return loss, model
        return name, ""
    df[["loss", "model"]] = df["run"].apply(lambda x: pd.Series(_split(x)))

    LOSSES = [
        "mse", "mae", "huber", "whitened_mse", "log_power_spectral",
        "multires_stft", "eegm2", "distdf", "cmin_logcosh", "content_aware_l1",
    ]
    TOKENIZERS = [
        "dilated_cnn", "linear", "stft_magnitude", "complex_stft",
        "wavelet_packet", "welch_psd",
    ]

    # Loss-axis: pick whichever transformer tokenizer has the most loss
    # rows (since the loss-axis sweep can pick any tokenizer; default is
    # "linear" post-channel-mask).
    tok_counts = df[df["model"].isin(TOKENIZERS)].groupby("model").size()
    loss_axis_tok = tok_counts.idxmax() if len(tok_counts) else "linear"
    loss_df = df[df["model"] == loss_axis_tok].copy()
    loss_df["row"] = loss_df["loss"]
    _heatmap(
        loss_df, "row", LOSSES,
        f"Loss-axis NMSE per band (tokenizer={loss_axis_tok})",
        args.runs_root / "band_eval_loss_axis.png",
    )

    tok_df = df[(df["loss"] == "mse") & df["model"].isin(TOKENIZERS)].copy()
    tok_df["row"] = tok_df["model"]
    _heatmap(
        tok_df, "row", TOKENIZERS,
        "Tokenizer-axis NMSE per band (loss=mse)",
        args.runs_root / "band_eval_tokenizer_axis.png",
    )

    ar_df = df[df["model"] == "linear_ar"].copy()
    ar_df["row"] = ar_df["loss"]
    _heatmap(
        ar_df, "row", LOSSES,
        "Linear-VAR NMSE per band (MVAR(3), C^2 p + C params)",
        args.runs_root / "band_eval_linear_ar_axis.png",
    )


if __name__ == "__main__":
    main()
