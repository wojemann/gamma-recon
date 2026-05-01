"""Per-run reconstruction-vs-truth plots.

For each ``results/overfit_runs/<run>/model.pt``, loads the model,
reconstructs the cached overfit batch, and writes
``results/overfit_runs/<run>/reconstruction.png`` showing one segment
(all 8 channels, true vs predicted overlaid).

By default we plot segment 0. Use ``--segment-idx`` to pick a different
one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gamma_encoder.models.full_model import GammaEncoderConfig, GammaEncoderModel
from gamma_encoder.models.linear_ar import LinearARModel
from gamma_encoder.training.overfit import build_tokenizer


def _load_model(ckpt_path: Path, segments: torch.Tensor, region_ids: torch.Tensor):
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
    return model, model_type, blob.get("loss"), blob.get("tokenizer")


def _plot(true_seg: np.ndarray, pred_seg: np.ndarray, fs: float,
          title: str, out_path: Path,
          masked_channels: np.ndarray | None = None) -> None:
    """true_seg, pred_seg: (C, T). masked_channels: (C,) bool of which
    channels were hidden from the encoder (highlighted in title)."""
    C, T = true_seg.shape
    t = np.arange(T) / fs
    fig, axes = plt.subplots(C, 1, figsize=(10, 1.4 * C), sharex=True)
    if C == 1:
        axes = [axes]
    for c in range(C):
        ax = axes[c]
        ax.plot(t, true_seg[c], lw=0.8, color="black", label="true" if c == 0 else None)
        ax.plot(t, pred_seg[c], lw=0.8, color="tab:red", alpha=0.85,
                label="pred" if c == 0 else None)
        suffix = "  [MASKED]" if masked_channels is not None and bool(masked_channels[c]) else ""
        ax.set_ylabel(f"ch {c}{suffix}", fontsize=8)
        if masked_channels is not None and bool(masked_channels[c]):
            ax.set_facecolor((1.0, 0.95, 0.85))  # cream tint on masked rows
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, default=Path("results/overfit_batch.pt"))
    p.add_argument("--runs-root", type=Path, default=Path("results/overfit_runs"))
    p.add_argument("--segment-idx", type=int, default=0)
    args = p.parse_args()

    payload = torch.load(args.batch, map_location="cpu", weights_only=False)
    segments = payload["segments"].float()                # (B, C, T)
    region_ids = payload["region_ids"].long()
    fs = float(payload["fs"])
    seg_idx = int(args.segment_idx)

    written = 0
    for run_dir in sorted(args.runs_root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "model.pt").exists():
            continue
        blob = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
        mask_k = int((blob.get("config") or {}).get("mask_n_regions") or 0)
        model, model_type, loss_name, tok_name = _load_model(
            run_dir / "model.pt", segments, region_ids
        )
        B, C, T = segments.shape
        masked_row = None
        if mask_k > 0:
            # Same eval seed as run_band_eval — single fixed mask for plotting.
            unique = torch.unique(region_ids)
            n_unique = unique.numel()
            g = torch.Generator().manual_seed(12345)
            mask = torch.zeros(B, C, dtype=torch.bool)
            for b in range(B):
                perm = torch.randperm(n_unique, generator=g)[:mask_k]
                chosen = unique[perm]
                in_chosen = (region_ids.unsqueeze(0) == chosen.unsqueeze(1)).any(dim=0)
                mask[b] = in_chosen
            masked_row = mask[seg_idx].numpy()
            with torch.no_grad():
                recon = model(segments, region_ids, mask_channels=mask)
        else:
            with torch.no_grad():
                recon = model(segments, region_ids)
        true_seg = segments[seg_idx].numpy()
        pred_seg = recon[seg_idx].numpy()
        label = tok_name if model_type == "transformer" else model_type
        out_path = run_dir / "reconstruction.png"
        title = f"{loss_name} + {label}  (segment {seg_idx})"
        if mask_k > 0:
            n_masked_c = int(mask[seg_idx].sum().item())
            title += f"  | mask {mask_k} regions ({n_masked_c}/{C} ch)"
        _plot(true_seg, pred_seg, fs, title=title, out_path=out_path,
              masked_channels=masked_row)
        print(f"wrote {out_path}")
        written += 1
    print(f"\n{written} plots written")


if __name__ == "__main__":
    main()
