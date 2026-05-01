"""Run a small overfit sweep over (loss, tokenizer) combos.

Reads the cached batch at ``results/overfit_batch.pt`` and runs each
configuration through :func:`gamma_encoder.training.overfit.run_overfit`
for ``--steps`` steps. Per-run artifacts land under
``results/overfit_runs/<loss>__<tokenizer>/``; comparison plots and a
summary CSV land under ``results/overfit_runs/``.

Default sweep: loss-axis ablation (10 losses with dilated_cnn) and
tokenizer-axis ablation (6 tokenizers with mse), 1000 steps each.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gamma_encoder.training.overfit import run_overfit


# Order matters for the comparison plots: keep MSE first so it anchors
# the legend.
LOSSES = [
    "mse",
    "mae",
    "huber",
    "whitened_mse",
    "log_power_spectral",
    "multires_stft",
    "eegm2",
    "distdf",
    "cmin_logcosh",
    "content_aware_l1",
]

TOKENIZERS = [
    "dilated_cnn",
    "linear",
    "stft_magnitude",
    "complex_stft",
    "wavelet_packet",
    "welch_psd",
]


def _load_metrics(run_dir: Path) -> pd.DataFrame:
    """Load per-step ``loss`` from ``metrics.jsonl``."""
    rows = []
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            if "step" in obj and "loss" in obj:
                rows.append({"step": obj["step"], "loss": obj["loss"]})
    return pd.DataFrame(rows)


def _plot_curve(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["step"], df["loss"], lw=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_axis_comparison(
    runs: List[Tuple[str, Path]], title: str, out_path: Path
) -> None:
    """Overlay loss curves for a list of (label, run_dir) pairs.

    Each loss has its own scale, so we plot per step normalized to its
    own initial value (``loss / loss_at_step_1``). That makes "fraction
    of initial loss reached" comparable across loss families.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, run_dir in runs:
        df = _load_metrics(run_dir)
        if df.empty:
            continue
        normalized = df["loss"] / df["loss"].iloc[0]
        ax.plot(df["step"], normalized, lw=1.2, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("loss / initial loss")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _run_one(
    loss_name: str,
    tokenizer_name: str,
    batch_path: Path,
    out_root: Path,
    steps: int,
    lr: float,
    mask_n_channels: int,
    model_type: str = "transformer",
) -> Path:
    label = tokenizer_name if model_type == "transformer" else model_type
    run_dir = out_root / f"{loss_name}__{label}"
    if (run_dir / "summary.json").exists():
        print(f"[skip] {run_dir.name} already has summary.json")
        return run_dir
    print(f"\n=== {loss_name} + {label} (mask {mask_n_channels} ch) ===")
    rep = run_overfit(
        batch_path=batch_path,
        tokenizer_name=tokenizer_name,
        loss_name=loss_name,
        model_type=model_type,
        steps=steps,
        lr=lr,
        mask_n_channels=mask_n_channels,
        out_dir=run_dir,
        device="cpu",
        log_every=max(1, steps // 20),
    )
    df = _load_metrics(run_dir)
    _plot_curve(df, f"{loss_name} + {tokenizer_name}", run_dir / "loss_curve.png")
    print(
        f"  init={rep.initial_loss:.4f}  final={rep.final_loss:.4f}  "
        f"min={rep.min_loss:.4f}  params={rep.n_params:,}  {rep.seconds:.0f}s"
    )
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--batch",
        type=Path,
        default=Path("results/overfit_batch.pt"),
    )
    p.add_argument("--out-root", type=Path, default=Path("results/overfit_runs"))
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--axis",
        choices=("all", "loss", "tokenizer", "linear_ar"),
        default="all",
        help="which axis of the sweep to run",
    )
    p.add_argument(
        "--mask-n-channels", type=int, default=4,
        help="number of channels to mask per segment per step (channel-mask "
             "pretraining). 0 = full reconstruction (legacy)."
    )
    p.add_argument(
        "--loss-axis-tokenizer", default="linear",
        help="tokenizer used for the loss-axis sweep (default: linear — fast)"
    )
    args = p.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    completed: list[Tuple[str, str, Path]] = []  # (loss, tokenizer, run_dir)

    if args.axis in ("all", "loss"):
        for loss in LOSSES:
            run_dir = _run_one(
                loss_name=loss,
                tokenizer_name=args.loss_axis_tokenizer,
                batch_path=args.batch,
                out_root=args.out_root,
                steps=args.steps,
                lr=args.lr,
                mask_n_channels=args.mask_n_channels,
            )
            completed.append((loss, args.loss_axis_tokenizer, run_dir))

    if args.axis in ("all", "tokenizer"):
        for tok in TOKENIZERS:
            if tok == args.loss_axis_tokenizer and args.axis == "all":
                # already covered as mse + <loss_axis_tokenizer> in the loss axis
                continue
            run_dir = _run_one(
                loss_name="mse",
                tokenizer_name=tok,
                batch_path=args.batch,
                out_root=args.out_root,
                steps=args.steps,
                lr=args.lr,
                mask_n_channels=args.mask_n_channels,
            )
            completed.append(("mse", tok, run_dir))

    if args.axis in ("all", "linear_ar"):
        for loss in LOSSES:
            run_dir = _run_one(
                loss_name=loss,
                tokenizer_name="dilated_cnn",  # ignored
                batch_path=args.batch,
                out_root=args.out_root,
                steps=args.steps,
                lr=args.lr,
                mask_n_channels=args.mask_n_channels,
                model_type="linear_ar",
            )
            completed.append((loss, "linear_ar", run_dir))

    summary_rows: list[dict] = []
    for loss, tok, run_dir in completed:
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)
        with open(run_dir / "config.json") as f:
            config = json.load(f)
        summary_rows.append(
            {
                "loss": loss,
                "tokenizer": tok,
                "initial_loss": summary["initial_loss"],
                "final_loss": summary["final_loss"],
                "min_loss": summary["min_loss"],
                "frac_remaining": summary["final_loss"] / summary["initial_loss"],
                "n_params": summary["n_params"],
                "seconds": summary["seconds"],
                "steps": config["steps"],
            }
        )
    df = pd.DataFrame(summary_rows)
    csv_path = args.out_root / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")
    print(df.to_string(index=False))

    if args.axis in ("all", "loss"):
        loss_runs = [
            (loss, args.out_root / f"{loss}__{args.loss_axis_tokenizer}")
            for loss in LOSSES
        ]
        loss_runs = [(l, p) for l, p in loss_runs if (p / "metrics.jsonl").exists()]
        if loss_runs:
            _plot_axis_comparison(
                loss_runs,
                f"Loss-axis sweep (tokenizer={args.loss_axis_tokenizer}, "
                f"{args.steps} steps, mask={args.mask_n_channels})",
                args.out_root / "comparison_loss_axis.png",
            )
            print(f"wrote {args.out_root / 'comparison_loss_axis.png'}")

    if args.axis in ("all", "tokenizer"):
        tok_runs = [(tok, args.out_root / f"mse__{tok}") for tok in TOKENIZERS]
        tok_runs = [(t, p) for t, p in tok_runs if (p / "metrics.jsonl").exists()]
        if tok_runs:
            _plot_axis_comparison(
                tok_runs,
                f"Tokenizer-axis sweep (loss=mse, {args.steps} steps)",
                args.out_root / "comparison_tokenizer_axis.png",
            )
            print(f"wrote {args.out_root / 'comparison_tokenizer_axis.png'}")

    if args.axis in ("all", "linear_ar"):
        ar_runs = [(loss, args.out_root / f"{loss}__linear_ar") for loss in LOSSES]
        ar_runs = [(l, p) for l, p in ar_runs if (p / "metrics.jsonl").exists()]
        if ar_runs:
            _plot_axis_comparison(
                ar_runs,
                f"Linear-AR loss sweep ({args.steps} steps)",
                args.out_root / "comparison_linear_ar_axis.png",
            )
            print(f"wrote {args.out_root / 'comparison_linear_ar_axis.png'}")


if __name__ == "__main__":
    main()
