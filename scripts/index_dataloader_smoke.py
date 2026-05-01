"""Integration smoke test: segment indices -> h5 -> faithful BaRISTA forward+backward.

Exercises the path that the server-stage trainer will use. Skips
preprocessing for the smoke (production loaders should preprocess once
per session and slice; per-segment notch+reref produces edge artifacts
on 3-s windows). The point here is to catch shape/dtype/integration
bugs before the server-launch round-trip.

Steps:
  1. Open ``segment_indices/sub_02/trial000__pretrain.parquet``.
  2. Take the first ``--batch`` rows.
  3. Open the trial h5; for each row, slice [start_sample : end_sample]
     across all atlas-resolvable, Laplacian-eligible channels (using
     ``BrainTreeTrial`` for the channel keep-set + region IDs but NOT
     its preprocessing).
  4. Stack into ``(B, C, T)`` tensor and z-score per (segment, channel)
     so values are in the regime the model was trained on.
  5. Run through ``GammaEncoderModel(encoder_kind="faithful")`` at
     laptop dims.
  6. MSE recon loss + backward; report stats.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u -m scripts.index_dataloader_smoke --profile local
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from configs.config import get_config
from gamma_encoder.data.braintreebank import BrainTreeTrial
from gamma_encoder.models.full_model import GammaEncoderConfig, GammaEncoderModel
from gamma_encoder.tokenizers.dilated_cnn import DilatedCNNTokenizer


def parse_session_id(session_id: str) -> tuple[int, int]:
    # "sub_02_trial_000" -> (2, 0)
    parts = session_id.split("_")
    return int(parts[1]), int(parts[3])


def slice_h5_batch(
    h5_path: Path,
    starts: np.ndarray,
    ends: np.ndarray,
    electrode_indices: list[int],
) -> np.ndarray:
    """Return (B, C, T) float64 array. T must be constant across rows."""
    T = int(ends[0] - starts[0])
    assert all(ends - starts == T), "non-constant window length"
    B = len(starts)
    C = len(electrode_indices)
    out = np.empty((B, C, T), dtype=np.float64)
    with h5py.File(h5_path, "r") as f:
        grp = f["data"]
        # Loop over channels (outer) since each channel is its own dataset.
        for c, eidx in enumerate(electrode_indices):
            ds = grp[f"electrode_{eidx}"]
            for b in range(B):
                out[b, c] = ds[int(starts[b]) : int(ends[b])]
    return out


def zscore_per_segment(x: np.ndarray) -> np.ndarray:
    # x: (B, C, T)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-8
    return (x - mean) / std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="local", choices=["local", "server"])
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-channels", type=int, default=8,
                        help="Subset of kept channels for the smoke (full ~150 is slow).")
    args = parser.parse_args()

    cfg = get_config(args.profile)
    bt_root = Path(cfg.braintree_data_root)
    repo_root = Path(__file__).resolve().parents[1]
    parquet_path = (
        repo_root / "segment_indices" / "sub_02"
        / f"trial{args.trial:03d}__pretrain.parquet"
    )
    print(f"loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("start_sample").head(args.batch).reset_index(drop=True)
    print(f"  {len(df)} rows, columns={list(df.columns)}")
    print(df[["session_id", "split", "start_sample", "end_sample", "label"]].to_string(index=False))

    # Resolve channel keep-set + region IDs via existing loader (no preprocessing).
    sub_id, trial_id = parse_session_id(df.iloc[0]["session_id"])
    assert trial_id == args.trial
    trial = BrainTreeTrial(subject_id=sub_id, trial_id=trial_id, data_root=bt_root)
    summary = trial.summary()
    print(f"\ntrial summary: {summary}")

    # Subset for smoke speed: take the first N kept channels.
    kept_names = trial.kept_names[: args.n_channels]
    region_by_name = {n: trial.region_vocab[trial.region_by_name[n]] for n in kept_names}
    raw_idx_by_name = {n: i for i, n in enumerate(trial.electrode_names)}
    electrode_indices = [raw_idx_by_name[n] for n in kept_names]
    region_ids = np.array([region_by_name[n] for n in kept_names], dtype=np.int64)
    print(f"using {len(kept_names)} channels: {kept_names[:5]}... region_ids={region_ids.tolist()}")

    # Slice h5 by parquet index ranges.
    starts = df["start_sample"].to_numpy()
    ends = df["end_sample"].to_numpy()
    h5_path = bt_root / f"sub_{sub_id}_trial{trial_id:03d}.h5"
    t0 = time.perf_counter()
    raw = slice_h5_batch(h5_path, starts, ends, electrode_indices)
    t_slice = time.perf_counter() - t0
    print(f"\nh5 slice: shape={raw.shape} dtype={raw.dtype} took {t_slice:.2f}s")
    print(f"  per-segment stats (raw): mean={raw.mean():+.3e}  std={raw.std():.3e}  "
          f"min={raw.min():+.3e}  max={raw.max():+.3e}")

    # Z-score per (segment, channel) so values are model-regime.
    seg = zscore_per_segment(raw).astype(np.float32)
    segments = torch.from_numpy(seg)
    region_t = torch.from_numpy(region_ids)
    B, C, T = segments.shape
    print(f"  z-scored: shape={tuple(segments.shape)} mean={segments.mean():+.3e} std={segments.std():.3e}")

    # Build faithful model at laptop dims.
    cfg_kwargs = dict(
        d_model=32, n_layers=6, n_heads=2, ff_mult=4,
        patch_samples=512,
        num_regions=max(int(region_t.max().item()) + 1, 64),
        max_seq_len=C * (T // 512) + 16,
        encoder_kind="faithful",
    )
    torch.manual_seed(0)
    model_cfg = GammaEncoderConfig(**cfg_kwargs)
    tokenizer = DilatedCNNTokenizer(d_model=model_cfg.d_model, patch_samples=model_cfg.patch_samples)
    model = GammaEncoderModel(tokenizer, model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nbuilt faithful encoder: {n_params:,} params (laptop dims)")

    # Forward + backward.
    model.train()
    t0 = time.perf_counter()
    recon = model(segments, region_t)
    loss = ((recon - segments) ** 2).mean()
    loss.backward()
    t_fb = time.perf_counter() - t0
    grad_max = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
    print(f"\nforward+backward took {t_fb:.2f}s")
    print(f"  recon shape={tuple(recon.shape)} mean={recon.mean().item():+.4f} std={recon.std().item():.4f}")
    print(f"  loss (MSE vs input z-score): {loss.item():.4f}")
    print(f"  max |grad|: {grad_max:.3e}")

    # Cross-check: end-start consistency on every row, and batch within h5 bounds.
    assert (ends - starts == 6144).all(), "non-uniform window length"
    with h5py.File(h5_path, "r") as f:
        n_samples = f["data"]["electrode_0"].shape[0]
    assert (ends <= n_samples).all() and (starts >= 0).all(), "out-of-bounds index"
    print(f"\nall {len(df)} rows within h5 bounds [0, {n_samples}); window length 6144 confirmed")

    print("\nOK: indices -> h5 -> faithful encoder -> loss -> backward integration verified")


if __name__ == "__main__":
    main()
