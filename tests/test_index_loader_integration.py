"""Integration tests: segment-index parquets -> h5 -> faithful BaRISTA.

Real-data integration. Skips automatically when the BrainTreebank h5
files or the committed segment_indices parquets aren't available, so
the suite still runs fast and offline elsewhere. The convention follows
``test_braintreebank_loader.py``'s split between fast synthetic tests
and gated real-data tests.

What this catches:
  - parquet schema drift (column rename, dtype change)
  - start_sample / end_sample drifting outside h5 bounds
  - faithful encoder shape contract changes (patchify, region embed,
    decoder)
  - z-scored segments producing non-finite outputs through the model
  - integration regressions between BrainTreeTrial channel-resolution
    and the parquets' assumed channel pool
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from gamma_encoder.data.braintreebank import BrainTreeTrial
from gamma_encoder.models.full_model import GammaEncoderConfig, GammaEncoderModel
from gamma_encoder.tokenizers.dilated_cnn import DilatedCNNTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
SEG_ROOT = REPO_ROOT / "segment_indices" / "sub_02"
# Hardcoded laptop path; the test gates on file existence, so it skips cleanly
# anywhere the data isn't mounted (e.g., CI, server before the data sync).
BT_ROOT = Path("/Users/wojemann/local_data/BrainTree")
H5_PATH = BT_ROOT / "sub_2_trial000.h5"
PARQUET_PATH = SEG_ROOT / "trial000__pretrain.parquet"

pytestmark = pytest.mark.skipif(
    not H5_PATH.exists() or not PARQUET_PATH.exists(),
    reason=(
        "Real BrainTreebank h5 + committed segment_indices parquets required; "
        "this is a laptop-only integration test."
    ),
)


def _z_per_segment(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-8
    return (x - mean) / std


def _slice_h5_batch(h5_path, starts, ends, electrode_indices):
    T = int(ends[0] - starts[0])
    out = np.empty((len(starts), len(electrode_indices), T), dtype=np.float64)
    with h5py.File(h5_path, "r") as f:
        grp = f["data"]
        for c, eidx in enumerate(electrode_indices):
            ds = grp[f"electrode_{eidx}"]
            for b, (s, e) in enumerate(zip(starts, ends)):
                out[b, c] = ds[int(s) : int(e)]
    return out


def test_parquet_schema_has_expected_columns():
    df = pd.read_parquet(PARQUET_PATH)
    expected = {
        "session_id", "subject_id", "trial_id", "movie", "task",
        "session_role", "split", "split_seed",
        "start_sample", "end_sample",
        "label", "center_sample", "source_word_idx", "notes",
    }
    assert expected.issubset(set(df.columns)), \
        f"missing columns: {expected - set(df.columns)}"
    assert (df["end_sample"] - df["start_sample"] == 6144).all()
    assert df["task"].nunique() == 1 and df["task"].iloc[0] == "pretrain"


def test_index_rows_within_h5_bounds():
    df = pd.read_parquet(PARQUET_PATH)
    with h5py.File(H5_PATH, "r") as f:
        n_samples = f["data"]["electrode_0"].shape[0]
    assert (df["start_sample"] >= 0).all()
    assert (df["end_sample"] <= n_samples).all()


def test_index_to_faithful_encoder_forward_backward():
    """End-to-end: parquet -> h5 slice -> z-score -> faithful encoder
    -> MSE recon loss -> backward. Asserts finite outputs and grads.
    """
    df = pd.read_parquet(PARQUET_PATH).sort_values("start_sample").head(2)
    starts = df["start_sample"].to_numpy()
    ends = df["end_sample"].to_numpy()

    trial = BrainTreeTrial(subject_id=2, trial_id=0, data_root=BT_ROOT)
    kept = trial.kept_names[:4]  # tiny channel set for speed
    raw_idx_by_name = {n: i for i, n in enumerate(trial.electrode_names)}
    eidx = [raw_idx_by_name[n] for n in kept]
    region_ids = np.array(
        [trial.region_vocab[trial.region_by_name[n]] for n in kept],
        dtype=np.int64,
    )

    raw = _slice_h5_batch(H5_PATH, starts, ends, eidx)
    seg = _z_per_segment(raw).astype(np.float32)

    segments = torch.from_numpy(seg)
    region_t = torch.from_numpy(region_ids)
    B, C, T = segments.shape
    assert (B, T) == (2, 6144) and C == 4

    torch.manual_seed(0)
    cfg = GammaEncoderConfig(
        d_model=32, n_layers=2, n_heads=2, ff_mult=2,
        patch_samples=512,
        num_regions=max(int(region_t.max().item()) + 1, 64),
        max_seq_len=C * (T // 512) + 8,
        encoder_kind="faithful",
    )
    tok = DilatedCNNTokenizer(d_model=cfg.d_model, patch_samples=cfg.patch_samples)
    model = GammaEncoderModel(tok, cfg)
    model.train()

    recon = model(segments, region_t)
    assert recon.shape == segments.shape
    assert torch.isfinite(recon).all().item()

    loss = ((recon - segments) ** 2).mean()
    assert torch.isfinite(loss).item()
    loss.backward()
    grad_max = max(
        p.grad.abs().max().item()
        for p in model.parameters() if p.grad is not None
    )
    assert np.isfinite(grad_max) and grad_max > 0


def test_zscored_segments_are_in_model_regime():
    """A regression check: raw h5 amplitudes are O(100) microvolts, so
    feeding raw values would saturate the encoder. Confirm our slicing
    + z-score path keeps inputs near unit scale.
    """
    df = pd.read_parquet(PARQUET_PATH).sort_values("start_sample").head(2)
    starts = df["start_sample"].to_numpy()
    ends = df["end_sample"].to_numpy()

    trial = BrainTreeTrial(subject_id=2, trial_id=0, data_root=BT_ROOT)
    kept = trial.kept_names[:4]
    raw_idx_by_name = {n: i for i, n in enumerate(trial.electrode_names)}
    eidx = [raw_idx_by_name[n] for n in kept]

    raw = _slice_h5_batch(H5_PATH, starts, ends, eidx)
    seg = _z_per_segment(raw)
    # mean ~ 0, std ~ 1 per segment
    assert np.abs(seg.mean(axis=-1)).max() < 1e-6
    assert np.abs(seg.std(axis=-1) - 1.0).max() < 1e-3
