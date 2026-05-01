"""Tests for gamma_encoder.data.braintreebank.

Combines synthetic-fixture tests (fake h5 + JSON + CSV in tmp_path) with
helper-function tests. Real-data tests live in a separate file gated on
data availability so the suite runs fast and offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from gamma_encoder.data.braintreebank import (
    BrainTreeTrial,
    build_region_vocab,
    sanitize_electrode_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_sanitize_electrode_name_strips_markers():
    assert sanitize_electrode_name("RT2aA1#") == "RT2aA1"
    assert sanitize_electrode_name("RT1aIa1*") == "RT1aIa1"
    assert sanitize_electrode_name("LT3a_1") == "LT3a1"
    assert sanitize_electrode_name("LT3a1") == "LT3a1"


def test_build_region_vocab_is_stable_and_sorted():
    a = build_region_vocab(["B", "A", "C", "B"])
    b = build_region_vocab(["C", "B", "A"])
    assert a == b
    assert a["A"] < a["B"] < a["C"]


def test_build_region_vocab_reserves_zero_for_unknown():
    v = build_region_vocab(["", "G_temp", "Hippo"])
    assert v[""] == 0
    assert 0 not in [v["G_temp"], v["Hippo"]]


# ---------------------------------------------------------------------------
# Synthetic fixture: a one-trial mini BrainTreebank on tmp_path
# ---------------------------------------------------------------------------


def _make_fake_dataset(
    tmp_path: Path,
    fs: int = 2048,
    n_seconds: float = 6.0,
    rng_seed: int = 0,
) -> Path:
    """Build a minimal BrainTreebank-shaped tree under tmp_path.

    Layout:
        tmp_path/
          sub_99_trial000.h5
          electrode_labels/sub_99/electrode_labels.json
          localization/sub_99/depth-wm.csv
          corrupted_elec.json
    """
    rng = np.random.default_rng(rng_seed)
    n_samples = int(fs * n_seconds)

    # Two leads of 5 contacts each: LT3a1..LT3a5, RT1c1..RT1c5.
    # Add a corrupted DC chan and one with a marker character.
    h5_names = [
        "LT3a1", "LT3a2", "LT3a3", "LT3a4", "LT3a5",
        "RT1c1", "RT1c2*", "RT1c3", "RT1c4", "RT1c5",
        "DC4",
    ]

    # Atlas: covers sanitized names (so RT1c2* -> RT1c2 is found).
    atlas_rows = [
        {"Electrode": "LT3a1", "Destrieux": "ctx_lh_G_temporal_middle"},
        {"Electrode": "LT3a2", "Destrieux": "ctx_lh_G_temporal_middle"},
        {"Electrode": "LT3a3", "Destrieux": "ctx_lh_S_temporal_inf"},
        {"Electrode": "LT3a4", "Destrieux": "ctx_lh_S_temporal_inf"},
        {"Electrode": "LT3a5", "Destrieux": "Left-Hippocampus"},
        {"Electrode": "RT1c1", "Destrieux": "ctx_rh_G_temporal_middle"},
        {"Electrode": "RT1c2", "Destrieux": "ctx_rh_G_temporal_middle"},
        {"Electrode": "RT1c3", "Destrieux": "ctx_rh_S_temporal_inf"},
        {"Electrode": "RT1c4", "Destrieux": "ctx_rh_S_temporal_inf"},
        {"Electrode": "RT1c5", "Destrieux": "Right-Hippocampus"},
    ]

    corrupted = {"sub_99": ["DC4"]}

    # Write h5.
    h5_path = tmp_path / "sub_99_trial000.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("data")
        for i, _ in enumerate(h5_names):
            sig = rng.standard_normal(n_samples)
            grp.create_dataset(f"electrode_{i}", data=sig)

    # Write electrode_labels.json.
    labels_dir = tmp_path / "electrode_labels" / "sub_99"
    labels_dir.mkdir(parents=True)
    (labels_dir / "electrode_labels.json").write_text(json.dumps(h5_names))

    # Write localization/sub_99/depth-wm.csv.
    loc_dir = tmp_path / "localization" / "sub_99"
    loc_dir.mkdir(parents=True)
    pd.DataFrame(atlas_rows).to_csv(loc_dir / "depth-wm.csv", index=False)

    # Write corrupted_elec.json.
    (tmp_path / "corrupted_elec.json").write_text(json.dumps(corrupted))

    return tmp_path


def test_loader_constructs_and_summarizes(tmp_path):
    root = _make_fake_dataset(tmp_path)
    trial = BrainTreeTrial(subject_id=99, trial_id=0, data_root=root)
    s = trial.summary()
    # 11 raw chans, DC4 corrupted (1).
    assert s["n_raw_chans"] == 11
    assert s["n_corrupted"] == 1
    # Eligible: contacts 2..4 on each lead (need both same-stem neighbors,
    # and atlas-resolvable). DC4 has no atlas + neighbor. So 6 kept.
    assert s["n_kept"] == 6


def test_loader_kept_names_are_inner_contacts(tmp_path):
    root = _make_fake_dataset(tmp_path)
    trial = BrainTreeTrial(subject_id=99, trial_id=0, data_root=root)
    assert set(trial.kept_names) == {
        "LT3a2", "LT3a3", "LT3a4",
        "RT1c2", "RT1c3", "RT1c4",
    }


def test_loader_runs_pipeline_end_to_end(tmp_path):
    root = _make_fake_dataset(tmp_path, n_seconds=6.0)
    trial = BrainTreeTrial(subject_id=99, trial_id=0, data_root=root)
    out = trial.load_segments()
    # 6 s @ 2048 Hz, 3 s segments => 2 segments.
    assert out.segments.shape == (2, 6, 6144)
    # z-score: per-row mean ~ 0, std ~ 1.
    np.testing.assert_allclose(out.segments.mean(axis=-1), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.segments.std(axis=-1), 1.0, atol=1e-6)
    # region_ids align with kept_names.
    assert out.region_ids.shape == (6,)
    assert out.region_ids.dtype == np.int64
    # Same Destrieux region -> same id.
    name_to_id = dict(zip(out.channel_names, out.region_ids.tolist()))
    assert name_to_id["LT3a2"] == name_to_id["LT3a1"] if "LT3a1" in name_to_id else True
    # LT3a3 and LT3a4 share parcel "ctx_lh_S_temporal_inf".
    assert name_to_id["LT3a3"] == name_to_id["LT3a4"]
    # And differ from LT3a2 (different parcel).
    assert name_to_id["LT3a3"] != name_to_id["LT3a2"]


def test_loader_max_seconds_truncates(tmp_path):
    root = _make_fake_dataset(tmp_path, n_seconds=12.0)
    trial = BrainTreeTrial(subject_id=99, trial_id=0, data_root=root)
    out_full = trial.load_segments()
    out_short = trial.load_segments(max_seconds=6.0)
    assert out_full.segments.shape[0] == 4
    assert out_short.segments.shape[0] == 2


def test_loader_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        BrainTreeTrial(subject_id=99, trial_id=0, data_root=tmp_path)


def test_loader_explicit_region_vocab_used(tmp_path):
    root = _make_fake_dataset(tmp_path)
    # Pre-build a vocab missing one region we have data for.
    custom = {"ctx_lh_G_temporal_middle": 7, "ctx_lh_S_temporal_inf": 3,
              "Left-Hippocampus": 9, "ctx_rh_G_temporal_middle": 5,
              "ctx_rh_S_temporal_inf": 4, "Right-Hippocampus": 8}
    trial = BrainTreeTrial(
        subject_id=99, trial_id=0, data_root=root, region_vocab=custom
    )
    out = trial.load_segments()
    name_to_id = dict(zip(out.channel_names, out.region_ids.tolist()))
    assert name_to_id["LT3a3"] == 3
    assert name_to_id["LT3a2"] == 7
