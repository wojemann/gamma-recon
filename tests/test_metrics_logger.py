"""Tests for the MetricsLogger seam.

Each backend is exercised against the same minimal interface contract:
log_config -> log_step (multiple) -> log_summary -> close. Wandb is
covered via a fake module injected into ``sys.modules`` so the test
doesn't need the real dependency installed.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from gamma_encoder.training.logging import (
    JsonlLogger,
    MetricsLogger,
    MultiLogger,
    StdoutLogger,
)


# ---------------------------------------------------------------------------
# StdoutLogger
# ---------------------------------------------------------------------------


def test_stdout_logger_throttles_steps(capsys):
    lg = StdoutLogger(log_every=10)
    lg.log_config({"lr": 1e-3})
    for step in range(1, 21):
        lg.log_step(step, {"loss": 1.0 / step})
    lg.log_summary({"final_loss": 0.05})

    out = capsys.readouterr().out
    # Step 1 always logs; then every 10th. So steps {1, 10, 20}.
    assert "step      1" in out
    assert "step     10" in out
    assert "step     20" in out
    assert "step      2" not in out
    assert "step      9" not in out
    assert "config:" in out
    assert "summary:" in out


def test_stdout_logger_rejects_zero_log_every():
    with pytest.raises(ValueError):
        StdoutLogger(log_every=0)


# ---------------------------------------------------------------------------
# JsonlLogger
# ---------------------------------------------------------------------------


def test_jsonl_logger_writes_files(tmp_path: Path):
    lg = JsonlLogger(tmp_path)
    lg.log_config({"lr": 1e-3, "tokenizer": "dilated_cnn"})
    lg.log_step(1, {"loss": 0.5})
    lg.log_step(2, {"loss": 0.4})
    lg.log_summary({"final_loss": 0.4, "n_params": 12345})
    lg.close()

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["lr"] == 1e-3
    assert cfg["tokenizer"] == "dilated_cnn"

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["final_loss"] == 0.4
    assert summary["n_params"] == 12345

    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["step"] == 1 and rec0["loss"] == 0.5
    rec1 = json.loads(lines[1])
    assert rec1["step"] == 2 and rec1["loss"] == 0.4


def test_jsonl_logger_handles_torch_scalar(tmp_path: Path):
    """Torch / numpy scalars must be coerced via .item() so json works."""
    torch = pytest.importorskip("torch")
    lg = JsonlLogger(tmp_path)
    lg.log_step(1, {"loss": torch.tensor(0.123)})
    lg.close()
    rec = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert rec["loss"] == pytest.approx(0.123, rel=1e-5)


def test_jsonl_logger_close_is_idempotent(tmp_path: Path):
    lg = JsonlLogger(tmp_path)
    lg.log_step(1, {"loss": 0.0})
    lg.close()
    lg.close()  # must not raise


# ---------------------------------------------------------------------------
# WandbLogger (fake module to avoid real dependency)
# ---------------------------------------------------------------------------


class _FakeRun:
    def __init__(self):
        self.finished = False

    def finish(self):
        self.finished = True


class _FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.init_args = None
        self.config = types.SimpleNamespace(
            update=lambda d, allow_val_change=False: self._config_store.update(d)
        )
        self.summary = {}
        self.logged = []
        self._config_store: dict = {}
        self._run = _FakeRun()

    def init(self, **kwargs):
        self.init_args = kwargs
        if kwargs.get("config"):
            self._config_store.update(kwargs["config"])
        return self._run

    def log(self, metrics, step=None):
        self.logged.append((step, dict(metrics)))


def test_wandb_logger_routes_through_fake_module(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    from gamma_encoder.training.logging import WandbLogger

    lg = WandbLogger(project="p", run_name="r", config={"lr": 1e-3})
    lg.log_config({"tokenizer": "dilated_cnn"})
    lg.log_step(5, {"loss": 0.42})
    lg.log_summary({"final_loss": 0.1})
    lg.close()

    assert fake.init_args["project"] == "p"
    assert fake.init_args["name"] == "r"
    assert fake._config_store["lr"] == 1e-3
    assert fake._config_store["tokenizer"] == "dilated_cnn"
    assert fake.logged == [(5, {"loss": 0.42})]
    assert fake.summary["final_loss"] == 0.1
    assert fake._run.finished is True


# ---------------------------------------------------------------------------
# MultiLogger
# ---------------------------------------------------------------------------


class _RecordingLogger(MetricsLogger):
    def __init__(self):
        self.events: list = []

    def log_config(self, config):
        self.events.append(("config", dict(config)))

    def log_step(self, step, metrics):
        self.events.append(("step", step, dict(metrics)))

    def log_summary(self, summary):
        self.events.append(("summary", dict(summary)))

    def close(self):
        self.events.append(("close",))


def test_multi_logger_fans_out():
    a, b = _RecordingLogger(), _RecordingLogger()
    multi = MultiLogger([a, b])
    multi.log_config({"x": 1})
    multi.log_step(1, {"loss": 0.5})
    multi.log_summary({"y": 2})
    multi.close()
    assert a.events == b.events
    assert a.events[0] == ("config", {"x": 1})
    assert a.events[1] == ("step", 1, {"loss": 0.5})
    assert a.events[2] == ("summary", {"y": 2})
    assert a.events[3] == ("close",)


# ---------------------------------------------------------------------------
# Integration with run_overfit
# ---------------------------------------------------------------------------


def test_run_overfit_calls_logger(tmp_path: Path):
    """Smoke test: the overfit harness drives the logger through the
    full lifecycle on a tiny synthetic batch."""
    torch = pytest.importorskip("torch")
    from gamma_encoder.training.overfit import run_overfit

    # Build a minimal synthetic batch matching cache_overfit_batch's payload.
    B, C, T = 1, 2, 1024
    payload = {
        "segments": torch.randn(B, C, T),
        "region_ids": torch.randint(0, 4, (C,)),
        "fs": 2048.0,
    }
    batch_path = tmp_path / "batch.pt"
    torch.save(payload, batch_path)

    rec = _RecordingLogger()
    rep = run_overfit(
        batch_path=batch_path,
        steps=3,
        d_model=16,
        n_layers=1,
        n_heads=2,
        log_every=1,
        logger=rec,
    )

    kinds = [e[0] for e in rec.events]
    assert kinds[0] == "config"
    assert kinds.count("step") == 3
    assert "summary" in kinds
    # User-supplied logger must NOT be closed by the harness.
    assert ("close",) not in rec.events
    assert rep.steps == 3
    assert rep.initial_loss > 0
