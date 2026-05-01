"""Pluggable metrics-logging backends for training runs.

The training loop only knows about :class:`MetricsLogger`. Backends
(stdout, jsonl on disk, wandb) sit behind that interface so we can swap
or compose them without touching training code.

Interface
---------

- ``log_config(config)`` — once at the start of a run, with hyperparams.
- ``log_step(step, metrics)`` — every step (or every k steps).
- ``log_summary(summary)`` — once at the end, with aggregate stats.
- ``close()`` — flush / close any open files or remote handles.

Backends
--------

- :class:`StdoutLogger` — formatted prints, throttled by ``log_every``.
- :class:`JsonlLogger` — one JSON object per line in ``metrics.jsonl``;
  ``config.json`` and ``summary.json`` live next to it.
- :class:`WandbLogger` — lazy ``wandb`` import; only the import touches
  the dependency, so the rest of the code stays wandb-free on the laptop.
- :class:`MultiLogger` — fan out to several backends at once (e.g.,
  stdout + jsonl during a server run).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


class MetricsLogger(ABC):
    """Minimal logger interface used by the training loops."""

    @abstractmethod
    def log_config(self, config: Mapping[str, Any]) -> None:
        ...

    @abstractmethod
    def log_step(self, step: int, metrics: Mapping[str, float]) -> None:
        ...

    @abstractmethod
    def log_summary(self, summary: Mapping[str, Any]) -> None:
        ...

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class StdoutLogger(MetricsLogger):
    """Throttled-print logger.

    Prints config once, prints metrics every ``log_every`` steps (and on
    step 1), prints the summary once at the end. Format is intentionally
    boring so it composes well with shell pipes and tee.
    """

    def __init__(self, log_every: int = 25, prefix: str = "") -> None:
        if log_every < 1:
            raise ValueError("log_every must be >= 1")
        self.log_every = log_every
        self.prefix = prefix

    def log_config(self, config: Mapping[str, Any]) -> None:
        print(f"{self.prefix}config: {json.dumps(dict(config), default=str)}")

    def log_step(self, step: int, metrics: Mapping[str, float]) -> None:
        if step != 1 and step % self.log_every != 0:
            return
        body = "  ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())
        print(f"{self.prefix}step {step:6d}  {body}")

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        print(f"{self.prefix}summary: {json.dumps(dict(summary), default=str)}")


class JsonlLogger(MetricsLogger):
    """Writes metrics to ``<out_dir>/metrics.jsonl`` plus config/summary JSON.

    One JSON object per line for step metrics — easy to slice with
    ``pandas.read_json(..., lines=True)`` or ``jq``. ``config.json`` and
    ``summary.json`` are written as standalone files at run boundaries.
    """

    def __init__(self, out_dir: Path | str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.out_dir / "metrics.jsonl"
        self._fh = open(self._metrics_path, "w")

    def log_config(self, config: Mapping[str, Any]) -> None:
        with open(self.out_dir / "config.json", "w") as f:
            json.dump(dict(config), f, indent=2, default=str)

    def log_step(self, step: int, metrics: Mapping[str, float]) -> None:
        record = {"step": int(step), **{k: _to_json_safe(v) for k, v in metrics.items()}}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        with open(self.out_dir / "summary.json", "w") as f:
            json.dump(dict(summary), f, indent=2, default=str)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class WandbLogger(MetricsLogger):
    """Lazy-imported wandb backend.

    Importing this module never imports wandb; the dependency is touched
    only when the backend is instantiated. Useful for server runs where
    the laptop env doesn't have wandb installed.
    """

    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        import wandb  # noqa: WPS433 - intentional lazy import

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=run_name,
            entity=entity,
            config=dict(config) if config else None,
        )

    def log_config(self, config: Mapping[str, Any]) -> None:
        self._wandb.config.update(dict(config), allow_val_change=True)

    def log_step(self, step: int, metrics: Mapping[str, float]) -> None:
        self._wandb.log(dict(metrics), step=int(step))

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        for k, v in summary.items():
            self._wandb.summary[k] = _to_json_safe(v)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


class MultiLogger(MetricsLogger):
    """Fan-out logger that broadcasts every call to all backends."""

    def __init__(self, loggers: Iterable[MetricsLogger]) -> None:
        self.loggers = list(loggers)

    def log_config(self, config: Mapping[str, Any]) -> None:
        for lg in self.loggers:
            lg.log_config(config)

    def log_step(self, step: int, metrics: Mapping[str, float]) -> None:
        for lg in self.loggers:
            lg.log_step(step, metrics)

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        for lg in self.loggers:
            lg.log_summary(summary)

    def close(self) -> None:
        for lg in self.loggers:
            lg.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _to_json_safe(v: Any) -> Any:
    # numpy / torch scalars come through as objects with .item().
    item = getattr(v, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # pragma: no cover - defensive
            pass
    return v
