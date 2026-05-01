"""Single-batch overfit harness.

Loads a cached (segments, region_ids) batch from disk and trains the
model on the same fixed bytes for N steps. Reports per-step train loss
and (optionally) per-band NMSE via :mod:`gamma_eval`.

Use this to verify that any new (loss, tokenizer) combo can drive train
loss toward zero on a tiny batch — i.e. the pipeline is sound. If a
config can't overfit, there is a bug or a fundamental incompatibility,
and that's a result worth catching before scaling up.

Progress / metrics are pushed through a :class:`MetricsLogger` seam, so
this loop never hard-codes stdout vs jsonl vs wandb. Default backend
combo for laptop runs is stdout + jsonl when ``out_dir`` is given.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gamma_encoder.losses.base import ReconstructionLoss
from gamma_encoder.losses.cmin_logcosh import CMinLogCoshLoss
from gamma_encoder.losses.content_aware_l1 import ContentAwareL1Loss
from gamma_encoder.losses.distdf import DistDFLoss
from gamma_encoder.losses.eegm2 import EEGM2Loss
from gamma_encoder.losses.log_power_spectral import LogPowerSpectralL1Loss
from gamma_encoder.losses.mse import MSELoss
from gamma_encoder.losses.multires_stft import MultiResolutionSTFTLoss
from gamma_encoder.losses.robust import HuberLoss, MAELoss
from gamma_encoder.losses.whitened_mse import WhitenedMSELoss
from gamma_encoder.models.full_model import GammaEncoderConfig, GammaEncoderModel
from gamma_encoder.models.linear_ar import LinearARModel
from gamma_encoder.tokenizers.base import Tokenizer
from gamma_encoder.tokenizers.complex_stft import ComplexSTFTTokenizer
from gamma_encoder.tokenizers.dilated_cnn import DilatedCNNTokenizer
from gamma_encoder.tokenizers.linear import LinearTokenizer
from gamma_encoder.tokenizers.stft_magnitude import STFTMagnitudeTokenizer
from gamma_encoder.tokenizers.wavelet_packet import WaveletPacketTokenizer
from gamma_encoder.tokenizers.welch_psd import WelchPSDTokenizer
from gamma_encoder.training.logging import (
    JsonlLogger,
    MetricsLogger,
    MultiLogger,
    StdoutLogger,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def build_tokenizer(name: str, cfg: GammaEncoderConfig) -> Tokenizer:
    if name == "dilated_cnn":
        return DilatedCNNTokenizer(d_model=cfg.d_model, patch_samples=cfg.patch_samples)
    if name == "linear":
        return LinearTokenizer(d_model=cfg.d_model, patch_samples=cfg.patch_samples)
    if name == "stft_magnitude":
        return STFTMagnitudeTokenizer(d_model=cfg.d_model, patch_samples=cfg.patch_samples)
    if name == "complex_stft":
        return ComplexSTFTTokenizer(d_model=cfg.d_model, patch_samples=cfg.patch_samples)
    if name == "wavelet_packet":
        return WaveletPacketTokenizer(d_model=cfg.d_model, patch_samples=cfg.patch_samples)
    if name == "welch_psd":
        return WelchPSDTokenizer(d_model=cfg.d_model, patch_samples=cfg.patch_samples)
    raise ValueError(f"unknown tokenizer: {name}")


_WHITENED_MSE_PSD_PATH = Path("results/whitened_mse_psd.pt")


def build_loss(name: str) -> ReconstructionLoss:
    if name == "mse":
        return MSELoss()
    if name == "mae":
        return MAELoss()
    if name == "huber":
        return HuberLoss()
    if name == "whitened_mse":
        psd = None
        if _WHITENED_MSE_PSD_PATH.exists():
            psd = torch.load(_WHITENED_MSE_PSD_PATH, map_location="cpu", weights_only=True)
        return WhitenedMSELoss(psd=psd)
    if name == "log_power_spectral":
        return LogPowerSpectralL1Loss()
    if name == "multires_stft":
        return MultiResolutionSTFTLoss()
    if name == "eegm2":
        return EEGM2Loss()
    if name == "distdf":
        return DistDFLoss()
    if name == "cmin_logcosh":
        return CMinLogCoshLoss()
    if name == "content_aware_l1":
        return ContentAwareL1Loss()
    raise ValueError(f"unknown loss: {name}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class OverfitReport:
    tokenizer: str
    loss: str
    n_params: int
    initial_loss: float
    final_loss: float
    min_loss: float
    steps: int
    seconds: float
    loss_trace: list = field(default_factory=list)


def _default_logger(out_dir: Optional[Path], log_every: int) -> MetricsLogger:
    """Stdout-only when no ``out_dir``; stdout + jsonl otherwise."""
    backends: list[MetricsLogger] = [StdoutLogger(log_every=log_every, prefix="  ")]
    if out_dir is not None:
        backends.append(JsonlLogger(out_dir))
    return MultiLogger(backends) if len(backends) > 1 else backends[0]


def _sample_channel_mask(
    B: int, C: int, k: int, generator: torch.Generator, device: str
) -> torch.Tensor:
    """Random per-segment mask: pick k of C channels per batch element.

    Returns (B, C) bool tensor with exactly k True entries per row.
    """
    mask = torch.zeros(B, C, dtype=torch.bool)
    for b in range(B):
        idx = torch.randperm(C, generator=generator)[:k]
        mask[b, idx] = True
    return mask.to(device)


def run_overfit(
    batch_path: Path,
    tokenizer_name: str = "dilated_cnn",
    loss_name: str = "mse",
    model_type: str = "transformer",
    steps: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    log_every: int = 25,
    seed: int = 0,
    device: str = "cpu",
    d_model: int = 32,
    n_layers: int = 6,
    n_heads: int = 2,
    ar_order: int = 3,
    mask_n_channels: int = 0,
    out_dir: Optional[Path] = None,
    logger: Optional[MetricsLogger] = None,
) -> OverfitReport:
    torch.manual_seed(seed)

    # Loading torch.save'd dict written from numpy in cache_overfit_batch.
    payload = torch.load(batch_path, map_location="cpu", weights_only=False)
    segments = payload["segments"].to(device)            # (B, C, T) float32
    region_ids = payload["region_ids"].to(device)        # (C,) long
    fs = float(payload["fs"])
    patch_samples = 512
    if segments.shape[-1] % patch_samples != 0:
        raise RuntimeError(f"segment length {segments.shape[-1]} not divisible by {patch_samples}")
    B, C, T = segments.shape

    if model_type == "transformer":
        cfg = GammaEncoderConfig(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            patch_samples=patch_samples,
            num_regions=max(int(region_ids.max().item()) + 1, 64),
            max_seq_len=C * (T // patch_samples) + 16,
        )
        tokenizer = build_tokenizer(tokenizer_name, cfg)
        model = GammaEncoderModel(tokenizer, cfg).to(device)
    elif model_type == "linear_ar":
        model = LinearARModel(num_channels=C, order=ar_order).to(device)
        tokenizer_name = "none"
    else:
        raise ValueError(f"unknown model_type: {model_type}")
    loss_fn = build_loss(loss_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    owns_logger = logger is None
    if logger is None:
        logger = _default_logger(out_dir, log_every)

    run_config = {
        "model_type": model_type,
        "ar_order": ar_order if model_type == "linear_ar" else None,
        "tokenizer": tokenizer_name,
        "loss": loss_name,
        "steps": steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "device": device,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "patch_samples": patch_samples,
        "batch_shape": list(segments.shape),
        "fs": fs,
        "n_params": n_params,
        "mask_n_channels": mask_n_channels,
    }
    logger.log_config(run_config)

    trace: list = []
    model.train()
    t0 = time.time()
    init_loss: Optional[float] = None
    min_loss = float("inf")
    masking = mask_n_channels > 0
    if masking and mask_n_channels >= C:
        raise ValueError(f"mask_n_channels={mask_n_channels} must be < C={C}")
    mask_gen = torch.Generator().manual_seed(seed + 1)
    try:
        for step in range(1, steps + 1):
            if masking:
                mask = _sample_channel_mask(B, C, mask_n_channels, mask_gen, device)
                recon = model(segments, region_ids, mask_channels=mask)
                # Loss only on the masked channels' waveforms — variable
                # rows per batch element, so flatten via boolean indexing.
                pred_m = recon[mask]            # (B*k, T)
                true_m = segments[mask]         # (B*k, T)
                # Add a leading "channel" axis so loss_fn's (B, C, T) contract holds.
                loss = loss_fn(pred_m.unsqueeze(1), true_m.unsqueeze(1))
            else:
                recon = model(segments, region_ids)
                loss = loss_fn(recon, segments)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            v = float(loss.detach().cpu().item())
            trace.append((step, v))
            if init_loss is None:
                init_loss = v
            min_loss = min(min_loss, v)
            logger.log_step(step, {"loss": v})

        elapsed = time.time() - t0
        report = OverfitReport(
            tokenizer=tokenizer_name,
            loss=loss_name,
            n_params=n_params,
            initial_loss=float(init_loss),
            final_loss=float(trace[-1][1]),
            min_loss=float(min_loss),
            steps=steps,
            seconds=float(elapsed),
            loss_trace=trace,
        )

        logger.log_summary(
            {
                "initial_loss": report.initial_loss,
                "final_loss": report.final_loss,
                "min_loss": report.min_loss,
                "seconds": report.seconds,
                "n_params": report.n_params,
            }
        )

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"overfit_{tokenizer_name}_{loss_name}.json"
            with open(out_path, "w") as f:
                json.dump(asdict(report), f, indent=2)
            print(f"\nwrote {out_path}")
            ckpt_path = out_dir / "model.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": model_type,
                    "tokenizer": tokenizer_name,
                    "loss": loss_name,
                    "config": run_config,
                },
                ckpt_path,
            )
            print(f"wrote {ckpt_path}")
    finally:
        if owns_logger:
            logger.close()

    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--tokenizer", default="dilated_cnn")
    p.add_argument("--loss", default="mse")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()

    rep = run_overfit(
        batch_path=args.batch,
        tokenizer_name=args.tokenizer,
        loss_name=args.loss,
        steps=args.steps,
        lr=args.lr,
        out_dir=args.out_dir,
        device=args.device,
        log_every=args.log_every,
    )
    print(f"\ninit={rep.initial_loss:.6f}  final={rep.final_loss:.6f}  min={rep.min_loss:.6f}")
    print(f"params={rep.n_params:,}  elapsed={rep.seconds:.1f}s")


if __name__ == "__main__":
    main()
