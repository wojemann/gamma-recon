"""Smoke test for the faithful BaRISTA-style encoder.

Steps:
  1. Build our `GammaEncoderModel` with `encoder_kind="faithful"` at our
     laptop default (d=32, L=6, H=2). Forward + backward on the cached
     overfit batch with seed=0. Report param count, output stats, MSE
     vs target.
  2. Build the same model at BaRISTA's dimensions (d=64, L=12, H=4,
     mlp_ratio=4). Forward + backward, same input, seed=0. Report
     same stats — this is the "BaRISTA-architecture random-init" arm
     you'd compare a checkpoint-loaded run against.
  3. Inspect `pretrained_models/parcels_chans.ckpt` directly: list
     param-name groups + shapes. Cross-check which of our faithful
     model's parameters have a same-shape counterpart in the ckpt
     (structural parity score).
  4. Train arm (1) for 10 steps with MSE on the cached batch; print
     loss curve and save it to results/faithful_smoke/loss_curve.png.

Outputs:
  - stdout: all comparisons.
  - results/faithful_smoke/loss_curve.png
  - results/faithful_smoke/parity_report.txt
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from gamma_encoder.models.full_model import GammaEncoderConfig, GammaEncoderModel
from gamma_encoder.tokenizers.dilated_cnn import DilatedCNNTokenizer


BARISTA_CKPT = Path("/Users/wojemann/local_data/BaRISTA/pretrained_models/parcels_chans.ckpt")
BATCH_PATH = Path("results/overfit_batch.pt")
OUT_DIR = Path("results/faithful_smoke")


def _build(cfg_kwargs: dict, seed: int = 0) -> GammaEncoderModel:
    torch.manual_seed(seed)
    cfg = GammaEncoderConfig(encoder_kind="faithful", **cfg_kwargs)
    tok = DilatedCNNTokenizer(d_model=cfg.d_model, patch_samples=cfg.patch_samples)
    return GammaEncoderModel(tok, cfg)


def _stats(t: torch.Tensor) -> str:
    return (f"shape={tuple(t.shape)} mean={t.mean().item():+.4f} "
            f"std={t.std().item():.4f} min={t.min().item():+.4f} "
            f"max={t.max().item():+.4f}")


def _forward_backward(model: nn.Module, segments: torch.Tensor,
                       region_ids: torch.Tensor, label: str) -> dict:
    print(f"\n--- {label} ---")
    n = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: total={n:,} trainable={n_trainable:,}")
    model.train()
    recon = model(segments, region_ids)
    loss = ((recon - segments) ** 2).mean()
    loss.backward()
    grad_max = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
    print(f"  forward: recon {_stats(recon.detach())}")
    print(f"  loss (MSE vs input): {loss.item():.6f}")
    print(f"  max |grad|: {grad_max:.4e}")
    return {
        "params": n,
        "loss": loss.item(),
        "recon_mean": recon.mean().item(),
        "recon_std": recon.std().item(),
        "grad_max": grad_max,
    }


def _structural_parity(model: nn.Module, ckpt_path: Path) -> str:
    """Compare our faithful model's parameter shapes to the BaRISTA ckpt.

    Returns a multi-line report string and the parity score
    (#our_params with same-shape match in ckpt / #our_params).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # ckpt is a flat dict[str, Tensor]
    ckpt_shapes: dict[str, tuple] = {k: tuple(v.shape) for k, v in ckpt.items()}

    our = {k: tuple(v.shape) for k, v in model.state_dict().items()}

    # Group by leading prefix for compactness.
    def _top(name: str) -> str:
        return ".".join(name.split(".")[:3])

    our_groups: dict[str, list[tuple[str, tuple]]] = {}
    for k, s in our.items():
        our_groups.setdefault(_top(k), []).append((k, s))
    ckpt_groups: dict[str, list[tuple[str, tuple]]] = {}
    for k, s in ckpt_shapes.items():
        ckpt_groups.setdefault(_top(k), []).append((k, s))

    # Shape-match check: does each of our shapes appear in ckpt anywhere?
    ckpt_shape_multiset: dict[tuple, int] = {}
    for s in ckpt_shapes.values():
        ckpt_shape_multiset[s] = ckpt_shape_multiset.get(s, 0) + 1
    our_shape_multiset: dict[tuple, int] = {}
    for s in our.values():
        our_shape_multiset[s] = our_shape_multiset.get(s, 0) + 1

    matched = 0
    avail = dict(ckpt_shape_multiset)
    for s, count in our_shape_multiset.items():
        take = min(count, avail.get(s, 0))
        matched += take
        if s in avail:
            avail[s] -= take
    n_our = sum(our_shape_multiset.values())
    parity = matched / max(n_our, 1)

    lines = []
    lines.append(f"BaRISTA ckpt: {ckpt_path}")
    lines.append(f"  total tensors: {len(ckpt_shapes)}, "
                 f"total params: {sum(int(torch.tensor(s).prod().item()) for s in ckpt_shapes.values()):,}")
    lines.append(f"  top-level groups (count of tensors per group):")
    for g, items in sorted(ckpt_groups.items()):
        lines.append(f"    {g}: {len(items)}")
    lines.append("")
    lines.append("Our faithful model state_dict:")
    lines.append(f"  total tensors: {len(our)}, "
                 f"total params: {sum(int(torch.tensor(s).prod().item()) for s in our.values()):,}")
    lines.append(f"  top-level groups:")
    for g, items in sorted(our_groups.items()):
        lines.append(f"    {g}: {len(items)}")
    lines.append("")
    lines.append(f"Structural shape-parity (multiset shape match):")
    lines.append(f"  {matched} / {n_our} = {parity:.1%}")
    lines.append("")
    lines.append("Our tensors WITHOUT a same-shape ckpt counterpart:")
    avail2 = dict(ckpt_shape_multiset)
    for k, s in our.items():
        if avail2.get(s, 0) > 0:
            avail2[s] -= 1
        else:
            lines.append(f"  {k}: {s}")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = torch.load(BATCH_PATH, map_location="cpu", weights_only=False)
    segments = payload["segments"].float()
    region_ids = payload["region_ids"].long()
    fs = float(payload["fs"])
    B, C, T = segments.shape
    print(f"batch: B={B} C={C} T={T} fs={fs}")

    # Step 1: our default-size faithful model.
    laptop_kwargs = dict(
        d_model=32, n_layers=6, n_heads=2, ff_mult=4,
        patch_samples=512, num_regions=max(int(region_ids.max().item()) + 1, 64),
        max_seq_len=C * (T // 512) + 16,
    )
    model_ours = _build(laptop_kwargs, seed=0)
    out_ours = _forward_backward(model_ours, segments, region_ids,
                                  label="ours @ laptop dims (faithful, d=32 L=6 H=2)")

    # Step 2: our model at BaRISTA dimensions.
    barista_kwargs = dict(
        d_model=64, n_layers=12, n_heads=4, ff_mult=4,
        patch_samples=512, num_regions=max(int(region_ids.max().item()) + 1, 64),
        max_seq_len=C * (T // 512) + 16,
    )
    model_barista_dims = _build(barista_kwargs, seed=0)
    out_barista = _forward_backward(model_barista_dims, segments, region_ids,
                                     label="ours @ BaRISTA dims (faithful, d=64 L=12 H=4) — random init seed=0")

    # Step 3: structural parity vs the actual BaRISTA ckpt.
    print("\n--- Structural parity vs BaRISTA pretrained ckpt ---")
    if BARISTA_CKPT.exists():
        report = _structural_parity(model_barista_dims, BARISTA_CKPT)
        print(report)
        (OUT_DIR / "parity_report.txt").write_text(report)
        print(f"\nwrote {OUT_DIR / 'parity_report.txt'}")
    else:
        print(f"ckpt not found at {BARISTA_CKPT} — skipped")

    # Step 4: 10-step loss curve on arm 1.
    print("\n--- 10-step loss curve (ours @ laptop dims) ---")
    torch.manual_seed(0)
    model = _build(laptop_kwargs, seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses: list[float] = []
    model.train()
    for step in range(10):
        recon = model(segments, region_ids)
        loss = ((recon - segments) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"  step {step:02d}  loss={loss.item():.6f}")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(len(losses)), losses, marker="o", color="tab:blue")
    ax.set_xlabel("step"); ax.set_ylabel("MSE loss")
    ax.set_title("Faithful encoder (laptop dims) — 10-step smoke")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_curve.png", dpi=110)
    plt.close(fig)
    print(f"\nwrote {OUT_DIR / 'loss_curve.png'}")

    print("\nSummary:")
    print(f"  ours laptop:    init_loss={out_ours['loss']:.4f}  "
          f"final(after 10 steps)={losses[-1]:.4f}  "
          f"params={out_ours['params']:,}")
    print(f"  ours barista:   init_loss={out_barista['loss']:.4f}  "
          f"params={out_barista['params']:,}")


if __name__ == "__main__":
    main()
