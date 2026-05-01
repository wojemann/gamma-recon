"""Cache a small fixed batch from Subject 2 for overfit-sweep experiments.

Selects ``n_chans`` channels with the most diverse Destrieux parcels we
can get, takes the first ``n_segments`` non-overlapping 3-s windows from
trial 0, and saves the result as a torch ``.pt`` file under
``results/overfit_batch.pt``.

Why a fixed batch: every (loss, tokenizer) configuration must train
against identical bytes for the overfit sweep to be a fair comparison.

Run:

    KMP_DUPLICATE_LIB_OK=TRUE python -m scripts.cache_overfit_batch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from configs.config import get_config
from gamma_encoder.data.braintreebank import BrainTreeTrial


def select_diverse_channels(
    region_by_name: dict,
    kept_names: list,
    n: int,
    seed: int = 0,
) -> list:
    """Pick ``n`` channels covering as many distinct Destrieux regions as possible.

    Greedy: walk a shuffled order, accept a channel if its region hasn't
    been picked yet. If we run out of new regions before reaching ``n``,
    fill with the next channels regardless of region.
    """
    rng = np.random.default_rng(seed)
    order = list(kept_names)
    rng.shuffle(order)

    picked: list = []
    seen_regions: set = set()
    for name in order:
        region = region_by_name[name]
        if region not in seen_regions:
            picked.append(name)
            seen_regions.add(region)
            if len(picked) == n:
                return picked
    # Fallback fill.
    for name in order:
        if name not in picked:
            picked.append(name)
            if len(picked) == n:
                break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="local", choices=["local", "server"])
    parser.add_argument("--subject", type=int, default=2)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--n-chans", type=int, default=8)
    parser.add_argument("--n-segments", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .pt path. Defaults to <repo>/results/overfit_batch.pt.",
    )
    args = parser.parse_args()

    cfg = get_config(args.profile)
    repo_root = Path(__file__).resolve().parents[1]
    out_path = args.out or (repo_root / "results" / "overfit_batch.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trial = BrainTreeTrial(
        subject_id=args.subject,
        trial_id=args.trial,
        data_root=cfg.braintree_data_root,
    )
    print("trial summary:", trial.summary())

    # Need >= n_segments * 3 s of data; pad a touch for the IIR transient.
    needed_seconds = args.n_segments * 3.0 + 5.0
    out = trial.load_segments(max_seconds=needed_seconds)
    if out.segments.shape[0] < args.n_segments:
        raise RuntimeError(
            f"Only got {out.segments.shape[0]} segments; need {args.n_segments}"
        )

    picked = select_diverse_channels(
        region_by_name=trial.region_by_name,
        kept_names=out.channel_names,
        n=args.n_chans,
        seed=args.seed,
    )
    chan_idx = np.array([out.channel_names.index(n) for n in picked])
    seg = out.segments[: args.n_segments][:, chan_idx, :]  # (n_seg, n_chan, T)
    region_ids = out.region_ids[chan_idx]
    regions_picked = [trial.region_by_name[n] for n in picked]

    payload = {
        "segments": torch.from_numpy(seg).float(),  # (16, 8, 6144) float32
        "region_ids": torch.from_numpy(region_ids).long(),  # (8,)
        "channel_names": picked,
        "regions": regions_picked,
        "region_vocab": trial.region_vocab,
        "fs": float(out.fs),
        "segment_samples": int(out.segment_samples),
        "subject_id": args.subject,
        "trial_id": args.trial,
        "selection_seed": args.seed,
    }
    torch.save(payload, out_path)
    print(f"\nsaved {out_path}")
    print(f"  segments shape: {tuple(payload['segments'].shape)} {payload['segments'].dtype}")
    print(f"  channels: {picked}")
    print(f"  regions:  {regions_picked}")
    print(f"  region_ids: {region_ids.tolist()}")


if __name__ == "__main__":
    main()
