# Moving from laptop to 2× A40 server

What you need to change before this code can pretrain Subject 2 on the
server. Audited 2026-05-01 against `main` (commit `f519730` plus the
region-masking change).

## Must-fix before the server

### 1. Build a real pretraining entrypoint

There isn't one yet. `gamma_encoder/training/overfit.py` is a
single-batch laptop harness. `scripts/run_overfit_sweep.py` is a laptop
sweep. Neither streams Subject 2's full ~14h.

Build `gamma_encoder/training/pretrain.py` with:
- A multi-trial / multi-session DataLoader over the BrainTreebank
  Subject 2 sessions (sessions 1–5 for laptop-stage held-out semantics
  or 1–7 once we move to the server, per the methodology in
  `CLAUDE.md`).
- The 80/10/10 pretraining-internal segment split (separate from the
  downstream task split).
- A multi-epoch loop with checkpointing, early stopping on
  pretraining-validation loss, and an LR scheduler.
- The same masked-region pretraining objective the laptop sweep uses
  (`mask_n_regions`, default `floor(n_unique_regions / 3)`).
- Pretraining-validation NMSE logged per epoch via the same
  `gamma_eval` evaluator the laptop pipeline uses.

The `MetricsLogger` ABC in `gamma_encoder/training/logging.py` already
has a `WandbLogger`; flip that on for server runs.

### 2. Streaming / multi-trial data loader

`gamma_encoder/data/braintreebank.py` reads one trial's h5 once and
returns the segmented tensor in memory. That's fine for a single trial
but not for ~14h.

Wrap that in a `torch.utils.data.Dataset` that:
- Indexes `(session, trial, segment_idx)` triples.
- Reads each trial's h5 lazily (open + close around each segment, or
  hold a few open file handles in a small LRU).
- Applies the existing preprocess primitives.
- Returns `(segments_tensor, region_ids_tensor)` — channel set is
  fixed per subject, so `region_ids` is constant across segments for a
  given subject.

Add a `DataLoader` with `num_workers > 0` (h5 read is the bottleneck).

### 3. DDP wiring for 2× A40

Currently single-process. Add to the new pretrain entrypoint:
- `torch.distributed.init_process_group(backend="nccl")` from
  `torchrun --nproc_per_node=2`.
- `DistributedDataParallel(model, device_ids=[local_rank])`.
- `DistributedSampler` on the dataset.
- Rank-zero gates around the `WandbLogger` and checkpoint writes.

Nothing in `GammaEncoderModel` or `LinearVARModel` is DDP-hostile
(no register_buffer-with-non-module-parents, no in-place state).

### 4. Mixed precision + gradient clipping

`overfit.py` uses bare `torch.optim.AdamW` with no autocast. For
135-channel batches on A40s, you'll want:
```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    recon = model(segments, region_ids, mask_channels=mask)
    loss = loss_fn(...)
scaler.scale(loss).backward()
scaler.unscale_(optim)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optim)
scaler.update()
```
bfloat16 over fp16 because RoPE + RMSNorm are easier to keep stable in
bf16. Some of the spectral losses (especially `whitened_mse` and
`distdf` on the FFT side) may want explicit `.float()` casts inside
the autocast block — verify on a 100-step run before committing.

### 5. Re-estimate the whitened-MSE PSD on real training data

`results/whitened_mse_psd.pt` was built from the 16-segment cached
laptop batch. That estimate is not trustworthy — its scale (~1e6 in the
on-the-fly variant) makes any cross-loss comparison meaningless.

Before training, run a one-shot script that:
- Loads all preprocessed pretraining-train segments.
- Welch-PSDs each channel, averages over segments.
- Saves to `results/whitened_mse_psd.pt`.

The `WhitenedMSELoss` constructor already accepts a precomputed `psd`
tensor and `gamma_encoder/training/overfit.py` already loads from this
path; no code changes needed beyond regenerating the file.

The `whitened_mse__linear_ar` masked run produced NaN — diagnose
before re-enabling. Likely the PSD has zeros where the masked
zero-padded inputs put no power, leading to division-by-zero. A small
floor (e.g. clamp PSD ≥ 1e-6 before sqrt) is the obvious fix.

### 6. Path profile sanity-check

`configs/config.py` already exposes `local` (your laptop) and `server`
(`/mnt/sauce/littlab/users/wojemann/BrainTree/`) profiles, and every
script that needs paths uses `get_config(args.profile)`. Just verify
the server path is correct on your actual server before the first run
— it's a one-line fix if the mount point is different.

## Likely needs tuning, not blocking

### 7. Batch size / channel count

The laptop overfit batch is `(16, 8, 6144)`. Subject 2 after Laplacian
has ~135 channels. Memory-wise, `(B=4, C=135, T=6144)` fp32 is ~13 MB
per batch element raw; with the transformer's `(B, C·n_patches,
d_model)` token layout that becomes `B × C × 12 × 64` ≈ 100k tokens per
batch element. At `d_model=64, n_layers=12, n_heads=8`:
- Attention is `O(n_tokens²)` per layer per batch element. 100k tokens
  squared is too much; you'll need to reduce somewhere.
- Easy first cuts: smaller batch (`B=2` per GPU = 4 effective with DDP)
  + bf16. Verify with a forward/backward smoke before kicking off the
  full ablation.
- If it's still too much, the server can also reduce `n_patches` per
  segment (longer patch → fewer tokens). 12 patches × 256-sample
  patches at `fs=2048` is 1.5s; the segment is 3s so you could halve
  to 6 patches without changing the segment definition.

### 8. Two losses are fragile under masking

- `whitened_mse__linear_ar` produced NaN on the masked sweep —
  see point 5.
- `distdf__linear_ar` produced NaN. The Bures-Wasserstein term involves
  matrix-square-root of covariances, which goes singular when masked
  rows zero-pad. Either skip this loss for the linear_ar baseline, or
  add a small ridge to the covariance estimate inside `distdf.py`
  before the matrix-sqrt.
- `log_power_spectral` plateaued near 1.0 with oscillation at
  `lr=1e-3`. Try `lr=3e-4` for that loss specifically before declaring
  it a non-converger.

### 9. Move the OpenMP hack into a server-no-op

`KMP_DUPLICATE_LIB_OK=TRUE` is required for the Mac libomp clash but
will be a silent no-op on Linux. Safe to leave as is; just don't be
surprised that nothing complains on the server.

## What does not need to change

- The model code (`gamma_encoder/models/`) is device-agnostic.
- The loss code (`gamma_encoder/losses/`) is device-agnostic.
- The tokenizer code (`gamma_encoder/tokenizers/`) is device-agnostic.
- The preprocess primitives (`gamma_encoder/data/preprocess.py`) are
  pure numpy, run anywhere.
- The `gamma_eval` harness is pure scipy/numpy, runs anywhere.
- The test suite is CPU-pure; it'll run as a sanity-check on the server
  too.

## Checklist

```
[ ] write gamma_encoder/training/pretrain.py with the multi-session loop
[ ] wrap braintreebank.py in a torch Dataset + DataLoader, num_workers
[ ] add DDP scaffolding (torchrun, DistributedSampler, rank-zero gates)
[ ] add bf16 autocast + GradScaler + clip_grad_norm_(1.0) to the loop
[ ] regenerate results/whitened_mse_psd.pt from real pretraining data
[ ] add a PSD floor inside WhitenedMSELoss before the sqrt
[ ] add a covariance ridge inside DistDFLoss before matrix-sqrt
[ ] verify configs/config.py "server" profile mount point is correct
[ ] one-config smoke run (mse + linear, single epoch) before the ablation
[ ] kick off the loss ablation, then the tokenizer ablation
```
