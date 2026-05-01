# Pipeline / models / tests overview

A single-file walkthrough of everything built in `gamma-recon/` so far,
written so you can read top-to-bottom and check that it matches what you
asked for. Where this overlaps with `notes/codebase_walkthrough.md` and
`notes/linear_ar_walkthrough.md`, those are the deeper dives; this one
is the index.

## 1. Data path (laptop stage)

### `gamma_encoder/data/preprocess.py`

Pure-numpy primitives, no torch, no h5 — this is the "what does
preprocessing actually do" module, written so the operations are
testable without touching real data:

- `notch_filter(x, fs, freqs=(60,120,180,240,300,360), Q=30)` —
  iirnotch + `lfilter` (causal, BaRISTA-matched; not zero-phase). Q=30
  → ~2 Hz wide notches.
- `parse_electrode_name` — strips BaRISTA's `*#_` markers from electrode
  names (defensive copy of their `_elec_name_strip`).
- `build_laplacian_neighbors(names, corrupted)` — for each `<stem><N>`
  channel, finds `<stem><N-1>` and `<stem><N+1>` neighbors along the
  depth lead, drops endpoints and any channel with a corrupted neighbor.
- `apply_laplacian_reref(x, neighbors)` — `x_i' = x_i − mean(x_{N-1},
  x_{N+1})`.
- `segment_signal(x, fs, segment_seconds)` — non-overlapping windows.
- `zscore_segment(seg)` — per-channel z-score within each segment.

Pipeline order: **notch → laplacian reref → segment → z-score**. This
matches BaRISTA's recipe (the order matters because reref mixes
power-line content across channels, so notch must come first).

### `gamma_encoder/data/braintreebank.py`

H5 → preprocessed-segments loader for Subject 2. Wraps the preprocess
primitives. Reads the trial's electrode labels and corrupted-channel
list from BrainTree, builds the laplacian neighbor list ourselves
(BaRISTA's `clean_laplacian.json` isn't shipped with our copy of the
dataset).

### `scripts/cache_overfit_batch.py` + `results/overfit_batch.pt`

A fixed `(16 segments, 8 channels, 6144 samples) @ fs=2048` batch from
sub_2 trial 0, with `region_ids` and `fs`. Channels are picked greedily
to span 8 different Destrieux parcels. This is the canonical "does the
laptop pipeline work" artifact and it should not change between runs
unless we deliberately rebuild it.

## 2. Models

### `gamma_encoder/models/full_model.py` — `GammaEncoderModel`

The transformer model. Layout: tokenizer → optional mask substitution →
spatial embedding → encoder → linear decoder.

- **Tokenizer** maps `(B, C, T)` → `(B, C, n_patches, d_model)`, channel
  by channel (channel-independent).
- **Channel masking.** If `mask_channels=(B, C) bool` is passed, the
  rows of the tokenized sequence corresponding to True positions are
  replaced by a learnable `mask_token` parameter. This happens *after*
  tokenization, *before* the encoder — so the encoder never sees the
  masked channels' content. Spatial embeddings are still added at
  masked positions, so the encoder knows *which* channels it's being
  asked to predict.
- **Spatial encoder** — `nn.Embedding(num_regions, d_model)` lookup over
  Destrieux parcel IDs. Region-level only; we do not use BaRISTA's
  channel-level or lobe-level variants.
- **Encoder** — `cfg.encoder_kind` selects between:
  - `"stock"` (default): `nn.TransformerEncoderLayer` pre-norm + sinusoidal
    PE. Used for the main ablation.
  - `"faithful"`: BaRISTA's stack — RMSNorm + RoPE + SwiGLU MLP. Built
    in `gamma_encoder/models/faithful.py` for the reproducibility
    baseline only; not part of the loss/tokenizer ablation.
- **Decoder** — single linear layer from `d_model` back to
  `patch_samples`, then `unpatchify` to `(B, C, T)`.

### `gamma_encoder/models/faithful.py` — BaRISTA-faithful encoder

Faithful reimplementation of BaRISTA's encoder for the reproducibility
baseline:

- `RMSNorm` (eps=1e-6, fp32 variance computation).
- `RotaryEmbedding` (sliced-halves LLaMA convention, base=10000).
- `RotarySelfAttention` — RoPE applied to Q/K only, V untouched.
- `GatedTransformerMLP` (SwiGLU, mlp_ratio=4).
- `FaithfulEncoderLayer` (pre-norm), `FaithfulTransformerEncoder`
  (stack + final RMSNorm).

The smoke test in `scripts/faithful_smoke.py` verifies the random and
weight-reset versions of this stack run forward+backward and that the
parameter shapes parity-match BaRISTA's checkpoint at ~58% multiset shape
overlap (most divergence is in the spatial embedder and decoder, which
we don't need to load).

### `gamma_encoder/models/linear_ar.py` — `LinearVARModel` (alias `LinearARModel`)

The MVAR baseline — one `nn.Conv1d(C, C, kernel_size=p)` applied to a
left-padded input, implementing a strictly causal full vector-AR(p)
predictor:

```
y[c, t] = b[c] + Σ_{c'} Σ_{k=1..p} W[c, c', k] * x[c', t-k]
```

- `p = 3` by default (200 params for `C=8`).
- **Init:** `W[:, :, p-1] = I_C` (lag-1 identity), bias = 0. So at step
  zero the model implements `y[c, t] = x[c, t-1]` per channel — each
  channel is a 1-sample-delayed copy of itself, no cross-coupling. Off-
  diagonal coupling has to be learned from gradient signal.
- **Channel masking.** If `mask_channels` is passed, masked input rows
  are zero-filled before the conv, so masked channels cannot leak into
  the prediction.

Why MVAR rather than per-channel AR: it is the smallest linear, strictly
causal model that has the same input access pattern as the transformer's
per-token aggregation. Anything the transformer does that VAR can't is
therefore non-linear or non-causal, not a spatial-aggregation issue.
Detailed walkthrough in `notes/linear_ar_walkthrough.md`.

### Tokenizers (`gamma_encoder/tokenizers/`)

All inherit from `Tokenizer` (ABC). Shape contract:
`(B, C, n_patches, L) → (B, C, n_patches, d_model)`. `patchify` and
`unpatchify` helpers in `tokenizers/base.py`.

- `dilated_cnn.py` — BaRISTA-style stack. `nn.Linear(hidden *
  patch_samples, d_model)` head replaces BaRISTA's MLP-pool — this is
  a deliberate divergence (an `nn.AdaptiveAvgPool1d(1)` head smooths
  high frequencies, working against the gamma-fidelity probe).
- `linear.py` — single `nn.Linear(patch_samples, d_model)` per patch.
  Fast; default for the loss-axis sweep.
- `stft_magnitude.py`, `complex_stft.py` — STFT-based, magnitude-only
  vs full complex.
- `wavelet_packet.py` — pywt wavelet-packet decomposition + linear.
- `welch_psd.py` — Welch PSD per patch + linear (drops time resolution
  within each patch — known floor).

## 3. Losses (`gamma_encoder/losses/`)

`base.py` defines the `ReconstructionLoss` ABC. Each loss takes
`(pred, true)` of shape `(B, C, T)` (z-scored). Built so far:

| Name | Module | What it does |
|------|--------|--------------|
| `mse` | `mse.py` | Plain `((pred - true)**2).mean()`. Baseline. |
| `mae`, `huber` | `robust.py` | L1 / Huber baselines (sanity check that any improvement comes from spectrum awareness, not outlier handling). |
| `whitened_mse` | `whitened_mse.py` | FFT, divide by √PSD (precomputed and cached at `results/whitened_mse_psd.pt`), MSE in whitened space. Currently the PSD is built from the cached batch — needs re-estimation from real data on the server. Numerically unstable when masked rows are zero-padded; flag for follow-up. |
| `log_power_spectral` | `log_power_spectral.py` | L1 on log-power spectra. |
| `multires_stft` | `multires_stft.py` | Sum of STFT-magnitude L1 across `[(64,16,64), (256,64,256), (1024,256,1024)]`. Audio-domain SOTA. |
| `eegm2` | `eegm2.py` | Temporal L1 + frequency-domain spectral loss. EEG-SSL precedent. |
| `distdf` | `distdf.py` | Bures-Wasserstein on joint (X, Y) and (X, Ŷ) covariances + MSE. |
| `cmin_logcosh` | `cmin_logcosh.py` | Circular-min log-cosh — minimum log-cosh over all circular shifts. Time-shift-invariant. |
| `content_aware_l1` | `content_aware_l1.py` | BrainBERT-style content-reweighted L1. |

## 4. Training harness

### `gamma_encoder/training/overfit.py` — `run_overfit`

The single-batch harness. Loads the cached batch, builds a
(model, tokenizer, loss) trio, runs `steps` Adam steps on the same batch
of bytes. Used for laptop sanity-checking only.

Region masking (current behavior):
- `mask_n_regions: int = 0` parameter. 0 disables masking
  (full-reconstruction legacy mode).
- `_sample_region_mask(B, region_ids, k_regions, gen, device)` picks
  `k_regions` distinct region IDs per batch element and masks every
  channel whose region is in that set. With one channel per region (the
  cached laptop batch), this reduces to channel masking; with multiple
  channels per region (real Subject 2 data) it masks whole regions at a
  time.
- Loss is computed only on the masked channels' waveforms — the
  unmasked channels' near-perfect identity reconstruction would
  otherwise dominate.

Logging is via `gamma_encoder/training/logging.py` (`MetricsLogger`
ABC + `StdoutLogger`, `JsonlLogger`, `WandbLogger`, `MultiLogger`). The
loop never hard-codes a backend; pick the combo at the call site.

### `scripts/run_overfit_sweep.py`

Walks `(loss × tokenizer × model_type)` axes:
- **Loss axis** — 10 losses × `linear` tokenizer (default — fast).
- **Tokenizer axis** — `mse` × 6 tokenizers.
- **MVAR axis** — 10 losses × `linear_ar` baseline.

Total: 25 unique configs. CLI:
```
python -u scripts/run_overfit_sweep.py --axis all --steps 500 \
    --loss-axis-tokenizer linear
```
`--mask-n-regions` defaults to `floor(n_unique_regions / 3)` from the
batch's `region_ids` (currently 8 unique → mask 2 regions per segment).
Each run lands at `results/overfit_runs/<loss>__<label>/` with
`config.json`, `metrics.jsonl`, `summary.json`, `model.pt`,
`loss_curve.png`, `reconstruction.png`.

### `scripts/run_band_eval.py`

For every saved `model.pt`:
1. Reconstructs the cached batch (using a deterministic eval region-mask
   with `_EVAL_MASK_SEED = 12345` if the run was trained masked).
2. Computes per-band NMSE via `gamma_eval.evaluator.ReconstructionEvaluator`
   over delta-theta (1-8), alpha-beta (8-30), low-gamma (30-50),
   high-gamma (50-200) Hz.
3. Writes `band_eval.csv` and three heatmaps (`band_eval_loss_axis.png`,
   `band_eval_tokenizer_axis.png`, `band_eval_linear_ar_axis.png`).

For masked runs the evaluator scores **only the masked channels'
waveforms**, not all channels — same supervision signal as training.

### `scripts/plot_reconstructions.py`

For every saved `model.pt`, plots one segment (8 channels, true vs
predicted overlaid) at `results/overfit_runs/<run>/reconstruction.png`.
Masked rows get a cream tint and a `[MASKED]` suffix in the y-label.

### `scripts/faithful_smoke.py`

A 4-step smoke for the BaRISTA-faithful encoder: random-init forward +
backward, weight-reset-from-checkpoint forward + backward, parity report
of parameter shapes vs BaRISTA's `parcels_chans.ckpt`, 10-step loss
curve. Output at `results/faithful_smoke/`.

## 5. Evaluation harness (`gamma_eval/`)

Built before the encoder framework, used as the eval substrate.

- `metrics/reconstruction.py` — band-resolved NMSE, log-spectral-distance.
  `evaluate_reconstruction(true, pred, fs, bands)` filters per band with
  zero-phase 4th-order Butterworth (`scipy.signal.sosfiltfilt`) and
  computes per-channel `NMSE = MSE / Var(true)`.
- `metrics/prefilter.py` — for real (long-session) data, filter once
  end-to-end then slice per segment to avoid filter edge artifacts.
- `synthetic/signals.py` — 1/f + burst generator; used by tests and
  `gamma_eval/demo.py`.
- `evaluator.py` — `ReconstructionEvaluator` accumulator over many
  segments; aggregates per-band per-channel NMSE arrays + connectivity.
- `demo.py` — synthetic BaRISTA-failure-pattern demo (oracle / smoothed /
  low_freq_only).

## 6. Tests (`tests/`)

```
pytest tests/ -v
```

| File | What it covers |
|------|----------------|
| `test_preprocess.py` | notch behavior on synthetic 60Hz; laplacian neighbor parsing; segment shape; z-score statistics. |
| `test_braintreebank.py` | end-to-end loader on the real h5 (sub_2 trial 0): shape, fs, region_ids alignment with the laplacian-clean channel list. |
| `test_tokenizers.py` | shape contract for every tokenizer; gradient flow; per-tokenizer parameter count sanity. |
| `test_losses.py` | shape contract for every loss; finite-output sanity; `whitened_mse` PSD-cache load path; symmetry where expected. |
| `test_full_model.py` | `GammaEncoderModel` end-to-end forward shape; spatial-embedding lookup; encoder kind selector. |
| `test_faithful_encoder.py` | RoPE rotation correctness, RMSNorm fp32 variance, attention masking, layer parity tests. |
| `test_linear_ar.py` | 20 tests: `LinearVARModel` shape, strict causality (perturbing `signal[..., k]` must not change recon at indices `≤ k`), parameter count `C²p + C`, hand-built cross-channel coupling, end-to-end fit to a synthetic VAR(3) process with known `A1, A2, A3` (recovered within 0.07 after 400 Adam steps). |
| `test_channel_masking.py` | 8 tests: transformer mask preserves shape; transformer rejects wrong-shape mask; perturbing a masked channel does NOT change recon (load-bearing causality property, `< 1e-5`); same property for MVAR (`< 1e-6`); unmasked channels DO drive output (sanity); MVAR can learn to predict masked channels from neighbors on a synthetic 4-channel rolled signal; region-mask sampler groups channels by region (`[0,0,1,1,2,2]` → exactly 2 channels masked, sharing a region); sampler rejects `k_regions >= n_unique`. |
| `test_overfit.py` | 1-step overfit smoke for `(mse, dilated_cnn)`. |

Current count: 88+ tests, all pass.

## 7. What is NOT built yet

These are the gaps before this can train Subject 2 on the server:

- **No real pretraining entrypoint.** `run_overfit` takes a single
  cached batch; there is no streaming DataLoader over Subject 2's full
  ~14h. `gamma_encoder/data/braintreebank.py` returns segments
  per-trial in memory but there is no multi-trial / multi-session loop
  yet.
- **No DDP / multi-GPU.** The model is built `.to("cpu")` only.
- **No mixed precision / gradient clipping.** Plain AdamW; will need
  `torch.cuda.amp.autocast` + `GradScaler` + `clip_grad_norm_` for
  full Subject 2 batches on A40s.
- **`whitened_mse` PSD cache is laptop-stage.** Built from the 16-segment
  cached batch; needs re-estimation from full pretraining sessions
  before the loss is meaningful in cross-loss comparisons. Also
  numerically unstable on masked-channel inputs (NaN observed in
  `whitened_mse__linear_ar`); the on-the-fly path or a sane PSD floor
  needs to be re-examined.
- **`distdf` is numerically fragile** under masked-channel inputs (NaN
  observed on the masked sweep). Diagnose before the server.
- **`log_power_spectral` lr=1e-3 is too hot.** Plateaued ~1.0 with
  oscillation; try `lr=3e-4` for that loss specifically.
- **No checkpointing / early stopping / lr scheduling.**
- **No tracking of pretraining-internal validation loss** (the 80/10/10
  segment split inside the pretraining sessions).

## 8. Config / repro conventions

- `configs/config.py` defines `local` and `server` path profiles. Every
  script that touches the filesystem takes `--profile {local,server}`
  and resolves real paths via `get_config(profile)`.
- `KMP_DUPLICATE_LIB_OK=TRUE` is the Mac libomp workaround; not needed
  on Linux.
- Tests run pure-CPU; nothing is GPU-only.

## 9. Where to look for the "why" behind specific decisions

- `notes/codebase_walkthrough.md` — fuller per-file walkthrough of the
  encoder framework.
- `notes/linear_ar_walkthrough.md` — MVAR baseline (init, MVAR-vs-AR
  rationale, masked-pretraining specifics).
- `notes/laptop_overfit_smoke_results.md` — the 1000-step full-recon
  sweep numbers (pre-masking).
- `CLAUDE.md` — the project thesis, Subject 2 splits, BaRISTA-faithful
  vs. gamma-fidelity divergences, methodology discipline.
