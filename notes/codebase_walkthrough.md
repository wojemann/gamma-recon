# Codebase walkthrough — gamma_encoder framework

State as of 2026-04-30. Written so the next person reading the repo
(human or LLM) can understand what's here in 15 minutes before
running experiments.

If you only read one section, read **"Data flow at a glance"**.

---

## What's in this repo right now

Two packages live side-by-side:

- **`gamma_eval/`** — pre-existing evaluation harness. Pre-filtered
  per-band NMSE, gamma-envelope connectivity. Not changed in this
  session. 49 tests.
- **`gamma_encoder/`** — the model framework being built in this
  session. Tokenizer + spatial encoder + transformer + decoder + losses
  + training loop. New code; minimum needed for the overfit-sweep
  sanity check on Subject 2 data. 29 tests.

Plus:

- **`tests/`** — pytest suite, 78 tests total, all passing.
- **`scripts/cache_overfit_batch.py`** — pulls a small fixed batch off
  Subject 2 and saves it to `results/overfit_batch.pt`.
- **`configs/config.py`** — `local` / `server` path-profile selector.
- **`results/`** — cached batch (`overfit_batch.pt`) lives here. JSON
  output of overfit runs goes under `results/overfit_runs/`.
- **`brain_tree/`** — prior exploratory work (NOT part of the framework).

The reference BaRISTA repo at `/Users/wojemann/local_data/BaRISTA/` is
read-only.

---

## Data flow at a glance

Raw HDF5 trial → preprocessed segments → tokens → transformer →
reconstructed waveform → loss → gradient.

```
+-----------------------+   +------------------+   +----------------+
| BrainTreebank h5      |   | preprocess.py    |   | braintreebank  |
| (164 chans, 2048 Hz)  +-->+ notch -> reref   +-->+ .py loader     |
| sub_2_trial000..006   |   | -> segment       |   | (joins atlas)  |
+-----------------------+   | -> z-score       |   +-------+--------+
                            +------------------+           |
                                                           v
                            +----------------------------------------+
                            | TrialSegments                          |
                            |   segments (n_seg, C, T=6144)          |
                            |   region_ids (C,)                      |
                            |   channel_names (C,)                   |
                            +-------+--------------------------------+
                                    |
                                    v        scripts/cache_overfit_batch.py
                            +----------------+
                            | overfit_batch  |
                            |   .pt cached   |
                            +-------+--------+
                                    |
                                    v
+--------------------------------------------------------------+
| GammaEncoderModel                                            |
|                                                              |
|   patchify (T -> n_patches=12, L=512)                        |
|   |                                                          |
|   v                                                          |
|   Tokenizer  (B, C, n, L)  ->  (B, C, n, d_model)            |
|   |                                                          |
|   + RegionSpatialEncoder(region_ids)  broadcast over n       |
|   |                                                          |
|   v                                                          |
|   reshape -> (B, C*n, d_model)                               |
|   SmallTransformerEncoder (sinusoidal PE, stock attn)        |
|   reshape -> (B, C, n, d_model)                              |
|   |                                                          |
|   v                                                          |
|   LinearPatchDecoder (d_model -> L per token)                |
|   |                                                          |
|   v                                                          |
|   unpatchify -> (B, C, T)                                    |
+--------------------------------------------------------------+
                                    |
                                    v
                            +----------------+
                            | ReconstructionLoss  e.g. MSELoss |
                            +----------------------------------+
```

---

## Module-by-module

### `gamma_encoder/data/preprocess.py`

Pure-numpy preprocessing primitives. No I/O, no torch.

- `notch_filter(data, fs, freqs, q)` — `scipy.signal.iirnotch` per
  freq, applied via causal `lfilter`. Defaults match BaRISTA: Q=30 at
  [60, 120, 180, 240, 300, 360] Hz. Note 120/180 sit inside high-gamma
  (50–200 Hz) — narrow but worth flagging, see CLAUDE.md.
- `parse_electrode_name(name)` — `"LT3a1"` → `("LT3a", 1)` via regex.
- `build_laplacian_neighbors(all_electrodes, excluded)` — returns
  `{name: (neighbor_minus, neighbor_plus)}` for each channel that has
  both ±1 stem-neighbors AND none of (self, either neighbor) is
  corrupted. Channels at lead endpoints, or with corrupted neighbors,
  are absent from the result.
- `apply_laplacian_reref(data, names, neighbors)` — subtracts the mean
  of the two neighbors from each channel. Returns `(reref_array,
  kept_names)`.
- `segment_signal(data, segment_samples, step)` — non-overlapping
  windows by default. Returns shape (n_seg, n_chan, segment_samples).
- `zscore_segment(segments)` — per-row mean/std normalization with
  eps=1e-8 to avoid NaN on flat channels.

Tested in `tests/test_preprocess.py` (15 tests).

### `gamma_encoder/data/braintreebank.py`

Wraps the on-disk BrainTreebank layout in a class.

- `sanitize_electrode_name(name)` — strips `*`, `#`, `_` markers that
  the labels JSON adds but the atlas CSV doesn't. Matches BaRISTA's
  `_elec_name_strip`.
- `build_region_vocab(region_names)` — stable Destrieux→int mapping,
  sorted alphabetically. Reserves id 0 for empty/unknown if present.
- `BrainTreeTrial(subject_id, trial_id, data_root, ...)` — the loader.
  Reads JSON/CSV at construction time (cheap), opens h5 only on
  `load_segments()`. Returns a `TrialSegments` dataclass with
  `segments`, `region_ids`, `channel_names`, `fs`, `segment_samples`.
- `load_segments(max_seconds=None, notch=True)` — runs the full
  pipeline. `max_seconds` truncates the trial for fast caching/testing.

Default behavior verified on Subject 2 trial000:
164 raw chans → 29 corrupted dropped → 95 kept (after also requiring
both Laplacian neighbors and atlas membership). 25 unique Destrieux
regions across the kept channels.

Tested in `tests/test_braintreebank_loader.py` (8 tests, including a
synthetic-fixture end-to-end test).

### `gamma_encoder/tokenizers/`

- `base.py` — `Tokenizer` ABC plus `patchify(signal, L)` and
  `unpatchify(patches)` helpers. Shape contract for tokenizers:
  `(B, C, n, L) -> (B, C, n, d_model)`.
- `dilated_cnn.py` — `DilatedCNNTokenizer`. 1D conv stack with
  exponentially growing dilations (1, 2, 4, 8, 16), GELU, then adaptive
  average pool over time and a linear projection to `d_model`. Operates
  per (channel, patch). Faithful in *spirit* to BaRISTA's
  `TSEncoder2D` (repeated dilated-conv smoothing of high-frequency
  content) but simpler and 1D for clarity. The properties that matter
  for the gamma-fidelity hypothesis are preserved.

Other tokenizers (Linear, STFT, WPD) are not yet implemented.

### `gamma_encoder/models/`

- `spatial_encoder.py` — `RegionSpatialEncoder`. Vanilla `nn.Embedding`
  table over region ids. Returns `(B, C, 1, d_model)` so it broadcasts
  cleanly across the patch axis when added to tokenizer output.
- `encoder.py` — `SmallTransformerEncoder`. Defaults: d=32, 6 layers, 2
  heads, ff_mult=4, GELU, pre-norm. Stock
  `nn.TransformerEncoderLayer`s with **sinusoidal positional encoding**
  on the (channel × patch) sequence axis. BaRISTA uses RoPE + RMSNorm
  + GatedMLP — for an overfit-sweep sanity check the stock components
  are equivalent capacity, and swapping in a faithful BaRISTA stack is
  a follow-up that doesn't block the experiment matrix.
- `decoder.py` — `LinearPatchDecoder`. Single `nn.Linear(d_model,
  patch_samples)` applied per token. The downstream caller reshapes
  `(B, C, n, L)` back to `(B, C, T)` via `unpatchify`.
- `full_model.py` — `GammaEncoderConfig` dataclass + `GammaEncoderModel`
  composing tokenizer + spatial encoder + transformer + decoder.
  Forward signature: `model(signal: (B, C, T), region_ids: (B, C) or
  (C,)) -> (B, C, T)`. Validates that
  `tokenizer.{d_model, patch_samples}` match `cfg`.

### `gamma_encoder/losses/`

- `base.py` — `ReconstructionLoss` ABC. Forward signature
  `(pred: (B,C,T), target: (B,C,T)) -> scalar`.
- `mse.py` — `MSELoss`. Plain `F.mse_loss(pred, target)`. The BaRISTA
  baseline. Six other losses on the experiment plan (MAE, Whitened
  MSE, Multi-Res STFT, EEGM2, DistDF, cMin-LogCosh) not yet
  implemented.

### `gamma_encoder/training/overfit.py`

The single-batch overfit harness.

- Loads a cached batch (`results/overfit_batch.pt`) saved by
  `scripts/cache_overfit_batch.py`.
- Builds tokenizer + loss from string names via
  `build_tokenizer(name, cfg)` / `build_loss(name)`. Add new entries to
  these registries when you add a tokenizer or loss.
- Trains for `--steps` steps with AdamW. No LR schedule, no eval.
- Logs per-step loss, prints every `--log-every` steps.
- On exit writes `results/overfit_runs/overfit_<tok>_<loss>.json`
  containing the full loss trace and meta (params, init/final/min,
  elapsed seconds).

Run via:

```
KMP_DUPLICATE_LIB_OK=TRUE python -u -m gamma_encoder.training.overfit \
    --batch results/overfit_batch.pt \
    --tokenizer dilated_cnn --loss mse \
    --steps 500 --lr 1e-3 \
    --out-dir results/overfit_runs
```

Note the `python -u` (unbuffered) — without it, piping the output
through `tee` or `tail` will buffer the loss prints until the run
completes.

### `scripts/cache_overfit_batch.py`

CLI tool. Picks `--n-chans` channels (default 8) from a single
BrainTreebank trial, prioritizing diverse Destrieux parcels via a
greedy region-deduplication walk. Takes the first `--n-segments`
non-overlapping 3-s windows. Saves a torch dict to
`results/overfit_batch.pt`.

Default invocation produced (sub_2 trial 0):

```
segments shape: (16, 8, 6144) torch.float32
channels: ['LT3cHb5', 'RT3bHb6', 'LT3a7', 'LT2aA10',
           'LT3d6', 'RT1c8', 'LT1bIb5', 'RT3aHa11']
unique Destrieux parcels: 8
```

---

## Why these design choices

### Why z-score per segment (not per channel across the whole session)

Matches BaRISTA. Their NMSE numbers in Table 13 assume z-scored
segments, so per-channel-per-segment is the apples-to-apples
normalization for our sanity check. Mathematically also makes NMSE =
1 - R² per channel, which is interpretable independent of overall scale.

### Why sinusoidal PE instead of RoPE

Time-to-first-result trade. RoPE attention is ~30 lines from scratch
but plumbing it correctly with multi-head is enough work to defer until
the pipeline is verifiably alive. Stock `nn.TransformerEncoderLayer`
exercises the rest of the system (data → tokenizer → attention →
decoder → loss → gradient) and is sufficient capacity for the
overfit-sweep test. The faithful BaRISTA stack is a one-PR follow-up
that does not change the experimental conclusions of the loss/
tokenizer ablations (since attention positional details are held
constant across all configs).

### Why one latent vector per (channel, patch), not per channel

Matches BaRISTA's segment-as-sequence approach. Each (channel, patch)
becomes one token; the transformer attends over all C × n_patches
tokens jointly. This is what lets the model represent within-channel
temporal structure AND cross-channel spatial structure in the same
attention.

### Why a `--max-seconds` knob on the loader

The trials are 110–224 minutes long; loading a full trial is ~3.5 GB
in float64 across kept channels. For the overfit batch we need maybe a
minute of data, so we just read the prefix. Saves both wall time and
memory during dev.

### Why the channel name sanitization

The h5 labels file uses `*` (probably "noisy") and `#` (probably
"out-of-brain") markers; the localization CSVs don't. Without
stripping, atlas join would fail for every starred/hashed channel and
the kept-set would be much smaller than it should be. BaRISTA does
exactly this in `_elec_name_strip`.

### Why drop a channel if EITHER neighbor is corrupted

Laplacian reref subtracts the mean of the two neighbors. If either
neighbor is corrupted, that subtraction injects noise into a
nominally-clean channel. The conservative choice is to skip the
channel. Matches BaRISTA's `clean_laplacian` filter.

---

## What is NOT yet built

These are explicit gaps in scope, not bugs:

1. **Real-batch overfit run not yet verified.** Code exists
   (`gamma_encoder.training.overfit`), unit-test single-segment overfit
   passes, but the full 500-step run on `results/overfit_batch.pt` was
   interrupted before producing a clean log. Running this is the next
   action; expected behavior is loss → ~0 (or at minimum monotonic
   decrease over 500 steps).
2. **Other tokenizers**: Linear, STFT-magnitude, Wavelet Packet
   Decomposition. Each is a single new file under
   `gamma_encoder/tokenizers/` plus a registry entry in `overfit.py`.
3. **Other losses**: MAE/Huber, Whitened MSE, Multi-Resolution STFT,
   EEGM2 composite, DistDF, cMin-LogCosh. Each gets a synthetic-data
   unit test FIRST per CLAUDE.md, then plugs into the overfit harness.
4. **Sweep harness**: a thin script that runs the cross-product of
   (loss × tokenizer) configs against the same cached batch and
   summarizes results. Trivial once #2 and #3 exist.
5. **Faithful BaRISTA encoder stack** (RoPE + RMSNorm + GatedMLP).
   Optional improvement; doesn't change experimental conclusions.
6. **Masked reconstruction** during training. BaRISTA masks ~30% of
   tokens and reconstructs only those; we currently do full
   reconstruction. Add when the framework moves past overfit-sweep
   into actual pretraining.
7. **Streaming data loader** for actual training (not just cached
   overfit batches). Will read a session, chunk-load, yield batches
   asynchronously. Build when Phase 1 starts.

---

## How to run things

```
# environment (M5)
conda activate gamma-env
export KMP_DUPLICATE_LIB_OK=TRUE   # libomp duplicate guard, see CLAUDE.md

# tests
pytest tests/ -q

# cache the overfit batch (writes results/overfit_batch.pt)
python -m scripts.cache_overfit_batch

# run a single overfit configuration
python -u -m gamma_encoder.training.overfit \
    --batch results/overfit_batch.pt \
    --tokenizer dilated_cnn --loss mse \
    --steps 500 --lr 1e-3 \
    --out-dir results/overfit_runs
```

Inspect the per-run JSON:

```
import json
data = json.load(open("results/overfit_runs/overfit_dilated_cnn_mse.json"))
print(data["initial_loss"], data["final_loss"], data["min_loss"])
```

---

## Known footguns

- `python -u` matters when piping. Without it, `tail`, `tee`, etc.
  buffer stdout until the program exits, and you can't see live
  progress.
- `KMP_DUPLICATE_LIB_OK=TRUE` is required because the conda env's
  scipy/MKL and pip-installed torch each ship libomp. Long-term fix is
  to install torch via conda-forge into the same env so there is only
  one libomp.
- The overfit batch's `region_ids` use a *trial-local* region vocab.
  If you ever cache batches from multiple trials/subjects and load them
  together, you must rebuild a shared vocab — otherwise the same
  integer means different things across batches. The `BrainTreeTrial`
  constructor accepts an explicit `region_vocab` arg for this.

---

## Where to look first when something breaks

| Symptom                                  | Suspect first                                    |
| ---------------------------------------- | ------------------------------------------------ |
| Loss stays flat                          | Optimizer step missing, or loss is constant in input — check `MSELoss` shape assertion fired |
| Shape error in `model.forward`           | `cfg.patch_samples * n_patches != T` (T=6144)    |
| Region embedding goes out of range       | `cfg.num_regions <= region_ids.max()` — overfit script bumps it to ≥ 64 by default |
| Loader returns 0 kept channels           | `corrupted_elec.json[sub_N]` plus atlas-missing names exhausting all candidates; check `trial.summary()` |
| All NaN after first step                 | Almost certainly an unfilled IIR transient — extend `--max-seconds` so segmenter starts after settling, OR increase z-score eps |
| `KeyError` joining atlas                 | Forgot to `sanitize_electrode_name(...)` somewhere on either side of the join |
