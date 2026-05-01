# Project: high-frequency-faithful brain encoders

## What this project is

A research project building toward a paper on improving high-frequency
(gamma) reconstruction in self-supervised brain-state encoder models.
Working title: something like "Spectral losses and tokenization for
gamma-faithful neural foundation models."

## The thesis

Current brain-state encoders (BaRISTA, BrainBERT, KenazLBM, foundation
models for iEEG/sEEG) systematically fail to reconstruct high-frequency
neural activity (gamma band, 50–200 Hz). The hypothesis is that this
failure is driven by:

1. **MSE loss is biased toward low frequencies** because neural signals
   follow a 1/f^alpha power law (alpha ≈ 1.5 in iEEG). MSE weights all
   time-domain samples equally, but the time-domain power is dominated by
   slow components, so models minimize loss by predicting the slow
   structure and ignoring high-frequency bursts.
2. **CNN tokenizers smooth high-frequency content** through repeated
   convolutions, even before the model has a chance to represent gamma
   activity in its latent space.

We hypothesize that fixing these two issues will produce models whose
reconstructions are gamma-faithful, and that gamma-faithful reconstruction
correlates with downstream task performance on tasks known to be
gamma-driven (speech onset, speech/non-speech, volume, pitch, face/emotion
perception).

## Smoking guns motivating the project

**1. BaRISTA Table 13.** Their own paper reports per-band reconstruction
quality. Their high-frequency NMSE is greater than 1.0 across every
configuration they tried — meaning their model is *worse than predicting
the mean* in the gamma band. This is published, in their appendix, and
they don't dwell on it.

**2. Pilot result on Subject 2 (this project).** Logistic regression on
high-gamma (50–200 Hz) band power alone is the strongest predictor of
speech onset on Subject 2 by a substantial margin compared to other
bands. See `notebooks/01_gamma_predicts_speech.ipynb`. This is the
empirical anchor for the paper: gamma reconstruction matters because
gamma carries the task-relevant signal.

## Key prior art

### Brain encoder backbones

- **BaRISTA** (`14145_BaRISTA_Brain_Scale_Info.pdf`): primary architectural
  reference. Lightweight (d=64, 12 layers), channel-flexible,
  raw-waveform input, MSE loss in latent space + linear head to raw
  signal. We reimplement their architecture as one configuration in our
  own framework — see "Implementation strategy" below.
- **BrainBERT** (Wang et al. 2023, `2302.14367v1.pdf`): predecessor;
  STFT/superlet tokenizer, content-aware L1 loss that reweights
  high-magnitude regions. Important precedent for non-trivial
  reconstruction loss.
- **PopT** (Chau et al. 2025, `2406.03044v4.pdf`): discriminative-only
  pretraining, no reconstruction. Argues reconstruction is the wrong
  objective. Counterargument we need to address — see "Framing".
- **KenazLBM** (`2025.08.10.669538v1.pdf`): preprocesses by *filtering
  out* everything above 121 Hz, sidestepping the question entirely. Loss
  function not specified in paper.
- **BrainState/AR-βVAE** (Johnson et al. 2024, `2024_06_01_596884v1_full.pdf`):
  asymmetric recurrent variational autoencoder with **circular minimum
  log-cosh (cMin-LogCosh)** loss — a time-shift-invariant loss for
  asymmetric encoder-decoder forecasting setups. Different axis of
  concern from spectral losses (temporal alignment vs frequency
  weighting), worth one slot in the loss ablation.

### Loss-function precedents

- **Merk et al. 2025** (`2508.10160v1.pdf`): proposes 1/f-scaled MAE in
  the *spectrogram domain* for a Welch-PSD-tokenized DBS LFP model.
  Targets are spectrograms, not waveforms, so the loss is not directly
  swappable into a waveform-prediction model. The closest waveform
  analog is **whitened MSE** (FFT the waveform, scale by 1/√PSD, MSE),
  which we adopt as our principled spectrum-aware loss. Cite Merk as
  closest precedent for spectral-domain 1/f correction.
- **EEGM2** (Yu et al. 2025, arXiv:2502.17873): EEG SSL framework that
  combines temporal L1 with frequency-domain spectral loss. Direct
  EEG-domain precedent for composite losses. Must be acknowledged and
  beaten.
- **DistDF** (Wang et al. 2026, `8696_DistDF_Time_series_Foreca.pdf`):
  Bures-Wasserstein discrepancy on joint distributions, addresses MSE's
  autocorrelation bias by aligning means and covariances of (X,Y) and
  (X,Ŷ). Theoretically motivated alternative; same Gaussian assumption
  as MSE but adds covariance alignment.
- **Multi-resolution STFT loss** (audio domain SOTA — Parallel WaveGAN,
  HiFi-GAN, UnivNet, DAC). Sum of STFT-magnitude L1 losses across
  multiple (window, hop, FFT) configurations. The mature recipe for
  high-fidelity waveform reconstruction. Adapted to neural data here.

## The dataset: BrainTreebank

- 10 epilepsy patients, 26 sessions watching movies, sEEG at 2048 Hz,
  ~16 electrodes per subject (1688 total).
- Labels available: sentence onset, speech vs non-speech, word volume,
  word pitch, scene labels, face counts, GPT-2 surprisal, etc.
- Documentation: `2411.08343v1.pdf`.
- Dataset URL: `https://braintreebank.dev`.
- Data is on the M5 at `/Users/wojemann/local_data/BrainTree/`.
- BaRISTA reference repo (read-only) at `/Users/wojemann/local_data/BaRISTA/`.

### Verified Subject-2 facts (from session exploration)

- **Sampling rate: 2048 Hz** confirmed via trigger-index spacing in
  `subject_timings/sub_2_trial000_timings.csv` (median ~2038 idx/s).
- **Channels: 164 in sub_2 trial000.h5** (`/data/electrode_{0..163}`,
  float64), but `electrode_labels/sub_2/electrode_labels.json` lists
  154 names — mismatch to reconcile in the loader.
- **Trial length: ~19.18M samples ≈ 156 min per trial.**
- **Atlas labels present.** `localization/sub_2/depth-wm.csv` gives per
  electrode DesikanKilliany / Destrieux / DKT parcels;
  `localization/elec_coords_full.csv` gives Destrieux + coords across
  subjects. → Region-level (parcel) spatial encoding feasible without
  lobe fallback.
- **Bad channels:** 29 entries for sub_2 in `corrupted_elec.json` (incl.
  DC4, DC10). Drop before training.
- **Word-level features precomputed:** `transcripts/venom/features.csv`
  contains start, end, pitch, RMS, delta_rms, delta_pitch, GPT-2
  surprisal, face_num, brightness, speaker, etc. Sentence onset
  derivable; trees in `trees/venom/tree.conllu` are time-aligned.

### BaRISTA repo caveat

Their public repo contains **only finetuning code + 3 pretrained
checkpoints** (chans/parcels/lobes spatial groupings). No pretraining
script, no MSE-reconstruction loss code. We must implement pretraining
ourselves — which is the plan anyway. Their model + spatial-encoder
modules are still useful as architectural reference and weight-loadable
sanity check.

## Subject 2 splits (per BaRISTA Appendix Tables 5 and 6)

Subject 2 has 7 sessions. Three levels of splits operate in the BaRISTA
protocol that are easy to conflate:

### Session-level split (Table 5)

| Session | Duration (hr) | Role (per BaRISTA) |
|---------|---------------|--------------------|
| 1 | 2.60 | Pretraining |
| 2 | 2.42 | Pretraining |
| 3 | 2.66 | Pretraining |
| 4 | 3.00 | Pretraining |
| 5 | 3.73 | Pretraining |
| 6 | 1.85 | Pretraining-stage validation |
| 7 | 3.52 | Downstream test |

### Pretraining segment split

Within the pretraining sessions, non-overlapping 3-second segments are
split 80/10/10 into pretraining train/valid/test. Used internally during
the pretraining loop for masked-reconstruction loss tracking, early
stopping, and learning-rate scheduling.

### Downstream task segment split (Table 6)

For each downstream task, labeled 3-second segments are extracted from
test session(s) and split 80/10/10 into downstream train/valid/test for
linear probe finetuning.

For Subject 2 (per BaRISTA Table 6 from session 7):

| Task | Train | Valid | Test |
|------|-------|-------|------|
| Sentence Onset | 1,036 | 129 | 129 |
| Speech/Non-Speech | 1,470 | 183 | 183 |
| Channel Reconstruction | 3,385 | 422 | 422 |

### Two-stage development workflow

The data is used differently in laptop development vs server evaluation.
This is intentional — the laptop phase is for building infrastructure;
the server phase is for the real ablation.

#### Stage 1: laptop development (M5)

Sessions 6 and 7 are treated as **fake test sets** during this stage.
This is for developing and debugging the evaluation pipeline — the
plumbing for running a trained model over a held-out session, computing
band-resolved NMSE, computing connectivity, getting downstream AUC.
You need *something* held out to build that infrastructure against, and
sessions 6/7 are convenient for that purpose.

Anything we look at during this stage — including casual peeks at
session 6/7 metrics — is considered contaminated for the purposes of
final reporting.

#### Stage 2: server evaluation

When moving to the server for real experiments, **fold sessions 6 and 7
back into training**. Reasoning: we've peeked at them during laptop
development, so they can't honestly be reported as held-out test
performance. By using all 7 sessions for training, we (a) maximize
training data for the within-subject experiments, and (b) avoid making
contaminated test claims.

Test-set performance for the within-subject experiments comes from
**other subjects' held-out sessions** (cross-subject evaluation, where
the model trained on Subject 2 is evaluated on a session from another
subject that was never used for any decisions). This is also more
faithful to BaRISTA's evaluation protocol, which reports performance
averaged across multiple test sessions.

Alternative if cross-subject evaluation is too far from the within-subject
focus of the project: hold out one of sessions 6/7 entirely from laptop
development (don't load it, don't compute on it) and use that session as
the real test set. The other becomes part of training. Decide which
approach before the laptop phase begins.

### Methodology discipline

- The full ablation matrix must be decided before the server evaluation
  begins. Discoveries that suggest adding a new loss or tokenizer
  mid-experiment break the protocol.
- Hyperparameters chosen during the laptop development phase are frozen
  before the server evaluation phase. No retuning if numbers look
  unexpected.
- The paper will explicitly describe this protocol so reviewers can
  verify nothing in the laptop phase contaminated the final test set.

## Conventions

### Preprocessing (BaRISTA-matched)

Pipeline order on raw h5 channels: **notch filter → Laplacian
rereference → segment → z-score**.

1. **Notch filter** — `scipy.signal.iirnotch` with Q=30, applied via
   `scipy.signal.lfilter` (causal; matches BaRISTA — not zero-phase
   `filtfilt`), at frequencies `[60, 120, 180, 240, 300, 360]` Hz. US
   power-line fundamental + harmonics.
   - **Gamma-band caveat:** 60 Hz sits at the boundary of the
     low-gamma band (30–50 Hz) and 120/180 Hz sit *inside* high-gamma
     (50–200 Hz). Q=30 makes each notch ~2 Hz wide, so a small fraction
     of high-gamma is removed. This is BaRISTA's published recipe; we
     match it for comparability. If it ever looks like the gamma loss
     work is sensitive to notch width, swap to `filtfilt` with the same
     Q and re-run; that's a downstream debugging knob, not Phase-1
     scope.

2. **Laplacian rereference** — for each electrode named `<stem><N>`
   (e.g., `LT3a1`), neighbors are `<stem><N-1>` and `<stem><N+1>` along
   the depth lead. Reref:
   `x_i' = x_i − mean(x_{N-1}, x_{N+1})`. Channels at lead endpoints
   (only one neighbor) are excluded. This must happen *before*
   segmenting and *after* notch — order matters because reref mixes
   power-line content across channels.
   - **Channel selection** for Subject 2: start from h5's 164 channels,
     drop the 29 in `corrupted_elec.json[sub_2]`, then drop any
     channel whose required neighbor is also corrupted or missing.
     The resulting set is the BaRISTA `clean_laplacian` equivalent
     (their per-subject `clean_laplacian.json` is not shipped with our
     BrainTree copy, so we construct it).

3. **Segment** into non-overlapping 3-second windows (6,144 samples at
   2,048 Hz).

4. **Z-score** each segment independently per channel (BaRISTA's
   choice; keeps NMSE values comparable to their Table 13).

### Patching

- **Patch** each 3-second segment into n=12 temporal patches of length
  L=512 (250 ms each). Tokenizer operates per patch, per channel.

## Development decisions made

1. **Within-subject development first**, cross-subject later. Use Subject
   2 (most data: ~14.4 hours pretraining + 2 held-out sessions). The 1/f
   mechanism doesn't depend on subject generalization, so this isolates
   the variable we're testing.
2. **Backbone: BaRISTA architecture** (d=64, 12 layers, dilated-CNN
   tokenizer at default config) plus a smaller variant (d=32, 6 layers)
   for the size-sensitivity check.
3. **Gamma definition: 50–200 Hz** (broadband gamma). Not concerned with
   ultra-high-gamma above 200 Hz.
4. **Compute target: server with 2× A40s** for training. M5 MacBook for
   development/debugging only. No serious training on Mac/MPS.
5. **Treat BaRISTA's repo as read-only.** It's at
   `/Users/wojemann/local_data/BaRISTA/`; reference but never modify.
   Their data loader is fair game to wrap in a thin adapter if useful.

## Implementation strategy

Build our own training framework from scratch in this repo. BaRISTA's
architecture is one configuration within the framework, not a starting
codebase to modify. Reasons: we'll be swapping tokenizers and losses
extensively, which would mean rewriting most of BaRISTA's code anyway,
and a config-driven framework keeps the experiment matrix tractable.

Key abstractions to build:

```python
class Tokenizer(nn.Module):
    """Maps (batch, channels, n_samples) -> (batch, channels, n_tokens, d_model)."""

class ReconstructionLoss(nn.Module):
    """Computes loss between predicted and target signals (or latents)."""
```

with concrete implementations swapped via config. The encoder transformer
and spatial encoding live in single, well-tested modules parameterized by
config.

**Sanity check, not full reproduction.** With `tokenizer=DilatedCNN,
loss=MSE`, our framework should produce reconstruction NMSE numbers in
the right ballpark of BaRISTA's Table 13 (low-freq ~0.4, high-freq > 1).
Off by 10% is fine. Off by 2x means there's a bug.

**Spatial encoding and reconstruction target are fixed design choices,
not experimental variables.** We use BaRISTA's region-level (atlas
parcel) spatial encoding across all configurations because (a) it was
their best-performing variant, (b) it provides a useful inductive bias
for the network-reconstruction validation experiment, and (c) holding it
constant isolates the effect of loss/tokenizer choices. The
reconstruction target is the per-channel raw waveform (BaRISTA's
downstream reconstruction setup), which is the natural substrate for
asking gamma-fidelity questions. Note: this requires per-channel atlas
parcel labels for every subject we train on. For Subject 2 these labels
are confirmed present (`localization/sub_2/depth-wm.csv`, Destrieux
column) — no lobe-level fallback needed.

## Experiment plan

### Loss ablation matrix (7 configurations, default tokenizer)

Each row addresses a different aspect of the 1/f problem or
autocorrelation structure:

1. **MSE** (baseline) — BaRISTA-matched, the configuration we're trying
   to beat.
2. **MAE / Huber** (robust regression baseline) — sanity check that any
   improvement over MSE comes from spectrum awareness, not just outlier
   handling.
3. **Whitened MSE** — FFT the patch, divide by √PSD (estimated from
   training data), MSE in whitened space. Principled spectrum-aware loss
   that treats every frequency bin equally. The cleanest test of the
   1/f hypothesis.
4. **Multi-resolution STFT loss** — sum of STFT-magnitude L1 across
   multiple (window, hop, FFT) configurations, e.g., [(64, 16, 64),
   (256, 64, 256), (1024, 256, 1024)]. The audio-domain SOTA recipe for
   high-fidelity waveform reconstruction.
5. **EEGM2-style composite** — temporal L1 + frequency-domain spectral
   loss. Direct EEG-SSL precedent; must be matched or beaten.
6. **DistDF (Bures-Wasserstein + MSE)** — covariance-aligning composite
   loss; addresses MSE's autocorrelation bias.
7. **cMin-LogCosh** (BrainState) — time-shift-invariant variant.
   Addresses temporal alignment as orthogonal axis to spectral weighting.

If a clear winner emerges from the first six, an additional composite
combining the winner with multi-resolution STFT is worth a single extra
row. This is decided BEFORE seeing test-set numbers.

### Tokenizer ablation matrix (4 configurations, best loss from above)

Each row varies the time/frequency resolution and feature-learning
properties:

1. **Dilated CNN** (BaRISTA default) — raw waveform, full phase, learned
   features.
2. **Linear projection on raw patch** — raw waveform, full phase, no
   feature learning. Establishes whether learned features matter.
3. **STFT magnitude + linear** (BrainBERT-style) — frequency-resolved,
   no phase, fixed time resolution.
4. **Wavelet packet decomposition + linear** (Hi-WaveTST-style) —
   frequency-resolved, phase preserved, adaptive resolution. Specifically
   designed for high-frequency feature extraction.

Total core ablation: 7 losses + 4 tokenizers, with the BaRISTA-default
configuration shared between both = ~10 unique training runs.

### Phased experiments

- **Phase 1 (laptop, sessions 1-5 for training, 6/7 as fake test sets)**:
  Build framework. Reimplement BaRISTA architecture as baseline config.
  Validate that pretraining loss decreases on the pretraining-internal
  validation split. Confirm preprocessing preserves gamma. Build the
  full evaluation pipeline against fake-test sessions 6/7. Develop the
  ablation pipeline on a small subset.
- **Phase 2 (server, sessions 1-7 all in training)**: Loss ablation —
  train all 7 loss configurations with default tokenizer using all of
  Subject 2's sessions for training. Evaluate on held-out test data
  (cross-subject sessions, or one of sessions 6/7 fully untouched on
  laptop — decided before Phase 1).
- **Phase 3 (server)**: Tokenizer ablation — train 4 tokenizer
  configurations with best loss from Phase 2. Same training data and
  test set as Phase 2.
- **Phase 4 (server)**: Network reconstruction validation. Compute
  gamma envelope correlation matrices on real and reconstructed signals
  from the test set; compare with Frobenius distance and off-diagonal
  Pearson.
- **Phase 5+**: Cross-subject generalization, model size sensitivity,
  pretrained decoder probes.

## Repo structure

```
gamma-recon/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── gamma_eval/                  # evaluation harness (already built)
│   ├── metrics/
│   │   ├── reconstruction.py    # core metric functions
│   │   └── prefilter.py         # prefilter-once-slice-many path
│   ├── synthetic/signals.py     # 1/f + burst generator for testing
│   ├── evaluator.py             # ReconstructionEvaluator wrapper
│   └── demo.py                  # synthetic BaRISTA-failure-pattern demo
├── gamma_encoder/               # model framework (skeleton built; see "Framework state")
│   ├── models/                  # encoder transformer, decoder, full model, spatial encoder
│   ├── tokenizers/              # DilatedCNN built; Linear/STFT/WPD pending
│   ├── losses/                  # MSE built; WhitenedMSE/MultiResSTFT/EEGM2/DistDF/cMin-LogCosh pending
│   ├── data/                    # BrainTreebank loader + preprocess primitives
│   └── training/                # overfit harness + MetricsLogger seam
├── tests/                       # pytest suite (88 tests pass)
├── notebooks/                   # currently empty; pilot notebook not yet migrated
├── notes/                       # markdown notes from Claude Code sessions
├── scripts/                     # entry-point training scripts
├── configs/
│   └── config.py                # path profile selector (local/server) — already built
├── results/                     # plots, tables, checkpoints metadata
└── brain_tree/                  # prior/side analysis (NOT part of the main framework)
    ├── notebooks/trial0_sentence_onset_gamma_baseline.ipynb
    └── scripts/                 # holdout analysis, per-trial analysis, BrainTreebank utils
```

**Note:** `brain_tree/` contains prior exploratory work (holdout
validation, per-trial speech-onset analysis with ROC curves, etc.). It
is separate from the gamma_encoder framework being built here. The
pilot result referenced as `notebooks/01_gamma_predicts_speech.ipynb` in
the thesis section is the analysis in
`brain_tree/notebooks/trial0_sentence_onset_gamma_baseline.ipynb` —
that notebook has not been migrated/renamed yet.

## Framework state (gamma_encoder)

What's built and exercised end-to-end. A more thorough walkthrough lives
at `notes/codebase_walkthrough.md`.

### Modules

- **`data/preprocess.py`** — pure-numpy primitives: `notch_filter`
  (iirnotch + lfilter, BaRISTA-matched), `parse_electrode_name`
  (defensive: strips `*#_` markers à la BaRISTA's `_elec_name_strip`),
  `build_laplacian_neighbors`, `apply_laplacian_reref`, `segment_signal`,
  `zscore_segment`.
- **`data/braintreebank.py`** — h5 → preprocessed-segments loader for
  Subject 2. Wraps the preprocess primitives.
- **`tokenizers/base.py`** — `Tokenizer` ABC with shape contract
  `(B, C, n_patches, L) → (B, C, n_patches, d_model)`, plus
  `patchify`/`unpatchify` helpers.
- **`tokenizers/dilated_cnn.py`** — dilated-CNN tokenizer. **Deliberate
  divergence from BaRISTA's `TSEncoder2D`:** see "BaRISTA-faithful vs.
  gamma-fidelity" below.
- **`models/{encoder,decoder,spatial_encoder,full_model}.py`** —
  stock-PyTorch transformer encoder (pre-norm, sinusoidal PE), linear
  patch decoder, region-embedding spatial encoder, glue config +
  `GammaEncoderModel`. The encoder backbone is **stock**, not the
  RoPE+RMSNorm+GatedMLP stack BaRISTA uses; that's a deferred faithfulness
  swap, not a thesis-relevant variable.
- **`losses/{base,mse}.py`** — `ReconstructionLoss` ABC + `MSELoss`.
  Other six losses pending.
- **`training/overfit.py`** — single-batch overfit harness. CLI:
  `python -u -m gamma_encoder.training.overfit --batch <pt> --tokenizer <name> --loss <name> --steps N --lr X --out-dir <dir> --log-every K`.
- **`training/logging.py`** — `MetricsLogger` ABC with backends
  `StdoutLogger`, `JsonlLogger`, `WandbLogger` (lazy import — no wandb
  dep on the laptop), and `MultiLogger` for fan-out. The training loop
  never hard-codes a backend; pick the combo at the call site.

### Cached overfit batch + verified end-to-end

`results/overfit_batch.pt` — fixed batch from sub_2 trial 0, shape
`(16 segments, 8 channels, 6144 samples)` with `region_ids` and
`fs=2048`. Channels selected greedily for diverse Destrieux parcels.
This is the canonical "does the pipeline work" artifact; it should not
change between runs unless we deliberately rebuild it. Builder:
`scripts/cache_overfit_batch.py`.

End-to-end smoke verified (2026-04-30): on this batch with
`tokenizer=dilated_cnn`, `loss=mse`, `d_model=32`, `n_layers=6`, `lr=1e-3`,
200 steps drives loss from **2.90 → 0.17** (~6% of initial) in ~108 s on
the M5 CPU. Loss is still decreasing at step 200 — convergence isn't
reached, but monotonic descent over the full run confirms the pipeline
is sound. Artifacts at `results/overfit_runs/mse_dilated_first/`
(`config.json`, `metrics.jsonl`, `summary.json`,
`overfit_dilated_cnn_mse.json`).

**Full 15-cell sweep at 1000 steps (2026-04-30):** all 10 losses ×
`dilated_cnn` and 5 non-default tokenizers × `mse` overfit successfully
on the cached batch — every cell descended monotonically, no nan/inf,
no failure-to-converge. Pipeline is sound across the entire ablation
matrix. Reference floor: **`mse + dilated_cnn` final loss = 0.0702**
(`frac_remaining ≈ 2.4%`); this is what every (loss, tokenizer) combo
must beat *on real data*, not on this batch. Tight cluster in the
1.5–4% range: `huber, distdf, cmin_logcosh, mse, complex_stft,
stft_magnitude, linear, wavelet_packet`. Slower convergers (not
failures, may need more steps or lower lr): `content_aware_l1` (11%),
`multires_stft` (12%), `eegm2` (17%), `log_power_spectral` (31%, plateaued
with oscillation around 1.0 — `lr=1e-3` likely too hot). `welch_psd`
tokenizer floored at 5.3% (expected — drops per-patch time resolution).
Full numbers, timing, and per-loss flags in
`notes/laptop_overfit_smoke_results.md`; raw artifacts in
`results/overfit_runs/`.

**Convergence-speed on a 16-segment batch is not a quality ranking.**
`frac_remaining` mostly reflects loss-specific scale and curvature;
real-data band-resolved NMSE is the actual probe.

**`whitened_mse` is not real-data ready.** The 1000-step sweep used
on-the-fly target PSD whitening, which produced an initial-loss scale
of ~3.9e9 and makes any cross-loss or composite comparison meaningless.
Fix before wiring it into anything: estimate PSD once on training data
(or apply a sane floor on the on-the-fly estimate) and freeze it.

### BaRISTA-faithful vs. gamma-fidelity divergences

Three places where our default config diverges from BaRISTA on purpose.
Future sessions should not "fix" these without understanding the
rationale.

1. **Tokenizer head (deliberate, thesis-motivated).** BaRISTA's
   `TSEncoder2D` keeps a learned MLP from the full conv-stack output to
   the d_model token (`barista/models/tokenizer.py:49`,
   `temporal_pooler = MLP(d_input=temporal_subsegment_len, d_out=d_h)`).
   An earlier draft of our tokenizer used `nn.AdaptiveAvgPool1d(1)` to
   collapse time, which is a free low-pass on the conv output and works
   directly against the gamma-fidelity probe. Replaced with
   `nn.Linear(hidden * patch_samples, d_model)` operating on the
   flattened conv output. Adds ~262k params at default
   (hidden=16, L=512, d_model=32) — fine for laptop overfit, may want
   to revisit on server. Conv1×1-then-Linear is a cheaper alternative if
   tokenizer params dwarf the transformer.
2. **Encoder backbone (stock, faithfulness-deferred).** BaRISTA uses
   RoPE attention, RMSNorm, GatedMLP. We use stock
   `nn.TransformerEncoderLayer` with pre-norm + sinusoidal PE. This is
   a faithfulness gap, not a thesis-relevant variable; swap for the
   "BaRISTA-faithful baseline" run when reproducing Table 13.
3. **Conv-block structure (faithfulness-deferred).** BaRISTA's
   `ConvBlock` is two `SamePadConv` layers + LayerNorm + residual; ours
   is one conv + GELU, no norm, no residual. Same parity bucket as the
   encoder: matters for the faithful baseline, not for the loss/tokenizer
   ablation.

A 4th not-quite-divergence worth flagging: BaRISTA's
`_laplacian_rereference` pulls neighbor signals from the **full
unfiltered electrode list** (including corrupted channels) and notches
them on the spot, while ours notches all channels first and only allows
non-corrupted neighbors. Numerically equivalent on a correctly-curated
clean list; semantically different. We construct the clean list
ourselves (BaRISTA's `clean_laplacian.json` isn't shipped with our
BrainTree copy) so the two pipelines may diverge by a few channels for
Subject 2. Diff before reporting reproduction numbers.

### Where to pick up next

Items 1–3 (overfit to convergence, wire next loss, wire next tokenizer)
are done as of the 2026-04-30 sweep. Updated priority order:

1. **`gamma_eval` band-resolved NMSE on the overfit batch** for a few
   of the converged checkpoints. This is the first signal of whether
   any of these losses actually move high-gamma NMSE downward on real
   data; convergence-speed on the cached batch can't tell us that.
   Until this is in hand, none of this approaches the server.
2. **Fix `whitened_mse` PSD handling** before it's included in any
   cross-loss comparison. Estimate PSD once on training data and
   freeze it; the on-the-fly path produces uninterpretable
   initial-loss scales (~4e9).
3. **One retry of `log_power_spectral` at `lr=3e-4`** before drawing
   conclusions about its convergence behavior. `lr=1e-3` plateaus
   around 1.0 with mild oscillation from step ~300 onward.
4. **Tiny notebook to plot `metrics.jsonl` curves side-by-side.**
   `comparison_loss_axis.png` / `comparison_tokenizer_axis.png` exist
   from the sweep but a notebook makes incremental additions easier.
   `pd.read_json(..., lines=True)` is enough — no plotting infra
   needed yet.

Everything beyond #4 (sweep harness, faithful BaRISTA stack, masked
reconstruction, streaming loader, Subject 2 full-session pretraining)
is server-side work and shouldn't be attempted on the M5.

## Pre-computed segment indices (sub_2)

Lookup tables of `(start_sample, end_sample)` ranges into the BrainTreebank
h5 files for sub_2 trials 000–006. **No neural data is copied** — these are
just timestamps + labels + metadata so the training/finetuning loops can
load 3-s windows by indexing the h5 directly.

Generator: `scripts/build_segment_indices.py` (path-profile aware,
`--profile local|server`). Idempotent and seeded; re-running produces
byte-identical parquets.

Output layout (committed to the repo):

```
segment_indices/
├── manifest.json                          # per-trial counts, seeds, sanity checks
└── sub_02/
    ├── trial000__pretrain.parquet
    ├── trial000__sentence_onset.parquet
    ├── trial000__speech_nonspeech.parquet
    ├── trial000__volume.parquet
    ├── trial000__optical_flow.parquet
    └── ... (5 files × 7 trials = 35 parquets)
```

Two evaluation schemes covered:

1. **`pretrain`** — non-overlapping 3-s windows tiling the trigger-bounded
   valid window (between the timing CSV's `beginning` and `end` markers).
   Random 80/10/10 train/valid/test split with seed 42. The same tiling
   doubles as the **channel-reconstruction** candidate pool — no separate
   file for that task.
2. **`main`** (BaRISTA Appendix A) for four binary tasks:
   - `sentence_onset` — positives = words with `is_onset==1`
   - `speech_nonspeech` — positives = every word
   - `volume` — top vs. bottom quartile of `rms`, middle two quartiles dropped
   - `optical_flow` — top vs. bottom quartile of `max_global_magnitude`
     (this column choice is a guess; easy to swap)

   Positives are 3-s windows centered on the relevant word/onset.
   Negatives are 3-s windows containing **no** word interval (BaRISTA's
   stricter rule, not PopT's center-1-s rule). Greedy chronological
   non-overlap (keep first); kept positives win the cross-prune against
   negatives. Class-balanced (subsample larger class, seed 42), then
   80/10/10 random split (seed 42).

The chronological / Appendix-K "extended" scheme is **not** generated —
deliberately dropped to simplify the experimental paradigm.

### Row schema (every parquet)

```
session_id        str    "sub_02_trial_000"
subject_id        int    2
trial_id          int    0..6
movie             str    transcript subdir name (e.g., "venom")
task              str    pretrain | sentence_onset | speech_nonspeech | volume | optical_flow
session_role      str    pretrain | downstream_val | downstream_test
                         (CLAUDE.md mapping: trials 000-004=pretrain, 005=val, 006=test)
split             str    train | valid | test
split_seed        int    42
start_sample      int64  inclusive sample index into the h5
end_sample        int64  exclusive; (end - start) == 6144 always
label             int8   0/1 for binary tasks; -1 for pretrain rows
center_sample     int64  word/onset sample for centered positives, else -1
source_word_idx   int64  row index into the movie's features.csv, else -1
notes             str    e.g., "no_speech_negative", "volume_top_quartile_pos"
```

### Time → sample mapping

Word times in `transcripts/<movie>/features.csv` are movie-relative
seconds. Mapping to h5 sample indices uses the per-trial trigger table
(`subject_timings/sub_2_trial<NNN>_timings.csv`):

```python
sample_idx = round(np.interp(word_time_s,
                             triggers["movie_time"],
                             triggers["index"]))
```

Triggers are dense (~12 Hz) and `diff` is non-constant, so per-trigger
interpolation absorbs mild clock drift without sub-sample bias. The
"valid window" for pretrain tiling is the `[beginning.index, end.index)`
range from the same CSV.

### Sanity-check vs. BaRISTA Table 6 (sub_2 row)

Comparing against trial006 (the downstream-test session per CLAUDE.md's
session-role mapping):

| Task | Expected (train/val/test) | Got | Delta |
|------|---------------------------|-----|-------|
| sentence_onset | 1036 / 129 / 129 | 1040 / 130 / 130 | +0.4% |
| speech_nonspeech | 1470 / 183 / 183 | 1498 / 187 / 187 | +1.9% |
| channel_recon (= trial006 pretrain tiling) | 3385 / 422 / 422 | 2954 / 369 / 370 | **−12.7%** |

The labeled-task counts match within ~2%. The channel_recon shortfall is
explained by valid-window choice: `4229 × 3 s = 3.52 hr` (Table 5
trial007 row exactly), so BaRISTA appears to tile the full h5
(~3.52 hr) rather than the trigger-bounded valid window (~3.08 hr we
use). Easy knob to flip in `pretrain_tiling()` if exact replication of
that table cell becomes important.

### How to use at training/finetuning time

```python
import h5py, pandas as pd

df = pd.read_parquet("segment_indices/sub_02/trial006__sentence_onset.parquet")
train = df[df["split"] == "train"]
with h5py.File(f"sub_2_trial{int(train.iloc[0].trial_id):03d}.h5") as f:
    for _, row in train.iterrows():
        seg = f["data/electrode_42"][row.start_sample : row.end_sample]  # length 6144
        label = row.label
```

`split` and `session_role` are independent: `session_role` tags the
trial's intended role per CLAUDE.md (pretrain / downstream_val /
downstream_test), `split` is the per-task 80/10/10 random fold.
Filter however the experiment requires — every trial gets indices for
every task, the trainer decides which to use.

### Notes / caveats

- **Sentence onsets <3 s apart get pruned by greedy non-overlap.** Rare
  in practice; it's the same code path as `speech_nonspeech` so it
  isn't a special case.
- **Negative pool = pretrain tiling filtered to no-speech windows** — so
  negative starts are always multiples of 6144 from `valid_window[0]`,
  while positive starts can be anywhere. That asymmetry is intentional
  (negatives have to fit the no-speech rule cheaply).
- **The optical-flow column (`max_global_magnitude`) is a guess.** The
  BTB / PopT papers reference an "optical flow" feature without naming
  the exact column; `max_mean_magnitude` and `max_median_magnitude`
  are alternatives. Swap in `_build_main_task` if needed.
- **Reproducibility.** All seeds are constants at the top of the script:
  `SEED_PRETRAIN = SEED_MAIN = 42`. Re-running with the same data
  produces byte-identical parquets.

## Evaluation harness conventions

- Signals are arrays of shape `(n_channels, n_samples)` or, when batched,
  `(n_segments, n_channels, n_samples)`.
- Default frequency bands: delta-theta (1–8 Hz), alpha-beta (8–30 Hz),
  low-gamma (30–50 Hz), high-gamma (50–200 Hz).
- **Use the pre-filtered evaluation path for real data.** Filter the long
  session signal once via `prefilter_signal`, then call
  `evaluator.accumulate_segment(pre_true, pre_pred, start, stop)` per
  segment. Avoids filter edge artifacts on short segments.
- Connectivity is computed once at the end on the pooled session signal
  via `evaluator.compute_connectivity(true_pool, pred_pool)`. NOT
  per-segment — short windows produce noisy correlation matrices.
- Sampling rate `fs=2048` matches BrainTreebank.
- NMSE definition: `MSE / Var(true)`, computed per-channel, then
  averaged. Assumes z-scored input segments (BaRISTA convention).

## Style / approach guidance for Claude Code sessions

- **Don't make architectural decisions silently.** If something requires
  a non-obvious choice (band edges, filter type, normalization), surface
  the choice explicitly with a brief justification.
- **Tests use synthetic data first.** Any new metric or loss should have
  a test using `gamma_eval.synthetic.signals` that verifies expected
  behavior before being applied to real data.
- **No Mac training.** If asked to run pretraining, default to "you
  should run this on the server"; don't try to make it work on MPS. The
  M5 is fine for forward-pass sanity checks, single-batch debugging, and
  data loader testing.
- **The user has a strong neuroscience background.** Don't over-explain
  basic concepts (1/f, gamma, BOLD coupling). Do explain ML choices.
- **Reconstruction quality is a *probe*, not necessarily the final
  pretraining objective.** The thesis is about what the loss/tokenizer
  choice reveals about what the model represents, not about whether
  reconstruction is the right pretraining objective per se. This frames
  around the PopT counterargument.
- **Everything outside `/Users/wojemann/local_data/gamma-recon/` is
  read-only.** This includes BaRISTA at
  `/Users/wojemann/local_data/BaRISTA/`, BrainTreebank data at
  `/Users/wojemann/local_data/BrainTree/`, and every other sibling
  directory. Reading is fine; any write/edit/delete/move/rename outside
  gamma-recon requires explicit user approval before acting. Cache
  files, intermediate outputs, scratch artifacts — all of it lives
  inside the repo (`results/`, `notes/`, etc.).
- **Sessions 6 and 7 are fake test sets during laptop development.**
  Anything you compute on them during this stage is for building the
  evaluation pipeline. The plan is to fold them back into training when
  moving to the server, since they'll be contaminated by laptop-stage
  peeking. The real test set comes from elsewhere (see Subject 2 splits
  section).
- **Keep notes on what didn't work.** When you (Claude Code) go down a
  wrong path and have to course-correct, add the correction to this
  CLAUDE.md so future sessions don't make the same mistake.
- **Path-profile requirement for runnable code.** Any script/notebook that
  accesses filesystem paths should expose a path profile selector
  (`local` or `server`) so runs from CLI or IDE can choose the correct
  hardcoded path set without editing code.

## Commands

Set up environment (M5): use the existing `gamma-env` conda env.
```
conda activate gamma-env
```

**Mac OpenMP gotcha.** The conda env's MKL/scipy and pip-installed torch
both ship libomp, which trips the duplicate-OpenMP guard at import time.
Workaround: `export KMP_DUPLICATE_LIB_OK=TRUE`. Long-term fix is to
install torch via the conda-forge channel into the env so there's only
one libomp, but the env var is fine for laptop dev.

Run tests:
```
pytest tests/ -v
```

Run the synthetic demo:
```
python -m gamma_eval.demo
```

Expected demo output (pre-filtered path):
- oracle: all NMSE ~0, conn_r ~1
- smoothed: delta-theta ~0, high-gamma ~0.68, conn_r ~1
- low_freq_only: low-freq bands ~0, high-gamma ~1.0, conn_r ~0.2
