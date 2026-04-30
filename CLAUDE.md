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

- **BaRISTA** (`14145_BaRISTA_Brain_Scale_Info.pdf`): primary architectural
  reference. Lightweight (d=64, 12 layers), channel-flexible, raw-waveform
  input, MSE loss in latent space + linear head to raw signal. We
  reimplement their architecture as one configuration in our own
  framework — see "Implementation strategy" below.
- **BrainBERT** (Wang et al. 2023, `2302.14367v1.pdf`): predecessor;
  STFT/superlet tokenizer, content-aware L1 loss. Important precedent for
  non-trivial reconstruction loss.
- **PopT** (Chau et al. 2025, `2406.03044v4.pdf`): discriminative-only
  pretraining, no reconstruction. Argues reconstruction is the wrong
  objective. Counterargument we need to address — see "Framing".
- **Merk et al. 2025** (`2508.10160v1.pdf`): proposes a 1/f-scaled MAE in
  spectrogram domain for DBS LFP modeling. Closest direct precedent to
  the loss work, but tiny model, no gamma evaluation, no downstream task
  analysis.
- **KenazLBM** (`2025.08.10.669538v1.pdf`): preprocesses by *filtering
  out* everything above 121 Hz, sidestepping the question entirely. Loss
  function not specified in paper.

## The dataset: BrainTreebank

- 10 epilepsy patients, 26 sessions watching movies, sEEG at 2048 Hz,
  ~16 electrodes per subject (1688 total).
- Labels available: sentence onset, speech vs non-speech, word volume,
  word pitch, scene labels, face counts, GPT-2 surprisal, etc.
- Documentation: `2411.08343v1.pdf`.
- Dataset URL: `https://braintreebank.dev`.
- Data is on the M5 at `~/projects/BrainTree/data/`.

## Development decisions made

1. **Within-subject development first**, cross-subject later. Use Subject
   2 (most data: ~14 hours pretraining + validation + test session). The
   1/f mechanism doesn't depend on subject generalization, so this
   isolates the variable we're testing.
2. **Backbone: BaRISTA architecture** (d=64, 12 layers, dilated-CNN
   tokenizer at default config) plus a smaller variant (d=32, 6 layers)
   for the size-sensitivity check.
3. **Gamma definition: 50–200 Hz** (broadband gamma). Not concerned with
   ultra-high-gamma above 200 Hz.
4. **Compute target: server with 2× A40s** for training. M5 MacBook for
   development/debugging only. No serious training on Mac/MPS.
5. **Treat BaRISTA's repo as read-only.** It's at `~/projects/BaRISTA/`;
   reference but never modify. Their data loader is fair game to wrap in
   a thin adapter if useful.

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
parcel labels for every subject we train on. For Subject 2 specifically,
verify these labels are present in the BrainTreebank metadata before
designing the data loader around them — fall back to lobe-level encoding
(BaRISTA's second-best variant) only if parcel labels are unavailable.

## Experiment plan

Five-week sequence (rough; iterate as needed):

- **Week 1**: Build framework skeleton. Reimplement BaRISTA architecture
  as the baseline config. Validate against Table 13 numbers. Confirm
  preprocessing preserves gamma.
- **Week 2**: Loss ablation — MSE, MAE, log-cosh (sanity), Merk-style
  1/f-scaled MAE, **whitened MSE** (the principled version), log-power
  spectral L1, composite (time MSE + spectral L1), magnitude-reweighted
  L1 (BrainBERT adaptation). Hold tokenizer = DilatedCNN.
- **Week 3**: Tokenizer ablation — DilatedCNN, linear, STFT, superlet,
  raw-patch. Hold loss = best from Week 2.
- **Week 4**: Network reconstruction validation. Compute gamma envelope
  correlation matrices on real and reconstructed signals; compare with
  Frobenius distance and off-diagonal Pearson.
- **Week 5+**: Cross-subject generalization, downstream task evaluation.

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
├── gamma_encoder/               # to be built (the model framework)
│   ├── models/                  # encoder transformer, predictor, decoder
│   ├── tokenizers/              # DilatedCNN, Linear, STFT, Superlet, etc.
│   ├── losses/                  # MSE, MAE, WhitenedMSE, LogPowerSpectralL1, ...
│   ├── data/                    # BrainTreebank loader (may wrap BaRISTA's)
│   └── training/                # training loop, config schemas
├── tests/                       # pytest suite (49 tests as of now)
├── notebooks/
│   └── 01_gamma_predicts_speech.ipynb   # pilot result
├── notes/                       # markdown notes from Claude Code sessions
├── scripts/                     # entry-point training scripts
├── configs/                     # experiment configs
└── results/                     # plots, tables, checkpoints metadata
```

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
- **Treat BaRISTA's repo (`~/projects/BaRISTA/`) as read-only.** Reference
  it but never modify it.
- **Keep notes on what didn't work.** When you (Claude Code) go down a
  wrong path and have to course-correct, add the correction to this
  CLAUDE.md so future sessions don't make the same mistake.
- **Path-profile requirement for runnable code.** Any script/notebook that
  accesses filesystem paths should expose a path profile selector
  (`local` or `server`) so runs from CLI or IDE can choose the correct
  hardcoded path set without editing code.

## Commands

Set up environment (M5, first time):
```
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

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
