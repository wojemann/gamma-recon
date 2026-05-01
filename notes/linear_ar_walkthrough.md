# How the LinearVAR predictions are generated

A walkthrough of the multivariate-AR (MVAR / VAR) baseline used in the
laptop-stage band-resolved sweep, end to end. Goal: when you look at
`band_eval_linear_ar_axis.png` and want to know exactly what computation
produced those numbers, the answer is here.

## What the model is

`gamma_encoder/models/linear_ar.py:LinearVARModel` implements a single
**causal vector AR(p) FIR predictor with full cross-channel mixing**,
default `p = 3`. For a signal with `C` channels:

```
y[c, t] = b[c] + Σ_{c'=1..C} Σ_{k=1..p}  W[c, c', k] * x[c', t-k]
```

— each output channel at time `t` is a linear function of the past `p`
samples of *every* channel, including itself. Samples before `t = 0`
are treated as zeros (left-padded). Parameters:

- `W ∈ R^{C × C × p}` — full VAR coefficient tensor (lag `k` matrix is
  `W[:, :, k]`),
- `b ∈ R^{C}` — per-channel bias.

For Subject-2's overfit batch (`C=8, p=3`) that's `8·8·3 + 8 = 200`
learnable parameters — about three orders of magnitude smaller than
the 360k-param transformer it sits next to in the sweep, but with the
*same kind* of cross-channel pooling. This is the point of the model:
it gives the loss exactly enough capacity to mix across channels (the
way attention does in the transformer) and nothing more.

The forward signature matches the transformer's,
`model(signal, region_ids) -> recon`, with `signal` and `recon` both
shape `(B, C, T)`. `region_ids` is accepted for interface compatibility
and ignored — VAR has no spatial encoder.

### Why MVAR rather than per-channel AR

A previous version of this baseline applied the *same* `(w_1, w_2, w_3, b)`
to every channel independently — 4 total parameters, no cross-channel
mixing. The fairness problem there was that the transformer's attention
explicitly pools across channels, so any gap between the two models
conflated "the loss can't drive cross-channel learning" with "the
model has no spatial pathway at all." The MVAR variant closes that
gap: it is the smallest linear, strictly causal model that has the
same input access pattern as the transformer's per-token aggregation.
Anything the transformer does that VAR can't is therefore non-linear or
non-causal, not a spatial-aggregation issue.

### Initialization

`W` is zeroed and then `W[:, :, p-1] = I_C` (identity at lag-1) and
`b = 0`. So at step zero the model implements `y[c, t] = x[c, t-1]` —
each channel is a 1-sample-delayed copy of itself, no cross-coupling.
This matters for two reasons:

1. The model starts producing target-shaped output at step 0 (delayed
   identity), so the initial loss is small for time-domain losses and
   essentially zero for shift-invariant losses.
2. Cross-channel coupling has to be *learned* from gradient signal —
   it is not present at init. So if the loss has no way to push toward
   spatial mixing (e.g. a magnitude-only spectral loss with no cross-channel
   coupling term), the off-diagonal `W` entries stay near zero and the
   model behaves like the per-channel identity baseline.

### Implementation detail

The forward pass is one `nn.Conv1d(C, C, kernel_size=p)` applied to a
left-padded input (`F.pad(signal, (p, 0))`). PyTorch's `Conv1d` follows
the cross-correlation convention `y[t] = Σ_k w_conv[k] * x[t+k]`, so
the conv kernel index `k_conv` corresponds to **AR lag `p - k_conv`**:
- `conv.weight[:, :, p-1]` — lag-1 matrix `A_1`,
- `conv.weight[:, :, p-2]` — lag-2 matrix `A_2`,
- `conv.weight[:, :, 0]`   — lag-`p` matrix `A_p`.

The output length after the pad is `T + 1`; the trailing sample is
sliced off so the recon shape matches the input.

`tests/test_linear_ar.py` verifies this end-to-end:
- `test_linear_ar_strict_causality` — perturbing `signal[..., k]` may
  not change recon at indices `≤ k`, even with random non-zero
  cross-channel weights.
- `test_linear_ar_cross_channel_coupling_works` — with a hand-built
  `W` where channel-0 drives channel-2 at lag-1, poking channel 0 at
  `t=10` changes channel 2 at `t=11` and only at `t=11`.
- `test_linear_var_recovers_known_coupling` — fits the model to a
  synthetic 2-channel VAR(3) process with known cross-coupling
  `A1, A2, A3`. After 400 Adam steps the learned conv kernel matches
  ground truth within 0.07.

## Channel-masked pretraining

As of the channel-masking pivot, training masks `k=4` of 8 channels
per segment per step. The MVAR forward zero-fills masked channels'
inputs before the conv, so the prediction for any output channel is
purely a function of the unmasked channels' lagged values plus the
bias. The harness slices the loss to the masked channels' outputs
only — that's the supervision signal.

What this probes specifically: **the off-diagonal entries of `W`**.
Identity-shift init has zero off-diagonals, so the masked-channel
prediction at step 0 is just the bias (≈ 0). Driving the masked
loss down requires learning non-zero `W[masked_c, unmasked_c', k]`
that capture cross-channel temporal coupling. So masked-channel MVAR
is a clean probe of "did the model learn cross-channel structure."

## How it gets trained

Same harness as every other config in the sweep:
`gamma_encoder/training/overfit.py:run_overfit`. The only difference
is that `model_type="linear_ar"` is passed in, which makes the harness
build a `LinearVARModel(num_channels=C, order=ar_order)` instead of a
`GammaEncoderModel + tokenizer`. Everything else is identical:

- AdamW, lr=1e-3, weight_decay=0, seed=0,
- 500 steps,
- single fixed batch (`results/overfit_batch.pt`, shape
  `(16, 8, 6144)`, `fs=2048`),
- one of the ten loss objectives (`mse`, `mae`, ..., `content_aware_l1`),
- `torch.save({state_dict, model_type, tokenizer, loss, config}, model.pt)`
  at the end. `config` carries `ar_order` so the eval scripts can
  rebuild the right shape.

Per-step loss is logged to `results/overfit_runs/<loss>__linear_ar/metrics.jsonl`
identically to the transformer runs.

## How the eval predictions are produced

`scripts/run_band_eval.py` walks every `results/overfit_runs/*` that
has a `model.pt`, runs inference on the **same cached batch the model
was trained on**, and pipes the result into
`gamma_eval.evaluator.ReconstructionEvaluator`. For the LinearVAR runs
the eval looks like:

```python
blob = torch.load(run_dir / "model.pt")
model = LinearARModel(                         # alias for LinearVARModel
    num_channels=segments.shape[1],
    order=blob["config"]["ar_order"] or 3,
)
model.load_state_dict(blob["state_dict"])
model.eval()

with torch.no_grad():
    recon = model(segments, region_ids)        # (16, 8, 6144)

evaluator = ReconstructionEvaluator(fs=2048)
evaluator.accumulate(segments.numpy(), recon.numpy())
metrics = evaluator.summarize()
```

`accumulate` iterates over the 16 segments. Per segment it calls
`evaluate_reconstruction(true, pred, fs, bands)`, which:

1. **Bandpass-filters both `true` and `pred`** to each of the four
   default bands — `delta_theta=(1, 8)`, `alpha_beta=(8, 30)`,
   `low_gamma=(30, 50)`, `high_gamma=(50, 200)` Hz — using a 4th-order
   zero-phase Butterworth (`scipy.signal.sosfiltfilt`).
2. **Computes per-channel NMSE** within each band:
   `nmse = MSE(true_band, pred_band) / Var(true_band)`. NMSE = 0 means
   perfect match, NMSE = 1 means no better than predicting the mean.
   This is the BaRISTA Table 13 metric.
3. Stores per-channel NMSE arrays per band in a `ReconstructionReport`.

After all 16 segments, `evaluator.summarize()` concatenates the
per-segment per-channel arrays (16×8 = 128 values per band) and reports
the mean. That mean is what shows up in `band_eval.csv` and the
heatmaps as e.g. `nmse_high_gamma_mean=0.38`.

### Caveat: same batch for train and eval

This is laptop-stage smoke testing. The model overfit on the cached
batch and is evaluated on the same batch — band-resolved NMSE here
measures *fitting capacity*, not generalization. The point at this
stage is to verify the pipeline and surface qualitative differences;
real test numbers come later from held-out data on the server.

## Why the LinearVAR scores look the way they do

A few things to keep in mind when reading the heatmap:

1. **Identity-shift floor.** Init is `y[c, t] = x[c, t-1]` per channel,
   which after bandpass-filtering both true and pred is just a 1-sample
   phase shift. The NMSE of that copy is small at every band because
   one sample at fs=2048 Hz is a tiny phase shift even for 200 Hz
   content (~17° at 200 Hz, ~0.7° at 8 Hz). So the LinearVAR's "good
   gamma NMSE" partly reflects this structural floor — it inherits
   gamma faithfulness from the input rather than learning it from the
   loss. Cross-channel coupling has to actually be learned to do
   *better* than this floor.

2. **Some losses don't move the AR weights at all.**
   `multires_stft`, `eegm2`, and `cmin_logcosh` show essentially flat
   loss curves on LinearVAR (init ≈ final). For the magnitude-spectral
   losses the magnitude spectrum of `y[c, t] = x[c, t-1]` matches that
   of `x[c, t]` channel-by-channel, so the loss is already at its
   minimum at init and gradients are ≈ 0. For `cmin_logcosh` the loss
   minimizes over circular shifts, so the 1-sample delay maps to the
   trivial shift and the loss is ≈ 0 at init too. These rows tell you
   the loss has **no leverage on identity-initialized cross-channel
   linear taps** — the off-diagonals never get a gradient. The
   band-resolved NMSE numbers there are essentially "what does the
   identity-shift init score?" regardless of the loss.

3. **The contrast with the transformer is the headline.** A 200-param
   VAR scoring `high_gamma ≈ 0.4` is not impressive in absolute terms
   — most of that comes from the identity-shift floor, with
   off-diagonal coupling adding modest improvement on losses that have
   leverage there (`mse`, `mae`, `huber`, `content_aware_l1`,
   `whitened_mse`, `distdf`, `log_power_spectral`). What's striking is
   that the 360k-param transformer + dilated CNN + reconstruction loss
   scores `high_gamma ≈ 1.24`, *worse than constant-mean prediction*.
   The transformer is actively destroying gamma content during its
   forward pass (most plausibly in the tokenizer's conv stack, which
   smooths high frequencies), and the loss doesn't punish that
   destruction because gamma carries a tiny fraction of the time-domain
   power. This is the project's central hypothesis, surfaced directly
   on the laptop batch.

## File map

- `gamma_encoder/models/linear_ar.py` — `LinearVARModel` (with
  `LinearARModel` alias for backwards compatibility)
- `tests/test_linear_ar.py` — shape, causality, parameter count,
  cross-channel coupling, VAR(3) coupling-matrix recovery
- `gamma_encoder/training/overfit.py` — `model_type="linear_ar"`
  branch + checkpoint saving (passes `num_channels=C`)
- `scripts/run_overfit_sweep.py` — `--axis linear_ar` adds the 10
  loss × linear_ar configs
- `scripts/run_band_eval.py` — band-resolved NMSE consumer (rebuilds
  `LinearVARModel` with `num_channels=segments.shape[1]`)
- `scripts/plot_reconstructions.py` — per-run waveform plots
- `results/overfit_runs/<loss>__linear_ar/` — config, metrics,
  checkpoint, loss curve, reconstruction plot
- `results/overfit_runs/band_eval_linear_ar_axis.png` — heatmap
