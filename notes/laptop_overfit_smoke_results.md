# Laptop overfit smoke results (2026-04-30)

Pipeline-soundness sweep across the loss and tokenizer axes. **Not** an
ablation — this is overfit-on-fixed-batch convergence-speed checking.
Final loss numbers are not a quality ranking.

## Setup

Cached batch `results/overfit_batch.pt`: 16 segments x 8 channels x 6144
samples from sub_2 trial 0 (region-diverse channels, fs=2048).
Model config: `d_model=32`, `n_layers=6`, `n_heads=2`, `patch_samples=512`,
1000 steps, `lr=1e-3`, `weight_decay=0`, `seed=0`, CPU.

Two axes swept independently:
- **Loss axis** (tokenizer fixed at `dilated_cnn`): mse, mae, huber,
  whitened_mse, log_power_spectral, multires_stft, eegm2, distdf,
  cmin_logcosh, content_aware_l1.
- **Tokenizer axis** (loss fixed at `mse`): dilated_cnn, linear,
  stft_magnitude, complex_stft, wavelet_packet, welch_psd.

`mse + dilated_cnn` is the shared cell.

## Headline findings

- Every (loss, tokenizer) combo descended monotonically. No nan/inf,
  no failure-to-converge.
- Tight cluster reaching ~2-3% of initial loss: mse, distdf, huber,
  complex_stft, stft_magnitude, wavelet_packet. These all behave like
  the pipeline is sound.
- Middle band (10-17% of initial): content_aware_l1, multires_stft,
  mae, eegm2.
- log_power_spectral was the worst at 31%. Its curve plateaued near
  ~1.0 from step ~300 onward with mild step-to-step oscillation
  (final 0.968, min 0.960) — lr=1e-3 looks too hot for this loss.
- whitened_mse hit `frac_remaining = 5.3e-7`, but only because the
  on-the-fly PSD whitening produces an initial loss of ~3.9e9.
  Relative descent here is not directly comparable.
- Tokenizer axis with mse: linear, complex_stft, wavelet_packet, and
  stft_magnitude all reach ~1.5-2.8% of initial — comparable floors
  to dilated_cnn (~2.4%). welch_psd was the outlier at 5.3% — expected,
  since welch_psd drops per-patch time resolution by design.

## Loss axis (tokenizer = dilated_cnn)

Sorted by `frac_remaining`.

| loss | initial | final | min | frac_remaining | seconds |
|------|---------|-------|-----|----------------|---------|
| whitened_mse | 3.879e9 | 2056 | 2056 | 5.30e-7 | 559 |
| mse | 2.898 | 0.07020 | 0.07020 | 2.42% | 578 |
| distdf | 2.814 | 0.06876 | 0.06876 | 2.44% | 1143 |
| huber | 0.9345 | 0.03264 | 0.03264 | 3.49% | 544 |
| cmin_logcosh | 0.8416 | 0.03378 | 0.03276 | 4.01% | 911 |
| content_aware_l1 | 1.583 | 0.1793 | 0.1793 | 11.32% | 571 |
| multires_stft | 12.95 | 1.505 | 1.504 | 11.62% | 627 |
| mae | 1.357 | 0.1740 | 0.1740 | 12.82% | 552 |
| eegm2 | 23.90 | 4.156 | 4.156 | 17.39% | 965 |
| log_power_spectral | 3.090 | 0.9683 | 0.9598 | 31.34% | 556 |

## Tokenizer axis (loss = mse)

Sorted by `frac_remaining`.

| tokenizer | initial | final | min | frac_remaining | n_params | seconds |
|-----------|---------|-------|-----|----------------|----------|---------|
| stft_magnitude | 12.04 | 0.1863 | 0.1863 | 1.55% | 130,048 | 20 |
| complex_stft | 4.134 | 0.06640 | 0.06635 | 1.61% | 164,896 | 21 |
| dilated_cnn | 2.898 | 0.07020 | 0.07020 | 2.42% | 360,544 | 578 |
| wavelet_packet | 2.482 | 0.06866 | 0.06866 | 2.77% | 111,584 | 99 |
| linear | 2.414 | 0.06877 | 0.06782 | 2.85% | 111,584 | 16 |
| welch_psd | 3.058 | 0.1612 | 0.1603 | 5.27% | 96,256 | 21 |

The dilated_cnn run is ~30x slower than linear/stft tokenizers because
of the conv stack on CPU; not a real-data concern.

## Reference number

**`mse + dilated_cnn` final loss = 0.0702** at 1000 steps on the cached
batch. This is the floor every new (loss, tokenizer) combo must beat
on real data, not on this batch.

## Flags for future sessions

- **whitened_mse** needs a fixed PSD estimate from training data (or a
  higher floor on the on-the-fly PSD) before it goes on real data.
  Initial-loss scale of ~4e9 makes any composite or cross-loss
  comparison meaningless. Fix this before wiring it into anything.
- **log_power_spectral** plateaus with oscillation around 1.0 from step
  ~300 — strongly suggests lr=1e-3 is too high. One retry at 3e-4
  before drawing conclusions.
- **distdf** at alpha=0.05 hit 0.0688, essentially identical to mse's
  0.0702. Expected — at alpha=0.05 it's 95% MSE and the BW term has
  little to bite on with a tiny batch and minimal autocorrelation
  variability. Validate distdf vs mse on real data; the smoke test
  cannot distinguish them.
- **eegm2** was the slowest dilated_cnn run at 965s; **distdf** next at
  1143s. Composite spectral term and eigendecomp respectively dominate
  per-step cost. Note for server compute budgeting.
- **content_aware_l1** at 11% sits where the L1-flavored losses sit
  (mae 13%). Magnitude-region reweighting doesn't speed up overfit
  convergence on this batch but isn't broken.

## Artifacts

- Per-run: `results/overfit_runs/<loss>__<tokenizer>/{config.json,
  metrics.jsonl, summary.json, loss_curve.png,
  overfit_<tokenizer>_<loss>.json}`
- Sweep-level: `results/overfit_runs/comparison_loss_axis.png`,
  `results/overfit_runs/comparison_tokenizer_axis.png`,
  `results/overfit_runs/summary_table.csv`,
  `results/overfit_runs/_sweep.log`

## What this does NOT establish

Nothing about real-data performance, gamma-faithful reconstruction, or
downstream task performance. Nothing about which loss is "best" —
overfit-on-tiny-batch convergence speed is a pipeline check, not a
thesis-relevant ranking. Differences in `frac_remaining` mostly reflect
loss-specific scale and curvature on a 16-segment batch, not modeling
quality.

The next decision point (CLAUDE.md "Where to pick up next" item 5) is
running band-resolved NMSE through `gamma_eval`'s pre-filtered eval
path on a held-out segment for a few of these checkpoints. That tells
us whether any of these losses actually move high-gamma NMSE downward
on real data. Until that is in hand, none of this approaches the server.
