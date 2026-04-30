# gamma-eval

Evaluation harness for measuring high-frequency reconstruction quality in
brain-state encoder models.

## Why this exists

Existing brain encoders (BaRISTA, BrainBERT, KenazLBM, etc.) all report
reconstruction quality as a single MSE/NMSE number averaged over all
frequencies. This obscures a known failure mode: **MSE-trained models
systematically attenuate high-frequency activity** because of the 1/f
power-law structure of neural data.

This harness provides band-resolved metrics, spectral-distance metrics, and
a connectivity-matrix comparison that together reveal *where in frequency*
a model's reconstruction is failing.

## What's in here

- `gamma_eval.metrics.reconstruction`: stateless metric functions
  (`nmse`, `band_resolved_nmse`, `log_spectral_distance`,
  `envelope_correlation_matrix`, `connectivity_similarity`,
  `evaluate_reconstruction`).
- `gamma_eval.evaluator.ReconstructionEvaluator`: stateful wrapper that
  accumulates metrics across batches, suitable for plugging into a
  training loop.
- `gamma_eval.synthetic.signals`: 1/f noise + oscillatory burst generator
  for testing the harness on controlled inputs.
- `gamma_eval.demo`: end-to-end demo reproducing a BaRISTA-Table-13-style
  failure pattern on synthetic data.

## Quick start

```python
from gamma_eval.evaluator import ReconstructionEvaluator

evaluator = ReconstructionEvaluator(fs=2048.0)

# Inside your validation loop:
metrics = evaluator.evaluate_batch(true_signal, predicted_signal)
print(metrics["nmse_high_gamma_mean"])  # the metric we actually care about
```

Run the demo to see what the metrics look like on synthetic data with
known failure modes:

```
python -m gamma_eval.demo
```

Run the test suite:

```
pytest tests/
```

## Conventions

- All signals are arrays of shape `(n_channels, n_samples)` or, when
  batched, `(n_segments, n_channels, n_samples)`.
- Default frequency bands: delta-theta (1–8 Hz), alpha-beta (8–30 Hz),
  low-gamma (30–50 Hz), high-gamma (50–200 Hz).
- The connectivity metric uses the high-gamma envelope and Pearson
  correlation across channels (Das & Menon 2022 style).
- Sampling rate `fs=2048` matches BrainTreebank's recording rate. Override
  for other datasets.

## What this doesn't do

- Doesn't include a model. Bring your own encoder (BaRISTA, etc.); the
  harness only consumes (true, predicted) tensor pairs.
- Doesn't include classification metrics for downstream tasks. That goes
  in a separate module (TODO).
- Doesn't load BrainTreebank data. That's the next module to build.
