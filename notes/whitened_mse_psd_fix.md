# Whitened-MSE PSD scaling fix

## The problem

In the 1000-step overfit sweep, `WhitenedMSELoss` reported initial loss
~3.9e9 — eight orders of magnitude larger than every other loss in the
table. Final loss landed at ~2056. Both numbers are nonsense for a
loss meant to be O(1).

## Where it came from

`WhitenedMSELoss` whitens the spectrum by `1/sqrt(PSD)` before
computing MSE, where PSD is the per-frequency-bin power. In the
"no-prior" mode (no PSD passed at construction), it estimates PSD
from the **target batch on each forward call**:

```python
psd = (target_f.abs() ** 2).mean(dim=tuple(range(target_f.dim() - 1)))
```

Two failure modes compound on a 16-segment, 8-channel batch:

1. **Tiny per-batch power for many bins.** With only 128 averaging
   units (16 × 8), high-frequency bins where the true power is small
   produce PSD estimates near machine precision — sometimes well
   below the `eps=1e-8` floor. `1/sqrt(eps)` then dominates and
   amplifies any prediction error in those bins by ~1e4.
2. **The whitened gradient drives `pred_f` toward `target_f` in those
   same exploding bins**, which the model can technically achieve at
   d_model=32 — but the loss-value scale never recovers because it's
   fundamentally a normalization-mismatch artifact, not a fitting
   problem.

The `frac_remaining = 5.3e-7` we saw is therefore meaningful as a
*relative* convergence indicator (loss did decrease monotonically by
nine orders of magnitude) but the absolute values can't be compared
to the other losses, and any cross-loss ranking that includes
whitened_mse is suspect until this is fixed.

## The fix

Estimate PSD **once**, with enough averaging to be stable, and pass
it to the loss at construction so it stays fixed across calls:

```python
# scripts/cache_whitened_mse_psd.py
segments = torch.load("results/overfit_batch.pt")["segments"]   # (16, 8, 6144)
spec = torch.fft.rfft(segments, n=segments.shape[-1], dim=-1)
psd = (spec.real**2 + spec.imag**2).mean(dim=(0, 1))            # (3073,)
torch.save(psd, "results/whitened_mse_psd.pt")
```

Then `gamma_encoder/training/overfit.py:build_loss` auto-loads it:

```python
if name == "whitened_mse":
    psd = None
    if _WHITENED_MSE_PSD_PATH.exists():
        psd = torch.load(_WHITENED_MSE_PSD_PATH, ...)
    return WhitenedMSELoss(psd=psd)
```

Diagnostics from the cached PSD on the overfit batch:

```
shape=(3073,)   min=1.77e-11   median=3.13e+02   max=4.16e+06
```

The dynamic range is still wide (low-frequency bins carry essentially
all the power), but with a fixed estimate every bin gets the same
scaling on every forward pass — so initial loss is set by the actual
mismatch between `pred_f` and `target_f` in whitened space, not by
a per-call rescaling drama.

## What to redo on real data

The cached PSD is computed on 128 averaging units (`16 × 8` from the
overfit batch). Good enough for laptop-stage smoke tests where every
loss sees the same fixed PSD. Before any real-data run on Subject 2,
recompute on a much larger sample — e.g. all of session 1, with
proper Welch averaging over overlapping sub-windows — and save to a
distinct path (`results/whitened_mse_psd_subj2_sess1.pt` or similar).
The current PSD is biased toward whatever the 16 cached segments
happen to contain.

## Status

- Cache script: `scripts/cache_whitened_mse_psd.py`.
- PSD artifact: `results/whitened_mse_psd.pt`.
- Auto-loaded by `gamma_encoder.training.overfit.build_loss`.
- Will be exercised by the 500-step sweep rerun and the band-resolved
  NMSE comparison that follows.
