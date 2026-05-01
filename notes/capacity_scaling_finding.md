# Capacity scaling finding (2026-05-01)

Short note capturing the laptop-stage experiment that disambiguated
"the masked task is data-limited" from "the laptop default model is too
small." Result: it was the latter. The masked region-prediction
objective is fully learnable on the cached batch given adequate
capacity + training time.

## Why this came up

After switching from channel masking to region masking and rerunning
the 25-config sweep, every reconstruction plot looked flat — the
masked-row predictions were near zero, no visible signal recovery.
Two possible reads:

1. **Data ceiling.** The cached batch has 8 channels each from a
   distinct Destrieux parcel, picked greedily for region diversity.
   With one channel per region, masking a region removes the only
   member of that region, and the remaining channels carry weak
   linear information about it. The masked floor is then capped by
   inter-region mutual information.
2. **Capacity ceiling.** The laptop default `d_model=32, n_layers=6`
   (~360k params for `dilated_cnn`, ~112k for `linear`) and 500
   training steps may simply be too small/short for the model to do
   the conditional memorization the masked objective requires.

These have different implications for the server. (1) says the
laptop batch isn't a useful smoke test for masking and we should
rebuild it with multi-electrode-per-region channels; (2) says the
laptop batch is fine, the dummy config was too small, and we should
run at full BaRISTA size (`d=64, l=12`) from the start on the server.

## Experiments

All on the same cached batch (`results/overfit_batch.pt`,
`(16, 8, 6144)` segments, fs=2048), `mse` loss, AdamW lr=1e-3, mask 2
of 8 regions per segment.

### Disambig 1 — full reconstruction (no mask)

| config | params | steps | unmasked final | min |
|---|---|---|---|---|
| `mse + linear` d=32 l=6 | 111,616 | 500 | **0.085** | 0.085 |
| `mse + dilated_cnn` d=32 l=6 | 360,576 | 500 | **0.098** | 0.097 |

Both small-config models reconstruct the batch almost perfectly with
no masking. So the model can memorize the data when allowed to see it
— optimization is fine, capacity is fine for the unmasked task.

### Disambig 2 — masked region prediction, capacity sweep

| config | params | steps | masked final | min | seconds |
|---|---|---|---|---|---|
| `mse + linear` d=32 l=6 (baseline) | 111,616 | 500 | 0.573 | 0.573 | ~7 |
| `mse + linear` d=64 l=12 | 670,080 | 1,000 | 0.200 | 0.173 | 38 |
| `mse + linear` d=64 l=12 | 670,080 | **10,000** | **0.036** | **0.022** | 384 |

Loss plateaus around step 4-5k at ~0.04 with bounce from the random
per-step mask draw. The 10k-step floor (0.036) is **below** the
small-model *unmasked* floor (0.085) — i.e., the masked task is
fully learnable on this batch.

## Conclusion

The "models look like they aren't learning" pattern in the original
sweep was an under-trained-small-model artifact, not a data-ceiling
artifact. With BaRISTA-default size (`d=64, n_layers=12`) and 10k
steps, the masked-region objective converges cleanly on the cached
8-channel batch even though every channel is from a different parcel.
The data-ceiling intuition (intra-region neighbors are the
load-bearing predictors, masking removes them) is not refuted in
general, but it does not bottleneck this batch.

The original framing in this conversation — "you can't predict a
masked region from inter-region context" — was wrong as a strict
ceiling claim. The model can do conditional memorization given enough
parameters; on real data with thousands of distinct segments it will
also lean on a learned prior over neural signals (1/f shape,
transient morphology, region-specific spectral signatures), which the
single-batch laptop run does not exercise.

## Implications

1. **Server should run at full BaRISTA size from step 0.** No reason
   to use the laptop default `d=32, n_layers=6`. The existing
   `run_overfit` and the future `run_pretrain` already take
   `d_model` / `n_layers` arguments.

2. **Laptop overfit batch is a usable smoke test for masking** as
   long as we're willing to run d=64 l=12 + ≥5k steps. The
   d=32 l=6 + 500-step preset is too cheap to demonstrate masked
   learning; raise the defaults if we're going to keep using the
   sweep as a smoke test.

3. **Existing 25-config region-masked sweep is misleading on the
   ablation question.** All those numbers were collected at d=32,
   l=6, 500 steps, where masked learning hadn't converged. Cross-
   loss / cross-tokenizer comparisons drawn from
   `results/overfit_runs/band_eval.csv` are convergence-speed
   comparisons, not asymptotic-quality comparisons. Real ablation
   numbers come from the server runs at full size.

4. **No need to rebuild `overfit_batch.pt`** with multi-electrode-
   per-region channels just for the masking smoke. The existing
   batch is fine; the only adjustment is to run with bigger model
   defaults when probing masked learning.

## Artifacts

Run directories:

- `results/disambig/mse__linear__unmasked/` — unmasked baseline,
  d=32 l=6, 500 steps
- `results/disambig/mse__dilated_cnn__unmasked/` — same with dilated CNN
- `results/disambig/mse__linear_d64_l12_s1000__masked/` — scaled
  config, 1000 steps
- `results/disambig/mse__linear_d64_l12_s10000__masked/` — scaled
  config, 10k steps (the headline number)
- `results/disambig/_capacity_test.log`,
  `results/disambig/_long_run.log` — full stdout from the runs

Reconstruction plots in each run dir show the visible improvement
from the d=32 l=6 baseline (flat masked rows) to the d=64 l=12 10k
result (masked rows track truth).
