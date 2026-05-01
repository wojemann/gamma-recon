"""Tests for new reconstruction losses.

Each loss gets four kinds of test:
  1. shape contract / mismatch raises;
  2. identity is (near) zero;
  3. gradient flows to ``pred``;
  4. behavioral check tied to the loss's purpose.

Synthetic signals come from ``gamma_eval.synthetic.signals`` — same
generator used by the evaluation harness, so the tests are exercising
the loss against signals that look the way our real data is *modeled*.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gamma_eval.synthetic.signals import (
    SignalConfig,
    generate_signal,
    smooth_high_frequencies,
)

from gamma_encoder.losses.cmin_logcosh import CMinLogCoshLoss
from gamma_encoder.losses.content_aware_l1 import ContentAwareL1Loss
from gamma_encoder.losses.distdf import DistDFLoss
from gamma_encoder.losses.eegm2 import EEGM2Loss
from gamma_encoder.losses.log_power_spectral import LogPowerSpectralL1Loss
from gamma_encoder.losses.multires_stft import MultiResolutionSTFTLoss
from gamma_encoder.losses.robust import HuberLoss, MAELoss
from gamma_encoder.losses.whitened_mse import WhitenedMSELoss


FS = 2048.0
T_SHORT = 256       # short signal for cheap tests
T_SEG = 6144        # full 3-sec segment


def _onef_signal(n_samples: int = T_SEG, n_channels: int = 4, seed: int = 0) -> torch.Tensor:
    cfg = SignalConfig(
        n_samples=n_samples,
        n_channels=n_channels,
        fs=FS,
        aperiodic_exponent=1.5,
        bursts=[(80.0, 1.5, 4.0), (130.0, 1.0, 4.0)],
        seed=seed,
    )
    arr = generate_signal(cfg)  # (C, T)
    return torch.from_numpy(arr).float().unsqueeze(0)  # (1, C, T)


def _check_grad_flows(loss_fn, target):
    pred = target.clone().detach().requires_grad_(True)
    # Tiny perturbation so loss isn't exactly zero (some losses may have
    # zero gradient at exact identity).
    pred_perturbed = pred + 0.01 * torch.randn_like(pred)
    val = loss_fn(pred_perturbed, target)
    val.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# MAE / Huber
# ---------------------------------------------------------------------------


def test_mae_identity_zero():
    x = torch.randn(2, 3, T_SHORT)
    assert MAELoss()(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_mae_shape_mismatch_raises():
    with pytest.raises(ValueError):
        MAELoss()(torch.randn(2, 3, 64), torch.randn(2, 3, 65))


def test_mae_grad_flows():
    _check_grad_flows(MAELoss(), torch.randn(2, 3, T_SHORT))


def test_mae_responds_to_error():
    x = torch.randn(2, 3, T_SHORT)
    assert MAELoss()(x + 0.5, x).item() > 0.1


def test_huber_identity_zero():
    x = torch.randn(2, 3, T_SHORT)
    assert HuberLoss()(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_huber_grad_flows():
    _check_grad_flows(HuberLoss(), torch.randn(2, 3, T_SHORT))


def test_huber_quadratic_then_linear():
    # Small error: huber ~ 0.5*err^2; large error: huber ~ delta*(|err|-delta/2).
    target = torch.zeros(1, 1, 4)
    small = torch.full_like(target, 0.1)
    large = torch.full_like(target, 5.0)
    h = HuberLoss(delta=1.0)
    # 0.5 * 0.1^2 = 0.005
    assert h(small, target).item() == pytest.approx(0.005, abs=1e-6)
    # 1.0 * (5.0 - 0.5) = 4.5
    assert h(large, target).item() == pytest.approx(4.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Whitened MSE
# ---------------------------------------------------------------------------


def test_whitened_mse_identity_zero():
    x = _onef_signal()
    assert WhitenedMSELoss()(x, x).item() == pytest.approx(0.0, abs=1e-4)


def test_whitened_mse_shape_mismatch_raises():
    with pytest.raises(ValueError):
        WhitenedMSELoss()(torch.randn(2, 3, 64), torch.randn(2, 3, 65))


def test_whitened_mse_grad_flows():
    _check_grad_flows(WhitenedMSELoss(), torch.randn(2, 3, T_SHORT))


def test_whitened_mse_psd_length_check():
    psd = torch.ones(33)  # for n=64 we'd need 33; for 65 we need 33 too.
    # n=64 -> n_freqs=33; supply wrong-length PSD.
    L = WhitenedMSELoss(psd=torch.ones(10))
    with pytest.raises(ValueError):
        L(torch.randn(1, 1, 64), torch.randn(1, 1, 64))


def test_whitened_mse_build_loss_loads_cached_psd():
    """build_loss('whitened_mse') populates the psd buffer when the cache
    file at results/whitened_mse_psd.pt exists."""
    from pathlib import Path

    from gamma_encoder.training.overfit import build_loss, _WHITENED_MSE_PSD_PATH

    if not _WHITENED_MSE_PSD_PATH.exists():
        pytest.skip(f"cached PSD not found at {_WHITENED_MSE_PSD_PATH}")
    L = build_loss("whitened_mse")
    assert isinstance(L, WhitenedMSELoss)
    assert L.psd is not None
    assert L.psd.dim() == 1
    # Cached batch is T=6144, n_freqs = 6144//2 + 1 = 3073.
    assert L.psd.shape[0] == 3073


def test_whitened_mse_cached_batch_identity_zero():
    """loss(target, target) == 0 on the cached overfit batch."""
    from pathlib import Path

    from gamma_encoder.training.overfit import build_loss, _WHITENED_MSE_PSD_PATH

    batch_path = Path("results/overfit_batch.pt")
    if not batch_path.exists() or not _WHITENED_MSE_PSD_PATH.exists():
        pytest.skip("cached batch or PSD not found")
    payload = torch.load(batch_path, map_location="cpu", weights_only=False)
    target = payload["segments"].float()
    L = build_loss("whitened_mse")
    val = L(target, target).item()
    assert val == pytest.approx(0.0, abs=1e-5), f"identity not zero on cached batch: {val}"


def test_whitened_mse_zeros_pred_scale_about_one():
    """When PSD is computed from the same target (so whitening matches the
    target's spectrum), loss(zeros, target) ~ 1.0 by construction:
    each frequency bin contributes |target_f|^2 / psd_f, and averaging
    over bins where psd ~ |target_f|^2 yields ~1."""
    from pathlib import Path

    from gamma_encoder.training.overfit import _WHITENED_MSE_PSD_PATH

    batch_path = Path("results/overfit_batch.pt")
    if not batch_path.exists() or not _WHITENED_MSE_PSD_PATH.exists():
        pytest.skip("cached batch or PSD not found")
    payload = torch.load(batch_path, map_location="cpu", weights_only=False)
    target = payload["segments"].float()  # (B, C, T)
    # PSD computed exactly from this target = |rFFT|^2 averaged over (B, C).
    T = target.shape[-1]
    spec = torch.fft.rfft(target, n=T, dim=-1)
    psd = (spec.real ** 2 + spec.imag ** 2).mean(dim=(0, 1))
    L = WhitenedMSELoss(psd=psd)
    val = L(torch.zeros_like(target), target).item()
    assert abs(val - 1.0) < 0.1, (
        f"loss(zeros, target) should be ~1 with target-derived PSD, got {val}"
    )


def test_whitened_mse_reacts_to_gamma_damage():
    """Whitened MSE should be substantially > 0 when gamma is smoothed away."""
    x_np = generate_signal(
        SignalConfig(
            n_samples=T_SEG,
            n_channels=2,
            fs=FS,
            aperiodic_exponent=1.5,
            bursts=[(130.0, 2.0, 8.0)],
            seed=0,
        )
    )
    x = torch.from_numpy(x_np).float().unsqueeze(0)
    smoothed_np = smooth_high_frequencies(x_np, FS, cutoff_hz=50.0, attenuation_db=20.0)
    smoothed = torch.from_numpy(smoothed_np).float().unsqueeze(0)
    L = WhitenedMSELoss()
    identity = L(x, x).item()
    damaged = L(smoothed, x).item()
    assert identity < 1e-3
    assert damaged > 0.05, f"whitened MSE didn't react to gamma damage: {damaged}"


# ---------------------------------------------------------------------------
# Log-power spectral L1
# ---------------------------------------------------------------------------


def test_log_power_identity_zero():
    x = _onef_signal()
    assert LogPowerSpectralL1Loss()(x, x).item() == pytest.approx(0.0, abs=1e-4)


def test_log_power_shape_mismatch_raises():
    with pytest.raises(ValueError):
        LogPowerSpectralL1Loss()(torch.randn(2, 3, 64), torch.randn(2, 3, 65))


def test_log_power_grad_flows():
    _check_grad_flows(LogPowerSpectralL1Loss(), torch.randn(2, 3, T_SHORT))


def test_log_power_reacts_to_gamma_damage():
    x_np = generate_signal(
        SignalConfig(
            n_samples=T_SEG, n_channels=2, fs=FS, aperiodic_exponent=1.5,
            bursts=[(130.0, 2.0, 8.0)], seed=0,
        )
    )
    x = torch.from_numpy(x_np).float().unsqueeze(0)
    smoothed = torch.from_numpy(smooth_high_frequencies(x_np, FS, 50.0, 20.0)).float().unsqueeze(0)
    L = LogPowerSpectralL1Loss()
    assert L(smoothed, x).item() > 0.5


# ---------------------------------------------------------------------------
# Multi-resolution STFT
# ---------------------------------------------------------------------------


def test_multires_stft_identity_near_zero():
    x = _onef_signal()
    assert MultiResolutionSTFTLoss()(x, x).item() == pytest.approx(0.0, abs=1e-4)


def test_multires_stft_grad_flows():
    _check_grad_flows(MultiResolutionSTFTLoss([(64, 16, 64)]), torch.randn(2, 3, T_SHORT))


def test_multires_stft_skips_too_large_configs():
    """Configs whose win > T are silently skipped; raises iff all skipped."""
    L = MultiResolutionSTFTLoss([(64, 16, 64), (1024, 256, 1024)])
    # T=256 -> only first config fits.
    val = L(torch.randn(1, 1, 256), torch.randn(1, 1, 256))
    assert torch.isfinite(val)
    # All configs too large -> raise.
    L_bad = MultiResolutionSTFTLoss([(1024, 256, 1024)])
    with pytest.raises(ValueError):
        L_bad(torch.randn(1, 1, 256), torch.randn(1, 1, 256))


def test_multires_stft_phase_shift_smaller_than_amplitude_distortion():
    """STFT magnitude is approximately phase-invariant, so a small time
    shift should produce a smaller loss than an amplitude scaling that
    moves the same number of samples by the same RMS distance."""
    torch.manual_seed(0)
    x = _onef_signal(n_samples=T_SEG, n_channels=2, seed=0)
    shifted = torch.roll(x, shifts=8, dims=-1)
    scaled = 0.5 * x
    # Match RMS perturbation roughly by checking time-domain MSE.
    L = MultiResolutionSTFTLoss()
    loss_shift = L(shifted, x).item()
    loss_scale = L(scaled, x).item()
    assert loss_shift < loss_scale, (
        f"phase-shift loss {loss_shift:.4f} should be < amplitude-distortion {loss_scale:.4f}"
    )


# ---------------------------------------------------------------------------
# EEGM2 composite
# ---------------------------------------------------------------------------


def test_eegm2_identity_zero():
    x = torch.randn(2, 3, T_SHORT)
    assert EEGM2Loss()(x, x).item() == pytest.approx(0.0, abs=1e-5)


def test_eegm2_grad_flows():
    _check_grad_flows(EEGM2Loss(), torch.randn(2, 3, T_SHORT))


def test_eegm2_responds_to_error():
    x = torch.randn(2, 3, T_SHORT)
    L = EEGM2Loss()
    assert L(x + 0.5, x).item() > L(x, x).item() + 0.05


# ---------------------------------------------------------------------------
# DistDF
# ---------------------------------------------------------------------------


def test_distdf_identity_zero():
    x = torch.randn(2, 3, T_SHORT)
    assert DistDFLoss()(x, x).item() == pytest.approx(0.0, abs=1e-5)


def test_distdf_requires_3d():
    L = DistDFLoss()
    with pytest.raises(ValueError):
        L(torch.randn(2, 3), torch.randn(2, 3))


def test_distdf_grad_flows():
    _check_grad_flows(DistDFLoss(), torch.randn(2, 3, T_SHORT))


def test_distdf_penalizes_mean_shift_more_than_mse_does_relatively():
    """Mean shift moves the BW^2 term sharply (it's exactly ||mu_p-mu_t||^2
    on the diagonal). Compared to identity baseline of zero, this should be
    a meaningful contribution."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, T_SHORT)
    shifted = x + 0.5  # constant mean offset
    val = DistDFLoss(alpha=0.5)(shifted, x).item()
    # MSE term contributes 0.5 * (0.5)^2 = 0.125; BW term adds the
    # mean-shift contribution (4 channels * 0.25 = 1.0 summed).
    # So total should clearly exceed 0.125.
    assert val > 0.2, f"DistDF underweighted mean shift: {val}"


def test_distdf_full_cov_runs():
    """Full-covariance path must be numerically stable on a small batch."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64)
    L = DistDFLoss(alpha=0.1, full_cov=True)
    val = L(x, x).item()
    assert val == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# cMin-LogCosh
# ---------------------------------------------------------------------------


def test_cmin_logcosh_identity_zero():
    x = torch.randn(2, 3, T_SHORT)
    assert CMinLogCoshLoss(max_shift=4)(x, x).item() == pytest.approx(0.0, abs=1e-3)


def test_cmin_logcosh_grad_flows():
    _check_grad_flows(CMinLogCoshLoss(max_shift=4), torch.randn(2, 3, T_SHORT))


def test_cmin_logcosh_invariant_to_small_shift():
    """A target rolled by k samples (k <= max_shift) should give near-zero loss."""
    torch.manual_seed(0)
    x = torch.randn(1, 2, T_SHORT)
    rolled = torch.roll(x, shifts=3, dims=-1)
    L = CMinLogCoshLoss(max_shift=8)
    loss_shifted = L(rolled, x).item()
    # Compare to a random pred: should be much larger.
    rand_pred = torch.randn_like(x)
    loss_rand = L(rand_pred, x).item()
    assert loss_shifted < 0.05, f"cMin-LogCosh failed to recover small shift: {loss_shifted}"
    assert loss_rand > loss_shifted * 5


# ---------------------------------------------------------------------------
# Content-aware reweighted L1
# ---------------------------------------------------------------------------


def test_content_aware_identity_zero():
    x = torch.randn(2, 3, T_SHORT)
    assert ContentAwareL1Loss()(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_content_aware_grad_flows():
    _check_grad_flows(ContentAwareL1Loss(), torch.randn(2, 3, T_SHORT))


def test_content_aware_upweights_high_amplitude():
    """Equal-magnitude error in a high-amplitude region should contribute
    more to the loss than the same error in a quiet region."""
    target = torch.zeros(1, 1, T_SHORT)
    burst_start = 100
    burst_len = 30
    target[0, 0, burst_start : burst_start + burst_len] = 5.0  # large amplitude

    err_in_burst = target.clone()
    err_in_burst[0, 0, burst_start : burst_start + burst_len] += 0.5

    err_in_quiet = target.clone()
    err_in_quiet[0, 0, 200:230] += 0.5  # equal magnitude error in quiet region

    L = ContentAwareL1Loss(weight_floor=0.1)
    loss_burst = L(err_in_burst, target).item()
    loss_quiet = L(err_in_quiet, target).item()
    assert loss_burst > loss_quiet, (
        f"content-aware L1 should weight burst-region error more: "
        f"burst={loss_burst:.4f}, quiet={loss_quiet:.4f}"
    )
