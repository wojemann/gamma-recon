"""Tests for the LinearVAR (multivariate AR) baseline model.

The model is a single causal vector-AR(p) predictor with full
cross-channel mixing — one ``nn.Conv1d(C, C, kernel_size=p)`` applied
to a left-padded input.

Tests verify:
  1. shape contract over multiple (B, C, T);
  2. parameter count == C*C*p + C;
  3. strict causality — perturbing input at index k cannot change recon
     at indices <= k (even with cross-channel coupling);
  4. gradient flow to conv weight and bias;
  5. MVAR-fit behavior — given a synthetic 2-channel VAR(3) process
     with known cross-coupling, learned weights match ground truth;
  6. region_ids=None is accepted (interface compatibility).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gamma_encoder.models.linear_ar import LinearARModel, LinearVARModel


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B,C,T", [(1, 1, 16), (2, 4, 64), (3, 8, 512), (1, 16, 6144)])
def test_linear_ar_shape_contract(B, C, T):
    model = LinearVARModel(num_channels=C, order=3)
    x = torch.randn(B, C, T)
    y = model(x)
    assert y.shape == (B, C, T)


def test_linear_ar_rejects_non_3d():
    model = LinearVARModel(num_channels=4, order=3)
    with pytest.raises(ValueError):
        model(torch.randn(4, 16))


def test_linear_ar_rejects_channel_mismatch():
    model = LinearVARModel(num_channels=4, order=3)
    with pytest.raises(ValueError):
        model(torch.randn(1, 5, 64))  # C=5 != configured 4


def test_linear_ar_invalid_order():
    with pytest.raises(ValueError):
        LinearVARModel(num_channels=2, order=0)


def test_linear_ar_invalid_num_channels():
    with pytest.raises(ValueError):
        LinearVARModel(num_channels=0, order=3)


def test_linear_ar_alias():
    """LinearARModel is kept as a backwards-compatible alias."""
    assert LinearARModel is LinearVARModel


# ---------------------------------------------------------------------------
# Parameter count: C^2 * p + C
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("C,order", [(1, 3), (4, 3), (8, 3), (8, 5), (16, 3)])
def test_linear_ar_param_count(C, order):
    model = LinearVARModel(num_channels=C, order=order)
    n = sum(p.numel() for p in model.parameters())
    expected = C * C * order + C
    assert n == expected, f"C={C} order={order}: expected {expected}, got {n}"


# ---------------------------------------------------------------------------
# Strict causality
# ---------------------------------------------------------------------------


def test_linear_ar_strict_causality():
    """Perturbing signal[..., k] must not change recon[..., :k+1].

    For VAR(p): y[c, t] = b[c] + Σ_{c', j=1..p} W[c, c', j] x[c', t-j].
    recon at time t depends only on x[..., < t], so changing x at
    index k can only affect recon[..., k+1:]. Holds across channels
    even with full cross-coupling.
    """
    torch.manual_seed(0)
    C = 3
    model = LinearVARModel(num_channels=C, order=3)
    # Randomize weights so all taps & cross-channel terms are non-zero,
    # otherwise causality could be trivially satisfied by zeros.
    with torch.no_grad():
        model.conv.weight.copy_(torch.randn_like(model.conv.weight) * 0.3)
        model.conv.bias.copy_(torch.randn_like(model.conv.bias) * 0.05)

    B, T = 2, 64
    x = torch.randn(B, C, T)
    base = model(x).detach()

    for k in [0, 1, 5, 30, T - 2, T - 1]:
        x_perturbed = x.clone()
        x_perturbed[:, :, k] += 7.5  # large kick on all channels at index k
        out = model(x_perturbed).detach()
        unchanged = (out[:, :, : k + 1] - base[:, :, : k + 1]).abs().max().item()
        assert unchanged < 1e-6, (
            f"causality violated at k={k}: max delta in recon[:, :, :{k+1}]={unchanged}"
        )
        if k + 1 < T:
            changed = (out[:, :, k + 1 :] - base[:, :, k + 1 :]).abs().max().item()
            assert changed > 1e-6, f"perturbation at k={k} had no downstream effect"


def test_linear_ar_cross_channel_coupling_works():
    """Perturbing channel c can affect channel c' != c at later times."""
    torch.manual_seed(0)
    C = 3
    model = LinearVARModel(num_channels=C, order=3)
    with torch.no_grad():
        # Build a coupling matrix where channel 0 strongly drives channel 2.
        model.conv.weight.zero_()
        # Self lag-1 identity (so the model is non-trivial elsewhere).
        model.conv.weight[:, :, 2] = torch.eye(C)
        # Cross term: channel 2 absorbs channel 0 at lag 1.
        model.conv.weight[2, 0, 2] = 0.7
        model.conv.bias.zero_()

    x = torch.randn(1, C, 32)
    base = model(x).detach()
    x_pert = x.clone()
    x_pert[0, 0, 10] += 5.0          # poke only channel 0 at t=10
    out = model(x_pert).detach()

    # Channel 2 at t=11 must respond (cross-coupling), strictly nothing earlier.
    assert (out[0, 2, :11] - base[0, 2, :11]).abs().max() < 1e-6
    assert (out[0, 2, 11] - base[0, 2, 11]).abs() > 0.5


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_linear_ar_grad_flows_to_weight_and_bias():
    model = LinearVARModel(num_channels=4, order=3)
    x = torch.randn(2, 4, 128)
    y = model(x)
    target = torch.randn_like(y)
    loss = ((y - target) ** 2).mean()
    loss.backward()
    assert model.conv.weight.grad is not None
    assert model.conv.bias.grad is not None
    assert model.conv.weight.grad.abs().sum() > 0
    assert model.conv.bias.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# region_ids interface compatibility
# ---------------------------------------------------------------------------


def test_linear_ar_region_ids_none_works():
    model = LinearVARModel(num_channels=2, order=3)
    x = torch.randn(1, 2, 64)
    y_default = model(x)
    y_explicit_none = model(x, region_ids=None)
    assert torch.allclose(y_default, y_explicit_none)


def test_linear_ar_region_ids_ignored():
    model = LinearVARModel(num_channels=4, order=3)
    x = torch.randn(1, 4, 64)
    y_a = model(x, region_ids=torch.zeros(4, dtype=torch.long))
    y_b = model(x, region_ids=torch.arange(4, dtype=torch.long))
    assert torch.allclose(y_a, y_b)


# ---------------------------------------------------------------------------
# MVAR-fit behavior
# ---------------------------------------------------------------------------


def _generate_var2_p3(
    n_samples: int,
    A1: np.ndarray,
    A2: np.ndarray,
    A3: np.ndarray,
    sigma: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Stable 2-channel VAR(3): x[t] = A1 x[t-1] + A2 x[t-2] + A3 x[t-3] + ε.

    Returns array of shape (2, n_samples).
    """
    rng = np.random.default_rng(seed)
    burn = 200
    total = n_samples + burn
    x = np.zeros((2, total), dtype=np.float64)
    eps = rng.normal(0.0, sigma, size=x.shape)
    for t in range(3, total):
        x[:, t] = A1 @ x[:, t - 1] + A2 @ x[:, t - 2] + A3 @ x[:, t - 3] + eps[:, t]
    return x[:, burn:].astype(np.float32)


def test_linear_var_recovers_known_coupling():
    """Train ~400 steps on a synthetic VAR(3) with known cross-channel
    coupling. Learned conv kernel should approximate ground truth.

    We generate 2 channels and the model fits the entire VAR matrix at
    each lag, including off-diagonal coupling terms.
    """
    torch.manual_seed(0)
    A1 = np.array([[0.5, 0.10], [0.05, 0.4]])
    A2 = np.array([[-0.2, 0.0], [0.0, -0.2]])
    A3 = np.array([[0.05, 0.0], [0.0, 0.05]])
    sig = _generate_var2_p3(4096, A1, A2, A3, sigma=0.1, seed=0)
    x = torch.from_numpy(sig).float().unsqueeze(0)  # (1, 2, 4096)

    model = LinearVARModel(num_channels=2, order=3)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    for _ in range(400):
        recon = model(x)
        # Predict next sample: skip the first p where the causal pad
        # creates synthetic zeros that bias the fit.
        loss = ((recon[..., 3:] - x[..., 3:]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # conv.weight has shape (C_out, C_in, K=order). Conv index p-k
    # corresponds to AR lag k.
    W = model.conv.weight.detach().numpy()  # (2, 2, 3)
    learned_A1 = W[:, :, 2]
    learned_A2 = W[:, :, 1]
    learned_A3 = W[:, :, 0]
    bias = model.conv.bias.detach().numpy()

    assert np.allclose(learned_A1, A1, atol=0.07), f"A1 off:\n{learned_A1}\nvs\n{A1}"
    assert np.allclose(learned_A2, A2, atol=0.07), f"A2 off:\n{learned_A2}\nvs\n{A2}"
    assert np.allclose(learned_A3, A3, atol=0.07), f"A3 off:\n{learned_A3}\nvs\n{A3}"
    assert np.max(np.abs(bias)) < 0.05, f"bias should be ~0, got {bias}"
