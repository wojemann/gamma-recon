"""Tests for the faithful BaRISTA-style encoder (RoPE + RMSNorm + SwiGLU).

The faithful encoder is the backbone for downstream experiments, so the
goal here is to lock down structural and behavioral contracts that
catch regressions in the RoPE/RMSNorm/SwiGLU stack independent of the
rest of the framework. Three layers of coverage:

  - Module-level: RMSNorm bias-free + fp32 variance, SwiGLU has the
    expected three-projection layout, RoPE makes attention
    position-sensitive.
  - Model-level: forward+backward at laptop dims and at BaRISTA dims
    produces finite outputs and grads to all parameters.
  - Cross-reference: structural shape parity vs the BaRISTA
    pretrained checkpoint (gates on the ckpt being mounted).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from gamma_encoder.models.faithful import (
    FaithfulTransformerEncoder,
    GatedTransformerMLP,
    RMSNorm,
    RotarySelfAttention,
)
from gamma_encoder.models.full_model import GammaEncoderConfig, GammaEncoderModel
from gamma_encoder.tokenizers.dilated_cnn import DilatedCNNTokenizer


BARISTA_CKPT = Path("/Users/wojemann/local_data/BaRISTA/pretrained_models/parcels_chans.ckpt")


# ---------------------------------------------------------------------------
# Module-level contracts
# ---------------------------------------------------------------------------


def test_rmsnorm_is_bias_free_and_unit_scale_init():
    norm = RMSNorm(d_hidden=16)
    # No bias parameter; only `weight`.
    names = [n for n, _ in norm.named_parameters()]
    assert names == ["weight"]
    assert torch.allclose(norm.weight, torch.ones(16))


def test_rmsnorm_normalizes_to_unit_rms():
    torch.manual_seed(0)
    norm = RMSNorm(d_hidden=32, eps=1e-6)
    x = torch.randn(4, 8, 32) * 7.5  # arbitrary scale
    y = norm(x)
    # Per-row RMS along the feature dim should be ~1 (weight init = 1).
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_swiglu_has_three_projections():
    """SwiGLU = silu(gate) * up; down. Three Linears, not two."""
    mlp = GatedTransformerMLP(d_model=32, mlp_ratio=4)
    assert isinstance(mlp.gate_proj, nn.Linear)
    assert isinstance(mlp.up_proj, nn.Linear)
    assert isinstance(mlp.down_proj, nn.Linear)
    assert mlp.gate_proj.out_features == 4 * 32
    assert mlp.up_proj.out_features == 4 * 32
    assert mlp.down_proj.in_features == 4 * 32


def test_rope_makes_attention_position_sensitive():
    """Permuting tokens changes the attention output (RoPE is positional)."""
    torch.manual_seed(0)
    attn = RotarySelfAttention(d_model=32, n_heads=2)
    x = torch.randn(1, 6, 32)
    perm = torch.tensor([5, 4, 3, 2, 1, 0])
    y = attn(x)
    y_perm = attn(x[:, perm, :])
    # If RoPE were absent, y[:, perm, :] would equal y_perm (permutation-equivariant).
    assert not torch.allclose(y[:, perm, :], y_perm, atol=1e-5)


def test_faithful_stack_gradient_flows_to_all_params():
    """Every parameter in the stack should receive a non-None grad."""
    torch.manual_seed(0)
    enc = FaithfulTransformerEncoder(d_model=16, n_layers=2, n_heads=2,
                                      mlp_ratio=4, max_seq_len=64)
    x = torch.randn(2, 12, 16, requires_grad=False)
    y = enc(x)
    y.sum().backward()
    missing = [n for n, p in enc.named_parameters() if p.grad is None]
    assert missing == [], f"parameters with no grad: {missing}"


# ---------------------------------------------------------------------------
# Model-level: end-to-end forward + backward at laptop and BaRISTA dims
# ---------------------------------------------------------------------------


def _build_model(d_model, n_layers, n_heads, C=4, T=6144, seed=0):
    torch.manual_seed(seed)
    cfg = GammaEncoderConfig(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, ff_mult=4,
        patch_samples=512, num_regions=64,
        max_seq_len=C * (T // 512) + 16,
        encoder_kind="faithful",
    )
    tok = DilatedCNNTokenizer(d_model=d_model, patch_samples=512)
    return GammaEncoderModel(tok, cfg), cfg


@pytest.mark.parametrize(
    "label,d_model,n_layers,n_heads",
    [
        ("laptop", 32, 6, 2),
        ("barista", 64, 12, 4),
    ],
)
def test_faithful_forward_backward(label, d_model, n_layers, n_heads):
    B, C, T = 2, 4, 6144
    torch.manual_seed(1)
    segments = torch.randn(B, C, T)
    region_ids = torch.arange(C, dtype=torch.long)

    model, _ = _build_model(d_model, n_layers, n_heads, C=C, T=T)
    model.train()
    recon = model(segments, region_ids)
    assert recon.shape == segments.shape
    assert torch.isfinite(recon).all().item()

    loss = ((recon - segments) ** 2).mean()
    assert torch.isfinite(loss).item()
    loss.backward()
    grad_max = max(p.grad.abs().max().item()
                   for p in model.parameters() if p.grad is not None)
    assert grad_max > 0 and torch.isfinite(torch.tensor(grad_max))


# ---------------------------------------------------------------------------
# Structural parity vs the BaRISTA pretrained checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not BARISTA_CKPT.exists(),
    reason="BaRISTA pretrained checkpoint not mounted",
)
def test_barista_dim_model_has_high_shape_parity_with_ckpt():
    """Multiset of our state_dict shapes should largely match the ckpt's.

    Catches regressions where the encoder stack drifts from BaRISTA's
    layout (e.g., losing the gate_proj, swapping LayerNorm in for
    RMSNorm, changing head count). We don't expect 100% — our spatial
    encoder, decoder, and tokenizer aren't shared with BaRISTA — but
    the bulk of transformer-block tensors should line up.
    """
    model, _ = _build_model(d_model=64, n_layers=12, n_heads=4)
    ckpt = torch.load(BARISTA_CKPT, map_location="cpu", weights_only=True)

    # Multiset match: each ckpt-shape can absorb at most as many of our
    # tensors as it appears in the ckpt itself.
    avail: dict[tuple, int] = {}
    for v in ckpt.values():
        s = tuple(v.shape)
        avail[s] = avail.get(s, 0) + 1
    n_our = 0
    matched = 0
    for v in model.state_dict().values():
        s = tuple(v.shape)
        n_our += 1
        if avail.get(s, 0) > 0:
            avail[s] -= 1
            matched += 1
    parity = matched / max(n_our, 1)
    # Loose threshold: regression alarm, not exact-match. Measured value
    # at the time the faithful encoder landed was 58.2% (121/208).
    assert parity >= 0.5, f"shape parity dropped to {parity:.1%} ({matched}/{n_our})"
