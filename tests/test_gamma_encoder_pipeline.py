"""Forward-pass + overfit smoke tests for the gamma_encoder pipeline.

Synthetic random tensors only — no real BrainTreebank data — so the
suite stays fast and offline.
"""

from __future__ import annotations

import torch

from gamma_encoder.losses.mse import MSELoss
from gamma_encoder.models.full_model import GammaEncoderConfig, GammaEncoderModel
from gamma_encoder.tokenizers.base import patchify, unpatchify
from gamma_encoder.tokenizers.dilated_cnn import DilatedCNNTokenizer


def _make_model(d_model=16, patch_samples=64, n_layers=2, num_regions=8):
    cfg = GammaEncoderConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=2,
        patch_samples=patch_samples,
        num_regions=num_regions,
        max_seq_len=512,
    )
    tok = DilatedCNNTokenizer(d_model=d_model, patch_samples=patch_samples, n_blocks=3, hidden=8)
    return GammaEncoderModel(tok, cfg), cfg


def test_patchify_roundtrip():
    sig = torch.randn(2, 3, 256)
    p = patchify(sig, patch_samples=64)
    assert p.shape == (2, 3, 4, 64)
    back = unpatchify(p)
    assert torch.equal(back, sig)


def test_dilated_cnn_tokenizer_shape():
    tok = DilatedCNNTokenizer(d_model=16, patch_samples=64, n_blocks=3, hidden=8)
    x = torch.randn(2, 3, 4, 64)
    out = tok(x)
    assert out.shape == (2, 3, 4, 16)


def test_full_model_forward_shape():
    model, cfg = _make_model()
    B, C = 2, 3
    T = cfg.patch_samples * 4
    sig = torch.randn(B, C, T)
    region_ids = torch.randint(0, cfg.num_regions, (C,))
    out = model(sig, region_ids)
    assert out.shape == sig.shape


def test_full_model_grad_flows():
    model, cfg = _make_model()
    B, C = 2, 3
    T = cfg.patch_samples * 4
    sig = torch.randn(B, C, T, requires_grad=False)
    region_ids = torch.randint(0, cfg.num_regions, (C,))
    out = model(sig, region_ids)
    loss = MSELoss()(out, sig)
    loss.backward()
    has_grad = [p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()]
    assert all(has_grad), "some params received no gradient"


def test_model_can_overfit_one_segment():
    """The smallest sanity check: with batch=1, a couple hundred steps
    of AdamW must drive MSE meaningfully below its initial value."""
    torch.manual_seed(0)
    model, cfg = _make_model(d_model=32, patch_samples=64, n_layers=2)
    sig = torch.randn(1, 2, cfg.patch_samples * 4)
    region_ids = torch.randint(0, cfg.num_regions, (2,))
    loss_fn = MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    init = loss_fn(model(sig, region_ids), sig).item()
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(sig, region_ids), sig)
        loss.backward()
        opt.step()
    final = loss.item()
    assert final < 0.25 * init, f"failed to overfit single segment: {init:.4f} -> {final:.4f}"
