"""Tests for channel-masked reconstruction.

Channel masking hides ``k`` of ``C`` channels from the encoder, so the
encoder must predict them from cross-channel context. Tests verify:

  1. Forward pass shape contract is preserved (still produces (B, C, T)).
  2. Masked channels' inputs do not influence anything in the
     transformer model: changing the masked channel's signal must NOT
     change the recon at any output position. (The encoder never sees
     them.)
  3. A boolean mask of the wrong shape is rejected (transformer).
  4. Loss-only-on-masked path: the harness's masked branch should drive
     loss down on a tiny synthetic batch where masked channels are
     predictable from unmasked ones.

The LinearVAR baseline ignores ``mask_channels`` on purpose — see
``gamma_encoder/models/linear_ar.py``. Zero-filling AR inputs would
corrupt system ID, so the AR runs unmasked and the caller scores
on the masked-channel positions only. That means the AR baseline
includes its own AR-self-history floor, which is the canonical thing
a cross-channel model must beat.
"""

from __future__ import annotations

import torch

from gamma_encoder.models.full_model import GammaEncoderConfig, GammaEncoderModel
from gamma_encoder.models.linear_ar import LinearVARModel
from gamma_encoder.tokenizers.linear import LinearTokenizer
from gamma_encoder.training.overfit import _sample_region_mask


def _build_transformer(C: int, T: int = 1024, patch: int = 256):
    cfg = GammaEncoderConfig(
        d_model=16, n_layers=2, n_heads=2, ff_mult=2,
        patch_samples=patch, num_regions=64, max_seq_len=C * (T // patch) + 8,
    )
    tok = LinearTokenizer(d_model=cfg.d_model, patch_samples=cfg.patch_samples)
    return GammaEncoderModel(tok, cfg)


def test_transformer_mask_preserves_shape():
    torch.manual_seed(0)
    B, C, T = 2, 4, 1024
    model = _build_transformer(C, T)
    x = torch.randn(B, C, T)
    mask = torch.zeros(B, C, dtype=torch.bool)
    mask[:, :2] = True
    region_ids = torch.arange(C)
    y = model(x, region_ids, mask_channels=mask)
    assert y.shape == (B, C, T)


def test_transformer_mask_rejects_wrong_shape():
    torch.manual_seed(0)
    B, C, T = 1, 4, 1024
    model = _build_transformer(C, T)
    x = torch.randn(B, C, T)
    bad = torch.zeros(B, C + 1, dtype=torch.bool)
    region_ids = torch.arange(C)
    try:
        model(x, region_ids, mask_channels=bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong mask shape")


def test_transformer_masked_channel_input_has_no_effect():
    """Perturbing a masked channel's signal must not change recon at all.

    This is the load-bearing property: masked channels are never seen by
    the encoder, so their content can't leak into the prediction.
    """
    torch.manual_seed(0)
    B, C, T = 1, 4, 1024
    model = _build_transformer(C, T)
    model.eval()
    x = torch.randn(B, C, T)
    mask = torch.zeros(B, C, dtype=torch.bool)
    mask[0, 0] = True
    mask[0, 2] = True
    region_ids = torch.arange(C)
    with torch.no_grad():
        y_base = model(x, region_ids, mask_channels=mask)
        x_pert = x.clone()
        x_pert[0, 0] += 5.0  # huge kick on a masked channel
        x_pert[0, 2] -= 3.0
        y_pert = model(x_pert, region_ids, mask_channels=mask)
    delta = (y_base - y_pert).abs().max().item()
    assert delta < 1e-5, f"masked channel content leaked: max |Δrecon|={delta}"


def test_linear_var_ignores_mask_channels():
    """LinearVAR must produce the SAME output regardless of mask_channels.

    The AR baseline runs unmasked on principle (zeroing inputs would
    corrupt cross-channel coefficient fitting). Passing a mask should
    be a no-op at the model level — the caller is responsible for
    scoring on masked positions.
    """
    torch.manual_seed(0)
    B, C, T = 1, 4, 256
    model = LinearVARModel(num_channels=C, order=3)
    with torch.no_grad():
        model.conv.weight.copy_(torch.randn_like(model.conv.weight) * 0.3)
        model.conv.bias.copy_(torch.randn_like(model.conv.bias) * 0.05)
    model.eval()
    x = torch.randn(B, C, T)
    mask = torch.zeros(B, C, dtype=torch.bool)
    mask[0, 1] = True
    with torch.no_grad():
        y_unmasked = model(x)
        y_masked = model(x, mask_channels=mask)
    assert torch.allclose(y_unmasked, y_masked)


def test_transformer_unmasked_channels_still_drive_output():
    """Sanity check: perturbing an UNmasked channel SHOULD change recon."""
    torch.manual_seed(0)
    B, C, T = 1, 4, 1024
    model = _build_transformer(C, T)
    model.eval()
    x = torch.randn(B, C, T)
    mask = torch.zeros(B, C, dtype=torch.bool)
    mask[0, 0] = True
    region_ids = torch.arange(C)
    with torch.no_grad():
        y_base = model(x, region_ids, mask_channels=mask)
        x_pert = x.clone()
        x_pert[0, 1] += 5.0      # unmasked channel
        y_pert = model(x_pert, region_ids, mask_channels=mask)
    delta = (y_base - y_pert).abs().max().item()
    assert delta > 1e-3, f"unmasked channel had no effect: max |Δ|={delta}"


def test_region_mask_groups_channels_by_region():
    """Channels sharing a region must be masked together (or not at all).

    Sampling 1 of 3 regions from [0,0,1,1,2,2] must yield exactly 2
    masked channels per row, and they must be the pair from the same region.
    """
    region_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    g = torch.Generator().manual_seed(0)
    mask = _sample_region_mask(B=8, region_ids=region_ids, k_regions=1,
                               generator=g, device="cpu")
    assert mask.shape == (8, 6)
    for b in range(8):
        m = mask[b]
        # exactly one full region's worth of channels masked
        assert m.sum().item() == 2
        # the two masked channels must share a region
        masked_regions = region_ids[m].tolist()
        assert masked_regions[0] == masked_regions[1]


def test_region_mask_rejects_too_many_regions():
    region_ids = torch.tensor([0, 1, 2])
    g = torch.Generator().manual_seed(0)
    try:
        _sample_region_mask(B=1, region_ids=region_ids, k_regions=3,
                            generator=g, device="cpu")
    except ValueError:
        return
    raise AssertionError("expected ValueError when k_regions >= n_unique")


def test_linear_var_can_learn_to_predict_masked_from_neighbors():
    """End-to-end: with a known cross-coupling structure, MVAR + masked
    loss should drive the masked-channel reconstruction loss down.
    """
    torch.manual_seed(0)
    B, C, T = 4, 4, 256
    # Synthetic data: every channel = ch0 shifted, so masking any
    # channel still leaves a perfect predictor (ch0) in the input.
    base = torch.randn(B, 1, T)
    x = torch.cat([torch.roll(base, shifts=k, dims=-1) for k in range(C)], dim=1)
    model = LinearVARModel(num_channels=C, order=3)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    mask = torch.zeros(B, C, dtype=torch.bool)
    mask[:, 1] = True  # always mask channel 1
    losses = []
    for _ in range(200):
        recon = model(x, mask_channels=mask)
        pred_m = recon[mask].view(B, 1, T)
        true_m = x[mask].view(B, 1, T)
        # Skip first p samples (causal pad creates synthetic zeros).
        loss = ((pred_m[..., 3:] - true_m[..., 3:]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], (
        f"masked-channel loss did not descend: init={losses[0]:.4f} "
        f"final={losses[-1]:.4f}"
    )
