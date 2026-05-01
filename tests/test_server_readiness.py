"""Server-only smoke tests.

These exist so you can run ``pytest tests/test_server_readiness.py -v``
on the server *before* kicking off the full Subject 2 ablation. Each
test either:

  - skips on the laptop (CUDA-only), or
  - skips until the corresponding server-only module exists (the
    streaming dataset, the pretrain entrypoint), or
  - runs everywhere but documents a numerical-robustness contract that
    matters once we leave the laptop batch behind (PSD floor, distdf
    grad clipping).

The point is to catch GPU/AMP/DDP regressions and "I forgot to add the
PSD floor" before they cost a multi-hour training run, not to
exhaustively test the server stack.
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

from gamma_encoder.losses.distdf import DistDFLoss
from gamma_encoder.losses.whitened_mse import WhitenedMSELoss
from gamma_encoder.models.full_model import GammaEncoderConfig, GammaEncoderModel
from gamma_encoder.models.linear_ar import LinearVARModel
from gamma_encoder.tokenizers.linear import LinearTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_small_transformer(C: int, T: int, patch: int = 256, d_model: int = 32):
    cfg = GammaEncoderConfig(
        d_model=d_model, n_layers=2, n_heads=2, ff_mult=2,
        patch_samples=patch, num_regions=64,
        max_seq_len=C * (T // patch) + 8,
    )
    tok = LinearTokenizer(d_model=d_model, patch_samples=patch)
    return GammaEncoderModel(tok, cfg)


_NEEDS_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required (server-only test)"
)
_NEEDS_MULTI_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="multi-GPU required (server-only test)",
)


# ---------------------------------------------------------------------------
# 1. Numerical robustness of losses under server-realistic inputs
# ---------------------------------------------------------------------------


def test_whitened_mse_psd_floor_is_finite_on_zero_bin():
    """A PSD with a near-zero bin must not NaN the loss.

    Reason this matters: when we re-estimate PSD from real pretraining
    data, some high-frequency bins above the notch may have very low
    power. Without the eps floor, ``rsqrt(psd)`` blows up and any
    error in those bins becomes inf -> NaN under backward.
    """
    T = 256
    psd = torch.full((T // 2 + 1,), 1.0)
    psd[10] = 0.0  # the test
    loss_fn = WhitenedMSELoss(psd=psd)
    pred = torch.randn(2, 3, T, requires_grad=True)
    target = torch.randn(2, 3, T)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss).item(), "PSD floor failed: zero bin produced inf/NaN"
    loss.backward()
    assert torch.isfinite(pred.grad).all().item(), "gradient through floored PSD is non-finite"


def test_whitened_mse_handles_silent_channel():
    """Masked-channel pred is ~0 at init for LinearVAR; loss must be finite there.

    On the masked path the harness slices recon[mask] and segments[mask]
    to (B*k, T). For LinearVAR at init, pred is the bias (~0), so the
    target carries all the energy. PSD is estimated from target — that's
    fine — but make sure the loss is finite when pred is uniformly zero.
    """
    T = 256
    loss_fn = WhitenedMSELoss(psd=None)  # on-the-fly PSD from target
    pred = torch.zeros(4, 1, T, requires_grad=True)
    target = torch.randn(4, 1, T)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(pred.grad).all().item()


def test_distdf_diag_finite_under_low_variance_pred():
    """DistDF (diagonal) must be finite when pred has near-zero variance.

    Reason: at init the LinearVAR's masked-channel prediction is ~0 with
    near-zero variance, so sigma_p ~ 0 and the BW closed form involves
    (sigma_p - sigma_t)^2 — fine algebraically but easy to break with
    a bad clamp. This test pins the contract.
    """
    loss_fn = DistDFLoss(alpha=0.05, full_cov=False)
    pred = torch.zeros(4, 1, 256, requires_grad=True)
    target = torch.randn(4, 1, 256)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(pred.grad).all().item()


def test_distdf_full_cov_finite_under_singular_pred():
    """Full-covariance DistDF: matrix-sqrt must not NaN on a low-rank pred.

    The eps*I ridge inside ``_matrix_sqrt_psd`` is what saves us when
    pred's empirical covariance is rank-deficient (e.g. from a
    near-collapsed AR model). If we ever drop or weaken that ridge,
    this test catches it.
    """
    loss_fn = DistDFLoss(alpha=0.05, full_cov=True)
    base = torch.randn(4, 1, 256)
    pred = base.repeat(1, 3, 1).clone().requires_grad_(True)  # rank-1 in C
    target = torch.randn(4, 3, 256)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(pred.grad).all().item()


def test_distdf_linear_ar_one_step_with_grad_clip_does_not_nan():
    """End-to-end: distdf + LinearVAR + masked path must stay finite for
    one step IF gradients are clipped.

    This documents the laptop-stage finding that distdf__linear_ar
    NaN'd without grad clipping. The server pretrain loop must clip
    gradients (clip_grad_norm_(1.0)) — this test verifies that with
    clipping the same combo is stable for one step.
    """
    torch.manual_seed(0)
    B, C, T = 4, 4, 256
    model = LinearVARModel(num_channels=C, order=3)
    loss_fn = DistDFLoss(alpha=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(B, C, T)
    mask = torch.zeros(B, C, dtype=torch.bool)
    mask[:, 1] = True
    for _ in range(5):
        recon = model(x, mask_channels=mask)
        pred_m = recon[mask].view(B, 1, T)
        true_m = x[mask].view(B, 1, T)
        loss = loss_fn(pred_m, true_m)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        assert torch.isfinite(loss).item(), "distdf + linear_ar NaN'd even with grad clipping"


# ---------------------------------------------------------------------------
# 2. clip_grad_norm + AdamW smoke (CPU is fine, but the server loop must
#    use these)
# ---------------------------------------------------------------------------


def test_clip_grad_norm_caps_norm():
    model = _build_small_transformer(C=4, T=512)
    x = torch.randn(2, 4, 512)
    region_ids = torch.arange(4)
    y = model(x, region_ids)
    loss = (y ** 2).mean() * 1e6  # huge loss → huge gradient
    loss.backward()
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # post-clip per-param L2 should not exceed 1.0 (with small slack).
    post = sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None).sqrt()
    assert post.item() <= 1.0 + 1e-4, f"clip failed: post-norm={post.item()}"
    assert total.item() > 1.0, "test broken — pre-clip norm should have been > 1"


# ---------------------------------------------------------------------------
# 3. CUDA + bf16 autocast (server-only)
# ---------------------------------------------------------------------------


@_NEEDS_CUDA
def test_cuda_forward_backward_smoke():
    """Stock GammaEncoderModel: one forward + backward on CUDA."""
    model = _build_small_transformer(C=8, T=1024).cuda()
    x = torch.randn(4, 8, 1024, device="cuda")
    region_ids = torch.arange(8, device="cuda")
    y = model(x, region_ids)
    loss = (y - x).pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss).item()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item()


@_NEEDS_CUDA
def test_cuda_bf16_autocast_forward_backward():
    """bf16 autocast + GradScaler smoke. bf16 doesn't need GradScaler in
    practice but the server loop will gate on amp dtype, so test both
    paths run cleanly.
    """
    model = _build_small_transformer(C=8, T=1024).cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 8, 1024, device="cuda")
    region_ids = torch.arange(8, device="cuda")
    optim.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x, region_ids)
        loss = (y - x).pow(2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    assert torch.isfinite(loss).item()


@_NEEDS_CUDA
def test_cuda_fp16_autocast_with_grad_scaler():
    """fp16 autocast path: needs GradScaler. Some spectral losses may
    misbehave in fp16; if the server picks fp16, this test pins the
    happy path."""
    model = _build_small_transformer(C=8, T=1024).cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(4, 8, 1024, device="cuda")
    region_ids = torch.arange(8, device="cuda")
    optim.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x, region_ids)
        loss = (y - x).pow(2).mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optim)
    scaler.update()
    assert torch.isfinite(loss).item()


@_NEEDS_CUDA
def test_linear_var_cuda_masked_smoke():
    """LinearVAR + masked channel path on CUDA, finite loss."""
    B, C, T = 4, 8, 1024
    model = LinearVARModel(num_channels=C, order=3).cuda()
    x = torch.randn(B, C, T, device="cuda")
    mask = torch.zeros(B, C, dtype=torch.bool, device="cuda")
    mask[:, :2] = True
    recon = model(x, mask_channels=mask)
    loss = ((recon[mask] - x[mask]) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss).item()


# ---------------------------------------------------------------------------
# 4. DDP smoke (multi-GPU server only)
# ---------------------------------------------------------------------------


@_NEEDS_MULTI_GPU
def test_ddp_wraps_model_and_takes_step():
    """Sanity that GammaEncoderModel is DDP-friendly.

    Spawn-style DDP testing is heavy for a unit test. We do the
    lighter check: init a process group with rank 0, world 1, wrap
    the model in DDP, run a step. This catches register_buffer issues
    or unused-parameter warnings without needing torchrun.
    """
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed not available")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", rank=0, world_size=1, init_method="env://"
        )
    try:
        model = _build_small_transformer(C=4, T=512).cuda(0)
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])
        x = torch.randn(2, 4, 512, device="cuda:0")
        region_ids = torch.arange(4, device="cuda:0")
        y = ddp(x, region_ids)
        loss = (y - x).pow(2).mean()
        loss.backward()
        assert torch.isfinite(loss).item()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# 5. Streaming dataset (skipped until module exists)
# ---------------------------------------------------------------------------


def test_streaming_dataset_module_exists():
    """The pretrain loop needs a streaming Dataset that yields
    (segments, region_ids) tuples without loading every trial into
    memory. Skips until ``gamma_encoder.data.dataset`` lands.
    """
    pytest.importorskip(
        "gamma_encoder.data.dataset",
        reason="streaming Dataset not built yet; build before server pretraining",
    )


def test_streaming_dataset_indexable_and_finite():
    """When the streaming dataset exists, it must:
    - implement __len__,
    - return (segments: (C, T), region_ids: (C,)) per __getitem__,
    - produce finite, z-scored values.
    """
    mod = pytest.importorskip("gamma_encoder.data.dataset")
    if not hasattr(mod, "BrainTreebankDataset"):
        pytest.skip("BrainTreebankDataset class not implemented yet")
    ds_cls = mod.BrainTreebankDataset
    # The class needs to accept a small synthetic config or the module
    # needs to expose a make_synthetic() helper. Document the contract:
    if not hasattr(mod, "make_synthetic"):
        pytest.skip("make_synthetic() helper for tests not implemented yet")
    ds = mod.make_synthetic(n_segments=4, n_channels=4, segment_samples=512)
    assert len(ds) == 4
    seg, rid = ds[0]
    assert seg.shape == (4, 512)
    assert rid.shape == (4,)
    assert torch.isfinite(seg).all().item()
    # z-scored => mean ~ 0, std ~ 1 per channel
    assert seg.mean(dim=-1).abs().max().item() < 0.5
    assert (seg.std(dim=-1) - 1.0).abs().max().item() < 0.5


# ---------------------------------------------------------------------------
# 6. Pretrain entrypoint (skipped until module exists)
# ---------------------------------------------------------------------------


def test_pretrain_entrypoint_module_exists():
    pytest.importorskip(
        "gamma_encoder.training.pretrain",
        reason="pretrain entrypoint not built yet; build before server run",
    )


def test_pretrain_run_smoke(tmp_path):
    """Two-step pretrain on a synthetic batch must drop loss and write a
    checkpoint. Skips until the pretrain module exposes ``run_pretrain``.
    """
    mod = pytest.importorskip(
        "gamma_encoder.training.pretrain",
        reason="pretrain entrypoint not built yet",
    )
    if not hasattr(mod, "run_pretrain"):
        pytest.skip("run_pretrain() not implemented yet")
    # Document the expected interface: takes a Dataset, returns a report
    # with at least .initial_loss and .final_loss, and writes a
    # checkpoint to out_dir / "model.pt".
    pytest.skip(
        "Test skeleton — populate when run_pretrain signature stabilizes. "
        "Expected: report.final_loss < report.initial_loss; "
        "(out_dir / 'model.pt').exists()"
    )


# ---------------------------------------------------------------------------
# 7. PSD recompute helper (skipped until script exists)
# ---------------------------------------------------------------------------


def test_psd_estimate_helper_exists():
    """The server flow re-estimates whitened-MSE PSD from real
    pretraining data. The helper that does this should live alongside
    the existing cache_whitened_mse_psd.py script and be importable.
    """
    try:
        importlib.import_module("scripts.cache_whitened_mse_psd")
    except ModuleNotFoundError:
        pytest.skip("scripts.cache_whitened_mse_psd not importable as a module")


def test_psd_estimate_floor_after_average():
    """PSD averaged over many segments must stay strictly positive in
    every bin (otherwise WhitenedMSELoss would divide by zero on the
    real data path). Synthetic check: random-Gaussian segments have
    nonzero power at every bin in expectation; verify and assert a
    realistic floor.
    """
    torch.manual_seed(0)
    n_seg, C, T = 64, 8, 512
    x = torch.randn(n_seg, C, T)
    Xf = torch.fft.rfft(x, n=T, dim=-1)
    psd = (Xf.abs() ** 2).mean(dim=(0, 1))  # (n_freqs,)
    assert psd.min().item() > 0.0
    # Once we estimate from real Subject 2 segments, document the
    # post-notch floor expectation here. For random Gaussians the floor
    # is comfortably > 0.5; on real notch-filtered data it can dip near
    # zero at 60/120/180 Hz, which is the case the eps in
    # WhitenedMSELoss must handle.
