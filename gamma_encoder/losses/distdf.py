"""DistDF loss: (1-alpha) * MSE + alpha * Bures-Wasserstein on joints.

Wang et al. 2026. Addresses MSE's autocorrelation bias by aligning the
joint distributions of (X, Y) and (X, Y_hat) — where X is the input
and Y/Y_hat are target/prediction — under a Gaussian assumption. The
Bures-Wasserstein-2 distance between two Gaussians N(mu_a, S_a) and
N(mu_b, S_b) is

    BW^2 = ||mu_a - mu_b||^2 + tr(S_a + S_b - 2*(S_a^{1/2} S_b S_a^{1/2})^{1/2})

In our reconstruction setting (no separate input X), we use the
single-stream simplification: align distributions of `target` and
`pred` directly by computing means and covariances over (batch *
channels) samples treated as iid d-dimensional vectors, where d is
chosen by stride-flattening the time axis. This is a faithful adaptation
of DistDF for autoencoder-style reconstruction (no autoregressive X).

For computational sanity, we treat each time-step as one sample of a
1-D variable and align the (mean, variance) per channel — i.e. the
diagonal Gaussian case. The full-covariance version is kept available
behind ``full_cov=True`` (uses matrix-square-root via eigendecomp of a
PSD matrix).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gamma_encoder.losses.base import ReconstructionLoss


def _matrix_sqrt_psd(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric matrix square root of a PSD matrix via eigendecomp.

    Adds eps*I to keep the eigendecomp well-conditioned.
    """
    d = m.shape[-1]
    eye = torch.eye(d, device=m.device, dtype=m.dtype)
    sym = 0.5 * (m + m.transpose(-1, -2)) + eps * eye
    evals, evecs = torch.linalg.eigh(sym)
    evals = evals.clamp_min(0.0)
    return evecs @ torch.diag_embed(evals.sqrt()) @ evecs.transpose(-1, -2)


def _bures_wasserstein_full(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """BW^2 between full-covariance Gaussians fit to pred and target.

    pred, target: (B, C, T). We treat (batch, time) as samples and fit
    a C-dimensional Gaussian to each. Returns a scalar.
    """
    B, C, T = pred.shape
    # Reshape to (samples, dim) = (B*T, C).
    p = pred.permute(0, 2, 1).reshape(-1, C)
    t = target.permute(0, 2, 1).reshape(-1, C)
    mu_p = p.mean(dim=0)
    mu_t = t.mean(dim=0)
    pc = p - mu_p
    tc = t - mu_t
    n = p.shape[0]
    cov_p = pc.t() @ pc / max(n - 1, 1)
    cov_t = tc.t() @ tc / max(n - 1, 1)

    sqrt_t = _matrix_sqrt_psd(cov_t, eps=eps)
    inner = sqrt_t @ cov_p @ sqrt_t
    sqrt_inner = _matrix_sqrt_psd(inner, eps=eps)
    trace_term = torch.trace(cov_p) + torch.trace(cov_t) - 2.0 * torch.trace(sqrt_inner)
    mean_term = (mu_p - mu_t).pow(2).sum()
    return mean_term + trace_term.clamp_min(0.0)


def _bures_wasserstein_diag(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Closed-form BW^2 for diagonal-covariance Gaussians.

    For diagonal cov, BW^2 = sum_i (mu_p_i - mu_t_i)^2 + (sigma_p_i - sigma_t_i)^2.
    Computed per channel over (batch, time) samples, then summed.
    """
    B, C, T = pred.shape
    p = pred.permute(1, 0, 2).reshape(C, -1)  # (C, B*T)
    t = target.permute(1, 0, 2).reshape(C, -1)
    mu_p = p.mean(dim=1)
    mu_t = t.mean(dim=1)
    var_p = p.var(dim=1, unbiased=False)
    var_t = t.var(dim=1, unbiased=False)
    sd_p = var_p.clamp_min(0.0).sqrt()
    sd_t = var_t.clamp_min(0.0).sqrt()
    return ((mu_p - mu_t) ** 2 + (sd_p - sd_t) ** 2).sum()


class DistDFLoss(ReconstructionLoss):
    """(1 - alpha) * MSE + alpha * BW^2.

    Default alpha=0.05 from the DistDF paper. ``full_cov=False`` (the
    default) uses the diagonal-Gaussian closed form, which is fast and
    backprop-stable. ``full_cov=True`` uses matrix-sqrt via eigendecomp;
    correct but ~10x slower and occasionally numerically delicate.
    """

    def __init__(self, alpha: float = 0.05, full_cov: bool = False) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.full_cov = bool(full_cov)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        if pred.dim() != 3:
            raise ValueError(f"DistDF expects (B, C, T); got {tuple(pred.shape)}")
        mse = F.mse_loss(pred, target)
        if self.full_cov:
            bw = _bures_wasserstein_full(pred, target)
        else:
            bw = _bures_wasserstein_diag(pred, target)
        return (1.0 - self.alpha) * mse + self.alpha * bw
