"""Linear vector-autoregression (VAR / MVAR) reconstruction model.

A multivariate AR(p) baseline that mixes information across channels
at each time step. Each output sample is a linear function of the
past ``order`` samples of *all* channels:

    y[c, t] = b[c] + Σ_{c', k=1..p} W[c, c', k] * x[c', t - k]

Compared to a per-channel scalar AR predictor, this gives the loss
function the same kind of cross-channel mixing capability the
transformer has (where attention pools tokens across channels). It
is the simplest "spatial aggregation" baseline that is still linear,
strictly causal, and parameter-efficient.

For ``num_channels=8, order=3`` this is exactly ``8*8*3 + 8 = 200``
parameters. Still tiny relative to the 360k-param transformer used
in the rest of the sweep.

Implementation: one ``nn.Conv1d(C, C, kernel_size=p)`` applied to a
left-padded input. Conv1d's cross-correlation convention means the
kernel index ``k_conv`` corresponds to AR lag ``p - k_conv``, so
``conv.weight[..., p-1]`` holds the lag-1 coefficients.

Initialization: identity 1-sample delay — ``y[c, t] = x[c, t-1]``,
no cross-channel coupling. So at step zero the model already produces
target-shaped output (1-sample-delayed copy), which is the floor any
loss has to improve upon. Cross-channel coupling has to be learned
from gradient signal.

Channel masking: ``mask_channels`` is accepted in ``forward`` for
interface parity with the transformer model but is intentionally a
no-op. Zero-filling the AR's input at masked channels would corrupt
system identification — the conv would fit cross-channel coefficients
against synthetic zeros instead of the true joint process. Instead,
the AR runs unmasked and the caller scores on whichever channels the
masking protocol calls out (so the AR's own self-history floor is
included in the baseline; that's the canonical thing the transformer
must beat).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearVARModel(nn.Module):
    """Causal multivariate AR(p) FIR predictor with full channel mixing."""

    def __init__(self, num_channels: int, order: int = 3) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        self.num_channels = int(num_channels)
        self.order = int(order)
        self.conv = nn.Conv1d(
            self.num_channels, self.num_channels, kernel_size=self.order, bias=True
        )
        # Init: y[c, t] = x[c, t-1]. Conv lag-1 lives at kernel index p-1.
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :, self.order - 1] = torch.eye(self.num_channels)
            self.conv.bias.zero_()

    def forward(
        self,
        signal: torch.Tensor,
        region_ids: torch.Tensor | None = None,
        mask_channels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ``mask_channels`` is accepted for interface parity with the
        # transformer model but is intentionally ignored. Zero-filling
        # the AR's input at masked channels would corrupt system
        # identification: the conv would fit cross-channel coefficients
        # against synthetic zeros instead of the true joint process.
        # Instead, the AR runs unmasked here and the caller scores on
        # whichever channels the masking protocol calls out (so the
        # baseline includes the channel's own AR self-history floor —
        # the canonical thing the transformer must beat).
        del mask_channels  # explicit no-op
        if signal.dim() != 3:
            raise ValueError(f"expected (B, C, T), got {tuple(signal.shape)}")
        if signal.shape[1] != self.num_channels:
            raise ValueError(
                f"expected {self.num_channels} channels, got {signal.shape[1]}"
            )
        B, C, T = signal.shape
        padded = F.pad(signal, (self.order, 0))     # (B, C, T+order)
        out = self.conv(padded)                     # (B, C, T+1)
        return out[..., :T]


# Backwards-compatible alias — registry/checkpoints use ``linear_ar``.
LinearARModel = LinearVARModel
