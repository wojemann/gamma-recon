"""Linear tokenizer: per-patch ``Linear(L, d_model)``.

Cheapest possible tokenizer. Establishes whether learned conv features
matter at all for the gamma-fidelity story; if a flat linear projection
on raw samples gives reconstruction comparable to a multi-block dilated
CNN tokenizer, that's a striking result.

No nonlinearities, no pooling, no smoothing. The linear map preserves
all frequency content (it's an invertible linear operator at d_model
sufficient).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gamma_encoder.tokenizers.base import Tokenizer


class LinearTokenizer(Tokenizer):
    """Flat linear projection from raw patch samples to d_model."""

    def __init__(self, d_model: int, patch_samples: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.patch_samples = int(patch_samples)
        self.proj = nn.Linear(self.patch_samples, self.d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 4:
            raise ValueError(f"expected (B, C, n, L), got {tuple(patches.shape)}")
        if patches.shape[-1] != self.patch_samples:
            raise ValueError(
                f"patch length {patches.shape[-1]} != configured {self.patch_samples}"
            )
        return self.proj(patches)
