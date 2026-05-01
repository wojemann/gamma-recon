"""Tokenizer interface.

A tokenizer maps a batch of raw waveform patches to per-(channel, patch)
latent vectors.

Shape contract:
    input  : (B, C, n_patches, L)   float
    output : (B, C, n_patches, d_model)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class Tokenizer(nn.Module):
    """Abstract tokenizer base.

    Subclasses must implement ``forward`` honoring the shape contract.
    Subclasses must expose ``d_model`` as an attribute.
    """

    d_model: int
    patch_samples: int

    @abstractmethod
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Convert raw patches to latent tokens.

        Parameters
        ----------
        patches : torch.Tensor
            Shape (B, C, n_patches, L). L must equal ``self.patch_samples``.

        Returns
        -------
        torch.Tensor
            Shape (B, C, n_patches, d_model).
        """
        raise NotImplementedError


def patchify(signal: torch.Tensor, patch_samples: int) -> torch.Tensor:
    """Split a (B, C, T) signal into non-overlapping patches.

    Returns shape (B, C, n_patches, patch_samples). Requires that T is
    divisible by ``patch_samples``.
    """
    if signal.dim() != 3:
        raise ValueError(f"expected (B, C, T), got {tuple(signal.shape)}")
    B, C, T = signal.shape
    if T % patch_samples != 0:
        raise ValueError(
            f"T={T} not divisible by patch_samples={patch_samples}"
        )
    n_patches = T // patch_samples
    return signal.view(B, C, n_patches, patch_samples)


def unpatchify(patches: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`patchify`. (B, C, n, L) -> (B, C, n*L)."""
    if patches.dim() != 4:
        raise ValueError(f"expected (B, C, n, L), got {tuple(patches.shape)}")
    B, C, n, L = patches.shape
    return patches.reshape(B, C, n * L)
