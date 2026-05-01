"""Region-level spatial encoder (Destrieux atlas embedding lookup).

Each channel has an integer Destrieux region id; the spatial encoder
embeds it into ``d_model``-space and broadcasts across patches so it can
be added to the tokenizer output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RegionSpatialEncoder(nn.Module):
    """Embedding lookup over Destrieux region ids.

    Parameters
    ----------
    num_regions : int
        Vocabulary size (max region id + 1).
    d_model : int
        Embedding dimension.
    """

    def __init__(self, num_regions: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_regions, d_model)

    def forward(self, region_ids: torch.Tensor) -> torch.Tensor:
        """region_ids: (B, C) or (C,) -> (B or 1, C, 1, d_model)."""
        if region_ids.dim() == 1:
            emb = self.embedding(region_ids)        # (C, d_model)
            return emb[None, :, None, :]            # (1, C, 1, d_model)
        if region_ids.dim() == 2:
            emb = self.embedding(region_ids)        # (B, C, d_model)
            return emb[:, :, None, :]               # (B, C, 1, d_model)
        raise ValueError(f"region_ids must be 1D or 2D, got shape {tuple(region_ids.shape)}")
