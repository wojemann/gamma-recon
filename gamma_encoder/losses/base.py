"""Reconstruction loss interface."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """Abstract base class.

    Subclasses implement ``forward(pred, target) -> scalar tensor``.
    Both inputs share shape (B, C, T).
    """

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
