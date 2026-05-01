"""Full encoder model composed from tokenizer + spatial + transformer + decoder.

Forward pass:
    signal (B, C, T)
        -> patchify     (B, C, n_patches, L)
        -> tokenizer    (B, C, n_patches, d_model)
        -> + spatial    (B, C, n_patches, d_model)
        -> reshape      (B, C * n_patches, d_model)
        -> transformer  (B, C * n_patches, d_model)
        -> reshape      (B, C, n_patches, d_model)
        -> decoder      (B, C, n_patches, L)
        -> unpatchify   (B, C, T)

Returns the reconstructed waveform.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from gamma_encoder.models.decoder import LinearPatchDecoder
from gamma_encoder.models.encoder import SmallTransformerEncoder
from gamma_encoder.models.faithful import FaithfulTransformerEncoder
from gamma_encoder.models.spatial_encoder import RegionSpatialEncoder
from gamma_encoder.tokenizers.base import Tokenizer, patchify, unpatchify


@dataclass
class GammaEncoderConfig:
    d_model: int = 32
    n_layers: int = 6
    n_heads: int = 2
    ff_mult: int = 4
    patch_samples: int = 512
    num_regions: int = 64
    max_seq_len: int = 1024
    dropout: float = 0.0
    encoder_kind: str = "stock"  # "stock" or "faithful" (BaRISTA RoPE+RMSNorm+SwiGLU)


class GammaEncoderModel(nn.Module):
    """Compose tokenizer, spatial encoder, transformer, decoder."""

    def __init__(self, tokenizer: Tokenizer, cfg: GammaEncoderConfig) -> None:
        super().__init__()
        if tokenizer.d_model != cfg.d_model:
            raise ValueError(
                f"tokenizer.d_model={tokenizer.d_model} != cfg.d_model={cfg.d_model}"
            )
        if tokenizer.patch_samples != cfg.patch_samples:
            raise ValueError(
                f"tokenizer.patch_samples={tokenizer.patch_samples} "
                f"!= cfg.patch_samples={cfg.patch_samples}"
            )
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.spatial = RegionSpatialEncoder(cfg.num_regions, cfg.d_model)
        if cfg.encoder_kind == "stock":
            self.encoder = SmallTransformerEncoder(
                d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                ff_mult=cfg.ff_mult, max_seq_len=cfg.max_seq_len, dropout=cfg.dropout,
            )
        elif cfg.encoder_kind == "faithful":
            self.encoder = FaithfulTransformerEncoder(
                d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                mlp_ratio=cfg.ff_mult, max_seq_len=cfg.max_seq_len, dropout=cfg.dropout,
            )
        else:
            raise ValueError(f"unknown encoder_kind: {cfg.encoder_kind!r}")
        self.decoder = LinearPatchDecoder(cfg.d_model, cfg.patch_samples)

    def forward(self, signal: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        """signal: (B, C, T); region_ids: (B, C) or (C,) -> reconstructed (B, C, T)."""
        B, C, T = signal.shape
        patches = patchify(signal, self.cfg.patch_samples)            # (B, C, n, L)
        tokens = self.tokenizer(patches)                              # (B, C, n, d)
        tokens = tokens + self.spatial(region_ids)                    # broadcast
        n = tokens.size(2)
        seq = tokens.reshape(B, C * n, self.cfg.d_model)
        seq = self.encoder(seq)
        tokens_out = seq.reshape(B, C, n, self.cfg.d_model)
        recon_patches = self.decoder(tokens_out)                      # (B, C, n, L)
        return unpatchify(recon_patches)                              # (B, C, T)
