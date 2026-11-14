"""Mission-success network.

Terrain is seen through one or both of
  * a small CNN on the 80 m LOLA patch (trained from scratch), and
  * frozen NASA-IBM Lunar Foundation Model embeddings of a 25.6 km DTM/slope/aspect tile,
plus an MLP on tabular planning features (sunlight, Earth, energy, craters, terrain stats).
Outputs 4 heads (touchdown, lit0, energy_ok, comms_ok); P(success) = product of sigmoids.
"""

from __future__ import annotations

import torch
from torch import nn

from .model import Block

LFM_DIM = 768


class SuccessNet(nn.Module):
    def __init__(self, n_tab: int, in_ch: int = 3, n_heads: int = 4, width: int = 32, use_cnn: bool = True, emb_dim: int = 0):
        super().__init__()
        self.use_cnn = use_cnn
        self.emb_dim = emb_dim
        feat = 0
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_ch, width, 3, 1, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                Block(width, width),
                Block(width, 2 * width, 2),
                Block(2 * width, 4 * width, 2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            feat += 4 * width
        if emb_dim:
            # frozen foundation-model features (center + global = 2 x 768) → compact projection
            self.emb = nn.Sequential(nn.LayerNorm(emb_dim), nn.Dropout(0.2), nn.Linear(emb_dim, 128), nn.GELU())
            feat += 128
        self.tab = nn.Sequential(nn.Linear(n_tab, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        feat += 64
        self.fuse = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, n_heads))

    def forward(self, patch: torch.Tensor | None, tab: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """Returns logits (B, heads)."""
        parts = []
        if self.use_cnn:
            parts.append(self.cnn(patch))
        if self.emb_dim:
            parts.append(self.emb(emb))
        parts.append(self.tab(tab))
        return self.fuse(torch.cat(parts, 1))

    @staticmethod
    def success(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits).prod(1)
