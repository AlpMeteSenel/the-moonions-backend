"""Touchdown-safety network (v2): a physics-informed residual model.

logit P(safe touchdown) = logit(physics estimate on the 80 m map) + f(terrain, context)

The physics baseline (same Monte-Carlo simulator run on the coarse map) is the prior; the network
learns how coarse terrain translates into fine-scale risk (sub-80 m slopes, hazard-avoidance
success, tip-over). If it learns nothing it falls back to physics instead of doing worse.
Terrain is seen through a small CNN on the 80 m patch and/or frozen NASA-IBM Lunar Foundation
Model embeddings. A second head predicts P(no divert needed) as an auxiliary task.
"""

from __future__ import annotations

import torch
from torch import nn

from .model import Block


def logit(p: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


class TouchdownNet(nn.Module):
    def __init__(self, n_tab: int, in_ch: int = 3, width: int = 32, use_cnn: bool = True, emb_dim: int = 0):
        super().__init__()
        self.use_cnn, self.emb_dim = use_cnn, emb_dim
        feat = 0
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_ch, width, 3, 2, 1, bias=False),  # stride-2 stem: 4x fewer FLOPs
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
            self.emb = nn.Sequential(nn.LayerNorm(emb_dim), nn.Dropout(0.3), nn.Linear(emb_dim, 128), nn.GELU())
            feat += 128
        self.tab = nn.Sequential(nn.Linear(n_tab, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        feat += 64
        self.head = nn.Sequential(nn.Linear(feat, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 2))
        nn.init.zeros_(self.head[-1].weight)  # start exactly at the physics prior
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, patch, tab, emb, prior_td, prior_direct):
        parts = []
        if self.use_cnn:
            parts.append(self.cnn(patch))
        if self.emb_dim:
            parts.append(self.emb(emb))
        parts.append(self.tab(tab))
        r = self.head(torch.cat(parts, 1))
        return torch.stack([logit(prior_td) + r[:, 0], logit(prior_direct) + r[:, 1]], 1)
