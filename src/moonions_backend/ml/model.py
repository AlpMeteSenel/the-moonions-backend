"""Small residual CNN for landing-hazard classification (fits a 4 GB GPU easily)."""

from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(y + self.skip(x))


class HazardCNN(nn.Module):
    def __init__(self, in_ch: int = 3, n_classes: int = 3, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.body = nn.Sequential(Block(width, width), Block(width, 2 * width, 2), Block(2 * width, 4 * width, 2), Block(4 * width, 4 * width, 2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(4 * width, n_classes))

    def forward(self, x):
        return self.head(self.body(self.stem(x)))
