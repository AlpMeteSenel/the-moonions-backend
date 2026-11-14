"""Train the hazard CNN on the auto-labelled dataset.

  uv sync --extra ml
  uv run python -m moonions_backend.ml.dataset --n 20000
  uv run python -m moonions_backend.ml.train --epochs 30
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import MODELS
from .dataset import CLASSES, COARSE_RES_M, OUT, PATCH
from .model import HazardCNN


def augment(x: torch.Tensor) -> torch.Tensor:
    # terrain has no preferred orientation: random 90° rotations and flips
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, dims=(2, 3))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(3,))
    return x


def main(epochs: int, batch: int, lr: float):
    d = np.load(OUT)
    X, y = torch.from_numpy(d["X"]), torch.from_numpy(d["y"])
    # spatial split by longitude band so train/val don't overlap geographically
    lon = d["meta"][:, 1]
    val = (lon > 120) & (lon < 180)
    tr_ds = TensorDataset(X[~val], y[~val])
    va_ds = TensorDataset(X[val], y[val])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = HazardCNN(in_ch=X.shape[1], n_classes=len(CLASSES)).to(dev)
    counts = torch.bincount(y[~val], minlength=len(CLASSES)).float()
    weight = (counts.sum() / (counts + 1)).to(dev)
    weight = weight / weight.mean()
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(tr_ds) // batch + 1))
    best = 0.0
    for ep in range(epochs):
        model.train()
        for xb, yb in DataLoader(tr_ds, batch_size=batch, shuffle=True, drop_last=False):
            xb, yb = augment(xb).to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        conf = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
        with torch.no_grad():
            for xb, yb in DataLoader(va_ds, batch_size=512):
                pred = model(xb.to(dev)).argmax(1).cpu()
                for t, p in zip(yb, pred):
                    conf[t, p] += 1
        recall = (conf.diag().float() / conf.sum(1).clamp(min=1).float())
        bal_acc = float(recall.mean())
        print(f"epoch {ep+1:3d} loss {loss.item():.3f} val balanced-acc {bal_acc:.3f} recall {[round(float(r),3) for r in recall]}")
        if bal_acc > best:
            best = bal_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "in_ch": X.shape[1],
                    "n_classes": len(CLASSES),
                    "classes": CLASSES,
                    "patch": PATCH,
                    "patch_res_m": COARSE_RES_M,
                    "val_balanced_acc": bal_acc,
                    "confusion": conf.tolist(),
                    "version": "hazard_cnn_v1",
                },
                MODELS / "hazard_cnn.pt",
            )
    print("best val balanced accuracy", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    main(a.epochs, a.batch, a.lr)
