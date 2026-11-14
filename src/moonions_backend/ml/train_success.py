"""Train + evaluate the mission-success model against rule-based and physics baselines.

  uv run python -m moonions_backend.ml.train_success --variants cnn,lfm,both --epochs 40

Variants differ only in how terrain is seen:
  cnn  : small CNN on the 80 m LOLA patch (from scratch)
  lfm  : frozen NASA-IBM Lunar Foundation Model embeddings (needs data/cache/lfm_embeddings.npz)
  both : CNN + LFM embeddings
All share the tabular planning features and the same geographic split. The deployed model is
chosen by VALIDATION success-ranking (never by test). Writes data/models/success_net.pt and
data/models/success_model_card.json (with every variant's test results for comparison).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import CACHE, MODELS
from .simulate import DEFAULT
from .success_dataset import HEADS, OUT, TAB_FEATURES
from .success_model import SuccessNet

CKPT = MODELS / "success_net.pt"
CARD = MODELS / "success_model_card.json"
EMB = CACHE / "lfm_embeddings.npz"
HOLDOUT_LON = (60.0, 120.0)
VAL_LON = (-60.0, -30.0)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pick_quality(pred: np.ndarray, truth: np.ndarray, date_idx: np.ndarray, k: int = 10, n_sets: int = 4000, seed: int = 0) -> dict:
    """Choose the best of k random candidate sites for the same date; compare the truth of the pick."""
    rng = np.random.default_rng(seed)
    regrets, picked, oracle, hit = [], [], [], []
    dates = np.unique(date_idx)
    for _ in range(n_sets):
        d = rng.choice(dates)
        idx = np.flatnonzero(date_idx == d)
        if len(idx) < k:
            continue
        c = rng.choice(idx, k, replace=False)
        p = c[np.argmax(pred[c])]
        best = truth[c].max()
        picked.append(truth[p])
        oracle.append(best)
        regrets.append(best - truth[p])
        hit.append(truth[p] >= best - 0.05)
    return {
        "mean_success_of_pick": float(np.mean(picked)),
        "mean_best_possible": float(np.mean(oracle)),
        "mean_regret": float(np.mean(regrets)),
        "near_best_rate": float(np.mean(hit)),
    }


def evaluate(pred_succ, pred_td, truth_succ, truth_td, date_idx) -> dict:
    return {
        "success_spearman": float(spearmanr(pred_succ, truth_succ).statistic),
        "success_auc": auc(pred_succ, truth_succ >= 0.5),
        "touchdown_spearman": float(spearmanr(pred_td, truth_td).statistic) if pred_td is not None else None,
        "touchdown_mae": float(np.mean(np.abs(pred_td - truth_td))) if pred_td is not None else None,
        "pick_best_of_10": pick_quality(pred_succ, truth_succ, date_idx),
    }


def calibration(pred, truth, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pred >= a) & ((pred < b) if b < 1 else (pred <= b))
        if m.sum() >= 20:
            out.append({"bin": [round(float(a), 2), round(float(b), 2)], "predicted": float(pred[m].mean()), "actual": float(truth[m].mean()), "n": int(m.sum())})
    return out


def load_embeddings(meta: np.ndarray) -> np.ndarray:
    e = np.load(EMB)
    table = {(round(float(a), 5), round(float(b), 5)): i for i, (a, b) in enumerate(zip(e["lat"], e["lon"]))}
    rows = np.array([table.get((round(float(a), 5), round(float(b), 5)), -1) for a, b in meta[:, :2]])
    if (rows < 0).any():
        raise RuntimeError(f"{(rows < 0).sum()} samples have no LFM embedding; run lfm_embed --from-dataset")
    return np.concatenate([e["center"], e["global"]], 1).astype(np.float32)[rows]


def train_variant(variant: str, d, emb, epochs: int, batch: int, lr: float, dev: str):
    Xp, Xt, Y, S = d["X_patch"], d["X_tab"], d["Y"], d["success"]
    meta = d["meta"]
    lon = meta[:, 1]
    test = (lon >= HOLDOUT_LON[0]) & (lon < HOLDOUT_LON[1])
    val = (lon >= VAL_LON[0]) & (lon < VAL_LON[1])
    train = ~test & ~val
    use_cnn = variant in ("cnn", "both")
    use_emb = variant in ("lfm", "both")
    mu, sd = Xt[train].mean(0), Xt[train].std(0) + 1e-6
    Xtn = ((Xt - mu) / sd).astype(np.float32)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()  # noqa: E731
    E = emb if use_emb else np.zeros((len(S), 1), np.float32)
    P = Xp if use_cnn else np.zeros((len(S), 1, 1, 1), np.float32)
    tr_ds = TensorDataset(T(P[train]), T(Xtn[train]), T(E[train]), T(Y[train]))
    torch.manual_seed(0)
    model = SuccessNet(n_tab=Xt.shape[1], in_ch=Xp.shape[1], n_heads=Y.shape[1], use_cnn=use_cnn, emb_dim=E.shape[1] if use_emb else 0).to(dev)
    head_w = torch.tensor([2.0, 1.0, 1.0, 1.0], device=dev)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(tr_ds) // batch + 1))

    def predict(mask):
        model.eval()
        outs = []
        idx = np.flatnonzero(mask)
        with torch.no_grad():
            for i in range(0, len(idx), 2048):
                sl = idx[i : i + 2048]
                outs.append(torch.sigmoid(model(T(P[sl]).to(dev) if use_cnn else None, T(Xtn[sl]).to(dev), T(E[sl]).to(dev) if use_emb else None)).cpu().numpy())
        return np.concatenate(outs)

    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for xp, xt, xe, y in DataLoader(tr_ds, batch_size=batch, shuffle=True):
            if use_cnn:
                xp = torch.rot90(xp, int(torch.randint(0, 4, (1,))), dims=(2, 3)).to(dev)
            logits = model(xp if use_cnn else None, xt.to(dev), xe.to(dev) if use_emb else None)
            y = y.to(dev)
            loss = (bce(logits, y) * head_w).mean()
            loss = loss + nn.functional.binary_cross_entropy(SuccessNet.success(logits).clamp(1e-5, 1 - 1e-5), y.prod(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        pv = predict(val)
        rho = float(spearmanr(pv.prod(1), S[val]).statistic)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[{variant}] epoch {ep+1:3d} loss {loss.item():.4f} val success ρ {rho:.4f} touchdown ρ {spearmanr(pv[:,0], Y[val,0]).statistic:.4f}", flush=True)
        if rho > best:
            best, best_state = rho, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pt = predict(test)
    res = evaluate(pt.prod(1), pt[:, 0], S[test], Y[test, 0], meta[test, 2].astype(int))
    res["val_success_spearman"] = best
    res["head_auc"] = {h: auc(pt[:, i], Y[test, i] >= 0.5) for i, h in enumerate(HEADS) if i > 0}
    ckpt = {
        "state_dict": model.state_dict(), "mu": mu, "sd": sd, "tab_features": TAB_FEATURES, "heads": HEADS,
        "n_tab": Xt.shape[1], "in_ch": Xp.shape[1], "use_cnn": use_cnn, "emb_dim": E.shape[1] if use_emb else 0,
        "variant": variant, "version": f"success_net_v1_{variant}",
    }  # fmt: skip
    return res, ckpt, calibration(pt.prod(1), S[test]), (train, val, test)


def main(variants: list[str], epochs: int, batch: int, lr: float):
    d = np.load(OUT)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb = None
    if any(v in ("lfm", "both") for v in variants):
        emb = load_embeddings(d["meta"])
    results, ckpts, calib = {}, {}, {}
    for v in variants:
        results[v], ckpts[v], calib[v], split = train_variant(v, d, emb, epochs, batch, lr, dev)
        torch.save(ckpts[v], MODELS / f"success_net_{v}.pt")
    train, val, test = split
    chosen = max(variants, key=lambda v: results[v]["val_success_spearman"])
    S, Y, Xt, meta = d["success"], d["Y"], d["X_tab"], d["meta"]
    date_idx = meta[test, 2].astype(int)
    rule = evaluate(d["rule"][test] / 100.0, None, S[test], Y[test, 0], date_idx)
    rule["touchdown_spearman"] = float(spearmanr(-Xt[test, TAB_FEATURES.index("slope_p90_deg")], Y[test, 0]).statistic)
    phys = d["phys80"][test]
    phys_ev = evaluate(phys * Y[test, 1:].prod(1), phys, S[test], Y[test, 0], date_idx)

    card = {
        "name": "Moonions mission-success model",
        "version": ckpts[chosen]["version"],
        "variant": chosen,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "task": "Predict the probability that a landing at a given site and touchdown time succeeds (safe touchdown, light at landing, enough energy, Earth link) from planning-level NASA data.",
        "labels": "Monte-Carlo landing simulator on LOLA 20 m south-polar terrain + JPL DE421 sunlight/Earth geometry + battery simulation.",
        "inputs": {
            "terrain_patch": "LOLA 80 m DEM, 32x32 px (2.5 km): detrended height, slope, curvature (variants cnn/both)",
            "foundation_model": "NASA-IBM Lunar Foundation Model (ViT-B, frozen) embeddings of a 25.6 km DTM/slope/aspect tile at 100 m/px (variants lfm/both)",
            "tabular": TAB_FEATURES,
        },
        "foundation_model": {"name": "NASA-IBM Lunar Foundation Model", "repo": "nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model", "license": "Apache-2.0", "used": chosen in ("lfm", "both")},
        "heads": HEADS,
        "simulator": DEFAULT.to_dict(),
        "data": {
            "samples": int(len(S)), "train": int(train.sum()), "val": int(val.sum()), "test": int(test.sum()),
            "split": f"geographic: test = longitudes {HOLDOUT_LON[0]:g}..{HOLDOUT_LON[1]:g}°E, val = {VAL_LON[0]:g}..{VAL_LON[1]:g}°E, never seen in training",
            "region": "south polar cap, latitudes -80.6..-90",
            "mean_true_success": float(S.mean()),
        },  # fmt: skip
        "results_test": {"ai_model": results[chosen], "rule_based_score": rule, "physics_80m_baseline": phys_ev},
        "variants_test": results,
        "head_auc_test": results[chosen]["head_auc"],
        "calibration_test": calib[chosen],
        "limitations": [
            "Trained on the lunar south polar region only; elsewhere it is extrapolating.",
            "20 m terrain is the finest wide-coverage truth; boulders smaller than ~20 m are not represented.",
            "Navigation dispersion, hazard-avoidance range and power system are generic assumptions (see simulator).",
            "Light, energy and Earth-link outcomes are physics the model re-learns; its main added value is the touchdown head and speed.",
        ],
    }  # fmt: skip
    torch.save(ckpts[chosen], CKPT)
    CARD.write_text(json.dumps(card, indent=2))
    print("chosen by validation:", chosen)
    print(json.dumps({k: {m: card["results_test"][k][m] for m in ("success_spearman", "success_auc", "touchdown_spearman", "pick_best_of_10")} for k in card["results_test"]}, indent=1))
    for v in variants:
        r = results[v]
        print(f"{v:5s} test success ρ {r['success_spearman']:.4f} AUC {r['success_auc']:.4f} touchdown ρ {r['touchdown_spearman']:.4f} MAE {r['touchdown_mae']:.4f} pick regret {r['pick_best_of_10']['mean_regret']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="cnn,lfm,both")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    main(a.variants.split(","), a.epochs, a.batch, a.lr)
