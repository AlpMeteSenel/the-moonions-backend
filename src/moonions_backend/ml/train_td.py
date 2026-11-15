"""Train + evaluate the v2 touchdown model (ensembles, 3 terrain-input variants) vs baselines.

  uv run python -m moonions_backend.ml.train_td --variants cnn,lfm,both --seeds 5 --epochs 60

Writes data/models/td_ensemble.pt and data/models/success_model_card.json.
Deployment choice = best VALIDATION touchdown error; test numbers are only reported.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from ..config import CACHE, MODELS
from .simulate import DEFAULT, time_outcomes
from .td_dataset import OUT, TAB
from .td_model import TouchdownNet

CKPT = MODELS / "td_ensemble.pt"
CARD = MODELS / "success_model_card.json"
EMB = CACHE / "lfm_embeddings_td.npz"
HOLDOUT_LON = (60.0, 120.0)
VAL_LON = (-60.0, -30.0)
VARIANT_LABEL = {"cnn": "Own CNN on 80 m terrain", "lfm": "NASA-IBM Lunar Foundation Model", "both": "CNN + foundation model"}


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pick_quality(pred, truth, groups, k=10, n_sets=6000, seed=0, require_viable=True):
    rng = np.random.default_rng(seed)
    picked, oracle, hit = [], [], []
    gids = np.unique(groups)
    while len(picked) < n_sets // 2 and n_sets > 0:
        n_sets -= 1
        idx = np.flatnonzero(groups == rng.choice(gids))
        if len(idx) < k:
            continue
        c = rng.choice(idx, k, replace=False)
        if require_viable and truth[c].max() <= 0:
            continue
        p = c[np.argmax(pred[c] + rng.uniform(0, 1e-9, k))]  # random tie-break
        picked.append(truth[p])
        oracle.append(truth[c].max())
        hit.append(truth[p] >= truth[c].max() - 0.05)
    return {
        "mean_success_of_pick": float(np.mean(picked)),
        "mean_best_possible": float(np.mean(oracle)),
        "mean_regret": float(np.mean(np.array(oracle) - np.array(picked))),
        "near_best_rate": float(np.mean(hit)),
        "n_sets": len(picked),
    }


def calibration(pred, truth, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pred >= a) & ((pred < b) if b < 1 else (pred <= b))
        if m.sum() >= 15:
            out.append({"bin": [round(float(a), 2), round(float(b), 2)], "predicted": float(pred[m].mean()), "actual": float(truth[m].mean()), "n": int(m.sum())})
    return out


def td_metrics(pred, truth, rng_groups):
    safe = truth >= 0.9
    return {
        "touchdown_spearman": float(spearmanr(pred, truth).statistic),
        "touchdown_mae": float(np.mean(np.abs(pred - truth))),
        "touchdown_brier": float(np.mean((pred - truth) ** 2)),
        "safe_site_auc": auc(pred, safe),
        "pick_best_of_10_sites": pick_quality(pred, truth, rng_groups, require_viable=False),
    }


def gates_for_dates(lat, lon, elev, horizon, n_dates=48):
    """Deterministic light / power / Earth-link outcomes for every site x date → (N, D) in {0,1}."""
    from ..ai import _horizon_at
    from .sky import body_vectors_km, sky_many

    steps = int(DEFAULT.mission_hours / DEFAULT.step_hours) + 1
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    dates = [t0 + timedelta(days=22.83 * k, hours=(7 * k) % 24) for k in range(n_dates)]
    G = np.zeros((len(lat), n_dates), np.float32)
    for j, d in enumerate(dates):
        v = body_vectors_km([d + timedelta(hours=DEFAULT.step_hours * s) for s in range(steps)])
        s = sky_many(lat, lon, v, height_m=elev)
        sc = s["sun_el"] - _horizon_at(horizon, s["sun_az"])
        ec = s["earth_el"] - _horizon_at(horizon, s["earth_az"])
        o = time_outcomes(sc, s["sun_el"], ec, DEFAULT)
        G[:, j] = o["lit0"] * o["energy_ok"] * o["comms_ok"]
    return G


def train_one(variant, seed, d, emb, split, epochs, batch, lr, dev, verbose, patience=12):
    """GPU-resident training: data lives on the GPU, shuffling/augmentation happen there, mixed precision."""
    train, val = split["train"], split["val"]
    use_cnn, use_emb = variant in ("cnn", "both"), variant in ("lfm", "both")
    G = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)  # noqa: E731
    Xt, Y, prior = G(d["Xtn"]), G(d["Y"][:, :2]), G(d["prior"])
    P = G(d["X_patch"]).contiguous(memory_format=torch.channels_last) if use_cnn else None
    E = G(emb) if use_emb else None
    tr_idx = torch.from_numpy(np.flatnonzero(train)).to(dev)
    va_idx = torch.from_numpy(np.flatnonzero(val)).to(dev)
    torch.manual_seed(seed)
    m = TouchdownNet(n_tab=Xt.shape[1], in_ch=d["X_patch"].shape[1], use_cnn=use_cnn, emb_dim=E.shape[1] if use_emb else 0).to(dev)
    m = m.to(memory_format=torch.channels_last)
    steps = max(1, len(tr_idx) // batch)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    w = torch.tensor([1.0, 0.5], device=dev)

    def forward(idx, augment=False):
        p = None
        if use_cnn:
            p = P[idx]
            if augment:
                p = torch.rot90(p, int(torch.randint(0, 4, (1,))), dims=(2, 3))
                if torch.rand(1) < 0.5:
                    p = torch.flip(p, dims=(3,))
        return m(p, Xt[idx], E[idx] if use_emb else None, prior[idx, 0], prior[idx, 1])

    def predict_idx(idx):
        m.eval()
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            out = torch.cat([torch.sigmoid(forward(idx[i : i + 8192]).float()) for i in range(0, len(idx), 8192)])
        return out.cpu().numpy()

    def predict(mask):
        return predict_idx(torch.from_numpy(np.flatnonzero(mask)).to(dev))

    yv = d["Y"][val, 0]
    best, best_state, curve, bad = 1e9, None, [], 0
    for ep in range(epochs):
        m.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        tot = torch.zeros((), device=dev)
        for k in range(steps):
            idx = perm[k * batch : (k + 1) * batch]
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                o = forward(idx, augment=True)
            loss = (bce(o.float(), Y[idx]) * w).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.detach()
        pv = predict_idx(va_idx)
        mae = float(np.mean(np.abs(pv[:, 0] - yv)))
        rho = float(spearmanr(pv[:, 0], yv).statistic)
        curve.append({"epoch": ep + 1, "train_loss": float(tot) / steps, "val_mae": mae, "val_spearman": rho})
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            print(f"  [{variant} s{seed}] epoch {ep+1:3d} train loss {float(tot)/steps:.4f}  val MAE {mae:.4f}  val ρ {rho:.4f}", flush=True)
        if mae < best - 1e-5:
            best, best_state, bad = mae, {k: v.detach().float().cpu().clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    return m, predict, best, curve


def main(variants, n_seeds, epochs, batch, lr):
    raw = np.load(OUT)
    d = {k: raw[k] for k in raw.files}
    lon, lat = d["lon"], d["lat"]
    test = (lon >= HOLDOUT_LON[0]) & (lon < HOLDOUT_LON[1])
    val = (lon >= VAL_LON[0]) & (lon < VAL_LON[1])
    train = ~test & ~val
    split = {"train": train, "val": val, "test": test}
    mu, sd = d["X_tab"][train].mean(0), d["X_tab"][train].std(0) + 1e-6
    d["Xtn"] = ((d["X_tab"] - mu) / sd).astype(np.float32)
    i_td, i_dir = TAB.index("phys80_td"), TAB.index("phys80_direct")
    d["prior"] = np.stack([d["X_tab"][:, i_td], d["X_tab"][:, i_dir]], 1).astype(np.float32)
    emb = None
    if any(v in ("lfm", "both") for v in variants):
        e = np.load(EMB)
        key = {(round(float(a), 6), round(float(b), 6)): i for i, (a, b) in enumerate(zip(e["lat"], e["lon"]))}
        rows = np.array([key[(round(float(a), 6), round(float(b), 6))] for a, b in zip(lat, lon)])
        emb = np.concatenate([e["center"], e["global"]], 1).astype(np.float32)[rows]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    Y = d["Y"]
    groups_test = np.random.default_rng(0).integers(0, 40, test.sum())  # random candidate pools for site picking

    results, ensembles, curves, preds = {}, {}, {}, {}
    for v in variants:
        members, pv, pt = [], [], []
        print(f"== variant {v} ({VARIANT_LABEL[v]})", flush=True)
        for s in range(n_seeds):
            m, predict, best, curve = train_one(v, s, d, emb, split, epochs, batch, lr, dev, verbose=(s == 0))
            members.append({k: t.float().cpu() for k, t in m.state_dict().items()})
            print(f"  seed {s}: stopped after {len(curve)} epochs", flush=True)
            pv.append(predict(val)[:, 0])
            pt.append(predict(test))
            curves.setdefault(v, []).append(curve)
            print(f"  seed {s}: best val MAE {best:.4f}", flush=True)
        pv, pt = np.mean(pv, 0), np.stack(pt)
        mean_t, std_t = pt[:, :, 0].mean(0), pt[:, :, 0].std(0)
        r = td_metrics(mean_t, Y[test, 0], groups_test)
        r["val_mae"] = float(np.mean(np.abs(pv - Y[val, 0])))
        r["val_spearman"] = float(spearmanr(pv, Y[val, 0]).statistic)
        r["uncertainty_error_spearman"] = float(spearmanr(std_t, np.abs(mean_t - Y[test, 0])).statistic)
        results[v], ensembles[v], preds[v] = r, members, (mean_t, std_t)
        print(f"  {v}: val MAE {r['val_mae']:.4f} | test MAE {r['touchdown_mae']:.4f} ρ {r['touchdown_spearman']:.4f} safe-AUC {r['safe_site_auc']:.4f}", flush=True)

    chosen = min(variants, key=lambda v: results[v]["val_mae"])
    phys = d["prior"][test, 0]
    rule = d["rule"][test] / 100.0
    base = {
        "physics_80m_baseline": td_metrics(phys, Y[test, 0], groups_test),
        "rule_based_score": td_metrics(rule, Y[test, 0], groups_test),
    }
    base["rule_based_score"]["touchdown_mae"] = None  # a 0-100 score, not a probability
    base["rule_based_score"]["touchdown_brier"] = None

    # mission level: test sites x 48 dates, deterministic light/power/comms gates
    G = gates_for_dates(lat[test], lon[test], d["elev"][test], d["horizon"][test])
    nd = G.shape[1]
    # pools = viable (lit, powered, in contact) site-dates for the same touchdown date: the AI must
    # pick the safest of them, as a planner would after the physics checks
    groups = np.repeat(np.arange(nd)[None], test.sum(), 0).ravel()
    viable = G.ravel() > 0
    truth_m = np.repeat(Y[test, 0][:, None], nd, 1).ravel()[viable]
    mission = {}
    for name, p in (("ai_model", preds[chosen][0]), ("physics_80m_baseline", phys), ("rule_based_score", rule)):
        mission[name] = pick_quality(np.repeat(p[:, None], nd, 1).ravel()[viable], truth_m, groups[viable], k=10, require_viable=False)
    mission["share_of_site_dates_viable"] = float((G > 0).mean())

    card = {
        "name": "Plan Banana landing-success model",
        "version": f"td_ensemble_v2_{chosen}",
        "variant": chosen,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "task": "Predict the probability of a safe touchdown at a site from planning-level NASA maps; mission success = AI touchdown × deterministic light, power and Earth-link checks.",
        "labels": "Monte-Carlo landing simulator on LOLA 20 m south-polar terrain: navigation dispersion, hazard-avoidance diverts, slopes against the GRAIL geoid, sub-grid roughness, crew-ops slope limit and low-gravity dynamic tip-over.",
        "design": "Physics-informed residual: the network corrects the same simulator run on the 80 m map, in log-odds. Ensemble of 5 networks → mean prediction + uncertainty.",
        "inputs": {
            "terrain_patch": "LOLA 80 m DEM, 32x32 px (2.5 km): detrended height / slope / curvature (variants cnn, both)",
            "foundation_model": "NASA-IBM Lunar Foundation Model (ViT-B, frozen) embeddings of a 25.6 km DTM/slope/aspect tile (variants lfm, both)",
            "tabular": TAB,
        },
        "foundation_model": {"name": "NASA-IBM Lunar Foundation Model", "repo": "nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model", "license": "Apache-2.0", "used": chosen in ("lfm", "both")},
        "simulator": DEFAULT.to_dict(),
        "data": {
            "samples": int(len(Y)), "train": int(train.sum()), "val": int(val.sum()), "test": int(test.sum()),
            "split": f"geographic: test = longitudes {HOLDOUT_LON[0]:g}..{HOLDOUT_LON[1]:g}°E, val = {VAL_LON[0]:g}..{VAL_LON[1]:g}°E, never seen in training",
            "region": "south polar cap, latitudes -80.6..-90",
            "mean_true_touchdown": float(Y[:, 0].mean()),
        },  # fmt: skip
        "results_test": {"ai_model": results[chosen], **base},
        "mission_test": mission,
        "variants_test": {v: {**results[v], "label": VARIANT_LABEL[v]} for v in variants},
        "calibration_test": calibration(preds[chosen][0], Y[test, 0]),
        "training_curves": {v: [{k: round(x[k], 5) for k in x} for x in curves[v][0]] for v in variants},
        "limitations": [
            "Trained on the lunar south polar region only; elsewhere it is extrapolating.",
            "Truth is LOLA 20 m terrain plus a statistical sub-grid roughness term; individual boulders are not mapped.",
            "Lander, navigation and hazard-avoidance parameters are generic HLS-class assumptions (see simulator).",
            "Light, power and Earth link are computed exactly (ephemeris + terrain horizon), not learned.",
        ],
    }
    torch.save(
        {"members": ensembles[chosen], "mu": mu, "sd": sd, "tab_features": TAB, "variant": chosen, "use_cnn": chosen in ("cnn", "both"),
         "emb_dim": (emb.shape[1] if chosen in ("lfm", "both") else 0), "in_ch": int(d["X_patch"].shape[1]), "version": card["version"],
         "all_variants": {v: {"members": ensembles[v], "use_cnn": v in ("cnn", "both"), "emb_dim": (emb.shape[1] if (emb is not None and v in ("lfm", "both")) else 0)} for v in variants}},
        CKPT,
    )  # fmt: skip
    CARD.write_text(json.dumps(card, indent=1))
    print("deployed (chosen by validation):", chosen)
    print(json.dumps({"touchdown": {k: {m: card["results_test"][k][m] for m in ("touchdown_spearman", "touchdown_mae", "safe_site_auc")} for k in card["results_test"]}, "mission": mission}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="cnn,lfm,both")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    main(a.variants.split(","), a.seeds, a.epochs, a.batch, a.lr)
