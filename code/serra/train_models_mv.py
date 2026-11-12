"""Multivariate (temperature-aware) networks, PROTOCOL amendment 1.

Inputs: standardised log(1+P), P-missing flag, standardised day-of-year anomalies of
Tmax, Tmin and Hargreaves PET, temperature-missing flag (missing values filled with
day-of-year climatology, i.e. anomaly 0). Targets: next-day standardised log(1+P) and
next-day PET anomaly, equal weight. Same architecture, training rule, ablation and
seeds as train_models.py.
Output: results/scores_mv/<region>/<station_id>.npz
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REF, EVAL, load_region, deseasonalise  # noqa
from train_models import prepare as prepare_p
from train_models import WINDOW, HIDDEN, BATCH, LR, MAX_EPOCHS, PATIENCE, WARMUP, LAM0, LAM1, LAM_PROJ_ABL, SEEDS


def load_temperature(region):
    T = pd.read_csv(os.path.join(ROOT, "data", region, "temperature_long.csv"), dtype={"station_id": str})
    T["date"] = pd.to_datetime(T["date"])
    return {s: g.set_index("date").reindex(DATES)[["tmax", "tmin", "pet"]] for s, g in T.groupby("station_id")}


def prepare(p, temp):
    fp, tp = prepare_p(p)                                   # (n,2), (n,)
    an = {c: deseasonalise(temp[c].values, DATES) for c in ["tmax", "tmin", "pet"]}
    miss_t = ~(np.isfinite(an["tmax"]) & np.isfinite(an["tmin"]) & np.isfinite(an["pet"]))
    cols = [np.nan_to_num(an[c], nan=0.0) for c in ["tmax", "tmin", "pet"]]
    feats = np.column_stack([fp, *cols, miss_t.astype(float)]).astype(np.float32)
    targets = np.column_stack([tp, an["pet"]]).astype(np.float32)   # NaN where missing
    return feats, targets


def run_station(job):
    region, sid, p, temp = job
    out = os.path.join(ROOT, "results", "scores_mv", region, f"{sid}.npz")
    if os.path.exists(out):
        return sid, "cached"
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    feats, target = prepare(p, temp)
    n = len(DATES)
    ends = np.arange(WINDOW - 1, n - 1)
    idx = ends[:, None] + np.arange(-WINDOW + 1, 1)[None, :]
    Xall = torch.from_numpy(feats[idx])
    yall = target[ends + 1]
    tyear = DATES.year.values[ends + 1]
    okp = np.isfinite(yall[:, 0])
    tr = (tyear >= REF[0]) & (tyear <= REF[1] - 3) & okp
    va = (tyear > REF[1] - 3) & (tyear <= REF[1]) & okp
    ev = (DATES.year.values[ends] >= EVAL[0])

    class Proj(nn.Module):
        def __init__(s):
            super().__init__(); s.l1 = nn.Linear(HIDDEN, HIDDEN); s.l2 = nn.Linear(HIDDEN, HIDDEN)
        def forward(s, h):
            return h + s.l2(torch.relu(s.l1(h)))

    class Net(nn.Module):
        def __init__(s, kind):
            super().__init__()
            s.rnn = (nn.GRU if kind == "gru" else nn.RNN)(feats.shape[1], HIDDEN, batch_first=True)
            s.head = nn.Linear(HIDDEN, 2)
            s.proj = Proj() if kind != "gru" else None
        def forward(s, x):
            h, _ = s.rnn(x)
            return h, s.head(h[:, -1])

    def tloss(yh, y):
        m = torch.isfinite(y)
        d = torch.where(m, yh - torch.nan_to_num(y), torch.zeros_like(yh))
        return (d ** 2).sum(0) / m.sum(0).clamp(min=1)     # per-target MSE

    res = {}
    for kind in ["rmrnn", "lam0", "gru"]:
        for k in range(SEEDS):
            seed = 1000 * k + int(sid[-4:]) if sid[-4:].isdigit() else 1000 * k
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            net = Net(kind)
            opt = torch.optim.Adam(net.parameters(), lr=LR)
            Xtr, ytr = Xall[tr], torch.from_numpy(yall[tr])
            Xva, yva = Xall[va], torch.from_numpy(yall[va])
            best, best_state, bad = np.inf, None, 0
            for ep in range(MAX_EPOCHS):
                if kind == "rmrnn":
                    frac = max(0, ep - WARMUP) / max(MAX_EPOCHS - WARMUP - 1, 1)
                    lam = 0.0 if ep < WARMUP else LAM0 * (LAM1 / LAM0) ** frac
                net.train()
                perm = rng.permutation(len(ytr))
                for b in range(0, len(perm), BATCH):
                    bi = torch.from_numpy(perm[b:b + BATCH])
                    h, yh = net(Xtr[bi])
                    loss = tloss(yh, ytr[bi]).mean()
                    if kind == "rmrnn" and lam > 0:
                        loss = loss + lam * ((h[:, :-1] - net.proj(h[:, 1:])) ** 2).sum(-1).mean()
                    elif kind in ("lam0", "rmrnn"):
                        hd = h.detach()
                        loss = loss + LAM_PROJ_ABL * ((hd[:, :-1] - net.proj(hd[:, 1:])) ** 2).sum(-1).mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                net.eval()
                with torch.no_grad():
                    vm = float(tloss(net(Xva)[1], yva).mean())
                if kind == "rmrnn" and ep < WARMUP:
                    continue
                if vm < best - 1e-5:
                    best, bad = vm, 0
                    best_state = {kk: v.clone() for kk, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            net.load_state_dict(best_state); net.eval()
            yhat = np.full(n, np.nan, np.float32); yhat_pet = np.full(n, np.nan, np.float32)
            dfc = np.full(n, np.nan, np.float32); qs = []
            with torch.no_grad():
                for b in range(0, len(ends), 4096):
                    h, yh = net(Xall[b:b + 4096])
                    yhat[ends[b:b + 4096] + 1] = yh[:, 0].numpy(); yhat_pet[ends[b:b + 4096] + 1] = yh[:, 1].numpy()
                    qs.append(torch.linalg.norm(h[:, 1:] - h[:, :-1], dim=-1).sum(1).numpy())
                    if net.proj is not None:
                        dfc[ends[b:b + 4096]] = torch.linalg.norm(h[:, -2] - net.proj(h[:, -1]), dim=-1).numpy()
            q = np.concatenate(qs)
            res[f"yhat_{kind}_s{k}"] = yhat; res[f"yhatpet_{kind}_s{k}"] = yhat_pet
            res[f"qpath_{kind}_s{k}"] = np.float32(q[ev].mean())
            res[f"valmse_{kind}_s{k}"] = np.float32(best); res[f"epochs_{kind}_s{k}"] = np.int32(ep + 1)
            if net.proj is not None:
                res[f"defect_{kind}_s{k}"] = dfc
    res["target"] = target[:, 0]; res["target_pet"] = target[:, 1]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **res)
    return sid, "done"


def stations_with_temperature(region):
    m = pd.read_csv(os.path.join(ROOT, "data", region, "temperature_meta.csv"), dtype={"station_id": str, "source_id": str})
    return list(m[m.source_id.notna() & (m.source_id != "")].station_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="texas,germany,australia,taiwan")
    ap.add_argument("--procs", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    jobs = []
    for r in a.regions.split(","):
        meta, P = load_region(r)
        T = load_temperature(r)
        sids = [s for s in meta.station_id if s in set(stations_with_temperature(r))][: a.limit or None]
        jobs += [(r, s, P[s].values, T[s]) for s in sids]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for i, (sid, st) in enumerate(pool.imap_unordered(run_station, jobs), 1):
            print(f"[{i}/{len(jobs)}] {sid} {st} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
