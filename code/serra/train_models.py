"""Train RMRNN, lambda=0 ablation and GRU per station and seed; store daily
forecasts, backward-reconstruction defects and path increments.

Output: results/scores/<region>/<station_id>.npz with arrays of length len(DATES):
  yhat_<model>_s<k>   forecast of standardised log precipitation for that date
  defect_<model>_s<k> defect at the window ending on that date (RMRNN models)
  qpath_<model>_s<k>  scalar: mean path increment over evaluation windows
"""
import argparse, os, sys, time
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REF, EVAL, load_region, doy_index, smooth_circular  # noqa

WINDOW, HIDDEN, BATCH, LR, MAX_EPOCHS, PATIENCE, WARMUP = 30, 32, 256, 1e-3, 40, 5, 3
LAM0, LAM1, LAM_PROJ_ABL = 0.1, 0.01, 0.1
SEEDS = 5


def prepare(p):
    """Standardised log1p precipitation, missing -> doy climatology, plus flag."""
    x = np.log1p(np.asarray(p, float))
    miss = ~np.isfinite(x)
    ref = (DATES.year >= REF[0]) & (DATES.year <= REF[1])
    doy = doy_index(DATES)
    clim = np.array([np.nanmean(x[ref & (doy == d)]) for d in range(1, 366)])
    clim = smooth_circular(np.nan_to_num(clim, nan=np.nanmean(clim)), 15)
    xf = np.where(miss, clim[doy - 1], x)
    mu, sd = xf[ref].mean(), xf[ref].std()
    z = (xf - mu) / sd
    target = np.where(miss, np.nan, (x - mu) / sd)
    return np.stack([z, miss.astype(float)], 1).astype(np.float32), target.astype(np.float32)


def run_station(job):
    region, sid, p = job
    out = os.path.join(ROOT, "results", "scores", region, f"{sid}.npz")
    if os.path.exists(out):
        return sid, "cached"
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    feats, target = prepare(p)
    n = len(DATES)
    ends = np.arange(WINDOW - 1, n - 1)                # window end index t; target t+1
    idx = ends[:, None] + np.arange(-WINDOW + 1, 1)[None, :]
    Xall = torch.from_numpy(feats[idx])                 # (N, W, 2)
    yall = target[ends + 1]
    tyear = DATES.year.values[ends + 1]
    tr = (tyear >= REF[0]) & (tyear <= REF[1] - 3) & np.isfinite(yall)
    va = (tyear > REF[1] - 3) & (tyear <= REF[1]) & np.isfinite(yall)
    ev = (DATES.year.values[ends] >= EVAL[0])

    class Proj(nn.Module):
        def __init__(s):
            super().__init__(); s.l1 = nn.Linear(HIDDEN, HIDDEN); s.l2 = nn.Linear(HIDDEN, HIDDEN)
        def forward(s, h):
            return h + s.l2(torch.relu(s.l1(h)))

    class Net(nn.Module):
        def __init__(s, kind):
            super().__init__()
            s.kind = kind
            s.rnn = (nn.GRU if kind == "gru" else nn.RNN)(2, HIDDEN, batch_first=True)
            s.head = nn.Linear(HIDDEN, 1)
            s.proj = Proj() if kind != "gru" else None
        def forward(s, x):
            h, _ = s.rnn(x)
            return h, s.head(h[:, -1]).squeeze(-1)

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
                    loss = ((yh - ytr[bi]) ** 2).mean()
                    if kind == "rmrnn" and lam > 0:
                        rm = ((h[:, :-1] - net.proj(h[:, 1:])) ** 2).sum(-1).mean()
                        loss = loss + lam * rm
                    elif kind == "lam0":
                        hd = h.detach()
                        rm = ((hd[:, :-1] - net.proj(hd[:, 1:])) ** 2).sum(-1).mean()
                        loss = loss + LAM_PROJ_ABL * rm
                    elif kind == "rmrnn":   # warm-up: still fit projector on detached states
                        hd = h.detach()
                        loss = loss + LAM_PROJ_ABL * ((hd[:, :-1] - net.proj(hd[:, 1:])) ** 2).sum(-1).mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                net.eval()
                with torch.no_grad():
                    vm = float(((net(Xva)[1] - yva) ** 2).mean())
                if kind == "rmrnn" and ep < WARMUP:
                    continue          # checkpoints are eligible only once the penalty is active
                if vm < best - 1e-5:
                    best, bad = vm, 0
                    best_state = {kk: v.clone() for kk, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            net.load_state_dict(best_state); net.eval()
            yhat = np.full(n, np.nan, np.float32); dfc = np.full(n, np.nan, np.float32)
            qs = []
            with torch.no_grad():
                for b in range(0, len(ends), 4096):
                    h, yh = net(Xall[b:b + 4096])
                    yhat[ends[b:b + 4096] + 1] = yh.numpy()
                    qs.append(torch.linalg.norm(h[:, 1:] - h[:, :-1], dim=-1).sum(1).numpy())
                    if net.proj is not None:
                        dfc[ends[b:b + 4096]] = torch.linalg.norm(h[:, -2] - net.proj(h[:, -1]), dim=-1).numpy()
            q = np.concatenate(qs)
            res[f"yhat_{kind}_s{k}"] = yhat
            res[f"qpath_{kind}_s{k}"] = np.float32(q[ev].mean())
            res[f"valmse_{kind}_s{k}"] = np.float32(best)
            res[f"epochs_{kind}_s{k}"] = np.int32(ep + 1)
            if net.proj is not None:
                res[f"defect_{kind}_s{k}"] = dfc
    res["target"] = target
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **res)
    return sid, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="texas,germany,australia,taiwan")
    ap.add_argument("--procs", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    jobs = []
    for r in a.regions.split(","):
        meta, P = load_region(r)
        sids = list(meta.station_id)[: a.limit or None]
        jobs += [(r, s, P[s].values) for s in sids]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for i, (sid, st) in enumerate(pool.imap_unordered(run_station, jobs), 1):
            print(f"[{i}/{len(jobs)}] {sid} {st} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
