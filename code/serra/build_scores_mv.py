"""Panels for the multivariate analysis (PROTOCOL amendment 1), restricted to the
stations with temperature. Adds D8 (multivariate RM defect), D9 (its lambda=0
ablation), D10 (daily SPEI-30), D11 (daily SPEI-90), D12 (multivariate P residual,
stored for completeness), and copies D1, D3, D7 and all covariates from the
univariate panel for the same stations.
Output: results/panel_mv/<region>.npz, results/panel_mv/<region>_events.csv
"""
import os, sys
import numpy as np
import pandas as pd
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS, load_region, deseasonalise, spei_daily
from train_models_mv import load_temperature, stations_with_temperature

SEEDS = 5


def one(job):
    region, sid, p, pet = job
    z = np.load(os.path.join(ROOT, "results", "scores_mv", region, f"{sid}.npz"))
    out = {}
    for name, key in [("D8", "rmrnn"), ("D9", "lam0")]:
        out[name] = np.nanmean([deseasonalise(np.log(z[f"defect_{key}_s{k}"] + 1e-8), DATES) for k in range(SEEDS)], 0)
    out["D10"] = spei_daily(p, pet, DATES, 30)
    out["D11"] = spei_daily(p, pet, DATES, 90)
    yh = np.nanmean([z[f"yhat_rmrnn_s{k}"] for k in range(SEEDS)], 0)
    out["D12"] = deseasonalise(z["target"] - yh, DATES)
    return sid, out


def main():
    os.makedirs(os.path.join(ROOT, "results", "panel_mv"), exist_ok=True)
    for r in (sys.argv[1].split(",") if len(sys.argv) > 1 else REGIONS):
        meta, P = load_region(r)
        T = load_temperature(r)
        keep = set(stations_with_temperature(r))
        sids = [s for s in meta.station_id if s in keep]
        with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", 12))) as pool:
            res = dict(pool.map(one, [(r, s, P[s].values, T[s].pet.values) for s in sids]))
        uni = np.load(os.path.join(ROOT, "results", "panel", f"{r}.npz"))
        ui = {s: i for i, s in enumerate(uni["station_id"])}
        arr = {k: np.stack([res[s][k] for s in sids]).astype(np.float32) for k in ["D8", "D9", "D10", "D11", "D12"]}
        for k in uni.files:
            if k != "station_id":
                arr[k] = np.stack([uni[k][ui[s]] for s in sids])
        np.savez_compressed(os.path.join(ROOT, "results", "panel_mv", f"{r}.npz"), station_id=np.array(sids), **arr)
        ev = pd.read_csv(os.path.join(ROOT, "results", "panel", f"{r}_events.csv"), dtype={"station_id": str})
        ev = ev[ev.station_id.isin(keep)]
        ev.to_csv(os.path.join(ROOT, "results", "panel_mv", f"{r}_events.csv"), index=False)
        print(r, "stations", len(sids), "events 2001-2024", int((pd.to_datetime(ev.onset_date).dt.year >= 2001).sum()), flush=True)


if __name__ == "__main__":
    main()
