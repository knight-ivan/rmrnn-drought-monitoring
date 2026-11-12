"""Assemble daily detector scores, reference soil-moisture events and
covariate anomalies for every station; one parquet-like npz per region.

Output: results/panel/<region>.npz  (arrays [station, day])
  D1 RM defect (lambda>0), D2 defect (lambda=0), D3 daily SPI-90,
  D4 monthly SPI-3 (value on month-end days, NaN elsewhere), D5 forecast residual,
  smpct (root-zone soil-wetness percentile), t2m_a, et_a (deseasonalised anomalies)
and results/panel/<region>_events.csv
"""
import os, sys
import numpy as np
import pandas as pd
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, DATES, REGIONS, load_region, deseasonalise, spi_daily,
                    spi3_monthly, sm_percentile, drought_events)

SEEDS = 5
POWER_REF = (1981, 2010)


def one(job):
    region, sid, p = job
    z = np.load(os.path.join(ROOT, "results", "scores", region, f"{sid}.npz"))
    out = {}
    for name, key in [("D1", "rmrnn"), ("D2", "lam0")]:
        s = [deseasonalise(np.log(z[f"defect_{key}_s{k}"] + 1e-8), DATES) for k in range(SEEDS)]
        out[name] = np.nanmean(s, 0)
    out["D3"] = spi_daily(p, DATES)
    m = spi3_monthly(p, DATES)
    d4 = np.full(len(DATES), np.nan)
    pos = DATES.get_indexer(m.index)
    d4[pos[pos >= 0]] = m.values[pos >= 0]
    out["D4"] = d4
    yh = np.nanmean([z[f"yhat_rmrnn_s{k}"] for k in range(SEEDS)], 0)
    out["D5"] = deseasonalise(z["target"] - yh, DATES)
    pw = pd.read_csv(os.path.join(ROOT, "data", region, "power", f"{sid}.csv"),
                     parse_dates=["date"]).set_index("date").reindex(DATES)
    out["smpct"] = np.full(len(DATES), np.nan)
    sel = DATES.year >= 1981
    out["smpct"][sel] = sm_percentile(pw["GWETROOT"].values[sel], DATES[sel])
    out["t2m_a"] = deseasonalise(pw["T2M"].values, DATES, ref=POWER_REF)
    out["et_a"] = deseasonalise(pw["EVPTRNS"].values, DATES, ref=POWER_REF)
    out["sm_a"] = deseasonalise(pw["GWETROOT"].values, DATES, ref=POWER_REF)
    ev = drought_events(out["smpct"], DATES)
    for e in ev:
        e["station_id"] = sid
    return sid, out, ev


def main():
    os.makedirs(os.path.join(ROOT, "results", "panel"), exist_ok=True)
    regions = sys.argv[1].split(",") if len(sys.argv) > 1 else REGIONS
    for r in regions:
        meta, P = load_region(r)
        jobs = [(r, s, P[s].values) for s in meta.station_id]
        with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", 14))) as pool:
            res = pool.map(one, jobs)
        keys = list(res[0][1])
        arr = {k: np.stack([o[k] for _, o, _ in res]).astype(np.float32) for k in keys}
        np.savez_compressed(os.path.join(ROOT, "results", "panel", f"{r}.npz"),
                            station_id=np.array([s for s, _, _ in res]), **arr)
        ev = pd.DataFrame([e for _, _, es in res for e in es])
        ev.to_csv(os.path.join(ROOT, "results", "panel", f"{r}_events.csv"), index=False)
        print(r, "stations", len(res), "events", len(ev),
              "events 2001-2024", int((pd.to_datetime(ev.onset_date).dt.year >= 2001).sum()), flush=True)


if __name__ == "__main__":
    main()
