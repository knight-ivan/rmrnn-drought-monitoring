"""Small intermediates for the post hoc checks of Amendment 3 (review_checks.py), so that
those checks run without the raw downloads and the per-seed score files.

Output: results/checks/intermediates.npz with, for every region,
  <region>_panel_ids, <region>_panel_mv_ids   station order of the score panels
  <region>_sm                                  raw MERRA-2 root-zone soil wetness (GWETROOT),
                                               [station, day 1961-2024], NaN before 1981
  <region>_scores_sse, <region>_scores_mv_sse  squared forecast error summed by calendar year
                                               2001-2024: [station, model (regularised, lambda=0,
                                               GRU), seed, year]
  <region>_scores_n, <region>_scores_mv_n      number of evaluated days [station, year]
Needs data/<region>/power/ and results/scores*/ (not in the archive).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS, EVAL

yr = DATES.year.values
ev = (yr >= EVAL[0]) & (yr <= EVAL[1]); NY = EVAL[1] - EVAL[0] + 1
out = {}
for r in REGIONS:
    for sub, sc in [("panel", "scores"), ("panel_mv", "scores_mv")]:
        sids = [str(s) for s in np.load(os.path.join(ROOT, "results", sub, f"{r}.npz"))["station_id"]]
        out[f"{r}_{sub}_ids"] = np.array(sids)
        if sub == "panel":
            sm = np.full((len(sids), len(DATES)), np.nan, np.float32)
            for i, s in enumerate(sids):
                pw = pd.read_csv(os.path.join(ROOT, "data", r, "power", f"{s}.csv"),
                                 parse_dates=["date"]).set_index("date").reindex(DATES)
                sm[i] = pw["GWETROOT"].values
            out[f"{r}_sm"] = sm
        sse = np.full((len(sids), 3, 5, NY), np.nan); n = np.zeros((len(sids), NY))
        for i, s in enumerate(sids):
            z = np.load(os.path.join(ROOT, "results", sc, r, f"{s}.npz")); y = z["target"]; m = ev & np.isfinite(y)
            yy = yr[m] - EVAL[0]; n[i] = np.bincount(yy, minlength=NY)
            for a, mod in enumerate(["rmrnn", "lam0", "gru"]):
                for k in range(5):
                    sse[i, a, k] = np.bincount(yy, weights=(z[f"yhat_{mod}_s{k}"][m] - y[m]) ** 2, minlength=NY)
        out[f"{r}_{sc}_sse"] = sse; out[f"{r}_{sc}_n"] = n
    print(r, "done", flush=True)
np.savez_compressed(os.path.join(ROOT, "results", "checks", "intermediates.npz"), **out)
