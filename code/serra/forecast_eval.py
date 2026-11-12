"""Forecast evaluation on 2001-2024 (PROTOCOL.md): MSE, MAE, MSE skill vs
day-of-year climatology, AUC for exceedance of the station's 1961-1990 P95,
path increment; paired station-level Wilcoxon tests and TOST equivalence.
Output: results/forecast/station_metrics.csv, results/forecast/summary.csv,
        results/forecast/paired.csv
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS, REF, EVAL, year_mask, doy_index, smooth_circular, load_region

MODELS, SEEDS, MARGIN = ["rmrnn", "lam0", "gru"], 5, 0.02
OUT = os.path.join(ROOT, "results", "forecast")
SCORES = "scores"
if "--mv" in sys.argv:   # PROTOCOL amendment 1
    OUT = os.path.join(ROOT, "results", "forecast_mv"); SCORES = "scores_mv"


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    ev, ref = year_mask(DATES, EVAL), year_mask(DATES, REF)
    doy = doy_index(DATES)
    for r in REGIONS:
        meta, P = load_region(r)
        for s in meta.station_id:
            f = os.path.join(ROOT, "results", SCORES, r, f"{s}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f); y = z["target"]
            clim = np.array([np.nanmean(y[ref & (doy == d)]) for d in range(1, 366)])
            clim = smooth_circular(clim, 15)[doy - 1]
            p = P[s].values
            p95 = np.nanpercentile(p[ref], 95)
            m = ev & np.isfinite(y)
            evt = (p[m] >= p95).astype(int)
            mse_c = np.mean((y[m] - clim[m]) ** 2)
            for mod in MODELS:
                for k in range(SEEDS):
                    yh = z[f"yhat_{mod}_s{k}"][m]
                    mse = np.mean((yh - y[m]) ** 2)
                    rows.append(dict(region=r, station_id=s, model=mod, seed=k, mse=mse,
                                     mae=np.mean(np.abs(yh - y[m])), msss=1 - mse / mse_c,
                                     auc=roc_auc_score(evt, yh) if 0 < evt.sum() < len(evt) else np.nan,
                                     qpath=float(z[f"qpath_{mod}_s{k}"])))
    S = pd.DataFrame(rows); S.to_csv(os.path.join(OUT, "station_metrics.csv"), index=False)
    st = S.groupby(["region", "station_id", "model"])[["mse", "mae", "msss", "auc", "qpath"]].mean().reset_index()
    summ = st.groupby(["region", "model"]).agg(n=("station_id", "nunique"),
                                               **{f"{c}_{f}": (c, f) for c in ["mse", "mae", "msss", "auc", "qpath"]
                                                  for f in ["mean", "std"]}).reset_index()
    allr = st.groupby("model").agg(n=("station_id", "nunique"),
                                   **{f"{c}_{f}": (c, f) for c in ["mse", "mae", "msss", "auc", "qpath"]
                                      for f in ["mean", "std"]}).reset_index().assign(region="all")
    pd.concat([summ, allr]).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pr = []
    W = st.pivot_table(index=["region", "station_id"], columns="model")
    for grp in list(REGIONS) + ["all"]:
        w = W if grp == "all" else W.loc[[grp]] if grp in W.index.get_level_values(0) else None
        if w is None:
            continue
        for other in ["lam0", "gru"]:
            for met in ["mse", "mae", "auc", "qpath"]:
                a, b = w[(met, "rmrnn")].values, w[(met, other)].values
                rel = (a - b) / b
                d = dict(group=grp, comp=f"rmrnn-{other}", metric=met, n=len(a),
                         median_rel_diff=float(np.median(rel)), mean_rel_diff=float(np.mean(rel)),
                         wilcoxon_p=float(stats.wilcoxon(a, b).pvalue) if len(a) > 5 else np.nan)
                if met == "mse":   # TOST on relative MSE difference, margin +/-2%
                    se = rel.std(ddof=1) / np.sqrt(len(rel))
                    p_lo = 1 - stats.t.cdf((rel.mean() + MARGIN) / se, len(rel) - 1)
                    p_hi = stats.t.cdf((rel.mean() - MARGIN) / se, len(rel) - 1)
                    d.update(tost_p=float(max(p_lo, p_hi)),
                             ci90_lo=float(rel.mean() - stats.t.ppf(.95, len(rel) - 1) * se),
                             ci90_hi=float(rel.mean() + stats.t.ppf(.95, len(rel) - 1) * se))
                pr.append(d)
    pd.DataFrame(pr).to_csv(os.path.join(OUT, "paired.csv"), index=False)
    print(pd.concat([summ, allr]).round(3).to_string())


if __name__ == "__main__":
    main()
