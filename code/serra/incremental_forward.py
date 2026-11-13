"""Prospective validation of the incremental-information analysis
(PROTOCOL.md, Amendment 2; post hoc).

covariate_analysis.py (pre-specified) scores the weekly logistic models for
"a reference drought begins within 60 days" by five-fold cross-validation over
contiguous blocks of years. In that scheme the training folds include years that
FOLLOW the test block, so its AUC measures retrospective discrimination, not
forecasting skill. This script repeats the comparison under schemes in which
the model that predicts a week never sees later data:

  blocked          pre-specified scheme (reproduces incremental.csv on 1982-2024)
  blocked_purged   same folds; training weeks within GAP days of a test block,
                   on either side, are removed
  forward          forward chaining, expanding window, refitted every year:
                   weeks of year Y are predicted by a model fitted to weeks
                   t < (1 January Y) - GAP, so that every training label
                   (onset in (t, t + 60]) was known before year Y began
  forward_6y       as forward, refitted every six years (2001, 2007, 2013, 2019)

GAP equals the 60-day prediction horizon. The forward schemes are scored on
2001-2024, the evaluation period of the paper (training always starts in 1982);
the blocked schemes are scored on 2001-2024 as well, and `blocked` also on
1982-2024 as pre-specified. Features are prospective by construction: the
networks and all reference distributions are fitted on 1961-1990.

Intervals: region-year cluster bootstrap (1000 resamples) of the out-of-sample
predictions, as in covariate_analysis.py.

Output: results/covariates[_mv]/incremental_forward.csv
Usage:  python3 code/serra/incremental_forward.py [--mv]
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS, EVAL

MV = "--mv" in sys.argv
HORIZON = GAP = 60
NBOOT = 1000
PAN = os.path.join(ROOT, "results", "panel_mv" if MV else "panel")
OUT = os.path.join(ROOT, "results", "covariates_mv" if MV else "covariates")
if MV:
    COLS = {"spei90": "D11", "spei30": "D10", "spi30": "D7"}; MEANS = ["D8", "D9", "D1"]
    FEATS = {"spei90": ["spei90"], "spei90+D8": ["spei90", "D8"],
             "base": ["spei90", "spei30", "spi30"], "base+D8": ["spei90", "spei30", "spi30", "D8"],
             "base+D9": ["spei90", "spei30", "spi30", "D9"], "base+D1": ["spei90", "spei30", "spi30", "D1"]}
    COMPS = [("spei90+D8", "spei90"), ("base+D8", "base"), ("base+D9", "base"), ("base+D1", "base")]
    SEED = 11
else:
    COLS = {"spi90": "D3", "spi30": "D7"}; MEANS = ["D1", "D2", "D5"]
    FEATS = {"base": ["spi90"], "D1": ["spi90", "D1"], "D2": ["spi90", "D2"], "D5": ["spi90", "D5"],
             "SPI30": ["spi90", "spi30"], "SPI30+D1": ["spi90", "spi30", "D1"]}
    COMPS = [("D1", "base"), ("D2", "base"), ("D5", "base"), ("SPI30", "base"), ("SPI30+D1", "SPI30")]
    SEED = 7
T0 = int(np.searchsorted(DATES, pd.Timestamp("1982-01-01")))
YEAR_START = {y: int(np.searchsorted(DATES, pd.Timestamp(f"{y}-01-01"))) for y in range(1982, 2026)}


def weekly(r):
    """Weekly design matrix of one region (identical to covariate_analysis*.py, plus the day index t)."""
    pn = np.load(os.path.join(PAN, f"{r}.npz"))
    ev = pd.read_csv(os.path.join(PAN, f"{r}_events.csv"), dtype={"station_id": str})
    t = np.arange(T0, len(DATES) - HORIZON, 7)
    rows = []
    for i, s in enumerate(pn["station_id"]):
        inside = np.zeros(len(DATES), bool); soon = np.zeros(len(DATES), bool)
        for _, x in ev[ev.station_id == s].iterrows():
            inside[x.onset: x.end + 1] = True
            soon[max(0, x.onset - HORIZON): x.onset] = True
        d = {"y": soon[t].astype(int)}
        d.update({k: pn[v][i][t] for k, v in COLS.items()})
        d.update({k: pd.Series(pn[k][i]).rolling(30, min_periods=20).mean().values[t] for k in MEANS})
        d = pd.DataFrame(d).assign(year=DATES[t].year, t=t, inside=inside[t], region=r)
        rows.append(d[~d.inside].dropna())
    return pd.concat(rows, ignore_index=True)


def fit_predict(d, tr, te, cols):
    return LogisticRegression(max_iter=1000).fit(d.loc[tr, cols], d.y[tr]).predict_proba(d.loc[te, cols])[:, 1]


def predictions(d, scheme):
    """Out-of-sample predictions of every model under `scheme` (NaN where a week is never predicted)."""
    pr = {k: np.full(len(d), np.nan) for k in FEATS}
    t, yr = d.t.values, d.year.values
    if scheme in ("blocked", "blocked_purged"):
        for fy in np.array_split(np.sort(d.year.unique()), 5):
            te = np.isin(yr, fy); tr = ~te
            if scheme == "blocked_purged":
                a, b = YEAR_START[int(fy[0])], YEAR_START[int(fy[-1]) + 1] - 1
                tr &= ~((t >= a - GAP) & (t <= b + GAP))
            for k, c in FEATS.items():
                pr[k][te] = fit_predict(d, tr, te, c)
    else:
        step = 1 if scheme == "forward" else 6
        for y0 in range(EVAL[0], EVAL[1] + 1, step):
            te = (yr >= y0) & (yr < y0 + step); tr = t < YEAR_START[y0] - GAP
            for k, c in FEATS.items():
                pr[k][te] = fit_predict(d, tr, te, c)
    return pr


def auc_fast(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(d, pr, sel, rng, scheme, period, group):
    y = d.y.values[sel]; P = {k: v[sel] for k, v in pr.items()}
    cl = (d.region + "_" + d.year.astype(str)).values[sel]
    uc = np.unique(cl); ib = {c: np.flatnonzero(cl == c) for c in uc}
    est = {k: (roc_auc_score(y, P[k]), float(np.mean((P[k] - y) ** 2))) for k in FEATS}
    boots = []
    for _ in range(NBOOT):
        s = np.concatenate([ib[c] for c in rng.choice(uc, len(uc))]); yy = y[s]
        if yy.min() == yy.max():
            continue
        boots.append({k: (auc_fast(yy, P[k][s]), np.mean((P[k][s] - yy) ** 2)) for k in FEATS})
    res = []
    for a, b in COMPS:
        da = np.array([x[a][0] - x[b][0] for x in boots]); db = np.array([x[a][1] - x[b][1] for x in boots])
        res.append(dict(scheme=scheme, period=period, gap_days=0 if scheme == "blocked" else GAP, group=group,
                        model=a, reference=b, n=int(len(y)), prevalence=float(y.mean()),
                        auc_ref=est[b][0], auc_model=est[a][0], dauc=est[a][0] - est[b][0],
                        dauc_lo=np.percentile(da, 2.5), dauc_hi=np.percentile(da, 97.5),
                        brier_ref=est[b][1], dbrier=est[a][1] - est[b][1],
                        dbrier_lo=np.percentile(db, 2.5), dbrier_hi=np.percentile(db, 97.5)))
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    D = {r: weekly(r) for r in REGIONS}
    D["all"] = pd.concat([D[r] for r in REGIONS], ignore_index=True)
    res = []
    # pass 1: pre-specified scheme on 1982-2024, same random stream as covariate_analysis*.py
    rng = np.random.default_rng(SEED); PR = {}
    for g in REGIONS + ["all"]:
        PR[(g, "blocked")] = predictions(D[g], "blocked")
        res += evaluate(D[g], PR[(g, "blocked")], np.ones(len(D[g]), bool), rng, "blocked", "1982-2024", g)
        print(g, "blocked (pre-specified) done", flush=True)
    # pass 2: all schemes on 2001-2024
    rng = np.random.default_rng(SEED + 1000)
    for g in REGIONS + ["all"]:
        ev = (D[g].year >= EVAL[0]).values
        for scheme in ["blocked", "blocked_purged", "forward", "forward_6y"]:
            pr = PR[(g, scheme)] if (g, scheme) in PR else predictions(D[g], scheme)
            res += evaluate(D[g], pr, ev, rng, scheme, "2001-2024", g)
        print(g, "forward-chaining done", flush=True)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "incremental_forward.csv"), index=False)
    # check against the published table
    ref = pd.read_csv(os.path.join(OUT, "incremental.csv"))
    mine = R[(R.scheme == "blocked") & (R.period == "1982-2024")]
    key = "model" if MV else "added"
    m = ref.merge(mine, left_on=["group", key], right_on=["group", "model"], suffixes=("_ref", ""))
    err = max(float(np.abs(m[c] - m[c + "_ref"]).max()) for c in ["dauc", "dauc_lo", "dauc_hi"])
    print(f"reproduction check: max abs difference from incremental.csv {err:.2e} ({len(m)} rows)")
    assert len(m) == len(ref) and err < 1e-6


if __name__ == "__main__":
    main()
