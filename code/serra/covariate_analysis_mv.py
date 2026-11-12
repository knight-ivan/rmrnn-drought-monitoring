"""PROTOCOL amendment 1, secondary analyses on the multivariate subset.
(a) Association of the D8 (and D9) monthly anomaly with SPI-3, soil wetness, T2m, ET.
(b) Incremental value for onset within 60 days: base = SPEI-90 + SPEI-30 + SPI-30;
    added = 30-day mean of D8 (and D9 for comparison); also base SPEI-90 alone + D8.
Output: results/covariates_mv/association.csv, results/covariates_mv/incremental.csv
"""
import os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS

OUT = os.path.join(ROOT, "results", "covariates_mv")
PAN = os.path.join(ROOT, "results", "panel_mv")
HORIZON, NBOOT = 60, 1000


def monthly(r):
    pn = np.load(os.path.join(PAN, f"{r}.npz")); mi = pd.PeriodIndex(DATES, freq="M"); rows = []
    for i, s in enumerate(pn["station_id"]):
        df = pd.DataFrame({k: pn[k][i] for k in ["D8", "D9", "sm_a", "t2m_a", "et_a"]}, index=DATES)
        g = df.groupby(mi).mean()
        g["spi3"] = pd.Series(pn["D4"][i], index=DATES).groupby(mi).last()
        g["station_id"] = f"{r}_{s}"; g["ym"] = f"{r}_" + g.index.astype(str)
        rows.append(g[g.index.year >= 1981])
    return pd.concat(rows)


def association():
    res = []
    for g in REGIONS + ["all"]:
        d = pd.concat([monthly(x) for x in REGIONS]) if g == "all" else monthly(g)
        d = d.dropna(subset=["D8", "D9", "spi3", "sm_a", "t2m_a", "et_a"])
        for c in ["spi3", "sm_a", "t2m_a", "et_a", "D8", "D9"]:
            d[c] = (d[c] - d[c].mean()) / d[c].std()
        for y in ["D8", "D9"]:
            m = smf.ols(f"{y} ~ spi3 + sm_a + t2m_a + et_a + C(station_id)", d).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ym)[0]})
            dm = d.groupby("station_id")[y].transform("mean")
            r2w = 1 - (m.resid ** 2).sum() / ((d[y] - dm) ** 2).sum()
            for v in ["spi3", "sm_a", "t2m_a", "et_a"]:
                ci = m.conf_int().loc[v]
                res.append(dict(group=g, outcome=y, term=v, coef=m.params[v], lo=ci[0], hi=ci[1],
                                p=m.pvalues[v], n=int(m.nobs), r2_within=r2w))
    pd.DataFrame(res).to_csv(os.path.join(OUT, "association.csv"), index=False)


def weekly(r):
    pn = np.load(os.path.join(PAN, f"{r}.npz"))
    ev = pd.read_csv(os.path.join(PAN, f"{r}_events.csv"), dtype={"station_id": str})
    t = np.arange(np.searchsorted(DATES, pd.Timestamp("1982-01-01")), len(DATES) - HORIZON, 7)
    rows = []
    for i, s in enumerate(pn["station_id"]):
        e = ev[ev.station_id == s]
        inside = np.zeros(len(DATES), bool); soon = np.zeros(len(DATES), bool)
        for _, x in e.iterrows():
            inside[x.onset: x.end + 1] = True; soon[max(0, x.onset - HORIZON): x.onset] = True
        m30 = {k: pd.Series(pn[k][i]).rolling(30, min_periods=20).mean().values for k in ["D8", "D9", "D1"]}
        d = pd.DataFrame({"y": soon[t].astype(int), "spei90": pn["D11"][i][t], "spei30": pn["D10"][i][t],
                          "spi30": pn["D7"][i][t], "D8": m30["D8"][t], "D9": m30["D9"][t], "D1": m30["D1"][t],
                          "year": DATES[t].year, "inside": inside[t], "region": r})
        rows.append(d[~d.inside].dropna())
    return pd.concat(rows)


def incremental():
    rng = np.random.default_rng(11)
    feats = {"spei90": ["spei90"], "spei90+D8": ["spei90", "D8"],
             "base": ["spei90", "spei30", "spi30"], "base+D8": ["spei90", "spei30", "spi30", "D8"],
             "base+D9": ["spei90", "spei30", "spi30", "D9"], "base+D1": ["spei90", "spei30", "spi30", "D1"]}
    comps = [("spei90+D8", "spei90"), ("base+D8", "base"), ("base+D9", "base"), ("base+D1", "base")]
    res = []
    for g in REGIONS + ["all"]:
        d = pd.concat([weekly(x) for x in REGIONS]) if g == "all" else weekly(g)
        folds = np.array_split(np.sort(d.year.unique()), 5)
        pr = {k: np.full(len(d), np.nan) for k in feats}
        for fy in folds:
            te = d.year.isin(fy).values
            for k, c in feats.items():
                pr[k][te] = LogisticRegression(max_iter=1000).fit(d.loc[~te, c], d.y[~te]).predict_proba(d.loc[te, c])[:, 1]
        y = d.y.values
        cl = (d.region + "_" + d.year.astype(str)).values
        uc = np.unique(cl); ib = {c: np.flatnonzero(cl == c) for c in uc}
        est = {k: roc_auc_score(y, pr[k]) for k in feats}
        boots = []
        for _ in range(NBOOT):
            sel = np.concatenate([ib[c] for c in rng.choice(uc, len(uc))])
            if y[sel].min() == y[sel].max():
                continue
            boots.append({k: roc_auc_score(y[sel], pr[k][sel]) for k in feats})
        for a, b in comps:
            dd = np.array([x[a] - x[b] for x in boots])
            res.append(dict(group=g, model=a, reference=b, n=len(d), prevalence=float(y.mean()),
                            auc_ref=est[b], auc_model=est[a], dauc=est[a] - est[b],
                            dauc_lo=np.percentile(dd, 2.5), dauc_hi=np.percentile(dd, 97.5)))
        print(g, "done", flush=True)
    pd.DataFrame(res).to_csv(os.path.join(OUT, "incremental.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    association(); incremental()
