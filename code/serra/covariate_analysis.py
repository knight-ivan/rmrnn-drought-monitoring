"""Physical association and incremental value of the defect (Reviewer 2, c.2).

(a) Monthly association, 1981-2024: station-month means of the deseasonalised
    defect score (D1, and D2 for comparison) regressed on SPI-3 and on MERRA-2
    root-zone soil-wetness, 2-m temperature and evapotranspiration anomalies,
    with station fixed effects and standard errors clustered by calendar month
    (year-month), which absorbs cross-station dependence.
(b) Incremental value: weekly logistic models for "a reference drought onset in
    the next 60 days" (days inside an ongoing event excluded) with SPI-90 alone
    vs SPI-90 plus 30-day means of D1, D2 or D5; 5-fold blocked-by-year CV;
    Delta AUC and Delta Brier with year-cluster bootstrap CIs.
Output: results/covariates/association.csv, results/covariates/incremental.csv
"""
import os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS

OUT = os.path.join(ROOT, "results", "covariates")
HORIZON = 60
NBOOT = 1000


def monthly_panel(r):
    pn = np.load(os.path.join(ROOT, "results", "panel", f"{r}.npz"))
    rows = []
    mi = pd.PeriodIndex(DATES, freq="M")
    for i, s in enumerate(pn["station_id"]):
        df = pd.DataFrame({k: pn[k][i] for k in ["D1", "D2", "D5", "sm_a", "t2m_a", "et_a"]}, index=DATES)
        g = df.groupby(mi).mean()
        spi3 = pd.Series(pn["D4"][i], index=DATES).groupby(mi).last()
        g["spi3"] = spi3
        g["station_id"] = s; g["ym"] = g.index.astype(str)
        rows.append(g[g.index.year >= 1981])
    return pd.concat(rows)


def association():
    res = []
    for r in REGIONS + ["all"]:
        if r == "all":
            d = pd.concat([monthly_panel(x).assign(region=x) for x in REGIONS if
                           os.path.exists(os.path.join(ROOT, "results", "panel", f"{x}.npz"))])
            d["station_id"] = d.region + "_" + d.station_id
            d["ym"] = d.region + "_" + d.ym
        else:
            if not os.path.exists(os.path.join(ROOT, "results", "panel", f"{r}.npz")):
                continue
            d = monthly_panel(r)
        d = d.dropna(subset=["D1", "D2", "spi3", "sm_a", "t2m_a", "et_a"])
        # standardise covariates within the analysis sample so coefficients are comparable
        for c in ["spi3", "sm_a", "t2m_a", "et_a", "D1", "D2"]:
            d[c] = (d[c] - d[c].mean()) / d[c].std()
        for y in ["D1", "D2"]:
            for spec, f in [("spi3_only", f"{y} ~ spi3 + C(station_id)"),
                            ("full", f"{y} ~ spi3 + sm_a + t2m_a + et_a + C(station_id)")]:
                m = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ym)[0]})
                for v in ["spi3", "sm_a", "t2m_a", "et_a"]:
                    if v in m.params:
                        ci = m.conf_int().loc[v]
                        res.append(dict(group=r, outcome=y, spec=spec, term=v, coef=m.params[v],
                                        lo=ci[0], hi=ci[1], p=m.pvalues[v], n=int(m.nobs),
                                        r2_within=np.nan))
                # within-station R2 (variation explained beyond station means)
                dm = d.groupby("station_id")[y].transform("mean")
                ss_tot = ((d[y] - dm) ** 2).sum(); ss_res = (m.resid ** 2).sum()
                res[-1]["r2_within"] = 1 - ss_res / ss_tot
    pd.DataFrame(res).to_csv(os.path.join(OUT, "association.csv"), index=False)


def weekly_design(r):
    pn = np.load(os.path.join(ROOT, "results", "panel", f"{r}.npz"))
    ev = pd.read_csv(os.path.join(ROOT, "results", "panel", f"{r}_events.csv"), dtype={"station_id": str})
    rows = []
    t_idx = np.arange(np.searchsorted(DATES, pd.Timestamp("1982-01-01")), len(DATES) - HORIZON, 7)
    for i, s in enumerate(pn["station_id"]):
        e = ev[ev.station_id == s]
        inside = np.zeros(len(DATES), bool); onset_soon = np.zeros(len(DATES), bool)
        for _, x in e.iterrows():
            inside[x.onset: x.end + 1] = True
            onset_soon[max(0, x.onset - HORIZON): x.onset] = True
        m30 = {k: pd.Series(pn[k][i]).rolling(30, min_periods=20).mean().values for k in ["D1", "D2", "D5"]}
        d = pd.DataFrame({"y": onset_soon[t_idx].astype(int), "spi90": pn["D3"][i][t_idx],
                          "spi30": pn["D7"][i][t_idx],
                          "D1": m30["D1"][t_idx], "D2": m30["D2"][t_idx], "D5": m30["D5"][t_idx],
                          "year": DATES[t_idx].year, "inside": inside[t_idx]})
        rows.append(d[~d.inside].dropna())
    return pd.concat(rows)


def incremental():
    rng = np.random.default_rng(7)
    res = []
    for r in REGIONS + ["all"]:
        if r == "all":
            d = pd.concat([weekly_design(x).assign(region=x) for x in REGIONS if
                           os.path.exists(os.path.join(ROOT, "results", "panel", f"{x}.npz"))])
        else:
            if not os.path.exists(os.path.join(ROOT, "results", "panel", f"{r}.npz")):
                continue
            d = weekly_design(r).assign(region=r)
        years = np.sort(d.year.unique())
        folds = np.array_split(years, 5)
        feats = {"base": ["spi90"], "D1": ["spi90", "D1"], "D2": ["spi90", "D2"], "D5": ["spi90", "D5"],
                 "SPI30": ["spi90", "spi30"], "SPI30+D1": ["spi90", "spi30", "D1"]}
        preds = {k: np.full(len(d), np.nan) for k in feats}
        for fy in folds:
            te = d.year.isin(fy).values
            for k, cols in feats.items():
                m = LogisticRegression(max_iter=1000).fit(d.loc[~te, cols], d.y[~te])
                preds[k][te] = m.predict_proba(d.loc[te, cols])[:, 1]
        y = d.y.values
        def metrics(sel):
            out = {}
            for k in preds:
                out[k] = (roc_auc_score(y[sel], preds[k][sel]), np.mean((preds[k][sel] - y[sel]) ** 2))
            return out
        base = metrics(np.ones(len(d), bool))
        clusters = (d.region + "_" + d.year.astype(str)).values
        uc = np.unique(clusters); idx_by = {c: np.flatnonzero(clusters == c) for c in uc}
        boots = []
        for _ in range(NBOOT):
            pick = rng.choice(uc, len(uc))
            sel = np.concatenate([idx_by[c] for c in pick])
            yy = y[sel]
            if yy.min() == yy.max():
                continue
            boots.append({k: (roc_auc_score(yy, preds[k][sel]), np.mean((preds[k][sel] - yy) ** 2)) for k in preds})
        for k, ref in [("D1", "base"), ("D2", "base"), ("D5", "base"), ("SPI30", "base"), ("SPI30+D1", "SPI30")]:
            da = np.array([b[k][0] - b[ref][0] for b in boots])
            db = np.array([b[k][1] - b[ref][1] for b in boots])
            res.append(dict(group=r, added=k, reference=ref, n=len(d), prevalence=float(y.mean()),
                            auc_base=base[ref][0], auc_model=base[k][0],
                            dauc=base[k][0] - base[ref][0], dauc_lo=np.percentile(da, 2.5), dauc_hi=np.percentile(da, 97.5),
                            brier_base=base[ref][1], dbrier=base[k][1] - base[ref][1],
                            dbrier_lo=np.percentile(db, 2.5), dbrier_hi=np.percentile(db, 97.5)))
        print(r, "incremental done", flush=True)
    pd.DataFrame(res).to_csv(os.path.join(OUT, "incremental.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if "incremental" not in sys.argv:
        association()
    incremental()
