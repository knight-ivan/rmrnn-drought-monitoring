"""Event-based warning evaluation at matched alarm budgets (PROTOCOL.md).

Outputs (results/warning/):
  thresholds.csv           calibrated threshold per region x detector x budget
  station_year.csv         per station-year: events, hits, false alarms, exposure
  event_rows.csv           per event x detector (primary budget): hit, timing
  summary.csv              region x detector x budget: POD, FA/yr, timing + bootstrap CIs
  paired.csv               paired differences vs D1 with bootstrap CIs
  amoc.csv                 threshold-free POD vs FA/yr curves on 2001-2024
"""
import os, sys
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS, CAL, EVAL, year_mask

DETS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]   # D6 = exploratory combination; D7 = SPI-30 (post hoc)
PAIRS = [("D1", "D2"), ("D1", "D3"), ("D1", "D4"), ("D1", "D5"), ("D1", "D7"), ("D6", "D3"), ("D6", "D1"), ("D7", "D3")]
K, REFRACT, PRE, POST = 0.5, 90, 60, 120
BUDGETS = [1.0, 0.5, 2.0]          # alarms per station-year in calibration (primary first)
NBOOT = 2000
OUT = os.path.join(ROOT, "results", "warning")
PANEL = "panel"
if "--mv" in sys.argv:   # PROTOCOL amendment 1: multivariate subset
    DETS = ["D1", "D3", "D7", "D8", "D9", "D10", "D11"]
    PAIRS = [("D8", "D11"), ("D8", "D10"), ("D8", "D9"), ("D8", "D1"), ("D10", "D11"), ("D1", "D3")]
    OUT = os.path.join(ROOT, "results", "warning_mv")
    PANEL = "panel_mv"


@njit(cache=True)
def cusum_alarm_flags(z, h, k, refract):
    n = len(z); a = np.zeros(n, np.bool_); c = 0.0; last = -10 ** 9
    for t in range(n):
        v = z[t]
        if np.isnan(v):
            continue
        c = max(0.0, c - v - k)
        if c >= h:
            if t - last > refract:
                a[t] = True; last = t
            c = 0.0
    return a


@njit(cache=True)
def thresh_alarm_flags(z, c, refract):
    n = len(z); a = np.zeros(n, np.bool_); prev = False; last = -10 ** 9
    for t in range(n):
        v = z[t]
        if np.isnan(v):
            continue
        b = v <= c
        if b and (not prev) and t - last > refract:
            a[t] = True; last = t
        prev = b
    return a


def alarms(det, z, thr):
    if det == "D4":
        return thresh_alarm_flags(z, thr, REFRACT)
    return cusum_alarm_flags(z, thr, K, REFRACT)


GRID = {d: np.concatenate([np.linspace(0.05, 0.95, 19), np.arange(1, 30.01, 0.25), np.arange(31, 120.1, 1.0)]) for d in DETS}
GRID["D4"] = np.arange(-3.0, 2.0001, 0.025)
CALM, EVM = year_mask(DATES, CAL), year_mask(DATES, EVAL)
_RATE_CACHE = {}


def calibration_rates(Z, det, key):
    if key not in _RATE_CACHE:
        years = (CALM.sum() / 365.25) * Z.shape[0]
        _RATE_CACHE[key] = np.array([sum(alarms(det, z, g)[CALM].sum() for z in Z) / years for g in GRID[det]])
    return _RATE_CACHE[key]


def calibrate(Z, det, budget, key):
    """Grid threshold whose pooled 1991-2000 alarm rate is closest to `budget`
    alarms per station-year. If the budget is not attainable the closest rate is
    used and flagged in thresholds.csv."""
    rates = calibration_rates(Z, det, key)
    j = int(np.argmin(np.abs(rates - budget)))
    return float(GRID[det][j]), float(rates[j])


def event_windows(ev, n):
    ok = np.zeros(n, bool)
    for _, e in ev.iterrows():
        ok[max(0, e.onset - PRE): e.end + 1] = True
    return ok


def score_station(a, ev, okwin):
    """Rows per evaluated event and station-year false alarm/exposure."""
    idx = np.flatnonzero(a & EVM)
    rows = []
    for _, e in ev.iterrows():
        if not EVM[e.onset]:
            continue
        w = idx[(idx >= e.onset - PRE) & (idx <= e.onset + POST)]
        rows.append((e.onset, DATES[e.onset].year, len(w) > 0, (w[0] - e.onset) if len(w) else np.nan))
    fa_days = idx[~okwin[idx]]
    fa_year = pd.Series(DATES[fa_days].year).value_counts()
    expo = pd.Series(DATES[EVM & ~okwin].year).value_counts() / 365.25
    return rows, fa_year, expo


def main():
    os.makedirs(OUT, exist_ok=True)
    thr_rows, sy_rows, ev_rows, amoc_rows = [], [], [], []
    for r in REGIONS:
        f = os.path.join(ROOT, "results", PANEL, f"{r}.npz")
        if not os.path.exists(f):
            print("missing", r); continue
        pn = np.load(f); sids = pn["station_id"]
        evs = pd.read_csv(os.path.join(ROOT, "results", PANEL, f"{r}_events.csv"), dtype={"station_id": str})
        okw = {s: event_windows(evs[evs.station_id == s], len(DATES)) for s in sids}
        for det in DETS:
            if det == "D6":
                # exploratory: CUSUM on the re-standardised mean of the D1 and D3 anomalies
                from common import deseasonalise
                Z = np.stack([deseasonalise(0.5 * (pn["D1"][i].astype(float) + pn["D3"][i].astype(float)), DATES)
                              for i in range(len(sids))])
            else:
                Z = pn[det].astype(np.float64)
            for b in BUDGETS:
                thr, ach = calibrate(Z, det, b, (r, det))
                thr_rows.append(dict(region=r, det=det, budget=b, thr=thr, achieved_cal_rate=ach,
                                     attained=bool(abs(ach - b) <= 0.1 * b)))
                for i, s in enumerate(sids):
                    rows, fa_y, expo = score_station(alarms(det, Z[i], thr), evs[evs.station_id == s], okw[s])
                    for y in range(EVAL[0], EVAL[1] + 1):
                        ry = [x for x in rows if x[1] == y]
                        sy_rows.append(dict(region=r, det=det, budget=b, station_id=s, year=y,
                                            n_ev=len(ry), n_hit=sum(x[2] for x in ry),
                                            fa=int(fa_y.get(y, 0)), expo=float(expo.get(y, 0.0))))
                    if b == BUDGETS[0]:
                        ev_rows += [dict(region=r, det=det, station_id=s, onset=o, year=y, hit=h, timing=t)
                                    for o, y, h, t in rows]
            # threshold-free curve on the evaluation period
            for g in GRID[det][:: 3]:
                nev = nh = fa = 0; ex = 0.0
                for i, s in enumerate(sids):
                    rows, fa_y, expo = score_station(alarms(det, Z[i], g), evs[evs.station_id == s], okw[s])
                    nev += len(rows); nh += sum(x[2] for x in rows); fa += int(fa_y.sum()); ex += float(expo.sum())
                amoc_rows.append(dict(region=r, det=det, thr=g, pod=nh / max(nev, 1), far_per_yr=fa / max(ex, 1e-9)))
            print(r, det, "done", flush=True)
    pd.DataFrame(thr_rows).to_csv(os.path.join(OUT, "thresholds.csv"), index=False)
    SY = pd.DataFrame(sy_rows); SY.to_csv(os.path.join(OUT, "station_year.csv"), index=False)
    EV = pd.DataFrame(ev_rows); EV.to_csv(os.path.join(OUT, "event_rows.csv"), index=False)
    pd.DataFrame(amoc_rows).to_csv(os.path.join(OUT, "amoc.csv"), index=False)
    bootstrap(SY, EV)


def bootstrap(SY, EV):
    """Cluster bootstrap over calendar years within each region (paired across
    detectors: every detector is evaluated on the same resampled years)."""
    rng = np.random.default_rng(20261112)
    years = np.arange(EVAL[0], EVAL[1] + 1); ny = len(years)
    regions = list(SY.region.unique())
    budgets = list(SY.budget.unique())
    agg = SY.groupby(["region", "det", "budget", "year"])[["n_ev", "n_hit", "fa", "expo"]].sum()
    V = {}
    for key, g in agg.groupby(level=[0, 1, 2]):
        V[key] = g.droplevel([0, 1, 2]).reindex(years).fillna(0).values.T   # 4 x ny
    W = {r: np.stack([np.bincount(rng.integers(0, ny, ny), minlength=ny) for _ in range(NBOOT)]) for r in regions}
    T = {}
    for (r, d), g in EV[EV.hit.astype(bool)].groupby(["region", "det"]):
        T[(r, d)] = (g.year.values - EVAL[0], g.timing.values.astype(float))
    summ, pair = [], []
    for gname in regions + ["all"]:
        regs = regions if gname == "all" else [gname]
        for b in budgets:
            est, boots = {}, {}
            for det in DETS:
                tot = sum(V[(r, det, b)].sum(1) for r in regs)
                est[det] = (tot[1] / max(tot[0], 1), tot[2] / max(tot[3], 1e-9))
                bt = sum(W[r] @ V[(r, det, b)].T for r in regs)       # NBOOT x 4
                boots[det] = np.stack([bt[:, 1] / np.maximum(bt[:, 0], 1), bt[:, 2] / np.maximum(bt[:, 3], 1e-9)], 1)
                tim, tl, th = np.nan, np.nan, np.nan
                if b == budgets[0]:
                    rr = [r for r in regs if (r, det) in T]
                    if rr:
                        tim = float(np.median(np.concatenate([T[(r, det)][1] for r in rr])))
                        tb = []
                        for k in range(NBOOT):
                            vals = np.concatenate([np.repeat(T[(r, det)][1], W[r][k][T[(r, det)][0]]) for r in rr])
                            tb.append(np.median(vals) if len(vals) else np.nan)
                        tl, th = np.nanpercentile(tb, [2.5, 97.5])
                lo, hi = np.nanpercentile(boots[det], [2.5, 97.5], axis=0)
                summ.append(dict(group=gname, budget=b, det=det, n_events=int(tot[0]), n_hits=int(tot[1]),
                                 n_fa=int(tot[2]), exposure_yr=float(tot[3]),
                                 pod=est[det][0], pod_lo=lo[0], pod_hi=hi[0],
                                 far=est[det][1], far_lo=lo[1], far_hi=hi[1],
                                 timing_med=tim, timing_lo=tl, timing_hi=th))
            for a_, c_ in PAIRS:
                dd = boots[a_] - boots[c_]
                lo, hi = np.nanpercentile(dd, [2.5, 97.5], axis=0)
                pair.append(dict(group=gname, budget=b, comp=f"{a_}-{c_}",
                                 dpod=est[a_][0] - est[c_][0], dpod_lo=lo[0], dpod_hi=hi[0],
                                 dfar=est[a_][1] - est[c_][1], dfar_lo=lo[1], dfar_hi=hi[1],
                                 p_dpod_le0=float(np.mean(dd[:, 0] <= 0))))
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.DataFrame(pair).to_csv(os.path.join(OUT, "paired.csv"), index=False)
    # paired timing on events detected by both D1 and the comparator
    # (positive difference = D1 alarms earlier)
    pt = []
    for gname in regions + ["all"]:
        regs = regions if gname == "all" else [gname]
        for a_, det in PAIRS:
            E1 = EV[(EV.det == a_) & EV.hit.astype(bool)]
            Ec = EV[(EV.det == det) & EV.hit.astype(bool)]
            m = E1.merge(Ec, on=["region", "station_id", "onset", "year"], suffixes=("_1", "_c"))
            m = m[m.region.isin(regs)]
            if not len(m):
                continue
            diff = (m.timing_c - m.timing_1).values
            yi = m.year.values - EVAL[0]
            bm = []
            for k in range(NBOOT):
                vals = np.concatenate([np.repeat(diff[(m.region == r).values], W[r][k][yi[(m.region == r).values]]) for r in regs])
                bm.append(np.median(vals) if len(vals) else np.nan)
            lo, hi = np.nanpercentile(bm, [2.5, 97.5])
            pt.append(dict(group=gname, comp=f"{det} minus {a_}", n_both=len(m), median_diff=float(np.median(diff)),
                           lo=lo, hi=hi, frac_first_earlier=float(np.mean(diff > 0)), frac_tie=float(np.mean(diff == 0))))
    pd.DataFrame(pt).to_csv(os.path.join(OUT, "paired_timing.csv"), index=False)


if __name__ == "__main__":
    if "boot" in sys.argv:
        bootstrap(pd.read_csv(os.path.join(OUT, "station_year.csv")), pd.read_csv(os.path.join(OUT, "event_rows.csv")))
    else:
        main()
