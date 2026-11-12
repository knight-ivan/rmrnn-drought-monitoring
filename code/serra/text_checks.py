"""Descriptive quantities that the manuscript text quotes but no other script tabulates.

Every sentence of the paper that describes a figure (composites, operating curves,
station traces) or quotes a summary not in a table is backed by a row written here,
and make_tables.py turns these rows into LaTeX macros.

Outputs (results/checks/):
  composites.csv      event-aligned mean anomalies: minimum, its lag, values at fixed lags
  amoc_compare.csv    operating curves: POD of each detector at fixed false-alarm rates
  trace_alarms.csv    alarms of D1/D3/D7 relative to the reference onsets of the Fig. traces stations
  ci_width.csv        year-cluster bootstrap vs naive binomial 95% interval width for POD
  msss_same.csv       precipitation-only vs temperature-aware MSESS on the same stations
  donors.csv          temperature donor stations per region and their distances
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS
from evaluate_warning import alarms

OUT = os.path.join(ROOT, "results", "checks")
LAGS = np.arange(-180, 181)
FIXED = [-60, -30, 0, 30, 60]
TRACES = [("texas", "USC00411048", "2010-01-01", "2012-03-31"),
          ("germany", "GME00122746", "2017-09-01", "2019-06-30"),
          ("australia", "ASN00075032", "2017-06-01", "2019-12-31"),
          ("taiwan", "466990", "2019-11-01", "2021-08-31")]


def composites():
    rows = []
    for sub, keys in [("panel", ["D1", "D2", "D3", "D7"]), ("panel_mv", ["D1", "D8", "D9", "D10", "D11"])]:
        for r in REGIONS:
            pn = np.load(os.path.join(ROOT, "results", sub, f"{r}.npz"))
            ev = pd.read_csv(os.path.join(ROOT, "results", sub, f"{r}_events.csv"), dtype={"station_id": str})
            ev = ev[pd.to_datetime(ev.onset_date).dt.year >= 2001]
            si = {s: i for i, s in enumerate(pn["station_id"])}
            acc = {k: [] for k in keys}
            for _, e in ev.iterrows():
                t = LAGS + e.onset
                if t[-1] >= len(DATES):
                    continue
                for k in keys:
                    acc[k].append(pn[k][si[e.station_id]][t])
            for k in keys:
                x = np.nanmean(acc[k], 0); j = int(np.nanargmin(x))
                row = dict(subset=sub, region=r, det=k, n_onsets=len(acc[k]), min_value=x[j], min_lag=int(LAGS[j]))
                row.update({f"v{l:+d}": x[l + 180] for l in FIXED})
                rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "composites.csv"), index=False)


def amoc_compare():
    A = pd.read_csv(os.path.join(ROOT, "results", "warning", "amoc.csv")); rows = []
    for r in REGIONS:
        for d in ["D1", "D2", "D3", "D7"]:
            a = A[(A.region == r) & (A.det == d)].sort_values("far_per_yr")
            for f in [0.25, 0.5, 1.0, 1.5, 2.0]:
                rows.append(dict(region=r, det=d, far=f, pod=float(np.interp(f, a.far_per_yr, a.pod))))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "amoc_compare.csv"), index=False)


def trace_alarms():
    T = pd.read_csv(os.path.join(ROOT, "results", "warning", "thresholds.csv")); rows = []
    for r, sid, a, b in TRACES:
        pn = np.load(os.path.join(ROOT, "results", "panel", f"{r}.npz")); i = list(pn["station_id"]).index(sid)
        ev = pd.read_csv(os.path.join(ROOT, "results", "panel", f"{r}_events.csv"), dtype={"station_id": str})
        ons = [x for x in pd.to_datetime(ev[ev.station_id == sid].onset_date) if pd.Timestamp(a) <= x <= pd.Timestamp(b)]
        sel = (DATES >= a) & (DATES <= b); d = DATES[sel]
        for det in ["D1", "D3", "D7"]:
            h = T[(T.region == r) & (T.det == det) & (T.budget == 1.0)].thr.iloc[0]
            al = d[np.flatnonzero(alarms(det, pn[det][i].astype(float), h)[sel])]
            for o in ons:   # first alarm within the hit window [-60, +120] of this onset
                lag = [(x - o).days for x in al if -60 <= (x - o).days <= 120]
                rows.append(dict(region=r, station_id=sid, onset=o.date(), det=det,
                                 first_alarm_lag=lag[0] if lag else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "trace_alarms.csv"), index=False)


def ci_width():
    s = pd.read_csv(os.path.join(ROOT, "results", "warning", "summary.csv")); s = s[s.budget == 1.0]; rows = []
    for _, x in s.iterrows():
        naive = 2 * 1.96 * np.sqrt(x.pod * (1 - x.pod) / x.n_events)
        rows.append(dict(group=x.group, det=x.det, n_events=x.n_events, pod=x.pod,
                         cluster_width=x.pod_hi - x.pod_lo, naive_width=naive, ratio=(x.pod_hi - x.pod_lo) / naive))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ci_width.csv"), index=False)


def msss_same():
    a = pd.read_csv(os.path.join(ROOT, "results", "forecast", "station_metrics.csv"), dtype={"station_id": str})
    b = pd.read_csv(os.path.join(ROOT, "results", "forecast_mv", "station_metrics.csv"), dtype={"station_id": str})
    a = a[a.model == "rmrnn"].groupby(["region", "station_id"]).msss.mean()
    b = b[b.model == "rmrnn"].groupby(["region", "station_id"]).msss.mean()
    m = pd.concat([a.rename("msss_p"), b.rename("msss_mv")], axis=1, join="inner").reset_index()
    rows = [dict(region=g, n=len(x), msss_p=x.msss_p.mean(), msss_mv=x.msss_mv.mean(),
                 frac_improved=(x.msss_mv > x.msss_p).mean())
            for g, x in list(m.groupby("region")) + [("all", m)]]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "msss_same.csv"), index=False)


def donors():
    rows = []
    for r in REGIONS:
        m = pd.read_csv(os.path.join(ROOT, "data", r, "temperature_meta.csv")).dropna(subset=["distance_km"])
        d = m[m.distance_km > 0].distance_km
        rows.append(dict(region=r, n_with_temperature=len(m), n_donor=len(d),
                         median_donor_km=d.median() if len(d) else np.nan, max_donor_km=d.max() if len(d) else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "donors.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    amoc_compare(); ci_width(); msss_same(); donors(); trace_alarms(); composites()
    print("checks written to", OUT)
