"""Further post hoc checks (PROTOCOL.md, Amendment 3).

(a) Detection by deadline. Timing compared on events detected by both detectors
    conditions on joint detection. The unconditional summary is the fraction of ALL
    reference onsets for which a detector has alarmed by day d relative to onset
    (alarms from day -60 on), for d = -60..+120; a missed event never counts.
    Paired differences at fixed deadlines carry year-cluster bootstrap intervals.
(b) Reference events from a fixed historical climatology. The pre-specified events
    use day-of-year soil-wetness percentiles computed over all years 1981-2024, so an
    onset in an early evaluation year depends on later years. Here the percentile
    reference is 1981-2000 only and is applied unchanged to 2001-2024; events are
    re-derived by the same run rules and all detectors are re-scored at their
    calibrated thresholds (calibration does not use the events). The events are also
    written to results/panel[_mv]/<region>_events_ref1981_2000.csv for
    incremental_forward.py --fixedref.
(c) Forecast equivalence with year-cluster resampling (not with --mv; covers both
    analyses): the station-level relative MSE differences are recomputed on
    resampled calendar years within region (2000 replicates), which keeps the
    dependence among stations that share weather.
(d) MERRA-2 grid cells (0.5 x 0.625 degrees) shared by stations (not with --mv).

Inputs: results/panel*/ (scores, events), results/warning*/ (thresholds, event rows,
spi3_dryside*.csv from matching_checks.py), results/checks/intermediates.npz
(extract_intermediates.py), data/<region>/station_meta.csv.
Outputs (results/warning[_mv]/): deadline.csv, deadline_diff.csv, deadline_cross.csv, fixedref_summary.csv,
fixedref_paired.csv, fixedref_events.csv; results/forecast[_mv]/cluster_equivalence.csv;
results/checks/grid_cells.csv.

Usage: python3 code/serra/review_checks.py [--mv]
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS, EVAL, doy_index, drought_events
import evaluate_warning as EW
import matching_checks as MC

MV = "--mv" in sys.argv
NBOOT, SEED, NY = MC.NBOOT, MC.SEED, MC.NY
DEADLINES = np.arange(-60, 121)
REFYEARS = (1981, 2000)
PAIRS = ([("D8", "D11"), ("D8", "D10"), ("D8", "D9"), ("D8", "D1")] if MV else
         [("D1", "D3"), ("D1", "D7"), ("D1", "D2"), ("D6", "D3"), ("D7", "D3"), ("D1", "D4")])
INTER = os.path.join(ROOT, "results", "checks", "intermediates.npz")


# ---------------------------------------------------------------- (a) detection by deadline
def deadline(W):
    E = pd.read_csv(os.path.join(EW.OUT, "event_rows.csv"), dtype={"station_id": str})
    if not MV:   # monthly SPI-3 with the dry-side threshold replaces the pre-specified D4
        D = pd.read_csv(os.path.join(EW.OUT, "spi3_dryside_events.csv"), dtype={"station_id": str})
        E = pd.concat([E[E.det != "D4"], D], ignore_index=True)
    dets = sorted(E.det.unique(), key=lambda d: int(d[1:]))
    H = {}                                      # (region, det) -> [year, deadline] cumulative hits; N -> [year]
    for (r, d), g in E.groupby(["region", "det"]):
        h = np.zeros((NY, len(DEADLINES))); n = np.bincount(g.year.values - EVAL[0], minlength=NY).astype(float)
        hit = g[g.hit.astype(bool)]
        for y, t in zip(hit.year.values - EVAL[0], hit.timing.values):
            h[y, int(t) + 60:] += 1
        H[(r, d)] = (h, n)
    rows, diff, cross = [], [], []
    for grp in REGIONS + ["all"]:
        regs = REGIONS if grp == "all" else [grp]
        est, boot = {}, {}
        for d in dets:
            h = sum(H[(r, d)][0].sum(0) for r in regs); n = sum(H[(r, d)][1].sum() for r in regs)
            est[d] = h / n
            bh = sum(W[r] @ H[(r, d)][0] for r in regs); bn = sum(W[r] @ H[(r, d)][1] for r in regs)
            boot[d] = bh / np.maximum(bn, 1)[:, None]
            lo, hi = np.percentile(boot[d], [2.5, 97.5], axis=0)
            rows += [dict(group=grp, det=d, deadline=int(dl), frac=est[d][j], lo=lo[j], hi=hi[j])
                     for j, dl in enumerate(DEADLINES)]
        for a, b in PAIRS:
            dd = boot[a] - boot[b]
            lo, hi = np.percentile(dd, [2.5, 97.5], axis=0)
            diff += [dict(group=grp, comp=f"{a}-{b}", deadline=int(dl), frac_first=est[a][j], frac_second=est[b][j],
                          diff=est[a][j] - est[b][j], lo=lo[j], hi=hi[j]) for j, dl in enumerate(DEADLINES)]
            ahead = DEADLINES[est[a] > est[b]]       # deadlines at which the first detector is ahead
            cross.append(dict(group=grp, comp=f"{a}-{b}", first_ahead_from=float(ahead.min()) if len(ahead) else np.nan,
                              first_ahead_until=float(ahead.max()) if len(ahead) else np.nan,
                              max_lead=float((est[a] - est[b]).max()), at_deadline=int(DEADLINES[np.argmax(est[a] - est[b])])))
    pd.DataFrame(rows).to_csv(os.path.join(EW.OUT, "deadline.csv"), index=False)
    pd.DataFrame(diff).to_csv(os.path.join(EW.OUT, "deadline_diff.csv"), index=False)
    pd.DataFrame(cross).to_csv(os.path.join(EW.OUT, "deadline_cross.csv"), index=False)


# ---------------------------------------------------------------- (b) fixed-reference events
def sm_percentile_ref(sm, ref=None, pool=15):
    """Day-of-year percentile (0-100) of soil wetness against the values of the reference
    years in a +/-pool-day window; ref=None uses all years (the pre-specified definition)."""
    sm = np.asarray(sm, float); doy = doy_index(DATES); yr = DATES.year.values
    ok = np.isfinite(sm) & (yr >= 1981)
    inref = ok if ref is None else ok & (yr >= ref[0]) & (yr <= ref[1])
    out = np.full(len(sm), np.nan)
    for d in range(1, 366):
        dd = np.abs(((doy - d + 182) % 365) - 182) <= pool
        base = np.sort(sm[dd & inref]); sel = (doy == d) & ok
        out[sel] = 100.0 * np.searchsorted(base, sm[sel], side="right") / len(base)
    return out


def fixed_reference(W):
    Z = np.load(INTER)
    AL, EV0 = MC.load()
    if not MV:   # dry-side SPI-3 alarms replace the pre-specified D4
        T = pd.read_csv(os.path.join(EW.OUT, "spi3_dryside.csv")); T = T[T.budget == 1.0].set_index("group")
        for r in REGIONS:
            pn = np.load(os.path.join(ROOT, "results", EW.PANEL, f"{r}.npz"))
            for i, s in enumerate(pn["station_id"]):
                AL[(r, "D4", str(s))] = np.flatnonzero(EW.alarms("D4", pn["D4"][i].astype(np.float64),
                                                                 float(T.loc[r].thr_dryside)) & MC.EVM)
    EV1, rows = {}, []
    for r in REGIONS:
        ids = [str(s) for s in Z[f"{r}_panel_ids"]]; sm = Z[f"{r}_sm"]
        out = []
        for s in [str(x) for x in Z[f"{r}_{EW.PANEL}_ids"]]:
            x = sm[ids.index(s)]
            chk = drought_events(sm_percentile_ref(x), DATES)          # must reproduce the stored events
            assert np.array_equal(np.array([[e["onset"], e["end"]] for e in chk]).reshape(-1, 2), EV0[(r, s)]), (r, s)
            ev = drought_events(sm_percentile_ref(x, REFYEARS), DATES)
            EV1[(r, s)] = np.array([[e["onset"], e["end"]] for e in ev], int).reshape(-1, 2)
            out += [dict(e, station_id=s) for e in ev]
        ev = pd.DataFrame(out)
        ev.to_csv(os.path.join(ROOT, "results", EW.PANEL, f"{r}_events_ref{REFYEARS[0]}_{REFYEARS[1]}.csv"), index=False)
        both = lambda E: sum(int(MC.EVM[e[:, 0]].sum()) for (rr, s), e in E.items() if rr == r and len(e))
        days = lambda E: sum(int(sum(min(b, len(DATES) - 1) - a + 1 for a, b in e if MC.EVM[a])) for (rr, s), e in E.items() if rr == r)
        rows.append(dict(region=r, n_events_prespecified=both(EV0), n_events_fixedref=both(EV1),
                         drought_days_prespecified=days(EV0), drought_days_fixedref=days(EV1)))
    ev = pd.DataFrame(rows)
    ev = pd.concat([ev, pd.DataFrame([dict(region="all", **ev.drop(columns="region").sum().to_dict())])], ignore_index=True)
    ev.to_csv(os.path.join(EW.OUT, "fixedref_events.csv"), index=False)
    V, EVT, _ = MC.score_all(AL, EV1, 60, 120, "prespecified")
    S, P = MC.summarise(V, EVT, W, pairs=PAIRS)
    ch = MC.chance(AL, EV1); ch = ch[(ch.pre == 60) & (ch.post == 120)][["group", "det", "pod_chance", "lo", "hi"]]
    S.merge(ch.rename(columns={"lo": "chance_lo", "hi": "chance_hi"}), on=["group", "det"], how="left").to_csv(
        os.path.join(EW.OUT, "fixedref_summary.csv"), index=False)
    P.to_csv(os.path.join(EW.OUT, "fixedref_paired.csv"), index=False)


# ---------------------------------------------------------------- (c) forecast equivalence
def forecast_cluster():
    Z = np.load(INTER); rng = np.random.default_rng(SEED + 4)
    for sc, sub in [("scores", "forecast"), ("scores_mv", "forecast_mv")]:
        W = MC.weights(1, rng); res = []
        ref = pd.read_csv(os.path.join(ROOT, "results", sub, "paired.csv"))
        rel = {}        # (region, comp) -> (point [stations], boot [NBOOT, stations])
        for r in REGIONS:
            sse, n = Z[f"{r}_{sc}_sse"], Z[f"{r}_{sc}_n"]
            mse = sse.sum(-1) / n.sum(-1)[:, None, None]                         # station, model, seed
            bm = np.einsum("by,smky->bsmk", W[r], sse) / (W[r] @ n.T)[:, :, None, None]
            m, b = mse.mean(-1), bm.mean(-1)                                     # mean over seeds
            for j, other in [(1, "lam0"), (2, "gru")]:
                rel[(r, other)] = ((m[:, 0] - m[:, j]) / m[:, j], (b[:, :, 0] - b[:, :, j]) / b[:, :, j])
        for grp in REGIONS + ["all"]:
            regs = REGIONS if grp == "all" else [grp]
            for other in ["lam0", "gru"]:
                pt = np.concatenate([rel[(r, other)][0] for r in regs])
                bt = np.concatenate([rel[(r, other)][1] for r in regs], axis=1).mean(1)
                x = ref[(ref.group == grp) & (ref.comp == f"rmrnn-{other}") & (ref.metric == "mse")].iloc[0]
                assert abs(pt.mean() - x.mean_rel_diff) < 1e-6, (grp, other, pt.mean(), x.mean_rel_diff)
                lo, hi = np.percentile(bt, [5, 95])
                res.append(dict(group=grp, comp=f"rmrnn-{other}", n=len(pt), mean_rel_diff=pt.mean(),
                                station_ci90_lo=x.ci90_lo, station_ci90_hi=x.ci90_hi,
                                cluster_ci90_lo=lo, cluster_ci90_hi=hi,
                                equivalent_station=bool(x.ci90_lo > -0.02 and x.ci90_hi < 0.02),
                                equivalent_cluster=bool(lo > -0.02 and hi < 0.02)))
        pd.DataFrame(res).to_csv(os.path.join(ROOT, "results", sub, "cluster_equivalence.csv"), index=False)


# ---------------------------------------------------------------- (d) reanalysis grid cells
def grid_cells():
    rows = []
    for r in REGIONS:
        m = pd.read_csv(os.path.join(ROOT, "data", r, "station_meta.csv"))
        cell = list(zip(np.round(m.lat / 0.5).astype(int), np.round(m.lon / 0.625).astype(int)))
        c = pd.Series(cell).value_counts()
        rows.append(dict(region=r, stations=len(m), cells=len(c), max_stations_per_cell=int(c.max()),
                         stations_sharing_a_cell=int(c[c > 1].sum())))
    d = pd.DataFrame(rows)
    d = pd.concat([d, pd.DataFrame([dict(region="all", stations=d.stations.sum(), cells=d.cells.sum(),
                                         max_stations_per_cell=d.max_stations_per_cell.max(),
                                         stations_sharing_a_cell=d.stations_sharing_a_cell.sum())])], ignore_index=True)
    d.to_csv(os.path.join(ROOT, "results", "checks", "grid_cells.csv"), index=False)


if __name__ == "__main__":
    W = MC.weights(1, np.random.default_rng(SEED))      # same draws as evaluate_warning.bootstrap
    deadline(W); print("deadline done", flush=True)
    fixed_reference(W); print("fixed reference done", flush=True)
    if not MV:
        forecast_cluster(); print("forecast cluster done", flush=True)
        grid_cells()
