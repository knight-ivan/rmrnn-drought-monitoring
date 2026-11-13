"""Robustness of the event-based scoring (PROTOCOL.md, Amendment 2; post hoc).

The pre-specified rule (evaluate_warning.py) credits an event with a hit when any
alarm falls in [onset - 60 d, onset + 120 d]. Three questions are examined here.

(a) Matching. When the windows of two events at one station overlap, the
    pre-specified rule can credit the same alarm to both. The ONE-TO-ONE rule
    assigns every alarm to at most one event: events are processed in
    chronological order and each takes the earliest unassigned alarm in its
    window (for windows of equal length this maximises the number of matched
    events). Under this rule a matched alarm is never a false alarm; a false
    alarm is an unmatched alarm outside [onset - pre, end] of every event.
(b) Window. Both rules are re-scored for pre in {0, 30, 60} and post in
    {60, 90, 120} days (the false-alarm window moves with `pre`).
    For each window the POD expected from alarms that carry no information is
    estimated by shifting every station's alarm sequence circularly within
    2001-2024 by a random offset (NSHIFT replicates), which keeps the number of
    alarms and their spacing and removes their alignment with the events.
(c) Dependence. Station-level onsets are grouped into regional episodes, and
    the year-cluster bootstrap is repeated with circular blocks of 2, 3 and 4
    consecutive years, which allows for dependence carried by multi-year droughts.

(d) SPI-3 threshold. The alarm count of the month-end down-crossing rule (D4) is
    not monotone in its threshold c, and the pre-specified grid search can meet
    the budget with a positive (wet-side) threshold. D4 is re-calibrated with the
    grid restricted to c <= 0 (closest attainable rate) and re-scored.

Thresholds are those calibrated by evaluate_warning.py (thresholds.csv); nothing is
re-tuned. With rule = prespecified, pre = 60, post = 120 and block = 1 the script
reproduces summary.csv, paired.csv and paired_timing.csv exactly (checked at the end).

Outputs (results/warning[_mv]/):
  matching_overlap.csv   how often windows overlap and alarms are credited twice
  matching_summary.csv   POD, FAR, timing per rule x window x group x detector
  matching_paired.csv    paired dPOD, dFAR and timing differences per rule x window
  matching_chance.csv    POD of randomly timed alarms (same number and spacing) per window
  episodes.csv           regional episodes of station-level onsets
  spi3_dryside.csv       D4 with its threshold restricted to c <= 0 (not with --mv)
  blockboot.csv          interval widths under multi-year block resampling

Usage: python3 code/serra/matching_checks.py [--mv]
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS, EVAL, deseasonalise
import evaluate_warning as EW          # reads --mv from sys.argv

NBOOT = 2000
SEED = 20261112                        # same seed and draw order as evaluate_warning.bootstrap
WINDOWS = [(pre, post) for pre in (60, 30, 0) for post in (120, 90, 60)]
RULES = ["prespecified", "one_to_one"]
BLOCKS = [1, 2, 3, 4]
NSHIFT = 500                           # random circular shifts for the no-information reference
EPISODE_GAP = 30                       # days: onsets closer than this are linked into one episode
YEARS = np.arange(EVAL[0], EVAL[1] + 1)
NY = len(YEARS)
YR = DATES.year.values
EVM = EW.EVM


def load():
    """Alarm days in 2001-2024 at the calibrated primary threshold, and all events."""
    T = pd.read_csv(os.path.join(EW.OUT, "thresholds.csv"))
    al, ev = {}, {}
    for r in REGIONS:
        pn = np.load(os.path.join(ROOT, "results", EW.PANEL, f"{r}.npz"))
        sids = [str(s) for s in pn["station_id"]]
        E = pd.read_csv(os.path.join(ROOT, "results", EW.PANEL, f"{r}_events.csv"), dtype={"station_id": str})
        for s in sids:
            ev[(r, s)] = E[E.station_id == s].sort_values("onset")[["onset", "end"]].values.astype(int)
        for det in EW.DETS:
            if det == "D6":
                Z = np.stack([deseasonalise(0.5 * (pn["D1"][i].astype(float) + pn["D3"][i].astype(float)), DATES)
                              for i in range(len(sids))])
            else:
                Z = pn[det].astype(np.float64)
            thr = float(T[(T.region == r) & (T.det == det) & (T.budget == 1.0)].thr.iloc[0])
            for i, s in enumerate(sids):
                al[(r, det, s)] = np.flatnonzero(EW.alarms(det, Z[i], thr) & EVM)
    return al, ev


def score(al, ev, pre, post, rule):
    """Score one station x detector.
    Returns per-event rows (onset, year, hit, timing, index of credited alarm),
    false-alarm days, and the non-drought exposure mask."""
    ok = np.zeros(len(DATES), bool)
    for o, e in ev:
        ok[max(0, o - pre): e + 1] = True
    used = np.zeros(len(al), bool)
    rows = []
    for o, e in ev:
        if not EVM[o]:
            continue
        lo, hi = np.searchsorted(al, o - pre, "left"), np.searchsorted(al, o + post, "right")
        j = -1
        if rule == "one_to_one":
            free = np.flatnonzero(~used[lo:hi])
            if len(free):
                j = lo + int(free[0])
        elif hi > lo:
            j = lo
        if j >= 0:
            used[j] = True
        rows.append((o, YR[o], j >= 0, float(al[j] - o) if j >= 0 else np.nan, j))
    fa = al[~ok[al] & ~used] if rule == "one_to_one" else al[~ok[al]]
    return rows, fa, ok


def score_all(AL, EV, pre, post, rule):
    """Station-year table V[(region, det)] = 4 x NY (events, hits, false alarms, exposure)
    and the per-event table."""
    V, ev_rows, diag = {}, [], []
    for (r, det, s), al in AL.items():
        rows, fa, ok = score(al, EV[(r, s)], pre, post, rule)
        v = V.setdefault((r, det), np.zeros((4, NY)))
        for o, y, h, t, j in rows:
            v[0, y - EVAL[0]] += 1; v[1, y - EVAL[0]] += h
            ev_rows.append((r, det, s, o, y, h, t))
        v[2] += np.bincount(YR[fa] - EVAL[0], minlength=NY)
        v[3] += np.bincount(YR[EVM & ~ok] - EVAL[0], minlength=NY) / 365.25
        if rule == "prespecified":
            js = np.array([x[4] for x in rows if x[2]], int)
            onsets = np.array([x[0] for x in rows])
            n_shared = int(len(js) - len(np.unique(js)))         # hits credited to an alarm already used
            late = int(np.sum(~ok[al[np.unique(js)]])) if len(js) else 0   # credited alarms also counted false
            allo = EV[(r, s)][:, 0]
            gap = np.array([o - allo[allo < o].max() if (allo < o).any() else 10 ** 6 for o in onsets])
            diag.append(dict(region=r, det=det, n_events=len(rows), n_hits=len(js), n_alarms=len(al),
                             hits_sharing_an_alarm=n_shared, credited_alarms_also_false=late,
                             events_overlapping_previous=int(np.sum(gap <= pre + post))))
    EVT = pd.DataFrame(ev_rows, columns=["region", "det", "station_id", "onset", "year", "hit", "timing"])
    return V, EVT, pd.DataFrame(diag)


def weights(block, rng):
    """NBOOT x NY resampling weights for each region. block = 1: years drawn with
    replacement (as evaluate_warning.py). block > 1: circular blocks of consecutive years."""
    W = {}
    for r in REGIONS:
        if block == 1:
            W[r] = np.stack([np.bincount(rng.integers(0, NY, NY), minlength=NY) for _ in range(NBOOT)])
        else:
            nb = int(np.ceil(NY / block)); w = np.zeros((NBOOT, NY), int)
            for b in range(NBOOT):
                idx = ((rng.integers(0, NY, nb)[:, None] + np.arange(block)[None, :]) % NY).ravel()[:NY]
                w[b] = np.bincount(idx, minlength=NY)
            W[r] = w
    return W


def summarise(V, EVT, W, timing=True, dets=None, pairs=None):
    dets = dets or EW.DETS; pairs = pairs or EW.PAIRS
    hit = EVT[EVT.hit.astype(bool)]
    S, P = [], []
    for g in REGIONS + ["all"]:
        regs = REGIONS if g == "all" else [g]
        est, boots = {}, {}
        for det in dets:
            tot = sum(V[(r, det)].sum(1) for r in regs)
            bt = sum(W[r] @ V[(r, det)].T for r in regs)
            est[det] = (tot[1] / max(tot[0], 1), tot[2] / max(tot[3], 1e-9))
            boots[det] = np.stack([bt[:, 1] / np.maximum(bt[:, 0], 1), bt[:, 2] / np.maximum(bt[:, 3], 1e-9)], 1)
            lo, hi = np.nanpercentile(boots[det], [2.5, 97.5], axis=0)
            row = dict(group=g, det=det, n_events=int(tot[0]), n_hits=int(tot[1]), n_fa=int(tot[2]),
                       exposure_yr=float(tot[3]), pod=est[det][0], pod_lo=lo[0], pod_hi=hi[0],
                       far=est[det][1], far_lo=lo[1], far_hi=hi[1])
            if timing:
                h = hit[(hit.det == det) & hit.region.isin(regs)]
                tb = _boot_median(h, h.timing.values, regs, W)
                row.update(timing_med=float(np.median(h.timing)) if len(h) else np.nan,
                           timing_lo=tb[0], timing_hi=tb[1])
            S.append(row)
        for a, c in pairs:
            dd = boots[a] - boots[c]
            lo, hi = np.nanpercentile(dd, [2.5, 97.5], axis=0)
            row = dict(group=g, comp=f"{a}-{c}", dpod=est[a][0] - est[c][0], dpod_lo=lo[0], dpod_hi=hi[0],
                       dfar=est[a][1] - est[c][1], dfar_lo=lo[1], dfar_hi=hi[1])
            if timing:   # events detected by both; positive = first detector earlier
                m = hit[hit.det == a].merge(hit[hit.det == c], on=["region", "station_id", "onset", "year"],
                                            suffixes=("_1", "_c"))
                m = m[m.region.isin(regs)]
                diff = (m.timing_c - m.timing_1).values
                tb = _boot_median(m, diff, regs, W)
                row.update(n_both=len(m), dtiming=float(np.median(diff)) if len(m) else np.nan,
                           dtiming_lo=tb[0], dtiming_hi=tb[1],
                           frac_first_earlier=float(np.mean(diff > 0)) if len(m) else np.nan)
            P.append(row)
    return pd.DataFrame(S), pd.DataFrame(P)


def _boot_median(frame, values, regs, W):
    if not len(frame):
        return np.nan, np.nan
    yi = frame.year.values - EVAL[0]; reg = frame.region.values
    parts = [(values[reg == r], yi[reg == r], W[r]) for r in regs if (reg == r).any()]
    out = np.empty(NBOOT)
    for k in range(NBOOT):
        v = np.concatenate([np.repeat(x, w[k][y]) for x, y, w in parts])
        out[k] = np.median(v) if len(v) else np.nan
    return tuple(np.nanpercentile(out, [2.5, 97.5]))


def chance(AL, EV):
    """POD of randomly timed alarms: circular shift of each station's alarm days within 2001-2024."""
    rng = np.random.default_rng(SEED + 2)
    ev_idx = np.flatnonzero(EVM); a0, n = int(ev_idx[0]), len(ev_idx)
    hits = {}                                   # (pre, post, region, det) -> NSHIFT hit counts
    nev = {}
    for (r, det, s), al in AL.items():
        on = EV[(r, s)][:, 0]; on = on[EVM[on]]
        nev[(r, det)] = nev.get((r, det), 0) + len(on)
        off = rng.integers(1, n, NSHIFT)
        for k in range(NSHIFT):
            sh = np.sort(a0 + (al - a0 + off[k]) % n)
            for pre, post in WINDOWS:
                h = np.searchsorted(sh, on + post, "right") > np.searchsorted(sh, on - pre, "left")
                key = (pre, post, r, det)
                if key not in hits:
                    hits[key] = np.zeros(NSHIFT)
                hits[key][k] += h.sum()
    rows = []
    for pre, post in WINDOWS:
        for det in EW.DETS:
            tot = sum(hits[(pre, post, r, det)] for r in REGIONS); ne = sum(nev[(r, det)] for r in REGIONS)
            for g, hh, m in [(r, hits[(pre, post, r, det)], nev[(r, det)]) for r in REGIONS] + [("all", tot, ne)]:
                rows.append(dict(pre=pre, post=post, group=g, det=det, n_events=m, pod_chance=float(hh.mean() / m),
                                 lo=float(np.percentile(hh, 2.5) / m), hi=float(np.percentile(hh, 97.5) / m)))
    return pd.DataFrame(rows)


def spi3_dryside(EV, W):
    """D4 re-calibrated on 1991-2000 with c <= 0 at every budget and scored by the
    pre-specified rule; D1 at the same budget gives the paired difference."""
    T = pd.read_csv(os.path.join(EW.OUT, "thresholds.csv"))
    grid = EW.GRID["D4"][EW.GRID["D4"] <= 1e-9]
    out = []
    for budget in EW.BUDGETS:
        AL2, cal = {}, []
        for r in REGIONS:
            pn = np.load(os.path.join(ROOT, "results", EW.PANEL, f"{r}.npz"))
            Z = pn["D4"].astype(np.float64); Z1 = pn["D1"].astype(np.float64); sids = [str(s) for s in pn["station_id"]]
            years = (EW.CALM.sum() / 365.25) * Z.shape[0]
            rates = np.array([sum(EW.alarms("D4", z, g)[EW.CALM].sum() for z in Z) / years for g in grid])
            j = int(np.argmin(np.abs(rates - budget)))
            t0 = T[(T.region == r) & (T.det == "D4") & (T.budget == budget)].iloc[0]
            t1 = float(T[(T.region == r) & (T.det == "D1") & (T.budget == budget)].thr.iloc[0])
            cal.append(dict(group=r, thr_prespecified=float(t0.thr), cal_rate_prespecified=float(t0.achieved_cal_rate),
                            thr_dryside=float(grid[j]), cal_rate_dryside=float(rates[j]),
                            attained_dryside=bool(abs(rates[j] - budget) <= 0.1 * budget)))
            for i, s in enumerate(sids):
                AL2[(r, "D4", s)] = np.flatnonzero(EW.alarms("D4", Z[i], float(grid[j])) & EVM)
                AL2[(r, "D1", s)] = np.flatnonzero(EW.alarms("D1", Z1[i], t1) & EVM)
        V, EVT, _ = score_all(AL2, EV, 60, 120, "prespecified")
        S, P = summarise(V, EVT, W, dets=["D1", "D4"], pairs=[("D1", "D4")])
        o = S[S.det == "D4"].merge(P[["group", "dpod", "dpod_lo", "dpod_hi", "dfar", "dfar_lo", "dfar_hi"]], on="group")
        out.append(o.merge(pd.DataFrame(cal), on="group", how="left").assign(budget=budget))
    return pd.concat(out, ignore_index=True)


def episodes(EV):
    """Regional episodes: station-level onsets in 2001-2024 are ordered in time within
    a region and a new episode starts when more than EPISODE_GAP days separate two
    successive onsets."""
    rows = []
    for r in REGIONS:
        o = np.sort(np.concatenate([e[EVM[e[:, 0]], 0] for (rr, s), e in EV.items() if rr == r and len(e)]))
        st = len([1 for (rr, s) in EV if rr == r])
        ep = np.concatenate([[0], np.cumsum(np.diff(o) > EPISODE_GAP)])
        size = np.bincount(ep)
        yrs = pd.Series(YR[o]).value_counts()
        rows.append(dict(region=r, stations=st, station_onsets=len(o), episodes=int(ep.max() + 1),
                         median_onsets_per_episode=float(np.median(size)), max_onsets_per_episode=int(size.max()),
                         share_onsets_in_episodes_ge5=float(size[size >= 5].sum() / len(o)),
                         years_with_onsets=int(len(yrs)), share_onsets_in_top5_years=float(yrs.head(5).sum() / len(o)),
                         median_onsets_per_station=float(len(o) / st)))
    d = pd.DataFrame(rows)
    tot = dict(region="all", stations=int(d.stations.sum()), station_onsets=int(d.station_onsets.sum()),
               episodes=int(d.episodes.sum()), median_onsets_per_episode=np.nan, max_onsets_per_episode=int(d.max_onsets_per_episode.max()),
               share_onsets_in_episodes_ge5=float((d.share_onsets_in_episodes_ge5 * d.station_onsets).sum() / d.station_onsets.sum()),
               years_with_onsets=int(d.years_with_onsets.sum()),
               share_onsets_in_top5_years=np.nan, median_onsets_per_station=float(d.station_onsets.sum() / d.stations.sum()))
    return pd.concat([d, pd.DataFrame([tot])], ignore_index=True)


def main():
    AL, EV = load()
    rng = np.random.default_rng(SEED)
    W1 = weights(1, rng)                       # identical draws to evaluate_warning.bootstrap
    S, P, D = [], [], []
    for rule in RULES:
        for pre, post in WINDOWS:
            V, EVT, diag = score_all(AL, EV, pre, post, rule)
            s, p = summarise(V, EVT, W1)
            S.append(s.assign(rule=rule, pre=pre, post=post)); P.append(p.assign(rule=rule, pre=pre, post=post))
            if rule == "prespecified":
                D.append(diag.assign(pre=pre, post=post))
            print(rule, pre, post, "done", flush=True)
    S = pd.concat(S, ignore_index=True); P = pd.concat(P, ignore_index=True)
    S.to_csv(os.path.join(EW.OUT, "matching_summary.csv"), index=False)
    P.to_csv(os.path.join(EW.OUT, "matching_paired.csv"), index=False)
    D = pd.concat(D, ignore_index=True).groupby(["pre", "post", "region", "det"], as_index=False, sort=False).sum(numeric_only=True)
    tot = D.groupby(["pre", "post", "det"], as_index=False, sort=False).sum(numeric_only=True).assign(region="all")
    pd.concat([D, tot], ignore_index=True).to_csv(os.path.join(EW.OUT, "matching_overlap.csv"), index=False)
    episodes(EV).to_csv(os.path.join(EW.OUT, "episodes.csv"), index=False)
    chance(AL, EV).to_csv(os.path.join(EW.OUT, "matching_chance.csv"), index=False)
    print("chance reference done", flush=True)
    if "D4" in EW.DETS:
        spi3_dryside(EV, W1).to_csv(os.path.join(EW.OUT, "spi3_dryside.csv"), index=False)

    # multi-year block resampling at the pre-specified window, both rules
    B = []
    rngb = np.random.default_rng(SEED + 1)
    for block in BLOCKS:
        W = W1 if block == 1 else weights(block, rngb)
        for rule in RULES:
            V, EVT, _ = score_all(AL, EV, 60, 120, rule)
            s, p = summarise(V, EVT, W)
            B.append(p.assign(kind="paired", rule=rule, block_years=block))
            B.append(s.rename(columns={"det": "comp"}).assign(kind="detector", rule=rule, block_years=block))
        print("block", block, "done", flush=True)
    pd.concat(B, ignore_index=True).to_csv(os.path.join(EW.OUT, "blockboot.csv"), index=False)

    # check: the pre-specified setting reproduces the published tables exactly
    ref = pd.read_csv(os.path.join(EW.OUT, "summary.csv")); ref = ref[ref.budget == 1.0]
    mine = S[(S.rule == "prespecified") & (S.pre == 60) & (S.post == 120)]
    m = ref.merge(mine, on=["group", "det"], suffixes=("_ref", ""))
    cols = ["pod", "pod_lo", "pod_hi", "far", "far_lo", "far_hi", "timing_med", "timing_lo", "timing_hi"]
    err = max(float(np.nanmax(np.abs(m[c] - m[c + "_ref"]))) for c in cols)
    refp = pd.read_csv(os.path.join(EW.OUT, "paired.csv")); refp = refp[refp.budget == 1.0]
    minep = P[(P.rule == "prespecified") & (P.pre == 60) & (P.post == 120)]
    mp = refp.merge(minep, on=["group", "comp"], suffixes=("_ref", ""))
    errp = max(float(np.nanmax(np.abs(mp[c] - mp[c + "_ref"]))) for c in ["dpod", "dpod_lo", "dpod_hi", "dfar", "dfar_lo", "dfar_hi"])
    print(f"reproduction check: max abs difference summary {err:.2e}, paired {errp:.2e}")
    assert len(m) == len(ref) and len(mp) == len(refp) and err < 1e-9 and errp < 1e-9


if __name__ == "__main__":
    main()
