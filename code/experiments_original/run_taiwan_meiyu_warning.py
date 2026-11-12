"""
run_taiwan_meiyu_warning.py
===========================
REAL-EVENT Mei-yu (plum-rain, 梅雨) onset study.

Unlike a typhoon --- whose rainfall is directly observed and therefore
carries no warning lead --- the Mei-yu onset is a synoptic REGIME TRANSITION:
the seasonal shift from the spring pattern into the sustained East-Asian
frontal-rain regime over Taiwan in roughly mid-May to mid-June. It is the wet
analogue of the slow-onset drought: a change in the rainfall-generating
process rather than a single visible spike. The operational question is the
same as for drought:

    Does the RM-defect detector flag the Mei-yu regime transition EARLIER
    than an operational precipitation-accumulation index, at matched ARL_0?

Design (same 12 stations and archive as the drought study)
----------------------------------------------------------
  Train      : 2012-2016 daily precipitation (one model per station/seed).
  Per test year Y in 2017-2022 (each year's Mei-yu onset = one event):
    Spring null  : 1 Mar -- 30 Apr of year Y, the pre-Mei-yu baseline used to
                   calibrate every detector to a common ARL_0 (operationally
                   realistic: calibrate each spring on recent data, then watch
                   for the transition). No day-of-year deseasonalisation is
                   used --- the Mei-yu signal *is* seasonal, so we detect the
                   within-year shift against the spring baseline instead.
    Monitor      : 1 May -- 15 Jul of year Y; first alarm = detected onset.
  Detectors (UPWARD shift; matched ARL_0):
    cusum_rm : CUSUM on the standardised log RM defect (the regime shift
               perturbs the hidden-state path).
    cusum_pr : CUSUM on a 15-day precipitation accumulation -- the operational
               wet-onset baseline (Mei-yu is a multi-week sustained regime).
    raw_pr   : raw threshold on the 15-day accumulation.
  Lead = alarm(precip) - alarm(RM-defect); >0 = RM earlier.
  Onset reference (context only): first monitor day whose 15-day accumulation
  exceeds the 80th percentile of the spring 15-day accumulation.

  Repeated over 10 seeds; pooled over (station, year, seed); reported per
  location, per year, and integrated.

Usage
  /opt/anaconda3/bin/python run_taiwan_meiyu_warning.py [--seeds 10]
"""

import argparse
import importlib.util
import json
import os
import sys
from datetime import date

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE        = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(os.path.dirname(HERE))
BASE_DIR    = os.path.dirname(SCRIPTS_DIR)
EXP_DIR     = os.path.join(BASE_DIR, "experiments")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
os.makedirs(EXP_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "common"))

DEFAULT_CSV = os.path.join(
    BASE_DIR, "Data Sets", "raingel_drought_2012_2022",
    "raingel", "daily_precip_long.csv")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

wsr = _load("wsr", os.path.join(HERE, "run_warning_sr.py"))
drt = _load("drt", os.path.join(HERE, "run_taiwan_drought_warning.py"))

WINDOW       = drt.WINDOW
MAX_MISSING_FRAC = drt.MAX_MISSING_FRAC
STATION_NAMES = drt.STATION_NAMES
to_date      = drt.to_date

TRAIN_END    = date(2016, 12, 31)
TEST_YEARS   = list(range(2017, 2023))     # 2017..2022
ACC_WINDOW   = 15                          # precip accumulation (days)
TARGET_ARL0  = 60.0                        # ~1 false alarm per 2-month spring
HEAVY_PCTILE = 80                          # onset reference percentile


def spring_mon_masks(win_dates, year):
    spring = np.array([(d.year == year and d.month in (3, 4)) for d in win_dates])
    mon = np.array([(d.year == year and ((d.month == 5) or (d.month == 6)
                     or (d.month == 7 and d.day <= 15))) for d in win_dates])
    return spring, mon


def run_station(sid, dates, arr_mm, seed):
    name = STATION_NAMES.get(sid, sid)
    if float(np.mean(np.isnan(arr_mm))) > MAX_MISSING_FRAC:
        return None
    d = np.array(dates)
    train_len = int(np.sum(d <= TRAIN_END))
    series_norm, clean_mm = drt.preprocess_series(arr_mm, train_len)
    X, y = drt.make_windows(series_norm, WINDOW)
    win_dates = d[WINDOW:]
    if (win_dates <= TRAIN_END).sum() < 100:
        return None
    model = drt.train_model(X[(win_dates <= TRAIN_END)], y[(win_dates <= TRAIN_END)], seed)

    # streams over all window-aligned days (computed once, sliced per year)
    z_r = wsr._logdefect(drt.defect_stream(model, X))
    acc = np.convolve(clean_mm, np.ones(ACC_WINDOW), mode="full")[:len(clean_mm)][WINDOW:]

    per_year = []
    for Y in TEST_YEARS:
        spring, mon = spring_mon_masks(win_dates, Y)
        if spring.sum() < 40 or mon.sum() < 40:
            continue
        # calibrate on this year's spring (pre-Mei-yu) baseline
        rmu, rsd = z_r[spring].mean(), z_r[spring].std() + 1e-8
        pmu, psd = acc[spring].mean(), acc[spring].std() + 1e-8
        zr_null = (z_r[spring] - rmu) / rsd
        zp_null = (acc[spring] - pmu) / psd
        SIGN = +1.0
        h_rm = wsr.cusum_calibrate(zr_null, TARGET_ARL0, sign=SIGN)
        h_pr = wsr.cusum_calibrate(zp_null, TARGET_ARL0, sign=SIGN)

        zr_mon = (z_r[mon] - rmu) / rsd
        zp_mon = (acc[mon] - pmu) / psd
        mon_dates = win_dates[mon]
        # onset reference: first monitor day with 15-day acc > spring P80
        spring_p80 = float(np.percentile(acc[spring], HEAVY_PCTILE))
        heavy = np.where(acc[mon] > spring_p80)[0]
        onset_idx = int(heavy[0]) if len(heavy) else None

        a_rm = wsr.cusum_first_alarm(zr_mon, h_rm, sign=SIGN)
        a_pr = wsr.cusum_first_alarm(zp_mon, h_pr, sign=SIGN)
        rec = {"year": Y,
               "alarm_rm": str(mon_dates[a_rm]) if a_rm is not None else None,
               "alarm_pr": str(mon_dates[a_pr]) if a_pr is not None else None,
               "onset": str(mon_dates[onset_idx]) if onset_idx is not None else None,
               "det_rm": a_rm is not None, "det_pr": a_pr is not None,
               "lead": (a_pr - a_rm) if (a_rm is not None and a_pr is not None) else None,
               "delay_rm": (a_rm - onset_idx) if (a_rm is not None and onset_idx is not None) else None,
               "delay_pr": (a_pr - onset_idx) if (a_pr is not None and onset_idx is not None) else None}
        per_year.append(rec)
    return {"sid": sid, "name": name, "years": per_year}


def _med(a):
    return float(np.median(a)) if len(a) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    import data as datamod
    if not os.path.exists(args.csv):
        print(f"DATA NOT FOUND: {args.csv}"); sys.exit(1)
    M, date_strs, stations = datamod.load_precip_matrix(args.csv)
    dates = [to_date(s) for s in date_strs]
    seeds = [args.seed + 1000 * s for s in range(args.seeds)]
    print("Mei-yu (plum-rain) onset — real-event regime-transition lead study")
    print(f"  {len(stations)} stations, test years {TEST_YEARS[0]}-{TEST_YEARS[-1]}, "
          f"ARL0={TARGET_ARL0:.0f}, seeds={args.seeds}")
    print(f"  train<= {TRAIN_END}, spring null Mar-Apr, monitor May1-Jul15, "
          f"acc window {ACC_WINDOW}d\n")

    by_station = {sid: {"name": STATION_NAMES.get(sid, sid), "lead": [],
                        "det_rm": 0, "det_pr": 0, "n": 0} for sid in stations}
    by_year = {Y: {"lead": [], "det_rm": 0, "det_pr": 0, "n": 0} for Y in TEST_YEARS}
    pooled = []
    for si, seed in enumerate(seeds):
        for j, sid in enumerate(stations):
            r = run_station(sid, dates, M[:, j], seed + j)
            if r is None:
                continue
            for rec in r["years"]:
                bs = by_station[sid]; byr = by_year[rec["year"]]
                bs["n"] += 1; byr["n"] += 1
                if rec["det_rm"]: bs["det_rm"] += 1; byr["det_rm"] += 1
                if rec["det_pr"]: bs["det_pr"] += 1; byr["det_pr"] += 1
                if rec["lead"] is not None:
                    bs["lead"].append(rec["lead"]); byr["lead"].append(rec["lead"])
                    pooled.append(rec["lead"])
        print(f"  seed {seed} done ({si+1}/{len(seeds)})", flush=True)

    # ── per location ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-LOCATION Mei-yu onset lead of RM-defect over precip (median, days)")
    print("=" * 70)
    print(f"{'Station':12s} {'det RM':>8s} {'det PR':>8s} {'med lead':>9s} {'n':>4s}")
    print("-" * 70)
    per_station = []
    for sid in stations:
        b = by_station[sid]
        if b["n"] == 0:
            continue
        lm = _med(b["lead"])
        print(f"{b['name']:12s} {b['det_rm']:>5d}/{b['n']:<2d} {b['det_pr']:>5d}/{b['n']:<2d} "
              f"{('%+.0f'%lm) if lm is not None else '—':>9s} {len(b['lead']):>4d}")
        per_station.append({"sid": sid, "name": b["name"], "n": b["n"],
                            "det_rate_rm": b["det_rm"]/b["n"], "det_rate_pr": b["det_pr"]/b["n"],
                            "median_lead_days": lm, "leads": b["lead"]})

    # ── per year ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-YEAR Mei-yu onset lead (median over stations x seeds)")
    print("=" * 70)
    print(f"{'Year':6s} {'det RM':>8s} {'det PR':>8s} {'med lead':>9s} {'n both':>7s}")
    print("-" * 70)
    per_year_out = []
    for Y in TEST_YEARS:
        b = by_year[Y]
        if b["n"] == 0:
            continue
        lm = _med(b["lead"])
        print(f"{Y:<6d} {b['det_rm']:>5d}/{b['n']:<2d} {b['det_pr']:>5d}/{b['n']:<2d} "
              f"{('%+.0f'%lm) if lm is not None else '—':>9s} {len(b['lead']):>7d}")
        per_year_out.append({"year": Y, "n": b["n"],
                             "det_rate_rm": b["det_rm"]/b["n"], "det_rate_pr": b["det_pr"]/b["n"],
                             "median_lead_days": lm, "n_both": len(b["lead"])})

    pooled = np.array(pooled, dtype=float)
    st_meds = np.array([p["median_lead_days"] for p in per_station
                        if p["median_lead_days"] is not None], dtype=float)
    det_rm = float(np.mean([by_station[s]["det_rm"]/by_station[s]["n"]
                            for s in stations if by_station[s]["n"]]))
    det_pr = float(np.mean([by_station[s]["det_pr"]/by_station[s]["n"]
                            for s in stations if by_station[s]["n"]]))
    print("\n" + "=" * 70)
    print("INTEGRATED — Mei-yu onset regime-transition warning")
    print("=" * 70)
    print(f"  mean per-station RM detection rate: {det_rm:.0%}")
    print(f"  mean per-station PR detection rate: {det_pr:.0%}")
    if len(pooled):
        print(f"  pooled (station x year x seed):     n={len(pooled)}")
        print(f"  median pooled lead (days):          {np.median(pooled):+.1f}")
        print(f"  mean pooled lead (days):            {pooled.mean():+.1f} ± {pooled.std():.1f}")
        print(f"  fraction RM earlier:                {np.mean(pooled > 0):.0%}")
    if len(st_meds):
        print(f"  median of per-station median leads: {np.median(st_meds):+.1f}")
        print(f"  stations with positive median lead: "
              f"{int(np.sum(st_meds > 0))}/{len(st_meds)}")

    out = {
        "description": (
            "Real-event Mei-yu (plum-rain) onset regime-transition warning on "
            "real CWA daily station precipitation. For each test year an "
            "upward-shift CUSUM on the RM defect is compared with a CUSUM on a "
            "15-day precipitation accumulation, each calibrated to a common "
            "ARL_0 on that year's pre-Mei-yu spring (Mar-Apr) and monitored over "
            "May 1-Jul 15. Lead = alarm(precip) - alarm(RM); >0 = RM earlier. "
            "Pooled over 12 stations x 6 years x 10 seeds. Unlike a typhoon, "
            "Mei-yu onset is a regime change, so a lead is physically possible."),
        "config": {"window": WINDOW, "acc_window": ACC_WINDOW,
                   "target_arl0": TARGET_ARL0, "n_seeds": args.seeds,
                   "test_years": TEST_YEARS, "train_end": str(TRAIN_END)},
        "integrated": {
            "mean_det_rate_rm": det_rm, "mean_det_rate_pr": det_pr,
            "pooled_n": int(len(pooled)),
            "pooled_median_lead": _med(pooled),
            "pooled_mean_lead": float(pooled.mean()) if len(pooled) else None,
            "pooled_sd_lead": float(pooled.std()) if len(pooled) else None,
            "pooled_frac_rm_earlier": float(np.mean(pooled > 0)) if len(pooled) else None,
            "median_of_station_median_leads": _med(st_meds),
            "n_stations_positive_median": int(np.sum(st_meds > 0)) if len(st_meds) else 0,
        },
        "per_station": per_station,
        "per_year": per_year_out,
    }
    jpath = os.path.join(EXP_DIR, "taiwan_meiyu_warning_results.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {jpath}")


if __name__ == "__main__":
    main()
