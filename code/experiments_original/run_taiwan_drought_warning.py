"""
run_taiwan_drought_warning.py
=============================
REAL-EVENT early-warning lead-time study on the 2020--2021 Taiwan drought
(the worst in 56 years; the 2020 typhoon season produced no landfall for
the first time in ~60 years, and June-2020--May-2021 rainfall was about a
third of normal).

This is the real-data analogue of run_warning_sr.py, which runs the same
Shiryaev--Roberts / CUSUM detector machinery on SYNTHETIC injected change
points. Here the detectors are driven by REAL daily CWA station
precipitation through the real drought onset, and we ask the paper's
operational question directly:

    Does the RM-defect detector raise the drought alarm EARLIER than an
    operational SPI-style precipitation detector, at matched ARL_0?

Design (per station, stations act as natural replications of one regional event)
--------------------------------------------------------------------------------
  Train      : 2012-2018 daily precipitation (pre-event).
  Null calib : 2019 (a near-normal year) -- every detector is calibrated to
               a common ARL_0 on this no-change window.
  Monitor    : 2020-01-01 .. 2021-08-31 (the onset and peak of the drought).
  Reference onset : 2020-06-01 (start of the documented June-2020 deficit
               window); used only to label alarms as real detections vs.
               pre-onset false alarms. The headline lead metric is the
               PAIRED difference alarm(SPI) - alarm(RM-defect) and does not
               depend on this date.

Detectors (all calibrated to the SAME ARL_0 on the 2019 null window)
  - cusum_rm  : one-sided CUSUM on the standardised log RM-defect
                r_t = || g_phi(h_{t+1}) - h_t ||  (downward shift: the dry,
                highly regular drought regime makes the hidden path more
                backward-coherent, lowering the defect).
  - cusum_spi : one-sided CUSUM on the standardised 90-day precipitation
                accumulation (SPI-3 proxy), downward shift -- the operational
                baseline.
  - raw_spi   : raw threshold on the same SPI-3 proxy.
  - sr_rm     : Shiryaev--Roberts on the RM defect (reported for completeness).

Model: model_ext.RMRNNJoint (Elman cell, full BPTT, RM lambda 0.1->0.01),
the same model used in run_taiwan_precip.py.

Data: produced by
  fetch_taiwan_hydromet.py --source raingel \
     --stations 467410,467440,467480,467490,467530,467550,467650,467420,\
467770,466920,466990,467610 --years 2012-2022 --outdir <...>

Outputs
  experiments/taiwan_drought_warning_results.json
  figures/taiwan_drought_warning.pdf
  stdout : per-station alarm dates and the aggregate lead table.

Usage
  /opt/anaconda3/bin/python run_taiwan_drought_warning.py [--csv PATH] [--seed 42]
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

# ── paths ─────────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(os.path.dirname(HERE))
BASE_DIR    = os.path.dirname(SCRIPTS_DIR)
EXP_DIR     = os.path.join(BASE_DIR, "experiments")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "common"))

DEFAULT_CSV = os.path.join(
    BASE_DIR, "Data Sets", "raingel_drought_2012_2022",
    "raingel", "daily_precip_long.csv")

# Reuse the genuine SR/CUSUM detector functions from run_warning_sr.py.
_spec = importlib.util.spec_from_file_location(
    "wsr", os.path.join(HERE, "run_warning_sr.py"))
wsr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wsr)

# ── configuration ─────────────────────────────────────────────────────────────
WINDOW       = 30
HIDDEN_DIM   = 32
N_EPOCHS     = 200
WARMUP       = 5
LAMBDA_RM    = 0.1
LAMBDA_DECAY = 0.1
LR           = 5e-3
BATCH_SIZE   = 32
SPI_WINDOW   = 90               # 3-month accumulation (SPI-3 proxy)
TARGET_ARL0  = 365.0            # ~1 false alarm per year on the null
MAX_MISSING_FRAC = 0.25

TRAIN_END   = date(2016, 12, 31)         # train 2012-2016
NULL_START  = date(2017, 1, 1)           # 3-year null for stable ARL_0
NULL_END    = date(2019, 12, 31)
MON_START   = date(2020, 1, 1)
MON_END     = date(2021, 8, 31)
ONSET_REF   = date(2020, 6, 1)           # documented deficit-window start
DESEASON_END = date(2019, 12, 31)        # climatology from pre-2020 years

STATION_NAMES = {
    "466920": "Taipei",    "466990": "Hualien",   "467410": "Tainan",
    "467420": "467420",    "467440": "Kaohsiung", "467480": "467480",
    "467490": "Taichung",  "467530": "Alishan",   "467550": "Yushan",
    "467610": "Chenggong", "467650": "SunMoonLake","467770": "467770",
}


def to_date(s: str) -> date:
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def preprocess_series(arr_mm, train_len):
    """Interpolate short gaps, log1p, standardise on the training split."""
    arr = arr_mm.astype(float).copy()
    nan_idx = np.where(np.isnan(arr))[0]
    for i in nan_idx:
        left, right = i - 1, i + 1
        while right < len(arr) and np.isnan(arr[right]):
            right += 1
        if left >= 0 and right < len(arr) and (right - left) <= 5:
            arr[i] = arr[left] + (arr[right] - arr[left]) * (i - left) / (right - left)
    arr = np.clip(np.where(np.isnan(arr), 0.0, arr), 0.0, None)
    arr_log = np.log1p(arr)
    tr = arr_log[:train_len]
    mu, sig = tr.mean(), tr.std().clip(min=1e-8)
    return (arr_log - mu) / sig, arr  # standardised series, and clean mm


def make_windows(series, window):
    N = len(series) - window
    X = np.stack([series[i:i + window, np.newaxis] for i in range(N)])
    y = series[window:, np.newaxis]
    return X, y


def defect_stream(model, X):
    """Per-day RM defect r_t = || g_phi(h_{t+1}) - h_t || using the terminal
    pair of each 30-day window (matches the windowed training protocol).
    Returns array of length len(X), aligned to the window-end day."""
    h_all, _ = model._forward(X)            # (N, WINDOW, p)
    h_last = h_all[:, -1, :]                # h_{t+1}
    h_prev = h_all[:, -2, :]                # h_t
    g_last, _ = model.proj.forward(h_last)  # g(h_{t+1}) approximates h_t
    return np.linalg.norm(g_last - h_prev, axis=1)   # (N,)


def deseason(x, win_dates, ref_mask):
    """Remove the day-of-year climatology (estimated on ref_mask days, i.e.
    pre-drought years) so that the drought anomaly is not masked by the strong
    seasonal cycle of precipitation and of the hidden-state dynamics. Returns
    the deseasonalised series x - clim(doy)."""
    doy = np.array([min(d.timetuple().tm_yday, 365) for d in win_dates])
    clim = np.full(366, np.nan)
    for dd in range(1, 366):
        sel = ref_mask & (doy == dd)
        if sel.any():
            clim[dd] = x[sel].mean()
    idx = np.arange(366)
    good = ~np.isnan(clim)
    clim = np.interp(idx, idx[good], clim[good], period=365)
    k = 15  # circular +/-15-day smoothing of the climatology
    ext = np.concatenate([clim[-k:], clim, clim[:k]])
    sm = np.convolve(ext, np.ones(2 * k + 1) / (2 * k + 1), "same")[k:-k]
    return x - sm[doy]


def train_model(X_tr, y_tr, seed):
    from model_ext import RMRNNJoint
    model = RMRNNJoint(1, HIDDEN_DIM, 1, lambda_rm=LAMBDA_RM, seed=seed)
    rng = np.random.RandomState(seed)
    n_tr = len(X_tr)
    for epoch in range(N_EPOCHS):
        frac = max(0, epoch - WARMUP) / max(N_EPOCHS - WARMUP - 1, 1)
        model.lambda_rm = LAMBDA_RM * (LAMBDA_DECAY ** frac) if epoch >= WARMUP else 0.0
        idx = rng.choice(n_tr, size=min(BATCH_SIZE, n_tr), replace=False)
        model.train_step(X_tr[idx], y_tr[idx], lr=LR)
    return model


def run_station(sid, dates, arr_mm, seed):
    name = STATION_NAMES.get(sid, sid)
    miss = float(np.mean(np.isnan(arr_mm)))
    if miss > MAX_MISSING_FRAC:
        print(f"  {name:11s} ({sid})  SKIPPED — {miss:.0%} missing")
        return None

    # chronological index helpers
    d = np.array(dates)  # list of date objects
    train_len = int(np.sum(d <= TRAIN_END))
    series_norm, clean_mm = preprocess_series(arr_mm, train_len)

    X, _y = make_windows(series_norm, WINDOW)
    win_dates = d[WINDOW:]                    # date for each window-end day
    if len(X) < 200:
        print(f"  {name:11s} ({sid})  SKIPPED — too few windows")
        return None

    # train on pre-event windows only
    tr_mask = win_dates <= TRAIN_END
    if tr_mask.sum() < 100:
        print(f"  {name:11s} ({sid})  SKIPPED — too few training windows")
        return None
    model = train_model(X[tr_mask], _y[tr_mask], seed)

    # ── defect and SPI-3 streams over the full window-aligned timeline ────────
    r = defect_stream(model, X)                       # RM defect, per day
    z_r = wsr._logdefect(r)                            # Gaussianise
    spi = np.convolve(clean_mm, np.ones(SPI_WINDOW), mode="full")[:len(clean_mm)]
    spi = spi[WINDOW:]                                # align to win_dates

    # Remove the seasonal cycle (climatology from pre-2020 years), otherwise the
    # drought anomaly is swamped by the wet-/dry-season swing of both streams.
    ref_mask = win_dates <= DESEASON_END
    z_r = deseason(z_r, win_dates, ref_mask)
    spi = deseason(spi, win_dates, ref_mask)

    # multi-year null window (2017-2019) statistics + calibration
    null_mask = (win_dates >= NULL_START) & (win_dates <= NULL_END)
    mon_mask  = (win_dates >= MON_START) & (win_dates <= MON_END)
    if null_mask.sum() < 60 or mon_mask.sum() < 60:
        print(f"  {name:11s} ({sid})  SKIPPED — short null/monitor window")
        return None

    rmu, rsd = z_r[null_mask].mean(), z_r[null_mask].std() + 1e-8
    smu, ssd = spi[null_mask].mean(), spi[null_mask].std() + 1e-8
    zr_null = (z_r[null_mask] - rmu) / rsd
    zs_null = (spi[null_mask] - smu) / ssd

    # downward-shift monitoring: drought lowers both the RM defect and SPI-3
    SIGN = -1.0
    B_star, m0, _, _ = wsr.calibrate_sr_B(SIGN * zr_null, TARGET_ARL0)
    h_rm  = wsr.cusum_calibrate(SIGN * zr_null, TARGET_ARL0, sign=1.0)
    h_spi = wsr.cusum_calibrate(zs_null, TARGET_ARL0, sign=SIGN)
    c_spi = wsr.raw_calibrate(zs_null, TARGET_ARL0, sign=SIGN)

    # monitoring streams
    zr_mon = (z_r[mon_mask] - rmu) / rsd
    zs_mon = (spi[mon_mask] - smu) / ssd
    mon_dates = win_dates[mon_mask]

    def alarm_date(idx):
        return mon_dates[idx] if idx is not None else None

    a_sr  = wsr.sr_first_alarm(SIGN * zr_mon, B_star, m0)
    a_rm  = wsr.cusum_first_alarm(SIGN * zr_mon, h_rm, sign=1.0)
    a_spi = wsr.cusum_first_alarm(zs_mon, h_spi, sign=SIGN)
    a_raw = wsr.raw_first_alarm(zs_mon, c_spi, sign=SIGN)

    def days_after_onset(dt):
        return (dt - ONSET_REF).days if dt is not None else None

    res = {
        "sid": sid, "name": name, "n_train_win": int(tr_mask.sum()),
        "alarm": {
            "sr_rm":   str(alarm_date(a_sr))  if a_sr  is not None else None,
            "cusum_rm":  str(alarm_date(a_rm))  if a_rm  is not None else None,
            "cusum_spi": str(alarm_date(a_spi)) if a_spi is not None else None,
            "raw_spi":   str(alarm_date(a_raw)) if a_raw is not None else None,
        },
        "delay_after_onset_days": {
            "cusum_rm":  days_after_onset(alarm_date(a_rm)),
            "cusum_spi": days_after_onset(alarm_date(a_spi)),
        },
    }
    # paired lead (positive = RM-defect detector earlier than SPI baseline)
    if a_rm is not None and a_spi is not None:
        res["lead_days_rm_vs_spi"] = (alarm_date(a_spi) - alarm_date(a_rm)).days
    else:
        res["lead_days_rm_vs_spi"] = None
    return res


def _med(a):
    return float(np.median(a)) if len(a) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=10,
                    help="number of random seeds for the robustness check")
    # Optional date-window overrides (defaults reproduce the 2020-21 Taiwan run).
    ap.add_argument("--train-end")
    ap.add_argument("--null-start")
    ap.add_argument("--null-end")
    ap.add_argument("--mon-start")
    ap.add_argument("--mon-end")
    ap.add_argument("--onset-ref")
    ap.add_argument("--deseason-end")
    ap.add_argument("--tag", default="taiwan",
                    help="label for output files and the report header")
    ap.add_argument("--no-names", action="store_true",
                    help="do not apply the Taiwan station-name lookup")
    args = ap.parse_args()

    # Apply window overrides to the module-level globals used throughout.
    global TRAIN_END, NULL_START, NULL_END, MON_START, MON_END, ONSET_REF, DESEASON_END, STATION_NAMES
    if args.train_end:    TRAIN_END    = to_date(args.train_end)
    if args.null_start:   NULL_START   = to_date(args.null_start)
    if args.null_end:     NULL_END     = to_date(args.null_end)
    if args.mon_start:    MON_START    = to_date(args.mon_start)
    if args.mon_end:      MON_END      = to_date(args.mon_end)
    if args.onset_ref:    ONSET_REF    = to_date(args.onset_ref)
    if args.deseason_end: DESEASON_END = to_date(args.deseason_end)
    if args.no_names:     STATION_NAMES = {}

    import data as datamod
    if not os.path.exists(args.csv):
        print(f"DATA NOT FOUND: {args.csv}")
        sys.exit(1)

    M, date_strs, stations = datamod.load_precip_matrix(args.csv)
    dates = [to_date(s) for s in date_strs]
    seeds = [args.seed + 1000 * s for s in range(args.seeds)]
    print("2020--2021 Taiwan drought — real-event early-warning lead study")
    print(f"  {len(date_strs)} days ({date_strs[0]}..{date_strs[-1]}), "
          f"{len(stations)} stations, ARL0={TARGET_ARL0:.0f}, seeds={args.seeds}")
    print(f"  train<= {TRAIN_END}, null {NULL_START}..{NULL_END}, "
          f"monitor {MON_START}..{MON_END}, onset ref {ONSET_REF}\n")

    def dnum(s):
        return (to_date(s) - MON_START).days if s else None

    # per-station accumulators across seeds
    acc = {sid: {"name": STATION_NAMES.get(sid, sid),
                 "rm_day": [], "spi_day": [], "lead": [],
                 "det_rm": 0, "det_spi": 0, "n": 0} for sid in stations}
    pooled_leads = []          # every (station, seed) lead where both detect

    for si, seed in enumerate(seeds):
        for j, sid in enumerate(stations):
            r = run_station(sid, dates, M[:, j], seed + j)
            if r is None:
                continue
            a = acc[sid]; a["n"] += 1
            rmd = dnum(r["alarm"]["cusum_rm"]); spid = dnum(r["alarm"]["cusum_spi"])
            if rmd is not None:  a["det_rm"] += 1;  a["rm_day"].append(rmd)
            if spid is not None: a["det_spi"] += 1; a["spi_day"].append(spid)
            if r["lead_days_rm_vs_spi"] is not None:
                a["lead"].append(r["lead_days_rm_vs_spi"])
                pooled_leads.append(r["lead_days_rm_vs_spi"])
        print(f"  seed {seed} done ({si+1}/{len(seeds)})", flush=True)

    evaluated = [sid for sid in stations if acc[sid]["n"] > 0]

    # ── per-location (separate stations) summary ──────────────────────────────
    print("\n" + "=" * 78)
    print("PER-LOCATION lead of RM-defect over SPI-3 (median over "
          f"{args.seeds} seeds; +days = RM earlier)")
    print("=" * 78)
    print(f"{'Station':12s} {'det RM':>7s} {'det SPI':>8s} "
          f"{'med RM alarm':>13s} {'med SPI alarm':>14s} {'med lead':>9s}")
    print("-" * 78)
    per_station = []
    for sid in evaluated:
        a = acc[sid]
        rm_med = _med(a["rm_day"]); spi_med = _med(a["spi_day"]); lead_med = _med(a["lead"])
        def fmt_day(x):
            return str(MON_START + __import__("datetime").timedelta(days=int(x))) if x is not None else "—"
        print(f"{a['name']:12s} {a['det_rm']:>4d}/{a['n']:<2d} {a['det_spi']:>5d}/{a['n']:<2d} "
              f"{fmt_day(rm_med):>13s} {fmt_day(spi_med):>14s} "
              f"{('%+d' % lead_med) if lead_med is not None else '—':>9s}")
        per_station.append({
            "sid": sid, "name": a["name"], "n_seeds": a["n"],
            "det_rate_rm": a["det_rm"] / a["n"], "det_rate_spi": a["det_spi"] / a["n"],
            "median_rm_alarm": fmt_day(rm_med), "median_spi_alarm": fmt_day(spi_med),
            "median_lead_days": lead_med,
            "leads": a["lead"],
        })

    # ── integrated (island-wide) summary ──────────────────────────────────────
    pooled = np.array(pooled_leads, dtype=float)
    station_med_leads = np.array([p["median_lead_days"] for p in per_station
                                  if p["median_lead_days"] is not None], dtype=float)
    det_rm_rate = float(np.mean([acc[s]["det_rm"] for s in evaluated]) /
                        max(args.seeds, 1))
    det_spi_rate = float(np.mean([acc[s]["det_spi"] for s in evaluated]) /
                         max(args.seeds, 1))
    print("\n" + "=" * 78)
    print("INTEGRATED (island-wide) — RM-defect vs SPI-3 CUSUM, matched ARL_0")
    print("=" * 78)
    print(f"  stations evaluated:                 {len(evaluated)}")
    print(f"  mean per-station RM detection rate: {det_rm_rate:.0%}")
    print(f"  mean per-station SPI detection rate:{det_spi_rate:.0%}")
    if len(pooled):
        print(f"  pooled (station x seed) leads:      n={len(pooled)}")
        print(f"  median pooled lead (days):          {np.median(pooled):+.1f}")
        print(f"  mean pooled lead (days):            {pooled.mean():+.1f} ± {pooled.std():.1f}")
        print(f"  fraction RM earlier (pooled):       {np.mean(pooled > 0):.0%}")
    if len(station_med_leads):
        print(f"  median of per-station median leads: {np.median(station_med_leads):+.1f}")
        print(f"  stations with positive median lead: "
              f"{int(np.sum(station_med_leads > 0))}/{len(station_med_leads)}")

    out = {
        "description": (
            "Seed-robust real-event early-warning lead study on the 2020-2021 "
            "Taiwan drought using real CWA daily station precipitation "
            "(Raingel/CODiS archive). Per station, a fresh RMRNNJoint is trained "
            "for each of N seeds; detectors (CUSUM/SR on the deseasonalised RM "
            "defect; CUSUM/raw on a deseasonalised 90-day SPI-3 proxy) are "
            "calibrated to a common ARL_0 on a 2017-2019 null window and "
            "monitored over 2020-01..2021-08. Lead = alarm(SPI-3) - "
            "alarm(RM-defect); positive = RM detector earlier. Reported both "
            "per location and integrated island-wide."),
        "config": {
            "window": WINDOW, "hidden_dim": HIDDEN_DIM, "n_epochs": N_EPOCHS,
            "lambda_rm": LAMBDA_RM, "lambda_decay": LAMBDA_DECAY, "lr": LR,
            "spi_window": SPI_WINDOW, "target_arl0": TARGET_ARL0,
            "n_seeds": args.seeds, "base_seed": args.seed,
            "train_end": str(TRAIN_END), "null": [str(NULL_START), str(NULL_END)],
            "monitor": [str(MON_START), str(MON_END)], "onset_ref": str(ONSET_REF),
            "deseasonalised": True,
        },
        "n_stations": len(evaluated),
        "integrated": {
            "mean_det_rate_rm": det_rm_rate, "mean_det_rate_spi": det_spi_rate,
            "pooled_n": int(len(pooled)),
            "pooled_median_lead": _med(pooled),
            "pooled_mean_lead": float(pooled.mean()) if len(pooled) else None,
            "pooled_sd_lead": float(pooled.std()) if len(pooled) else None,
            "pooled_frac_rm_earlier": float(np.mean(pooled > 0)) if len(pooled) else None,
            "median_of_station_median_leads": _med(station_med_leads),
            "n_stations_positive_median": int(np.sum(station_med_leads > 0)) if len(station_med_leads) else 0,
        },
        "per_station": per_station,
    }
    jpath = os.path.join(EXP_DIR, f"{args.tag}_drought_warning_results.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {jpath}")

    # ── figure: per-station median alarm day (RM vs SPI) across seeds ──────────
    plotable = [p for p in per_station
                if p["median_lead_days"] is not None]
    plotable.sort(key=lambda p: p["median_lead_days"], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    yy = np.arange(len(plotable))
    for i, p in enumerate(plotable):
        rmd = dnum(p["median_rm_alarm"]); spid = dnum(p["median_spi_alarm"])
        ax.plot([rmd, spid], [i, i], color="gray", lw=1, alpha=0.6, zorder=1)
        ax.scatter(rmd, i, c="#1f77b4", zorder=3)
        ax.scatter(spid, i, c="#ff7f0e", marker="s", zorder=3)
    ax.scatter([], [], c="#1f77b4", label="RM-defect CUSUM")
    ax.scatter([], [], c="#ff7f0e", marker="s", label="SPI-3 CUSUM")
    ax.axvline((ONSET_REF - MON_START).days, color="k", ls="--", lw=1,
               label=f"deficit ref ({ONSET_REF})")
    ax.set_yticks(yy); ax.set_yticklabels([p["name"] for p in plotable], fontsize=8)
    ax.set_xlabel(f"days after {MON_START} (median first-alarm over {args.seeds} seeds)")
    ax.set_title("2020--2021 Taiwan drought: RM-defect vs SPI-3 first alarm "
                 "(matched ARL$_0$, seed-median)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fpath = os.path.join(FIG_DIR, f"{args.tag}_drought_warning.pdf")
    fig.savefig(fpath); plt.close(fig)
    print(f"Saved: {fpath}")


if __name__ == "__main__":
    main()
