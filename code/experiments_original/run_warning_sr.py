"""
run_warning_sr.py
=================
Genuine Shiryaev-Roberts (SR) warning-layer experiment for the
RM_Precipitation paper (Sections 4 and the ARL_0 calibration figure).

WHY THIS SCRIPT EXISTS
----------------------
The forecast-skill tables (Sections 6) were produced by genuine 1,000-rep
runs (run_section5.py).  The *warning* layer of the paper -- the SR
threshold B*, the ARL_0 calibration figure, and the FAR/detection/lead
numbers -- previously had NO generating code and NO data file; the only
SR thresholds on disk (results.json, ~5.5e6) came from a 6-epoch toy run,
and arl_calibration.pdf was a hand-drawn "pilot".  This script computes
those quantities for real and reproducibly.

It reuses the genuine RMRNN backbone (GRU + backward projector g_phi,
rm_defects), the synthetic ClimateSpec-style simulator, and the
ARL_0-calibration philosophy already present in run_hybrid_extension.py,
and adds:
  * a proper Shiryaev-Roberts recursion R_t = (1+R_{t-1}) Lambda_t,
  * a Monte-Carlo ARL_0(B) calibration -> B* at a target ARL_0,
  * two regimes: a sustained DROUGHT change and a FLASH-FLOOD change,
  * matched-ARL_0 comparators (CUSUM on the operational indicator; a raw
    indicator threshold), so detection rate / lead are compared fairly.

IMPORTANT HONESTY NOTE
----------------------
All precipitation here is SYNTHETIC (ClimateSpec generators); these are
testbeds, not real CWA / CHIRPS / ERA5-Land observations, and the two
"case studies" are injected change-points, not the real 2020-21 drought
or 2023 Typhoon Haikui.  Numbers produced here should be reported as
synthetic-pilot warning results.

Usage
-----
    python run_warning_sr.py --reps 200            # full-ish
    python run_warning_sr.py --reps 20 --quick     # smoke test
"""
from __future__ import annotations
import argparse, importlib.util, json, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import the genuine machinery from run_hybrid_extension.py (handles the
# space/digit characters in the absolute path that block normal imports).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "rhe", str(_HERE / "run_hybrid_extension.py"))
rhe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rhe)

STEPS_PER_YEAR = rhe.STEPS_PER_YEAR
CONTEXT        = rhe.CONTEXT
HIDDEN         = rhe.HIDDEN
N_EPOCHS       = rhe.N_EPOCHS
BATCH_LEN      = rhe.BATCH_LEN
RMRNN          = rhe.RMRNN
simulate_multivar = rhe.simulate_multivar
inject_drought    = rhe.inject_drought
build_features    = rhe.build_features


# ---------------------------------------------------------------------------
# Flash-flood regime injection (heavy-rain onset), analogue of inject_drought
# ---------------------------------------------------------------------------

def inject_flood(data: np.ndarray, t_change: int, seed: int,
                 severity: float = 1.0) -> np.ndarray:
    """
    Inject a flash-flood regime starting at t_change.  Mirrors a typhoon
    outer-rainband onset: a short, intense wet excursion across channels.
      - Rain: large positive excursion (extreme accumulation)
      - Temp: -1.5 sigma (cloud/rain cooling)
      - SM:   +0.10 (rapid wetting)
      - u10/v10: strong cyclonic anomaly (moisture flux)
    The joint excursion drives hidden-state trajectories outside the
    pre-change training distribution, inflating r_t at flood onset.
    """
    rng = np.random.RandomState(seed + 7777)
    d = data.copy()
    T = data.shape[0]
    flood_len = T - t_change
    if flood_len <= 0:
        return d
    # short, intense event: ramp up over 2 days, sustained heavy regime
    ramp = np.clip(np.arange(flood_len) / 2.0, 0.0, 1.0)
    rain_extra = severity * (8.0 * ramp + rng.gamma(1.5, 6.0, size=flood_len) * ramp)
    d[t_change:, 0] = np.clip(data[t_change:, 0] + rain_extra, 0.0, None)
    d[t_change:, 1] = data[t_change:, 1] - (1.5 * ramp + 0.3 * rng.randn(flood_len))
    d[t_change:, 2] = np.clip(data[t_change:, 2] + 0.10 * ramp, 0.02, 0.60)
    d[t_change:, 3] = data[t_change:, 3] - (3.0 * ramp + 0.5 * rng.randn(flood_len))
    d[t_change:, 4] = data[t_change:, 4] + (3.0 * ramp + 0.5 * rng.randn(flood_len))
    return d.astype(np.float32)


# ---------------------------------------------------------------------------
# Indicators used by the comparators (operational baselines)
# ---------------------------------------------------------------------------

def spi3_proxy(rain: np.ndarray, window: int, mu0: float, sd0: float) -> np.ndarray:
    """Standardised 3-month accumulation (SPI-3 proxy): rolling sum z-scored
    on the null statistics (mu0, sd0)."""
    acc = np.convolve(rain, np.ones(window), mode="full")[:len(rain)]
    return (acc - mu0) / sd0


# ---------------------------------------------------------------------------
# Shiryaev-Roberts machinery on the standardised RM defect
# ---------------------------------------------------------------------------
# The raw RM defect r_t = ||h_t - g_phi(h_{t+1})||^2 is a positive,
# right-skewed (chi-square-like) quantity.  We Gaussianise by a log
# transform before standardising, so that the SR score
#   Lambda_t = exp(delta z_t - delta^2/2),   z_t = (log r_t - mu0)/sd0
# has approximately unit mean under the null (E_inf[Lambda] ~ 1) and the
# SR statistic R_t = (1+R_{t-1}) Lambda_t behaves like the classical SR
# detector, for which ARL_0 grows ~linearly in the threshold B.
SR_DELTA = 1.0


def _logdefect(defects: np.ndarray) -> np.ndarray:
    return np.log(np.clip(defects, 1e-12, None))


def sr_null_mean(null_z: np.ndarray, delta: float = SR_DELTA) -> float:
    """Empirical null mean of the raw score exp(delta z - delta^2/2).
    Dividing the score by this makes E_null[Lambda]=1, so that R_t - t is
    (approximately) a martingale and ARL_0 grows ~linearly in B."""
    return float(np.mean(np.exp(delta * null_z - 0.5 * delta * delta))) + 1e-12


def sr_score(z: np.ndarray, m0: float, delta: float = SR_DELTA) -> np.ndarray:
    return np.exp(delta * z - 0.5 * delta * delta) / m0


def sr_arl0_curve(null_z: np.ndarray, B_grid: np.ndarray, m0: float,
                  delta: float = SR_DELTA) -> np.ndarray:
    """
    Monte-Carlo ARL_0(B) on a long null score stream using the standard
    reset-on-crossing estimator: run the SR recursion; each time R_t >= B,
    record the run length and reset R=0.  ARL_0(B) = mean run length.
    Returns ARL_0 for each B in B_grid.
    """
    Lam = sr_score(null_z, m0, delta)
    n = len(Lam)
    arl = np.empty(len(B_grid))
    for j, B in enumerate(B_grid):
        R = 0.0
        last = 0
        rls = []
        for t in range(n):
            R = (1.0 + R) * Lam[t]
            if R >= B:
                rls.append(t - last + 1)
                R = 0.0
                last = t + 1
        arl[j] = np.mean(rls) if rls else float(n)
    return arl


def calibrate_sr_B(null_z: np.ndarray, target_arl0: float,
                   delta: float = SR_DELTA
                   ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Return (B_star, m0, B_grid, arl_curve). B_star solves ARL_0(B)=target
    by log-log interpolation on the Monte-Carlo curve."""
    m0 = sr_null_mean(null_z, delta)
    B_grid = np.geomspace(1.5, max(20.0, 6.0 * target_arl0), 24)
    arl = sr_arl0_curve(null_z, B_grid, m0, delta)
    lb, la = np.log(B_grid), np.log(np.clip(arl, 1e-6, None))
    tgt = np.log(target_arl0)
    if tgt <= la[0]:
        B_star = float(B_grid[0])
    elif tgt >= la[-1]:
        B_star = float(B_grid[-1])
    else:
        B_star = float(np.exp(np.interp(tgt, la, lb)))
    return B_star, m0, B_grid, arl


def sr_first_alarm(z: np.ndarray, B: float, m0: float,
                   delta: float = SR_DELTA) -> Optional[int]:
    R = 0.0
    for t in range(len(z)):
        R = (1.0 + R) * np.exp(delta * z[t] - 0.5 * delta * delta) / m0
        if R >= B:
            return t
    return None


# ---------------------------------------------------------------------------
# CUSUM and raw-threshold comparators (calibrated to the SAME ARL_0)
# ---------------------------------------------------------------------------

def cusum_calibrate(null_ind: np.ndarray, target_arl0: float,
                    k: float = 0.5, sign: float = 1.0) -> float:
    """One-sided CUSUM threshold h calibrated to ARL_0 by reset-on-crossing
    on the null indicator stream.  sign=+1 detects an upward shift, -1 a
    downward shift."""
    s = sign * null_ind
    grid = np.geomspace(0.5, 200.0, 40)
    def arl_of(h):
        S = 0.0; last = 0; rls = []
        for t in range(len(s)):
            S = max(0.0, S + s[t] - k)
            if S >= h:
                rls.append(t - last + 1); S = 0.0; last = t + 1
        return np.mean(rls) if rls else float(len(s))
    arls = np.array([arl_of(h) for h in grid])
    la = np.log(np.clip(arls, 1e-6, None)); lh = np.log(grid)
    tgt = np.log(target_arl0)
    if tgt <= la[0]:   return float(grid[0])
    if tgt >= la[-1]:  return float(grid[-1])
    return float(np.exp(np.interp(tgt, la, lh)))


def cusum_first_alarm(ind: np.ndarray, h: float, k: float = 0.5,
                      sign: float = 1.0) -> Optional[int]:
    S = 0.0
    for t in range(len(ind)):
        S = max(0.0, S + sign * ind[t] - k)
        if S >= h:
            return t
    return None


def raw_calibrate(null_ind: np.ndarray, target_arl0: float,
                  sign: float = 1.0) -> float:
    """Raw-threshold detector calibrated to ARL_0: choose c so the expected
    run length to first crossing of sign*ind>=c on null equals target."""
    s = sign * null_ind
    q = 100.0 * (1.0 - 1.0 / max(target_arl0, 2.0))
    return float(np.percentile(s, min(99.99, q)))


def raw_first_alarm(ind: np.ndarray, c: float, sign: float = 1.0) -> Optional[int]:
    cross = np.where(sign * ind >= c)[0]
    return int(cross[0]) if len(cross) else None


# ---------------------------------------------------------------------------
# One replication: train, calibrate all detectors to matched ARL_0, evaluate
# ---------------------------------------------------------------------------

# Detection-window design (in steps = days on this daily substrate).
BURN_IN  = 30      # pre-change monitoring exposure
# Post-change horizon is regime-specific: drought needs a long horizon so the
# slow SPI-3 accumulation baseline (90-day window) gets a fair chance to react;
# heavy-rain onset is detectable quickly.
MAX_DELAY = {"drought": 180, "flood": 30}
DETECTORS = ("sr_rm", "cusum_rm", "cusum_ind", "raw_ind")


def _indicator(rain_sup: np.ndarray, regime: str, spi_window: int,
               imu: Optional[float] = None, isd: Optional[float] = None):
    """Operational indicator stream + its null standardisation.
    Drought -> standardised SPI-3 proxy (90-day accumulation).
    Flood   -> standardised daily precipitation."""
    if regime == "drought":
        ind = np.convolve(rain_sup, np.ones(spi_window), mode="full")[:len(rain_sup)]
    else:
        ind = rain_sup.astype(float)
    if imu is None:
        imu, isd = float(ind.mean()), float(ind.std()) + 1e-8
    return (ind - imu) / isd, imu, isd


def run_one(seed: int, regime: str, target_arl0: float,
            spi_window: int = 90) -> Optional[Dict]:
    """Train once; calibrate every detector to the SAME target ARL_0 on a null
    window; then measure (i) detection rate / delay on a change stream and
    (ii) false-alarm rate on a matched-length NULL stream (no change)."""
    n_tr_raw = int(7.0 * STEPS_PER_YEAR)
    n_ca_raw = int(10.0 * STEPS_PER_YEAR)

    cp_year = (seed % 2) + 11
    rng_cp = np.random.RandomState(seed + 1)
    cp_doy = rng_cp.randint(75, 120)
    t_change = cp_year * STEPS_PER_YEAR + cp_doy

    data_null, _ = simulate_multivar(14.0, seed)            # no-change stream
    if regime == "drought":
        data_c = inject_drought(data_null, t_change, seed)
        # drought: dry regime is MORE backward-coherent -> RM defect shifts DOWN;
        # SPI-3 shifts DOWN. Both monitored as downward shifts.
        z_sign, ind_sign = -1.0, -1.0
    else:
        data_c = inject_flood(data_null, t_change, seed)
        # heavy-rain regime -> RM defect and precipitation shift UP.
        z_sign, ind_sign = +1.0, +1.0

    Xc, yc = build_features(data_c, None, CONTEXT, train_end=n_tr_raw)
    Xn, _  = build_features(data_null, None, CONTEXT, train_end=n_tr_raw)
    n = Xc.shape[0]
    n_tr = min(n_tr_raw, n - 2 * CONTEXT)
    n_ca = min(n_ca_raw, n - CONTEXT)
    if n_tr < 50 or n_ca - n_tr < 50:
        return None
    if n - n_ca < CONTEXT:
        return None

    # train on guaranteed pre-change window (identical in data_c and data_null)
    model = RMRNN(Xc.shape[1], HIDDEN, seed=seed, lam=100.0)
    bl = min(BATCH_LEN, n_tr)
    rng_tr = np.random.RandomState(seed + 1001)
    for _ in range(N_EPOCHS):
        model.train_epoch(Xc[:n_tr], yc[:n_tr], batch_len=bl, rng=rng_tr)

    # --- null statistics + calibration on the no-change calibration window ---
    H_ca, _ = model.forward(Xn[n_tr:n_ca])
    z_raw_ca = _logdefect(model.rm_defects(H_ca))
    mu0, sd0 = float(z_raw_ca.mean()), float(z_raw_ca.std()) + 1e-8
    z_ca = (z_raw_ca - mu0) / sd0

    rain_ca = data_null[:, 0][CONTEXT:CONTEXT + n][n_tr:n_ca]
    ind_ca, imu, isd = _indicator(rain_ca, regime, spi_window)

    B_star, m0, _, _ = calibrate_sr_B(z_sign * z_ca, target_arl0)
    h_rm  = cusum_calibrate(z_sign * z_ca, target_arl0, sign=1.0)
    h_ind = cusum_calibrate(ind_ca, target_arl0, sign=ind_sign)
    c_raw = raw_calibrate(ind_ca, target_arl0, sign=ind_sign)

    # --- build CHANGE-stream and NULL-stream monitoring segments ---
    t_cp = max(0, int(t_change) - CONTEXT - n_ca)
    start = max(0, t_cp - BURN_IN)
    stop  = t_cp + MAX_DELAY[regime]

    def streams(Xsrc, data_src):
        H, _ = model.forward(Xsrc[n_ca:])
        z = (_logdefect(model.rm_defects(H)) - mu0) / sd0
        rain_te = data_src[:, 0][CONTEXT:CONTEXT + n][n_ca:]
        ind, _, _ = _indicator(rain_te, regime, spi_window, imu, isd)
        return z, ind

    z_c, ind_c = streams(Xc, data_c)       # change present
    z_n, ind_n = streams(Xn, data_null)    # null
    L = stop - start                        # matched horizon length

    def eval_change(alarm_fn):
        a = alarm_fn(start, stop, z_c, ind_c)
        if a is None:
            return {"detected": 0, "delay": None}
        ta = start + a
        if ta < t_cp:                       # pre-change alarm = miss (not early)
            return {"detected": 0, "delay": None}
        return {"detected": 1, "delay": int(ta - t_cp)}

    def eval_null(alarm_fn):
        a = alarm_fn(start, start + L, z_n, ind_n)
        return {"false_alarm": int(a is not None)}

    def fn_sr(s, e, z, ind):
        seg = z_sign * z[s:e]
        return sr_first_alarm(seg, B_star, m0)
    def fn_cusum_rm(s, e, z, ind):
        return cusum_first_alarm(z_sign * z[s:e], h_rm, sign=1.0)
    def fn_cusum_ind(s, e, z, ind):
        return cusum_first_alarm(ind[s:e], h_ind, sign=ind_sign)
    def fn_raw(s, e, z, ind):
        return raw_first_alarm(ind[s:e], c_raw, sign=ind_sign)

    fns = {"sr_rm": fn_sr, "cusum_rm": fn_cusum_rm,
           "cusum_ind": fn_cusum_ind, "raw_ind": fn_raw}

    out = {"B_star": B_star, "t_cp": t_cp, "L": L}
    for name, fn in fns.items():
        rec = eval_change(fn); rec.update(eval_null(fn))
        out[name] = rec
    return out


# ---------------------------------------------------------------------------
# Sweep + aggregation
# ---------------------------------------------------------------------------

def _binom_se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 0.0) / max(n, 1)))


def aggregate(recs: List[Dict], name: str) -> Dict:
    det = [r[name]["detected"] for r in recs]
    fa  = [r[name]["false_alarm"] for r in recs]
    delays = [r[name]["delay"] for r in recs if r[name]["delay"] is not None]
    dr, fr = float(np.mean(det)), float(np.mean(fa))
    n = len(recs)
    return {
        "detection_rate": dr,
        "detection_rate_se": _binom_se(dr, n),
        "false_alarm_rate": fr,
        "false_alarm_rate_se": _binom_se(fr, n),
        "mean_delay": float(np.mean(delays)) if delays else None,
        "sd_delay": float(np.std(delays)) if len(delays) > 1 else None,
        "median_delay": float(np.median(delays)) if delays else None,
        "n_detected": len(delays),
    }


def run_regime(regime: str, reps: int, target_arl0: float, seed0: int) -> Dict:
    recs = []
    B_stars = []
    t0 = time.time()
    for rep in range(reps):
        r = run_one(seed0 + rep, regime=regime, target_arl0=target_arl0)
        if r is not None:
            recs.append(r); B_stars.append(r["B_star"])
    res = {d: aggregate(recs, d) for d in DETECTORS}
    # Paired lead of the RM-defect CUSUM over the SPI-3/indicator CUSUM, on reps
    # where BOTH detect: lead = delay(indicator) - delay(rm); positive = RM earlier.
    # This is the paper's actual claim (lead vs the operational baseline).
    leads = [r["cusum_ind"]["delay"] - r["cusum_rm"]["delay"]
             for r in recs
             if r["cusum_rm"]["delay"] is not None and r["cusum_ind"]["delay"] is not None]
    res["rm_vs_ind_lead"] = {
        "n_both_detect": len(leads),
        "mean_lead": float(np.mean(leads)) if leads else None,
        "median_lead": float(np.median(leads)) if leads else None,
        "sd_lead": float(np.std(leads)) if len(leads) > 1 else None,
        "frac_rm_earlier": float(np.mean([x > 0 for x in leads])) if leads else None,
    }
    res["B_star_mean"] = float(np.mean(B_stars))
    res["B_star_sd"] = float(np.std(B_stars))
    res["B_star_ci"] = [float(np.percentile(B_stars, 2.5)),
                        float(np.percentile(B_stars, 97.5))]
    res["n_reps_used"] = len(recs)
    res["target_arl0"] = target_arl0
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


def cusum_arl0_curve(x: np.ndarray, h_grid: np.ndarray,
                     k: float = 0.5, sign: float = 1.0) -> np.ndarray:
    """Monte-Carlo ARL_0(h) for a one-sided CUSUM by reset-on-crossing."""
    arl = np.empty(len(h_grid))
    for j, h in enumerate(h_grid):
        S = 0.0; last = 0; rls = []
        for t in range(len(x)):
            S = max(0.0, S + sign * x[t] - k)
            if S >= h:
                rls.append(t - last + 1); S = 0.0; last = t + 1
        arl[j] = np.mean(rls) if rls else float(len(x))
    return arl


def make_arl_figure(out_png: Path, target_d: float, target_f: float,
                    seed0: int = 0):
    """Genuine CUSUM ARL_0(h) calibration figure (the detector actually used):
    drought = CUSUM on the RM defect (downward); heavy-rain = CUSUM on
    precipitation (upward). Single panel, target lines and h* marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # long null stream (years 7-13) from a model trained on years 1-7
    data, _ = simulate_multivar(14.0, seed0)
    n_tr = int(7.0 * STEPS_PER_YEAR); n_hi = int(13.0 * STEPS_PER_YEAR)
    X, y = build_features(data, None, CONTEXT, train_end=n_tr)
    model = RMRNN(X.shape[1], HIDDEN, seed=seed0 + 1001, lam=100.0)
    rng = np.random.RandomState(seed0 + 2002)
    for _ in range(N_EPOCHS):
        model.train_epoch(X[:n_tr], y[:n_tr], batch_len=min(BATCH_LEN, n_tr), rng=rng)
    Hn, _ = model.forward(X[n_tr:n_hi])
    z = _logdefect(model.rm_defects(Hn)); z = (z - z.mean()) / (z.std() + 1e-8)
    n = X.shape[0]
    rain_null = data[:, 0][CONTEXT:CONTEXT + n][n_tr:n_hi]
    p = (rain_null - rain_null.mean()) / (rain_null.std() + 1e-8)

    h_grid = np.geomspace(1.0, 60.0, 40)
    curves = {
        "drought": (cusum_arl0_curve(z, h_grid, sign=-1.0), target_d,
                    "Drought: CUSUM on RM defect (synthetic)", "tab:blue"),
        "heavyrain": (cusum_arl0_curve(p, h_grid, sign=+1.0), target_f,
                      "Heavy-rain: CUSUM on precipitation (synthetic)", "tab:orange"),
    }

    def hstar(arl, tgt):
        la, lh = np.log(np.clip(arl, 1e-6, None)), np.log(h_grid)
        t = np.log(tgt)
        if t <= la[0]: return float(h_grid[0])
        if t >= la[-1]: return float(h_grid[-1])
        return float(np.exp(np.interp(t, la, lh)))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    info = {}
    for key, (arl, tgt, label, color) in curves.items():
        hs = hstar(arl, tgt)
        ax.plot(h_grid, arl, color=color, lw=2, label=label)
        ax.axhline(tgt, color=color, ls=":", lw=1, alpha=0.6)
        ax.axvline(hs, color=color, ls="--", lw=1, alpha=0.8)
        info[key] = {"h_star": hs, "target_arl0": float(tgt)}
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"CUSUM threshold $h$")
    ax.set_ylabel(r"Empirical $\mathrm{ARL}_0$")
    ax.set_title(r"CUSUM $\mathrm{ARL}_0$ calibration (Monte-Carlo, synthetic data)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--arl0_drought", type=float, default=500.0)
    ap.add_argument("--arl0_flood", type=float, default=1000.0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default="00 results")
    ap.add_argument("--figdir", type=str, default="figures")
    args = ap.parse_args()
    if args.quick:
        args.reps = min(args.reps, 20)

    root = _HERE.parent.parent.parent          # .../RM_Precipitation
    out_dir = root / args.out
    fig_dir = root / args.figdir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"SR warning experiment  reps={args.reps}  "
          f"ARL0(drought)={args.arl0_drought:.0f}  ARL0(flood)={args.arl0_flood:.0f}",
          flush=True)

    def _line(tag, r):
        s = r["sr_rm"]; c = r["cusum_rm"]; i = r["cusum_ind"]
        print(f"  {tag} done ({r['elapsed_sec']}s)  B*={r['B_star_mean']:.0f}  "
              f"SR(rm) det={s['detection_rate']:.2f}/FAR={s['false_alarm_rate']:.2f}  "
              f"CUSUM(rm) det={c['detection_rate']:.2f}/FAR={c['false_alarm_rate']:.2f}/"
              f"delay={c['mean_delay']}  CUSUM(ind) det={i['detection_rate']:.2f}",
              flush=True)

    drought = run_regime("drought", args.reps, args.arl0_drought, args.seed0)
    _line("drought", drought)
    flood = run_regime("flood", args.reps, args.arl0_flood, args.seed0)
    _line("flood  ", flood)

    print("  generating ARL_0 calibration figure ...", flush=True)
    figinfo = make_arl_figure(fig_dir / "arl_calibration.png",
                              args.arl0_drought, args.arl0_flood, args.seed0)

    out = {
        "description": (
            "Genuine Shiryaev-Roberts warning-layer calibration and detection "
            "on SYNTHETIC ClimateSpec data (NOT real CWA/CHIRPS/ERA5-Land). "
            "RM defect r_t=||h_t-g_phi(h_{t+1})||^2 from a trained GRU+g_phi "
            "RMRNN; log-transformed and standardised on a null calibration "
            "window; SR recursion R_t=(1+R_{t-1})exp(delta z_t-delta^2/2). "
            "B* calibrated to target ARL_0 by Monte-Carlo reset-on-crossing. "
            "Comparators (CUSUM on the operational indicator; raw indicator "
            "threshold) calibrated to the SAME ARL_0 for a fair comparison. "
            "Drought = sustained multi-channel suppression; flood = heavy-rain "
            "onset. Case studies are injected change-points, not real events."),
        "config": {"reps": args.reps, "seed0": args.seed0,
                   "context": CONTEXT, "hidden_dim": HIDDEN,
                   "n_epochs": N_EPOCHS, "sr_delta": SR_DELTA,
                   "spi_window": 90},
        "drought": drought,
        "flood": flood,
        "figure_B_star": figinfo,
    }
    path = out_dir / "warning_sr_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> wrote {path}")
    print(f"  -> wrote {fig_dir/'arl_calibration.pdf'} (and .png)")


if __name__ == "__main__":
    main()
