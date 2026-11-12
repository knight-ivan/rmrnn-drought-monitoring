"""
run_ghcn_stability.py
=====================
Table 12 in P2 — GHCN Monthly station-level stability study.

Design
------
  - Dataset  : GHCN Monthly v4 (GHCNm), monthly precipitation totals
  - Selection: 1000 stations with >= 600 complete months, < 5% missing,
               stratified by Koppen climate zone (167 per zone × 6 zones)
  - Target   : next-month log-precipitation (standardised per station)
  - Input    : 12-month sliding window (captures annual cycle)
  - Split    : first 480 months training, remainder test
  - Models   : RMRNN, VanillaRNN (Elman base)
  - Reps     : one per station — 1000 natural replications across climate types

Data download (one-time, ~50 MB)
---------------------------------
  # GHCNm v4 monthly precipitation file:
  wget https://www.ncei.noaa.gov/data/ghcnm/v4/archive/ghcnm.prcp.latest.qcu.tar.gz
  tar -xzf ghcnm.prcp.latest.qcu.tar.gz
  # This produces ghcnm.prcp.vX.X.X.YYYYMMDD.qcu.dat and .inv files.
  # Set DATA_DIR below to the directory containing those files.

  Koppen zone metadata is derived from the station coordinates using a
  simplified Koppen-Geiger classification (see koppen_zone() below).

Outputs
-------
  experiments/ghcn_stability_results.npz  — per-station metrics
  figures/ghcn_Q_by_zone.pdf             — Q by climate zone (boxplot)
  stdout                                  — Table 12 and 13 values

Usage
-----
  python3 run_ghcn_stability.py [--n_stations 1000] [--workers 8] [--seed 42]
"""

import argparse
import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
EXP_DIR  = os.path.join(BASE_DIR, "experiments")
FIG_DIR  = os.path.join(BASE_DIR, "figures")
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DATA_DIR = os.environ.get(
    "GHCNM_DATA",
    os.path.join(BASE_DIR, "data", "ghcnm")
)

# ── hyperparameters ───────────────────────────────────────────────────────────
WINDOW       = 12     # input context: 12 months (annual cycle)
TRAIN_MONTHS = 480    # 40 years training
MIN_MONTHS   = 600    # minimum complete months for station selection
MAX_MISSING  = 0.05   # maximum missing fraction
N_EPOCHS     = 200
WARMUP       = 5
LAMBDA_RM    = 0.1
LAMBDA_DECAY = 0.1
LR           = 5e-3
HIDDEN_DIM   = 32
BATCH_SIZE   = 32

KOPPEN_ZONES = [
    "Tropical", "Arid", "Temperate", "Continental", "Polar", "Mediterranean"
]


# ── Koppen zone classifier (simplified, from lat/lon) ─────────────────────────

def koppen_zone(lat: float, lon: float) -> str:
    """
    Simplified Koppen-Geiger zone from coordinates.
    This is a coarse approximation; a full implementation would use
    the Kottek et al. (2006) gridded dataset.
    """
    alat = abs(lat)
    if alat < 15:
        return "Tropical"
    if alat > 66:
        return "Polar"
    # Mediterranean: 30–45° lat, west coasts (rough approximation)
    if 30 <= alat <= 45 and (lon < -10 or (100 < lon < 180 and lat > 0)):
        return "Mediterranean"
    # Arid: subtropical belts
    if 15 <= alat <= 35:
        return "Arid"
    # Continental: mid-high latitudes, continental interiors
    if alat > 45:
        return "Continental"
    return "Temperate"


# ── GHCN Monthly data loader ──────────────────────────────────────────────────

def find_ghcnm_files(data_dir: str):
    """Locate the .dat and .inv files in data_dir."""
    dat_files = [f for f in os.listdir(data_dir) if f.endswith(".qcu.dat")]
    inv_files = [f for f in os.listdir(data_dir) if f.endswith(".qcu.inv")]
    if not dat_files or not inv_files:
        raise FileNotFoundError(
            f"No GHCNm .qcu.dat / .qcu.inv files found in {data_dir}.\n"
            f"Download from:\n"
            f"  wget https://www.ncei.noaa.gov/data/ghcnm/v4/archive/"
            f"ghcnm.prcp.latest.qcu.tar.gz\n"
            f"Then: tar -xzf ghcnm.prcp.latest.qcu.tar.gz -C {data_dir}\n"
            f"Or set GHCNM_DATA env var."
        )
    return (os.path.join(data_dir, dat_files[0]),
            os.path.join(data_dir, inv_files[0]))


def load_ghcnm_inventory(inv_path: str) -> dict:
    """Parse .inv file → dict station_id -> (lat, lon, name)."""
    stations = {}
    with open(inv_path) as f:
        for line in f:
            sid  = line[0:11].strip()
            lat  = float(line[12:20])
            lon  = float(line[21:30])
            name = line[38:68].strip()
            stations[sid] = (lat, lon, name)
    return stations


def load_ghcnm_data(dat_path: str) -> dict:
    """
    Parse GHCNm v4 .dat file.

    Format: station_id(11) year(4) element(4) value*12 flag*12
    Each value is in tenths of mm; -9999 = missing.

    Returns dict: station_id -> np.ndarray of shape (n_years * 12,)
                  with NaN for missing, units mm.
    """
    station_monthly = {}   # sid -> dict year -> [12 values]

    with open(dat_path) as f:
        for line in f:
            if len(line) < 19:
                continue
            sid     = line[0:11]
            year    = int(line[11:15])
            element = line[15:19]
            if element.strip() != "PRCP":
                continue

            vals = []
            for m in range(12):
                start = 19 + m * 8
                v_str = line[start: start + 5]
                try:
                    v = int(v_str)
                except ValueError:
                    v = -9999
                vals.append(np.nan if v == -9999 else v / 10.0)   # tenths mm → mm

            if sid not in station_monthly:
                station_monthly[sid] = {}
            station_monthly[sid][year] = vals

    # Convert to contiguous arrays
    station_series = {}
    for sid, year_dict in station_monthly.items():
        years = sorted(year_dict.keys())
        if len(years) < 2:
            continue
        y_min, y_max = years[0], years[-1]
        n_months = (y_max - y_min + 1) * 12
        arr = np.full(n_months, np.nan)
        for yr in years:
            offset = (yr - y_min) * 12
            arr[offset: offset + 12] = year_dict[yr]
        station_series[sid] = arr
    return station_series


def select_stations(station_series: dict, inventory: dict,
                    n_total: int, min_months: int,
                    max_missing: float, seed: int) -> list:
    """
    Select n_total stations satisfying quality criteria,
    stratified across Koppen zones.
    Returns list of station_ids.
    """
    rng = np.random.RandomState(seed)

    # Filter by completeness
    candidates = {
        sid: arr for sid, arr in station_series.items()
        if (len(arr) >= min_months and
            np.isnan(arr).mean() < max_missing and
            sid in inventory)
    }

    # Assign Koppen zones
    zone_buckets = {z: [] for z in KOPPEN_ZONES}
    for sid, arr in candidates.items():
        lat, lon, _ = inventory[sid]
        zone = koppen_zone(lat, lon)
        zone_buckets[zone].append(sid)

    # Stratified sample
    n_per_zone = n_total // len(KOPPEN_ZONES)
    selected = []
    for zone in KOPPEN_ZONES:
        pool_z = zone_buckets[zone]
        n_draw = min(n_per_zone, len(pool_z))
        chosen = rng.choice(pool_z, size=n_draw, replace=False).tolist()
        selected.extend(chosen)

    # Top up to n_total if any zone was under-represented
    remaining = [s for s in candidates if s not in set(selected)]
    rng.shuffle(remaining)
    selected.extend(remaining[: n_total - len(selected)])
    return selected[:n_total]


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess_station(arr: np.ndarray, train_months: int):
    """
    Impute missing values (linear interpolation), log-transform,
    standardise on training window.

    Returns (series_norm, mean_tr, std_tr) where series_norm is the
    full normalised series.
    """
    # Interpolate short gaps (< 6 months)
    arr = arr.copy()
    nan_idx = np.where(np.isnan(arr))[0]
    for i in nan_idx:
        left  = i - 1
        right = i + 1
        while right < len(arr) and np.isnan(arr[right]):
            right += 1
        if left >= 0 and right < len(arr) and (right - left) <= 6:
            arr[i] = arr[left] + (arr[right] - arr[left]) * (i - left) / (right - left)
    # Replace remaining NaN with 0 (arid stations)
    arr = np.where(np.isnan(arr), 0.0, arr)
    arr = np.clip(arr, 0.0, None)

    # Log-transform (add 1 mm offset to handle zeros)
    arr_log = np.log1p(arr)

    # Standardise on training split
    tr = arr_log[:train_months]
    mu  = tr.mean()
    sig = tr.std().clip(min=1e-8)
    return (arr_log - mu) / sig, mu, sig


def make_windows(series: np.ndarray, window: int):
    """
    Sliding-window sequences: X (N, window, 1), y (N, 1).
    """
    N = len(series) - window
    X = np.stack([series[i: i + window, np.newaxis] for i in range(N)])
    y = series[window:, np.newaxis]
    return X, y


# ── single station worker ─────────────────────────────────────────────────────

def _run_station(args):
    """Train RMRNN and VanillaRNN on one station. Returns metrics dict."""
    sid, raw_arr, lat, lon, seed = args
    sys.path.insert(0, SRC_DIR)
    from model import RMRNN as _RMRNN
    from baselines import VanillaRNN as _VanillaRNN

    # Clip to first 600 months if shorter
    arr = raw_arr[:max(MIN_MONTHS, TRAIN_MONTHS + 60)]

    series_norm, _, _ = preprocess_station(arr, TRAIN_MONTHS)

    X_all, y_all = make_windows(series_norm, WINDOW)
    X_tr, y_tr = X_all[:TRAIN_MONTHS - WINDOW], y_all[:TRAIN_MONTHS - WINDOW]
    X_te, y_te = X_all[TRAIN_MONTHS - WINDOW:], y_all[TRAIN_MONTHS - WINDOW:]

    if len(X_tr) < 20 or len(X_te) < 5:
        return None

    rng = np.random.RandomState(seed)
    n_tr = len(X_tr)

    station_results = {}
    for model_name, lambda_rm in [("RMRNN", LAMBDA_RM), ("VanillaRNN", 0.0)]:
        if lambda_rm > 0:
            model = _RMRNN(1, HIDDEN_DIM, 1, lambda_rm=lambda_rm, seed=seed)
        else:
            model = _VanillaRNN(1, HIDDEN_DIM, 1, seed=seed)

        conv_epoch = N_EPOCHS   # default if not converged
        prev_loss = np.inf

        for epoch in range(N_EPOCHS):
            if lambda_rm > 0:
                frac = max(0, (epoch - WARMUP)) / max(N_EPOCHS - WARMUP - 1, 1)
                model.lambda_rm = LAMBDA_RM * (LAMBDA_DECAY ** frac) if epoch >= WARMUP else 0.0

            idx = rng.choice(n_tr, size=min(BATCH_SIZE, n_tr), replace=False)
            info = model.train_step(X_tr[idx], y_tr[idx], lr=LR)

            # Convergence: task loss change < 1e-4
            cur_loss = info["loss_task"]
            if abs(prev_loss - cur_loss) < 1e-4 and epoch > WARMUP:
                conv_epoch = epoch
                break
            prev_loss = cur_loss

        # Evaluate on test
        if hasattr(model, 'params'):
            from model import rmrnn_forward_batch
            h = rmrnn_forward_batch(model.params, X_te)
            y_pred = h[:, -1, :] @ model.params.Wr.T + model.params.br
            Q = model.quasi_martingale_total(X_te[:min(50, len(X_te))])
        else:
            y_pred = model.predict(X_te)
            Q = 0.0

        mse = float(np.mean((y_pred - y_te) ** 2))
        mae = float(np.mean(np.abs(y_pred - y_te)))
        defect_norm = Q / max(WINDOW - 1, 1)

        station_results[model_name] = {
            "mse": mse, "mae": mae, "Q": Q,
            "defect_norm": defect_norm, "conv_epoch": conv_epoch,
        }

    zone = koppen_zone(lat, lon)
    return {"sid": sid, "lat": lat, "lon": lon, "zone": zone,
            **station_results}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_stations", type=int, default=1000)
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    print("GHCN Monthly — station-level stability study")
    print(f"Target: {args.n_stations} stations  |  {N_EPOCHS} epochs each\n")

    # ── load GHCNm data ───────────────────────────────────────────────────────
    try:
        dat_path, inv_path = find_ghcnm_files(DATA_DIR)
    except FileNotFoundError as e:
        print(f"\nDATA NOT FOUND:\n{e}")
        sys.exit(1)

    print(f"Loading inventory from {inv_path} ...")
    inventory = load_ghcnm_inventory(inv_path)
    print(f"Loading station data from {dat_path} ...")
    station_series = load_ghcnm_data(dat_path)
    print(f"  {len(station_series)} stations with precipitation data loaded.")

    selected = select_stations(station_series, inventory,
                               args.n_stations, MIN_MONTHS,
                               MAX_MISSING, args.seed)
    print(f"  {len(selected)} stations selected after quality filter + stratification.\n")

    # ── build worker argument list ────────────────────────────────────────────
    worker_args = []
    for i, sid in enumerate(selected):
        lat, lon, _ = inventory[sid]
        worker_args.append(
            (sid, station_series[sid], lat, lon, args.seed + i)
        )

    # ── run in parallel ───────────────────────────────────────────────────────
    with Pool(processes=args.workers) as pool:
        raw_results = pool.map(_run_station, worker_args)

    results = [r for r in raw_results if r is not None]
    print(f"\n{len(results)} stations completed successfully.")

    # ── aggregate overall ─────────────────────────────────────────────────────
    def agg(results, model, key):
        vals = np.array([r[model][key] for r in results])
        return vals.mean(), vals.std(), vals

    print("\n" + "="*70)
    print("TABLE 12 — GHCN station-level stability (overall)")
    print("="*70)
    print(f"{'Model':15s}  {'Q mean±SD':>18s}  {'DefectNorm':>12s}  "
          f"{'TestMSE':>10s}  {'Conv.Ep.':>10s}")
    print("-"*70)
    for mn in ("RMRNN", "VanillaRNN"):
        Qm, Qs, _   = agg(results, mn, "Q")
        DNm, DNs, _ = agg(results, mn, "defect_norm")
        MSEm, _, _  = agg(results, mn, "mse")
        CEm, _, _   = agg(results, mn, "conv_epoch")
        print(f"{mn:15s}  {Qm:7.2f}±{Qs:6.2f}  {DNm:8.4f}±{DNs:6.4f}  "
              f"{MSEm:10.4f}  {CEm:10.1f}")

    # ── aggregate by Koppen zone ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("TABLE 13 — GHCN stability by Koppen climate zone")
    print("="*70)
    print(f"{'Zone':20s}  {'n':>5s}  {'Q RMRNN':>18s}  {'Q Baseline':>18s}")
    print("-"*70)
    for zone in KOPPEN_ZONES:
        zone_res = [r for r in results if r["zone"] == zone]
        if not zone_res:
            print(f"{zone:20s}  {'0':>5s}  {'—':>18s}  {'—':>18s}")
            continue
        Qrm_m, Qrm_s, _ = agg(zone_res, "RMRNN",      "Q")
        Qbl_m, Qbl_s, _ = agg(zone_res, "VanillaRNN", "Q")
        n = len(zone_res)
        print(f"{zone:20s}  {n:>5d}  {Qrm_m:7.2f}±{Qrm_s:6.2f}  "
              f"{Qbl_m:7.2f}±{Qbl_s:6.2f}")

    # ── save ─────────────────────────────────────────────────────────────────
    save_dict = {
        "station_ids": np.array([r["sid"] for r in results], dtype=object),
        "zones":       np.array([r["zone"] for r in results], dtype=object),
        "lats":        np.array([r["lat"]  for r in results]),
        "lons":        np.array([r["lon"]  for r in results]),
    }
    for mn in ("RMRNN", "VanillaRNN"):
        save_dict[f"{mn}/Q"]          = np.array([r[mn]["Q"]          for r in results])
        save_dict[f"{mn}/mse"]        = np.array([r[mn]["mse"]        for r in results])
        save_dict[f"{mn}/defect_norm"]= np.array([r[mn]["defect_norm"]for r in results])
        save_dict[f"{mn}/conv_epoch"] = np.array([r[mn]["conv_epoch"] for r in results])

    npz_path = os.path.join(EXP_DIR, "ghcn_stability_results.npz")
    np.savez(npz_path, **save_dict)
    print(f"\nSaved: {npz_path}")

    # ── figure: Q boxplot by zone ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    zone_Q_rm = []
    zone_Q_bl = []
    labels_used = []
    for zone in KOPPEN_ZONES:
        z_res = [r for r in results if r["zone"] == zone]
        if z_res:
            zone_Q_rm.append([r["RMRNN"]["Q"]      for r in z_res])
            zone_Q_bl.append([r["VanillaRNN"]["Q"] for r in z_res])
            labels_used.append(zone)

    x = np.arange(len(labels_used))
    w = 0.35
    bp1 = ax.boxplot(zone_Q_rm, positions=x - w/2, widths=w*0.8,
                     patch_artist=True,
                     boxprops=dict(facecolor="#1f77b4", alpha=0.6),
                     medianprops=dict(color="navy", lw=2))
    bp2 = ax.boxplot(zone_Q_bl, positions=x + w/2, widths=w*0.8,
                     patch_artist=True,
                     boxprops=dict(facecolor="#ff7f0e", alpha=0.6),
                     medianprops=dict(color="darkred", lw=2))
    ax.set_xticks(x)
    ax.set_xticklabels(labels_used, fontsize=9)
    ax.set_ylabel("Quasi-martingale total $Q$")
    ax.set_title(f"GHCN $Q$ by Koppen climate zone ({len(results)} stations)")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["RMRNN", "Baseline"],
              fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "ghcn_Q_by_zone.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
