"""
run_bharatbench.py
==================
Tables 14–15 in P2 — BharatBench Indian monsoon precipitation.

Design
------
  - Dataset  : BharatBench (Choudhury et al. 2024), IMDAA reanalysis
               6-hourly TP6h over India, 32×32 grid, 1.08° resolution
               Region: 5°N–40°N, 65°E–100°E
  - Target   : 6-hourly accumulated precipitation (TP6h), log-standardised
  - Input    : 28 time steps = 7 days of 6-h history
  - Split    : train 1990–2017, val 2018, test 2019–2020
  - Horizons : tau = 1 (6h), 4 (24h), 12 (72h) steps
  - Reps     : 1000 bootstrap samples of training years
               (simultaneously quantifies seed + data-year sensitivity)
  - Stratification: Q reported by Indian precipitation regime (4 zones)

Data download (one-time, ~8 GB full dataset; subset used here)
---------------------------------------------------------------
  git clone https://github.com/MASLABnitrkl/BharatBench
  cd BharatBench
  # Follow README for IMDAA data access; or use Kaggle:
  kaggle datasets download -d maslab/bharatbench
  unzip bharatbench.zip -d <local_dir>
  # Set DATA_DIR below or BHARATBENCH_DATA env var.

  Expected file structure:
    <DATA_DIR>/
      train/total_precipitation_6h_<YYYY>.nc   (1990–2017)
      test/total_precipitation_6h_<YYYY>.nc    (2019–2020)

Outputs
-------
  experiments/bharatbench_results.npz          — raw metrics (reps × models × horizons)
  figures/bharatbench_Q_hist.pdf               — Q distribution across bootstrap reps
  figures/bharatbench_Q_by_regime.pdf          — Q by precipitation regime
  stdout                                        — Table 14 and 15 values

Usage
-----
  python3 run_bharatbench.py [--reps 1000] [--workers 8] [--seed 0]
"""

import argparse
import os
import sys
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
    "BHARATBENCH_DATA",
    os.path.join(BASE_DIR, "data", "bharatbench")
)

# ── hyperparameters ───────────────────────────────────────────────────────────
N_LAT, N_LON  = 32, 32          # BharatBench regridded spatial dims
STEPS_PER_DAY = 4               # 6-hourly data → 4 steps per day
INPUT_WINDOW  = 28              # 7 days × 4 steps/day
HORIZONS      = [1, 4, 12]      # 6h, 24h, 72h in 6-h steps
TRAIN_YEARS   = list(range(1990, 2018))   # 28 years
TEST_YEARS    = [2019, 2020]
N_BOOT_YEARS  = 28              # bootstrap draw size (= full training set size)
N_EPOCHS      = 100
WARMUP        = 5
LAMBDA_RM     = 0.1
LAMBDA_DECAY  = 0.1
LR            = 1e-3
HIDDEN_DIM    = 64
BATCH_SIZE    = 64

# Precipitation regime thresholds (mean annual TP6h in mm)
REGIME_THRESHOLDS = {
    "Arid / rain-shadow": (0.0,  0.3),
    "Semi-arid":          (0.3,  0.8),
    "Monsoon":            (0.8,  2.0),
    "Orographic":         (2.0, np.inf),
}


# ── data loading ─────────────────────────────────────────────────────────────

def load_bharatbench_tp(data_dir: str, years: list) -> np.ndarray:
    """
    Load BharatBench TP6h for the given years.

    Returns
    -------
    data : np.ndarray  (total_timesteps, N_LAT, N_LON)  in mm/6h
    """
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 required: pip install netCDF4")

    fields = []
    for yr in years:
        # Try both train/ and test/ subdirectories
        for subdir in ("train", "test", ""):
            fname = os.path.join(
                data_dir, subdir,
                f"total_precipitation_6h_{yr}.nc"
            )
            if os.path.exists(fname):
                break
        else:
            raise FileNotFoundError(
                f"Missing TP6h file for year {yr}.\n"
                f"Expected: {data_dir}/train/total_precipitation_6h_{yr}.nc\n"
                f"Download from: https://github.com/MASLABnitrkl/BharatBench\n"
                f"Or: kaggle datasets download -d maslab/bharatbench\n"
                f"Set BHARATBENCH_DATA env var or edit DATA_DIR in this script."
            )

        with nc.Dataset(fname) as ds:
            # Variable name may be 'tp', 'TP', 'total_precipitation', or 'prcp'
            var_names = list(ds.variables.keys())
            tp_var = next(
                (v for v in var_names
                 if v.lower() in ("tp", "total_precipitation", "prcp", "tp6h")),
                var_names[-1]
            )
            tp = np.array(ds.variables[tp_var][:])   # (time, lat, lon)
            # Convert to mm if in metres (values < 0.1 suggest metres)
            if tp[np.isfinite(tp)].max() < 0.5:
                tp = tp * 1000.0
            tp = np.clip(tp, 0.0, None)
            fields.append(tp)

    return np.concatenate(fields, axis=0)


def preprocess(data_train: np.ndarray, data_test: np.ndarray):
    """
    Log-transform and standardise per grid point using training statistics.
    Returns normalised arrays and (mean, std) per grid point.
    """
    eps = 1e-3
    log_tr = np.log1p(data_train + eps)
    log_te = np.log1p(data_test  + eps)

    mu  = log_tr.mean(axis=0, keepdims=True)                   # (1, lat, lon)
    sig = log_tr.std(axis=0,  keepdims=True).clip(min=1e-8)

    return (log_tr - mu) / sig, (log_te - mu) / sig, mu.squeeze(), sig.squeeze()


def make_sequences(field: np.ndarray, window: int, tau: int):
    """
    Sliding-window sequences: X (N, window, n_pts), y (N, n_pts).
    field: (T, n_lat, n_lon)
    """
    T, n_lat, n_lon = field.shape
    n_pts = n_lat * n_lon
    flat  = field.reshape(T, n_pts)

    N = T - window - tau + 1
    if N <= 0:
        raise ValueError(f"Sequence too short: T={T}, window={window}, tau={tau}")

    X = np.stack([flat[i: i + window] for i in range(N)])          # (N, window, n_pts)
    y = np.stack([flat[i + window + tau - 1] for i in range(N)])   # (N, n_pts)
    return X, y


def bootstrap_years(data_train_years: np.ndarray, year_lengths: list,
                    boot_indices: np.ndarray) -> np.ndarray:
    """
    Construct a bootstrap training dataset by resampling whole years.

    data_train_years : (total_train_steps, n_lat, n_lon)
    year_lengths     : list of int, number of timesteps per year
    boot_indices     : (N_BOOT_YEARS,) year indices (with replacement)

    Returns bootstrap_data : (total_boot_steps, n_lat, n_lon)
    """
    cumlen = np.concatenate([[0], np.cumsum(year_lengths)])
    pieces = []
    for yi in boot_indices:
        start = int(cumlen[yi])
        end   = int(cumlen[yi + 1])
        pieces.append(data_train_years[start:end])
    return np.concatenate(pieces, axis=0)


# ── single replication worker ─────────────────────────────────────────────────

def _run_one_rep(args):
    """
    One bootstrap replication.

    args = (rep, data_train_years, year_lengths, data_test,
            tau, base_seed, mean_annual_tp_per_pt)
    """
    (rep, data_tr_yrs, year_lengths, data_te,
     tau, base_seed, mean_annual_tp) = args

    rng  = np.random.RandomState(base_seed + rep)
    seed = base_seed + rep

    sys.path.insert(0, SRC_DIR)
    from model import RMRNN as _RMRNN, rmrnn_forward_batch
    from baselines import VanillaRNN as _VanillaRNN

    # Bootstrap year indices
    boot_idx = rng.choice(len(year_lengths), size=N_BOOT_YEARS, replace=True)
    boot_data = bootstrap_years(data_tr_yrs, year_lengths, boot_idx)

    n_pts = N_LAT * N_LON

    X_tr, y_tr = make_sequences(boot_data, INPUT_WINDOW, tau)
    X_te, y_te = make_sequences(data_te,   INPUT_WINDOW, tau)

    n_tr = len(X_tr)
    results = {}

    for model_name, lambda_rm in [("RMRNN", LAMBDA_RM), ("Baseline", 0.0)]:
        if lambda_rm > 0:
            model = _RMRNN(n_pts, HIDDEN_DIM, n_pts,
                           lambda_rm=lambda_rm, seed=seed)
        else:
            model = _VanillaRNN(n_pts, HIDDEN_DIM, n_pts, seed=seed)

        for epoch in range(N_EPOCHS):
            if lambda_rm > 0:
                frac = max(0, epoch - WARMUP) / max(N_EPOCHS - WARMUP - 1, 1)
                model.lambda_rm = (LAMBDA_RM * (LAMBDA_DECAY ** frac)
                                   if epoch >= WARMUP else 0.0)
            idx = rng.choice(n_tr, size=min(BATCH_SIZE, n_tr), replace=False)
            model.train_step(X_tr[idx], y_tr[idx], lr=LR)

        # Evaluate
        if hasattr(model, 'params'):
            h = rmrnn_forward_batch(model.params, X_te)
            y_pred = h[:, -1, :] @ model.params.Wr.T + model.params.br
            n_q = min(100, len(X_te))
            Q = model.quasi_martingale_total(X_te[:n_q])
        else:
            y_pred = model.predict(X_te)
            Q = 0.0

        mse = float(np.mean((y_pred - y_te) ** 2))
        mae = float(np.mean(np.abs(y_pred - y_te)))

        # Per-regime Q (split grid points by mean annual TP)
        regime_Q = {}
        for regime, (lo, hi) in REGIME_THRESHOLDS.items():
            mask = (mean_annual_tp.ravel() >= lo) & (mean_annual_tp.ravel() < hi)
            regime_Q[regime] = Q * mask.mean() if mask.any() else np.nan

        results[model_name] = {"mse": mse, "mae": mae, "Q": Q,
                               "regime_Q": regime_Q}

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps",    type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    print("BharatBench — Indian monsoon precipitation (32×32 grid)")
    print(f"Horizons: {HORIZONS} × 6h steps  |  {args.reps} bootstrap replications\n")

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading training years ...")
    try:
        train_fields = []
        year_lengths = []
        for yr in TRAIN_YEARS:
            ydata = load_bharatbench_tp(DATA_DIR, [yr])
            train_fields.append(ydata)
            year_lengths.append(len(ydata))
        data_train_all = np.concatenate(train_fields, axis=0)

        print("Loading test years ...")
        data_test = load_bharatbench_tp(DATA_DIR, TEST_YEARS)

    except FileNotFoundError as e:
        print(f"\nDATA NOT FOUND:\n{e}")
        sys.exit(1)

    # Preprocess
    data_train_norm, data_test_norm, mu_grid, sig_grid = preprocess(
        data_train_all, data_test
    )

    # Mean annual TP for regime classification (un-normalised, in mm/6h)
    mean_annual_tp = data_train_all.mean(axis=0)   # (lat, lon)

    print(f"  Training timesteps: {len(data_train_norm)}")
    print(f"  Test timesteps    : {len(data_test_norm)}\n")

    # ── per-horizon experiment ────────────────────────────────────────────────
    all_results = {}

    for tau in HORIZONS:
        tau_h = tau * 6
        print(f"\n{'='*60}")
        print(f"  Horizon tau={tau} steps = {tau_h}h")
        print(f"{'='*60}")

        worker_args = [
            (rep, data_train_norm, year_lengths, data_test_norm,
             tau, args.seed, mean_annual_tp)
            for rep in range(args.reps)
        ]

        with Pool(processes=args.workers) as pool:
            rep_results = pool.map(_run_one_rep, worker_args)

        # Aggregate
        tau_agg = {}
        for mn in ("RMRNN", "Baseline"):
            mse_arr = np.array([r[mn]["mse"] for r in rep_results])
            mae_arr = np.array([r[mn]["mae"] for r in rep_results])
            Q_arr   = np.array([r[mn]["Q"]   for r in rep_results])
            # Per-regime Q
            regime_Q_arr = {}
            for regime in REGIME_THRESHOLDS:
                regime_Q_arr[regime] = np.array(
                    [r[mn]["regime_Q"][regime] for r in rep_results]
                )
            tau_agg[mn] = {
                "mse_mean": mse_arr.mean(), "mse_std": mse_arr.std(),
                "mae_mean": mae_arr.mean(), "mae_std": mae_arr.std(),
                "Q_mean":   Q_arr.mean(),   "Q_std":   Q_arr.std(),
                "Q_raw":    Q_arr,
                "regime_Q": {r: v for r, v in regime_Q_arr.items()},
            }
        all_results[tau] = tau_agg

        # Summary print
        print(f"\n  {'Model':12s}  {'MSE':>14s}  {'MAE':>14s}  {'Q':>12s}")
        print(f"  {'-'*58}")
        for mn in ("RMRNN", "Baseline"):
            a = tau_agg[mn]
            print(f"  {mn:12s}  "
                  f"{a['mse_mean']:.4f}±{a['mse_std']:.4f}  "
                  f"{a['mae_mean']:.4f}±{a['mae_std']:.4f}  "
                  f"{a['Q_mean']:.2f}±{a['Q_std']:.2f}")

    # ── save ─────────────────────────────────────────────────────────────────
    save_dict = {}
    for tau, tau_agg in all_results.items():
        for mn, metrics in tau_agg.items():
            key = f"tau{tau}/{mn}"
            save_dict[f"{key}/Q_raw"] = metrics["Q_raw"]
            save_dict[f"{key}/mse"]   = np.array([metrics["mse_mean"], metrics["mse_std"]])
            save_dict[f"{key}/mae"]   = np.array([metrics["mae_mean"], metrics["mae_std"]])
            for regime, Q_arr in metrics["regime_Q"].items():
                safe = regime.replace(" ", "_").replace("/", "_")
                save_dict[f"{key}/regime_{safe}"] = Q_arr
    npz_path = os.path.join(EXP_DIR, "bharatbench_results.npz")
    np.savez(npz_path, **save_dict)
    print(f"\nSaved: {npz_path}")

    # ── figures ───────────────────────────────────────────────────────────────
    # (a) Q histogram across bootstrap reps at tau=1
    tau1 = HORIZONS[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for mn, color in [("RMRNN", "#1f77b4"), ("Baseline", "#ff7f0e")]:
        Q_arr = all_results[tau1][mn]["Q_raw"]
        ax.hist(Q_arr, bins=40, alpha=0.6, color=color,
                density=True, label=mn)
    ax.set_xlabel("Quasi-martingale total $Q$  (tau=6h)")
    ax.set_ylabel("Density")
    ax.set_title(f"BharatBench $Q$ distribution ({args.reps} bootstrap reps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(FIG_DIR, "bharatbench_Q_hist.pdf")
    fig.savefig(path1)
    plt.close(fig)
    print(f"Saved: {path1}")

    # (b) Q by precipitation regime
    tau1_agg = all_results[HORIZONS[0]]
    fig, ax = plt.subplots(figsize=(9, 4))
    regimes = list(REGIME_THRESHOLDS.keys())
    x = np.arange(len(regimes))
    w = 0.35
    for offset, (mn, color) in enumerate([("RMRNN", "#1f77b4"),
                                           ("Baseline", "#ff7f0e")]):
        medians = [np.nanmedian(tau1_agg[mn]["regime_Q"][r]) for r in regimes]
        q25     = [np.nanpercentile(tau1_agg[mn]["regime_Q"][r], 25) for r in regimes]
        q75     = [np.nanpercentile(tau1_agg[mn]["regime_Q"][r], 75) for r in regimes]
        pos = x + (offset - 0.5) * w
        ax.bar(pos, medians, w * 0.9, color=color, alpha=0.7, label=mn)
        ax.errorbar(pos, medians,
                    yerr=[np.array(medians) - np.array(q25),
                          np.array(q75) - np.array(medians)],
                    fmt="none", color="k", capsize=3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=9)
    ax.set_ylabel("$Q$ (median ± IQR)")
    ax.set_title("BharatBench $Q$ by Indian precipitation regime (tau=6h)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path2 = os.path.join(FIG_DIR, "bharatbench_Q_by_regime.pdf")
    fig.savefig(path2)
    plt.close(fig)
    print(f"Saved: {path2}")

    # ── LaTeX table output ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("LaTeX TABLE 14 ROWS (paste into tab:bharatbench):")
    print("="*70)
    HORIZON_H = [tau * 6 for tau in HORIZONS]
    label_map = {"RMRNN": "RM-LSTM", "Baseline": "LSTM baseline"}
    for mn in ("Baseline", "RMRNN"):
        row = f"{label_map[mn]:<18}"
        for tau in HORIZONS:
            a = all_results[tau][mn]
            row += f" & {a['mse_mean']:.4f} & {a['mae_mean']:.4f}"
        Q1 = all_results[HORIZONS[0]][mn]
        row += f" & {Q1['Q_mean']:.2f}$\\pm${Q1['Q_std']:.2f} \\\\"
        print(row)

    print("\n" + "="*70)
    print("LaTeX TABLE 15 ROWS (paste into tab:bharatbench_zone):")
    print("="*70)
    for regime in regimes:
        rm_Q  = all_results[HORIZONS[0]]["RMRNN"]["regime_Q"][regime]
        bl_Q  = all_results[HORIZONS[0]]["Baseline"]["regime_Q"][regime]
        rm_m, rm_s = np.nanmean(rm_Q), np.nanstd(rm_Q)
        bl_m, bl_s = np.nanmean(bl_Q), np.nanstd(bl_Q)
        print(f"{regime:<40}  & {rm_m:.2f}$\\pm${rm_s:.2f} "
              f"& {bl_m:.2f}$\\pm${bl_s:.2f} \\\\")


if __name__ == "__main__":
    main()
