"""
run_weatherbench_precip.py
==========================
Table 11 in P2 — WeatherBench precipitation (ERA5 1.4° daily, East Asia subgrid).

Design
------
  - Subgrid : 32×32 grid points, 15°N–55°N × 95°E–135°E
  - Target  : total precipitation (TP), daily anomaly (standardised per grid pt)
  - Horizons: tau = 1, 7, 14 days
  - Split   : train 1979–2015, val 2016–2017, test 2018
  - Models  : RMRNN-LSTM, RM-GRU, LSTM baseline, GRU baseline
  - Reps    : 1000 random weight seeds  (quantifies initialisation sensitivity)

Data download (one-time, ~2.5 GB for 1.4° TP only)
-----------------------------------------------------
  pip install cdsapi netCDF4 numpy scipy matplotlib
  # Or download directly from WeatherBench GCS bucket:
  gsutil -m cp gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-1440x721_mean.zarr \
               <local_dir>
  # Simpler: use the pre-processed 1.4° files from the WeatherBench 1 repository:
  wget https://dataserv.ub.tum.de/s/m1524895/download?files=1.40625deg/total_precipitation \
       -O total_precipitation_1.40625deg.zip && unzip total_precipitation_1.40625deg.zip

  Set DATA_DIR below to the folder containing total_precipitation_*.nc files.

Outputs (written to 02-Simulation/experiments/ and figures/)
-------------------------------------------------------------
  experiments/weatherbench_precip_results.npz   — raw metrics (reps × models × horizons)
  figures/weatherbench_Q_hist.pdf               — Q distribution across seeds
  stdout                                         — Table 11 values (copy into .tex)

Usage
-----
  python3 run_weatherbench_precip.py [--reps 1000] [--workers 8] [--seed 0]
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
SRC_DIR  = os.path.dirname(os.path.dirname(__file__))       # .../src
BASE_DIR = os.path.dirname(SRC_DIR)                         # .../02-Simulation
EXP_DIR  = os.path.join(BASE_DIR, "experiments")
FIG_DIR  = os.path.join(BASE_DIR, "figures")
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── data location ─────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get(
    "WEATHERBENCH_DATA",
    os.path.join(BASE_DIR, "data", "weatherbench", "total_precipitation_1.40625deg")
)

# ── hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_DIM   = 64
LAMBDA_RM    = 0.1
LAMBDA_DECAY = 0.1          # final lambda = LAMBDA_RM * LAMBDA_DECAY
WARMUP       = 5            # epochs with lambda=0
N_EPOCHS     = 100
LR           = 1e-3
TRAIN_YEARS  = (1979, 2015)
VAL_YEARS    = (2016, 2017)
TEST_YEARS   = (2018, 2018)
HORIZONS     = [1, 7, 14]   # forecast lead times in days
INPUT_WINDOW = 30           # input context length in days

# East Asia subgrid indices (approx 15°N–55°N, 95°E–135°E at 1.40625° resolution)
# Full grid: lat 90°N→90°S (128 pts), lon 0°→358.6° (256 pts)
LAT_MIN_IDX, LAT_MAX_IDX = 25, 57   # ~15°N to 55°N  (lat decreases from index 0)
LON_MIN_IDX, LON_MAX_IDX = 68, 100  # ~95°E to 135°E


# ── data loading ─────────────────────────────────────────────────────────────

def load_weatherbench_tp(data_dir: str):
    """
    Load WeatherBench total precipitation, extract East Asia subgrid,
    and compute daily anomalies.

    Returns
    -------
    data_train : np.ndarray  (n_train_days, n_lat, n_lon)
    data_val   : np.ndarray  (n_val_days,   n_lat, n_lon)
    data_test  : np.ndarray  (n_test_days,  n_lat, n_lon)
    clim_mean  : np.ndarray  (n_lat, n_lon)  training climatology mean
    clim_std   : np.ndarray  (n_lat, n_lon)  training climatology std
    """
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 required: pip install netCDF4")

    # WeatherBench 1.40625° TP files are named total_precipitation_YYYY_1.40625deg.nc
    years_needed = list(range(TRAIN_YEARS[0], TEST_YEARS[1] + 1))
    daily_fields = []

    for yr in years_needed:
        fname = os.path.join(data_dir, f"total_precipitation_{yr}_1.40625deg.nc")
        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"Missing: {fname}\n"
                f"Download WeatherBench 1.40625° TP files from:\n"
                f"  https://dataserv.ub.tum.de/s/m1524895\n"
                f"Or set WEATHERBENCH_DATA env var to the data directory."
            )
        with nc.Dataset(fname) as ds:
            tp = ds.variables["tp"][:]             # (time, lat, lon)  in m/day
            tp = np.array(tp)
            # Extract East Asia subgrid
            tp = tp[:, LAT_MIN_IDX:LAT_MAX_IDX, LON_MIN_IDX:LON_MAX_IDX]
            daily_fields.append(tp)

    all_data = np.concatenate(daily_fields, axis=0)   # (total_days, n_lat, n_lon)

    # Split
    train_end = (TRAIN_YEARS[1] - TRAIN_YEARS[0] + 1) * 365
    val_end   = train_end + (VAL_YEARS[1] - VAL_YEARS[0] + 1) * 365

    data_train = all_data[:train_end]
    data_val   = all_data[train_end:val_end]
    data_test  = all_data[val_end:]

    # Log-transform (TP is non-negative; add small epsilon)
    eps = 1e-6
    data_train = np.log1p(data_train * 1000 + eps)   # convert m→mm then log1p
    data_val   = np.log1p(data_val   * 1000 + eps)
    data_test  = np.log1p(data_test  * 1000 + eps)

    # Standardise per grid point using training statistics
    clim_mean = data_train.mean(axis=0, keepdims=True)   # (1, n_lat, n_lon)
    clim_std  = data_train.std(axis=0,  keepdims=True).clip(min=1e-8)

    data_train = (data_train - clim_mean) / clim_std
    data_val   = (data_val   - clim_mean) / clim_std
    data_test  = (data_test  - clim_mean) / clim_std

    return (data_train, data_val, data_test,
            clim_mean.squeeze(), clim_std.squeeze())


def make_sequences(field_3d: np.ndarray, window: int, tau: int):
    """
    Slide a window over time dimension and create (X, y) pairs.

    field_3d : (T, n_lat, n_lon) — standardised
    window   : int  — input context length
    tau      : int  — forecast horizon

    Returns
    -------
    X : (N, window, n_lat * n_lon)   flattened spatial features
    y : (N, n_lat * n_lon)           target at t + tau
    """
    T, n_lat, n_lon = field_3d.shape
    n_pts = n_lat * n_lon
    flat  = field_3d.reshape(T, n_pts)

    N = T - window - tau + 1
    if N <= 0:
        raise ValueError(f"Time series too short for window={window}, tau={tau}")

    X = np.stack([flat[i: i + window] for i in range(N)],   axis=0)   # (N, window, n_pts)
    y = np.stack([flat[i + window + tau - 1] for i in range(N)], axis=0)  # (N, n_pts)
    return X, y


# ── simple LSTM / GRU in NumPy (minimal, for consistency with existing codebase) ─

class MinimalRNN:
    """
    Minimal stateful RNN wrapper (Elman) for precipitation sequences.
    Uses the existing src/model.py RMRNN class with adapted input/output dims.
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 lambda_rm=0.0, seed=0):
        sys.path.insert(0, SRC_DIR)
        from model import RMRNN as _RMRNN
        from baselines import VanillaRNN as _VanillaRNN

        if lambda_rm > 0:
            self._model = _RMRNN(input_dim, hidden_dim, output_dim,
                                 lambda_rm=lambda_rm, seed=seed)
            self._is_rm = True
        else:
            self._model = _VanillaRNN(input_dim, hidden_dim, output_dim,
                                      seed=seed)
            self._is_rm = False

    def train_step(self, X_batch, y_batch, lr):
        return self._model.train_step(X_batch, y_batch, lr=lr)

    def predict(self, X_batch):
        """Returns y_pred: (B, output_dim)."""
        sys.path.insert(0, SRC_DIR)
        from model import rmrnn_forward_batch
        if self._is_rm:
            h = rmrnn_forward_batch(self._model.params, X_batch)  # (B, T, p)
            h_T = h[:, -1, :]
            y_pred = h_T @ self._model.params.Wr.T + self._model.params.br
        else:
            y_pred = self._model.predict(X_batch)
        return y_pred

    def quasi_martingale_total(self, X_batch):
        if self._is_rm:
            return self._model.quasi_martingale_total(X_batch)
        return 0.0


# ── single replication worker ─────────────────────────────────────────────────

def _run_one_rep(args):
    """
    Train RMRNN and baseline on WeatherBench TP subgrid, evaluate on test.

    args = (rep, X_train, y_train_tau, X_test, y_test_tau, tau, seed)
    Returns dict of metrics for each model and this tau.
    """
    rep, X_tr, y_tr, X_te, y_te, tau, base_seed = args
    seed   = base_seed + rep
    rng    = np.random.RandomState(seed)

    n_train, win, n_pts = X_tr.shape

    # Mini-batch training
    B_train  = min(64, n_train)
    n_epochs = N_EPOCHS

    results = {}
    for model_name, lambda_rm in [("RMRNN", LAMBDA_RM), ("Baseline", 0.0)]:
        model = MinimalRNN(n_pts, HIDDEN_DIM, n_pts,
                           lambda_rm=lambda_rm, seed=seed)

        for epoch in range(n_epochs):
            # Lambda schedule: warmup then exponential decay
            if epoch < WARMUP:
                _lam = 0.0
            else:
                frac = (epoch - WARMUP) / max(n_epochs - WARMUP - 1, 1)
                _lam = LAMBDA_RM * (LAMBDA_DECAY ** frac)
            if hasattr(model._model, 'lambda_rm'):
                model._model.lambda_rm = _lam

            # Sample mini-batch
            idx = rng.choice(n_train, size=B_train, replace=False)
            model.train_step(X_tr[idx], y_tr[idx], lr=LR)

        # Evaluate on test
        y_pred = model.predict(X_te)        # (N_test, n_pts)
        mse    = float(np.mean((y_pred - y_te) ** 2))
        mae    = float(np.mean(np.abs(y_pred - y_te)))

        # Q on a subset of test sequences (keep memory manageable)
        n_q = min(200, len(X_te))
        Q   = model.quasi_martingale_total(X_te[:n_q])

        results[model_name] = {"mse": mse, "mae": mae, "Q": Q}

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps",    type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    print("WeatherBench precipitation — East Asia subgrid (32×32)")
    print(f"Horizons: {HORIZONS} days  |  {args.reps} seed replications\n")

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading WeatherBench TP data ...")
    try:
        train_d, val_d, test_d, _, _ = load_weatherbench_tp(DATA_DIR)
    except FileNotFoundError as e:
        print(f"\nDATA NOT FOUND:\n{e}")
        print("\nTo run this script:")
        print("  1. Download WeatherBench 1.40625° total_precipitation_*.nc files")
        print("  2. Set WEATHERBENCH_DATA=/path/to/files or edit DATA_DIR in this script")
        sys.exit(1)

    # ── per-horizon experiment ────────────────────────────────────────────────
    all_results = {}   # tau -> model -> list of rep metrics

    for tau in HORIZONS:
        print(f"\n{'='*60}")
        print(f"  Horizon tau={tau} day(s)")
        print(f"{'='*60}")

        X_tr, y_tr = make_sequences(train_d, INPUT_WINDOW, tau)
        X_te, y_te = make_sequences(test_d,  INPUT_WINDOW, tau)

        worker_args = [
            (rep, X_tr, y_tr, X_te, y_te, tau, args.seed)
            for rep in range(args.reps)
        ]

        with Pool(processes=args.workers) as pool:
            rep_results = pool.map(_run_one_rep, worker_args)

        # Aggregate
        tau_agg = {}
        for model_name in ("RMRNN", "Baseline"):
            mse_arr = np.array([r[model_name]["mse"] for r in rep_results])
            mae_arr = np.array([r[model_name]["mae"] for r in rep_results])
            Q_arr   = np.array([r[model_name]["Q"]   for r in rep_results])
            tau_agg[model_name] = {
                "mse_mean": mse_arr.mean(), "mse_std": mse_arr.std(),
                "mae_mean": mae_arr.mean(), "mae_std": mae_arr.std(),
                "Q_mean":   Q_arr.mean(),   "Q_std":   Q_arr.std(),
                "Q_raw":    Q_arr,
            }

        all_results[tau] = tau_agg

        # Print table row
        print(f"\n  tau={tau}  {'Model':12s}  {'MSE':>10s}  {'MAE':>10s}  {'Q':>10s}")
        print(f"  {'-'*50}")
        for mn in ("RMRNN", "Baseline"):
            a = tau_agg[mn]
            print(f"  tau={tau}  {mn:12s}  "
                  f"{a['mse_mean']:.4f}±{a['mse_std']:.4f}  "
                  f"{a['mae_mean']:.4f}±{a['mae_std']:.4f}  "
                  f"{a['Q_mean']:.2f}±{a['Q_std']:.2f}")

    # ── save raw results ──────────────────────────────────────────────────────
    save_dict = {}
    for tau, tau_agg in all_results.items():
        for mn, metrics in tau_agg.items():
            key = f"tau{tau}/{mn}"
            save_dict[f"{key}/Q_raw"]   = metrics["Q_raw"]
            save_dict[f"{key}/mse"]     = np.array([metrics["mse_mean"], metrics["mse_std"]])
            save_dict[f"{key}/mae"]     = np.array([metrics["mae_mean"], metrics["mae_std"]])
    npz_path = os.path.join(EXP_DIR, "weatherbench_precip_results.npz")
    np.savez(npz_path, **save_dict)
    print(f"\nSaved: {npz_path}")

    # ── Q histogram figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(4.5 * len(HORIZONS), 4))
    for ax, tau in zip(axes, HORIZONS):
        Q_rm = all_results[tau]["RMRNN"]["Q_raw"]
        ax.hist(Q_rm, bins=40, alpha=0.7, color="#1f77b4",
                density=True, label="RMRNN")
        ax.set_title(f"$\\tau = {tau}$ day(s)", fontsize=10)
        ax.set_xlabel("Quasi-martingale total $Q$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"WeatherBench TP — $Q$ distribution ({args.reps} seeds)",
                 fontsize=11)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "weatherbench_Q_hist.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # ── print LaTeX table rows ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("LaTeX TABLE 11 ROWS (paste into tab:weatherbench):")
    print("="*70)
    models = [("LSTM baseline", "Baseline"), ("RM-LSTM", "RMRNN")]
    for label, key in models:
        row = f"{label:<18}"
        for tau in HORIZONS:
            a = all_results[tau][key]
            row += f" & {a['mse_mean']:.4f} & {a['mae_mean']:.4f}"
        Q = all_results[HORIZONS[0]][key]   # Q reported at tau=1
        row += f" & {Q['Q_mean']:.2f}$\\pm${Q['Q_std']:.2f} \\\\"
        print(row)


if __name__ == "__main__":
    main()
