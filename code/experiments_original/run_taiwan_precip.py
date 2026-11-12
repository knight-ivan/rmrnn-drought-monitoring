"""
run_taiwan_precip.py
====================
Real-data case study — Taiwan daily station precipitation (CWA/CODiS archive,
rebuilt by Raingel/historical_weather).

Design
------
  - Dataset  : daily precipitation, 5 major CWA stations, 2010-2020
               (Taipei 466920, Hualien 466990, Tainan 467410,
                Kaohsiung 467440, Taichung 467490)
  - Target   : next-day log-precipitation, standardised per station
  - Input    : 30-day sliding window
  - Split    : first 80% of days training (chronological), remainder test
  - Models   : RMRNNJoint (full BPTT) with RM on (lambda=0.1) vs off (lambda=0)
               — an exact RM-regularisation ablation on one architecture.
  - Metrics  : forecast MSE/MAE; hidden-path stability Q_path (raw increments,
               comparable across arms) and learned-defect Q_rm; and heavy-rain
               EARLY-WARNING skill — next-day exceedance of the per-station 95th
               percentile, scored by AUC and a rate-calibrated CSI/F1.
  - Reps     : one per station — natural replications across climate regimes
               (north / east / south / south / central Taiwan)

Data
----
  Produced by:
    python3 fetch_taiwan_hydromet.py --source raingel \
        --stations 466920,466990,467410,467440,467490 \
        --years 2010-2020 --outdir "<.../Data Sets/raingel_taiwan>"
  which writes raingel/daily_precip_long.csv (tidy date,station_id,precip_mm).

Outputs
-------
  experiments/taiwan_precip_results.npz  — per-station metrics
  figures/taiwan_precip_Q.pdf            — Q and Test MSE per station
  stdout                                  — per-station + aggregate table

Usage
-----
  python3 run_taiwan_precip.py [--csv PATH] [--window 30] [--seed 42]
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))        # experiments/precipitation/
SCRIPTS_DIR = os.path.dirname(os.path.dirname(HERE))            # scripts/
BASE_DIR    = os.path.dirname(SCRIPTS_DIR)                      # RM_Precipitation/
EXP_DIR     = os.path.join(BASE_DIR, "experiments")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "common"))

DEFAULT_CSV = os.path.join(
    BASE_DIR, "Data Sets", "raingel_taiwan", "raingel", "daily_precip_long.csv"
)

# ── hyperparameters (mirrors run_ghcn_stability.py for comparability) ──────────
WINDOW       = 30      # input context: 30 days
TRAIN_FRAC   = 0.80    # chronological train fraction
N_EPOCHS     = 200
WARMUP       = 5
LAMBDA_RM    = 0.1
LAMBDA_DECAY = 0.1
LR           = 5e-3
HIDDEN_DIM   = 32
BATCH_SIZE   = 32
THRESH_PCTILE = 95     # heavy-rain early-warning threshold (per-station pctile)
MAX_MISSING_FRAC = 0.25  # drop stations with >25% missing days (degenerate)
MIN_THR_MM = 5.0       # drop stations whose P95 wet threshold is below this

# Friendly station names for well-known CWA synoptic stations; any id not
# listed falls back to the raw station id in output.
STATION_NAMES = {
    "466880": "Banqiao",   "466900": "Tamsui",    "466910": "Anbu",
    "466920": "Taipei",    "466940": "Keelung",   "466990": "Hualien",
    "467050": "Xinwu",     "467060": "Suao",      "467080": "Yilan",
    "467410": "Tainan",    "467440": "Kaohsiung", "467490": "Taichung",
    "467530": "Alishan",   "467540": "Dawu",      "467550": "Yushan",
    "467590": "Hengchun",  "467610": "Chenggong", "467650": "SunMoonLake",
    "467660": "Taitung",
}


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess_series(arr_mm: np.ndarray, train_len: int):
    """Interpolate short gaps, log1p, standardise on the training split.

    Returns (series_norm, mu, sig) over the full series; mu/sig let callers
    map predictions back to log-mm via  x*sig + mu  then expm1.
    """
    arr = arr_mm.astype(float).copy()
    # Linear-interpolate gaps up to 5 days; clip remaining NaN to 0 mm.
    nan_idx = np.where(np.isnan(arr))[0]
    for i in nan_idx:
        left = i - 1
        right = i + 1
        while right < len(arr) and np.isnan(arr[right]):
            right += 1
        if left >= 0 and right < len(arr) and (right - left) <= 5:
            arr[i] = arr[left] + (arr[right] - arr[left]) * (i - left) / (right - left)
    arr = np.where(np.isnan(arr), 0.0, arr)
    arr = np.clip(arr, 0.0, None)

    arr_log = np.log1p(arr)
    tr = arr_log[:train_len]
    mu = tr.mean()
    sig = tr.std().clip(min=1e-8)
    return (arr_log - mu) / sig, mu, sig


def make_windows(series: np.ndarray, window: int):
    """Sliding windows: X (N, window, 1), y (N, 1) one-step-ahead."""
    N = len(series) - window
    X = np.stack([series[i: i + window, np.newaxis] for i in range(N)])
    y = series[window:, np.newaxis]
    return X, y


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC-AUC (probability a positive outranks a negative)."""
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    # Average ranks for ties.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_rank = np.zeros(counts.size)
    np.add.at(sum_rank, inv, ranks)
    ranks = (sum_rank / counts)[inv]
    auc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def warning_metrics(scores: np.ndarray, labels: np.ndarray, tau: float) -> dict:
    """Early-warning skill for predicting next-day heavy-rain exceedance.

    A warning is issued when the model's continuous forecast score exceeds the
    same threshold tau used to define the true exceedance label.

    Returns AUC (threshold-free ranking skill) plus a confusion-matrix-based
    Critical Success Index (CSI), F1, precision and recall at tau.
    """
    labels = labels.astype(bool)
    pred = scores > tau
    tp = int(np.sum(pred & labels))
    fp = int(np.sum(pred & ~labels))
    fn = int(np.sum(~pred & labels))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0   # = TP/(TP+FP+FN)
    return {
        "auc": roc_auc(scores, labels),
        "csi": csi, "f1": f1, "precision": prec, "recall": rec,
        "n_pos": int(labels.sum()),
    }


# ── single-station training ───────────────────────────────────────────────────

def run_station(sid: str, arr_mm: np.ndarray, seed: int) -> dict:
    # RMRNNJoint trains the forward cell by full BPTT, so the RM gradient
    # actually shapes the hidden trajectory. The baseline is the *same*
    # architecture with lambda_rm = 0 (plain BPTT Elman RNN) — an exact
    # RM-on / RM-off comparison rather than a different baseline class.
    from model_ext import RMRNNJoint, GRUJoint
    from model import hidden_increment_total

    name = STATION_NAMES.get(sid, sid)
    n_days = len(arr_mm)
    train_len = int(n_days * TRAIN_FRAC)

    # ── data-quality filter ───────────────────────────────────────────────────
    # Stations with heavy missingness collapse to a near-constant series once
    # gaps are zero-filled; standardising then divides by ~0 and BPTT diverges.
    miss = float(np.mean(np.isnan(arr_mm)))
    if miss > MAX_MISSING_FRAC:
        print(f"  {name:11s} ({sid})  SKIPPED — {miss:.0%} missing")
        return None

    series_norm, mu, sig = preprocess_series(arr_mm, train_len)

    X_all, y_all = make_windows(series_norm, WINDOW)
    split = train_len - WINDOW
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_te, y_te = X_all[split:], y_all[split:]
    if len(X_tr) < 50 or len(X_te) < 20:
        print(f"  {name:11s} ({sid})  SKIPPED — too few windows")
        return None

    # Heavy-rain early-warning threshold: per-station 95th percentile of daily
    # precipitation, estimated on the training split, mapped into the model's
    # standardised-log output space  tau = (log1p(thr_mm) - mu) / sig.
    train_mm = np.clip(np.nan_to_num(arr_mm[:train_len], nan=0.0), 0.0, None)
    thr_mm = float(np.percentile(train_mm, THRESH_PCTILE))
    if thr_mm < MIN_THR_MM or sig < 1e-3:
        print(f"  {name:11s} ({sid})  SKIPPED — degenerate "
              f"(P{THRESH_PCTILE}={thr_mm:.1f}mm, sig={sig:.2e})")
        return None
    tau = (np.log1p(thr_mm) - mu) / sig

    rng = np.random.RandomState(seed)
    n_tr = len(X_tr)
    out = {"sid": sid, "name": STATION_NAMES.get(sid, sid),
           "n_train": n_tr, "n_test": len(X_te)}

    for model_name, lam in [("RMRNN", LAMBDA_RM), ("VanillaRNN", 0.0), ("GRU", 0.0)]:
        is_rm = model_name == "RMRNN"
        is_gru = model_name == "GRU"
        if is_gru:
            model = GRUJoint(1, HIDDEN_DIM, 1, seed=seed)
        else:
            model = RMRNNJoint(1, HIDDEN_DIM, 1, lambda_rm=lam, seed=seed)

        conv_epoch = N_EPOCHS
        prev = np.inf
        for epoch in range(N_EPOCHS):
            if is_rm:
                frac = max(0, epoch - WARMUP) / max(N_EPOCHS - WARMUP - 1, 1)
                model.lambda_rm = LAMBDA_RM * (LAMBDA_DECAY ** frac) if epoch >= WARMUP else 0.0
            idx = rng.choice(n_tr, size=min(BATCH_SIZE, n_tr), replace=False)
            info = model.train_step(X_tr[idx], y_tr[idx], lr=LR)
            cur = info["loss_task"]
            if abs(prev - cur) < 1e-4 and epoch > WARMUP:
                conv_epoch = epoch
                break
            prev = cur

        h, _ = model._forward(X_te)
        y_pred = h[:, -1, :] @ model.cell["Wr"].T + model.cell["br"]
        h_tr, _ = model._forward(X_tr)
        y_pred_tr = h_tr[:, -1, :] @ model.cell["Wr"].T + model.cell["br"]
        if not (np.isfinite(y_pred).all() and np.isfinite(h).all()):
            print(f"  {name:11s} ({sid})  SKIPPED — {model_name} diverged")
            return None
        # Comparable across both arms: raw hidden-path increments, and the
        # learned-backward-map quasi-martingale defect.
        Q_path = hidden_increment_total(h)
        Q_rm = model.quasi_martingale_total(X_te[:min(50, len(X_te))])

        # Early-warning skill: predict next-day heavy-rain exceedance (y > tau).
        # A mean-regression forecast rarely reaches the event magnitude tau, so
        # the warning decision threshold is *calibrated* on the training
        # predictions to match the climatological exceedance rate (no test
        # leakage). AUC itself is threshold-free.
        labels = (y_te.flatten() > tau)
        base_rate = float(np.mean(y_tr.flatten() > tau))
        if base_rate > 0:
            tau_dec = float(np.quantile(y_pred_tr.flatten(), 1.0 - base_rate))
        else:
            tau_dec = float("inf")
        wm = warning_metrics(y_pred.flatten(), labels, tau_dec)

        out[model_name] = {
            "mse": float(np.mean((y_pred - y_te) ** 2)),
            "mae": float(np.mean(np.abs(y_pred - y_te))),
            "Q_path": Q_path,
            "Q_rm": float(Q_rm),
            "defect_norm": Q_path / max(WINDOW - 1, 1),
            "conv_epoch": conv_epoch,
            "auc": wm["auc"], "csi": wm["csi"], "f1": wm["f1"],
            "precision": wm["precision"], "recall": wm["recall"],
        }
    out["thr_mm"] = thr_mm
    out["n_pos_test"] = int((y_te.flatten() > tau).sum())
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global WINDOW
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    WINDOW = args.window

    import data as datamod
    if not os.path.exists(args.csv):
        print(f"DATA NOT FOUND: {args.csv}\n"
              f"Generate it with fetch_taiwan_hydromet.py --source raingel ...")
        sys.exit(1)

    M, dates, stations = datamod.load_precip_matrix(args.csv)
    print("Taiwan daily precipitation — real-data case study")
    print(f"  {M.shape[0]} days  ({dates[0]} -> {dates[-1]})  |  "
          f"{len(stations)} stations  |  window={WINDOW}d\n")

    results = []
    for j, sid in enumerate(stations):
        r = run_station(sid, M[:, j], args.seed + j)
        if r is None:
            continue  # run_station already printed the skip reason
        results.append(r)
        rm, bl = r["RMRNN"], r["VanillaRNN"]
        print(f"  {r['name']:10s} ({sid})  "
              f"RMRNN: MSE={rm['mse']:.4f} Qpath={rm['Q_path']:.2f}  |  "
              f"Vanilla: MSE={bl['mse']:.4f} Qpath={bl['Q_path']:.2f}")

    if not results:
        print("No stations produced results.")
        sys.exit(1)

    # ── aggregate table ───────────────────────────────────────────────────────
    def agg(model, key):
        v = np.array([r[model][key] for r in results])
        return v.mean(), v.std()

    print("\n" + "=" * 72)
    print("TAIWAN PRECIPITATION — aggregate over stations")
    print("=" * 72)
    print(f"{'Model':12s}  {'TestMSE':>16s}  {'TestMAE':>16s}  "
          f"{'Q_path':>16s}  {'Conv.Ep.':>9s}")
    print("-" * 72)
    for mn in ("RMRNN", "VanillaRNN", "GRU"):
        msem, mses = agg(mn, "mse")
        maem, maes = agg(mn, "mae")
        qm, qs = agg(mn, "Q_path")
        cem, _ = agg(mn, "conv_epoch")
        print(f"{mn:12s}  {msem:7.4f}±{mses:6.4f}  {maem:7.4f}±{maes:6.4f}  "
              f"{qm:7.2f}±{qs:6.2f}  {cem:9.1f}")

    rm_mse, _ = agg("RMRNN", "mse")
    bl_mse, _ = agg("VanillaRNN", "mse")
    rm_q, _ = agg("RMRNN", "Q_path")
    bl_q, _ = agg("VanillaRNN", "Q_path")
    print("-" * 72)
    print(f"  Test-MSE change (RMRNN vs Vanilla):  {100*(rm_mse-bl_mse)/bl_mse:+.1f}%")
    print(f"  Q_path change (RMRNN vs Vanilla):    {100*(rm_q-bl_q)/abs(bl_q):+.1f}%"
          if bl_q != 0 else "  Q_path change: n/a")

    # ── early-warning skill (heavy-rain exceedance) ───────────────────────────
    print("\n" + "=" * 72)
    print(f"HEAVY-RAIN EARLY WARNING — next-day exceedance of station P{THRESH_PCTILE}")
    print("=" * 72)
    npos = np.array([r["n_pos_test"] for r in results])
    print(f"  threshold per station: {[round(r['thr_mm'],1) for r in results]} mm")
    print(f"  positive test events:  {npos.tolist()}  (total {int(npos.sum())})\n")
    print(f"{'Model':12s}  {'AUC':>14s}  {'CSI':>14s}  {'F1':>14s}  "
          f"{'Recall':>14s}")
    print("-" * 72)
    for mn in ("RMRNN", "VanillaRNN", "GRU"):
        am, asd = agg(mn, "auc")
        cm, csd = agg(mn, "csi")
        fm, fsd = agg(mn, "f1")
        rm_, rsd = agg(mn, "recall")
        print(f"{mn:12s}  {am:6.3f}±{asd:5.3f}  {cm:6.3f}±{csd:5.3f}  "
              f"{fm:6.3f}±{fsd:5.3f}  {rm_:6.3f}±{rsd:5.3f}")
    rm_auc, _ = agg("RMRNN", "auc")
    bl_auc, _ = agg("VanillaRNN", "auc")
    print("-" * 72)
    print(f"  AUC change (RMRNN vs Vanilla): {100*(rm_auc-bl_auc)/bl_auc:+.1f}%")

    # ── save ─────────────────────────────────────────────────────────────────
    save = {
        "station_ids": np.array([r["sid"] for r in results], dtype=object),
        "names":       np.array([r["name"] for r in results], dtype=object),
    }
    save["thr_mm"] = np.array([r["thr_mm"] for r in results])
    save["n_pos_test"] = np.array([r["n_pos_test"] for r in results])
    for mn in ("RMRNN", "VanillaRNN", "GRU"):
        for key in ("mse", "mae", "Q_path", "Q_rm", "defect_norm", "conv_epoch",
                    "auc", "csi", "f1", "precision", "recall"):
            save[f"{mn}/{key}"] = np.array([r[mn][key] for r in results])
    npz_path = os.path.join(EXP_DIR, "taiwan_precip_results.npz")
    np.savez(npz_path, **save)
    print(f"\nSaved: {npz_path}")

    # ── figure: per-station Q and Test MSE ────────────────────────────────────
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(x - w/2, [r["RMRNN"]["Q_path"] for r in results], w, label="RMRNN", color="#1f77b4")
    ax1.bar(x + w/2, [r["VanillaRNN"]["Q_path"] for r in results], w, label="Vanilla", color="#ff7f0e")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel(r"Path-increment total $Q_{\mathrm{path}}$"); ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y"); ax1.set_title("Hidden-path stability per station")

    ax2.bar(x - w/2, [r["RMRNN"]["mse"] for r in results], w, label="RMRNN", color="#1f77b4")
    ax2.bar(x + w/2, [r["VanillaRNN"]["mse"] for r in results], w, label="Vanilla", color="#ff7f0e")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Test MSE (standardised log-precip)"); ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y"); ax2.set_title("One-step forecast error per station")
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "taiwan_precip_Q.pdf")
    fig.savefig(fig_path); plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
