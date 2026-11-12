"""
run_taiwan_precip_deepbaselines.py
==================================
Genuine deep temporal baselines on the REAL Taiwan CWA daily station archive,
to sit alongside the RMRNN / RMRNN(lambda=0) / GRU comparison of
run_taiwan_precip.py (Table tab:realdata in the paper).

Two additional, standard sequence models trained by the SAME protocol
(30-day window, chronological 80/20 split, next-day standardised
log-precipitation target, per-station 95th-percentile heavy-rain warning):

  * LSTM : single-layer PyTorch nn.LSTM -> linear readout.  Recurrent, so the
           hidden-path increment total Q_path is defined and reported.
  * TCN  : a small causal dilated temporal convolutional network (no recurrent
           hidden state, so Q_path is not applicable -- reported as --).

These answer the reviewer question "how does the RM-regularised recurrent model
compare with off-the-shelf deep sequence models on real data?"  Metrics are
framework-agnostic (MSE, MAE, warning AUC/CSI), so they are directly comparable
to the NumPy models in run_taiwan_precip.py.

Usage:  python3 run_taiwan_precip_deepbaselines.py [--seed 42]
Writes: experiments/taiwan_precip_deepbaselines.npz
"""
from __future__ import annotations
import argparse, os, sys, importlib.util
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(os.path.dirname(HERE))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)
EXP_DIR = os.path.join(BASE_DIR, "experiments")
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "common"))

# reuse the EXACT preprocessing / windowing / metrics / constants from the
# main real-data script so the comparison is apples-to-apples
_spec = importlib.util.spec_from_file_location("rtp", os.path.join(HERE, "run_taiwan_precip.py"))
rtp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rtp)

WINDOW, TRAIN_FRAC, HIDDEN_DIM = rtp.WINDOW, rtp.TRAIN_FRAC, rtp.HIDDEN_DIM
THRESH_PCTILE, MAX_MISSING_FRAC, MIN_THR_MM = rtp.THRESH_PCTILE, rtp.MAX_MISSING_FRAC, rtp.MIN_THR_MM
N_EPOCHS, LR = 200, 5e-3


class LSTMNet(nn.Module):
    def __init__(self, h=HIDDEN_DIM):
        super().__init__()
        self.lstm = nn.LSTM(1, h, batch_first=True)
        self.fc = nn.Linear(h, 1)
    def forward(self, x):
        H, _ = self.lstm(x)            # (B,T,h)
        return self.fc(H[:, -1, :]), H


class TCN(nn.Module):
    """Causal dilated temporal CNN (3 blocks, dilations 1/2/4)."""
    def __init__(self, ch=32, k=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        cin = 1
        for d in (1, 2, 4):
            pad = (k - 1) * d
            self.blocks.append(nn.Conv1d(cin, ch, k, padding=pad, dilation=d))
            cin = ch
        self.relu = nn.ReLU()
        self.fc = nn.Linear(ch, 1)
        self._pads = [(k - 1) * d for d in (1, 2, 4)]
    def forward(self, x):              # x: (B,T,1)
        z = x.transpose(1, 2)          # (B,1,T)
        for conv, pad in zip(self.blocks, self._pads):
            z = conv(z)
            if pad: z = z[:, :, :-pad]  # causal: drop right padding
            z = self.relu(z)
        return self.fc(z[:, :, -1]), None


def train_eval(ModelCls, X_tr, y_tr, X_te, y_te, seed, recurrent):
    torch.manual_seed(seed)
    model = ModelCls()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xtr = torch.tensor(X_tr, dtype=torch.float32); ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xte = torch.tensor(X_te, dtype=torch.float32)
    lossf = nn.MSELoss()
    bs = min(32, len(Xtr)); rng = np.random.RandomState(seed)
    model.train()
    for ep in range(N_EPOCHS):
        idx = rng.choice(len(Xtr), size=bs, replace=False)
        opt.zero_grad()
        pred, _ = model(Xtr[idx])
        loss = lossf(pred, ytr[idx])
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        yp, H = model(Xte)
        yp = yp.numpy()
        q_path = float("nan")
        if recurrent and H is not None:
            Hn = H.numpy()                       # (B,T,h)
            inc = np.linalg.norm(np.diff(Hn, axis=1), axis=2).sum(axis=1)  # per-seq path length
            q_path = float(inc.mean())
    return yp, q_path


def run_station(sid, arr_mm, seed):
    name = rtp.STATION_NAMES.get(sid, sid)
    n = len(arr_mm); train_len = int(n * TRAIN_FRAC)
    if float(np.mean(np.isnan(arr_mm))) > MAX_MISSING_FRAC:
        return None
    series, mu, sig = rtp.preprocess_series(arr_mm, train_len)
    X, y = rtp.make_windows(series, WINDOW)
    split = train_len - WINDOW
    X_tr, y_tr, X_te, y_te = X[:split], y[:split], X[split:], y[split:]
    if len(X_tr) < 50 or len(X_te) < 20:
        return None
    train_mm = np.clip(np.nan_to_num(arr_mm[:train_len], nan=0.0), 0.0, None)
    thr_mm = float(np.percentile(train_mm, THRESH_PCTILE))
    if thr_mm < MIN_THR_MM or sig < 1e-3:
        return None
    tau = (np.log1p(thr_mm) - mu) / sig
    labels = (y_te.flatten() > tau)
    base = float(np.mean(y_tr.flatten() > tau))

    out = {"sid": sid, "name": name}
    for mn, Cls, rec in [("LSTM", LSTMNet, True), ("TCN", TCN, False)]:
        yp, qpath = train_eval(Cls, X_tr, y_tr, X_te, y_te, seed, rec)
        # decision threshold calibrated on TRAIN preds to the climatological rate
        with torch.no_grad():
            m = Cls(); torch.manual_seed(seed)  # not used; predictions below from trained model
        # recompute train preds from the trained model for calibration
        # (re-run train_eval would retrain; instead calibrate on test-free train preds)
        out[mn] = {"yp": yp, "qpath": qpath, "tau": tau, "labels": labels, "base": base,
                   "y_te": y_te}
    out["thr_mm"] = thr_mm; out["n_pos_test"] = int(labels.sum())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=rtp.DEFAULT_CSV); args = ap.parse_args()
    import data as datamod
    M, dates, stations = datamod.load_precip_matrix(args.csv)
    print(f"Deep baselines on real CWA archive: {M.shape[0]} days, {len(stations)} stations")
    # we need train-prediction calibration; redo per-station with calibrated decision threshold
    res = {"LSTM": [], "TCN": []}
    npos = []
    for j, sid in enumerate(stations):
        arr = M[:, j]; name = rtp.STATION_NAMES.get(sid, sid)
        n = len(arr); train_len = int(n * TRAIN_FRAC)
        if float(np.mean(np.isnan(arr))) > MAX_MISSING_FRAC: continue
        series, mu, sig = rtp.preprocess_series(arr, train_len)
        X, y = rtp.make_windows(series, WINDOW); split = train_len - WINDOW
        X_tr, y_tr, X_te, y_te = X[:split], y[:split], X[split:], y[split:]
        if len(X_tr) < 50 or len(X_te) < 20: continue
        train_mm = np.clip(np.nan_to_num(arr[:train_len], nan=0.0), 0.0, None)
        thr_mm = float(np.percentile(train_mm, THRESH_PCTILE))
        if thr_mm < MIN_THR_MM or sig < 1e-3: continue
        tau = (np.log1p(thr_mm) - mu) / sig
        labels = (y_te.flatten() > tau); base = float(np.mean(y_tr.flatten() > tau))
        npos.append(int(labels.sum()))
        for mn, Cls, rec in [("LSTM", LSTMNet, True), ("TCN", TCN, False)]:
            # train once, get both train & test preds for calibration + scoring
            torch.manual_seed(args.seed + j); model = Cls()
            opt = torch.optim.Adam(model.parameters(), lr=LR); lossf = nn.MSELoss()
            Xtr = torch.tensor(X_tr, dtype=torch.float32); ytr = torch.tensor(y_tr, dtype=torch.float32)
            rng = np.random.RandomState(args.seed + j); bs = min(32, len(Xtr))
            model.train()
            for ep in range(N_EPOCHS):
                idx = rng.choice(len(Xtr), size=bs, replace=False)
                opt.zero_grad(); pred, _ = model(Xtr[idx]); loss = lossf(pred, ytr[idx])
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                yp_te, H = model(torch.tensor(X_te, dtype=torch.float32))
                yp_tr, _ = model(Xtr)
                yp_te = yp_te.numpy().flatten(); yp_tr = yp_tr.numpy().flatten()
                qpath = float("nan")
                if rec and H is not None:
                    Hn = H.numpy(); qpath = float(np.linalg.norm(np.diff(Hn, axis=1), axis=2).sum(axis=1).mean())
            mse = float(np.mean((yp_te - y_te.flatten())**2)); mae = float(np.mean(np.abs(yp_te - y_te.flatten())))
            tau_dec = float(np.quantile(yp_tr, 1.0 - base)) if base > 0 else float("inf")
            wm = rtp.warning_metrics(yp_te, labels, tau_dec)
            res[mn].append({"mse": mse, "mae": mae, "qpath": qpath, "auc": wm["auc"], "csi": wm["csi"]})
        print(f"  {name:11s} LSTM mse={res['LSTM'][-1]['mse']:.3f} auc={res['LSTM'][-1]['auc']:.3f} | "
              f"TCN mse={res['TCN'][-1]['mse']:.3f} auc={res['TCN'][-1]['auc']:.3f}", flush=True)

    print("\n=== aggregate (mean±SD over %d stations) ===" % len(res["LSTM"]))
    save = {}
    for mn in ("LSTM", "TCN"):
        for key in ("mse", "mae", "qpath", "auc", "csi"):
            v = np.array([r[key] for r in res[mn]], dtype=float)
            save[f"{mn}/{key}"] = v
        def ms(k):
            v = np.array([r[k] for r in res[mn]], dtype=float); v = v[np.isfinite(v)]
            return (v.mean(), v.std()) if len(v) else (float('nan'), float('nan'))
        print(f"{mn:5s}  MSE {ms('mse')[0]:.3f}±{ms('mse')[1]:.3f}  MAE {ms('mae')[0]:.3f}±{ms('mae')[1]:.3f}  "
              f"Qpath {ms('qpath')[0]:.2f}  AUC {ms('auc')[0]:.3f}±{ms('auc')[1]:.3f}  CSI {ms('csi')[0]:.3f}±{ms('csi')[1]:.3f}")
    np.savez(os.path.join(EXP_DIR, "taiwan_precip_deepbaselines.npz"), **save)
    print("saved experiments/taiwan_precip_deepbaselines.npz")


if __name__ == "__main__":
    main()
