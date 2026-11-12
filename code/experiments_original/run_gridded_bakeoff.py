"""
run_gridded_bakeoff.py
======================
Gridded SYNTHETIC nowcasting bake-off: the forecast-skill comparison the
spatial deep models actually need (they are image/grid-sequence models and our
station testbeds are not gridded).

A 2D synthetic precipitation field (spatially correlated, seasonal, AR(1) in
time, gamma intensity) is generated; given the last L frames each model
predicts the next frame.  Models, all trained by the same protocol:

  baselines : Persistence (last frame), Climatology (seasonal-mean frame)
  spatial DL: ConvLSTM, PredRNN (ST-LSTM core), U-Net (L frames -> next frame)
  recurrent : GRU (per-cell, shared weights) and RMRNN (same + reverse-
              martingale backward-coherence penalty -- the paper's method)

Scored by next-frame RMSE and MAE over the grid on a held-out test split.
This isolates one question: is the RM-regularised recurrent model's *forecast
skill* competitive with off-the-shelf spatial deep models?  (It is not a
warning experiment.)  All data are synthetic; no real gridded product is used.

Usage: python3 run_gridded_bakeoff.py [--epochs 40] [--seed 0]
Writes experiments/gridded_bakeoff_results.json
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
EXP = os.path.join(os.path.dirname(os.path.dirname(HERE)), "..", "experiments")
EXP = os.path.normpath(os.path.join(BASE, "experiments")) if False else \
      os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments"))

GRID = 12          # HxW
T = 720            # frames (days)
L = 6              # input frames
HID = 16


def simulate_grid(seed):
    rng = np.random.RandomState(seed)
    # spatial smoothing kernel for correlated fields
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    t = np.arange(T)
    season = 1.0 + 0.5 * np.cos(2 * np.pi * (t / 365.0 - 0.55))   # wet/dry cycle
    # AR(1) latent field, spatially smoothed each step
    from scipy.ndimage import gaussian_filter
    z = np.zeros((T, GRID, GRID))
    for k in range(1, T):
        z[k] = 0.9 * z[k - 1] + np.sqrt(1 - 0.81) * gaussian_filter(rng.randn(GRID, GRID), 1.2)
    # orographic gradient (west wetter), gamma intensity gated by wet prob
    oro = 1.0 + 0.4 * (1 - xx / (GRID - 1))
    p_wet = np.clip(0.25 * season[:, None, None] * oro[None] * (1 + 0.3 * z), 0.01, 0.95)
    wet = (rng.rand(T, GRID, GRID) < p_wet).astype(float)
    scale = np.clip(4.0 * season[:, None, None] * oro[None] * np.exp(0.3 * z), 0.1, None)
    rain = wet * rng.gamma(0.9, scale)
    return np.log1p(rain).astype(np.float32), season    # work in log1p space


# ---- models -----------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, ch, hid):
        super().__init__(); self.hid = hid
        self.conv = nn.Conv2d(ch + hid, 4 * hid, 3, padding=1)
    def forward(self, x, h, c):
        z = self.conv(torch.cat([x, h], 1))
        i, f, o, g = z.chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, hid=HID):
        super().__init__(); self.cell = ConvLSTMCell(1, hid); self.head = nn.Conv2d(hid, 1, 1); self.hid = hid
    def forward(self, x):                       # x: (B,L,1,H,W)
        B = x.size(0); h = x.new_zeros(B, self.hid, GRID, GRID); c = torch.zeros_like(h)
        for t in range(x.size(1)):
            h, c = self.cell(x[:, t], h, c)
        return self.head(h)                     # (B,1,H,W)

class STLSTMCell(nn.Module):
    """PredRNN spatiotemporal-LSTM core: standard memory C plus a spatial
    memory M flowing through the layer."""
    def __init__(self, ch, hid):
        super().__init__(); self.hid = hid
        self.cx = nn.Conv2d(ch, 7 * hid, 3, padding=1)
        self.ch = nn.Conv2d(hid, 4 * hid, 3, padding=1)
        self.cm = nn.Conv2d(hid, 3 * hid, 3, padding=1)
        self.o1 = nn.Conv2d(2 * hid, hid, 1)
    def forward(self, x, h, c, m):
        gx = self.cx(x); gh = self.ch(h); gm = self.cm(m)
        ix, fx, gx_, ix2, fx2, gx2, ox = gx.chunk(7, 1)
        ih, fh, gh_, oh = gh.chunk(4, 1)
        im, fm, gm_ = gm.chunk(3, 1)
        i = torch.sigmoid(ix + ih); f = torch.sigmoid(fx + fh); g = torch.tanh(gx_ + gh_)
        c = f * c + i * g
        i2 = torch.sigmoid(ix2 + im); f2 = torch.sigmoid(fx2 + fm); g2 = torch.tanh(gx2 + gm_)
        m = f2 * m + i2 * g2
        o = torch.sigmoid(ox + oh + self.o1(torch.cat([c, m], 1)))
        h = o * torch.tanh(self.o1(torch.cat([c, m], 1)))
        return h, c, m

class PredRNN(nn.Module):
    def __init__(self, hid=HID):
        super().__init__(); self.cell = STLSTMCell(1, hid); self.head = nn.Conv2d(hid, 1, 1); self.hid = hid
    def forward(self, x):
        B = x.size(0); h = x.new_zeros(B, self.hid, GRID, GRID); c = torch.zeros_like(h); m = torch.zeros_like(h)
        for t in range(x.size(1)):
            h, c, m = self.cell(x[:, t], h, c, m)
        return self.head(h)

class UNet(nn.Module):
    def __init__(self, cin=L, base=16):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(cin, base, 3, padding=1), nn.ReLU(), nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.p = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(nn.Conv2d(base, 2*base, 3, padding=1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.d1 = nn.Sequential(nn.Conv2d(3*base, base, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):                       # x: (B,L,H,W)
        e1 = self.e1(x); e2 = self.e2(self.p(e1))
        d = self.up(e2)
        # pad to match if odd
        if d.shape[-1] != e1.shape[-1]:
            d = nn.functional.interpolate(d, size=e1.shape[-2:])
        d = self.d1(torch.cat([d, e1], 1))
        return self.out(d)

class CellGRU(nn.Module):
    """Per-cell GRU (shared weights across cells); optional RM backward map."""
    def __init__(self, hid=HID, rm=False):
        super().__init__(); self.gru = nn.GRU(1, hid, batch_first=True); self.fc = nn.Linear(hid, 1)
        self.rm = rm
        if rm:
            self.g = nn.Sequential(nn.Linear(hid, 2*hid), nn.Tanh(), nn.Linear(2*hid, hid))
    def forward(self, x):                        # x: (Ncell, L, 1)
        H, _ = self.gru(x); return self.fc(H[:, -1, :]), H
    def rm_penalty(self, H):
        # backward coherence: g(h_{t+1}) ~ h_t
        if not self.rm: return torch.tensor(0.0)
        hnext = H[:, 1:, :].reshape(-1, H.size(-1)); hprev = H[:, :-1, :].reshape(-1, H.size(-1))
        return ((self.g(hnext) - hprev) ** 2).mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0); args = ap.parse_args()
    torch.manual_seed(args.seed)
    field, season = simulate_grid(args.seed)               # (T,H,W) log1p
    # windows of L frames -> next frame
    Xs = np.stack([field[i:i+L] for i in range(T-L)])      # (N,L,H,W)
    ys = field[L:]                                          # (N,H,W)
    n = len(Xs); ntr = int(0.8*n)
    Xtr, ytr, Xte, yte = Xs[:ntr], ys[:ntr], Xs[ntr:], ys[ntr:]
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)[:, None]      # (N,1,H,W)
    Xte_t = torch.tensor(Xte); yte_t = torch.tensor(yte)[:, None]

    def rmse_mae(pred):
        e = (pred - yte).reshape(len(yte), -1)
        return float(np.sqrt((e**2).mean())), float(np.abs(e).mean())

    res = {}
    # baselines
    res["Persistence"] = rmse_mae(Xte[:, -1])
    clim = field[:ntr+L].mean(0)                  # mean frame from training period
    res["Climatology"] = rmse_mae(np.broadcast_to(clim, yte.shape))

    def train_seq(model, seqfmt):
        opt = torch.optim.Adam(model.parameters(), 3e-3); lossf = nn.MSELoss()
        bs = 32; rng = np.random.RandomState(args.seed)
        for ep in range(args.epochs):
            idx = rng.choice(ntr, bs, replace=False)
            xb = Xtr_t[idx]; yb = ytr_t[idx]
            if seqfmt == "conv": xb = xb[:, :, None]          # (B,L,1,H,W)
            elif seqfmt == "img": pass                         # (B,L,H,W)
            opt.zero_grad(); pred = model(xb); loss = lossf(pred, yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xb = Xte_t[:, :, None] if seqfmt == "conv" else Xte_t
            pred = model(xb).squeeze(1).numpy()
        return rmse_mae(pred)

    t0 = time.time()
    torch.manual_seed(args.seed); res["ConvLSTM"] = train_seq(ConvLSTM(), "conv")
    torch.manual_seed(args.seed); res["PredRNN"] = train_seq(PredRNN(), "conv")
    torch.manual_seed(args.seed); res["U-Net"] = train_seq(UNet(), "img")

    # per-cell GRU / RMRNN: reshape (N,L,H,W)->(N*HW, L,1)
    def train_cell(rm):
        torch.manual_seed(args.seed); model = CellGRU(rm=rm)
        opt = torch.optim.Adam(model.parameters(), 3e-3); lossf = nn.MSELoss()
        # build per-cell training set: each (window, cell) is a sample
        Xc = torch.tensor(Xtr.transpose(0,2,3,1).reshape(-1, L, 1))   # (N*HW, L,1)
        yc = torch.tensor(ytr.reshape(-1, 1))
        rng = np.random.RandomState(args.seed); N = Xc.size(0); bs = 512
        for ep in range(args.epochs):
            idx = rng.choice(N, bs, replace=False)
            opt.zero_grad(); pred, H = model(Xc[idx]); loss = lossf(pred, yc[idx])
            if rm: loss = loss + 0.1 * model.rm_penalty(H)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            Xcte = torch.tensor(Xte.transpose(0,2,3,1).reshape(-1, L, 1))
            pred,_ = model(Xcte); pred = pred.numpy().reshape(len(yte), GRID, GRID)
        return rmse_mae(pred)
    res["GRU (cell)"] = train_cell(False)
    res["RMRNN (cell)"] = train_cell(True)

    print(f"Gridded synthetic bake-off ({GRID}x{GRID}, T={T}, L={L}, epochs={args.epochs}, {time.time()-t0:.0f}s)")
    print(f"{'Model':14s} {'RMSE':>8s} {'MAE':>8s}")
    for k, (r, m) in res.items():
        print(f"{k:14s} {r:8.4f} {m:8.4f}")
    out = {"config": {"grid": GRID, "T": T, "L": L, "epochs": args.epochs, "seed": args.seed,
                      "note": "synthetic gridded next-frame nowcasting; log1p space; RMSE/MAE over grid"},
           "results": {k: {"rmse": v[0], "mae": v[1]} for k, v in res.items()}}
    os.makedirs(EXP, exist_ok=True)
    json.dump(out, open(os.path.join(EXP, "gridded_bakeoff_results.json"), "w"), indent=2)
    print("saved experiments/gridded_bakeoff_results.json")


if __name__ == "__main__":
    main()
