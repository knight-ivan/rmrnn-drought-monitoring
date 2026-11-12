# /mnt/data/rmrnn_pilot/rmrnn_pilot_study.py
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(7)
np.random.seed(7)
random.seed(7)
torch.set_num_threads(2)


class SyntheticPrecipDataset(Dataset):
    def __init__(self, n_sequences=400, seq_len=24, anomaly=False, cp_range=(10, 18), include_cp=False):
        xs, ys, cps = [], [], []
        for _ in range(n_sequences):
            x, y, cp = self.generate_one(seq_len, anomaly, cp_range)
            xs.append(x.astype(np.float32))
            ys.append(y.astype(np.float32))
            cps.append(cp)
        self.X = torch.tensor(np.stack(xs))
        self.y = torch.tensor(np.stack(ys))
        self.cp = np.array(cps)
        self.include_cp = include_cp

    @staticmethod
    def generate_one(seq_len, anomaly=False, cp_range=(10, 18)):
        T = seq_len + 1
        t = np.arange(T)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.8, 1.3)
        drift = np.random.uniform(-0.02, 0.02)

        temp = 0.9 * amp * np.sin(2 * np.pi * t / 16 + phase) + drift * t + 0.12 * np.random.randn(T)
        humidity = 0.7 * np.cos(2 * np.pi * t / 17 + phase / 2) - 0.3 * temp + 0.12 * np.random.randn(T)

        omega = np.zeros(T)
        for i in range(1, T):
            omega[i] = 0.78 * omega[i - 1] + 0.16 * np.random.randn() + 0.05 * np.sin(2 * np.pi * i / 10 + phase)

        cp = -1
        if anomaly:
            cp = np.random.randint(cp_range[0], cp_range[1])
            end = min(T, cp + np.random.randint(4, 8))
            omega[cp:end] += np.linspace(0.8, 1.6, end - cp) + 0.08 * np.random.randn(end - cp)
            humidity[cp:end] += np.linspace(0.3, 0.8, end - cp)
            temp[cp:end] -= np.linspace(0.1, 0.4, end - cp)
            if end < T:
                omega[end:] += 0.4
                humidity[end:] += 0.2

        latent = 0.35 * humidity - 0.22 * temp + 0.85 * omega + 0.15 * np.sin(2 * np.pi * t / 8 + phase)
        hetero = 0.10 + 0.10 * np.maximum(0, omega) + 0.06 * np.maximum(0, humidity)
        precip = np.maximum(0, latent + hetero * np.random.randn(T))

        if anomaly and cp >= 0:
            precip[cp:min(T, cp + 4)] += np.linspace(0.8, 1.8, min(T, cp + 4) - cp)
            if cp + 4 < T:
                precip[cp + 4:min(T, cp + 7)] += 1.2

        X = np.column_stack([precip[:-1], temp[:-1], humidity[:-1], omega[:-1]])
        y = precip[1:, None]
        return X, y, cp

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.include_cp:
            return self.X[idx], self.y[idx], int(self.cp[idx])
        return self.X[idx], self.y[idx]


class SequenceStandardizer:
    def fit(self, X, y):
        self.x_mean = X.mean(dim=(0, 1), keepdim=True)
        self.x_std = X.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        self.y_mean = y.mean(dim=(0, 1), keepdim=True)
        self.y_std = y.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        return self

    def transform_X(self, X):
        return (X - self.x_mean) / self.x_std

    def transform_y(self, y):
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y):
        return y * self.y_std + self.y_mean


class GRUForecast(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, lambda_reg=0.0):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.lambda_reg = lambda_reg

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.fc(h), h

    def loss_components(self, x, y):
        yhat, h = self(x)
        pred = nn.functional.mse_loss(yhat, y)
        reg = nn.functional.mse_loss(h[:, :-1, :], h[:, 1:, :]) if self.lambda_reg else torch.tensor(0.0, device=x.device)
        return pred + self.lambda_reg * reg, pred, reg


def train_model(model, train_loader, val_loader, epochs=6, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    hist = []

    for epoch in range(epochs):
        model.train()
        tr_pred = tr_reg = n = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            total, pred, reg = model.loss_components(xb, yb)
            total.backward()
            opt.step()
            bs = xb.shape[0]
            tr_pred += pred.item() * bs
            tr_reg += reg.item() * bs
            n += bs

        model.eval()
        va_pred = vn = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                _, pred, _ = model.loss_components(xb, yb)
                va_pred += pred.item() * xb.shape[0]
                vn += xb.shape[0]
        va_pred /= vn
        hist.append((epoch + 1, tr_pred / n, tr_reg / n, va_pred))

        if va_pred < best_val:
            best_val = va_pred
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, hist


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(mse(a, b)))


def get_pred_hidden(model, ds):
    model.eval()
    with torch.no_grad():
        yhat, h = model(ds.X)
    return yhat, h


def forecast_metrics(model, ds, scaler):
    yhat, _ = get_pred_hidden(model, ds)
    pred = scaler.inverse_y(yhat).numpy().reshape(-1)
    true = scaler.inverse_y(ds.y).numpy().reshape(-1)
    return {"rmse": rmse(true, pred), "mae": mae(true, pred)}


def residual_sequences(model, ds):
    _, h = get_pred_hidden(model, ds)
    return torch.norm(h[:, 1:, :] - h[:, :-1, :], dim=-1).numpy()


def point_threshold_from_validation(residuals, sigma=3.0):
    flat = residuals.reshape(-1)
    return float(flat.mean() + sigma * flat.std())


def first_alert_time_point(residual_seq, threshold):
    hits = np.where(residual_seq > threshold)[0]
    return int(hits[0]) if len(hits) else None


def sr_scores_from_residuals(residual_seq, mu0, sigma0):
    sigma0 = max(float(sigma0), 1e-6)
    z = np.clip((residual_seq - mu0) / sigma0, -3.0, 6.0)
    lamb = np.exp(np.maximum(0.0, z))
    R = 0.0
    out = []
    for lam in lamb:
        R = (1.0 + R) * float(lam)
        out.append(R)
    return np.array(out)


def calibrate_sr_threshold(val_residuals, target_arl0=100):
    flat = val_residuals.reshape(-1)
    mu0, sigma0 = float(flat.mean()), float(flat.std() + 1e-6)
    all_scores = []
    for seq in val_residuals:
        all_scores.extend(sr_scores_from_residuals(seq, mu0, sigma0))
    percentile = (1.0 - 1.0 / target_arl0) * 100.0
    B = float(np.percentile(all_scores, percentile))
    return B, mu0, sigma0


def evaluate_detection(model, val_ds, normal_ds, anom_ds, method="point"):
    val_res = residual_sequences(model, val_ds)
    normal_res = residual_sequences(model, normal_ds)
    anom_res = residual_sequences(model, anom_ds)

    if method == "point":
        threshold = point_threshold_from_validation(val_res, sigma=3.0)

        def detector(seq):
            return first_alert_time_point(seq, threshold)

        descriptor = {"threshold": threshold}
    else:
        B, mu0, sigma0 = calibrate_sr_threshold(val_res, target_arl0=100)

        def detector(seq):
            scores = sr_scores_from_residuals(seq, mu0, sigma0)
            hits = np.where(scores >= B)[0]
            return int(hits[0]) if len(hits) else None

        descriptor = {"threshold_B": B, "mu0": mu0, "sigma0": sigma0}

    false_alarms = 0
    for seq in normal_res:
        false_alarms += int(detector(seq) is not None)

    detects = 0
    delays = []
    early_hits = 0
    for seq, cp in zip(anom_res, anom_ds.cp):
        t = detector(seq)
        if t is not None:
            detects += 1
            if t < cp:
                early_hits += 1
            else:
                delays.append(t - cp)

    return {
        **descriptor,
        "false_alarm_rate_normal": false_alarms / len(normal_res),
        "detection_rate_anomaly": detects / len(anom_res),
        "early_warning_rate": early_hits / len(anom_res),
        "mean_delay_after_cp": float(np.mean(delays)) if delays else None,
        "mean_residual_normal": float(normal_res.mean()),
        "mean_residual_anomaly": float(anom_res.mean()),
    }


def main():
    outdir = Path("/mnt/data/rmrnn_pilot")
    outdir.mkdir(parents=True, exist_ok=True)

    train_ds = SyntheticPrecipDataset(n_sequences=360, anomaly=False)
    val_ds = SyntheticPrecipDataset(n_sequences=100, anomaly=False)
    test_normal_ds = SyntheticPrecipDataset(n_sequences=120, anomaly=False, include_cp=True)
    test_anom_ds = SyntheticPrecipDataset(n_sequences=120, anomaly=True, include_cp=True)

    scaler = SequenceStandardizer().fit(train_ds.X, train_ds.y)
    for ds in [train_ds, val_ds, test_normal_ds, test_anom_ds]:
        ds.X = scaler.transform_X(ds.X)
        ds.y = scaler.transform_y(ds.y)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    gru, gru_hist = train_model(GRUForecast(lambda_reg=0.0), train_loader, val_loader)
    rmrnn, rmrnn_hist = train_model(GRUForecast(lambda_reg=0.15), train_loader, val_loader)

    results = {
        "forecast": {
            "GRU": {
                "normal": forecast_metrics(gru, test_normal_ds, scaler),
                "anomaly": forecast_metrics(gru, test_anom_ds, scaler),
            },
            "RMRNN": {
                "normal": forecast_metrics(rmrnn, test_normal_ds, scaler),
                "anomaly": forecast_metrics(rmrnn, test_anom_ds, scaler),
            },
        },
        "detection": {
            "GRU_point": evaluate_detection(gru, val_ds, test_normal_ds, test_anom_ds, method="point"),
            "RMRNN_point": evaluate_detection(rmrnn, val_ds, test_normal_ds, test_anom_ds, method="point"),
            "GRU_sr": evaluate_detection(gru, val_ds, test_normal_ds, test_anom_ds, method="sr"),
            "RMRNN_sr": evaluate_detection(rmrnn, val_ds, test_normal_ds, test_anom_ds, method="sr"),
        },
        "training_history": {"GRU": gru_hist, "RMRNN": rmrnn_hist},
    }

    (outdir / "results.json").write_text(json.dumps(results, indent=2))

    plt.figure(figsize=(7, 4))
    plt.plot([e[0] for e in gru_hist], [e[3] for e in gru_hist], label="GRU val pred loss")
    plt.plot([e[0] for e in rmrnn_hist], [e[3] for e in rmrnn_hist], label="RMRNN val pred loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE (scaled)")
    plt.title("Pilot training comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "training_curve.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
