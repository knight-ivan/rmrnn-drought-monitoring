from typing import Optional
"""
Baseline models: EMA and simple linear RNN (no RM regularisation)
==================================================================
These are the "vanilla" comparators for the RMRNN simulation study.
Implemented in pure NumPy to match the main model.
"""

import numpy as np
import copy
from model import AdamState, tanh, dtanh


# ---------------------------------------------------------------------------
# Exponential Moving Average (EMA)  — the classic non-RM baseline
# ---------------------------------------------------------------------------

class EMABaseline:
    """
    h_t = alpha * h_{t-1} + (1-alpha) * x_t,   alpha fixed.

    The EMA with fixed alpha is NOT a reverse martingale (Remark rem:ema in the
    paper): E[h_t | h_{t+1}] != h_{t+1} in general.  Included as a sanity check.
    """

    def __init__(self, alpha: float = 0.9):
        assert 0 < alpha < 1
        self.alpha = alpha

    def forward_batch(self, x_batch: np.ndarray) -> np.ndarray:
        """
        x_batch : (B, T, d)
        Returns h_batch : (B, T, d)
        """
        B, T, d = x_batch.shape
        h = np.zeros((B, d))
        h_list = []
        for t in range(T):
            h = self.alpha * h + (1 - self.alpha) * x_batch[:, t, :]
            h_list.append(h.copy())
        return np.stack(h_list, axis=1)

    def predict(self, x_batch: np.ndarray) -> np.ndarray:
        return self.forward_batch(x_batch)[:, -1, :]   # (B, d)


# ---------------------------------------------------------------------------
# Vanilla linear RNN (no RM regularisation)
# ---------------------------------------------------------------------------

class LinearRNNParams:
    def __init__(self, d: int, p: int, output_dim: int,
                 rho_init: float = 0.5, seed: int = 1):
        rng = np.random.RandomState(seed)
        Wh_raw = rng.randn(p, p) * 0.1
        sv = np.linalg.svd(Wh_raw, compute_uv=False)
        self.Wh = Wh_raw * (rho_init / sv[0])
        self.Wx = rng.randn(p, d) * 0.1
        self.bh = np.zeros(p)
        self.Wr = rng.randn(output_dim, p) * 0.1
        self.br = np.zeros(output_dim)


class VanillaRNN:
    """
    Simple tanh-RNN trained with task loss only (no RM regularisation).
    Used to show that RM training improves hidden-state convergence properties.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 rho_init: float = 0.5, seed: int = 1):
        self.params = LinearRNNParams(input_dim, hidden_dim, output_dim,
                                      rho_init=rho_init, seed=seed)
        self.adam = AdamState()
        self.hidden_dim = hidden_dim

    def forward_batch(self, x_batch: np.ndarray) -> np.ndarray:
        B, T, d = x_batch.shape
        p = self.params.Wh.shape[0]
        h = np.zeros((B, p))
        h_list = []
        for t in range(T):
            pre = h @ self.params.Wh.T + x_batch[:, t, :] @ self.params.Wx.T + self.params.bh
            h = tanh(pre)
            h_list.append(h.copy())
        return np.stack(h_list, axis=1)    # (B, T, p)

    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray,
                   lr: float = 1e-3) -> dict:
        params = self.params
        B, T, d = x_batch.shape

        h_batch = self.forward_batch(x_batch)
        h_T = h_batch[:, -1, :]
        y_pred = h_T @ params.Wr.T + params.br
        res = y_pred - y_batch
        loss = 0.5 * float(np.mean(np.sum(res**2, axis=-1)))

        dy = res / B
        dWr = dy.T @ h_T
        dbr = dy.sum(axis=0)

        all_grads = {
            "Wh": np.zeros_like(params.Wh),
            "Wx": np.zeros_like(params.Wx),
            "bh": np.zeros_like(params.bh),
            "Wr": dWr, "br": dbr,
        }
        self.params = self.adam.step(params, all_grads, lr=lr)
        return {"loss_task": loss, "loss_rm": 0.0, "loss_total": loss}

    def convergence_profile(self, x_batch: np.ndarray) -> dict:
        h_batch = self.forward_batch(x_batch)
        B, T, p = h_batch.shape
        h_final = h_batch[:, -1, :]
        diff = h_batch - h_final[:, np.newaxis, :]
        error_vs_t = np.mean(np.linalg.norm(diff, axis=-1), axis=0)
        return {"error_vs_t": error_vs_t,
                "defect_norms": np.zeros(T - 1),
                "Q_cumulative": np.zeros(T - 1)}
