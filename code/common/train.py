"""
Training loop and evaluation utilities for RMRNN
=================================================
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Callable

from model import RMRNN
from data import DataBatch, get_batch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: RMRNN,
    scenario: str,
    *,
    n_steps: int = 2000,
    batch: int = 64,
    T: int = 50,
    input_dim: int = 1,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 200,
    scenario_kwargs: dict | None = None,
) -> list[dict]:
    """
    Train an RMRNN model on a given synthetic scenario.

    Returns a list of log dicts (one per logged step) with keys:
        step, loss_task, loss_rm, loss_total, Q_empirical, rho_upper
    """
    if scenario_kwargs is None:
        scenario_kwargs = {}

    model = model.to(device)
    opt = Adam(model.parameters(), lr=lr)
    log = []

    for step in range(1, n_steps + 1):
        data: DataBatch = get_batch(
            scenario, batch=batch, T=T, input_dim=input_dim,
            device=device, **scenario_kwargs
        )

        opt.zero_grad()
        loss, info = model.total_loss(data.x_seq, data.y_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            Q = model.quasi_martingale_total(data.x_seq)
            rho = model.cell.contraction_rate()
            record = {
                "step":       step,
                "loss_task":  info["loss_task"],
                "loss_rm":    info["loss_rm"],
                "loss_total": info["loss_total"],
                "Q_empirical": Q,
                "rho_upper":  rho,
            }
            log.append(record)
            if log_every > 0:
                print(
                    f"step {step:5d} | "
                    f"L_task={info['loss_task']:.4f} "
                    f"L_rm={info['loss_rm']:.4f} "
                    f"Q={Q:.4f} "
                    f"rho<={rho:.3f}"
                )

    return log


# ---------------------------------------------------------------------------
# Evaluation: hidden-state convergence
# ---------------------------------------------------------------------------

@torch.no_grad()
def convergence_profile(
    model: RMRNN,
    data: DataBatch,
    device: str = "cpu",
) -> dict:
    """
    Evaluate hidden-state convergence on a single batch.

    Returns a dict with:
        h_seq         : (batch, T, hidden_dim)  all hidden states
        h_final       : (batch, hidden_dim)     h_T (proxy for h_infty)
        error_vs_t    : (T,)  mean ||h_t - h_T|| over batch (convergence curve)
        defect_norms  : (T-1,) mean ||delta_t|| over batch
        Q_cumulative  : (T-1,) cumulative defect sum from t to T-1
    """
    model.eval()
    x_seq = data.x_seq.to(device)

    h_seq, y_pred, loss_rm = model(x_seq)

    h_final = h_seq[:, -1, :]                            # proxy for h_infty
    diff = h_seq - h_final.unsqueeze(1)                  # (batch, T, p)
    error_vs_t = diff.norm(dim=-1).mean(dim=0)           # (T,)

    defects = model.defect_sequence(x_seq)               # (batch, T-1, p)
    defect_norms = defects.norm(dim=-1).mean(dim=0)      # (T-1,)

    # cumulative Q from position t to end
    Q_cumulative = defect_norms.flip(0).cumsum(0).flip(0)  # (T-1,)

    model.train()
    return {
        "h_seq":        h_seq.cpu(),
        "h_final":      h_final.cpu(),
        "error_vs_t":   error_vs_t.cpu(),
        "defect_norms": defect_norms.cpu(),
        "Q_cumulative": Q_cumulative.cpu(),
        "y_pred":       y_pred.cpu(),
        "y_target":     data.y_target.cpu(),
    }


# ---------------------------------------------------------------------------
# Baseline: vanilla LSTM (no RM regularisation)
# ---------------------------------------------------------------------------

class VanillaLSTM(nn.Module):
    """Single-layer LSTM baseline for comparison."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_seq, _ = self.lstm(x_seq)           # (batch, T, hidden_dim)
        y_pred = self.readout(h_seq[:, -1, :])
        return h_seq, y_pred

    def total_loss(self, x_seq, y_target):
        _, y_pred = self.forward(x_seq)
        loss = nn.functional.mse_loss(y_pred, y_target)
        info = {"loss_task": loss.item(), "loss_rm": 0.0, "loss_total": loss.item()}
        return loss, info


class VanillaGRU(nn.Module):
    """Single-layer GRU baseline for comparison."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_seq, _ = self.gru(x_seq)
        y_pred = self.readout(h_seq[:, -1, :])
        return h_seq, y_pred

    def total_loss(self, x_seq, y_target):
        _, y_pred = self.forward(x_seq)
        loss = nn.functional.mse_loss(y_pred, y_target)
        info = {"loss_task": loss.item(), "loss_rm": 0.0, "loss_total": loss.item()}
        return loss, info
