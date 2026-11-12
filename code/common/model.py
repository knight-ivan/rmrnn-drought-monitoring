from typing import Optional
"""
RMRNN: Reverse-Martingale Guided Recurrent Neural Network (NumPy implementation)
=================================================================================
Implements a small contractive RNN + backward (reverse-conditional) MLP in pure
NumPy.  Gradients are computed analytically via the chain rule.

Classes:
    RMRNNParams   : named container for all learnable parameters
    rmrnn_forward : run the forward RNN, collect hidden states
    backward_fwd  : evaluate the backward (reverse-conditional) MLP
    rm_loss_grad  : compute RM loss and its gradient w.r.t. backward MLP params
    train_step    : one Adam step on the total loss
    RMRNN         : high-level stateful model class
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

def tanh(x):    return np.tanh(x)
def dtanh(x):   return 1.0 - np.tanh(x) ** 2   # derivative of tanh at x


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class Params:
    """All learnable parameters for RMRNN."""
    # --- forward RNN cell ---
    Wh: np.ndarray    # (p, p)  hidden-to-hidden
    Wx: np.ndarray    # (p, d)  input-to-hidden
    bh: np.ndarray    # (p,)
    # --- backward MLP (2-layer): approx E[h_t | h_{t+1}] ---
    W1: np.ndarray    # (m, p)  first layer
    b1: np.ndarray    # (m,)
    W2: np.ndarray    # (p, m)  second layer (residual: output = h + W2*tanh(W1*h+b1)+b2)
    b2: np.ndarray    # (p,)
    # --- readout ---
    Wr: np.ndarray    # (output_dim, p)
    br: np.ndarray    # (output_dim,)


def init_params(d: int, p: int, m: int, output_dim: int,
                rho_init: float = 0.5, seed: int = 0) -> Params:
    """
    Initialise parameters.

    d          : input dimension
    p          : hidden dimension
    m          : backward MLP hidden dimension (= 2*p typically)
    rho_init   : target initial spectral norm for Wh (contraction rate)
    """
    rng = np.random.RandomState(seed)

    # Forward RNN: Wh initialised with controlled spectral norm
    Wh_raw = rng.randn(p, p) * 0.1
    sv = np.linalg.svd(Wh_raw, compute_uv=False)
    Wh = Wh_raw * (rho_init / sv[0])   # scale so sigma_max = rho_init

    Wx = rng.randn(p, d) * 0.1
    bh = np.zeros(p)

    # Backward MLP: initialise near identity (W2 = 0 → g(h) = h)
    W1 = rng.randn(m, p) * 0.01
    b1 = np.zeros(m)
    W2 = np.zeros((p, m))              # residual: start as g(h) = h
    b2 = np.zeros(p)

    # Readout
    Wr = rng.randn(output_dim, p) * 0.1
    br = np.zeros(output_dim)

    return Params(Wh=Wh, Wx=Wx, bh=bh, W1=W1, b1=b1, W2=W2, b2=b2, Wr=Wr, br=br)


# ---------------------------------------------------------------------------
# Forward RNN
# ---------------------------------------------------------------------------

def rmrnn_forward(params: Params, x_seq: np.ndarray,
                  h0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Run forward RNN over a sequence.

    Args:
        x_seq : (T, d)   one sequence
        h0    : (p,)     initial hidden state (zeros if None)
    Returns:
        h_seq : (T, p)   hidden states h_1 .. h_T
    """
    T, d = x_seq.shape
    p = params.Wh.shape[0]
    h = np.zeros(p) if h0 is None else h0.copy()
    h_seq = np.empty((T, p))
    for t in range(T):
        pre = params.Wh @ h + params.Wx @ x_seq[t] + params.bh
        h = tanh(pre)
        h_seq[t] = h
    return h_seq


def rmrnn_forward_batch(params: Params, x_batch: np.ndarray,
                        h0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Batched forward RNN.

    Args:
        x_batch : (B, T, d)
    Returns:
        h_batch : (B, T, p)
    """
    B, T, d = x_batch.shape
    p = params.Wh.shape[0]
    h = np.zeros((B, p)) if h0 is None else h0.copy()
    h_all = np.empty((B, T, p))
    for t in range(T):
        pre = h @ params.Wh.T + x_batch[:, t, :] @ params.Wx.T + params.bh
        h = tanh(pre)
        h_all[:, t, :] = h
    return h_all


# ---------------------------------------------------------------------------
# Backward MLP (residual):  g(h) = h + W2 tanh(W1 h + b1) + b2
# ---------------------------------------------------------------------------

def backward_fwd(params: Params, h: np.ndarray) -> tuple:
    """
    Evaluate backward MLP at h.

    Args:
        h  : (B, p)  or (p,)
    Returns:
        g  : same shape as h   — approximate E[h_t | h_{t+1}]
        z1 : pre-activation of hidden layer  (B, m)
        a1 : post-activation of hidden layer (B, m)
    """
    squeeze = h.ndim == 1
    if squeeze:
        h = h[np.newaxis, :]
    z1 = h @ params.W1.T + params.b1      # (B, m)
    a1 = tanh(z1)                          # (B, m)
    g  = h + a1 @ params.W2.T + params.b2 # (B, p)  residual
    if squeeze:
        g, z1, a1 = g[0], z1[0], a1[0]
    return g, z1, a1


# ---------------------------------------------------------------------------
# RM loss and its gradient w.r.t. backward MLP params
# ---------------------------------------------------------------------------

def rm_loss_and_grad(params: Params, h_seq: np.ndarray) -> tuple:
    """
    Compute RM loss and gradient for a single sequence.

        L_RM = (1/(T-1)) sum_{t=0}^{T-2}  || g(h_{t+1}) - h_t ||^2

    where h_t = h_seq[t].

    Args:
        h_seq : (T, p)
    Returns:
        loss  : scalar
        grads : dict with keys 'W1','b1','W2','b2'
    """
    T, p = h_seq.shape
    if T < 2:
        zero = np.zeros(1)[0]
        return zero, {k: np.zeros_like(getattr(params, k))
                      for k in ('W1','b1','W2','b2')}

    h_t    = h_seq[:-1]   # (T-1, p)  targets
    h_next = h_seq[1:]    # (T-1, p)  inputs to backward MLP

    g, z1, a1 = backward_fwd(params, h_next)   # (T-1, p), (T-1, m), (T-1, m)
    residual = g - h_t                          # (T-1, p)  = g - target
    loss = 0.5 * np.mean(np.sum(residual**2, axis=-1))

    # gradient of loss w.r.t. g  (upstream)
    dg = residual / (T - 1)                    # (T-1, p)

    # backward through residual:  g = h_next + a1 @ W2.T + b2
    dW2 = dg.T @ a1                            # (p, m)
    db2 = dg.sum(axis=0)                       # (p,)

    # backward through tanh layer
    da1 = dg @ params.W2                       # (T-1, m)
    dz1 = da1 * dtanh(z1)                      # (T-1, m)

    dW1 = dz1.T @ h_next                       # (m, p)
    db1 = dz1.sum(axis=0)                      # (m,)

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    return float(loss), grads


# ---------------------------------------------------------------------------
# Task loss and gradient w.r.t. readout params
# ---------------------------------------------------------------------------

def task_loss_and_grad(params: Params, h_T: np.ndarray,
                       y: np.ndarray) -> tuple:
    """
    MSE readout loss:  L_task = (1/B) sum_b || Wr h_{T,b} + br - y_b ||^2

    Args:
        h_T : (B, p)
        y   : (B, output_dim)
    Returns:
        loss : scalar
        grads: dict with keys 'Wr','br'
    """
    B = h_T.shape[0]
    y_pred = h_T @ params.Wr.T + params.br    # (B, output_dim)
    res = y_pred - y                           # (B, output_dim)
    loss = 0.5 * np.mean(np.sum(res**2, axis=-1))

    dy = res / B                               # (B, output_dim)
    dWr = dy.T @ h_T                           # (output_dim, p)
    dbr = dy.sum(axis=0)                       # (output_dim,)

    return float(loss), {"Wr": dWr, "br": dbr}


# ---------------------------------------------------------------------------
# Adam optimiser state
# ---------------------------------------------------------------------------

@dataclass
class AdamState:
    m: dict = field(default_factory=dict)   # first moment
    v: dict = field(default_factory=dict)   # second moment
    t: int = 0

    def step(self, params: Params, grads: dict,
             lr: float = 1e-3, beta1: float = 0.9,
             beta2: float = 0.999, eps: float = 1e-8) -> Params:
        self.t += 1
        t = self.t
        updates = {}
        for key, g in grads.items():
            if key not in self.m:
                self.m[key] = np.zeros_like(g)
                self.v[key] = np.zeros_like(g)
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * g**2
            m_hat = self.m[key] / (1 - beta1**t)
            v_hat = self.v[key] / (1 - beta2**t)
            updates[key] = getattr(params, key) - lr * m_hat / (np.sqrt(v_hat) + eps)
        import copy
        new_params = copy.copy(params)
        for key, val in updates.items():
            object.__setattr__(new_params, key, val)
        return new_params


# ---------------------------------------------------------------------------
# Model-agnostic path-stability diagnostic
# ---------------------------------------------------------------------------

def hidden_increment_total(h_batch: np.ndarray) -> float:
    """Quasi-martingale proxy computable from *any* hidden trajectory:

        Q_path = sum_{t} mean_b || h_{t+1} - h_t ||.

    Unlike RMRNN.quasi_martingale_total (which evaluates the learned backward
    map g), this uses only the raw forward increments of the hidden path, so it
    is directly comparable across architectures (RMRNN vs a vanilla RNN). A
    hidden path that has stabilised into a quasi-martingale has small
    per-step increments and hence a small Q_path.

    Args:
        h_batch : (B, T, p) hidden states.
    Returns:
        scalar Q_path.
    """
    incr = h_batch[:, 1:, :] - h_batch[:, :-1, :]        # (B, T-1, p)
    return float(np.mean(np.linalg.norm(incr, axis=-1), axis=0).sum())


# ---------------------------------------------------------------------------
# High-level RMRNN class
# ---------------------------------------------------------------------------

class RMRNN:
    """
    Stateful RMRNN model.

    Attributes:
        params     : current parameters
        adam       : Adam optimiser state
        lambda_rm  : RM regularisation weight
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 lambda_rm: float = 0.1, backward_expand: int = 2,
                 rho_init: float = 0.5, seed: int = 0):
        m = hidden_dim * backward_expand
        self.params = init_params(input_dim, hidden_dim, m, output_dim,
                                  rho_init=rho_init, seed=seed)
        self.adam = AdamState()
        self.lambda_rm = lambda_rm
        self.hidden_dim = hidden_dim

    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray,
                   lr: float = 1e-3) -> dict:
        """
        One Adam step over a mini-batch.

        Args:
            x_batch : (B, T, d)
            y_batch : (B, output_dim)
        Returns:
            info : dict with loss values
        """
        B, T, d = x_batch.shape
        params = self.params

        # ---- forward pass ----
        h_batch = rmrnn_forward_batch(params, x_batch)   # (B, T, p)

        # ---- RM loss: average over batch ----
        loss_rm_total = 0.0
        grads_back = {k: np.zeros_like(getattr(params, k))
                      for k in ('W1','b1','W2','b2')}
        for b in range(B):
            loss_b, grads_b = rm_loss_and_grad(params, h_batch[b])
            loss_rm_total += loss_b
            for k in grads_back:
                grads_back[k] += grads_b[k]
        loss_rm = loss_rm_total / B
        for k in grads_back:
            grads_back[k] /= B

        # ---- task loss ----
        h_T = h_batch[:, -1, :]
        loss_task, grads_out = task_loss_and_grad(params, h_T, y_batch)

        # ---- total loss gradient ----
        all_grads = {}
        for k in ('W1','b1','W2','b2'):
            all_grads[k] = self.lambda_rm * grads_back[k]
        for k in ('Wr','br'):
            all_grads[k] = grads_out[k]
        # Wh, Wx, bh: zero gradient (frozen in this implementation;
        # in a full BPTT version the forward RNN params are also updated)
        for k in ('Wh','Wx','bh'):
            all_grads[k] = np.zeros_like(getattr(params, k))

        self.params = self.adam.step(params, all_grads, lr=lr)

        loss_total = loss_task + self.lambda_rm * loss_rm
        return {
            "loss_task":  loss_task,
            "loss_rm":    loss_rm,
            "loss_total": loss_total,
        }

    def convergence_profile(self, x_batch: np.ndarray) -> dict:
        """
        Compute convergence diagnostics on a batch.

        Returns:
            error_vs_t   : (T,)   mean ||h_t - h_T|| (proxy: h_T ≈ h_infty)
            defect_norms : (T-1,) mean ||delta_t|| = ||g(h_{t+1}) - h_{t+1}||
            Q_cumulative : (T-1,) cumulative defect tail sum from t
        """
        params = self.params
        h_batch = rmrnn_forward_batch(params, x_batch)    # (B, T, p)
        B, T, p = h_batch.shape

        h_final = h_batch[:, -1, :]                        # (B, p)
        diff = h_batch - h_final[:, np.newaxis, :]         # (B, T, p)
        error_vs_t = np.mean(np.linalg.norm(diff, axis=-1), axis=0)  # (T,)

        # defects: delta_t = g(h_{t+1}) - h_{t+1}
        h_next = h_batch[:, 1:, :].reshape(-1, p)          # (B*(T-1), p)
        g, _, _ = backward_fwd(params, h_next)
        delta = g - h_next                                  # (B*(T-1), p)
        delta = delta.reshape(B, T - 1, p)
        defect_norms = np.mean(np.linalg.norm(delta, axis=-1), axis=0)  # (T-1,)

        Q_cum = np.cumsum(defect_norms[::-1])[::-1]        # (T-1,) tail sums

        return {
            "error_vs_t":   error_vs_t,
            "defect_norms": defect_norms,
            "Q_cumulative": Q_cum,
        }

    def contraction_rate(self) -> float:
        sv = np.linalg.svd(self.params.Wh, compute_uv=False)
        return float(sv[0])

    def quasi_martingale_total(self, x_batch: np.ndarray) -> float:
        prof = self.convergence_profile(x_batch)
        return float(prof["defect_norms"].sum())
