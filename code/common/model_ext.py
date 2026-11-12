"""
model_ext.py — Extended RM architectures: RMLSTM, RMGRU, variable-depth projector,
               and full BPTT for the joint-vs-staged comparison.
Supplements model.py (Elman RMRNN) without modifying it.
"""

import numpy as np
import copy
from typing import Optional
from model import AdamState, tanh, dtanh

# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

def dsigmoid_from_s(s):
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Dict-based Adam (works on plain dicts, not dataclasses)
# ---------------------------------------------------------------------------

class DictAdam:
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params: dict, grads: dict, lr: float = 1e-3) -> dict:
        self.t += 1
        t, b1, b2, eps = self.t, self.beta1, self.beta2, self.eps
        out = dict(params)
        for k, g in grads.items():
            self.m[k] = b1 * self.m.get(k, np.zeros_like(g)) + (1 - b1) * g
            self.v[k] = b2 * self.v.get(k, np.zeros_like(g)) + (1 - b2) * g**2
            m_hat = self.m[k] / (1 - b1**t)
            v_hat = self.v[k] / (1 - b2**t)
            out[k] = params[k] - lr * m_hat / (np.sqrt(v_hat) + eps)
        return out


# ---------------------------------------------------------------------------
# Variable-depth backward projector
# ---------------------------------------------------------------------------

class BackwardProjector:
    """
    Residual backward MLP g(h) ≈ E[h_t | h_{t+1}]:
      depth=1: g(h) = h + W1 h + b1            (linear residual)
      depth=2: g(h) = h + W2 tanh(W1 h+b1)+b2  (standard)
      depth=3: adds one more tanh layer
    Zero-initialised output weights so g ≈ identity at start.
    """

    def __init__(self, p: int, m: int, depth: int = 2, seed: int = 0):
        assert depth in (1, 2, 3)
        rng = np.random.RandomState(seed + 100)
        self.depth, self.p, self.m = depth, p, m

        if depth == 1:
            self.params = {'W1': np.zeros((p, p)), 'b1': np.zeros(p)}
        elif depth == 2:
            self.params = {
                'W1': rng.randn(m, p) * 0.01, 'b1': np.zeros(m),
                'W2': np.zeros((p, m)),        'b2': np.zeros(p),
            }
        else:
            self.params = {
                'W1': rng.randn(m, p) * 0.01, 'b1': np.zeros(m),
                'W2': rng.randn(m, m) * 0.01, 'b2': np.zeros(m),
                'W3': np.zeros((p, m)),        'b3': np.zeros(p),
            }
        self.adam = DictAdam()

    def forward(self, h):
        """h: (B, p) or (p,); returns (g, intermediates)."""
        squeeze = h.ndim == 1
        if squeeze:
            h = h[np.newaxis, :]
        pr = self.params
        if self.depth == 1:
            g = h + h @ pr['W1'].T + pr['b1']
            ints = {}
        elif self.depth == 2:
            z1 = h @ pr['W1'].T + pr['b1']
            a1 = tanh(z1)
            g = h + a1 @ pr['W2'].T + pr['b2']
            ints = {'z1': z1, 'a1': a1}
        else:
            z1 = h @ pr['W1'].T + pr['b1']
            a1 = tanh(z1)
            z2 = a1 @ pr['W2'].T + pr['b2']
            a2 = tanh(z2)
            g = h + a2 @ pr['W3'].T + pr['b3']
            ints = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        if squeeze:
            g = g[0]
        return g, ints

    def rm_loss_and_grad(self, h_seq):
        """h_seq: (T, p) → (loss, grads_dict)."""
        T, p = h_seq.shape
        if T < 2:
            return 0.0, {k: np.zeros_like(v) for k, v in self.params.items()}

        h_t    = h_seq[:-1]   # (T-1, p) targets
        h_next = h_seq[1:]    # (T-1, p) projector inputs
        g, ints = self.forward(h_next)
        residual = g - h_t                            # (T-1, p)
        loss = 0.5 * float(np.mean(np.sum(residual**2, axis=-1)))
        dg = residual / (T - 1)

        pr = self.params
        grads = {}
        if self.depth == 1:
            grads['W1'] = dg.T @ h_next
            grads['b1'] = dg.sum(0)
        elif self.depth == 2:
            z1, a1 = ints['z1'], ints['a1']
            grads['W2'] = dg.T @ a1
            grads['b2'] = dg.sum(0)
            da1 = dg @ pr['W2']
            dz1 = da1 * dtanh(z1)
            grads['W1'] = dz1.T @ h_next
            grads['b1'] = dz1.sum(0)
        else:
            z1, a1 = ints['z1'], ints['a1']
            z2, a2 = ints['z2'], ints['a2']
            grads['W3'] = dg.T @ a2
            grads['b3'] = dg.sum(0)
            da2 = dg @ pr['W3']
            dz2 = da2 * dtanh(z2)
            grads['W2'] = dz2.T @ a1
            grads['b2'] = dz2.sum(0)
            da1 = dz2 @ pr['W2']
            dz1 = da1 * dtanh(z1)
            grads['W1'] = dz1.T @ h_next
            grads['b1'] = dz1.sum(0)
        return loss, grads

    def jacobian(self, h):
        """Jacobian dg/dh at h (B, p) → (B, p, p)."""
        B, p = h.shape
        pr = self.params
        _, ints = self.forward(h)
        if self.depth == 1:
            J = np.eye(p)[np.newaxis] + pr['W1'].T[np.newaxis]  # (1, p, p)
            return np.broadcast_to(J, (B, p, p)).copy()
        elif self.depth == 2:
            dact = dtanh(ints['z1'])                    # (B, m)
            # dg/dh = I + W2 @ diag(dact) @ W1
            J = np.eye(p)[np.newaxis].repeat(B, 0)
            for b in range(B):
                J[b] += (pr['W2'] * dact[b, np.newaxis, :]) @ pr['W1']
            return J
        else:
            a1, a2 = ints['a1'], ints['a2']
            dact1, dact2 = dtanh(ints['z1']), dtanh(ints['z2'])
            J = np.eye(p)[np.newaxis].repeat(B, 0)
            for b in range(B):
                dg_da2 = pr['W3']                            # (p, m)
                da2_da1 = (pr['W2'] * dact2[b, np.newaxis, :])   # (m, m)
                da1_dh  = (pr['W1'] * dact1[b, np.newaxis, :]).T  # (p, m) wait...
                # chain: dg/dh = W3 @ diag(dact2) @ W2 @ diag(dact1) @ W1
                J[b] += dg_da2 @ (pr['W2'] * dact2[b]) @ (pr['W1'] * dact1[b, np.newaxis, :])
            return J

    def update(self, batch_grads, lr):
        self.params = self.adam.step(self.params, batch_grads, lr)


# ---------------------------------------------------------------------------
# Elman RMRNN with full BPTT (for joint-vs-staged comparison)
# ---------------------------------------------------------------------------

class RMRNNJoint:
    """
    Elman RMRNN with full BPTT through the forward cell.
    Used to compare joint end-to-end training (this class) vs the staged
    protocol in the baseline RMRNN (which freezes forward params).
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 lambda_rm=0.1, backward_expand=2, rho_init=0.5, seed=0):
        p, d = hidden_dim, input_dim
        m = p * backward_expand
        rng = np.random.RandomState(seed)

        Wh_raw = rng.randn(p, p) * 0.1
        sv = np.linalg.svd(Wh_raw, compute_uv=False)
        Wh = Wh_raw * (rho_init / sv[0])

        self.cell = {
            'Wh': Wh,
            'Wx': rng.randn(p, d) * 0.1,
            'bh': np.zeros(p),
            'Wr': rng.randn(output_dim, p) * 0.1,
            'br': np.zeros(output_dim),
        }
        self.proj = BackwardProjector(p, m, depth=2, seed=seed)
        self.adam_cell = DictAdam()
        self.lambda_rm = lambda_rm
        self.hidden_dim, self.p = p, p

    def _forward(self, x_batch):
        B, T, d = x_batch.shape
        Wh, Wx, bh = self.cell['Wh'], self.cell['Wx'], self.cell['bh']
        h = np.zeros((B, self.p))
        h_list, pre_list = [], []
        for t in range(T):
            pre = h @ Wh.T + x_batch[:, t, :] @ Wx.T + bh
            h = tanh(pre)
            h_list.append(h.copy())
            pre_list.append(pre.copy())
        return np.stack(h_list, 1), np.stack(pre_list, 1)   # (B,T,p)

    def train_step(self, x_batch, y_batch, lr=1e-3):
        B, T, d = x_batch.shape
        h_all, pre_all = self._forward(x_batch)   # (B,T,p)

        # ---- task loss ----
        Wr, br = self.cell['Wr'], self.cell['br']
        h_T = h_all[:, -1, :]
        y_pred = h_T @ Wr.T + br
        res = y_pred - y_batch
        loss_task = 0.5 * float(np.mean(np.sum(res**2, axis=-1)))

        # gradient of task loss at h_T
        dy = res / B                               # (B, out)
        dh_task_T = dy @ Wr                        # (B, p)

        # ---- RM loss ----
        loss_rm = 0.0
        # dh_rm[t] = upstream RM gradient at h_t   shape: (B, p)
        dh_rm = [np.zeros((B, self.p)) for _ in range(T)]

        for b in range(B):
            l_b, _ = self.proj.rm_loss_and_grad(h_all[b])
            loss_rm += l_b
        loss_rm /= B

        # Compute RM upstream gradients at each h_t analytically
        for t in range(T - 1):
            residual = h_all[:, t + 1, :] - h_all[:, t, :]  # simplified: no projector gradient through h
            pass

        # Full upstream at h_t from RM loss (using stored h_all):
        # L_RM = (1/(T-1)) * sum_{s=0}^{T-2} ||g(h_{s+1}) - h_s||^2
        # dL/dh_t = -delta_t/(T-1)  as target   (for t < T)
        # dL/dh_t = J_t^T delta_{t-1} / (T-1)  as input to g (for t > 0)
        h_next_all = h_all[:, 1:, :]      # (B, T-1, p)
        h_t_all    = h_all[:, :-1, :]     # (B, T-1, p)

        g_all = np.empty_like(h_next_all)
        for t in range(T - 1):
            g_t, _ = self.proj.forward(h_next_all[:, t, :])
            g_all[:, t, :] = g_t

        deltas = g_all - h_t_all          # (B, T-1, p), delta_t = g(h_{t+1}) - h_t

        dh_rm_arr = np.zeros((B, T, self.p))
        # As targets (h_t for t=0..T-2):
        dh_rm_arr[:, :T-1, :] -= deltas / (T - 1)
        # As projector inputs (h_{t+1} for t=0..T-2 → position t+1):
        # gradient through g: J^T @ delta_t / (T-1)
        # Use batched jacobian (expensive for large p; fine for p=8)
        J = self.proj.jacobian(h_next_all.reshape(-1, self.p))   # (B*(T-1), p, p)
        J = J.reshape(B, T-1, self.p, self.p)
        dh_rm_arr[:, 1:T, :] += np.einsum('btsq,bts->btq', J, deltas) / (T - 1)

        # ---- combine and BPTT ----
        dh_arr = dh_rm_arr * self.lambda_rm       # (B, T, p)
        dh_arr[:, -1, :] += dh_task_T             # task gradient at h_T

        Wh = self.cell['Wh']
        dWh = np.zeros_like(Wh)
        dWx = np.zeros_like(self.cell['Wx'])
        dbh = np.zeros(self.p)

        dh = np.zeros((B, self.p))
        for t in reversed(range(T)):
            dh += dh_arr[:, t, :]
            dpre = dh * dtanh(pre_all[:, t, :])    # (B, p)
            h_prev = h_all[:, t-1, :] if t > 0 else np.zeros((B, self.p))
            dWh += dpre.T @ h_prev
            dWx += dpre.T @ x_batch[:, t, :]
            dbh += dpre.sum(0)
            dh = dpre @ Wh

        # readout gradients
        dWr = dy.T @ h_T
        dbr = dy.sum(0)

        cell_grads = {'Wh': dWh, 'Wx': dWx, 'bh': dbh, 'Wr': dWr, 'br': dbr}
        self.cell = self.adam_cell.step(self.cell, cell_grads, lr)

        # ---- update projector ----
        proj_grads_sum = {k: np.zeros_like(v) for k, v in self.proj.params.items()}
        for b in range(B):
            _, pg = self.proj.rm_loss_and_grad(h_all[b])
            for k in proj_grads_sum:
                proj_grads_sum[k] += pg[k]
        for k in proj_grads_sum:
            proj_grads_sum[k] /= B
        self.proj.update(proj_grads_sum, lr)

        loss_total = loss_task + self.lambda_rm * loss_rm
        return {'loss_task': loss_task, 'loss_rm': loss_rm, 'loss_total': loss_total}

    def convergence_profile(self, x_batch):
        h_all, _ = self._forward(x_batch)
        B, T, p = h_all.shape
        h_final = h_all[:, -1, :]
        diff = h_all - h_final[:, np.newaxis, :]
        error_vs_t = np.mean(np.linalg.norm(diff, axis=-1), axis=0)

        h_next = h_all[:, 1:, :].reshape(-1, p)
        g, _ = self.proj.forward(h_next)
        delta = g - h_next
        delta = delta.reshape(B, T-1, p)
        defect_norms = np.mean(np.linalg.norm(delta, axis=-1), axis=0)
        Q_cum = np.cumsum(defect_norms[::-1])[::-1]
        return {'error_vs_t': error_vs_t, 'defect_norms': defect_norms,
                'Q_cumulative': Q_cum}

    def quasi_martingale_total(self, x_batch):
        return float(self.convergence_profile(x_batch)['defect_norms'].sum())


# ---------------------------------------------------------------------------
# Elman RMRNN with variable-depth projector (for depth ablation)
# ---------------------------------------------------------------------------

class RMRNNDepth:
    """Same as RMRNN but uses BackwardProjector with variable depth."""

    def __init__(self, input_dim, hidden_dim, output_dim,
                 lambda_rm=0.1, backward_expand=2, depth=2,
                 rho_init=0.5, seed=0):
        p, d = hidden_dim, input_dim
        m = p * backward_expand
        rng = np.random.RandomState(seed)
        Wh_raw = rng.randn(p, p) * 0.1
        sv = np.linalg.svd(Wh_raw, compute_uv=False)
        self.cell = {
            'Wh': Wh_raw * (rho_init / sv[0]),
            'Wx': rng.randn(p, d) * 0.1,
            'bh': np.zeros(p),
            'Wr': rng.randn(1, p) * 0.1,
            'br': np.zeros(1),
        }
        self.proj = BackwardProjector(p, m, depth=depth, seed=seed)
        self.adam_out = DictAdam()
        self.lambda_rm = lambda_rm
        self.p = p

    def _forward(self, x_batch):
        B, T, _ = x_batch.shape
        Wh, Wx, bh = self.cell['Wh'], self.cell['Wx'], self.cell['bh']
        h = np.zeros((B, self.p))
        h_list = []
        for t in range(T):
            h = tanh(h @ Wh.T + x_batch[:, t, :] @ Wx.T + bh)
            h_list.append(h.copy())
        return np.stack(h_list, 1)

    def train_step(self, x_batch, y_batch, lr=1e-3):
        B, T, _ = x_batch.shape
        h_all = self._forward(x_batch)

        # RM loss
        loss_rm, proj_grads = 0.0, {k: np.zeros_like(v) for k, v in self.proj.params.items()}
        for b in range(B):
            l, g = self.proj.rm_loss_and_grad(h_all[b])
            loss_rm += l
            for k in g: proj_grads[k] += g[k]
        loss_rm /= B
        for k in proj_grads: proj_grads[k] /= B

        # task loss
        h_T = h_all[:, -1, :]
        Wr, br = self.cell['Wr'], self.cell['br']
        res = h_T @ Wr.T + br - y_batch
        loss_task = 0.5 * float(np.mean(np.sum(res**2, axis=-1)))
        dy = res / B
        out_grads = {'Wr': dy.T @ h_T, 'br': dy.sum(0)}

        self.cell.update(self.adam_out.step({'Wr': self.cell['Wr'], 'br': self.cell['br']},
                                            out_grads, lr))
        self.proj.update(proj_grads, lr)

        return {'loss_task': loss_task, 'loss_rm': loss_rm,
                'loss_total': loss_task + self.lambda_rm * loss_rm}

    def convergence_profile(self, x_batch):
        h_all = self._forward(x_batch)
        B, T, p = h_all.shape
        h_final = h_all[:, -1, :]
        diff = h_all - h_final[:, np.newaxis, :]
        error_vs_t = np.mean(np.linalg.norm(diff, axis=-1), axis=0)
        h_next = h_all[:, 1:, :].reshape(-1, p)
        g, _ = self.proj.forward(h_next)
        delta = (g - h_next).reshape(B, T-1, p)
        defect_norms = np.mean(np.linalg.norm(delta, axis=-1), axis=0)
        return {'error_vs_t': error_vs_t, 'defect_norms': defect_norms,
                'Q_cumulative': np.cumsum(defect_norms[::-1])[::-1]}

    def quasi_martingale_total(self, x_batch):
        return float(self.convergence_profile(x_batch)['defect_norms'].sum())


# ---------------------------------------------------------------------------
# Vanilla LSTM baseline (no RM)
# ---------------------------------------------------------------------------

class VanillaLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, rho_init=0.5, seed=0):
        p, d = hidden_dim, input_dim
        rng = np.random.RandomState(seed)
        self.p = p
        self.cell = {
            'W': rng.randn(4*p, p+d) * 0.1,
            'b': np.concatenate([np.zeros(p), np.ones(p), np.zeros(2*p)]),  # forget bias=1
            'Wr': rng.randn(output_dim, p) * 0.1,
            'br': np.zeros(output_dim),
        }
        self.adam = DictAdam()

    def _step(self, h, c, x):
        p = self.p
        W, b = self.cell['W'], self.cell['b']
        hx = np.concatenate([h, x], axis=-1)
        pre = hx @ W.T + b
        i = sigmoid(pre[:, :p])
        f = sigmoid(pre[:, p:2*p])
        g = tanh(pre[:, 2*p:3*p])
        o = sigmoid(pre[:, 3*p:])
        c_new = f * c + i * g
        h_new = o * tanh(c_new)
        return h_new, c_new

    def forward_batch(self, x_batch):
        B, T, _ = x_batch.shape
        h, c = np.zeros((B, self.p)), np.zeros((B, self.p))
        h_list = []
        for t in range(T):
            h, c = self._step(h, c, x_batch[:, t, :])
            h_list.append(h.copy())
        return np.stack(h_list, 1)

    def train_step(self, x_batch, y_batch, lr=1e-3):
        B, T, _ = x_batch.shape
        h_all = self.forward_batch(x_batch)
        h_T = h_all[:, -1, :]
        Wr, br = self.cell['Wr'], self.cell['br']
        res = h_T @ Wr.T + br - y_batch
        loss = 0.5 * float(np.mean(np.sum(res**2, axis=-1)))
        dy = res / B
        grads = {'Wr': dy.T @ h_T, 'br': dy.sum(0),
                 'W': np.zeros_like(self.cell['W']),
                 'b': np.zeros_like(self.cell['b'])}
        self.cell = self.adam.step(self.cell, grads, lr)
        return {'loss_task': loss, 'loss_rm': 0.0, 'loss_total': loss}

    def convergence_profile(self, x_batch):
        h_all = self.forward_batch(x_batch)
        B, T, p = h_all.shape
        h_final = h_all[:, -1, :]
        diff = h_all - h_final[:, np.newaxis, :]
        error_vs_t = np.mean(np.linalg.norm(diff, axis=-1), axis=0)
        return {'error_vs_t': error_vs_t, 'defect_norms': np.zeros(T-1),
                'Q_cumulative': np.zeros(T-1)}

    def quasi_martingale_total(self, x_batch):
        return 0.0


# ---------------------------------------------------------------------------
# RMLSTM: LSTM with RM backward projector on h_t
# ---------------------------------------------------------------------------

class RMLSTM:
    """
    LSTM + reverse-martingale regularization.
    joint_cell=False: projector on h_t only  (rm_lstm_h)
    joint_cell=True : projector on h_t and c_t with weight alpha  (rm_lstm_cell)
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 lambda_rm=0.1, backward_expand=2, joint_cell=False,
                 depth=2, rho_init=0.5, seed=0):
        p, d = hidden_dim, input_dim
        m = p * backward_expand
        rng = np.random.RandomState(seed)
        self.p = p
        self.lambda_rm = lambda_rm
        self.joint_cell = joint_cell
        self.alpha_cell = 0.5   # weight on c_t term

        self.cell = {
            'W': rng.randn(4*p, p+d) * 0.1,
            'b': np.concatenate([np.zeros(p), np.ones(p), np.zeros(2*p)]),
            'Wr': rng.randn(output_dim, p) * 0.1,
            'br': np.zeros(output_dim),
        }
        self.proj_h = BackwardProjector(p, m, depth, seed)
        if joint_cell:
            self.proj_c = BackwardProjector(p, m, depth, seed + 1)
        self.adam_out = DictAdam()

    def _forward(self, x_batch):
        B, T, _ = x_batch.shape
        p = self.p
        W, b = self.cell['W'], self.cell['b']
        h, c = np.zeros((B, p)), np.zeros((B, p))
        h_list, c_list = [], []
        for t in range(T):
            hx = np.concatenate([h, x_batch[:, t, :]], axis=-1)
            pre = hx @ W.T + b
            i = sigmoid(pre[:, :p])
            f = sigmoid(pre[:, p:2*p])
            g = tanh(pre[:, 2*p:3*p])
            o = sigmoid(pre[:, 3*p:])
            c = f * c + i * g
            h = o * tanh(c)
            h_list.append(h.copy()); c_list.append(c.copy())
        return np.stack(h_list, 1), np.stack(c_list, 1)

    def train_step(self, x_batch, y_batch, lr=1e-3):
        B, T, _ = x_batch.shape
        h_all, c_all = self._forward(x_batch)

        # RM loss on h
        loss_rm_h, grads_h = 0.0, {k: np.zeros_like(v) for k, v in self.proj_h.params.items()}
        for b in range(B):
            l, g = self.proj_h.rm_loss_and_grad(h_all[b])
            loss_rm_h += l
            for k in g: grads_h[k] += g[k]
        loss_rm_h /= B
        for k in grads_h: grads_h[k] /= B

        loss_rm_c = 0.0
        if self.joint_cell:
            grads_c = {k: np.zeros_like(v) for k, v in self.proj_c.params.items()}
            for b in range(B):
                l, g = self.proj_c.rm_loss_and_grad(c_all[b])
                loss_rm_c += l
                for k in g: grads_c[k] += g[k]
            loss_rm_c /= B
            for k in grads_c: grads_c[k] /= B

        # task loss
        h_T = h_all[:, -1, :]
        Wr, br = self.cell['Wr'], self.cell['br']
        res = h_T @ Wr.T + br - y_batch
        loss_task = 0.5 * float(np.mean(np.sum(res**2, axis=-1)))
        dy = res / B
        out_grads = {'Wr': dy.T @ h_T, 'br': dy.sum(0)}
        self.cell.update(self.adam_out.step({'Wr': self.cell['Wr'], 'br': self.cell['br']},
                                            out_grads, lr))
        self.proj_h.update(grads_h, lr)
        if self.joint_cell:
            self.proj_c.update(grads_c, lr)

        loss_rm = loss_rm_h + (self.alpha_cell * loss_rm_c if self.joint_cell else 0.0)
        return {'loss_task': loss_task, 'loss_rm': loss_rm,
                'loss_total': loss_task + self.lambda_rm * loss_rm}

    def convergence_profile(self, x_batch):
        h_all, _ = self._forward(x_batch)
        B, T, p = h_all.shape
        h_final = h_all[:, -1, :]
        diff = h_all - h_final[:, np.newaxis, :]
        error_vs_t = np.mean(np.linalg.norm(diff, axis=-1), axis=0)
        h_next = h_all[:, 1:, :].reshape(-1, p)
        g, _ = self.proj_h.forward(h_next)
        delta = (g - h_next).reshape(B, T-1, p)
        defect_norms = np.mean(np.linalg.norm(delta, axis=-1), axis=0)
        return {'error_vs_t': error_vs_t, 'defect_norms': defect_norms,
                'Q_cumulative': np.cumsum(defect_norms[::-1])[::-1]}

    def quasi_martingale_total(self, x_batch):
        return float(self.convergence_profile(x_batch)['defect_norms'].sum())

    def forward_batch(self, x_batch):
        h_all, _ = self._forward(x_batch)
        return h_all


# ---------------------------------------------------------------------------
# Vanilla GRU baseline (no RM)
# ---------------------------------------------------------------------------

class VanillaGRU:
    def __init__(self, input_dim, hidden_dim, output_dim, rho_init=0.5, seed=0):
        p, d = hidden_dim, input_dim
        rng = np.random.RandomState(seed)
        self.p = p
        self.cell = {
            'Wzx': rng.randn(p, d) * 0.1, 'Wzh': rng.randn(p, p) * 0.1, 'bz': np.zeros(p),
            'Wrx': rng.randn(p, d) * 0.1, 'Wrh': rng.randn(p, p) * 0.1, 'br': np.zeros(p),
            'Wnx': rng.randn(p, d) * 0.1, 'Wnh': rng.randn(p, p) * 0.1, 'bn': np.zeros(p),
            'Wr_out': rng.randn(output_dim, p) * 0.1, 'br_out': np.zeros(output_dim),
        }
        self.adam = DictAdam()

    def _step(self, h, x):
        c = self.cell
        z = sigmoid(x @ c['Wzx'].T + h @ c['Wzh'].T + c['bz'])
        r = sigmoid(x @ c['Wrx'].T + h @ c['Wrh'].T + c['br'])
        n = tanh(x @ c['Wnx'].T + (r * h) @ c['Wnh'].T + c['bn'])
        return (1 - z) * n + z * h

    def forward_batch(self, x_batch):
        B, T, _ = x_batch.shape
        h = np.zeros((B, self.p))
        h_list = []
        for t in range(T):
            h = self._step(h, x_batch[:, t, :])
            h_list.append(h.copy())
        return np.stack(h_list, 1)

    def train_step(self, x_batch, y_batch, lr=1e-3):
        h_all = self.forward_batch(x_batch)
        h_T = h_all[:, -1, :]
        c = self.cell
        res = h_T @ c['Wr_out'].T + c['br_out'] - y_batch
        loss = 0.5 * float(np.mean(np.sum(res**2, axis=-1)))
        B = h_T.shape[0]
        dy = res / B
        grads = {k: np.zeros_like(v) for k, v in c.items()}
        grads['Wr_out'] = dy.T @ h_T
        grads['br_out'] = dy.sum(0)
        self.cell = self.adam.step(self.cell, grads, lr)
        return {'loss_task': loss, 'loss_rm': 0.0, 'loss_total': loss}

    def convergence_profile(self, x_batch):
        h_all = self.forward_batch(x_batch)
        B, T, p = h_all.shape
        h_final = h_all[:, -1, :]
        diff = h_all - h_final[:, np.newaxis, :]
        error_vs_t = np.mean(np.linalg.norm(diff, axis=-1), axis=0)
        return {'error_vs_t': error_vs_t, 'defect_norms': np.zeros(T-1),
                'Q_cumulative': np.zeros(T-1)}

    def quasi_martingale_total(self, x_batch):
        return 0.0


# ---------------------------------------------------------------------------
# RMGRU: GRU with RM backward projector on h_t
# ---------------------------------------------------------------------------

class RMGRU:
    def __init__(self, input_dim, hidden_dim, output_dim,
                 lambda_rm=0.1, backward_expand=2, depth=2,
                 rho_init=0.5, seed=0):
        p, d = hidden_dim, input_dim
        rng = np.random.RandomState(seed)
        self.p = p
        self.lambda_rm = lambda_rm
        self.cell = {
            'Wzx': rng.randn(p, d) * 0.1, 'Wzh': rng.randn(p, p) * 0.1, 'bz': np.zeros(p),
            'Wrx': rng.randn(p, d) * 0.1, 'Wrh': rng.randn(p, p) * 0.1, 'br': np.zeros(p),
            'Wnx': rng.randn(p, d) * 0.1, 'Wnh': rng.randn(p, p) * 0.1, 'bn': np.zeros(p),
            'Wr_out': rng.randn(output_dim, p) * 0.1, 'br_out': np.zeros(output_dim),
        }
        self.proj = BackwardProjector(p, p * backward_expand, depth, seed)
        self.adam_out = DictAdam()

    def _step(self, h, x):
        c = self.cell
        z = sigmoid(x @ c['Wzx'].T + h @ c['Wzh'].T + c['bz'])
        r = sigmoid(x @ c['Wrx'].T + h @ c['Wrh'].T + c['br'])
        n = tanh(x @ c['Wnx'].T + (r * h) @ c['Wnh'].T + c['bn'])
        return (1 - z) * n + z * h

    def forward_batch(self, x_batch):
        B, T, _ = x_batch.shape
        h = np.zeros((B, self.p))
        h_list = []
        for t in range(T):
            h = self._step(h, x_batch[:, t, :])
            h_list.append(h.copy())
        return np.stack(h_list, 1)

    def train_step(self, x_batch, y_batch, lr=1e-3):
        B, T, _ = x_batch.shape
        h_all = self.forward_batch(x_batch)

        loss_rm, proj_grads = 0.0, {k: np.zeros_like(v) for k, v in self.proj.params.items()}
        for b in range(B):
            l, g = self.proj.rm_loss_and_grad(h_all[b])
            loss_rm += l
            for k in g: proj_grads[k] += g[k]
        loss_rm /= B
        for k in proj_grads: proj_grads[k] /= B

        h_T = h_all[:, -1, :]
        c = self.cell
        res = h_T @ c['Wr_out'].T + c['br_out'] - y_batch
        loss_task = 0.5 * float(np.mean(np.sum(res**2, axis=-1)))
        dy = res / B
        out_grads_full = {k: np.zeros_like(v) for k, v in c.items()}
        out_grads_full['Wr_out'] = dy.T @ h_T
        out_grads_full['br_out'] = dy.sum(0)
        self.cell = self.adam_out.step(self.cell, out_grads_full, lr)
        self.proj.update(proj_grads, lr)

        return {'loss_task': loss_task, 'loss_rm': loss_rm,
                'loss_total': loss_task + self.lambda_rm * loss_rm}

    def convergence_profile(self, x_batch):
        h_all = self.forward_batch(x_batch)
        B, T, p = h_all.shape
        h_final = h_all[:, -1, :]
        diff = h_all - h_final[:, np.newaxis, :]
        error_vs_t = np.mean(np.linalg.norm(diff, axis=-1), axis=0)
        h_next = h_all[:, 1:, :].reshape(-1, p)
        g, _ = self.proj.forward(h_next)
        delta = (g - h_next).reshape(B, T-1, p)
        defect_norms = np.mean(np.linalg.norm(delta, axis=-1), axis=0)
        return {'error_vs_t': error_vs_t, 'defect_norms': defect_norms,
                'Q_cumulative': np.cumsum(defect_norms[::-1])[::-1]}

    def quasi_martingale_total(self, x_batch):
        return float(self.convergence_profile(x_batch)['defect_norms'].sum())


# ---------------------------------------------------------------------------
# Full-BPTT batched GRU (fair forecast baseline, drop-in for RMRNNJoint)
# ---------------------------------------------------------------------------

class GRUJoint:
    """A standard GRU trained end-to-end by full BPTT over batched windows,
    with a last-step linear readout. Interface matches RMRNNJoint so it can be
    used as a drop-in third arm in the forecast experiments: it exposes
    ``_forward(x)->(h_all,None)`` and ``cell['Wr'],cell['br']`` for the readout.
    It has no backward projector, so ``quasi_martingale_total`` returns NaN.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, seed=0):
        p, d = hidden_dim, input_dim
        rng = np.random.RandomState(seed)
        sc = 1.0 / np.sqrt(p)
        self.p = p
        self.cell = {
            # update gate z, reset gate s, candidate c
            'Wz': rng.randn(p, d) * sc, 'Uz': rng.randn(p, p) * sc, 'bz': np.zeros(p),
            'Ws': rng.randn(p, d) * sc, 'Us': rng.randn(p, p) * sc, 'bs': np.zeros(p),
            'Wc': rng.randn(p, d) * sc, 'Uc': rng.randn(p, p) * sc, 'bc': np.zeros(p),
            # last-step readout (named Wr/br to match RMRNNJoint)
            'Wr': rng.randn(output_dim, p) * sc, 'br': np.zeros(output_dim),
        }
        self.adam = DictAdam()
        self.lambda_rm = 0.0   # for API parity; unused

    def _forward(self, x_batch):
        c = self.cell
        B, T, d = x_batch.shape
        h = np.zeros((B, self.p))
        H = np.zeros((B, T, self.p))
        for t in range(T):
            x = x_batch[:, t, :]
            z = sigmoid(x @ c['Wz'].T + h @ c['Uz'].T + c['bz'])
            s = sigmoid(x @ c['Ws'].T + h @ c['Us'].T + c['bs'])
            cand = tanh(x @ c['Wc'].T + (s * h) @ c['Uc'].T + c['bc'])
            h = (1 - z) * h + z * cand
            H[:, t, :] = h
        return H, None

    def train_step(self, x_batch, y_batch, lr=1e-3):
        c = self.cell
        B, T, d = x_batch.shape
        p = self.p
        # ---- forward, caching gate activations ----
        h = np.zeros((B, p))
        Z = np.zeros((B, T, p)); S = np.zeros((B, T, p))
        C = np.zeros((B, T, p)); H = np.zeros((B, T, p))
        for t in range(T):
            x = x_batch[:, t, :]
            z = sigmoid(x @ c['Wz'].T + h @ c['Uz'].T + c['bz'])
            s = sigmoid(x @ c['Ws'].T + h @ c['Us'].T + c['bs'])
            cand = tanh(x @ c['Wc'].T + (s * h) @ c['Uc'].T + c['bc'])
            h = (1 - z) * h + z * cand
            Z[:, t, :], S[:, t, :], C[:, t, :], H[:, t, :] = z, s, cand, h

        h_T = H[:, -1, :]
        y_pred = h_T @ c['Wr'].T + c['br']
        res = y_pred - y_batch
        loss = 0.5 * float(np.mean(np.sum(res ** 2, axis=-1)))
        dy = res / B
        g = {k: np.zeros_like(v) for k, v in c.items()}
        g['Wr'] = dy.T @ h_T
        g['br'] = dy.sum(0)

        dh = np.zeros((B, p))
        dh_task_T = dy @ c['Wr']          # (B,p) task gradient at h_T
        for t in reversed(range(T)):
            x = x_batch[:, t, :]
            z, s, cand = Z[:, t, :], S[:, t, :], C[:, t, :]
            hp = H[:, t - 1, :] if t > 0 else np.zeros((B, p))
            if t == T - 1:
                dh = dh + dh_task_T
            dhp = (1 - z) * dh
            dcand = z * dh
            dz = (cand - hp) * dh
            dpre_c = dcand * (1 - cand ** 2)
            g['Wc'] += dpre_c.T @ x
            g['Uc'] += dpre_c.T @ (s * hp)
            g['bc'] += dpre_c.sum(0)
            ds_h = dpre_c @ c['Uc']        # grad wrt (s*hp)
            ds = ds_h * hp
            dhp += ds_h * s
            dpre_s = ds * s * (1 - s)
            g['Ws'] += dpre_s.T @ x
            g['Us'] += dpre_s.T @ hp
            g['bs'] += dpre_s.sum(0)
            dhp += dpre_s @ c['Us']
            dpre_z = dz * z * (1 - z)
            g['Wz'] += dpre_z.T @ x
            g['Uz'] += dpre_z.T @ hp
            g['bz'] += dpre_z.sum(0)
            dhp += dpre_z @ c['Uz']
            dh = dhp

        self.cell = self.adam.step(self.cell, g, lr)
        return {"loss_task": loss, "loss_rm": 0.0, "loss_total": loss}

    def quasi_martingale_total(self, x_batch):
        return float("nan")
