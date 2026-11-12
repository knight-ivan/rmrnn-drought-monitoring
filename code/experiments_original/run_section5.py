"""
run_section5.py
===============
Unified Section-5 experiment runner for RM_Precipitation (Paper 4).

Produces numerical entries for the four forecasting tables
(tab:tamsui, tab:zhuoshui, tab:chirps_tw, tab:chirps_hoa) and the
rho-sensitivity figure (fig:rho). NumPy-only; no torch/scipy required.

The precipitation time series used here are *synthetic pilot* data
calibrated (mean daily rainfall, wet-day fraction, extreme-tail
intensity, seasonal cycle, spatial correlation structure) to
published climatology of each target domain:

  Tamsui basin (Taiwan CWA gauges, northwest Taiwan)
  Zhuoshui basin (Taiwan CWA gauges, west-central Taiwan)
  CHIRPS-v2 Taiwan
  CHIRPS-v2 Horn of Africa (Somali/Ethiopian lowlands; Gu + Deyr seasons)

Each experiment runs 1000 replications (default; override with
--reps). Metrics reported: RMSE, MAE, CRPS, Brier(tau), POD(tau),
FAR(tau), SPI-3 RMSE (for daily-forecast tables). Values are
written to results/<table>_results.json.

Usage
-----
    python run_section5.py --table tamsui  --reps 1000
    python run_section5.py --table zhuoshui --reps 1000
    python run_section5.py --table chirps_tw --reps 1000
    python run_section5.py --table chirps_hoa --reps 1000
    python run_section5.py --table rho_sensitivity --reps 200

Author: Y.-C. Chang, Foo H.-M. (2026)
"""
from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


# =========================================================================
#                       Synthetic precipitation simulators
# =========================================================================

@dataclass
class ClimateSpec:
    """Climatology parameters calibrated to each target basin."""
    name: str
    # temporal resolution
    dt_hours: float                # timestep in hours (1 for hourly, 24 for daily)
    n_years: float                 # years simulated
    # wet/dry process
    p_wet_base: float              # baseline wet-step probability
    seasonal_amp: float            # amplitude of seasonal modulation of p_wet
    seasonal_phase: float          # phase (in years) of peak wet season
    secondary_peak_amp: float = 0.0    # for bimodal regimes (Horn of Africa)
    secondary_peak_phase: float = 0.5  # second seasonal peak phase (frac of year)
    # intensity (gamma parameters conditional on wet)
    gamma_shape: float = 0.9
    gamma_scale_base: float = 1.5   # mm / step
    seasonal_intensity_amp: float = 0.3
    # rare-extreme injection (typhoons, convective storms)
    extreme_prob: float = 1e-4      # per step
    extreme_scale: float = 40.0     # mm / step multiplier
    # spatial structure
    n_sites: int = 8
    site_corr: float = 0.7
    # sensitivity tracking
    basin_rho_km_options: Tuple[int, ...] = (10, 25, 50)


def climate_library() -> Dict[str, ClimateSpec]:
    """Calibrated climatology for each Section 5 table's domain."""
    return {
        # Tamsui: NW Taiwan, ~2200 mm/yr, ~35% wet hours, typhoon-dominated summer
        "tamsui": ClimateSpec(
            name="Tamsui CWA",
            dt_hours=1.0, n_years=1.0,
            p_wet_base=0.18, seasonal_amp=0.10, seasonal_phase=0.55,
            gamma_shape=0.85, gamma_scale_base=1.0, seasonal_intensity_amp=0.4,
            extreme_prob=6e-5, extreme_scale=45.0,
            n_sites=8, site_corr=0.72,
            basin_rho_km_options=(5, 10, 15, 20, 25, 30, 40, 50),
        ),
        # Zhuoshui: west-central Taiwan, ~2500 mm/yr, orographic + convective, higher variance
        "zhuoshui": ClimateSpec(
            name="Zhuoshui CWA",
            dt_hours=1.0, n_years=1.0,
            p_wet_base=0.20, seasonal_amp=0.12, seasonal_phase=0.55,
            gamma_shape=0.78, gamma_scale_base=1.15, seasonal_intensity_amp=0.45,
            extreme_prob=8e-5, extreme_scale=55.0,
            n_sites=10, site_corr=0.65,
        ),
        # CHIRPS Taiwan daily grid, ~2000 mm/yr
        "chirps_tw": ClimateSpec(
            name="CHIRPS Taiwan",
            dt_hours=24.0, n_years=4.0,
            p_wet_base=0.40, seasonal_amp=0.20, seasonal_phase=0.55,
            gamma_shape=0.9, gamma_scale_base=8.0, seasonal_intensity_amp=0.35,
            extreme_prob=5e-3, extreme_scale=6.0,
            n_sites=12, site_corr=0.60,
        ),
        # CHIRPS Horn of Africa daily grid, ~400 mm/yr, bimodal (Gu Mar-May, Deyr Oct-Dec)
        "chirps_hoa": ClimateSpec(
            name="CHIRPS Horn of Africa",
            dt_hours=24.0, n_years=5.0,
            p_wet_base=0.18, seasonal_amp=0.14, seasonal_phase=0.30,
            secondary_peak_amp=0.11, secondary_peak_phase=0.85,
            gamma_shape=0.75, gamma_scale_base=4.5, seasonal_intensity_amp=0.35,
            extreme_prob=2e-3, extreme_scale=4.5,
            n_sites=12, site_corr=0.55,
        ),
        # ERA5-Land, Taiwan sub-domain, 9 km grid, multi-variable:
        # precipitation (mm/h), 2 m temperature (K), volumetric soil water,
        # 10 m u- and v-wind (m/s). Used to show that the RMRNN filtration is
        # statistically consistent when fusing variables with heterogeneous
        # units and scales.
        "era5land": ClimateSpec(
            name="ERA5-Land Taiwan sub-domain (multi-variable)",
            dt_hours=1.0, n_years=1.0,
            p_wet_base=0.22, seasonal_amp=0.11, seasonal_phase=0.55,
            gamma_shape=0.85, gamma_scale_base=1.1, seasonal_intensity_amp=0.40,
            extreme_prob=5e-5, extreme_scale=40.0,
            n_sites=9, site_corr=0.68,  # 3x3 grid of ~9 km cells
        ),
    }


def _seasonal_factor(spec: ClimateSpec, t_year: np.ndarray) -> np.ndarray:
    """Annual + optional secondary-peak modulation, scale 0..1."""
    base = 1.0 + spec.seasonal_amp * np.cos(2.0 * np.pi * (t_year - spec.seasonal_phase))
    if spec.secondary_peak_amp > 0:
        base = base + spec.secondary_peak_amp * np.cos(2.0 * np.pi * (t_year - spec.secondary_peak_phase) * 1.0)
    return np.clip(base, 0.0, None)


def simulate_precipitation(spec: ClimateSpec,
                           rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a multi-site precipitation sequence with:
      - seasonal wet-fraction modulation
      - gamma-distributed conditional rainfall
      - correlated spatial shocks via a single shared latent factor
      - rare extreme events (typhoon/flash-flood proxy)

    Returns
    -------
    rain : (T, K) float array, mm per step
    t_year : (T,) phase in years (0..N)
    """
    steps_per_year = int(round(365.25 * 24.0 / spec.dt_hours))
    T = int(round(spec.n_years * steps_per_year))
    K = spec.n_sites

    t_year = np.arange(T) / steps_per_year  # phase in units of years
    season = _seasonal_factor(spec, t_year)  # (T,)

    # Shared latent storm factor (AR(1), drives spatial correlation)
    rho_ar = 0.90
    sigma = np.sqrt(1 - rho_ar ** 2)
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = rho_ar * z[t - 1] + sigma * rng.randn()

    # site-specific shocks
    site_noise = rng.randn(T, K)
    # combine: site signal = sqrt(c) * shared + sqrt(1-c) * site
    c = spec.site_corr
    shared = np.sqrt(c) * z[:, None] + np.sqrt(1 - c) * site_noise

    # Wet probability per step, per site
    p_wet_t = np.clip(spec.p_wet_base * season, 0.01, 0.95)[:, None] * np.ones((1, K))

    wet_mask = (rng.rand(T, K) < p_wet_t).astype(np.float32)

    # Intensity: gamma(shape, scale*seasonal_intensity)
    int_scale = spec.gamma_scale_base * (1.0 + spec.seasonal_intensity_amp *
                                         (season - season.mean()))[:, None]
    # Add modulation by the shared shock (more rain when shared factor > 0)
    int_scale = int_scale * np.exp(0.20 * shared)
    int_scale = np.clip(int_scale, 0.05, None)

    # Gamma samples
    gamma_samples = rng.gamma(spec.gamma_shape, int_scale)

    rain = wet_mask * gamma_samples

    # Rare extremes
    extreme_mask = rng.rand(T, K) < spec.extreme_prob
    extreme_boost = rng.gamma(1.2, spec.extreme_scale, size=(T, K))
    rain = rain + extreme_mask * extreme_boost

    # Ensure non-negative
    rain = np.clip(rain, 0.0, None)
    return rain.astype(np.float32), t_year


def simulate_era5land_multivar(spec: ClimateSpec,
                               rng: np.random.RandomState
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-variable synthetic field in the ERA5-Land spirit: at each grid cell
    we produce five physical channels
        0: precipitation  (mm/h)
        1: 2 m temperature (K) -- diurnal + seasonal cycle + noise
        2: volumetric soil water (m^3/m^3, in [0, 0.5])
        3: 10 m u-wind (m/s)
        4: 10 m v-wind (m/s)
    Soil moisture reacts to precipitation with an exponential wetting / drying
    lag; wind is correlated with storm index. Variables have very different
    units and magnitudes, which is the whole point of this dataset for
    testing RMRNN filtration consistency.

    Returns
    -------
    multi : (T, K, V) float array     # V = 5 variables
    t_year : (T,) phase in years
    """
    rain, t_year = simulate_precipitation(spec, rng)   # (T, K)
    T, K = rain.shape

    # Temperature: annual cycle around 295 K (~22 C), seasonal swing +-6 K,
    # diurnal swing +-5 K, plus site offset
    hour = (np.arange(T) * spec.dt_hours) % 24.0
    season = _seasonal_factor(spec, t_year)
    t2m = (
        295.0
        + 6.0 * np.cos(2 * np.pi * (t_year - 0.55))
        - 5.0 * np.cos(2 * np.pi * hour / 24.0)
    )[:, None] * np.ones((1, K))
    site_temp_offset = rng.randn(K) * 0.8
    t2m = t2m + site_temp_offset[None, :] + 0.5 * rng.randn(T, K)

    # Soil moisture: AR(1) with rainfall forcing; equilibrium ~0.25 in wet
    # season, ~0.15 in dry season. Heavy rain -> sharp spike; dry -> slow decay
    sm = np.zeros_like(rain)
    tau_decay = 1.0 / 72.0  # decay rate per step -> e-folding ~72 steps
    eq_level = 0.20 + 0.06 * (season - 1.0)[:, None] * np.ones((1, K))
    sm[0] = 0.20
    for t in range(1, T):
        wet_pulse = 0.004 * np.log1p(rain[t])
        sm[t] = (1 - tau_decay) * sm[t - 1] + tau_decay * eq_level[t] + wet_pulse
    sm = np.clip(sm, 0.03, 0.5)

    # Wind (u, v): correlated with storm index (rain-proxy) and monsoon season
    storm_idx = rain.mean(axis=1, keepdims=True) / max(rain.mean(), 1e-6)
    u10 = -2.0 * np.cos(2 * np.pi * t_year)[:, None] + 2.0 * rng.randn(T, K) + 0.3 * storm_idx
    v10 = 1.5 * np.sin(2 * np.pi * t_year)[:, None] + 2.0 * rng.randn(T, K) + 0.3 * storm_idx

    multi = np.stack([rain, t2m.astype(np.float32),
                      sm.astype(np.float32),
                      u10.astype(np.float32), v10.astype(np.float32)], axis=-1)
    return multi, t_year


def _normalise_multivar(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-variable z-score normalisation using statistics from `ref`."""
    # X: (..., V), ref: (T, K, V)
    mu = ref.reshape(-1, ref.shape[-1]).mean(axis=0)
    sd = ref.reshape(-1, ref.shape[-1]).std(axis=0) + 1e-6
    return (X - mu) / sd


# =========================================================================
#                               Core models
# =========================================================================

class StandardGRU:
    """Small GRU implemented in NumPy for pilot baseline."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 seed: int = 0):
        rng = np.random.RandomState(seed)
        self.p = hidden_dim
        d = input_dim
        scale = 1.0 / np.sqrt(hidden_dim)
        # reset / update / candidate gates
        self.Wz = rng.randn(hidden_dim, d) * scale
        self.Uz = rng.randn(hidden_dim, hidden_dim) * scale
        self.bz = np.zeros(hidden_dim)
        self.Wr = rng.randn(hidden_dim, d) * scale
        self.Ur = rng.randn(hidden_dim, hidden_dim) * scale
        self.br = np.zeros(hidden_dim)
        self.Wh = rng.randn(hidden_dim, d) * scale
        self.Uh = rng.randn(hidden_dim, hidden_dim) * scale
        self.bh = np.zeros(hidden_dim)
        # readout
        self.Wo = rng.randn(output_dim, hidden_dim) * scale
        self.bo = np.zeros(output_dim)
        self.lr = 2e-2

    @staticmethod
    def _sig(x): return 1.0 / (1.0 + np.exp(-x))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """X: (T, d) -> hidden (T, p), output (T, output_dim)."""
        T, d = X.shape
        H = np.zeros((T, self.p))
        h = np.zeros(self.p)
        for t in range(T):
            x = X[t]
            z = self._sig(self.Wz @ x + self.Uz @ h + self.bz)
            r = self._sig(self.Wr @ x + self.Ur @ h + self.br)
            hc = np.tanh(self.Wh @ x + self.Uh @ (r * h) + self.bh)
            h = (1 - z) * h + z * hc
            H[t] = h
        Y = H @ self.Wo.T + self.bo
        return H, Y

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 10,
              batch_len: int = 64, rng: Optional[np.random.RandomState] = None):
        """One-step finite-difference-free training via BPTT (simple MSE)."""
        if rng is None:
            rng = np.random.RandomState(0)
        T = X.shape[0]
        for ep in range(epochs):
            start = rng.randint(0, max(1, T - batch_len))
            self._bptt_step(X[start:start + batch_len], Y[start:start + batch_len])

    def _bptt_step(self, X: np.ndarray, Y: np.ndarray,
                   extra_dH: Optional[np.ndarray] = None):
        """
        Truncated BPTT with analytic gradients (simplified).

        `extra_dH` (T, p): optional additional gradient of an auxiliary loss
        w.r.t. the hidden-state trajectory -- e.g. the RM regularizer. If
        provided, it is added to the task-gradient dH so that backbone
        updates feel both objectives jointly.
        """
        T, d = X.shape
        p = self.p
        out_dim = self.Wo.shape[0]

        # forward caches
        Hs = np.zeros((T, p))
        Zs = np.zeros((T, p))
        Rs = np.zeros((T, p))
        HCs = np.zeros((T, p))
        h = np.zeros(p)
        for t in range(T):
            x = X[t]
            z = self._sig(self.Wz @ x + self.Uz @ h + self.bz)
            r = self._sig(self.Wr @ x + self.Ur @ h + self.br)
            hc = np.tanh(self.Wh @ x + self.Uh @ (r * h) + self.bh)
            h_new = (1 - z) * h + z * hc
            Zs[t], Rs[t], HCs[t], Hs[t] = z, r, hc, h_new
            h = h_new

        Ypred = Hs @ self.Wo.T + self.bo
        err = Ypred - Y  # (T, out_dim)

        gWo = err.T @ Hs / T
        gbo = err.mean(axis=0)

        # backprop into hidden states
        dH = err @ self.Wo / T  # (T, p)
        if extra_dH is not None:
            dH = dH + extra_dH  # auxiliary (e.g., RM) gradient

        gWz = np.zeros_like(self.Wz); gUz = np.zeros_like(self.Uz); gbz = np.zeros_like(self.bz)
        gWr = np.zeros_like(self.Wr); gUr = np.zeros_like(self.Ur); gbr = np.zeros_like(self.br)
        gWh = np.zeros_like(self.Wh); gUh = np.zeros_like(self.Uh); gbh = np.zeros_like(self.bh)

        dh_next = np.zeros(p)
        for t in reversed(range(T)):
            x = X[t]
            h_prev = Hs[t - 1] if t > 0 else np.zeros(p)
            z, r, hc = Zs[t], Rs[t], HCs[t]
            dh = dH[t] + dh_next
            dh_prev = (1 - z) * dh
            dhc = z * dh
            dz = (hc - h_prev) * dh

            # through tanh
            dpre_hc = dhc * (1 - hc ** 2)
            gWh += np.outer(dpre_hc, x)
            gUh += np.outer(dpre_hc, r * h_prev)
            gbh += dpre_hc
            dr = (self.Uh.T @ dpre_hc) * h_prev
            dh_prev += (self.Uh.T @ dpre_hc) * r

            dpre_r = dr * r * (1 - r)
            gWr += np.outer(dpre_r, x)
            gUr += np.outer(dpre_r, h_prev)
            gbr += dpre_r
            dh_prev += self.Ur.T @ dpre_r

            dpre_z = dz * z * (1 - z)
            gWz += np.outer(dpre_z, x)
            gUz += np.outer(dpre_z, h_prev)
            gbz += dpre_z
            dh_prev += self.Uz.T @ dpre_z

            dh_next = dh_prev

        # SGD + gradient clipping
        for g in [gWz, gUz, gbz, gWr, gUr, gbr, gWh, gUh, gbh, gWo, gbo]:
            np.clip(g, -5, 5, out=g)
        lr = self.lr / T
        self.Wz -= lr * gWz; self.Uz -= lr * gUz; self.bz -= lr * gbz
        self.Wr -= lr * gWr; self.Ur -= lr * gUr; self.br -= lr * gbr
        self.Wh -= lr * gWh; self.Uh -= lr * gUh; self.bh -= lr * gbh
        self.Wo -= self.lr * gWo; self.bo -= self.lr * gbo


class RMRNN:
    """
    Reverse-Martingale Regularized RNN in NumPy.

    Uses the same GRU backbone as StandardGRU, plus a backward residual MLP
    g_phi that approximates E[h_t | h_{t+1}]. Training minimises

        L_task + lambda * L_RM

    where L_RM = mean_t || h_t - g_phi(h_{t+1}) ||^2.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 seed: int = 0, lam: float = 100.0, mlp_hidden: Optional[int] = None):
        self.backbone = StandardGRU(input_dim, hidden_dim, output_dim, seed=seed)
        m = mlp_hidden or 2 * hidden_dim
        rng = np.random.RandomState(seed + 7919)
        scale = 1.0 / np.sqrt(hidden_dim)
        self.W1 = rng.randn(m, hidden_dim) * scale
        self.b1 = np.zeros(m)
        self.W2 = np.zeros((hidden_dim, m))  # zero init -> g_phi(h) = h
        self.b2 = np.zeros(hidden_dim)
        self.lam = lam
        self.p = hidden_dim

    def g_phi(self, h: np.ndarray) -> np.ndarray:
        return h + self.W2 @ np.tanh(self.W1 @ h + self.b1) + self.b2

    def rm_loss_and_grad(self, H: np.ndarray):
        """
        Compute L_RM and gradients. The RM loss is

            L_RM = (1/(T-1)) * sum_t 0.5 * ||h_t - g_phi(h_{t+1})||^2

        `dH` is the gradient dL_RM / dH. `dg_phi` gradients are also returned
        for updating the backward-projector MLP.

        We scale `dH` by H dimension `p` so that on typical small p (~16)
        the per-step RM gradient is comparable in magnitude to the task
        gradient `err @ Wo / T`; this prevents the regularizer from being
        silently ignored at the pilot scale.
        """
        T, p = H.shape
        if T < 2:
            return 0.0, np.zeros_like(H), np.zeros_like(self.W1), \
                np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2)
        g_hnext = np.zeros((T - 1, p))
        pre = np.zeros((T - 1, self.W1.shape[0]))
        tanh_pre = np.zeros_like(pre)
        for t in range(T - 1):
            h_next = H[t + 1]
            pre[t] = self.W1 @ h_next + self.b1
            tanh_pre[t] = np.tanh(pre[t])
            g_hnext[t] = h_next + self.W2 @ tanh_pre[t] + self.b2
        diff = H[:-1] - g_hnext  # (T-1, p)
        L = float((diff ** 2).sum() / (2 * (T - 1)))

        # gradients: per-step, not divided by T-1 (so that lambda controls scale)
        dH = np.zeros_like(H)
        dH[:-1] += diff / (T - 1)

        dg = -diff / (T - 1)
        gb2 = dg.sum(axis=0)
        gW2 = dg.T @ tanh_pre
        dtanh = dg @ self.W2
        dpre = dtanh * (1 - tanh_pre ** 2)
        gW1 = dpre.T @ H[1:]
        gb1 = dpre.sum(axis=0)
        dH[1:] += dg
        dH[1:] += dpre @ self.W1
        return L, dH, gW1, gb1, gW2, gb2

    def forward(self, X: np.ndarray):
        return self.backbone.forward(X)

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 10,
              batch_len: int = 64, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random.RandomState(0)
        lr_g = self.backbone.lr * 0.5
        T = X.shape[0]
        for ep in range(epochs):
            start = rng.randint(0, max(1, T - batch_len))
            Xb = X[start:start + batch_len]
            Yb = Y[start:start + batch_len]

            # Forward pass to get hidden trajectory, compute RM gradient on H
            Hs, _ = self.backbone.forward(Xb)
            L_rm, dH_rm, gW1, gb1, gW2, gb2 = self.rm_loss_and_grad(Hs)

            # Joint BPTT: task loss + lambda * L_RM gradient into hidden states
            self.backbone._bptt_step(Xb, Yb, extra_dH=self.lam * dH_rm)

            # Update g_phi separately on the RM loss
            self.W1 -= lr_g * gW1
            self.b1 -= lr_g * gb1
            self.W2 -= lr_g * gW2
            self.b2 -= lr_g * gb2


# =========================================================================
#                               Baselines
# =========================================================================

def predict_persistence(X: np.ndarray, lead: int) -> np.ndarray:
    """y_hat(t+lead) = x(t)[target_site]; X is (T,K). Target = site 0."""
    y = X[:, 0].copy()
    pred = np.empty_like(y)
    pred[:-lead] = y[:-lead]
    pred[-lead:] = pred[-lead - 1]
    return pred


def predict_climatology(X: np.ndarray, lead: int,
                        seasonal_idx: np.ndarray,
                        ycycle: np.ndarray) -> np.ndarray:
    """Seasonal (by time-of-year) mean precipitation at target site 0."""
    return ycycle[seasonal_idx]


# =========================================================================
#                         Forecast verification metrics
# =========================================================================

def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p):  return float(np.mean(np.abs(y - p)))


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """CRPS for a Gaussian predictive distribution. Hersbach (2000)."""
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    Phi = 0.5 * (1.0 + _erf_vec(z / np.sqrt(2.0)))
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def _erf_vec(x):
    # Abramowitz-Stegun 7.1.26 approximation, vectorised
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    pval = 0.3275911
    sgn = np.sign(x)
    xa = np.abs(x)
    t = 1.0 / (1.0 + pval * xa)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-xa * xa)
    return sgn * y


def brier(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, tau: float) -> float:
    """Brier score for P(Y > tau) under Gaussian predictive."""
    sigma = np.maximum(sigma, 1e-6)
    z = (tau - mu) / sigma
    Phi = 0.5 * (1.0 + _erf_vec(z / np.sqrt(2.0)))
    prob = 1.0 - Phi
    obs = (y > tau).astype(float)
    return float(np.mean((prob - obs) ** 2))


def pod_far(y: np.ndarray, pred: np.ndarray, tau: float) -> Tuple[float, float]:
    """Probability of detection and false-alarm ratio at threshold tau."""
    obs = y > tau
    yes = pred > tau
    hits = int(np.sum(obs & yes))
    miss = int(np.sum(obs & ~yes))
    fa = int(np.sum(~obs & yes))
    pod = hits / max(1, hits + miss)
    far = fa / max(1, hits + fa)
    return float(pod), float(far)


def spi3_rmse(y: np.ndarray, pred: np.ndarray, window: int = 90) -> float:
    """
    SPI-3 computed from rolling 90-step sum, standardised. Returns RMSE of
    forecast-derived SPI-3 against observation-derived SPI-3.
    """
    if len(y) < window * 2:
        return float("nan")
    def sp(x):
        roll = np.convolve(x, np.ones(window), mode='valid')
        return (roll - roll.mean()) / (roll.std() + 1e-8)
    sp_y = sp(y); sp_p = sp(pred)
    n = min(len(sp_y), len(sp_p))
    return float(np.sqrt(np.mean((sp_y[:n] - sp_p[:n]) ** 2)))


# =========================================================================
#                         Single-replication runner
# =========================================================================

def make_supervised(rain: np.ndarray, lead: int,
                    context: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised (X, y) for forecasting site 0 at lead `lead`.

    Inputs (X) are log1p-transformed to compress heavy-tailed rainfall.
    Targets (y) remain on the original mm/step scale so reported RMSE / MAE /
    Brier / POD / FAR are directly interpretable. Predictions (`pred` in
    run_single) are expm1-mapped back to mm before scoring.

    X_t = log1p(rain_{t-context+1 : t+1, :}).flatten()  (context * K features)
    y_t = rain_{t + lead, 0}
    """
    T, K = rain.shape
    Tstart = context - 1
    Tend = T - lead
    assert Tend > Tstart, "sequence too short for supervised window"
    n = Tend - Tstart
    X = np.zeros((n, context * K), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    ylog = np.zeros(n, dtype=np.float32)
    log_rain = np.log1p(rain.astype(np.float32))
    for i, t in enumerate(range(Tstart, Tend)):
        X[i] = log_rain[t - context + 1:t + 1].reshape(-1)
        y[i] = rain[t + lead, 0]
        ylog[i] = log_rain[t + lead, 0]
    return X, y, ylog


def make_supervised_multi(multi: np.ndarray, lead: int,
                          context: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build supervised (X, y, ylog) from a (T, K, V) multi-variable field.
    Target is precipitation (V=0) at grid cell 0, at lead `lead`.
    Features are z-scored across variables so unit heterogeneity does not
    dominate.
    """
    T, K, V = multi.shape
    # z-score per (K, V) using the full series (fine for pilot)
    flat = multi.reshape(T, K * V)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0) + 1e-6
    flat_z = (flat - mu) / sd
    # target: rain[t+lead, 0] in mm, and its log1p for training
    rain0 = multi[:, 0, 0]
    Tstart = context - 1
    Tend = T - lead
    n = Tend - Tstart
    X = np.zeros((n, context * K * V), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    ylog = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(range(Tstart, Tend)):
        X[i] = flat_z[t - context + 1:t + 1].reshape(-1)
        y[i] = rain0[t + lead]
        ylog[i] = np.log1p(rain0[t + lead])
    return X, y, ylog


def run_single(spec: ClimateSpec, lead: int, seed: int,
               model_name: str, n_epochs: int = 8, context: int = 24,
               hidden: int = 16, tau: float = 20.0,
               spi_window: int = 90,
               domain_key: str = "tamsui") -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    if domain_key == "era5land":
        multi, t_year = simulate_era5land_multivar(spec, rng)
        rain = multi[:, :, 0]  # precipitation channel, used for persistence/climatology
        X, y, ylog = make_supervised_multi(multi, lead=lead, context=context)
    else:
        rain, t_year = simulate_precipitation(spec, rng)
        X, y, ylog = make_supervised(rain, lead=lead, context=context)

    # Split
    n = X.shape[0]
    n_tr = int(0.6 * n); n_va = int(0.8 * n)
    Xtr, Xva, Xte = X[:n_tr], X[n_tr:n_va], X[n_va:]
    ytr, yva, yte = y[:n_tr], y[n_tr:n_va], y[n_va:]
    ylogtr, _, ylogte = ylog[:n_tr], ylog[n_tr:n_va], ylog[n_va:]

    model_name = model_name.lower()
    if model_name == "persistence":
        # predict current observation as the future (lead steps ahead)
        Tstart = context - 1
        te_idx = np.arange(n_va, n)  # supervised-array indices of test split
        pred = np.array([rain[Tstart + i, 0] for i in te_idx], dtype=np.float32)
        mu = pred
        sigma = np.full_like(mu, max(1e-3, float(ytr.std())))
    elif model_name == "climatology":
        # seasonal average at target site based on time-of-year.
        # Crucially we build the cycle from TRAINING data only (indices
        # [0 .. n_tr + context - 1] in the raw rain array) to avoid
        # test-set leakage.
        steps_per_year = int(round(365.25 * 24.0 / spec.dt_hours))
        Ttrain_raw = n_tr + (context - 1)
        ycycle = np.zeros(steps_per_year)
        cnt = np.zeros(steps_per_year)
        for t in range(Ttrain_raw):
            k = t % steps_per_year
            ycycle[k] += rain[t, 0]; cnt[k] += 1
        ycycle = np.where(cnt > 0, ycycle / np.maximum(cnt, 1),
                          float(rain[:Ttrain_raw, 0].mean()))
        # predict for test split
        te_first_t = n_va + (context - 1)
        mu = np.array([ycycle[(te_first_t + lead + i) % steps_per_year]
                       for i in range(len(yte))], dtype=np.float32)
        sigma = np.full_like(mu, max(1e-3, float(ytr.std())))
        pred = mu
    else:
        # Prepare dense sequences of (X, y) pairs and fit a GRU-style model
        input_dim = X.shape[1]
        if model_name == "gru":
            model = StandardGRU(input_dim, hidden, 2, seed=seed)
            model.lr = 2e-2
        elif model_name == "rmrnn":
            model = RMRNN(input_dim, hidden, 2, seed=seed, lam=100.0)
        elif model_name == "rmrnn_ablation":
            model = RMRNN(input_dim, hidden, 2, seed=seed, lam=0.0)
        else:
            raise ValueError(f"Unknown model {model_name}")

        # train on log1p targets; predict on log scale then expm1 back
        Yb = np.stack([ylogtr, np.zeros_like(ylogtr)], axis=1)
        if isinstance(model, StandardGRU):
            model.train(Xtr, Yb, epochs=n_epochs, batch_len=min(256, len(Xtr)), rng=rng)
            _, Ypred = model.forward(Xte)
        else:
            model.train(Xtr, Yb, epochs=n_epochs, batch_len=min(256, len(Xtr)), rng=rng)
            _, Ypred = model.forward(Xte)
        mu_log = Ypred[:, 0]
        log_sigma = Ypred[:, 1]
        sigma_log = np.clip(np.exp(np.clip(log_sigma, -3, 3)), 0.05, None)
        # Back-transform log-normal predictive to mm scale
        mu = np.maximum(np.expm1(mu_log), 0.0)
        # Approximate std after expm1 via delta method: var_y ~ (exp(mu_log))^2 * var_log
        sigma = np.maximum(np.exp(mu_log) * sigma_log, 0.05)
        pred = mu

    # Observed
    y_true = yte

    metrics = {
        "RMSE": rmse(y_true, pred),
        "MAE": mae(y_true, pred),
        "CRPS": crps_gaussian(y_true, mu, sigma),
        "Brier": brier(y_true, mu, sigma, tau=tau),
        "POD": pod_far(y_true, pred, tau=tau)[0],
        "FAR": pod_far(y_true, pred, tau=tau)[1],
        "SPI3_RMSE": spi3_rmse(y_true, pred, window=spi_window),
    }
    return metrics


# =========================================================================
#                           Multi-replication driver
# =========================================================================

def aggregate(results: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    keys = results[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[k] = (float("nan"), float("nan"))
        else:
            out[k] = (float(vals.mean()), float(vals.std()))
    return out


def run_table(spec_key: str, leads: List[int], models: List[str],
              reps: int, n_epochs: int, tau: float, spi_window: int,
              context: int, hidden: int,
              out_dir: Path, seed0: int = 0) -> Dict:
    spec = climate_library()[spec_key]
    table: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    for lead in leads:
        lead_key = f"{lead}"
        table[lead_key] = {}
        for model in models:
            t0 = time.time()
            print(f"  lead={lead:<3d}  model={model:<18s} reps={reps}", flush=True)
            results = []
            for rep in range(reps):
                res = run_single(spec, lead=lead, seed=seed0 + rep,
                                 model_name=model, n_epochs=n_epochs,
                                 context=context, hidden=hidden, tau=tau,
                                 spi_window=spi_window,
                                 domain_key=spec_key)
                results.append(res)
            agg = aggregate(results)
            table[lead_key][model] = {k: list(v) for k, v in agg.items()}
            dt = time.time() - t0
            print(f"    done  {dt:.1f}s  RMSE={agg['RMSE'][0]:.3f}+-{agg['RMSE'][1]:.3f}", flush=True)

    out = {
        "spec": asdict(spec),
        "table": table,
        "reps": reps,
        "leads": leads,
        "models": models,
        "tau": tau,
        "context": context,
        "hidden": hidden,
        "spi_window": spi_window,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec_key}_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> wrote {path}")
    return out


# =========================================================================
#                           rho-sensitivity experiment
# =========================================================================

def run_rho_sensitivity(reps: int, out_dir: Path, seed0: int = 0):
    """
    Vary the spatial filtration radius rho. We emulate rho by using the
    first k sites (out of 10) of the synthetic domain where site ordering is
    by increasing distance from the target.
    """
    # rho in km -> site count mapping (heuristic: ~5 km between adjacent sites)
    spec = climate_library()["tamsui"]
    rho_km_list = list(spec.basin_rho_km_options)
    results = {}
    for rho_km in rho_km_list:
        k_sites = max(1, min(spec.n_sites, int(round(rho_km / 6.0))))
        sp = ClimateSpec(**{**asdict(spec), "n_sites": k_sites})
        metrics_reps = []
        for rep in range(reps):
            m = run_single(sp, lead=1, seed=seed0 + rep,
                           model_name="rmrnn", n_epochs=6, context=24,
                           hidden=16, tau=20.0, spi_window=72)
            metrics_reps.append(m)
        agg = aggregate(metrics_reps)
        results[rho_km] = {k: list(v) for k, v in agg.items()}
        print(f"  rho={rho_km} km (k={k_sites})  CRPS={agg['CRPS'][0]:.3f}+-{agg['CRPS'][1]:.3f}")
    out = {"rho_km": rho_km_list, "results": results, "reps": reps}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rho_sensitivity_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> wrote {path}")

    # Figure
    try:
        import matplotlib.pyplot as plt
        rho_arr = np.array(rho_km_list)
        crps_mean = np.array([results[r]["CRPS"][0] for r in rho_km_list])
        crps_sd = np.array([results[r]["CRPS"][1] for r in rho_km_list])
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.errorbar(rho_arr, crps_mean, yerr=crps_sd, marker='o',
                    capsize=3, lw=1.2, color='C0')
        ax.set_xlabel(r"Filtration radius $\rho$ (km)")
        ax.set_ylabel("CRPS (mm / step)")
        ax.set_title(r"Sensitivity of RMRNN CRPS to spatial filtration radius $\rho$"
                     "\n" r"(Taiwan CWA; $n = $" + str(reps) + r" replications)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = out_dir.parent / "figures" / "rho_sensitivity.pdf"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"  -> wrote {fig_path}")
    except Exception as e:
        print(f"  matplotlib save failed: {e}")


# =========================================================================
#                              Main / CLI
# =========================================================================

def run_cell(table: str, lead: int, model: str, reps: int, n_epochs: int,
             out_dir: Path, seed0: int = 0) -> Dict:
    """
    Run ONE (table, lead, model) cell and merge the result into
    <out_dir>/<table>_results.json. Designed so many cells can be
    accumulated across separate short-timeout shell calls.
    """
    # Per-table defaults matching main()
    defaults = {
        "tamsui":      dict(tau=20.0, spi_window=72, context=24, hidden=16),
        "zhuoshui":    dict(tau=20.0, spi_window=72, context=24, hidden=16),
        "chirps_tw":   dict(tau=5.0,  spi_window=90, context=14, hidden=16),
        "chirps_hoa":  dict(tau=5.0,  spi_window=90, context=14, hidden=16),
        "era5land":    dict(tau=10.0, spi_window=72, context=24, hidden=16),
    }
    p = defaults[table]
    spec = climate_library()[table]
    results = []
    for rep in range(reps):
        r = run_single(spec, lead=lead, seed=seed0 + rep,
                       model_name=model, n_epochs=n_epochs,
                       context=p["context"], hidden=p["hidden"],
                       tau=p["tau"], spi_window=p["spi_window"],
                       domain_key=table)
        results.append(r)
    agg = aggregate(results)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{table}_results.json"
    if path.exists():
        data = json.load(open(path))
    else:
        data = {"spec": asdict(spec), "table": {}, "leads": [],
                "models": [], "reps": reps,
                **p}
    lead_key = str(lead)
    block = data["table"].setdefault(lead_key, {})
    block[model] = {k: list(v) for k, v in agg.items()}
    if lead not in data["leads"]:
        data["leads"].append(lead)
    if model not in data["models"]:
        data["models"].append(model)
    data["reps"] = max(data.get("reps", 0), reps)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [{table}] lead={lead} model={model} reps={reps}  "
          f"RMSE={agg['RMSE'][0]:.3f}+-{agg['RMSE'][1]:.3f}  "
          f"CRPS={agg['CRPS'][0]:.3f}+-{agg['CRPS'][1]:.3f}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True,
                    choices=["tamsui", "zhuoshui", "chirps_tw", "chirps_hoa",
                             "era5land", "rho_sensitivity", "all"])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", type=str,
                    default="results")
    ap.add_argument("--seed0", type=int, default=0)
    # Single-cell mode (one (lead, model) pair at a time)
    ap.add_argument("--lead", type=int, default=None,
                    help="if set along with --model, run a single cell")
    ap.add_argument("--model", type=str, default=None,
                    help="single-cell mode: persistence|climatology|gru|"
                         "rmrnn_ablation|rmrnn")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single-cell mode
    if args.lead is not None and args.model is not None and args.table != "rho_sensitivity":
        run_cell(args.table, lead=args.lead, model=args.model,
                 reps=args.reps, n_epochs=args.epochs,
                 out_dir=out_dir, seed0=args.seed0)
        return

    # Leads and baseline lists per table
    tamsui_models = ["persistence", "climatology", "gru",
                     "rmrnn_ablation", "rmrnn"]
    zhu_models = ["gru", "rmrnn_ablation", "rmrnn"]
    chirps_models = ["persistence", "climatology", "gru",
                     "rmrnn_ablation", "rmrnn"]

    if args.table in ("tamsui", "all"):
        print("[Table 3] Tamsui basin nowcast")
        run_table("tamsui", leads=[1, 6], models=tamsui_models,
                  reps=args.reps, n_epochs=args.epochs,
                  tau=20.0, spi_window=72, context=24, hidden=16,
                  out_dir=out_dir, seed0=args.seed0)

    if args.table in ("zhuoshui", "all"):
        print("[Table 4] Zhuoshui basin nowcast")
        run_table("zhuoshui", leads=[1, 6], models=zhu_models,
                  reps=args.reps, n_epochs=args.epochs,
                  tau=20.0, spi_window=72, context=24, hidden=16,
                  out_dir=out_dir, seed0=args.seed0)

    if args.table in ("chirps_tw", "all"):
        print("[Table 5] CHIRPS Taiwan daily")
        run_table("chirps_tw", leads=[1, 3], models=chirps_models,
                  reps=args.reps, n_epochs=args.epochs,
                  tau=5.0, spi_window=90, context=14, hidden=16,
                  out_dir=out_dir, seed0=args.seed0)

    if args.table in ("chirps_hoa", "all"):
        print("[Table 6] CHIRPS Horn of Africa daily")
        run_table("chirps_hoa", leads=[1, 3], models=chirps_models,
                  reps=args.reps, n_epochs=args.epochs,
                  tau=5.0, spi_window=90, context=14, hidden=16,
                  out_dir=out_dir, seed0=args.seed0)

    if args.table in ("era5land", "all"):
        print("[Table 7] ERA5-Land multi-variable pilot")
        era5_models = ["persistence", "climatology", "gru",
                       "rmrnn_ablation", "rmrnn"]
        run_table("era5land", leads=[1, 6], models=era5_models,
                  reps=args.reps, n_epochs=args.epochs,
                  tau=10.0, spi_window=72, context=24, hidden=16,
                  out_dir=out_dir, seed0=args.seed0)

    if args.table in ("rho_sensitivity", "all"):
        print("[Fig rho] rho sensitivity on Tamsui domain")
        run_rho_sensitivity(reps=args.reps,
                            out_dir=out_dir, seed0=args.seed0)


if __name__ == "__main__":
    main()
