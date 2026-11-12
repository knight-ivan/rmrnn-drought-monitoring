from typing import Optional
"""
Synthetic data generators for the three simulation scenarios
=============================================================

Scenario A — Stationary i.i.d.       (validates §7.1–7.10 / toy example)
Scenario B — AR(1) phi-mixing         (validates §7.11)
Scenario C — Piecewise-stationary     (validates §7.13 concept-drift)

Each function returns:
    x_batch  : (B, T, d)   input sequences
    y_batch  : (B, d)      regression target (the true mean of the process)
    meta     : dict        true parameters, change-point times, etc.
"""

import numpy as np
from typing import NamedTuple


class DataBatch(NamedTuple):
    x_batch: np.ndarray    # (B, T, d)
    y_batch: np.ndarray    # (B, d)
    meta:    dict


# ---------------------------------------------------------------------------
# Scenario A: i.i.d. Gaussian
# ---------------------------------------------------------------------------

def make_iid(B: int, T: int, d: int,
             mu_range: float = 1.0, sigma: float = 1.0,
             rng: Optional[np.random.RandomState] = None) -> DataBatch:
    """
    x_t ~ N(mu, sigma^2 I),  mu ~ Uniform[-mu_range, mu_range]^d.
    Target: y = mu.
    """
    if rng is None: rng = np.random.RandomState()
    mu = rng.uniform(-mu_range, mu_range, size=(B, d))
    noise = rng.randn(B, T, d) * sigma
    x = mu[:, np.newaxis, :] + noise
    return DataBatch(x, mu, {"true_mean": mu, "scenario": "iid", "sigma": sigma})


# ---------------------------------------------------------------------------
# Scenario B: AR(1) phi-mixing
# ---------------------------------------------------------------------------

def make_ar1(B: int, T: int, d: int,
             phi: float = 0.7, mu_range: float = 1.0, sigma: float = 1.0,
             rng: Optional[np.random.RandomState] = None) -> DataBatch:
    """
    x_t = phi * x_{t-1} + (1-phi)*mu + sqrt(1-phi^2)*sigma * eps_t,
    eps_t ~ i.i.d. N(0,I).
    Stationary with mean mu; phi-mixing coefficients phi(k) = phi^k.
    Target: y = mu.
    """
    if rng is None: rng = np.random.RandomState()
    assert 0 < phi < 1
    inno_std = sigma * np.sqrt(1 - phi**2)

    mu = rng.uniform(-mu_range, mu_range, size=(B, d))
    x  = mu.copy()
    x_list = []
    for _ in range(T):
        eps = rng.randn(B, d) * inno_std
        x = phi * x + (1 - phi) * mu + eps
        x_list.append(x.copy())

    x_batch = np.stack(x_list, axis=1)   # (B, T, d)
    return DataBatch(
        x_batch, mu,
        {"true_mean": mu, "scenario": "ar1", "phi": phi, "mixing_rate": phi}
    )


# ---------------------------------------------------------------------------
# Scenario C: Piecewise-stationary / concept drift
# ---------------------------------------------------------------------------

def make_concept_drift(B: int, T: int, d: int,
                       n_segments: int = 3, sigma: float = 1.0,
                       drift_mag: float = 1.5,
                       rng: Optional[np.random.RandomState] = None) -> DataBatch:
    """
    Sequence is split into n_segments equal blocks.  Each block has i.i.d.
    N(mu_k, sigma^2 I) inputs with a block-specific mean mu_k drawn uniformly
    in [-drift_mag, drift_mag]^d.

    Change points: t_k = k * seg_len  for k = 1, .., n_segments-1.
    Target: y = mu_{n_segments}  (mean of the last/current segment).
    """
    if rng is None: rng = np.random.RandomState()
    seg_len = T // n_segments
    change_points = [k * seg_len for k in range(1, n_segments)]

    mus = [rng.uniform(-drift_mag, drift_mag, size=(B, d))
           for _ in range(n_segments)]

    x_list = []
    for t in range(T):
        seg = min(t // seg_len, n_segments - 1)
        x_t = mus[seg] + rng.randn(B, d) * sigma
        x_list.append(x_t.copy())

    x_batch = np.stack(x_list, axis=1)   # (B, T, d)
    y_batch = mus[-1]
    return DataBatch(
        x_batch, y_batch,
        {
            "segment_means": mus,
            "change_points": change_points,
            "n_segments": n_segments,
            "scenario": "concept_drift",
        }
    )


# ---------------------------------------------------------------------------
# Real data: Taiwan daily station precipitation (Raingel / CODiS archive)
# ---------------------------------------------------------------------------

def load_precip_matrix(csv_path: str):
    """Read a tidy `date, station_id, precip_mm` CSV (as produced by
    fetch_taiwan_hydromet.py --source raingel) into a dense matrix.

    Returns (M, dates, stations) where
        M        : (n_days, n_stations) float array, NaN for missing values,
        dates    : list[str] of length n_days, sorted ascending,
        stations : list[str] of length n_stations, sorted.
    """
    import csv as _csv

    table: dict = {}            # date -> {station -> value}
    stations_set: set = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            date = row["date"]
            stn = row["station_id"]
            raw = (row.get("precip_mm") or "").strip()
            try:
                val = float(raw)
            except ValueError:
                val = np.nan
            table.setdefault(date, {})[stn] = val
            stations_set.add(stn)

    dates = sorted(table)
    stations = sorted(stations_set)
    M = np.full((len(dates), len(stations)), np.nan, dtype=float)
    s_idx = {s: j for j, s in enumerate(stations)}
    for i, date in enumerate(dates):
        for stn, val in table[date].items():
            M[i, s_idx[stn]] = val
    return M, dates, stations


def make_real_precip(B: Optional[int], T: int, d: Optional[int],
                     csv_path: Optional[str] = None,
                     log1p: bool = True,
                     fill: str = "zero",
                     stride: int = 1,
                     rng: Optional[np.random.RandomState] = None) -> DataBatch:
    """Sliding-window batches from real Taiwan daily station precipitation.

    Each window of length ``T`` is the input ``x`` and the *next* day's
    precipitation vector is the forecast target ``y`` (one-step-ahead),
    matching the paper's forecast-preserving early-warning setup.

    Parameters
    ----------
    csv_path : path to a tidy `date, station_id, precip_mm` CSV.
    T        : input window length (days).
    log1p    : apply log1p(mm) to compress the heavy right tail (recommended
               for precipitation extremes); stored in meta for inversion.
    fill     : how to handle missing days — "zero" (treat NaN as 0 mm, the
               usual convention for a no-record rain gauge day) or "drop"
               (skip any window containing a NaN).
    stride   : step between consecutive window starts.
    B, d     : ignored (kept for dispatcher signature compatibility); the
               actual batch size and feature dim come from the data.
    """
    if not csv_path:
        raise ValueError("make_real_precip requires csv_path=<daily_precip_long.csv>")

    M, dates, stations = load_precip_matrix(csv_path)   # (n_days, n_stations)
    n_days, n_stations = M.shape
    if n_days <= T:
        raise ValueError(f"Need more than T={T} days; CSV has only {n_days}.")

    raw = M.copy()
    if log1p:
        # Precipitation is non-negative; clip tiny negatives from rounding.
        M = np.log1p(np.clip(M, 0.0, None))

    nan_mask = np.isnan(M)
    if fill == "zero":
        M = np.where(nan_mask, 0.0, M)
    elif fill != "drop":
        raise ValueError("fill must be 'zero' or 'drop'.")

    x_list, y_list, start_idx = [], [], []
    for i in range(0, n_days - T, stride):
        window = M[i:i + T]          # (T, d)
        target = M[i + T]            # (d,)
        if fill == "drop" and (np.isnan(window).any() or np.isnan(target).any()):
            continue
        x_list.append(window)
        y_list.append(target)
        start_idx.append(i)

    if not x_list:
        raise ValueError("No valid windows produced (check T, fill, and data coverage).")

    x_batch = np.stack(x_list, axis=0)              # (B, T, d)
    y_batch = np.stack(y_list, axis=0)              # (B, d)
    return DataBatch(
        x_batch, y_batch,
        {
            "scenario": "real_precip",
            "stations": stations,
            "dates": dates,
            "window_start_idx": start_idx,
            "target_dates": [dates[i + T] for i in start_idx],
            "log1p": log1p,
            "fill": fill,
            "raw_matrix": raw,
            "source": csv_path,
        },
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def get_batch(scenario: str, B: int, T: int, d: int,
              rng: Optional[np.random.RandomState] = None,
              **kwargs) -> DataBatch:
    fns = {
        "iid":           make_iid,
        "ar1":           make_ar1,
        "concept_drift": make_concept_drift,
        "real_precip":   make_real_precip,
    }
    if scenario not in fns:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(fns)}")
    return fns[scenario](B=B, T=T, d=d, rng=rng, **kwargs)
