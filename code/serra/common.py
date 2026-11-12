"""Shared utilities: data loading, SPI, deseasonalisation, CUSUM, events."""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGIONS = ["texas", "germany", "australia", "taiwan"]
REF = (1961, 1990)
CAL = (1991, 2000)
EVAL = (2001, 2024)
DATES = pd.date_range("1961-01-01", "2024-12-31", freq="D")


def load_region(region):
    """Return (meta DataFrame, precip DataFrame [DATES x stations], NaN = missing)."""
    d = os.path.join(ROOT, "data", region)
    meta = pd.read_csv(os.path.join(d, "station_meta.csv"), dtype={"station_id": str})
    df = pd.read_csv(os.path.join(d, "daily_precip_long.csv"), dtype={"station_id": str})
    df["date"] = pd.to_datetime(df["date"])
    P = df.pivot_table(index="date", columns="station_id", values="precip_mm", aggfunc="first")
    P = P.reindex(DATES)[meta.station_id]
    return meta, P


def load_era5(region, sid):
    f = os.path.join(ROOT, "data", region, "era5land", f"{sid}.csv")
    e = pd.read_csv(f, parse_dates=["date"]).set_index("date").reindex(DATES)
    return e


def doy_index(dates):
    doy = dates.dayofyear.values.copy()
    doy[doy == 366] = 365
    return doy


def year_mask(dates, span):
    return (dates.year >= span[0]) & (dates.year <= span[1])


def smooth_circular(v, k=15):
    ext = np.concatenate([v[-k:], v, v[:k]])
    return np.convolve(ext, np.ones(2 * k + 1) / (2 * k + 1), "same")[k:-k]


def deseasonalise(x, dates, ref=REF, k=15):
    """z = (x - mu_doy) / sd_doy, with mu/sd estimated on ref years, circularly smoothed."""
    x = np.asarray(x, float)
    doy = doy_index(dates)
    m = year_mask(dates, ref) & np.isfinite(x)
    mu = np.full(365, np.nan); sd = np.full(365, np.nan)
    for d in range(1, 366):
        s = m & (doy == d)
        if s.sum() >= 5:
            mu[d - 1] = x[s].mean(); sd[d - 1] = x[s].std()
    idx = np.arange(365)
    for v in (mu, sd):
        g = np.isfinite(v)
        v[:] = np.interp(idx, idx[g], v[g], period=365)
    mu, sd = smooth_circular(mu, k), smooth_circular(sd, k)
    return (x - mu[doy - 1]) / np.maximum(sd[doy - 1], 1e-8)


def _zigamma_z(train, values):
    """Zero-inflated gamma CDF fitted on train, transformed to N(0,1)."""
    train = train[np.isfinite(train)]
    pos = train[train > 0]
    q = float(np.mean(train == 0))
    if len(pos) < 10:
        return np.full(len(values), np.nan)
    a, _, sc = stats.gamma.fit(pos, floc=0)
    p = q + (1 - q) * stats.gamma.cdf(np.maximum(values, 0), a, scale=sc)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = stats.norm.ppf(p)
    out[~np.isfinite(values)] = np.nan
    return out


def spi_daily(p, dates, window=90, min_obs=80, ref=REF, pool=15):
    """Daily SPI on a moving `window`-day total; gamma per day of year,
    pooling +/-`pool` days of the reference years (one value per year per day)."""
    p = pd.Series(np.asarray(p, float), index=dates)
    n = p.notna().rolling(window).sum()
    tot = p.fillna(0).rolling(window).sum() * window / n.clip(lower=1)
    tot[n < min_obs] = np.nan
    tot = tot.values
    doy = doy_index(dates)
    refm = year_mask(dates, ref)
    z = np.full(len(tot), np.nan)
    for d in range(1, 366):
        dd = np.abs(((doy - d + 182) % 365) - 182) <= pool
        train = tot[refm & dd]
        sel = doy == d
        z[sel] = _zigamma_z(train, tot[sel])
    return z


def spi3_monthly(p, dates, ref=REF, valid=0.9):
    """WMO SPI-3: monthly totals (>= valid coverage), 3-month sums, calendar-month
    zero-inflated gamma on ref years. Returns Series indexed by month end."""
    p = pd.Series(np.asarray(p, float), index=dates)
    mon = p.resample("ME").sum(min_count=1)
    cov = p.notna().resample("ME").mean()
    mon[cov < valid] = np.nan
    acc = mon.rolling(3, min_periods=3).sum()
    z = pd.Series(np.nan, index=acc.index)
    for m in range(1, 13):
        sel = acc.index.month == m
        tr = acc[sel & (acc.index.year >= ref[0]) & (acc.index.year <= ref[1])].values
        z[sel] = _zigamma_z(tr, acc[sel].values)
    return z


def cusum_alarms(z, h, k=0.5, refractory=90, sign=-1.0):
    """One-sided CUSUM for a shift in direction `sign` (-1 = downward).
    NaN scores leave the statistic unchanged. Reset after alarm; alarms are
    suppressed for `refractory` steps. Returns alarm indices."""
    z = np.asarray(z, float)
    c, last, out = 0.0, -10 ** 9, []
    for t, v in enumerate(z):
        if not np.isfinite(v):
            continue
        c = max(0.0, c + sign * v - k)
        if c >= h:
            if t - last > refractory:
                out.append(t); last = t
            c = 0.0
    return np.array(out, int)


def cusum_path(z, k=0.5, sign=-1.0):
    z = np.asarray(z, float); c = 0.0; out = np.zeros(len(z))
    for t, v in enumerate(z):
        if np.isfinite(v):
            c = max(0.0, c + sign * v - k)
        out[t] = c
    return out


def threshold_alarms(z, c, refractory=90, step=1):
    """Alarm when z <= c after having been above c (downcrossing), with refractory."""
    z = np.asarray(z, float)
    below = z <= c
    out, last, prev = [], -10 ** 9, False
    for t in range(len(z)):
        if not np.isfinite(z[t]):
            continue
        if below[t] and not prev and t - last > refractory / step:
            out.append(t); last = t
        prev = below[t]
    return np.array(out, int)


def sm_percentile(sm, dates, pool=15):
    """Empirical day-of-year percentile (0-100) of soil moisture over all years."""
    sm = np.asarray(sm, float)
    doy = doy_index(dates)
    out = np.full(len(sm), np.nan)
    for d in range(1, 366):
        dd = np.abs(((doy - d + 182) % 365) - 182) <= pool
        ref = np.sort(sm[dd & np.isfinite(sm)])
        sel = (doy == d) & np.isfinite(sm)
        out[sel] = 100.0 * np.searchsorted(ref, sm[sel], side="right") / len(ref)
    return out


def drought_events(pct, dates, thresh=20.0, min_len=30, merge_gap=30, quiet=60):
    """Run-theory events on the soil-moisture percentile series."""
    dry = np.nan_to_num(pct, nan=100.0) < thresh
    runs, t, n = [], 0, len(dry)
    while t < n:
        if dry[t]:
            s = t
            while t < n and dry[t]:
                t += 1
            runs.append([s, t - 1])
        else:
            t += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 < merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    ev, prev_end = [], -10 ** 9
    for s, e in merged:
        if e - s + 1 >= min_len and s - prev_end - 1 >= quiet:
            ev.append(dict(onset=s, end=e, onset_date=dates[s], end_date=dates[e],
                           duration=e - s + 1, severity=float(np.nanmin(pct[s:e + 1]))))
        prev_end = e
    return ev


def score_alarms(alarms, events, n_days, eval_mask, pre=60, post=120):
    """Event-level scoring. Returns per-event rows and false-alarm count/years."""
    alarms = np.asarray(alarms, int)
    alarms = alarms[eval_mask[alarms]] if len(alarms) else alarms
    rows = []
    for e in events:
        if not eval_mask[e["onset"]]:
            continue
        w = alarms[(alarms >= e["onset"] - pre) & (alarms <= e["onset"] + post)]
        rows.append(dict(onset=e["onset"], hit=len(w) > 0,
                         timing=int(w[0] - e["onset"]) if len(w) else np.nan))
    ok = np.zeros(n_days, bool)
    for e in events:
        ok[max(0, e["onset"] - pre): e["end"] + 1] = True
    fa = int(np.sum(~ok[alarms])) if len(alarms) else 0
    years = float(np.sum(eval_mask & ~ok)) / 365.25
    return rows, fa, years


def _loglogistic_z(train, values):
    """Three-parameter log-logistic fitted by unbiased probability-weighted moments
    (Vicente-Serrano et al. 2010, the standard SPEI estimator), transformed to N(0,1)."""
    from scipy.special import gamma as G
    x = np.sort(train[np.isfinite(train)])
    N = len(x)
    if N < 30:
        return np.full(len(values), np.nan)
    Fi = (np.arange(1, N + 1) - 0.35) / N
    w = [np.mean((1 - Fi) ** k * x) for k in range(3)]
    beta = (2 * w[1] - w[0]) / (6 * w[1] - w[0] - 6 * w[2])
    if not np.isfinite(beta) or beta <= 1:
        return np.full(len(values), np.nan)
    g = G(1 + 1 / beta) * G(1 - 1 / beta)
    alpha = (w[0] - 2 * w[1]) * beta / g
    gam = w[0] - alpha * g
    v = np.asarray(values, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 1.0 / (1.0 + (alpha / (v - gam)) ** beta)
    p = np.where(v <= gam, 1e-6, p)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = stats.norm.ppf(p)
    out[~np.isfinite(v)] = np.nan
    return out


def spei_daily(p, pet, dates, window=90, min_obs=None, ref=REF, pool=15):
    """Daily SPEI on a moving `window`-day sum of P - PET (days with both observed;
    rescaled to the full window), log-logistic per day of year with +/-pool pooling."""
    min_obs = min_obs or int(round(0.9 * window))
    d = pd.Series(np.asarray(p, float) - np.asarray(pet, float), index=dates)
    n = d.notna().rolling(window).sum()
    tot = d.fillna(0).rolling(window).sum() * window / n.clip(lower=1)
    tot[n < min_obs] = np.nan
    tot = tot.values
    doy = doy_index(dates)
    refm = year_mask(dates, ref)
    z = np.full(len(tot), np.nan)
    for dd_ in range(1, 366):
        dd = np.abs(((doy - dd_ + 182) % 365) - 182) <= pool
        sel = doy == dd_
        z[sel] = _loglogistic_z(tot[refm & dd], tot[sel])
    return z
