"""Figures for the SERRA manuscript (all from results/ produced by the pipeline)."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS, load_region

FIG = os.path.join(ROOT, "figures")
NAMES = {"texas": "Central Texas", "germany": "Germany", "australia": "SE Australia", "taiwan": "Taiwan"}
DLAB = {"D1": "D1 defect, λ>0", "D2": "D2 defect, λ=0", "D3": "D3 SPI-90 (daily)", "D4": "D4 SPI-3 (monthly)", "D5": "D5 forecast residual", "D7": "SPI-30 (daily)", "D6": "D1+D3 combined"}
COL = {"D1": "#1b6ca8", "D2": "#7fb3d5", "D3": "#d35400", "D4": "#f0a35e", "D5": "#6c757d", "D7": "#8e44ad", "D6": "#111111"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})


def _lines(name):
    import json
    j = json.load(open(os.path.join(ROOT, "data", "naturalearth", name)))
    out = []
    for f in j["features"]:
        g = f["geometry"]
        parts = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        out += [np.array(p) for p in parts]
    return out


def fig_map():
    coast = _lines("ne_50m_coastline.geojson"); bord = _lines("ne_50m_admin_0_boundary_lines_land.geojson")
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.0))
    for ax, r in zip(axs, REGIONS):
        m, _ = load_region(r)
        x0, x1, y0, y1 = m.lon.min() - 1.2, m.lon.max() + 1.2, m.lat.min() - 1.2, m.lat.max() + 1.2
        for L, c, w in [(coast, "#555", 0.6), (bord, "#999", 0.5)]:
            for p in L:
                if ((p[:, 0] > x0 - 5) & (p[:, 0] < x1 + 5) & (p[:, 1] > y0 - 5) & (p[:, 1] < y1 + 5)).any():
                    ax.plot(p[:, 0], p[:, 1], color=c, lw=w)
        ax.scatter(m.lon, m.lat, s=14, c="#1b6ca8", edgecolor="k", lw=0.3, zorder=3)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_title(f"{NAMES[r]} (n={len(m)})", fontsize=9)
        ax.set_xlabel("Longitude (°E)"); ax.set_aspect("equal")
        ax.grid(alpha=0.3, lw=0.4)
    axs[0].set_ylabel("Latitude (°N)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "station_map.pdf"))


def _grid22(height=5.9):
    """2 x 2 panels drawn at 7.2 in so that 11-pt lettering prints at about 8 pt
    when the figure is scaled to the text width."""
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "legend.fontsize": 10.5})
    fig, axs = plt.subplots(2, 2, figsize=(7.2, height), sharex=True, sharey=True)
    return fig, axs


def _finish22(fig, axs, xlabel, ylabel, ncol, bottom, name):
    for ax in axs[1]:
        ax.set_xlabel(xlabel)
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.4, handlelength=2.2)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(os.path.join(FIG, name))
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8})


def fig_amoc():
    A = pd.read_csv(os.path.join(ROOT, "results", "warning", "amoc.csv"))
    fig, axs = _grid22()
    for ax, r in zip(axs.ravel(), REGIONS):
        for d in ["D1", "D2", "D3", "D4", "D5", "D7"]:
            a = A[(A.region == r) & (A.det == d)].sort_values("far_per_yr")
            if d == "D4":
                a = a[a.thr <= 1e-9]          # dry-side thresholds only (c <= 0)
            if d == "D4":   # month-end threshold rule: alarm counts are not monotone in c
                ax.scatter(a.far_per_yr, a.pod, color=COL[d], s=9, label=DLAB[d], zorder=1)
            else:
                ax.plot(a.far_per_yr, a.pod, color=COL[d], lw=1.8, label=DLAB[d],
                        ls="--" if d in ("D2", "D5", "D7") else "-")
        ax.set_xlim(0, 2.5); ax.set_ylim(0, 1); ax.set_title(NAMES[r])
        ax.grid(alpha=0.3, lw=0.4)
    _finish22(fig, axs, "False alarms per station-year", "Probability of detection", 3, 0.11, "operating_curves.pdf")


def fig_composite():
    """Event-aligned composites: mean anomaly of D1, D2, D3, D7 and soil-wetness
    percentile around reference onsets (2001-2024). Onsets within 180 days of the
    end of the record are left out because their +180-day window is incomplete."""
    lags = np.arange(-180, 181)
    fig, axs = _grid22()
    for ax, r in zip(axs.ravel(), REGIONS):
        pn = np.load(os.path.join(ROOT, "results", "panel", f"{r}.npz"))
        ev = pd.read_csv(os.path.join(ROOT, "results", "panel", f"{r}_events.csv"), dtype={"station_id": str})
        ev = ev[pd.to_datetime(ev.onset_date).dt.year >= 2001]
        sidx = {s: i for i, s in enumerate(pn["station_id"])}
        acc = {k: [] for k in ["D1", "D2", "D3", "D7", "sm"]}
        for _, e in ev.iterrows():
            i = sidx[e.station_id]; t = lags + e.onset
            if t[-1] >= len(DATES):
                continue
            acc["D1"].append(pn["D1"][i][t]); acc["D2"].append(pn["D2"][i][t])
            acc["D3"].append(pn["D3"][i][t]); acc["D7"].append(pn["D7"][i][t])
            acc["sm"].append((pn["smpct"][i][t] - 50) / 28.9)
        for k, c, lab in [("D1", COL["D1"], "D1 defect, λ>0"), ("D2", COL["D2"], "D2 defect, λ=0"),
                          ("D3", COL["D3"], "SPI-90"), ("D7", "#8e44ad", "SPI-30"),
                          ("sm", "#2e7d32", "Soil wetness (rescaled percentile)")]:
            x = np.nanmean(acc[k], 0); ax.plot(lags, x, color=c, lw=1.6, label=lab)
        ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"{NAMES[r]} ({len(acc['D1'])} of {len(ev)} onsets)")
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.grid(alpha=0.3, lw=0.4)
    _finish22(fig, axs, "Days from soil-moisture onset", "Mean standardised anomaly", 3, 0.11, "event_composites.pdf")


def fig_traces(cases):
    """Representative single-station traces: 30-day running means of the daily
    anomalies, alarms of each detector at the primary budget (1 alarm per
    station-year), root-zone soil-wetness percentile and reference onsets."""
    from evaluate_warning import alarms
    T = pd.read_csv(os.path.join(ROOT, "results", "warning", "thresholds.csv"))
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11})
    fig, axs = plt.subplots(len(cases), 1, figsize=(7.4, 2.35 * len(cases)))
    for n, (ax, (r, sid, a, b, title)) in enumerate(zip(np.atleast_1d(axs), cases)):
        pn = np.load(os.path.join(ROOT, "results", "panel", f"{r}.npz"))
        i = list(pn["station_id"]).index(sid)
        ev = pd.read_csv(os.path.join(ROOT, "results", "panel", f"{r}_events.csv"), dtype={"station_id": str})
        ev = ev[ev.station_id == sid]
        sel = (DATES >= a) & (DATES <= b); d = DATES[sel]
        ax2 = ax.twinx()
        sm = pn["smpct"][i][sel]
        ax2.fill_between(d, 0, 100, where=sm < 20, color="#c8e6c9", alpha=0.7, lw=0)
        ax2.plot(d, sm, color="#2e7d32", lw=0.6, alpha=0.8)
        ax2.set_ylim(0, 100); ax2.set_ylabel("Soil wetness (pct.)", color="#2e7d32")
        for k, det in enumerate(["D1", "D3", "D7"]):
            z = pn[det][i].astype(float)
            zm = pd.Series(z).rolling(30, min_periods=20).mean().values[sel]
            ax.plot(d, zm, color=COL[det], lw=1.1, label=DLAB[det])
            h = T[(T.region == r) & (T.det == det) & (T.budget == 1.0)].thr.iloc[0]
            al = np.flatnonzero(alarms(det, z, h)[sel])
            ax.scatter(d[al], np.full(len(al), -2.3 - 0.35 * k), marker="v", color=COL[det], s=28, zorder=5)
        for _, e in ev.iterrows():
            if a <= pd.Timestamp(e.onset_date) <= b:
                ax.axvline(pd.Timestamp(e.onset_date), color="k", lw=0.9, ls="--")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylim(-3.2, 2.2); ax.set_title(title, loc="left")
        ax.set_ylabel("30-day anomaly")
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([], [], color="k", ls="--", lw=0.9)], l + ["reference onset"],
               loc="lower center", ncol=2, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(FIG, "event_traces.pdf"))
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})


def fig_association():
    A = pd.read_csv(os.path.join(ROOT, "results", "covariates", "association.csv"))
    A = A[(A.spec == "full")]
    terms = ["spi3", "sm_a", "t2m_a", "et_a"]; tl = ["SPI-3", "Soil wetness", "2-m temperature", "Evapotranspiration"]
    groups = REGIONS + ["all"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for ax, y in zip(axs, ["D1", "D2"]):
        for j, g in enumerate(groups):
            a = A[(A.group == g) & (A.outcome == y)].set_index("term").reindex(terms)
            yy = np.arange(len(terms)) + (j - 2) * 0.14
            ax.errorbar(a.coef, yy, xerr=[a.coef - a.lo, a.hi - a.coef], fmt="o", ms=3,
                        label=NAMES.get(g, "All regions"), color=plt.cm.tab10(j) if g != "all" else "k")
        ax.axvline(0, color="k", lw=0.5); ax.set_yticks(range(len(terms))); ax.set_yticklabels(tl)
        ax.set_title(DLAB[y], fontsize=9); ax.set_xlabel("Standardised coefficient (95% CI)")
    axs[1].legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(FIG, "association.pdf"))


def fig_composite_mv():
    """Event-aligned composites on the temperature-aware subset."""
    lags = np.arange(-180, 181)
    fig, axs = _grid22()
    spec = [("D1", COL["D1"], "D1 defect, precipitation only", "-"),
            ("D8", "#c0392b", "D8 defect, temperature-aware", "-"),
            ("D10", "#8e44ad", "SPEI-30", "--"), ("D11", "#d35400", "SPEI-90", "--"),
            ("sm", "#2e7d32", "Soil wetness (rescaled percentile)", "-")]
    for ax, r in zip(axs.ravel(), REGIONS):
        pn = np.load(os.path.join(ROOT, "results", "panel_mv", f"{r}.npz"))
        ev = pd.read_csv(os.path.join(ROOT, "results", "panel_mv", f"{r}_events.csv"), dtype={"station_id": str})
        ev = ev[pd.to_datetime(ev.onset_date).dt.year >= 2001]
        si = {s: i for i, s in enumerate(pn["station_id"])}
        acc = {k: [] for k, *_ in spec}
        for _, e in ev.iterrows():
            t = lags + e.onset
            if t[-1] >= len(DATES):
                continue
            i = si[e.station_id]
            for k, *_ in spec:
                acc[k].append((pn["smpct"][i][t] - 50) / 28.9 if k == "sm" else pn[k][i][t])
        for k, c, lab, ls in spec:
            ax.plot(lags, np.nanmean(acc[k], 0), color=c, lw=1.6, ls=ls, label=lab)
        ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"{NAMES[r]} ({len(acc['D1'])} of {len(ev)} onsets)")
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.grid(alpha=0.3, lw=0.4)
    _finish22(fig, axs, "Days from soil-moisture onset", "Mean standardised anomaly", 2, 0.15, "event_composites_mv.pdf")


def fig_deadline():
    """Detection by deadline (review_checks.py): fraction of all reference onsets detected by day d,
    pooled over regions (left), and paired differences with 95% bootstrap bands (right)."""
    W = os.path.join(ROOT, "results", "warning")
    L = pd.read_csv(os.path.join(W, "deadline.csv")); D = pd.read_csv(os.path.join(W, "deadline_diff.csv"))
    L = L[L.group == "all"]; D = D[D.group == "all"]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "legend.fontsize": 10})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for d, lab, ls in [("D1", "D1 defect, λ>0", "-"), ("D3", "D3 SPI-90", "-"), ("D7", "D7 SPI-30", "--"),
                       ("D4", "D4 SPI-3 (monthly)", ":"), ("D6", "D6 D1+D3 combined", "-.")]:
        x = L[L.det == d]; a1.plot(x.deadline, x.frac, color=COL[d], lw=1.8, ls=ls, label=lab)
    a1.axvline(0, color="k", lw=0.6); a1.set_xlim(-60, 120); a1.set_ylim(0, 0.8)
    a1.set_xlabel("Days from soil-moisture onset"); a1.set_ylabel("Fraction of onsets detected")
    a1.set_xticks([-60, -30, 0, 30, 60, 90, 120]); a1.grid(alpha=0.3, lw=0.4); a1.legend(frameon=False, loc="upper left", handlelength=2.4)
    for c, lab, col in [("D1-D3", "D1 − D3", COL["D1"]), ("D6-D3", "D6 − D3", COL["D6"])]:
        x = D[D.comp == c]
        a2.fill_between(x.deadline, x.lo, x.hi, color=col, alpha=0.18, lw=0); a2.plot(x.deadline, x["diff"], color=col, lw=1.8, label=lab)
    a2.axhline(0, color="k", lw=0.6); a2.axvline(0, color="k", lw=0.6); a2.set_xlim(-60, 120)
    a2.set_xticks([-60, -30, 0, 30, 60, 90, 120]); a2.grid(alpha=0.3, lw=0.4)
    a2.set_xlabel("Days from soil-moisture onset"); a2.set_ylabel("Difference in fraction detected")
    a2.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "deadline.pdf"))
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8})


def fig_deadline_regions():
    """Detection by deadline in each region."""
    L = pd.read_csv(os.path.join(ROOT, "results", "warning", "deadline.csv"))
    fig, axs = _grid22()
    for ax, r in zip(axs.ravel(), REGIONS):
        for d, lab, ls in [("D1", "D1 defect, λ>0", "-"), ("D2", "D2 defect, λ=0", "--"), ("D3", "D3 SPI-90", "-"),
                           ("D7", "D7 SPI-30", "--"), ("D4", "D4 SPI-3 (monthly)", ":"), ("D6", "D6 D1+D3 combined", "-.")]:
            x = L[(L.group == r) & (L.det == d)]; ax.plot(x.deadline, x.frac, color=COL[d], lw=1.6, ls=ls, label=lab)
        ax.axvline(0, color="k", lw=0.6); ax.set_xlim(-60, 120); ax.set_ylim(0, 0.9); ax.set_title(NAMES[r])
        ax.set_xticks([-60, 0, 60, 120]); ax.grid(alpha=0.3, lw=0.4)
    _finish22(fig, axs, "Days from soil-moisture onset", "Fraction of onsets detected", 3, 0.11, "deadline_regions.pdf")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    what = sys.argv[1:] or ["map", "amoc", "composite", "association", "traces", "deadline"]
    if "deadline" in what: fig_deadline(); fig_deadline_regions()
    if "map" in what: fig_map()
    if "amoc" in what: fig_amoc()
    if "composite" in what: fig_composite()
    if "association" in what: fig_association()
    if "composite_mv" in what: fig_composite_mv()
    if "traces" in what:
        m = {r: load_region(r)[0].set_index("station_id").name.to_dict() for r in REGIONS}
        cases = [("texas", "USC00411048", "2010-01-01", "2012-03-31", "2010–11 Texas drought"),
                 ("germany", "GME00122746", "2017-09-01", "2019-06-30", "2018 central-European drought"),
                 ("australia", "ASN00075032", "2017-06-01", "2019-12-31", "2018–19 south-eastern Australian drought"),
                 ("taiwan", "466990", "2019-11-01", "2021-08-31", "2020–21 Taiwan drought")]
        cases = [(r, s_, pd.Timestamp(a), pd.Timestamp(b), f"{t}: {str(m[r][s_]).title()} ({s_})") for r, s_, a, b, t in cases]
        fig_traces(cases)
