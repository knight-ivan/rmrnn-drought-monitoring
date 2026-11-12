"""Write every LaTeX table of the manuscript and supplement directly from the
result CSVs (no hand transcription). Output: tex/tables/*.tex and
tex/tables/numbers.tex (\\newcommand macros for numbers quoted in the text)."""
import os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, REGIONS, load_region

T = os.path.join(ROOT, "tex", "tables")
RN = {"texas": "Central Texas", "germany": "Germany", "australia": "SE Australia", "taiwan": "Taiwan", "all": "All regions"}
DN = {"D8": "D8 RM defect, multivariate", "D9": r"D9 defect, multivariate, $\lambda=0$",
      "D10": "D10 SPEI-30 (daily)", "D11": "D11 SPEI-90 (daily)", "D12": "D12 forecast residual, mv",
      "D1": "D1 RM defect", "D2": r"D2 defect, $\lambda=0$", "D3": "D3 SPI-90 (daily)", "D4": "D4 SPI-3 (monthly)",
      "D5": "D5 forecast residual", "D6": "D6 D1+D3 combined$^{\\dagger}$", "D7": "D7 SPI-30 (daily)$^{\\dagger}$"}
MACROS = []


_DIG = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))


def mac(name, val):
    """LaTeX macro names must be letters only: digits are spelled out (D1 -> Done)."""
    name = "".join(_DIG.get(c, c) for c in name if c.isalnum())
    if isinstance(val, str) and re.fullmatch(r"[+-]?[0-9.]+", val):
        val = f"\\ensuremath{{{val}}}"
    MACROS.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def mm(body):
    """Typeset negative numbers in table bodies with a math minus sign."""
    return re.sub(r"(?<![\w$\\{\-])-(?=\d)", "$-$", body)


def pfmt(p):
    return "$<10^{-4}$" if p < 1e-4 else f"{p:.4f}"


def f2(x, d=2):
    return f"{x:.{d}f}"


def ci(e, lo, hi, d=2, sign=False):
    fmt = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{fmt.format(e)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def data_table():
    rows = []
    ev = {}
    for r in REGIONS:
        m, P = load_region(r)
        e = pd.read_csv(os.path.join(ROOT, "results", "panel", f"{r}_events.csv"))
        e = e[pd.to_datetime(e.onset_date).dt.year >= 2001]
        ev[r] = len(e)
        miss = float(P[(P.index.year >= 1961)].isna().mean().mean())
        yr = P.resample("YE")
        ann = float((yr.sum() / yr.count().clip(lower=1) * 365.25).mean().median())   # observed-day mean x 365.25
        src = "CWA (Raingel)" if r == "taiwan" else "GHCN-Daily"
        rows.append(f"{RN[r]} & {src} & {len(m)} & {ann:.0f} & {100*miss:.1f} & {len(e)} & {np.median(e.duration):.0f} \\\\")
        mac(f"nst{r}", len(m)); mac(f"nev{r}", len(e))
    tot_st = sum(len(load_region(r)[0]) for r in REGIONS)
    mac("nstall", tot_st); mac("nevall", sum(ev.values()))
    tex = r"""\begin{table}[t]
\caption{Station networks and reference drought events. Annual precipitation is the median over
stations of the mean annual total (1961--2024; each year's mean over observed days times 365.25); missing is the mean percentage of missing days;
events are soil-moisture droughts with onset in 2001--2024.}\label{tab:data}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llrrrrr}
\toprule
Region & Source & Stations & Annual $P$ (mm) & Missing (\%) & Events & Duration (d)\\
\midrule
""" + "\n".join(rows) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "data.tex"), "w").write(tex)


def forecast_table():
    S = pd.read_csv(os.path.join(ROOT, "results", "forecast", "summary.csv"))
    P = pd.read_csv(os.path.join(ROOT, "results", "forecast", "paired.csv"))
    lines, plines = [], []
    for g in REGIONS + ["all"]:
        for j, (mod, lab) in enumerate([("rmrnn", "RMRNN"), ("lam0", r"$\lambda=0$"), ("gru", "GRU")]):
            s = S[(S.region == g) & (S.model == mod)].iloc[0]
            lines.append(f"{RN[g] if j == 0 else ''} & {lab} & {s.mse_mean:.3f} & {s.msss_mean:.3f} & {s.mae_mean:.3f} & {s.auc_mean:.3f} & {s.qpath_mean:.1f} \\\\")
        if g != "all":
            lines.append(r"\midrule")
        p0 = P[(P.group == g) & (P.comp == "rmrnn-lam0") & (P.metric == "mse")].iloc[0]
        p1 = P[(P.group == g) & (P.comp == "rmrnn-gru") & (P.metric == "mse")].iloc[0]
        q0 = P[(P.group == g) & (P.comp == "rmrnn-lam0") & (P.metric == "qpath")].iloc[0]
        plines.append(f"{RN[g]} & {int(p0.n)} & {100*p0.mean_rel_diff:+.2f} [{100*p0.ci90_lo:+.2f}, {100*p0.ci90_hi:+.2f}] & "
                      f"{100*p1.mean_rel_diff:+.2f} [{100*p1.ci90_lo:+.2f}, {100*p1.ci90_hi:+.2f}] & "
                      f"{100*q0.median_rel_diff:+.1f} & {pfmt(q0.wilcoxon_p)} \\\\")
        if g == "all":
            mac("qpathred", f"{-100*q0.median_rel_diff:.1f}")
            mac("msediffloall", f"{100*p0.ci90_lo:+.2f}"); mac("msediffhiall", f"{100*p0.ci90_hi:+.2f}")
            mac("msediffgruloall", f"{100*p1.ci90_lo:+.2f}"); mac("msediffgruhiall", f"{100*p1.ci90_hi:+.2f}")
            mac("tostmax", f"{P[(P.metric == 'mse')].tost_p.max():.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}")
    ms = S[S.model == "rmrnn"]
    mac("msssmin", f"{ms[ms.region != 'all'].msss_mean.min():.3f}"); mac("msssmax", f"{ms[ms.region != 'all'].msss_mean.max():.3f}")
    tex = r"""\begin{table}[t]
\caption{Next-day forecast accuracy on 2001--2024 (station means over five seeds, then averaged
over stations). MSE and MAE refer to standardised $\log(1+P)$; MSESS is the MSE skill score relative
to day-of-year climatology; AUC is for exceeding the station's 1961--1990 95th percentile;
$Q_{\mathrm{path}}$ is the mean hidden-state path increment over a 30-day window.}
\label{tab:forecast}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llrrrrr}
\toprule
Region & Model & MSE & MSESS & MAE & AUC & $Q_{\mathrm{path}}$\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}

\begin{table}[t]
\caption{Paired station-level comparisons of RMRNN. MSE differences are mean relative differences (\%)
with 90\% confidence intervals, which coincide with the TOST equivalence test at level 0.05: an interval
inside $\pm2\%$ establishes equivalence. $Q_{\mathrm{path}}$: median relative change (\%) against the
$\lambda=0$ ablation with its Wilcoxon signed-rank $p$-value.}
\label{tab:forecastpaired}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrccrr}
\toprule
Region & $n$ & $\Delta$MSE vs $\lambda=0$ (\%) & $\Delta$MSE vs GRU (\%) & $\Delta Q_{\mathrm{path}}$ (\%) & $p$\\
\midrule
""" + mm("\n".join(plines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "forecast.tex"), "w").write(tex)


def warning_table(budget=1.0, fname="warning.tex", label="tab:warning", dets=("D1", "D2", "D3", "D4", "D5", "D7", "D6")):
    S = pd.read_csv(os.path.join(ROOT, "results", "warning", "summary.csv"))
    Th = pd.read_csv(os.path.join(ROOT, "results", "warning", "thresholds.csv"))
    S = S[S.budget == budget]
    lines = []
    for g in REGIONS + ["all"]:
        n = int(S[(S.group == g)].n_events.iloc[0])
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{RN[g]}}} ({n} events)}}\\\\")
        for d in dets:
            s = S[(S.group == g) & (S.det == d)].iloc[0]
            flag = ""
            if g != "all":
                t = Th[(Th.region == g) & (Th.det == d) & (Th.budget == budget)].iloc[0]
                flag = "" if t.attained else "$^{\\ast}$"
            tim = "--" if not np.isfinite(s.timing_med) else f"{s.timing_med:+.0f} [{s.timing_lo:+.0f}, {s.timing_hi:+.0f}]"
            lines.append(f"\\quad {DN[d]}{flag} & {ci(s.pod, s.pod_lo, s.pod_hi)} & {ci(s.far, s.far_lo, s.far_hi)} & {tim} \\\\")
        if g != "all":
            lines.append(r"\midrule")
        if budget == 1.0:
            for d in dets:
                s = S[(S.group == g) & (S.det == d)].iloc[0]
                mac(f"pod{g}{d}", f"{s.pod:.3f}"); mac(f"far{g}{d}", f"{s.far:.2f}")
                if np.isfinite(s.timing_med):
                    mac(f"tim{g}{d}", f"{s.timing_med:+.0f}")
    timing_hdr = "Median timing (d)" if budget == 1.0 else ""
    tex = r"""\begin{table}[t]
\caption{Event-based warning skill on 2001--2024 at a matched budget of """ + f"{budget:g}" + r""" alarm(s) per
station-year (calibrated on 1991--2000). POD: probability of detection within $[-60,+120]$ days of a
reference onset; FAR: false alarms per station-year of non-drought exposure; timing: median of
(alarm $-$ onset) among hits, negative values meaning lead. Brackets: 95\% cluster-bootstrap intervals
(calendar years resampled within region). $^{\dagger}$Post hoc (not pre-specified).
$^{\ast}$Budget not attained within 10\% in calibration (closest attainable rate used).}
\label{""" + label + r"""}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccc}
\toprule
Detector & POD & FAR (per yr) & """ + timing_hdr + r"""\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, fname), "w").write(tex)


def paired_table():
    P = pd.read_csv(os.path.join(ROOT, "results", "warning", "paired.csv"))
    PT = pd.read_csv(os.path.join(ROOT, "results", "warning", "paired_timing.csv"))
    P = P[P.budget == 1.0]
    comps = [("D1-D2", "D2 minus D1", r"D1 vs D2 ($\lambda=0$)"), ("D1-D3", "D3 minus D1", "D1 vs D3 (SPI-90)"),
             ("D1-D4", "D4 minus D1", "D1 vs D4 (SPI-3)"), ("D1-D7", "D7 minus D1", r"D1 vs D7 (SPI-30)$^{\dagger}$"),
             ("D7-D3", "D3 minus D7", r"D7 vs D3$^{\dagger}$"), ("D6-D3", "D3 minus D6", r"D6 vs D3$^{\dagger}$")]
    lines = []
    for g in REGIONS + ["all"]:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{RN[g]}}}}}\\\\")
        for c, ct, lab in comps:
            p = P[(P.group == g) & (P.comp == c)].iloc[0]
            t = PT[(PT.group == g) & (PT.comp == ct)].iloc[0]
            lines.append(f"\\quad {lab} & {ci(p.dpod, p.dpod_lo, p.dpod_hi, 3, True)} & {ci(p.dfar, p.dfar_lo, p.dfar_hi, 2, True)} & "
                         f"{t.median_diff:+.1f} [{t.lo:+.1f}, {t.hi:+.1f}] & {int(t.n_both)} \\\\")
            key = c.replace("-", "")
            mac(f"dpod{g}{key}", f"{p.dpod:+.3f}"); mac(f"dpodlo{g}{key}", f"{p.dpod_lo:+.3f}"); mac(f"dpodhi{g}{key}", f"{p.dpod_hi:+.3f}")
            mac(f"dfar{g}{key}", f"{p.dfar:+.2f}"); mac(f"dfarlo{g}{key}", f"{p.dfar_lo:+.2f}"); mac(f"dfarhi{g}{key}", f"{p.dfar_hi:+.2f}")
            mac(f"dtim{g}{key}", f"{t.median_diff:.1f}".rstrip("0").rstrip(".")); mac(f"dtimlo{g}{key}", f"{t.lo:.1f}".rstrip("0").rstrip(".")); mac(f"dtimhi{g}{key}", f"{t.hi:.1f}".rstrip("0").rstrip("."))
            mac(f"fearly{g}{key}", f"{100*t.frac_first_earlier:.0f}")
        if g != "all":
            lines.append(r"\midrule")
    tex = r"""\begin{table}[t]
\caption{Paired comparisons at the primary budget (1 alarm per station-year). $\Delta$POD and
$\Delta$FAR are first minus second detector; $\Delta$timing is the median, over events detected by both,
of (second detector's alarm $-$ first detector's alarm), so positive values mean the first detector
alarms earlier. Brackets: 95\% cluster-bootstrap intervals. $^{\dagger}$Post hoc.}
\label{tab:paired}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccr}
\toprule
Comparison & $\Delta$POD & $\Delta$FAR (per yr) & $\Delta$timing (d) & $n$ both\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "paired.tex"), "w").write(tex)


def covariate_tables():
    A = pd.read_csv(os.path.join(ROOT, "results", "covariates", "association.csv"))
    A = A[A.spec == "full"]
    terms = [("spi3", "SPI-3"), ("sm_a", "Soil wetness"), ("t2m_a", "Temperature"), ("et_a", "Evapotranspiration")]
    lines = []
    for g in REGIONS + ["all"]:
        cells = []
        for y in ["D1", "D2"]:
            a = A[(A.group == g) & (A.outcome == y)].set_index("term")
            for t, _ in terms:
                cells.append(f"{a.loc[t].coef:+.2f}" + ("$^{\\ast}$" if a.loc[t].p < 0.05 else ""))
            cells.append(f"{a.r2_within.dropna().iloc[0]:.2f}")
            if y == "D1":
                for t, _ in terms:
                    mac(f"assoc{g}{t.replace('_', '')}", f"{a.loc[t].coef:+.2f}")
                mac(f"assocrtwo{g}", f"{a.r2_within.dropna().iloc[0]:.2f}")
        lines.append(f"{RN[g]} & " + " & ".join(cells) + " \\\\")
    tex = r"""\begin{table}[t]
\caption{Monthly association (1981--2024) between the defect anomaly and concurrent
hydrometeorological anomalies: standardised coefficients from regressions with station fixed effects;
$^{\ast}p<0.05$ with standard errors clustered by year-month; $R^2_w$ is the within-station $R^2$.}
\label{tab:assoc}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrrrrrr}
\toprule
& \multicolumn{5}{c}{D1 RM defect} & \multicolumn{5}{c}{D2 defect, $\lambda=0$}\\
\cmidrule{2-6}\cmidrule{7-11}
Region & SPI-3 & SM & T2m & ET & $R^2_w$ & SPI-3 & SM & T2m & ET & $R^2_w$\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "assoc.tex"), "w").write(tex)
    I = pd.read_csv(os.path.join(ROOT, "results", "covariates", "incremental.csv"))
    lab = {"D1": "SPI-90 + D1", "D2": "SPI-90 + D2", "D5": "SPI-90 + D5", "SPI30": "SPI-90 + SPI-30$^{\\dagger}$",
           "SPI30+D1": "SPI-90 + SPI-30 + D1$^{\\dagger}$"}
    lines = []
    for g in REGIONS + ["all"]:
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{RN[g]}}} (base AUC {I[(I.group == g) & (I.reference == 'base')].auc_base.iloc[0]:.3f}, prevalence {I[I.group == g].prevalence.iloc[0]:.3f})}}\\\\")
        for _, r in I[I.group == g].iterrows():
            ref = "SPI-90" if r.reference == "base" else "SPI-90 + SPI-30"
            nd = 4 if max(abs(r.dauc), abs(r.dauc_lo), abs(r.dauc_hi)) < 0.005 else 3
            lines.append(f"\\quad {lab[r.added]} & {ref} & {r.auc_model:.3f} & {r.dauc:+.{nd}f} [{r.dauc_lo:+.{nd}f}, {r.dauc_hi:+.{nd}f}] \\\\")
            nd = 4 if max(abs(r.dauc), abs(r.dauc_lo), abs(r.dauc_hi)) < 0.005 else 3
            key = r.added.replace('+', 'p').replace('30', 'thirty')
            mac(f"dauc{g}{key}", f"{r.dauc:+.{nd}f}")
            mac(f"dauclo{g}{key}", f"{r.dauc_lo:+.{nd}f}")
            mac(f"dauchi{g}{key}", f"{r.dauc_hi:+.{nd}f}")
        if g != "all":
            lines.append(r"\midrule")
    tex = r"""\begin{table}[t]
\caption{Incremental information about drought onset within 60 days (weekly logistic models,
five-fold cross-validation over blocks of years, 1982--2024). $\Delta$AUC is relative to the reference
model; brackets are 95\% region-year cluster-bootstrap intervals. Predictors other than SPI are
30-day means. $^{\dagger}$Post hoc.}
\label{tab:incremental}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llcc}
\toprule
Model & Reference & AUC & $\Delta$AUC\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "incremental.tex"), "w").write(tex)


def threshold_table():
    Th = pd.read_csv(os.path.join(ROOT, "results", "warning", "thresholds.csv"))
    lines = []
    for r in REGIONS:
        for d in ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]:
            cells = []
            for b in [0.5, 1.0, 2.0]:
                t = Th[(Th.region == r) & (Th.det == d) & (Th.budget == b)].iloc[0]
                cells.append(f"{t.thr:.2f} & {t.achieved_cal_rate:.2f}" + ("" if t.attained else "$^{\\ast}$"))
            lines.append(f"{RN[r] if d == 'D1' else ''} & {DN[d]} & " + " & ".join(cells) + " \\\\")
        if r != REGIONS[-1]:
            lines.append(r"\midrule")
    tex = r"""\begin{table}[h]
\caption{Calibrated thresholds ($h$ for CUSUM detectors, $c$ for the SPI-3 rule) and the alarm rate
they attain in 1991--2000 (alarms per station-year) for each budget. $^{\ast}$Not within 10\% of the
budget. $^{\dagger}$Post hoc.}\label{tab:thresholds}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llrrrrrr}
\toprule
& & \multicolumn{2}{c}{$\beta=0.5$} & \multicolumn{2}{c}{$\beta=1$} & \multicolumn{2}{c}{$\beta=2$}\\
Region & Detector & thr. & rate & thr. & rate & thr. & rate\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "thresholds.tex"), "w").write(tex)


def mv_tables():
    """Tables and macros for PROTOCOL amendment 1 (multivariate subset)."""
    W = os.path.join(ROOT, "results", "warning_mv")
    if not os.path.exists(os.path.join(W, "summary.csv")):
        return
    S = pd.read_csv(os.path.join(W, "summary.csv")); S = S[S.budget == 1.0]
    Th = pd.read_csv(os.path.join(W, "thresholds.csv"))
    dets = ("D8", "D9", "D11", "D10", "D1", "D3", "D7")
    lines = []
    for g in REGIONS + ["all"]:
        n = int(S[S.group == g].n_events.iloc[0])
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{RN[g]}}} ({n} events)}}\\\\")
        for d in dets:
            x = S[(S.group == g) & (S.det == d)].iloc[0]
            flag = ""
            if g != "all":
                t = Th[(Th.region == g) & (Th.det == d) & (Th.budget == 1.0)].iloc[0]
                flag = "" if t.attained else "$^{\\ast}$"
            tim = "--" if not np.isfinite(x.timing_med) else f"{x.timing_med:+.0f} [{x.timing_lo:+.0f}, {x.timing_hi:+.0f}]"
            lines.append(f"\\quad {DN[d]}{flag} & {ci(x.pod, x.pod_lo, x.pod_hi)} & {ci(x.far, x.far_lo, x.far_hi)} & {tim} \\\\")
            mac(f"mvpod{g}{d}", f"{x.pod:.3f}"); mac(f"mvfar{g}{d}", f"{x.far:.2f}")
        mac(f"mvnev{g}", n)
        if g != "all":
            lines.append(r"\midrule")
    nst = sum(len(np.load(os.path.join(ROOT, "results", "panel_mv", f"{r}.npz"))["station_id"]) for r in REGIONS)
    mac("mvnst", nst)
    for r in REGIONS:
        mac(f"mvnst{r}", len(np.load(os.path.join(ROOT, "results", "panel_mv", f"{r}.npz"))["station_id"]))
    tex = r"""\begin{table}[t]
\caption{Temperature-aware analysis (Section~\ref{sec:mv}; """ + str(nst) + r""" stations with daily temperature):
event-based warning skill on 2001--2024 at one alarm per station-year. Definitions as in
Table~\ref{tab:warning}. $^{\ast}$Budget not attained within 10\% in calibration.}
\label{tab:warningmv}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccc}
\toprule
Detector & POD & FAR (per yr) & Median timing (d)\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "warning_mv.tex"), "w").write(tex)
    P = pd.read_csv(os.path.join(W, "paired.csv")); P = P[P.budget == 1.0]
    PT = pd.read_csv(os.path.join(W, "paired_timing.csv"))
    comps = [("D8-D11", "D11 minus D8", "D8 vs D11 (SPEI-90)"), ("D8-D10", "D10 minus D8", "D8 vs D10 (SPEI-30)"),
             ("D8-D9", "D9 minus D8", r"D8 vs D9 ($\lambda=0$)"), ("D8-D1", "D1 minus D8", "D8 vs D1 (P only)"),
             ("D10-D11", "D11 minus D10", "D10 vs D11")]
    lines = []
    for g in REGIONS + ["all"]:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{RN[g]}}}}}\\\\")
        for c, ct, lab in comps:
            x = P[(P.group == g) & (P.comp == c)].iloc[0]
            t = PT[(PT.group == g) & (PT.comp == ct)].iloc[0]
            lines.append(f"\\quad {lab} & {ci(x.dpod, x.dpod_lo, x.dpod_hi, 3, True)} & {ci(x.dfar, x.dfar_lo, x.dfar_hi, 2, True)} & "
                         f"{t.median_diff:+.1f} [{t.lo:+.1f}, {t.hi:+.1f}] & {int(t.n_both)} \\\\")
            key = "mv" + c.replace("-", "")
            mac(f"dpod{g}{key}", f"{x.dpod:+.3f}"); mac(f"dpodlo{g}{key}", f"{x.dpod_lo:+.3f}"); mac(f"dpodhi{g}{key}", f"{x.dpod_hi:+.3f}")
            mac(f"dfar{g}{key}", f"{x.dfar:+.2f}"); mac(f"dfarlo{g}{key}", f"{x.dfar_lo:+.2f}"); mac(f"dfarhi{g}{key}", f"{x.dfar_hi:+.2f}")
            mac(f"dtim{g}{key}", f"{t.median_diff:.1f}".rstrip("0").rstrip("."))
            mac(f"dtimlo{g}{key}", f"{t.lo:.1f}".rstrip("0").rstrip(".")); mac(f"dtimhi{g}{key}", f"{t.hi:.1f}".rstrip("0").rstrip("."))
        if g != "all":
            lines.append(r"\midrule")
    tex = r"""\begin{table}[t]
\caption{Temperature-aware analysis: paired comparisons at one alarm per station-year (definitions as in
Table~\ref{tab:paired}; positive $\Delta$timing means the first detector alarms earlier).}
\label{tab:pairedmv}
\footnotesize\setlength{\tabcolsep}{2.5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccr}
\toprule
Comparison & $\Delta$POD & $\Delta$FAR (per yr) & $\Delta$timing (d) & $n$ both\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "paired_mv.tex"), "w").write(tex)
    C = os.path.join(ROOT, "results", "covariates_mv")
    if os.path.exists(os.path.join(C, "incremental.csv")):
        I = pd.read_csv(os.path.join(C, "incremental.csv"))
        lab = {"spei90+D8": "SPEI-90 + D8", "base+D8": "base + D8", "base+D9": "base + D9", "base+D1": "base + D1"}
        rl = {"spei90": "SPEI-90", "base": "base"}
        lines = []
        for g in REGIONS + ["all"]:
            lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{RN[g]}}}}}\\\\")
            for _, x in I[I.group == g].iterrows():
                nd = 4 if max(abs(x.dauc), abs(x.dauc_lo), abs(x.dauc_hi)) < 0.005 else 3
                lines.append(f"\\quad {lab[x.model]} & {rl[x.reference]} ({x.auc_ref:.3f}) & {x.auc_model:.3f} & "
                             f"{x.dauc:+.{nd}f} [{x.dauc_lo:+.{nd}f}, {x.dauc_hi:+.{nd}f}] \\\\")
                key = "mvdauc" + g + x.model.replace("+", "p").replace("90", "ninety")
                mac(key, f"{x.dauc:+.{nd}f}"); mac(key + "lo", f"{x.dauc_lo:+.{nd}f}"); mac(key + "hi", f"{x.dauc_hi:+.{nd}f}")
            if g != "all":
                lines.append(r"\midrule")
        tex = r"""\begin{table}[t]
\caption{Temperature-aware analysis: incremental information about onset within 60 days. ``base'' =
SPEI-90 + SPEI-30 + SPI-30; predictors other than the indices are 30-day means. AUCs from five-fold
cross-validation over blocks of years; brackets are 95\% region-year cluster-bootstrap intervals.}
\label{tab:incrementalmv}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llcc}
\toprule
Model & Reference (AUC) & AUC & $\Delta$AUC\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
        open(os.path.join(T, "incremental_mv.tex"), "w").write(tex)
    if os.path.exists(os.path.join(C, "association.csv")):
        A = pd.read_csv(os.path.join(C, "association.csv"))
        for g in REGIONS + ["all"]:
            a = A[(A.group == g) & (A.outcome == "D8")].set_index("term")
            for t in ["spi3", "sm_a", "t2m_a", "et_a"]:
                mac(f"mvassoc{g}{t}", f"{a.loc[t].coef:+.2f}"); mac(f"mvassocp{g}{t}", "yes" if a.loc[t].p < 0.05 else "no")
            mac(f"mvassocrtwo{g}", f"{a.r2_within.iloc[0]:.2f}")
        lines = []
        terms = ["spi3", "sm_a", "t2m_a", "et_a"]
        for g in REGIONS + ["all"]:
            cells = []
            for y in ["D8", "D9"]:
                a = A[(A.group == g) & (A.outcome == y)].set_index("term")
                cells += [f"{a.loc[t].coef:+.2f}" + ("$^{\\ast}$" if a.loc[t].p < 0.05 else "") for t in terms]
                cells.append(f"{a.r2_within.iloc[0]:.2f}")
            lines.append(f"{RN[g]} & " + " & ".join(cells) + " \\\\")
        tex = r"""\begin{table}[t]
\caption{Temperature-aware analysis: monthly association of the multivariate defect anomaly with
concurrent anomalies (as Table~\ref{M-tab:assoc} of the main paper). $^{\ast}p<0.05$.}
\label{tab:assocmv}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrrrrrr}
\toprule
& \multicolumn{5}{c}{D8 RM defect, multivariate} & \multicolumn{5}{c}{D9 defect, multivariate, $\lambda=0$}\\
\cmidrule{2-6}\cmidrule{7-11}
Region & SPI-3 & SM & T2m & ET & $R^2_w$ & SPI-3 & SM & T2m & ET & $R^2_w$\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
        open(os.path.join(T, "assoc_mv.tex"), "w").write(tex)

    D = os.path.join(ROOT, "results", "warning_mv", "drift.csv")
    if os.path.exists(D):
        Dd = pd.read_csv(D)
        lab = {"D1": "D1 defect (P only)", "D8": "D8 defect (temperature-aware)", "D9": r"D9 ablation ($\lambda=0$)",
               "D10": "SPEI-30", "D11": "SPEI-90", "t2m_a": "2-m temperature anomaly"}
        lines = []
        for r in REGIONS:
            for j, (_, x) in enumerate(Dd[Dd.region == r].iterrows()):
                lines.append(f"{RN[r] if j == 0 else ''} & {lab[x.score]} & {x.mean_1991_2000:+.2f} & {x.mean_2001_2024:+.2f} [{x.lo:+.2f}, {x.hi:+.2f}] \\\\")
            if r != REGIONS[-1]:
                lines.append(r"\midrule")
        tex = r"""\begin{table}[h]
\caption{Descriptive (post hoc) drift check: mean standardised anomaly relative to the 1961--1990
reference, averaged over stations, in 1991--2000 and 2001--2024 (95\% year-bootstrap interval).
Temperature anomalies are relative to 1981--2010 (MERRA-2).}\label{tab:drift}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llcc}
\toprule
Region & Score & 1991--2000 & 2001--2024\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
        open(os.path.join(T, "drift_mv.tex"), "w").write(tex)
    F = os.path.join(ROOT, "results", "forecast_mv")
    if os.path.exists(os.path.join(F, "paired.csv")):
        S2 = pd.read_csv(os.path.join(F, "summary.csv")); P2 = pd.read_csv(os.path.join(F, "paired.csv"))
        lines = []
        for g in REGIONS + ["all"]:
            p0 = P2[(P2.group == g) & (P2.comp == "rmrnn-lam0") & (P2.metric == "mse")].iloc[0]
            p1 = P2[(P2.group == g) & (P2.comp == "rmrnn-gru") & (P2.metric == "mse")].iloc[0]
            q0 = P2[(P2.group == g) & (P2.comp == "rmrnn-lam0") & (P2.metric == "qpath")].iloc[0]
            s0 = S2[(S2.region == g) & (S2.model == "rmrnn")].iloc[0]
            lines.append(f"{RN[g]} & {int(p0.n)} & {s0.msss_mean:.3f} & {100*p0.mean_rel_diff:+.2f} [{100*p0.ci90_lo:+.2f}, {100*p0.ci90_hi:+.2f}] & "
                         f"{100*p1.mean_rel_diff:+.2f} [{100*p1.ci90_lo:+.2f}, {100*p1.ci90_hi:+.2f}] & {100*q0.median_rel_diff:+.1f} \\\\")
            if g == "all":
                mac("mvqpathred", f"{-100*q0.median_rel_diff:.1f}")
                mac("mvmselo", f"{100*p0.ci90_lo:+.2f}"); mac("mvmsehi", f"{100*p0.ci90_hi:+.2f}")
                mac("mvmsegrulo", f"{100*p1.ci90_lo:+.2f}"); mac("mvmsegruhi", f"{100*p1.ci90_hi:+.2f}")
                mac("mvmsss", f"{s0.msss_mean:.3f}")
        tex = r"""\begin{table}[t]
\caption{Temperature-aware networks: next-day precipitation forecasts on 2001--2024 (MSESS of RMRNN;
paired relative MSE differences with 90\% CIs; median change in $Q_{\mathrm{path}}$ against the
$\lambda=0$ ablation).}
\label{tab:forecastmv}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrcccr}
\toprule
Region & $n$ & MSESS & $\Delta$MSE vs $\lambda=0$ (\%) & $\Delta$MSE vs GRU (\%) & $\Delta Q_{\mathrm{path}}$ (\%)\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
        open(os.path.join(T, "forecast_mv.tex"), "w").write(tex)


def check_macros():
    """Macros for descriptive statements in the text (results/checks/, written by text_checks.py).
    The assertions stop the build if a sentence in the paper would no longer match the results."""
    C = os.path.join(ROOT, "results", "checks")
    if not os.path.exists(os.path.join(C, "composites.csv")):
        return
    f1 = lambda v: f"{v:.1f}"; f2 = lambda v: f"{v:.2f}"
    rng = lambda name, s, fmt: (mac(name + "lo", fmt(s.min())), mac(name + "hi", fmt(s.max())))
    # event-aligned composites (Fig. composites, Fig. compositesmv)
    K = pd.read_csv(os.path.join(C, "composites.csv"))
    P = K[(K.subset == "panel") & (K.region != "taiwan")]
    g = lambda d: P[P.det == d]
    rng("compDonelag", g("D1").min_lag, str); rng("compDthreelag", g("D3").min_lag, str)
    rng("compDsevenlag", g("D7").min_lag, str)
    rng("compDonemin", g("D1").min_value, f2); rng("compDsevenmin", g("D7").min_value, f2)
    assert (g("D1").min_lag.abs() <= 7).all() and (g("D3").min_lag >= 28).all()
    assert (g("D7").min_value.values < g("D1").min_value.values).all()
    assert (g("D1")["v-30"].values < g("D1")["v-60"].values).all()
    tw = K[(K.subset == "panel") & (K.region == "taiwan") & (K.det == "D1")].iloc[0]
    mac("compDonemintaiwan", f2(tw.min_value)); mac("compDonelagtaiwan", str(tw.min_lag))
    mac("compDonepostaiwan", f2(tw["v+30"]))
    M = K[K.subset == "panel_mv"]
    rng("compmvDeightmin", M[M.det == "D8"].min_value, f2); rng("compmvDonemin", M[M.det == "D1"].min_value, f2)
    assert M[M.det == "D8"].min_value.min() > M[M.det == "D1"].min_value.max()
    # operating curves (Fig. amoc)
    A = pd.read_csv(os.path.join(C, "amoc_compare.csv")).pivot_table(index=["region", "far"], columns="det", values="pod")
    mid = A[A.index.get_level_values("far") >= 0.5]
    assert ((mid.D3 >= mid.D1) & (mid.D7 >= mid.D1)).all()
    mac("amocDoneDtwomaxdiff", f2((A.D1 - A.D2).abs().max()))
    # station traces (Fig. traces)
    R = pd.read_csv(os.path.join(C, "trace_alarms.csv")).pivot_table(index=["region", "onset"], columns="det",
                                                                        values="first_alarm_lag")
    ok = R.drop(index="taiwan", level="region")
    assert ok.D1.notna().all() and (ok.D1 <= ok.D3).all()
    rng("traceDonelag", ok.D1, lambda v: f"{v:.0f}"); rng("traceDthreeminusDone", ok.D3 - ok.D1, lambda v: f"{v:.0f}")
    assert (ok.D7 > ok.D1).all()
    rng("traceDsevenminusDone", ok.D7 - ok.D1, lambda v: f"{v:.0f}")
    tr = R.loc["taiwan"].iloc[0]
    assert np.isnan(tr.D1) and tr.D7 < 0 < tr.D3     # text: SPI-30 before onset, SPI-90 after
    mac("traceDseventaiwan", f"{-tr.D7:.0f}"); mac("traceDthreetaiwan", f"{tr.D3:.0f}")
    # dependence: year-cluster vs naive binomial interval width
    W = pd.read_csv(os.path.join(C, "ci_width.csv")); W = W[W.det.isin(["D1", "D3"])]
    mac("ciratioall", f1(W[(W.group == "all") & (W.det == "D1")].ratio.iloc[0]))
    rng("ciratio", W[W.group != "all"].ratio, f1)
    # same-station forecast skill
    S = pd.read_csv(os.path.join(C, "msss_same.csv")).set_index("region")
    mac("mvmsssp", f"{S.loc['all'].msss_p:.3f}"); mac("mvmsssmv", f"{S.loc['all'].msss_mv:.3f}")
    mac("mvmsssfrac", f"{100*S.loc['all'].frac_improved:.0f}")
    assert (S.msss_mv > S.msss_p).all()
    # temperature donors
    D = pd.read_csv(os.path.join(C, "donors.csv")).set_index("region")
    for r in ["texas", "germany", "australia"]:
        mac(f"ndonor{r}", str(int(D.loc[r].n_donor))); mac(f"kmdonor{r}", f"{D.loc[r].median_donor_km:.0f}")
    # base AUC of SPI-90 alone (Table incremental), GRU gap in Australia, drift
    I = pd.read_csv(os.path.join(ROOT, "results", "covariates", "incremental.csv"))
    for r in REGIONS:
        mac(f"aucbase{r}", f2(I[(I.group == r) & (I.added == "D1")].auc_base.iloc[0]))
    FP = pd.read_csv(os.path.join(ROOT, "results", "forecast_mv", "paired.csv"))
    x = FP[(FP.group == "australia") & (FP.comp == "rmrnn-gru") & (FP.metric == "mse")].iloc[0]
    mac("mvgruaustralia", f1(100 * x.mean_rel_diff)); assert x.ci90_lo > 0
    Dr = pd.read_csv(os.path.join(ROOT, "results", "warning_mv", "drift.csv")).set_index(["region", "score"])
    d8 = Dr.xs("D8", level="score"); tt = Dr.xs("t2m_a", level="score")
    rng("driftDeight", d8.mean_2001_2024, f2); rng("driftt", tt.mean_2001_2024, f2)
    assert (d8.lo > 0).all()
    for r in ["texas", "germany"]:
        mac(f"driftDeightcal{r}", f2(d8.loc[r].mean_1991_2000)); mac(f"drifttcal{r}", f2(tt.loc[r].mean_1991_2000))


if __name__ == "__main__":
    os.makedirs(T, exist_ok=True)
    data_table(); forecast_table(); warning_table(); paired_table(); covariate_tables(); threshold_table()
    mv_tables()
    warning_table(0.5, "warning_b05.tex", "tab:warning05", ("D1", "D2", "D3", "D4", "D5", "D7"))
    warning_table(2.0, "warning_b2.tex", "tab:warning2", ("D1", "D2", "D3", "D4", "D5", "D7"))
    check_macros()
    open(os.path.join(T, "numbers.tex"), "w").write("% generated by code/serra/make_tables.py\n" + "\n".join(MACROS) + "\n")
    print(len(MACROS), "macros")
