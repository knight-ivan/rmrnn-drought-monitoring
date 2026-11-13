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
DN = {"D8": r"D8 defect (T), $\lambda>0$", "D9": r"D9 defect (T), $\lambda=0$",
      "D10": "D10 SPEI-30 (daily)", "D11": "D11 SPEI-90 (daily)", "D12": "D12 forecast residual, mv",
      "D1": r"D1 defect, $\lambda>0$", "D2": r"D2 defect, $\lambda=0$", "D3": "D3 SPI-90 (daily)", "D4": "D4 SPI-3 (monthly)",
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
    tex = r"""\begin{table}[tbp]
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
        for j, (mod, lab) in enumerate([("rmrnn", "Regularised"), ("lam0", r"$\lambda=0$"), ("gru", "GRU")]):
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
    tex = r"""\begin{table}[tbp]
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

\begin{table}[tbp]
\caption{Paired station-level comparisons of the regularised network. MSE differences are mean relative differences (\%)
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


def chance_ref(sub="warning"):
    """Random-timing reference POD at the pre-specified window (matching_checks.py)."""
    C = pd.read_csv(os.path.join(ROOT, "results", sub, "matching_chance.csv"))
    return C[(C.pre == 60) & (C.post == 120)].set_index(["group", "det"])


def warning_table(budget=1.0, fname="warning.tex", label="tab:warning", dets=("D1", "D2", "D3", "D4", "D5", "D7", "D6")):
    S = pd.read_csv(os.path.join(ROOT, "results", "warning", "summary.csv"))
    Th = pd.read_csv(os.path.join(ROOT, "results", "warning", "thresholds.csv"))
    S = S[S.budget == budget]
    prim = budget == 1.0
    C = chance_ref() if prim else None
    ncol = 5 if prim else 3
    lines = []
    for g in REGIONS + ["all"]:
        n = int(S[(S.group == g)].n_events.iloc[0])
        lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{RN[g]}}} ({n} events)}}\\\\")
        for d in dets:
            s = S[(S.group == g) & (S.det == d)].iloc[0]
            flag = ""
            if g != "all":
                t = Th[(Th.region == g) & (Th.det == d) & (Th.budget == budget)].iloc[0]
                flag = "" if t.attained else "$^{\\ast}$"
            wet = Th[(Th.det == "D4") & (Th.budget == budget) & (Th.thr > 0)].region.tolist()
            if d == "D4" and (g in wet or (g == "all" and wet)):
                flag += "$^{\\ddagger}$"
            if prim:
                tim = "--" if not np.isfinite(s.timing_med) else f"{s.timing_med:+.0f} [{s.timing_lo:+.0f}, {s.timing_hi:+.0f}]"
                c = C.loc[(g, d)]
                lines.append(f"\\quad {DN[d]}{flag} & {ci(s.pod, s.pod_lo, s.pod_hi)} & {c.pod_chance:.2f} & {ci(s.far, s.far_lo, s.far_hi)} & {tim} \\\\")
                mac(f"chance{g}{d}", f"{c.pod_chance:.3f}"); mac(f"chancelo{g}{d}", f"{c.lo:.3f}"); mac(f"chancehi{g}{d}", f"{c.hi:.3f}")
            else:
                lines.append(f"\\quad {DN[d]}{flag} & {ci(s.pod, s.pod_lo, s.pod_hi)} & {ci(s.far, s.far_lo, s.far_hi)} \\\\")
        if g != "all":
            lines.append(r"\midrule")
        if budget == 1.0:
            for d in dets:
                s = S[(S.group == g) & (S.det == d)].iloc[0]
                mac(f"pod{g}{d}", f"{s.pod:.3f}"); mac(f"far{g}{d}", f"{s.far:.2f}")
                if np.isfinite(s.timing_med):
                    mac(f"tim{g}{d}", f"{s.timing_med:+.0f}")
    bud = "1 alarm" if prim else (f"{budget:g} alarms")
    extra = (r"""; timing: median of
(alarm $-$ onset) among hits, negative values meaning lead; random: POD of the same alarms after random
circular shifts within 2001--2024 (post hoc reference, Section~\ref{sec:scoring}); D1 ($\lambda>0$) is the
regularised defect and D2 ($\lambda=0$) its ablation""" if prim else "")
    hdr = (r"Detector & POD & Random & FAR (per yr) & Timing (d)\\" if prim
           else r"Detector & POD & FAR (per yr)\\")
    tex = r"""\begin{table}[tbp]
\caption{Event-based warning skill on 2001--2024 at a matched budget of """ + bud + r""" per
station-year (calibrated on 1991--2000). POD: probability of detection within $[-60,+120]$ days of a
reference onset; FAR: false alarms per station-year of non-drought exposure""" + extra + r""". Brackets: 95\%
cluster-bootstrap intervals (calendar years resampled within region). $^{\dagger}$Post hoc (not
pre-specified). $^{\ast}$Budget not attained within 10\% in calibration (closest attainable rate used).
$^{\ddagger}$The calibrated SPI-3 threshold is positive (wet side) in this region, or in a region pooled
here, so the entry is not informative about the index (dry-side calibration: Online Resource 1,
Table~""" + ("\\ref{S-tab:spithree}" if prim else "\\ref{tab:spithree}") + r""").}
\label{""" + label + r"""}
\footnotesize\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}l""" + ("cccc" if prim else "cc") + r"""}
\toprule
""" + hdr + r"""
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
    Th = pd.read_csv(os.path.join(ROOT, "results", "warning", "thresholds.csv"))
    WET = Th[(Th.det == "D4") & (Th.budget == 1.0) & (Th.thr > 0)].region.tolist()
    WET = WET + (["all"] if WET else [])
    comps = [("D1-D2", "D2 minus D1", r"D1 vs D2 ($\lambda=0$)"), ("D1-D3", "D3 minus D1", "D1 vs D3 (SPI-90)"),
             ("D1-D4", "D4 minus D1", "D1 vs D4 (SPI-3)"), ("D1-D7", "D7 minus D1", r"D1 vs D7 (SPI-30)$^{\dagger}$"),
             ("D7-D3", "D3 minus D7", r"D7 vs D3$^{\dagger}$"), ("D6-D3", "D3 minus D6", r"D6 vs D3$^{\dagger}$")]
    lines = []
    for g in REGIONS + ["all"]:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{RN[g]}}}}}\\\\")
        for c, ct, lab in comps:
            p = P[(P.group == g) & (P.comp == c)].iloc[0]
            t = PT[(PT.group == g) & (PT.comp == ct)].iloc[0]
            wetf = "$^{\\ddagger}$" if (c == "D1-D4" and g in WET) else ""
            lines.append(f"\\quad {lab}{wetf} & {ci(p.dpod, p.dpod_lo, p.dpod_hi, 3, True)} & {ci(p.dfar, p.dfar_lo, p.dfar_hi, 2, True)} & "
                         f"{t.median_diff:+.1f} [{t.lo:+.1f}, {t.hi:+.1f}] & {int(t.n_both)} \\\\")
            key = c.replace("-", "")
            mac(f"dpod{g}{key}", f"{p.dpod:+.3f}"); mac(f"dpodlo{g}{key}", f"{p.dpod_lo:+.3f}"); mac(f"dpodhi{g}{key}", f"{p.dpod_hi:+.3f}")
            mac(f"dfar{g}{key}", f"{p.dfar:+.2f}"); mac(f"dfarlo{g}{key}", f"{p.dfar_lo:+.2f}"); mac(f"dfarhi{g}{key}", f"{p.dfar_hi:+.2f}")
            mac(f"dtim{g}{key}", f"{t.median_diff:.1f}".rstrip("0").rstrip(".")); mac(f"dtimlo{g}{key}", f"{t.lo:.1f}".rstrip("0").rstrip(".")); mac(f"dtimhi{g}{key}", f"{t.hi:.1f}".rstrip("0").rstrip("."))
            mac(f"fearly{g}{key}", f"{100*t.frac_first_earlier:.0f}")
        if g != "all":
            lines.append(r"\midrule")
    tex = r"""\begin{table}[tbp]
\caption{Paired comparisons at the primary budget (1 alarm per station-year). $\Delta$POD and
$\Delta$FAR are first minus second detector; $\Delta$timing is the median, over events detected by both,
of (second detector's alarm $-$ first detector's alarm), so positive values mean the first detector
alarms earlier. Brackets: 95\% cluster-bootstrap intervals. $^{\dagger}$Post hoc. $^{\ddagger}$Distorted by
the wet-side SPI-3 threshold (Table~\ref{tab:warning}); with a dry-side threshold the pooled $\Delta$POD is
\spithreedrydpodall.}
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
    tex = r"""\begin{table}[tbp]
\caption{Monthly association (1981--2024) between the defect anomaly and concurrent
hydrometeorological anomalies: standardised coefficients from regressions with station fixed effects;
$^{\ast}p<0.05$ with standard errors clustered by year-month; $R^2_w$ is the within-station $R^2$.}
\label{tab:assoc}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrrrrrr}
\toprule
& \multicolumn{5}{c}{D1 defect, $\lambda>0$} & \multicolumn{5}{c}{D2 defect, $\lambda=0$}\\
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
    F = pd.read_csv(os.path.join(ROOT, "results", "covariates", "incremental_forward.csv"))
    F = F[(F.scheme == "forward") & (F.period == "2001-2024")]
    lab = {"D1": "SPI-90 + D1", "D2": "SPI-90 + D2", "D5": "SPI-90 + D5", "SPI30": "SPI-90 + SPI-30$^{\\dagger}$",
           "SPI30+D1": "SPI-90 + SPI-30 + D1$^{\\dagger}$"}

    def dfmt(r):
        nd = 4 if max(abs(r.dauc), abs(r.dauc_lo), abs(r.dauc_hi)) < 0.005 else 3
        return nd, f"{r.dauc:+.{nd}f} [{r.dauc_lo:+.{nd}f}, {r.dauc_hi:+.{nd}f}]"
    lines = []
    for g in REGIONS + ["all"]:
        fb = F[(F.group == g) & (F.reference == "base")].iloc[0]
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{RN[g]}}} (SPI-90 alone: AUC "
                     f"{I[(I.group == g) & (I.reference == 'base')].auc_base.iloc[0]:.3f} blocked, {fb.auc_ref:.3f} forward; "
                     f"prevalence {I[I.group == g].prevalence.iloc[0]:.3f} and {fb.prevalence:.3f})}}\\\\")
        for _, r in I[I.group == g].iterrows():
            f = F[(F.group == g) & (F.model == r.added)].iloc[0]
            nd, cell = dfmt(r); ndf, cellf = dfmt(f)
            lines.append(f"\\quad {lab[r.added]} & {r.auc_model:.3f} & {cell} & {f.auc_model:.3f} & {cellf} \\\\")
            key = r.added.replace('+', 'p').replace('30', 'thirty')
            mac(f"dauc{g}{key}", f"{r.dauc:+.{nd}f}")
            mac(f"dauclo{g}{key}", f"{r.dauc_lo:+.{nd}f}")
            mac(f"dauchi{g}{key}", f"{r.dauc_hi:+.{nd}f}")
            mac(f"fwddauc{g}{key}", f"{f.dauc:+.{ndf}f}")
            mac(f"fwddauclo{g}{key}", f"{f.dauc_lo:+.{ndf}f}")
            mac(f"fwddauchi{g}{key}", f"{f.dauc_hi:+.{ndf}f}")
        mac(f"fwdaucbase{g}", f"{fb.auc_ref:.2f}")
        f1 = F[(F.group == g) & (F.model == "D1")].iloc[0]     # text: gain from D1 beyond SPI-90, forward chaining
        assert (f1.dauc_lo > 0) == (g in ("germany", "australia", "all")), g
        if g != "all":
            lines.append(r"\midrule")
    tex = r"""\begin{table}[tbp]
\caption{Incremental information about drought onset within 60 days (weekly logistic models). Each model
is compared with the model without its last term; $\Delta$AUC is the difference in AUC and brackets are
95\% region-year cluster-bootstrap intervals. Blocked: pre-specified five-fold cross-validation over
blocks of years, 1982--2024 (retrospective discrimination). Forward: forward-chaining validation on
2001--2024, refitted every year on earlier weeks with a 60-day gap (post hoc). Predictors other than SPI
are 30-day means. $^{\dagger}$Post hoc.}
\label{tab:incremental}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccc}
\toprule
& \multicolumn{2}{c}{Blocked cross-validation} & \multicolumn{2}{c}{Forward chaining}\\
\cmidrule{2-3}\cmidrule{4-5}
Model & AUC & $\Delta$AUC & AUC & $\Delta$AUC\\
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
    C = chance_ref("warning_mv")
    lines = []
    for g in REGIONS + ["all"]:
        n = int(S[S.group == g].n_events.iloc[0])
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{RN[g]}}} ({n} events)}}\\\\")
        for d in dets:
            x = S[(S.group == g) & (S.det == d)].iloc[0]
            flag = ""
            if g != "all":
                t = Th[(Th.region == g) & (Th.det == d) & (Th.budget == 1.0)].iloc[0]
                flag = "" if t.attained else "$^{\\ast}$"
            tim = "--" if not np.isfinite(x.timing_med) else f"{x.timing_med:+.0f} [{x.timing_lo:+.0f}, {x.timing_hi:+.0f}]"
            c = C.loc[(g, d)]
            lines.append(f"\\quad {DN[d]}{flag} & {ci(x.pod, x.pod_lo, x.pod_hi)} & {c.pod_chance:.2f} & {ci(x.far, x.far_lo, x.far_hi)} & {tim} \\\\")
            mac(f"mvpod{g}{d}", f"{x.pod:.3f}"); mac(f"mvfar{g}{d}", f"{x.far:.2f}")
            mac(f"mvchance{g}{d}", f"{c.pod_chance:.3f}"); mac(f"mvchancelo{g}{d}", f"{c.lo:.3f}"); mac(f"mvchancehi{g}{d}", f"{c.hi:.3f}")
        mac(f"mvnev{g}", n)
        if g != "all":
            lines.append(r"\midrule")
    nst = sum(len(np.load(os.path.join(ROOT, "results", "panel_mv", f"{r}.npz"))["station_id"]) for r in REGIONS)
    mac("mvnst", nst)
    for r in REGIONS:
        mac(f"mvnst{r}", len(np.load(os.path.join(ROOT, "results", "panel_mv", f"{r}.npz"))["station_id"]))
    tex = r"""\begin{table}[tbp]
\caption{Temperature-aware analysis (Section~\ref{sec:mv}; """ + str(nst) + r""" stations with daily temperature):
event-based warning skill on 2001--2024 at one alarm per station-year. Definitions as in
Table~\ref{tab:warning}; (T) marks the temperature-aware network. $^{\ast}$Budget not attained within
10\% in calibration.}
\label{tab:warningmv}
\footnotesize\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccc}
\toprule
Detector & POD & Random & FAR (per yr) & Timing (d)\\
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
    tex = r"""\begin{table}[tbp]
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
        F = pd.read_csv(os.path.join(C, "incremental_forward.csv"))
        F = F[(F.scheme == "forward") & (F.period == "2001-2024")]
        lab = {"spei90+D8": "SPEI-90 + D8", "base+D8": "base + D8", "base+D9": "base + D9", "base+D1": "base + D1"}
        lines = []
        for g in REGIONS + ["all"]:
            i0 = I[(I.group == g) & (I.reference == "base")].iloc[0]; f0 = F[(F.group == g) & (F.reference == "base")].iloc[0]
            i1 = I[(I.group == g) & (I.reference == "spei90")].iloc[0]; f1 = F[(F.group == g) & (F.reference == "spei90")].iloc[0]
            lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{RN[g]}}} (AUC of SPEI-90 alone: {i1.auc_ref:.3f} blocked, {f1.auc_ref:.3f} forward; "
                         f"of base: {i0.auc_ref:.3f}, {f0.auc_ref:.3f})}}\\\\")
            for _, x in I[I.group == g].iterrows():
                f = F[(F.group == g) & (F.model == x.model)].iloc[0]
                nd = 4 if max(abs(x.dauc), abs(x.dauc_lo), abs(x.dauc_hi)) < 0.005 else 3
                ndf = 4 if max(abs(f.dauc), abs(f.dauc_lo), abs(f.dauc_hi)) < 0.005 else 3
                lines.append(f"\\quad {lab[x.model]} & {x.auc_model:.3f} & "
                             f"{x.dauc:+.{nd}f} [{x.dauc_lo:+.{nd}f}, {x.dauc_hi:+.{nd}f}] & {f.auc_model:.3f} & "
                             f"{f.dauc:+.{ndf}f} [{f.dauc_lo:+.{ndf}f}, {f.dauc_hi:+.{ndf}f}] \\\\")
                key = "mvdauc" + g + x.model.replace("+", "p").replace("90", "ninety")
                mac(key, f"{x.dauc:+.{nd}f}"); mac(key + "lo", f"{x.dauc_lo:+.{nd}f}"); mac(key + "hi", f"{x.dauc_hi:+.{nd}f}")
                key = "mvfwddauc" + g + x.model.replace("+", "p").replace("90", "ninety")
                mac(key, f"{f.dauc:+.{ndf}f}"); mac(key + "lo", f"{f.dauc_lo:+.{ndf}f}"); mac(key + "hi", f"{f.dauc_hi:+.{ndf}f}")
            if g != "all":
                lines.append(r"\midrule")
        tex = r"""\begin{table}[tbp]
\caption{Temperature-aware analysis: incremental information about onset within 60 days. ``base'' =
SPEI-90 + SPEI-30 + SPI-30; predictors other than the indices are 30-day means. Each model is compared
with the model without its last term. Blocked and forward as in Table~\ref{tab:incremental}; brackets are
95\% region-year cluster-bootstrap intervals.}
\label{tab:incrementalmv}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccc}
\toprule
& \multicolumn{2}{c}{Blocked cross-validation} & \multicolumn{2}{c}{Forward chaining}\\
\cmidrule{2-3}\cmidrule{4-5}
Model & AUC & $\Delta$AUC & AUC & $\Delta$AUC\\
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
        tex = r"""\begin{table}[tbp]
\caption{Temperature-aware analysis: monthly association of the temperature-aware defect anomaly with
concurrent anomalies (as Table~\ref{M-tab:assoc} of the main paper). $^{\ast}p<0.05$.}
\label{tab:assocmv}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrrrrrr}
\toprule
& \multicolumn{5}{c}{D8 defect (T), $\lambda>0$} & \multicolumn{5}{c}{D9 defect (T), $\lambda=0$}\\
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
        tex = r"""\begin{table}[tbp]
\caption{Temperature-aware networks: next-day precipitation forecasts on 2001--2024 (MSESS of the regularised network;
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


def robustness_tables():
    """Post hoc robustness checks (PROTOCOL Amendment 2): matching rule, scoring window,
    random-timing reference, dependence and forward-chaining validation.
    Inputs are written by matching_checks.py and incremental_forward.py."""
    W = {"": os.path.join(ROOT, "results", "warning"), "mv": os.path.join(ROOT, "results", "warning_mv")}
    S = {k: pd.read_csv(os.path.join(v, "matching_summary.csv")) for k, v in W.items()}
    P = {k: pd.read_csv(os.path.join(v, "matching_paired.csv")) for k, v in W.items()}
    C = {k: pd.read_csv(os.path.join(v, "matching_chance.csv")) for k, v in W.items()}
    O = pd.read_csv(os.path.join(W[""], "matching_overlap.csv"))
    E = pd.read_csv(os.path.join(W[""], "episodes.csv"))
    B = {k: pd.read_csv(os.path.join(v, "blockboot.csv")) for k, v in W.items()}
    RL = {"prespecified": "Pre-specified rule", "one_to_one": "One-to-one rule"}
    wins = [(60, 120), (60, 90), (60, 60), (30, 120), (30, 90), (30, 60), (0, 120), (0, 90), (0, 60)]

    def pooled(k, rule, pre, post):
        s = S[k][(S[k].group == "all") & (S[k].rule == rule) & (S[k].pre == pre) & (S[k].post == post)].set_index("det")
        p = P[k][(P[k].group == "all") & (P[k].rule == rule) & (P[k].pre == pre) & (P[k].post == post)].set_index("comp")
        c = C[k][(C[k].group == "all") & (C[k].pre == pre) & (C[k].post == post)].set_index("det")
        return s, p, c
    dp = lambda x: ci(x.dpod, x.dpod_lo, x.dpod_hi, 3, True)
    dt = lambda x: f"{x.dtiming:+.1f} [{x.dtiming_lo:+.1f}, {x.dtiming_hi:+.1f}]"

    # ---- Tables: matching rule x window (precipitation-only detectors)
    la, lb, lm = [], [], []
    for rule in RL:
        la.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{RL[rule]}}}}}\\\\")
        lb.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{RL[rule]}}}}}\\\\")
        lm.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{RL[rule]}}}}}\\\\")
        for pre, post in wins:
            s, p, c = pooled("", rule, pre, post)
            w = f"$[{'{-}' + str(pre) if pre else '0'},+{post}]$"
            la.append(f"{w} & {c.loc['D1'].pod_chance:.2f} & {s.loc['D1'].pod:.3f} & {s.loc['D3'].pod:.3f} & "
                      f"{dp(p.loc['D1-D3'])} & {dp(p.loc['D1-D2'])} \\\\")
            lb.append(f"{w} & {dp(p.loc['D1-D7'])} & {dp(p.loc['D7-D3'])} & {dt(p.loc['D1-D3'])} & {dt(p.loc['D1-D7'])} \\\\")
            s, p, c = pooled("mv", rule, pre, post)
            lm.append(f"{w} & {c.loc['D8'].pod_chance:.2f} & {s.loc['D8'].pod:.3f} & {s.loc['D11'].pod:.3f} & "
                      f"{dp(p.loc['D8-D11'])} & {dp(p.loc['D8-D9'])} \\\\")
        if rule == "prespecified":
            la.append(r"\midrule"); lb.append(r"\midrule"); lm.append(r"\midrule")
    cap_w = (r"Windows are $[-\text{pre},+\text{post}]$ days around the reference onset; $[-60,+120]$ under the "
             r"pre-specified rule is the primary analysis of the main paper. Thresholds are those calibrated for one "
             r"alarm per station-year and are not re-tuned. Brackets: 95\% year-cluster bootstrap intervals.")
    tex = r"""\begin{table}[tbp]
\caption{Sensitivity of the pooled warning comparisons to the matching rule and to the scoring window (post hoc).
""" + cap_w + r""" Random timing: POD of the D1 alarms after random circular shifts.}
\label{tab:matching}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccccc}
\toprule
& Random & \multicolumn{2}{c}{POD} & $\Delta$POD & $\Delta$POD\\
Window (d) & timing & D1 & D3 (SPI-90) & D1 $-$ D3 & D1 $-$ D2 ($\lambda=0$)\\
\midrule
""" + mm("\n".join(la)) + r"""
\botrule
\end{tabular*}
\end{table}

\begin{table}[tbp]
\caption{As Table~\ref{tab:matching}, for the comparisons that involve daily SPI-30 (D7) and for timing on
events detected by both detectors (positive: the first detector alarms earlier).}
\label{tab:matchingb}
\footnotesize\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccc}
\toprule
& $\Delta$POD & $\Delta$POD & $\Delta$timing (d) & $\Delta$timing (d)\\
Window (d) & D1 $-$ D7 & D7 $-$ D3 & D1 vs D3 & D1 vs D7\\
\midrule
""" + mm("\n".join(lb)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "matching.tex"), "w").write(tex)
    tex = r"""\begin{table}[tbp]
\caption{Temperature-aware analysis: sensitivity of the pooled comparisons to the matching rule and to the
scoring window (post hoc). """ + cap_w + r""" Random timing: POD of the D8 alarms after random circular
shifts.}
\label{tab:matchingmv}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccccc}
\toprule
& Random & \multicolumn{2}{c}{POD} & $\Delta$POD & $\Delta$POD\\
Window (d) & timing & D8 & D11 (SPEI-90) & D8 $-$ D11 & D8 $-$ D9 ($\lambda=0$)\\
\midrule
""" + mm("\n".join(lm)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "matching_mv.tex"), "w").write(tex)

    # ---- macros for the text: overlap, one-to-one, windows, random timing
    o = O[(O.pre == 60) & (O.post == 120) & (O.region == "all")].set_index("det")
    nev = int(o.loc["D1"].n_events)
    mac("noverlapall", int(o.loc["D1"].events_overlapping_previous))
    mac("pctoverlapall", f"{100 * o.loc['D1'].events_overlapping_previous / nev:.1f}")
    mac("nsharedDone", int(o.loc["D1"].hits_sharing_an_alarm)); mac("nsharedDthree", int(o.loc["D3"].hits_sharing_an_alarm))
    mac("nsharedmax", int(o.hits_sharing_an_alarm.max()))
    mac("nlatefalseDone", int(o.loc["D1"].credited_alarms_also_false)); mac("nlatefalseDthree", int(o.loc["D3"].credited_alarms_also_false))
    mac("nshift", "500")
    s1, p1, c1 = pooled("", "one_to_one", 60, 120); s0, p0, c0 = pooled("", "prespecified", 60, 120)
    mac("otopodallDone", f"{s1.loc['D1'].pod:.3f}"); mac("otopodallDthree", f"{s1.loc['D3'].pod:.3f}")
    mac("otofarallDone", f"{s1.loc['D1'].far:.2f}"); mac("otofarallDthree", f"{s1.loc['D3'].far:.2f}")
    for cmp_ in ["D1-D3", "D1-D2", "D1-D7"]:
        k = cmp_.replace("-", "")
        mac(f"otodpodall{k}", f"{p1.loc[cmp_].dpod:+.3f}"); mac(f"otodpodloall{k}", f"{p1.loc[cmp_].dpod_lo:+.3f}")
        mac(f"otodpodhiall{k}", f"{p1.loc[cmp_].dpod_hi:+.3f}")
    mac("ototimallDoneDthree", f"{p1.loc['D1-D3'].dtiming:.0f}")
    assert abs(p1.loc["D1-D3"].dpod - p0.loc["D1-D3"].dpod) < 0.005 and p1.loc["D1-D3"].dpod_hi < 0
    allw = P[""][(P[""].group == "all")]
    g = lambda cmp_: allw[allw.comp == cmp_]
    d13, d12, d17, d73 = g("D1-D3"), g("D1-D2"), g("D1-D7"), g("D7-D3")
    assert (d13.dpod_hi < 0).all() and (d17.dpod_hi < 0).all()          # text: fewer onsets than SPI-90 and SPI-30 in every setting
    assert ((d12.dpod_lo < 0) & (d12.dpod_hi > 0)).all()                # text: no detectable D1-D2 difference in any setting
    assert (d13.dtiming_lo > 0).all()                                   # text: earlier than SPI-90 in every setting
    mac("windpodDoneDthreemin", f"{d13.dpod.min():+.3f}"); mac("windpodDoneDthreemax", f"{d13.dpod.max():+.3f}")
    mac("windpodDoneDtwomin", f"{d12.dpod.min():+.3f}"); mac("windpodDoneDtwomax", f"{d12.dpod.max():+.3f}")
    mac("windpodDoneDtwohimax", f"{d12.dpod_hi.max():+.3f}"); mac("windpodDoneDtwolomin", f"{d12.dpod_lo.min():+.3f}")
    mac("windtimDoneDthreemin", f"{d13.dtiming.min():.0f}"); mac("windtimDoneDthreemax", f"{d13.dtiming.max():.0f}")
    z = allw[(allw.rule == "prespecified") & (allw.pre == 0) & (allw.post == 120)].set_index("comp")
    for cmp_ in ["D1-D3", "D7-D3", "D1-D7"]:
        k = cmp_.replace("-", "")
        mac(f"wzerodpod{k}", f"{z.loc[cmp_].dpod:+.3f}"); mac(f"wzerodpodlo{k}", f"{z.loc[cmp_].dpod_lo:+.3f}")
        mac(f"wzerodpodhi{k}", f"{z.loc[cmp_].dpod_hi:+.3f}")
    mac("wzerotimDoneDthree", f"{z.loc['D1-D3'].dtiming:.0f}")
    assert (d73[d73.pre == 0].dpod_hi < 0).all()                        # text: SPI-30 less sensitive than SPI-90 without pre-onset credit
    sz = S[""][(S[""].group == "all") & (S[""].rule == "prespecified") & (S[""].pre == 0) & (S[""].post == 120)].set_index("det")
    mac("wzeropodDone", f"{sz.loc['D1'].pod:.3f}"); mac("wzeropodDthree", f"{sz.loc['D3'].pod:.3f}")
    cz = C[""][(C[""].group == "all") & (C[""].pre == 0) & (C[""].post == 120)].set_index("det")
    mac("wzerochanceDone", f"{cz.loc['D1'].pod_chance:.2f}")
    # random-timing reference: statements in the text
    ch = chance_ref().reset_index(); chm = chance_ref("warning_mv").reset_index()
    sm = S[""][(S[""].rule == "prespecified") & (S[""].pre == 60) & (S[""].post == 120)].merge(ch, on=["group", "det"])
    al = sm[sm.group == "all"].set_index("det")
    for d in ["D1", "D2", "D3", "D7", "D6"]:
        assert al.loc[d].pod_lo > al.loc[d].hi                          # text: clearly above the reference
    assert al.loc["D5"].pod < al.loc["D5"].lo                           # text: forecast residual below the reference
    for r in ["germany", "australia"]:                                   # text: SPI-3 below the reference there
        x = sm[(sm.group == r) & (sm.det == "D4")].iloc[0]; assert x.pod < x.lo
    smm = S["mv"][(S["mv"].rule == "prespecified") & (S["mv"].pre == 60) & (S["mv"].post == 120)].merge(chm, on=["group", "det"])
    x = smm[(smm.group == "all") & (smm.det == "D8")].iloc[0]
    assert x.lo <= x.pod <= x.hi                                        # text: D8 indistinguishable from random timing
    for r in REGIONS:
        x = smm[(smm.group == r) & (smm.det == "D8")].iloc[0]
        mac(f"mvskill{r}", f"{x.pod - x.pod_chance:+.2f}")
    # SPI-3 with a dry-side threshold
    D4all = pd.read_csv(os.path.join(W[""], "spi3_dryside.csv"))
    S0 = pd.read_csv(os.path.join(W[""], "summary.csv"))
    lines = []
    for bud in [1.0, 0.5, 2.0]:
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{Budget {bud:g} alarm{'s' if bud != 1 else ''} per station-year}}}}\\\\")
        for g in REGIONS + ["all"]:
            x = D4all[(D4all.budget == bud) & (D4all.group == g)].iloc[0]
            p0 = S0[(S0.budget == bud) & (S0.group == g) & (S0.det == "D4")].pod.iloc[0]
            if g == "all":
                c0 = c1 = "--"
            else:
                c0 = f"{x.thr_prespecified:+.2f}"
                c1 = f"{x.thr_dryside:+.2f} ({x.cal_rate_dryside:.2f}" + ("" if x.attained_dryside else "$^{\\ast}$") + ")"
            lines.append(f"\\quad {RN[g]} & {c0} & {p0:.2f} & {c1} & {ci(x.pod, x.pod_lo, x.pod_hi)} & "
                         f"{ci(x.far, x.far_lo, x.far_hi)} & {ci(x.dpod, x.dpod_lo, x.dpod_hi, 3, True)} \\\\")
        if bud != 2.0:
            lines.append(r"\midrule")
    tex = r"""\begin{table}[tbp]
\caption{Monthly SPI-3 rule (D4) with its threshold restricted to the dry side, $c\le0$ (post hoc). The
alarm count of a month-end down-crossing rule is not monotone in $c$, and the pre-specified grid search
can meet a budget with a positive threshold. Columns: pre-specified threshold and POD; dry-side threshold
with the calibration rate it attains in 1991--2000; POD and FAR on 2001--2024 under the pre-specified
scoring rule; and the POD of the regularised defect (D1) minus that of SPI-3 at the same budget. Brackets:
95\% year-cluster bootstrap intervals. $^{\ast}$Budget not attained within 10\%.}
\label{tab:spithree}
\footnotesize\setlength{\tabcolsep}{1.5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccccc}
\toprule
& \multicolumn{2}{c}{Pre-specified} & \multicolumn{4}{c}{Dry-side threshold}\\
\cmidrule{2-3}\cmidrule{4-7}
Region & $c$ & POD & $c$ (rate) & POD & FAR (per yr) & $\Delta$POD, D1 $-$ D4\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "spi3_dryside.tex"), "w").write(tex)
    D4 = D4all[D4all.budget == 1.0].set_index("group")
    for r in REGIONS:
        mac(f"spithreethr{r}", f"{D4.loc[r].thr_prespecified:+.2f}"); mac(f"spithreedrythr{r}", f"{D4.loc[r].thr_dryside:+.2f}")
        mac(f"spithreedryrate{r}", f"{D4.loc[r].cal_rate_dryside:.2f}"); mac(f"spithreedrypod{r}", f"{D4.loc[r].pod:.2f}")
    assert D4.loc["germany"].thr_prespecified > 0 and D4.loc["australia"].thr_prespecified > 0
    mac("spithreedrypodall", f"{D4.loc['all'].pod:.3f}"); mac("spithreedryfarall", f"{D4.loc['all'].far:.2f}")
    mac("spithreedrydpodall", f"{D4.loc['all'].dpod:+.3f}"); mac("spithreedrydpodloall", f"{D4.loc['all'].dpod_lo:+.3f}")
    mac("spithreedrydpodhiall", f"{D4.loc['all'].dpod_hi:+.3f}")

    # ---- Table: episodes and overlap
    lines = []
    ov = O[(O.pre == 60) & (O.post == 120)].set_index(["region", "det"])
    for r in REGIONS + ["all"]:
        e = E[E.region == r].iloc[0]
        lines.append(f"{RN[r]} & {int(e.stations)} & {int(e.station_onsets)} & {int(e.episodes)} & "
                     f"{100 * e.share_onsets_in_episodes_ge5:.0f} & {int(ov.loc[(r, 'D1')].events_overlapping_previous)} & "
                     f"{int(ov.loc[(r, 'D1')].hits_sharing_an_alarm)} / {int(ov.loc[(r, 'D3')].hits_sharing_an_alarm)} & "
                     f"{int(ov.loc[(r, 'D1')].credited_alarms_also_false)} / {int(ov.loc[(r, 'D3')].credited_alarms_also_false)} \\\\")
        mac(f"nepisodes{r}", int(e.episodes))
    mac("shareepisodes", f"{100 * E[E.region == 'all'].share_onsets_in_episodes_ge5.iloc[0]:.0f}")
    tex = r"""\begin{table}[tbp]
\caption{Dependence among station-level events and overlap of scoring windows (2001--2024). Episodes: onsets
in a region linked when at most 30 days apart; share: percentage of station-level onsets in episodes of at
least five onsets; overlapping: events whose $[-60,+120]$-day window overlaps that of the preceding event at
the same station; shared: hits credited to an alarm that is also credited to another event; also false:
credited alarms that the pre-specified rule also counts as false alarms (D1 / D3).}
\label{tab:episodes}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrcc}
\toprule
Region & Stations & Onsets & Episodes & Share (\%) & Overlapping & Shared & Also false\\
\midrule
""" + "\n".join(lines) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "episodes.tex"), "w").write(tex)

    # ---- Table: multi-year block bootstrap
    lines = []; ratios = []
    for k, comps in [("", [("D1-D3", "D1 $-$ D3 (SPI-90)"), ("D1-D2", r"D1 $-$ D2 ($\lambda=0$)"), ("D1-D7", "D1 $-$ D7 (SPI-30)")]),
                     ("mv", [("D8-D11", "D8 $-$ D11 (SPEI-90)"), ("D8-D9", r"D8 $-$ D9 ($\lambda=0$)")])]:
        b = B[k][(B[k].group == "all") & (B[k].kind == "paired") & (B[k].rule == "prespecified")]
        for cmp_, lab in comps:
            x = b[b.comp == cmp_].set_index("block_years")
            lines.append(f"{lab} & {x.loc[1].dpod:+.3f} & " + " & ".join(f"[{x.loc[L].dpod_lo:+.3f}, {x.loc[L].dpod_hi:+.3f}]" for L in [1, 2, 3, 4]) + " \\\\")
            ratios += [(x.loc[L].dpod_hi - x.loc[L].dpod_lo) / (x.loc[1].dpod_hi - x.loc[1].dpod_lo) for L in [2, 3, 4]]
            if cmp_ in ("D1-D3", "D8-D11"):
                assert all(x.loc[L].dpod_hi < 0 for L in [1, 2, 3, 4])
            if cmp_ == "D1-D2":
                assert all(x.loc[L].dpod_lo < 0 < x.loc[L].dpod_hi for L in [1, 2, 3, 4])
    mac("blockratiolo", f"{min(ratios):.2f}"); mac("blockratiohi", f"{max(ratios):.2f}")
    tex = r"""\begin{table}[tbp]
\caption{Pooled $\Delta$POD at the primary budget with 95\% intervals from the year-cluster bootstrap (block
length 1, as in the main paper) and from circular block bootstraps that resample blocks of two, three and
four consecutive years within each region (post hoc).}
\label{tab:blockboot}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccccc}
\toprule
Comparison & $\Delta$POD & 1 year & 2 years & 3 years & 4 years\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "blockboot.tex"), "w").write(tex)

    # ---- Table: validation schemes for the incremental-information analysis (pooled)
    lines = []
    SCH = [("blocked", "1982-2024", "Blocked, 1982--2024"), ("blocked", "2001-2024", "Blocked"),
           ("blocked_purged", "2001-2024", "Blocked, 60-day gap"),
           ("forward", "2001-2024", "Forward, annual refit"), ("forward_6y", "2001-2024", "Forward, six-year refit")]
    for sub, comps, hd in [("covariates", [("D1", "SPI-90 + D1"), ("SPI30", "SPI-90 + SPI-30"), ("SPI30+D1", "SPI-90 + SPI-30 + D1")],
                            "Precipitation-only analysis"),
                           ("covariates_mv", [("spei90+D8", "SPEI-90 + D8"), ("base+D8", "base + D8"), ("base+D1", "base + D1")],
                            "Temperature-aware analysis")]:
        F = pd.read_csv(os.path.join(ROOT, "results", sub, "incremental_forward.csv")); F = F[F.group == "all"]
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{hd}}}: " + ", ".join(l for _, l in comps) + "}\\\\")
        for sc, per, lab in SCH:
            cells = []
            for m, _ in comps:
                x = F[(F.scheme == sc) & (F.period == per) & (F.model == m)].iloc[0]
                nd = 4 if max(abs(x.dauc), abs(x.dauc_lo), abs(x.dauc_hi)) < 0.005 else 3
                cells.append(f"{x.dauc:+.{nd}f} [{x.dauc_lo:+.{nd}f}, {x.dauc_hi:+.{nd}f}]")
            lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        if sub == "covariates":
            lines.append(r"\midrule")
            F0 = F[F.period == "2001-2024"]
            a = F0[F0.model == "SPI30+D1"]
            assert ((a.dauc_lo < 0) & (a.dauc_hi > 0) | (a.dauc_hi <= 0)).all()   # text: defect adds nothing to SPI-90 + SPI-30 in any scheme
            mac("fwdschemesDonemin", f"{F0[F0.model == 'D1'].dauc.min():+.3f}"); mac("fwdschemesDonemax", f"{F0[F0.model == 'D1'].dauc.max():+.3f}")
    tex = r"""\begin{table}[tbp]
\caption{Incremental information about onset within 60 days under five validation schemes, pooled over
regions: $\Delta$AUC of each model against the model without its last term, with 95\% region-year
cluster-bootstrap intervals. In the blocked schemes the training folds include years after the test block;
in the forward-chaining schemes a week is predicted only from earlier weeks, with a 60-day gap. The first
row is the pre-specified scheme, scored on 1982--2024; the other rows are post hoc and scored on
2001--2024. ``base'' = SPEI-90 + SPEI-30 + SPI-30.}
\label{tab:forward}
\footnotesize\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccc}
\toprule
Validation scheme & First model & Second model & Third model\\
\midrule
""" + mm("\n".join(lines)) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    open(os.path.join(T, "forward.tex"), "w").write(tex)


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
    for sub, key in [("panel", "ncomp"), ("panel_mv", "ncompmv")]:
        mac(key, int(K[(K.subset == sub) & (K.det == "D1")].n_onsets.sum()))
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
    mv_tables(); robustness_tables()
    warning_table(0.5, "warning_b05.tex", "tab:warning05", ("D1", "D2", "D3", "D4", "D5", "D7"))
    warning_table(2.0, "warning_b2.tex", "tab:warning2", ("D1", "D2", "D3", "D4", "D5", "D7"))
    check_macros()
    open(os.path.join(T, "numbers.tex"), "w").write("% generated by code/serra/make_tables.py\n" + "\n".join(MACROS) + "\n")
    print(len(MACROS), "macros")
