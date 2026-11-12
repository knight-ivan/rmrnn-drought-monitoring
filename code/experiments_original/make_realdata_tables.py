"""
make_realdata_tables.py
=======================
Build the real-data multi-landform result tables and comparison figure for the
JHM manuscript from the saved experiment outputs:

  forecast/path-stability :  experiments/{taiwan_precip_results.TAIWAN_backup,
                                          texas_precip_results,
                                          germany_precip_results,
                                          turkey_precip_results}.npz
  drought-warning lead    :  experiments/{taiwan,texas,germany,turkey}_drought_warning_results.json

Outputs
-------
  tables/realdata_forecast_table_body.tex
  tables/realdata_drought_table_body.tex
  figures/landform_comparison.pdf
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
EXP  = os.path.join(BASE, "experiments")
TAB  = os.path.join(BASE, "tables")
FIG  = os.path.join(BASE, "figures")
os.makedirs(TAB, exist_ok=True)

# region label -> (forecast npz, drought json)
# Ordered by decreasing drought-warning lead (Texas -> Germany -> Turkey -> Taiwan),
# which also traces a climatological axis (semi-arid flash-drought -> monsoonal
# orographic -> Mediterranean seasonal). Keep this order in tables and figure.
REGIONS = [
    ("Texas",       "texas_precip_results.npz",                "texas_drought_warning_results.json"),
    ("Germany",     "germany_precip_results.npz",              "germany_drought_warning_results.json"),
    ("Turkey",      "turkey_precip_results.npz",               "turkey_drought_warning_results.json"),
    ("Taiwan",      "taiwan_precip_results.TAIWAN_backup.npz", "taiwan_drought_warning_results.json"),
]
MODELS = ["RMRNN", "VanillaRNN", "GRU"]
MLAB   = {"RMRNN": r"\RMRNN", "VanillaRNN": "VanillaRNN", "GRU": "GRU"}

def ms(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return a.mean(), a.std()

# ---------- forecast / path-stability table ----------
fc = {}
for reg, npz, _ in REGIONS:
    d = np.load(os.path.join(EXP, npz), allow_pickle=True)
    n = len(d["station_ids"])
    fc[reg] = {"n": n}
    for m in MODELS:
        fc[reg][m] = {met: ms(d[f"{m}/{met}"]) for met in ("mse", "mae", "Q_path", "auc")}

lines = []
for ri, (reg, npz, _) in enumerate(REGIONS):
    R = fc[reg]
    if ri:
        lines.append(r"\midrule")
    for i, m in enumerate(MODELS):
        mse = R[m]["mse"]; mae = R[m]["mae"]; q = R[m]["Q_path"]; auc = R[m]["auc"]
        first = ("%s (%d)" % (reg, R["n"])) if i == 0 else ""
        lines.append(r"%s & %s & %.3f$\pm$%.3f & %.3f$\pm$%.3f & %.2f$\pm$%.2f & %.3f$\pm$%.3f \\"
                     % (first, MLAB[m], mse[0], mse[1], mae[0], mae[1], q[0], q[1], auc[0], auc[1]))
open(os.path.join(TAB, "realdata_forecast_table_body.tex"), "w").write("\n".join(lines) + "\n")

# ---------- drought-warning lead table ----------
dw = {}
for reg, _, js in REGIONS:
    J = json.load(open(os.path.join(EXP, js)))
    g = J["integrated"]
    # per-station median leads (for honest display of cross-station spread)
    st_leads = np.array([p["median_lead_days"] for p in J.get("per_station", [])
                         if p.get("median_lead_days") is not None], float)
    q25, q75 = (np.percentile(st_leads, [25, 75]) if st_leads.size else (np.nan, np.nan))
    dw[reg] = dict(n=J["n_stations"],
                   det_rm=g["mean_det_rate_rm"], det_spi=g["mean_det_rate_spi"],
                   med=g["pooled_median_lead"], mean=g["pooled_mean_lead"],
                   sd=g["pooled_sd_lead"], frac=g["pooled_frac_rm_earlier"],
                   npos=g["n_stations_positive_median"],
                   medstat=g["median_of_station_median_leads"],
                   st_leads=st_leads, q25=q25, q75=q75)
dl = []
for reg, _, js in REGIONS:
    D = dw[reg]
    dl.append(r"%s & %d & %.0f\%% & %.0f\%% & %+.0f & %+.0f$\pm$%.0f & %.0f\%% \\"
              % (reg, D["n"], 100*D["det_rm"], 100*D["det_spi"],
                 D["med"], D["mean"], D["sd"], 100*D["frac"]))
open(os.path.join(TAB, "realdata_drought_table_body.tex"), "w").write("\n".join(dl) + "\n")

# ---------- comparison figure ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
regs = [r for r, _, _ in REGIONS]
x = np.arange(len(regs)); w = 0.26
cols = {"RMRNN": "#1f77b4", "VanillaRNN": "#999999", "GRU": "#ff7f0e"}
for i, m in enumerate(MODELS):
    vals = [fc[r][m]["Q_path"][0] for r in regs]
    err  = [fc[r][m]["Q_path"][1] for r in regs]
    ax1.bar(x + (i-1)*w, vals, w, yerr=err, capsize=3, label=MLAB[m].replace("\\",""), color=cols[m])
ax1.set_xticks(x); ax1.set_xticklabels(regs)
ax1.set_ylabel(r"$Q_{\mathrm{path}}$ (hidden-state path instability)")
ax1.set_title("(a) Path stability preserved in every region")
ax1.legend(fontsize=8)

med = [dw[r]["med"] for r in regs]
frac = [100*dw[r]["frac"] for r in regs]
rng = np.random.default_rng(0)
ax2.axhline(0, color="k", lw=0.8)
for xi, r in zip(x, regs):
    D = dw[r]
    # light median bar
    ax2.bar(xi, D["med"], 0.5, color=("#2ca02c" if D["med"] > 0 else "#d62728"),
            alpha=0.30, zorder=1)
    # IQR whisker across stations (honest spread, robust to the skew)
    if np.isfinite(D["q25"]):
        ax2.plot([xi, xi], [D["q25"], D["q75"]], color="k", lw=1.4, zorder=2)
        ax2.plot([xi-0.08, xi+0.08], [D["q25"]]*2, color="k", lw=1.4, zorder=2)
        ax2.plot([xi-0.08, xi+0.08], [D["q75"]]*2, color="k", lw=1.4, zorder=2)
    # per-station median leads as jittered points (shows the full spread + sign split)
    sl = D["st_leads"]
    if sl.size:
        jx = xi + rng.uniform(-0.16, 0.16, sl.size)
        ax2.scatter(jx, sl, s=12, color="#1f77b4", alpha=0.55,
                    edgecolors="none", zorder=3)
ax2.set_xticks(x); ax2.set_xticklabels(regs)
ax2.set_ylabel("RM$-$SPI-3 lead (days), per station")
ax2.set_title("(b) Drought-warning lead varies across regions")
for xi, r in zip(x, regs):
    D = dw[r]
    top = D["q75"] if np.isfinite(D["q75"]) else D["med"]
    ax2.annotate(f"med {D['med']:+.0f} d\n{100*D['frac']:.0f}% earlier\n{D['npos']}/{D['n']} stn $>$0",
                 (xi, top), ha="center", va="bottom", fontsize=7,
                 xytext=(0, 4), textcoords="offset points")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "landform_comparison.pdf"))
print("Wrote tables/realdata_forecast_table_body.tex, tables/realdata_drought_table_body.tex,",
      "figures/landform_comparison.pdf")

# also echo a plain-text summary for the chat
print("\nFORECAST / PATH STABILITY")
for r in regs:
    R = fc[r]
    print(f"  {r:11s} n={R['n']:2d}  RMRNN MSE {R['RMRNN']['mse'][0]:.3f} "
          f"Qpath {R['RMRNN']['Q_path'][0]:.2f} | GRU MSE {R['GRU']['mse'][0]:.3f} "
          f"Qpath {R['GRU']['Q_path'][0]:.2f} | dQ {100*(R['RMRNN']['Q_path'][0]/R['VanillaRNN']['Q_path'][0]-1):+.1f}%")
print("\nDROUGHT WARNING LEAD (RM-defect vs SPI-3)")
for r in regs:
    D = dw[r]
    print(f"  {r:11s} n={D['n']:2d}  det {100*D['det_rm']:.0f}/{100*D['det_spi']:.0f}%  "
          f"median {D['med']:+.0f} d  frac-earlier {100*D['frac']:.0f}%  pos {D['npos']}/{D['n']}")
