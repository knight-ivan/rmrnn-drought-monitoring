"""Descriptive (post hoc) diagnostic: mean standardised anomaly of each score in
1991-2000 and 2001-2024 relative to its 1961-1990 reference, per region, with a
year-cluster bootstrap CI for the 2001-2024 mean. Output: results/warning_mv/drift.csv"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS

rng = np.random.default_rng(5)
rows = []
for r in REGIONS:
    pn = np.load(os.path.join(ROOT, "results", "panel_mv", f"{r}.npz"))
    for k in ["D1", "D8", "D9", "D10", "D11", "t2m_a"]:
        X = pn[k].astype(float)
        yr = DATES.year.values
        ym = pd.DataFrame(X.T).groupby(yr).mean().mean(1)     # yearly mean over stations
        e = ym.loc[2001:2024].values
        b = [np.mean(rng.choice(e, len(e))) for _ in range(2000)]
        rows.append(dict(region=r, score=k, mean_1991_2000=float(ym.loc[1991:2000].mean()),
                         mean_2001_2024=float(e.mean()), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5)))
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "warning_mv", "drift.csv"), index=False)
print(pd.DataFrame(rows).round(2).to_string())
