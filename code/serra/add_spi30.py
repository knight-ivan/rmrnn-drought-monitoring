"""Post hoc sensitivity comparator: daily SPI-30 (30-day totals, >= 27 observed
days, zero-inflated gamma per day of year on 1961-1990). Stored as D7 in the
region panels."""
import os, sys
import numpy as np
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATES, REGIONS, load_region, spi_daily


def one(p):
    return spi_daily(p, DATES, window=30, min_obs=27)


if __name__ == "__main__":
    for r in REGIONS:
        f = os.path.join(ROOT, "results", "panel", f"{r}.npz")
        pn = dict(np.load(f))
        meta, P = load_region(r)
        assert list(meta.station_id) == list(pn["station_id"])
        with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", 8))) as pool:
            d7 = pool.map(one, [P[s].values for s in meta.station_id])
        pn["D7"] = np.stack(d7).astype(np.float32)
        np.savez_compressed(f, **pn)
        print(r, "D7 added", flush=True)
