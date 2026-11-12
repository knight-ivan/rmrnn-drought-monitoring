"""Fetch long daily PRCP records from NOAA GHCN-Daily for one region.

Candidates: stations inside the region's bounding box whose PRCP inventory
spans 1961..2024. Each station's .dly file is downloaded, PRCP values with a
non-blank quality flag are set to missing, and stations are retained when
both the reference block (1961-1990) and the evaluation block (1991-2024)
are at least 90% complete. Output (long format):
  data/<region>/daily_precip_long.csv   date,station_id,precip_mm
  data/<region>/station_meta.csv        station_id,lat,lon,elev,name,complete_ref,complete_eval
"""
import argparse, os, urllib.request, concurrent.futures as cf
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GH = os.path.join(ROOT, "data", "ghcnd")
URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{}.dly"
BOX = {  # lat0, lat1, lon0, lon1 (original study regions)
    "texas": (29.0, 33.0, -101.0, -96.0),  # central Texas (widened from the original Hill-Country box)
    "germany": (47.5, 54.5, 6.0, 15.0),
    "australia": (-37.0, -28.0, 140.0, 152.0),  # Murray-Darling Basin
}
# Turkey (original study) dropped: no GHCN gauge in the box is >=50% complete
# over 1961-1990, so a WMO-standard SPI-3 reference cannot be fitted.


def stations():
    st = {}
    for ln in open(os.path.join(GH, "ghcnd-stations.txt")):
        st[ln[0:11]] = (float(ln[12:20]), float(ln[21:30]), float(ln[31:37]), ln[41:71].strip())
    return st


def inventory():
    inv = {}
    for ln in open(os.path.join(GH, "ghcnd-inventory.txt")):
        s = ln.split()
        if s[3] == "PRCP":
            inv[s[0]] = (int(s[4]), int(s[5]))
    return inv


def parse_dly(sid):
    path = os.path.join(GH, "dly", sid + ".dly")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(URL.format(sid), timeout=180) as r:
            open(path, "wb").write(r.read())
    out = []
    for ln in open(path):
        if ln[17:21] != "PRCP":
            continue
        y, m = int(ln[11:15]), int(ln[15:17])
        for d in range(31):
            v = int(ln[21 + 8 * d: 26 + 8 * d]); q = ln[27 + 8 * d]
            if v == -9999 or q != " ":
                continue
            try:
                out.append((pd.Timestamp(y, m, d + 1), v / 10.0))
            except ValueError:
                pass
    s = pd.Series(dict(out)).sort_index()
    return s[~s.index.duplicated()]


def completeness(s, a, b):
    idx = pd.date_range(f"{a}-01-01", f"{b}-12-31")
    return float(s.reindex(idx).notna().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=list(BOX))
    ap.add_argument("--first", type=int, default=1961)
    ap.add_argument("--last", type=int, default=2024)
    ap.add_argument("--min-complete", type=float, default=0.90)
    ap.add_argument("--max-stations", type=int, default=40)
    a = ap.parse_args()
    la0, la1, lo0, lo1 = BOX[a.region]
    st, inv = stations(), inventory()
    cand = [s for s, (la, lo, el, nm) in st.items()
            if la0 <= la <= la1 and lo0 <= lo <= lo1 and s in inv
            and inv[s][0] <= a.first and inv[s][1] >= a.last]
    print(a.region, "candidates:", len(cand))
    with cf.ThreadPoolExecutor(8) as ex:
        series = dict(zip(cand, ex.map(parse_dly, cand)))
    keep, meta = [], []
    for sid, s in series.items():
        cr, ce = completeness(s, a.first, 1990), completeness(s, 1991, a.last)
        la, lo, el, nm = st[sid]
        meta.append(dict(station_id=sid, lat=la, lon=lo, elev=el, name=nm,
                         complete_ref=round(cr, 4), complete_eval=round(ce, 4),
                         kept=bool(cr >= a.min_complete and ce >= a.min_complete)))
        if meta[-1]["kept"]:
            s = s[(s.index.year >= a.first) & (s.index.year <= a.last)]
            keep.append(pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"),
                                      "station_id": sid, "precip_mm": s.values}))
    # spatial thinning: greedy farthest-point selection among kept stations
    m = pd.DataFrame(meta)
    k = m[m.kept].sort_values("complete_eval", ascending=False).reset_index(drop=True)
    if len(k) > a.max_stations:
        xy = np.radians(k[["lat", "lon"]].values)
        chosen = [0]
        dmin = np.full(len(k), np.inf)
        while len(chosen) < a.max_stations:
            la, lo = xy[chosen[-1]]
            d = np.arccos(np.clip(np.sin(xy[:, 0]) * np.sin(la) + np.cos(xy[:, 0]) * np.cos(la) * np.cos(xy[:, 1] - lo), -1, 1))
            dmin = np.minimum(dmin, d)
            chosen.append(int(np.argmax(dmin)))
        sel = set(k.loc[chosen, "station_id"])
        m["kept"] = m.station_id.isin(sel)
        keep = [x for x in keep if x.station_id.iloc[0] in sel]
    meta = m.to_dict("records")
    od = os.path.join(ROOT, "data", a.region); os.makedirs(od, exist_ok=True)
    m = pd.DataFrame(meta).sort_values("station_id")
    m.to_csv(os.path.join(od, "station_meta_all_candidates.csv"), index=False)
    m[m.kept].drop(columns="kept").to_csv(os.path.join(od, "station_meta.csv"), index=False)
    pd.concat(keep).to_csv(os.path.join(od, "daily_precip_long.csv"), index=False)
    print(a.region, "kept", int(m.kept.sum()), "of", len(m))


if __name__ == "__main__":
    main()
