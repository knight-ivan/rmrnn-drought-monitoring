"""Daily Tmax/Tmin for every rain gauge (PROTOCOL amendment 1) and Hargreaves PET.

GHCN regions: own record if Tmax and Tmin are each >= 80% complete in 1961-1990 and
1991-2024, else nearest qualifying GHCN station within 150 km (donor). Taiwan: the
CWA station's own TxMaxAbs/TxMinAbs from the Raingel daily files.
Output: data/<region>/temperature_long.csv (date, station_id, tmax, tmin, pet)
        data/<region>/temperature_meta.csv (station_id, source_id, distance_km, completeness)
"""
import glob, os, sys, urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GH = os.path.join(ROOT, "data", "ghcnd")
URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{}.dly"
DATES = pd.date_range("1961-01-01", "2024-12-31")
MINC, MAXKM = 0.80, 150.0


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def dly_temp(sid):
    path = os.path.join(GH, "dly", sid + ".dly")
    if not os.path.exists(path):
        with urllib.request.urlopen(URL.format(sid), timeout=180) as r:
            open(path, "wb").write(r.read())
    out = {"TMAX": {}, "TMIN": {}}
    for ln in open(path):
        el = ln[17:21]
        if el not in out:
            continue
        y, m = int(ln[11:15]), int(ln[15:17])
        for d in range(31):
            v = int(ln[21 + 8 * d: 26 + 8 * d]); q = ln[27 + 8 * d]
            if v == -9999 or q != " ":
                continue
            try:
                out[el][pd.Timestamp(y, m, d + 1)] = v / 10.0
            except ValueError:
                pass
    df = pd.DataFrame({"tmax": pd.Series(out["TMAX"], dtype=float), "tmin": pd.Series(out["TMIN"], dtype=float)})
    return clean(df.reindex(DATES))


def clean(df):
    bad = (df.tmax < df.tmin) | (df.tmax.abs() > 60) | (df.tmin.abs() > 60)
    df.loc[bad, ["tmax", "tmin"]] = np.nan
    return df


def completeness(df):
    c = []
    for a, b in [(1961, 1990), (1991, 2024)]:
        sel = (df.index.year >= a) & (df.index.year <= b)
        c.append(min(df.tmax[sel].notna().mean(), df.tmin[sel].notna().mean()))
    return min(c)


def hargreaves(df, lat):
    """FAO-56 extraterrestrial radiation (eq. 21) and Hargreaves-Samani ET0 (mm/day)."""
    J = df.index.dayofyear.values
    phi = np.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    Ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    tmean = (df.tmax + df.tmin) / 2
    rng = np.sqrt(np.clip(df.tmax - df.tmin, 0, None))
    return 0.0023 * 0.408 * Ra * (tmean + 17.8) * rng


def ghcn_region(region):
    meta = pd.read_csv(os.path.join(ROOT, "data", region, "station_meta.csv"), dtype={"station_id": str})
    st = {}
    for ln in open(os.path.join(GH, "ghcnd-stations.txt")):
        st[ln[:11]] = (float(ln[12:20]), float(ln[21:30]))
    inv = {}
    for ln in open(os.path.join(GH, "ghcnd-inventory.txt")):
        s = ln.split()
        if s[3] in ("TMAX", "TMIN"):
            inv.setdefault(s[0], {})[s[3]] = (int(s[4]), int(s[5]))
    cand = [s for s, v in inv.items() if "TMAX" in v and "TMIN" in v
            and v["TMAX"][0] <= 1961 and v["TMAX"][1] >= 2024 and v["TMIN"][0] <= 1961 and v["TMIN"][1] >= 2024]
    cache, rows, frames = {}, [], []
    for _, r in meta.iterrows():
        dist = sorted((hav(r.lat, r.lon, *st[c]), c) for c in cand + [r.station_id] if c in st)
        chosen = None
        for d, c in dist:
            if d > MAXKM:
                break
            if c not in cache:
                cache[c] = dly_temp(c)
            if completeness(cache[c]) >= MINC:
                chosen = (c, d); break
        if chosen is None:
            rows.append(dict(station_id=r.station_id, source_id="", distance_km=np.nan, completeness=np.nan))
            continue
        c, d = chosen
        df = cache[c].copy()
        df["pet"] = hargreaves(df, r.lat)
        rows.append(dict(station_id=r.station_id, source_id=c, distance_km=round(d, 1), completeness=round(completeness(df), 4)))
        frames.append(df.assign(station_id=r.station_id).rename_axis("date").reset_index())
        print(region, r.station_id, "<-", c, f"{d:.1f} km", flush=True)
    return rows, frames


def taiwan():
    meta = pd.read_csv(os.path.join(ROOT, "data", "taiwan", "station_meta.csv"), dtype={"station_id": str})
    rows, frames = [], []
    for _, r in meta.iterrows():
        fr = []
        for f in sorted(glob.glob(os.path.join(ROOT, "data", "taiwan_raingel", "raingel", "raw", r.station_id, "*.csv"))):
            d = pd.read_csv(f, encoding="utf-8-sig")
            d = d.rename(columns={d.columns[0]: "date"})[["date", "TxMaxAbs", "TxMinAbs"]]
            fr.append(d)
        d = pd.concat(fr)
        d["date"] = pd.to_datetime(d.date, errors="coerce")
        d = d.dropna(subset=["date"]).drop_duplicates("date").set_index("date")
        df = pd.DataFrame({"tmax": pd.to_numeric(d.TxMaxAbs, errors="coerce"),
                           "tmin": pd.to_numeric(d.TxMinAbs, errors="coerce")}).reindex(DATES)
        df = clean(df)
        comp = completeness(df)
        if comp < MINC:
            rows.append(dict(station_id=r.station_id, source_id="", distance_km=np.nan, completeness=round(comp, 4)))
            continue
        df["pet"] = hargreaves(df, r.lat)
        rows.append(dict(station_id=r.station_id, source_id=r.station_id, distance_km=0.0, completeness=round(comp, 4)))
        frames.append(df.assign(station_id=r.station_id).rename_axis("date").reset_index())
    return rows, frames


if __name__ == "__main__":
    for region in sys.argv[1:] or ["texas", "germany", "australia", "taiwan"]:
        rows, frames = taiwan() if region == "taiwan" else ghcn_region(region)
        od = os.path.join(ROOT, "data", region)
        pd.DataFrame(rows).to_csv(os.path.join(od, "temperature_meta.csv"), index=False)
        T = pd.concat(frames)
        T["date"] = T["date"].dt.strftime("%Y-%m-%d")
        T[["date", "station_id", "tmax", "tmin", "pet"]].to_csv(os.path.join(od, "temperature_long.csv"), index=False)
        n = sum(1 for x in rows if x["source_id"])
        print(region, "with temperature:", n, "of", len(rows), flush=True)
