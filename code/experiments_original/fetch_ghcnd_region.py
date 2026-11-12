"""
fetch_ghcnd_region.py
=====================
Build a tidy daily precipitation CSV (date, station_id, precip_mm) from the
**real** NOAA GHCN-Daily archive for all stations inside a lat/lon bounding box
with adequate PRCP coverage. Output matches the long format consumed by
common/data.py::load_precip_matrix (same as the Taiwan raingel pipeline), so the
existing experiment scripts run unchanged on a new region.

Usage
-----
  python3 fetch_ghcnd_region.py --name texas_hill \
      --lat0 29.5 --lat1 31.5 --lon0 -100.0 --lon1 -97.5 \
      --years 1980-2024 --start-by 1985 --end-by 2020

Outputs
-------
  Data Sets/ghcnd_<name>/daily_precip_long.csv
  Data Sets/ghcnd_<name>/station_meta.csv   (id, lat, lon, name, n_days)
"""
import argparse, os, sys, io, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))        # scripts/experiments/precipitation
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # repo root
GHCND = os.path.join(BASE, "data", "ghcnd")
BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"

def _dl(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")

def load_stations(path):
    d = {}
    for ln in open(path):
        sid = ln[0:11]; lat = float(ln[12:20]); lon = float(ln[21:30])
        name = ln[41:71].strip()
        d[sid] = (lat, lon, name)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lat0", type=float, required=True)
    ap.add_argument("--lat1", type=float, required=True)
    ap.add_argument("--lon0", type=float, required=True)
    ap.add_argument("--lon1", type=float, required=True)
    ap.add_argument("--years", default="1980-2024")
    ap.add_argument("--start-by", type=int, default=1985,
                    help="station PRCP record must start at or before this year")
    ap.add_argument("--end-by", type=int, default=2020,
                    help="station PRCP record must end at or after this year")
    ap.add_argument("--max-stations", type=int, default=60)
    args = ap.parse_args()

    y0, y1 = (int(x) for x in args.years.split("-"))
    st = load_stations(os.path.join(GHCND, "ghcnd-stations.txt"))
    in_box = lambda la, lo: (args.lat0 <= la <= args.lat1 and args.lon0 <= lo <= args.lon1)
    cand = {s for s, (la, lo, nm) in st.items() if in_box(la, lo)}

    # inventory filter: PRCP with adequate temporal coverage
    keep = []
    for ln in open(os.path.join(GHCND, "ghcnd-inventory.txt")):
        sid = ln[0:11]; elem = ln[31:35]
        iy0 = int(ln[36:40]); iy1 = int(ln[41:45])
        if sid in cand and elem == "PRCP" and iy0 <= args.start_by and iy1 >= args.end_by:
            keep.append(sid)
    keep = sorted(set(keep))[: args.max_stations]
    print(f"[{args.name}] {len(cand)} stations in box; {len(keep)} pass PRCP coverage filter")

    outdir = os.path.join(BASE, "Data Sets", f"ghcnd_{args.name}")
    os.makedirs(outdir, exist_ok=True)
    rows = []          # (date, sid, mm)
    meta = []
    for i, sid in enumerate(keep, 1):
        try:
            txt = _dl(f"{BASE_URL}/all/{sid}.dly")
        except Exception as e:
            print(f"  ! {sid} download failed: {e}"); continue
        n = 0
        for ln in txt.splitlines():
            if ln[17:21] != "PRCP":
                continue
            yr = int(ln[11:15]); mo = int(ln[15:17])
            if yr < y0 or yr > y1:
                continue
            for day in range(31):
                base = 21 + day * 8
                val = ln[base:base + 5].strip()
                qflag = ln[base + 6:base + 7]
                if val in ("", "-9999"):
                    continue
                if qflag.strip():            # failed a quality check -> skip
                    continue
                mm = int(val) / 10.0         # tenths of mm -> mm
                if mm < 0:
                    continue
                rows.append((f"{yr:04d}-{mo:02d}-{day+1:02d}", sid, mm))
                n += 1
        la, lo, nm = st[sid]
        meta.append((sid, la, lo, nm, n))
        print(f"  [{i}/{len(keep)}] {sid} {nm[:22]:22s} {n} obs")

    # de-duplicate / sort and write long CSV
    rows.sort()
    csv_path = os.path.join(outdir, "daily_precip_long.csv")
    with open(csv_path, "w") as f:
        f.write("date,station_id,precip_mm\n")
        for d, s, mm in rows:
            f.write(f"{d},{s},{mm}\n")
    with open(os.path.join(outdir, "station_meta.csv"), "w") as f:
        f.write("station_id,lat,lon,name,n_days\n")
        for sid, la, lo, nm, n in meta:
            f.write(f"{sid},{la},{lo},{nm},{n}\n")
    print(f"[{args.name}] wrote {len(rows)} rows from {len(meta)} stations -> {csv_path}")

if __name__ == "__main__":
    main()
