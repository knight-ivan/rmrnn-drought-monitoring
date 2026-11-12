"""Fetch daily MERRA-2 covariates at each gauge from the NASA POWER point API
(no key): root-zone soil wetness (GWETROOT), profile soil wetness (GWETPROF),
2-m temperature (T2M), evapotranspiration energy flux (EVPTRNS) and corrected
precipitation (PRECTOTCORR), 1981-2024.

Root-zone wetness defines the reference drought events used to score every
detector; it is never an input to any detector.
Output: data/<region>/power/<station_id>.csv
"""
import argparse, json, os, time, urllib.request
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARS = "GWETROOT,GWETPROF,T2M,EVPTRNS,PRECTOTCORR"
API = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters={p}"
       "&community=AG&longitude={lon}&latitude={lat}&start=19810101&end=20241231&format=JSON")


def get(url, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=600) as r:
                return json.loads(r.read())
        except Exception as e:
            print("  retry", type(e).__name__, flush=True); time.sleep(20 * (k + 1))
    raise RuntimeError(url)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--region", required=True); a = ap.parse_args()
    meta = pd.read_csv(os.path.join(ROOT, "data", a.region, "station_meta.csv"), dtype={"station_id": str})
    od = os.path.join(ROOT, "data", a.region, "power"); os.makedirs(od, exist_ok=True)
    for _, r in meta.iterrows():
        out = os.path.join(od, f"{r.station_id}.csv")
        if os.path.exists(out):
            continue
        j = get(API.format(p=PARS, lat=round(r.lat, 4), lon=round(r.lon, 4)))
        df = pd.DataFrame(j["properties"]["parameter"])
        df.index = pd.to_datetime(df.index, format="%Y%m%d"); df.index.name = "date"
        df = df.where(df > -998)
        df.to_csv(out)
        print(a.region, r.station_id, len(df), flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
