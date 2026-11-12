"""Fetch daily ERA5-Land covariates at each gauge location (Open-Meteo
historical archive API, model=era5_land, no key required).

Variables: root-zone (28-100 cm) and 7-28 cm soil moisture, 2-m mean
temperature, FAO-56 reference evapotranspiration, and ERA5-Land precipitation.
The soil-moisture series defines the reference drought events used to score
all detectors; it is never an input to any detector.

Output: data/<region>/era5land/<station_id>.csv
"""
import argparse, json, os, time, urllib.request, urllib.error
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
       "&start_date={a}-01-01&end_date={b}-12-31&models=era5_land&timezone=GMT"
       "&daily=soil_moisture_28_to_100cm_mean,soil_moisture_7_to_28cm_mean,"
       "temperature_2m_mean,et0_fao_evapotranspiration,precipitation_sum")


def get(url, tries=8):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            wait = 65 if e.code == 429 else 10 * (k + 1)
            print("  HTTP", e.code, "waiting", wait, "s"); time.sleep(wait)
        except Exception as e:  # network hiccup
            print("  error", type(e).__name__, "retry"); time.sleep(10 * (k + 1))
    raise RuntimeError("failed: " + url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--first", type=int, default=1961)
    ap.add_argument("--last", type=int, default=2024)
    a = ap.parse_args()
    meta = pd.read_csv(os.path.join(ROOT, "data", a.region, "station_meta.csv"), dtype={"station_id": str})
    od = os.path.join(ROOT, "data", a.region, "era5land"); os.makedirs(od, exist_ok=True)
    for _, r in meta.iterrows():
        out = os.path.join(od, f"{r.station_id}.csv")
        if os.path.exists(out):
            continue
        frames = []
        for a0, b0 in [(a.first, 1992), (1993, a.last)]:  # two spans keep requests light
            d = get(API.format(lat=r.lat, lon=r.lon, a=a0, b=b0))["daily"]
            frames.append(pd.DataFrame(d))
            time.sleep(3)
        df = pd.concat(frames).rename(columns={"time": "date"})
        df.to_csv(out, index=False)
        print(a.region, r.station_id, len(df))


if __name__ == "__main__":
    main()
