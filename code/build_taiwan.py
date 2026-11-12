"""Build the Taiwan long-format file and station metadata from the Raingel
download (CWA synoptic stations), applying the same completeness rule as for
GHCN (>= 90% in 1961-1990 and in 1991-2024)."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COORD = {  # station: (name, lat, lon, elev m) -- CWA station list
    "467490": ("Taichung", 24.15, 120.68, 84), "467650": ("Sun Moon Lake", 23.88, 120.91, 1018),
    "467550": ("Yushan", 23.49, 120.96, 3845), "467770": ("Wuqi", 24.26, 120.52, 32),
    "467530": ("Alishan", 23.51, 120.81, 2413), "466910": ("Anbu", 25.18, 121.53, 838),
    "466920": ("Taipei", 25.04, 121.51, 6), "466940": ("Keelung", 25.13, 121.74, 27),
    "466900": ("Tamsui", 25.16, 121.45, 19), "466880": ("Banqiao", 25.00, 121.44, 10),
    "467610": ("Chenggong", 23.10, 121.37, 34), "466990": ("Hualien", 23.98, 121.61, 16),
    "467540": ("Dawu", 22.36, 120.90, 8), "467660": ("Taitung", 22.75, 121.15, 9),
    "467080": ("Yilan", 24.76, 121.76, 7), "467060": ("Suao", 24.60, 121.86, 25),
    "467590": ("Hengchun", 22.00, 120.75, 22), "467480": ("Chiayi", 23.50, 120.43, 27),
    "467050": ("Xinwu", 25.01, 121.05, 21), "467410": ("Tainan", 22.99, 120.20, 41),
    "467420": ("Yongkang", 23.04, 120.24, 8), "467440": ("Kaohsiung", 22.57, 120.32, 2),
}


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "taiwan_raingel", "raingel", "daily_precip_long.csv"),
                     dtype={"station_id": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df.date.dt.year >= 1961) & (df.date.dt.year <= 2024)]
    meta, keep = [], []
    for s, g in df.groupby("station_id"):
        g = g.drop_duplicates("date").set_index("date")["precip_mm"]
        def comp(a, b):
            idx = pd.date_range(f"{a}-01-01", f"{b}-12-31")
            return float(g.reindex(idx).notna().mean())
        cr, ce = comp(1961, 1990), comp(1991, 2024)
        nm, la, lo, el = COORD.get(s, ("?", float("nan"), float("nan"), float("nan")))
        ok = cr >= 0.9 and ce >= 0.9 and s in COORD
        meta.append(dict(station_id=s, lat=la, lon=lo, elev=el, name=nm,
                         complete_ref=round(cr, 4), complete_eval=round(ce, 4), kept=ok))
        if ok:
            keep.append(pd.DataFrame({"date": g.index.strftime("%Y-%m-%d"), "station_id": s,
                                      "precip_mm": g.values}))
    od = os.path.join(ROOT, "data", "taiwan"); os.makedirs(od, exist_ok=True)
    m = pd.DataFrame(meta)
    m.to_csv(os.path.join(od, "station_meta_all_candidates.csv"), index=False)
    m[m.kept].drop(columns="kept").to_csv(os.path.join(od, "station_meta.csv"), index=False)
    pd.concat(keep).dropna(subset=["precip_mm"]).to_csv(os.path.join(od, "daily_precip_long.csv"), index=False)
    print(m.to_string())


if __name__ == "__main__":
    main()
