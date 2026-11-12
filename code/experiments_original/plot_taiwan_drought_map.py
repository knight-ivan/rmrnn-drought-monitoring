"""
plot_taiwan_drought_map.py
==========================
Map the per-station 2020--2021 drought early-warning lead (RM-defect CUSUM
minus SPI-3 CUSUM, seed-median days; positive = RM earlier) onto Taiwan, to
read the result as an AREA EFFECT rather than an island-wide average.

Station coordinates are the real CWA/CODiS station locations
(https://codis.cwa.gov.tw/api/station_list, cached to /tmp/codis_stations.json
or re-fetched). Per-station leads come from
experiments/taiwan_drought_warning_results.json
(produced by run_taiwan_drought_warning.py).

Output: figures/taiwan_drought_map.pdf

Usage
  /opt/anaconda3/bin/python plot_taiwan_drought_map.py
"""

import json
import os
import sys
from urllib.request import Request, urlopen

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
EXP  = os.path.join(BASE, "experiments")
FIG  = os.path.join(BASE, "figures")
RESULTS = os.path.join(EXP, "taiwan_drought_warning_results.json")
CODIS_CACHE = "/tmp/codis_stations.json"
CODIS_URL = "https://codis.cwa.gov.tw/api/station_list"

# Satellite basemap (Esri World Imagery), fetched once for Taiwan's bounding box.
SAT_CACHE = "/tmp/taiwan_sat.png"
SAT_BBOX = (119.9, 21.85, 122.1, 25.35)   # xmin, ymin, xmax, ymax (lon/lat)
ESRI_EXPORT = (
    "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/"
    "MapServer/export?bbox={xmin},{ymin},{xmax},{ymax}"
    "&bboxSR=4326&imageSR=4326&size={w},{h}&format=png&f=image"
)


def load_satellite():
    """Return (image_array, extent) for the satellite basemap, fetching once."""
    if not os.path.exists(SAT_CACHE):
        xmin, ymin, xmax, ymax = SAT_BBOX
        w = 1200
        h = int(round(w * (ymax - ymin) / (xmax - xmin)))   # degree-proportional
        url = ESRI_EXPORT.format(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, w=w, h=h)
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=90) as r:
            open(SAT_CACHE, "wb").write(r.read())
    img = plt.imread(SAT_CACHE)
    xmin, ymin, xmax, ymax = SAT_BBOX
    return img, (xmin, xmax, ymin, ymax)


def load_codis():
    if not os.path.exists(CODIS_CACHE):
        req = Request(CODIS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=60) as r:
            open(CODIS_CACHE, "wb").write(r.read())
    codis = json.load(open(CODIS_CACHE))
    recs = {}
    for grp in codis["data"]:
        for r in grp["item"]:
            if r.get("longitude"):
                recs[r["stationID"]] = r
    return recs


def main():
    if not os.path.exists(RESULTS):
        print(f"Results not found: {RESULTS}")
        sys.exit(1)
    res = json.load(open(RESULTS))
    recs = load_codis()

    pts = []   # (lon, lat, lead_or_None, name, det_rm, det_spi)
    for p in res["per_station"]:
        r = recs.get(p["sid"])
        if not r:
            continue
        pts.append((r["longitude"], r["latitude"], p["median_lead_days"],
                    p["name"], p["det_rate_rm"], p["det_rate_spi"]))

    fig, ax = plt.subplots(figsize=(7.6, 9.4))
    # satellite basemap
    img, extent = load_satellite()
    ax.imshow(img, extent=extent, origin="upper", zorder=0)

    # diverging colour scale on the lead, symmetric about 0
    paired = [p[2] for p in pts if p[2] is not None]
    vmax = max(abs(min(paired)), abs(max(paired)))
    norm = plt.Normalize(-vmax, vmax)
    cmap = plt.get_cmap("RdBu")   # red = negative (SPI earlier), blue = positive (RM earlier)
    halo = [pe.withStroke(linewidth=2.2, foreground="black")]

    for lon, lat, lead, name, drm, dspi in pts:
        if lead is None:
            ax.scatter(lon, lat, s=110, marker="^", facecolor="white",
                       edgecolor="black", lw=1.2, zorder=3)
            lab = f"{name} (SPI n/a)"
        else:
            ax.scatter(lon, lat, s=90 + 3.2 * abs(lead) ** 0.8,
                       color=cmap(norm(lead)), edgecolor="white", lw=1.1, zorder=3)
            lab = f"{name} {lead:+.0f}d"
        ax.annotate(lab, (lon, lat), fontsize=6.6, ha="left", va="center",
                    color="white", path_effects=halo,
                    xytext=(5, 0), textcoords="offset points", zorder=4)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(23.7)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("2020--2021 drought: per-station early-warning lead\n"
                 "RM-defect CUSUM − SPI-3 CUSUM (seed median, days)",
                 fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label("lead (days);  blue = RM earlier,  red = SPI earlier")

    legend = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=10, label="SPI-3 never fires"),
    ]
    leg = ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.9)
    leg.set_zorder(5)
    ax.text(0.01, 0.01, "Imagery: Esri, Maxar, Earthstar Geographics",
            transform=ax.transAxes, fontsize=5.5, color="white",
            path_effects=halo, ha="left", va="bottom", zorder=5)
    fig.tight_layout()
    out = os.path.join(FIG, "taiwan_drought_map.pdf")
    fig.savefig(out, dpi=200); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
