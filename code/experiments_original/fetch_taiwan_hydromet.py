#!/usr/bin/env python3
"""
fetch_taiwan_hydromet.py

Local downloader for Taiwan hydrometeorological data sources used in the
RMRNN precipitation / drought paper.

Sources supported
-----------------
1. WRA open data API: reservoir water level, reservoir daily operation, or arbitrary resource UUIDs.
2. CWA open data API: automatic rainfall station observations.
3. NCDR DataHub API: token-authenticated rainfall / disaster sensor services.
4. Depositar CKAN API: package search + optional resource downloads.
5. TReAD / SciDM / TCCIP: manifest helper for manually downloaded NetCDF files.

Important
---------
This script is intended to be run on your local machine with open network access.
Some sources require registration / credentials. The script never redistributes
account-gated data; it downloads to a local folder under your control.

Examples
--------
# 1. WRA reservoir real-time water level data
python fetch_taiwan_hydromet.py --source wra --wra-resource reservoir_water_level --format JSON

# 2. WRA reservoir daily operation status
python fetch_taiwan_hydromet.py --source wra --wra-resource reservoir_daily_operation --format CSV

# 3. WRA arbitrary resource UUID from Swagger / data.gov.tw
python fetch_taiwan_hydromet.py --source wra --wra-resource-id 2be9044c-6e44-4856-aad5-dd108c2e6679 --format CSV

# 4. CWA automatic rainfall station observations
python fetch_taiwan_hydromet.py --source cwa_rain --cwa-token YOUR_CWA_TOKEN

# 5. NCDR DataHub rainfall service
python fetch_taiwan_hydromet.py --source ncdr --token YOUR_NCDR_TOKEN \
  --ncdr-service Disaster/Sensor/ObsRainfall \
  --param YY=2020 --param MM=01 --param Lon=121.5 --param Lat=24.0

# 6. NCDR rainfall by county, if enabled for your account
python fetch_taiwan_hydromet.py --source ncdr --token YOUR_NCDR_TOKEN \
  --ncdr-service Disaster/Sensor/ObsRainfallByCounty \
  --param County=臺北市 --param YYYY=2020

# 7. Depositar search only
python fetch_taiwan_hydromet.py --source depositar --query "drought Taiwan" --no-download

# 8. Depositar search and download linked resources
python fetch_taiwan_hydromet.py --source depositar --query "Zhuoshui Taiwan hydrology"

# 9. Make a manifest for manually downloaded TReAD NetCDF files
python fetch_taiwan_hydromet.py --source tread_manifest --tread-dir ./raw/TReAD

# 10. Long daily station rainfall from the Raingel/historical_weather archive
#     (rebuilt CWA/CODiS, 1896-present, no registration). Builds a tidy
#     date,station_id,precip_mm CSV at <outdir>/raingel/daily_precip_long.csv
python fetch_taiwan_hydromet.py --source raingel --stations 466920,467410 --years 1990-2020
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


WRA_RESOURCES = {
    # data.gov.tw dataset 45501: Reservoir water level data
    "reservoir_water_level": "2be9044c-6e44-4856-aad5-dd108c2e6679",
    # data.gov.tw dataset 41568: Reservoir daily operation status
    "reservoir_daily_operation": "51023e88-4c76-4dbc-bbb9-470da690d539",
}

CWA_DATASETS = {
    # data.gov.tw dataset 9177: Automatic rainfall station observations
    "automatic_rainfall": "O-A0002-001",
}


def safe_name(s: str, max_len: int = 160) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s.strip())
    out = "_".join(filter(None, out.split("_")))
    return out[:max_len] or "download"


DEFAULT_UA = "Mozilla/5.0 (compatible; taiwan-hydromet-fetch/1.0; +research)"


def _curl_get(url: str, headers: Dict[str, str], timeout: int) -> bytes:
    """Fallback for hosts that only serve HTTP/2 (urllib speaks only HTTP/1.1)."""
    import subprocess
    cmd = ["curl", "-sS", "--fail", "--max-time", str(timeout)]
    for k, v in headers.items():
        cmd += ["-H", f"{k}: {v}"]
    cmd.append(url)
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise URLError(f"curl fallback failed ({proc.returncode}): {proc.stderr.decode('utf-8', 'ignore')[:200]}")
    return proc.stdout


def request_url(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 90) -> bytes:
    merged = {"User-Agent": DEFAULT_UA}
    if headers:
        merged.update(headers)
    req = Request(url, headers=merged)
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        # Some endpoints (e.g. Depositar resource downloads) require HTTP/2,
        # which urllib cannot negotiate. Retry via curl when available.
        return _curl_get(url, merged, timeout)


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_params(param_items: Optional[List[str]]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    if not param_items:
        return params
    for item in param_items:
        if "=" not in item:
            raise ValueError(f"--param must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        params[key] = value
    return params


RAINGEL_RAW = "https://raw.githubusercontent.com/Raingel/historical_weather/main/data/{stn}/{stn}_{year}_daily.csv"


def parse_years(spec: str) -> List[int]:
    """Parse a year spec like '1990-2020' or '1990,1995,2001' or '2010'."""
    years: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            years.extend(range(int(lo), int(hi) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def fetch_raingel(args: argparse.Namespace) -> List[Path]:
    """Download long daily station rainfall from the Raingel/historical_weather
    archive (rebuilt CWA/CODiS, 1896-present, no registration) and build a tidy
    long-format CSV: date, station_id, precip_mm."""
    if not args.stations:
        raise ValueError("raingel requires --stations, e.g. 466920,467410")
    if not args.years:
        raise ValueError("raingel requires --years, e.g. 1990-2020")

    stations = [s.strip() for s in args.stations.split(",") if s.strip()]
    years = parse_years(args.years)
    outdir = Path(args.outdir) / "raingel"
    raw_dir = outdir / "raw"

    written: List[Path] = []
    tidy_rows: List[Dict[str, str]] = []
    skipped: List[str] = []

    for stn in stations:
        for year in years:
            url = RAINGEL_RAW.format(stn=stn, year=year)
            try:
                content = request_url(url, timeout=60)
            except Exception as e:
                skipped.append(f"{stn}_{year}: {type(e).__name__}")
                continue
            path = raw_dir / stn / f"{stn}_{year}_daily.csv"
            write_bytes(path, content)
            written.append(path)

            # Parse the daily file and extract the Precp (mm) column.
            text = content.decode("utf-8-sig", "ignore")
            reader = csv.reader(text.splitlines())
            try:
                header = next(reader)
            except StopIteration:
                continue
            # First column is the date; find the daily precipitation column.
            precp_idx = None
            for cand in ("Precp", "Precipitation.Accumulationf"):
                if cand in header:
                    precp_idx = header.index(cand)
                    break
            for row in reader:
                if not row or not row[0]:
                    continue
                precip = row[precp_idx] if (precp_idx is not None and precp_idx < len(row)) else ""
                # CWA/CODiS codes missing/no-observation as negative sentinels
                # (e.g. -9.8, -9.9); precipitation cannot be negative, so blank them.
                # "T" marks a trace amount; record it as 0.0.
                precip = precip.strip()
                if precip in ("T", "t"):
                    precip = "0.0"
                else:
                    try:
                        if float(precip) < 0:
                            precip = ""
                    except ValueError:
                        precip = ""
                tidy_rows.append({
                    "date": row[0],
                    "station_id": stn,
                    "precip_mm": precip,
                })

    if not written:
        raise ValueError(
            "No Raingel files downloaded. Check station ids and years "
            f"(skipped: {', '.join(skipped[:10])}{'...' if len(skipped) > 10 else ''})."
        )

    tidy_rows.sort(key=lambda r: (r["station_id"], r["date"]))
    tidy_path = outdir / "daily_precip_long.csv"
    with tidy_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "station_id", "precip_mm"])
        writer.writeheader()
        writer.writerows(tidy_rows)
    written.append(tidy_path)

    print(f"Downloaded {len(written) - 1} station-year files; "
          f"wrote {len(tidy_rows)} daily rows to {tidy_path}")
    if skipped:
        print(f"Skipped {len(skipped)} missing station-years (e.g. {', '.join(skipped[:5])}).")
    return written


def fetch_wra(args: argparse.Namespace) -> List[Path]:
    outdir = Path(args.outdir) / "wra"
    fmt = args.format.upper()

    if args.wra_resource_id:
        resource_id = args.wra_resource_id
        label = args.wra_resource or resource_id
    else:
        if args.wra_resource not in WRA_RESOURCES:
            raise ValueError(
                f"Unknown --wra-resource {args.wra_resource!r}. "
                f"Known: {', '.join(WRA_RESOURCES)} or pass --wra-resource-id UUID."
            )
        resource_id = WRA_RESOURCES[args.wra_resource]
        label = args.wra_resource

    params = {"format": fmt, "sort": "_importdate asc"}
    url = f"https://opendata.wra.gov.tw/api/v2/{resource_id}?{urlencode(params)}"
    suffix = "json" if fmt == "JSON" else fmt.lower()
    path = outdir / f"{safe_name(label)}.{suffix}"
    content = request_url(url)
    write_bytes(path, content)
    write_text(outdir / f"{safe_name(label)}.url.txt", url + "\n")
    print(f"Wrote {path}")
    return [path]


def fetch_cwa_rain(args: argparse.Namespace) -> List[Path]:
    outdir = Path(args.outdir) / "cwa"
    dataset = CWA_DATASETS["automatic_rainfall"]
    token = args.cwa_token or args.token or "rdec-key-123-45678-011121314"
    params = {"Authorization": token}
    if args.format:
        params["format"] = args.format.upper()
    url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/{dataset}?{urlencode(params)}"
    path = outdir / "automatic_rainfall_O-A0002-001.json"
    content = request_url(url)
    write_bytes(path, content)
    write_text(outdir / "automatic_rainfall_O-A0002-001.url.txt", url + "\n")
    print(f"Wrote {path}")
    return [path]


def fetch_ncdr(args: argparse.Namespace) -> List[Path]:
    if not args.token:
        raise ValueError("NCDR requires --token YOUR_NCDR_TOKEN.")
    if not args.ncdr_service:
        raise ValueError("NCDR requires --ncdr-service, e.g. Disaster/Sensor/ObsRainfall.")

    outdir = Path(args.outdir) / "ncdr"
    params = parse_params(args.param)
    query = urlencode(params)
    service = args.ncdr_service.strip("/")
    url = f"https://dataapi.ncdr.nat.gov.tw/api/V1/{service}"
    if query:
        url += "?" + query

    headers = {"Token": args.token}
    content = request_url(url, headers=headers)
    label = safe_name(service.replace("/", "_"))
    path = outdir / f"{label}.json"
    write_bytes(path, content)
    # Do not write token to disk; save sanitized URL only.
    write_text(outdir / f"{label}.url.txt", url + "\n")
    print(f"Wrote {path}")
    return [path]


def fetch_depositar(args: argparse.Namespace) -> List[Path]:
    if not args.query:
        raise ValueError("Depositar requires --query.")
    outdir = Path(args.outdir) / "depositar" / safe_name(args.query)
    outdir.mkdir(parents=True, exist_ok=True)

    search_url = "https://data.depositar.io/api/3/action/package_search?" + urlencode({"q": args.query})
    search_bytes = request_url(search_url)
    search_path = outdir / "package_search.json"
    write_bytes(search_path, search_bytes)
    write_text(outdir / "package_search.url.txt", search_url + "\n")

    written = [search_path]
    data = json.loads(search_bytes.decode("utf-8"))

    if args.no_download:
        print(f"Wrote {search_path}; resource downloads skipped.")
        return written

    results = data.get("result", {}).get("results", [])
    manifest_rows = []
    for pkg in results:
        pkg_name = pkg.get("name") or pkg.get("id") or "package"
        for i, res in enumerate(pkg.get("resources", []) or []):
            url = res.get("url")
            if not url:
                continue
            res_name = safe_name(res.get("name") or res.get("id") or f"resource_{i}")
            # Skip very large or non-file URLs when requested.
            if args.max_resources is not None and len(manifest_rows) >= args.max_resources:
                break
            try:
                content = request_url(url, timeout=180)
                suffix = Path(url.split("?")[0]).suffix
                if not suffix or len(suffix) > 10:
                    suffix = ".bin"
                path = outdir / safe_name(pkg_name) / f"{res_name}{suffix}"
                write_bytes(path, content)
                written.append(path)
                status = "downloaded"
                size = path.stat().st_size
            except Exception as e:
                path = Path("")
                status = f"failed: {type(e).__name__}: {e}"
                size = ""
            manifest_rows.append({
                "package": pkg_name,
                "resource_name": res_name,
                "url": url,
                "status": status,
                "local_path": str(path),
                "size_bytes": size,
            })

    manifest_path = outdir / "resource_download_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["package", "resource_name", "url", "status", "local_path", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    written.append(manifest_path)
    print(f"Wrote {search_path} and {manifest_path}")
    return written


def tread_manifest(args: argparse.Namespace) -> List[Path]:
    tread_dir = Path(args.tread_dir or (Path(args.outdir) / "TReAD"))
    outdir = Path(args.outdir) / "tread_manifest"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(tread_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".nc", ".nc4", ".cdf", ".grb", ".grib", ".zip"}:
            rows.append({
                "file": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    manifest_path = outdir / "tread_local_file_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    note = """TReAD retrieval note

TReAD / SciDM / TCCIP files are not redistributed here.
Recommended local procedure:
1. Open the SciDM DOI record for TReAD.
2. Log in with your registered account.
3. Select only the variables, temporal aggregation, years, and basin/region needed.
4. Download NetCDF or provider-packaged files into a local raw/TReAD folder.
5. Run:
   python fetch_taiwan_hydromet.py --source tread_manifest --tread-dir raw/TReAD

For the JHM paper, record product name, version, temporal aggregation,
spatial subset, date accessed, and the exact list of local files used.
"""
    note_path = outdir / "README_TReAD_manual_download.txt"
    write_text(note_path, note)
    print(f"Wrote {manifest_path}")
    return [manifest_path, note_path]


def write_run_manifest(paths: Iterable[Path], args: argparse.Namespace) -> None:
    rows = []
    for path in paths:
        if path.exists():
            rows.append({
                "source": args.source,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    if not rows:
        return
    manifest_path = Path(args.outdir) / "run_manifest.csv"
    new_file = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "path", "size_bytes", "sha256"])
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch Taiwan hydrometeorological datasets locally.")
    p.add_argument("--source", required=True,
                   choices=["wra", "cwa_rain", "ncdr", "depositar", "tread_manifest", "raingel"])
    p.add_argument("--outdir", default="taiwan_hydromet_raw")
    p.add_argument("--format", default="JSON", help="JSON, CSV, or XML when supported.")

    # WRA
    p.add_argument("--wra-resource", default="reservoir_water_level",
                   help=f"Known WRA resource alias: {', '.join(WRA_RESOURCES)}")
    p.add_argument("--wra-resource-id", help="Arbitrary WRA resource UUID from Swagger/data.gov.tw.")

    # Tokens
    p.add_argument("--token", help="NCDR token, or general token.")
    p.add_argument("--cwa-token", help="CWA OpenData API token. Defaults to public data.gov demo key.")

    # NCDR
    p.add_argument("--ncdr-service", help="NCDR service path after /api/V1/, e.g. Disaster/Sensor/ObsRainfall")
    p.add_argument("--param", action="append", help="Query parameter as key=value. Repeat as needed.")

    # Depositar
    p.add_argument("--query", help="Depositar CKAN package_search query.")
    p.add_argument("--no-download", action="store_true", help="Only save package_search metadata.")
    p.add_argument("--max-resources", type=int, default=None, help="Maximum Depositar resources to download.")

    # TReAD
    p.add_argument("--tread-dir", help="Directory containing manually downloaded TReAD files.")

    # Raingel historical daily station archive
    p.add_argument("--stations", help="Comma-separated CWA station ids, e.g. 466920,467410 (Raingel).")
    p.add_argument("--years", help="Year spec for Raingel, e.g. 1990-2020 or 1990,1995,2001.")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.source == "wra":
            paths = fetch_wra(args)
        elif args.source == "cwa_rain":
            paths = fetch_cwa_rain(args)
        elif args.source == "ncdr":
            paths = fetch_ncdr(args)
        elif args.source == "depositar":
            paths = fetch_depositar(args)
        elif args.source == "tread_manifest":
            paths = tread_manifest(args)
        elif args.source == "raingel":
            paths = fetch_raingel(args)
        else:
            raise ValueError(f"Unsupported source: {args.source}")
        write_run_manifest(paths, args)
        return 0
    except HTTPError as e:
        print(f"HTTP error {e.code}: {e.reason}")
        try:
            print(e.read().decode("utf-8", "ignore")[:2000])
        except Exception:
            pass
        return 2
    except URLError as e:
        print(f"Network error: {e}")
        return 3
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
