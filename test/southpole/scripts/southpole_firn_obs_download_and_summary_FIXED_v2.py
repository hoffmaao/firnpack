#!/usr/bin/env python3
"""
southpole/scripts/southpole_firn_obs_download_and_summary.py  (FIXED v2: ERA5 zip/netcdf handling)

This revision fixes the xarray open_dataset() failure you hit after successful CDS jobs:

Why it happened
---------------
When requesting *multiple* variables in NetCDF via cdsapi, CDS may return a **ZIP archive**
containing multiple NetCDF files (e.g., split by GRIB stepType: "instant" vs "accum"), even if you
requested NetCDF. The resulting file may be named *.nc but actually be a ZIP, so xarray can't detect
a NetCDF backend and raises:

  ValueError: did not find a match in any of xarray's currently installed IO backends ...

Fix
---
- Detect ZIP outputs by magic bytes (PK..)
- Extract all *.nc from the archive(s)
- Merge variables into a single NetCDF
- Open NetCDF explicitly with engine="netcdf4"

Also includes earlier fixes:
- Use 'data_format' (CDS API docs)
- Clamp latitude bounds so the bbox never crosses -90° at South Pole
- Avoid requesting future months by splitting requests into:
    * full years up to last_full_year
    * partial last year up to last closed month

Directory layout (expected)
---------------------------
southpole/
├── data
├── figures
├── processed
└── scripts

Raw downloads -> southpole/data/
Processed CSVs -> southpole/processed/
Figures -> southpole/figures/

Run from scripts directory
--------------------------
cd southpole/scripts
python southpole_firn_obs_download_and_summary.py

Env toggles
-----------
SKIP_ERA5=1   Skip ERA5 download/processing
SKIP_USAP=1   Skip USAP page parsing/guidance

Requirements
------------
pip install numpy pandas xarray netcdf4 matplotlib requests cdsapi
(optional) pip install beautifulsoup4
"""

from __future__ import annotations

import io
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Site + Sources
# =========================

SITE_NAME = "South Pole / SPICEcore (South Pole Station vicinity)"

LAT = -89.99
LON = -99.16  # degrees East (negative = West)

USP50_LAT = -89.54
USP50_LON = 137.04  # degrees East

SPLICE_YEAR_ERA5 = 1940

RHO_W = 1000.0
RHO_I = 917.0

# NOAA / NCEI direct URLs
NOAA_SP19_AGE_URL = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/sp19age.txt"
NOAA_SP19_DENS_URL = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/sp19dens.txt"
NOAA_BUIZERT_SPICE_TEMP_URL = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/buizert2021/buizert2021spice-temp-noaa.txt"
NOAA_BUIZERT_SPICE_ACCUM_URL = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/buizert2021/buizert2021spice-accum-noaa.txt"

USAP_DATASETS = {
    "601680": ("USP50 compaction/strain (borehole shortening)", "https://www.usap-dc.org/view/dataset/601680"),
    "601525": ("USP50 firn thermistor string (6-hourly)", "https://www.usap-dc.org/view/dataset/601525"),
    "601551": ("In-situ compilation (density/temp/grain/layers) [optional]", "https://www.usap-dc.org/view/dataset/601551"),
}

ERA5_DATASET = "reanalysis-era5-single-levels-monthly-means"
ERA5_VARS = ["2m_temperature", "mean_snowfall_rate", "mean_snow_evaporation_rate"]


# =========================
# Paths: enforce southpole/ layout
# =========================

def get_repo_dirs() -> Tuple[Path, Path, Path, Path, Path, Path]:
    scripts_dir = Path(__file__).resolve().parent
    root_dir = scripts_dir.parent  # southpole/
    data_dir = root_dir / "data"
    fig_dir = root_dir / "figures"
    proc_dir = root_dir / "processed"
    for d in (data_dir, fig_dir, proc_dir):
        d.mkdir(parents=True, exist_ok=True)

    noaa_dir = data_dir / "noaa_ncei"
    era5_dir = data_dir / "era5"
    usap_dir = data_dir / "usap_dc"
    for d in (noaa_dir, era5_dir, usap_dir):
        d.mkdir(parents=True, exist_ok=True)

    return root_dir, data_dir, proc_dir, fig_dir, noaa_dir, era5_dir


# =========================
# Download helpers
# =========================

def download_file(url: str, path: Path, overwrite: bool = False) -> None:
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        print(f"[download] exists: {path}")
        return
    import requests
    print(f"[download] {url}")
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=2**20):
            if chunk:
                f.write(chunk)
    tmp.replace(path)
    print(f"[download] wrote: {path}")


# =========================
# NOAA template parser
# =========================

def read_noaa_template_table(path: Path) -> pd.DataFrame:
    txt = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        raise ValueError(f"No data lines found in {path}")
    buf = io.StringIO("\n".join(lines))
    try:
        return pd.read_csv(buf, sep=r"\s+", engine="python")
    except Exception:
        buf = io.StringIO("\n".join(lines))
        return pd.read_csv(buf)

def pick_col(df: pd.DataFrame, needles: List[str], fallback: int = 0) -> str:
    cols = list(df.columns)
    cols_l = [str(c).lower() for c in cols]
    for n in needles:
        n = n.lower()
        for c, cl in zip(cols, cols_l):
            if cl == n:
                return c
    for n in needles:
        n = n.lower()
        for c, cl in zip(cols, cols_l):
            if n in cl:
                return c
    return cols[fallback]


# =========================
# ERA5 monthly (CDS API)
# =========================

def _clamp_lat(lat: float) -> float:
    return float(min(90.0, max(-90.0, lat)))

def build_cds_area(lat: float, lon: float, d: float = 0.5) -> List[str]:
    north = _clamp_lat(lat + d/2)
    south = _clamp_lat(lat - d/2)
    if south > north:
        south, north = north, south
    west = lon - d/2
    east = lon + d/2
    return [f"{north:.5f}", f"{west:.5f}", f"{south:.5f}", f"{east:.5f}"]

def last_closed_month_start(today: pd.Timestamp) -> pd.Timestamp:
    last_day_prev = today.replace(day=1) - pd.Timedelta(days=1)
    return pd.Timestamp(year=last_day_prev.year, month=last_day_prev.month, day=1)

def _is_zip_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            sig = f.read(4)
        return sig.startswith(b"PK\x03\x04")
    except Exception:
        return False

def _extract_zip(zip_path: Path, extract_dir: Path) -> List[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    return sorted([p for p in extract_dir.rglob("*.nc")])

def _open_and_merge_netcdfs(nc_files: List[Path]):
    import xarray as xr
    dsets = [xr.open_dataset(f, decode_times=True, engine="netcdf4") for f in nc_files]
    ds = xr.merge(dsets, compat="override", join="outer")
    for d in dsets:
        d.close()
    return ds

def cds_retrieve_era5_monthly(target: Path, area: List[str], start_year: int = 1940) -> None:
    try:
        import cdsapi
    except ImportError as e:
        raise SystemExit("Missing cdsapi. Install with: pip install cdsapi") from e

    if target.exists() and target.stat().st_size > 0:
        print(f"[era5] exists: {target}")
        return

    today = pd.Timestamp.today()
    end = last_closed_month_start(today)
    end_year = int(end.year)
    end_month = int(end.month)

    print(f"[era5] requesting {start_year}-01 through {end_year}-{end_month:02d} (last closed month)")

    client = cdsapi.Client()

    def _request(years: List[str], months: List[str], out: Path) -> None:
        req = {
            "product_type": "monthly_averaged_reanalysis",
            "variable": ERA5_VARS,
            "year": years,
            "month": months,
            "time": "00:00",
            "area": area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        print(f"[era5] retrieve -> {out.name}  years={years[0]}..{years[-1]}  months={months[0]}..{months[-1]}")
        client.retrieve(ERA5_DATASET, req, str(out))

    part1 = target.with_suffix(".part1.bin")
    part2 = target.with_suffix(".part2.bin")

    if end_year > start_year:
        years_full = [str(y) for y in range(start_year, end_year)]
        months_full = [f"{m:02d}" for m in range(1, 13)]
        _request(years_full, months_full, part1)

    years_last = [str(end_year)]
    months_last = [f"{m:02d}" for m in range(1, end_month + 1)]
    _request(years_last, months_last, part2)

    tmp_extract_root = target.parent / (target.stem + "_extract")
    if tmp_extract_root.exists():
        shutil.rmtree(tmp_extract_root, ignore_errors=True)
    tmp_extract_root.mkdir(parents=True, exist_ok=True)

    all_nc_files: List[Path] = []

    for idx, p in enumerate([part1, part2], start=1):
        if not p.exists() or p.stat().st_size == 0:
            continue
        if _is_zip_file(p):
            zdir = tmp_extract_root / f"part{idx}"
            ncs = _extract_zip(p, zdir)
            if not ncs:
                raise RuntimeError(f"CDS returned ZIP but no .nc found inside: {p}")
            all_nc_files.extend(ncs)
        else:
            # might already be NetCDF (just misnamed)
            newp = p.with_suffix(".nc")
            try:
                p.rename(newp)
                p = newp
            except Exception:
                pass
            all_nc_files.append(p)

    if not all_nc_files:
        raise RuntimeError("ERA5 download produced no readable files.")

    ds = _open_and_merge_netcdfs(all_nc_files)
    ds = ds.sortby("time")
    ds.to_netcdf(target)
    ds.close()
    print(f"[era5] wrote merged NetCDF: {target}")

    # Cleanup
    for p in (part1, part2):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    try:
        shutil.rmtree(tmp_extract_root, ignore_errors=True)
    except Exception:
        pass

def lon_to_dataset(ds_lon: np.ndarray, lon: float) -> float:
    lon_min = float(np.nanmin(ds_lon))
    lon_max = float(np.nanmax(ds_lon))
    if lon_min >= 0.0 and lon_max > 180.0:
        return lon % 360.0
    return lon

def load_era5_point(era5_file: Path, proc_dir: Path) -> pd.DataFrame:
    if os.getenv("SKIP_ERA5", "0").strip().lower() in ("1", "true", "yes"):
        print("[era5] SKIP_ERA5=1 -> skipping ERA5")
        return pd.DataFrame()

    area = build_cds_area(LAT, LON, d=0.5)
    cds_retrieve_era5_monthly(era5_file, area=area, start_year=SPLICE_YEAR_ERA5)

    import xarray as xr
    ds = xr.open_dataset(era5_file, decode_times=True, engine="netcdf4")

    def pick(*cands):
        for c in cands:
            if c in ds.data_vars:
                return c
        return None

    v_t2m = pick("t2m", "2m_temperature")
    v_msr = pick("msr", "mean_snowfall_rate")
    v_mser = pick("mser", "mean_snow_evaporation_rate")
    if any(v is None for v in (v_t2m, v_msr, v_mser)):
        raise KeyError(f"ERA5 missing expected vars; have: {list(ds.data_vars)}")

    lon_use = lon_to_dataset(ds["lon"].values, LON)
    pt = ds[[v_t2m, v_msr, v_mser]].sel(lat=_clamp_lat(LAT), lon=lon_use, method="nearest")
    df = pt.to_dataframe().reset_index().rename(columns={v_t2m:"t2m_K", v_msr:"snowfall_rate_kgm2s", v_mser:"snow_evap_rate_kgm2s"})
    ds.close()

    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month

    df["net_accum_rate_kgm2s"] = df["snowfall_rate_kgm2s"] + df["snow_evap_rate_kgm2s"]

    seconds_in_month = df["time"].dt.days_in_month.to_numpy(dtype=float) * 24.0 * 3600.0
    df["net_accum_m_we_month"] = (df["net_accum_rate_kgm2s"].to_numpy() * seconds_in_month) / 1000.0
    df["snowfall_m_we_month"] = (df["snowfall_rate_kgm2s"].to_numpy() * seconds_in_month) / 1000.0
    df["snow_evap_m_we_month"] = (df["snow_evap_rate_kgm2s"].to_numpy() * seconds_in_month) / 1000.0

    df["net_accum_m_iceeq_month"] = df["net_accum_m_we_month"] * (RHO_W / RHO_I)
    df["snowfall_m_iceeq_month"] = df["snowfall_m_we_month"] * (RHO_W / RHO_I)
    df["sublimation_m_iceeq_month"] = (-df["snow_evap_m_we_month"]) * (RHO_W / RHO_I)

    out_csv = proc_dir / "era5_monthly_point.csv"
    df.to_csv(out_csv, index=False)
    print(f"[processed] wrote: {out_csv}")
    return df


# =========================
# USAP manual guidance (reCAPTCHA)
# =========================

def usap_dir(data_dir: Path, uid: str) -> Path:
    d = data_dir / "usap_dc" / uid
    d.mkdir(parents=True, exist_ok=True)
    return d

def parse_usap_filenames_from_html(html: str) -> List[str]:
    exts = ("csv", "txt", "mat", "xlsx", "xls", "zip")
    pattern = r"[A-Za-z0-9_\-]+\.(?:" + "|".join(exts) + r")"
    found = re.findall(pattern, html)
    out, seen = [], set()
    for f in found:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

def print_usap_manual_instructions(data_dir: Path) -> None:
    if os.getenv("SKIP_USAP", "0").strip().lower() in ("1", "true", "yes"):
        print("[usap] SKIP_USAP=1 -> skipping USAP guidance")
        return

    import requests
    print("\n[usap] USAP-DC downloads require reCAPTCHA; please download via browser.")
    for uid, (label, url) in USAP_DATASETS.items():
        outdir = usap_dir(data_dir, uid)
        print(f"\n  - {uid}: {label}")
        print(f"    open: {url}")
        print(f"    save into: {outdir}")
        try:
            html = requests.get(url, timeout=180).text
            files = parse_usap_filenames_from_html(html)
        except Exception:
            files = []
        if files:
            if uid == "601525":
                hint = [f for f in files if "firn" in f.lower() and "temp" in f.lower()] or files
            elif uid == "601680":
                hint = [f for f in files if "borehole" in f.lower() or "pot" in f.lower() or "density" in f.lower()] or files
            else:
                hint = files
            print("    filenames detected on page:")
            for f in hint[:30]:
                print(f"      - {f}")
        else:
            print("    (could not parse filenames; download the dataset ZIP or the CSVs listed under 'Data Files')")


# =========================
# NOAA processing
# =========================

def load_sp19_depth_age(age_file: Path, proc_dir: Path) -> pd.DataFrame:
    df = read_noaa_template_table(age_file)
    depth_col = pick_col(df, ["depth"], 0)
    age_col = pick_col(df, ["age"], 1)
    out = df[[depth_col, age_col]].rename(columns={depth_col: "depth_m", age_col: "age"})
    out["depth_m"] = pd.to_numeric(out["depth_m"], errors="coerce")
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out = out.dropna().sort_values("depth_m").reset_index(drop=True)
    if out["age"].max() > 1000:
        out["age_yrBP"] = out["age"]
        out["year_CE"] = 1950.0 - out["age_yrBP"]
    out_csv = proc_dir / "sp19_depth_age.csv"
    out.to_csv(out_csv, index=False)
    print(f"[processed] wrote: {out_csv}")
    return out

def load_sp19_density(dens_file: Path, proc_dir: Path) -> pd.DataFrame:
    df = read_noaa_template_table(dens_file)
    depth_col = pick_col(df, ["depth"], 0)
    rho_col = pick_col(df, ["dens", "density", "rho"], 1)
    out = df[[depth_col, rho_col]].rename(columns={depth_col: "depth_m", rho_col: "rho_kgm3"})
    out["depth_m"] = pd.to_numeric(out["depth_m"], errors="coerce")
    out["rho_kgm3"] = pd.to_numeric(out["rho_kgm3"], errors="coerce")
    out = out.dropna().sort_values("depth_m").reset_index(drop=True)
    out_csv = proc_dir / "sp19_density.csv"
    out.to_csv(out_csv, index=False)
    print(f"[processed] wrote: {out_csv}")
    return out

def load_buizert_temp_accum(temp_file: Path, accum_file: Path, proc_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dfT = read_noaa_template_table(temp_file)
    dfA = read_noaa_template_table(accum_file)

    tcolT = pick_col(dfT, ["age", "year"], 0)
    vcolT = pick_col(dfT, ["temp", "temperature"], 1)
    outT = dfT[[tcolT, vcolT]].rename(columns={tcolT: "age_yrBP", vcolT: "temp"})
    outT["age_yrBP"] = pd.to_numeric(outT["age_yrBP"], errors="coerce")
    outT["temp"] = pd.to_numeric(outT["temp"], errors="coerce")
    outT = outT.dropna().sort_values("age_yrBP").reset_index(drop=True)
    outT["year_CE"] = 1950.0 - outT["age_yrBP"]
    outT.to_csv(proc_dir / "buizert2021_spice_temp.csv", index=False)

    tcolA = pick_col(dfA, ["age", "year"], 0)
    vcolA = pick_col(dfA, ["accum", "accumulation"], 1)
    outA = dfA[[tcolA, vcolA]].rename(columns={tcolA: "age_yrBP", vcolA: "accum"})
    outA["age_yrBP"] = pd.to_numeric(outA["age_yrBP"], errors="coerce")
    outA["accum"] = pd.to_numeric(outA["accum"], errors="coerce")
    outA = outA.dropna().sort_values("age_yrBP").reset_index(drop=True)
    outA["year_CE"] = 1950.0 - outA["age_yrBP"]
    outA.to_csv(proc_dir / "buizert2021_spice_accum.csv", index=False)

    return outT, outA


# =========================
# Plotting
# =========================

def plot_depth_age_density(depth_age: pd.DataFrame, density: pd.DataFrame, fig_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    if not density.empty:
        ax1.plot(density["rho_kgm3"], density["depth_m"])
        ax1.set_xlabel("Density (kg/m³)")
    ax1.set_ylabel("Depth (m)")
    ax1.invert_yaxis()
    ax1.set_title("SP19 firn density")

    if not depth_age.empty:
        if "year_CE" in depth_age.columns:
            ax2.plot(depth_age["year_CE"], depth_age["depth_m"])
            ax2.set_xlabel("Year (CE)")
        else:
            ax2.plot(depth_age["age"], depth_age["depth_m"])
            ax2.set_xlabel("Age")
        ax2.set_title("SP19 depth–age")

    fig.tight_layout()
    fig.savefig(fig_dir / "sp19_depth_age_density.png", dpi=200)
    plt.close(fig)

def plot_climate(era5: pd.DataFrame, buT: pd.DataFrame, buA: pd.DataFrame, fig_dir: Path) -> None:
    fig, (axT, axA) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    if not era5.empty:
        axT.plot(era5["time"], era5["t2m_K"], label="ERA5 t2m (K)")
        axA.plot(era5["time"], era5["net_accum_m_iceeq_month"], label="ERA5 net (m ice eq / month)")
        axA.plot(era5["time"], era5["snowfall_m_iceeq_month"], "--", alpha=0.7, label="ERA5 snowfall")
        axA.plot(era5["time"], era5["sublimation_m_iceeq_month"], ":", alpha=0.7, label="ERA5 sublimation (mag)")

    if not buT.empty:
        bt = buT[buT["year_CE"] >= 1550].copy()
        if not bt.empty:
            axT.plot(pd.to_datetime(bt["year_CE"].astype(int).astype(str)) + pd.Timedelta(days=182),
                     bt["temp"], label="Buizert2021 SPICE temp (last 400y)")
    if not buA.empty:
        ba = buA[buA["year_CE"] >= 1550].copy()
        if not ba.empty:
            axA.plot(pd.to_datetime(ba["year_CE"].astype(int).astype(str)) + pd.Timedelta(days=182),
                     ba["accum"], label="Buizert2021 SPICE accum (last 400y)")

    axT.set_ylabel("Temperature")
    axT.legend()
    axA.set_ylabel("Accumulation")
    axA.set_xlabel("Time")
    axA.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "south_pole_climate_quicklook.png", dpi=200)
    plt.close(fig)


# =========================
# Main
# =========================

def main() -> None:
    root_dir, data_dir, proc_dir, fig_dir, noaa_dir, era5_dir = get_repo_dirs()
    print(f"[site] {SITE_NAME}")
    print(f"[paths] root={root_dir}")
    print(f"        data={data_dir}  processed={proc_dir}  figures={fig_dir}")
    print(f"[coords] SPICEcore lat/lon={LAT:.2f}, {LON:.2f}; USP50 lat/lon={USP50_LAT:.2f}, {USP50_LON:.2f}")

    # NOAA downloads
    sp19_age_file = noaa_dir / "sp19age.txt"
    sp19_dens_file = noaa_dir / "sp19dens.txt"
    bu_temp_file = noaa_dir / "buizert2021spice-temp-noaa.txt"
    bu_accum_file = noaa_dir / "buizert2021spice-accum-noaa.txt"

    download_file(NOAA_SP19_AGE_URL, sp19_age_file)
    download_file(NOAA_SP19_DENS_URL, sp19_dens_file)
    download_file(NOAA_BUIZERT_SPICE_TEMP_URL, bu_temp_file)
    download_file(NOAA_BUIZERT_SPICE_ACCUM_URL, bu_accum_file)

    # ERA5
    era5_file = era5_dir / f"ERA5_monthly_south_pole_{SPLICE_YEAR_ERA5}_present.nc"
    era5 = load_era5_point(era5_file, proc_dir)

    # NOAA processing
    depth_age = load_sp19_depth_age(sp19_age_file, proc_dir)
    density = load_sp19_density(sp19_dens_file, proc_dir)
    buT, buA = load_buizert_temp_accum(bu_temp_file, bu_accum_file, proc_dir)

    # Figures
    plot_depth_age_density(depth_age, density, fig_dir)
    plot_climate(era5, buT, buA, fig_dir)

    # USAP guidance
    print_usap_manual_instructions(data_dir)

    print("\nDone.")

if __name__ == "__main__":
    main()
