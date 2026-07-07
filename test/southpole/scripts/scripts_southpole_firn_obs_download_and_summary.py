#!/usr/bin/env python3
"""
southpole/scripts/southpole_firn_obs_download_and_summary.py

South Pole / SPICEcore firn/ice observations: download + processing + quick-look plots.

Directory layout (expected):
southpole/
├── data
├── figures
├── processed
└── scripts

This script will:
  - Download raw files into southpole/data/ (with subfolders per source)
  - Write cleaned CSVs into southpole/processed/
  - Write PNG figures into southpole/figures/

Data streams targeted (best available public sources):
  1) Strain/compaction time series: USP50 borehole shortening (USAP-DC 601680)  [may need manual download]
  2) Firn compaction: same as (1)
  3) Accumulation:
       - Buizert et al. 2021 SPICEcore accumulation reconstruction (NOAA/NCEI)
       - ERA5 monthly net snowfall - sublimation (1940–present) (CDS; needs ~/.cdsapirc)
  4) Englacial/firn temperature:
       - Buizert et al. 2021 SPICEcore temperature reconstruction (NOAA/NCEI)
       - ERA5 monthly 2m air temperature (CDS)
       - USP50 firn thermistor string (USAP-DC 601525) [may need manual download]
  5) Grain size:
       - USAP-DC 601551 in-situ compilation (optional; often needs manual download)
  6) Depth-age stratigraphy + firn density:
       - SP19 age model (NOAA/NCEI) + SP19 density profile (NOAA/NCEI)

Requirements:
  pip install numpy pandas xarray netcdf4 matplotlib requests
Optional:
  pip install beautifulsoup4  (for USAP-DC link scraping)
ERA5:
  pip install cdsapi
  Create ~/.cdsapirc (Copernicus CDS API key)
  Docs: https://cds.climate.copernicus.eu/how-to-api

NOAA/NCEI direct files used:
  - SP19 age model:        https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/sp19age.txt
  - SP19 firn density:     https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/sp19dens.txt
  - Buizert2021 temp:      https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/buizert2021/buizert2021spice-temp-noaa.txt
  - Buizert2021 accum:     https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/buizert2021/buizert2021spice-accum-noaa.txt

USAP-DC dataset pages:
  - 601680 (USP50 compaction/strain): https://www.usap-dc.org/view/dataset/601680
  - 601525 (USP50 firn temps):        https://www.usap-dc.org/view/dataset/601525
  - 601551 (grain size compilation):  https://www.usap-dc.org/view/dataset/601551

Usage:
  python southpole_firn_obs_download_and_summary.py

Environment toggles:
  SKIP_ERA5=1     Skip ERA5 download/processing
  SKIP_USAP=1     Skip USAP-DC scraping attempts (manual-only workflow)
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Site + Sources
# =========================

SITE_NAME = "South Pole / SPICEcore (South Pole Station vicinity)"

# SPICEcore coords used in Buizert2021 metadata (approx):
LAT = -89.99
LON = -99.16  # degrees East (negative = West)

# USP50 site coords used in Stevens et al. datasets (not required for downloads here, but kept for reference)
USP50_LAT = -89.54
USP50_LON = 137.04  # degrees East

SPLICE_YEAR_ERA5 = 1940
BASELINE_START = 1950
BASELINE_END = 1980

# Densities for m w.e. -> m ice eq
RHO_W = 1000.0
RHO_I = 917.0

# NOAA / NCEI direct URLs
NOAA_SP19_AGE_URL = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/sp19age.txt"
NOAA_SP19_DENS_URL = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/sp19dens.txt"
NOAA_BUIZERT_SPICE_TEMP_URL = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/buizert2021/buizert2021spice-temp-noaa.txt"
NOAA_BUIZERT_SPICE_ACCUM_URL = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/buizert2021/buizert2021spice-accum-noaa.txt"

# USAP-DC datasets (may require manual download)
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

def get_repo_dirs() -> Tuple[Path, Path, Path, Path]:
    scripts_dir = Path(__file__).resolve().parent
    root_dir = scripts_dir.parent  # southpole/
    data_dir = root_dir / "data"
    fig_dir = root_dir / "figures"
    proc_dir = root_dir / "processed"
    for d in (data_dir, fig_dir, proc_dir):
        d.mkdir(parents=True, exist_ok=True)
    return root_dir, data_dir, proc_dir, fig_dir


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
    # Most NOAA tables are whitespace delimited
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

def download_era5_monthly_if_needed(era5_file: Path) -> None:
    if era5_file.exists() and era5_file.stat().st_size > 0:
        print(f"[era5] exists: {era5_file}")
        return
    try:
        import cdsapi
    except ImportError as e:
        raise SystemExit("Missing cdsapi. Install with: pip install cdsapi") from e

    end_year = pd.Timestamp.today().year
    years = [str(y) for y in range(SPLICE_YEAR_ERA5, end_year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]

    # Small bbox around South Pole. CDS expects [N, W, S, E]
    d = 0.5
    area = [LAT + d/2, LON - d/2, LAT - d/2, LON + d/2]

    req = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": ERA5_VARS,
        "year": years,
        "month": months,
        "time": "00:00",
        "area": area,
        "format": "netcdf",
    }

    print("[era5] requesting ERA5 monthly means from CDS...")
    c = cdsapi.Client()
    c.retrieve(ERA5_DATASET, req, str(era5_file))
    print(f"[era5] wrote: {era5_file}")


def lon_to_dataset(ds_lon: np.ndarray, lon: float) -> float:
    lon_min = float(np.nanmin(ds_lon))
    lon_max = float(np.nanmax(ds_lon))
    if lon_min >= 0.0 and lon_max > 180.0:
        return lon % 360.0
    return lon


def load_era5_point(era5_file: Path, proc_dir: Path) -> pd.DataFrame:
    import xarray as xr

    if os.getenv("SKIP_ERA5", "0").strip() in ("1", "true", "yes"):
        print("[era5] SKIP_ERA5=1 -> skipping ERA5")
        return pd.DataFrame()

    download_era5_monthly_if_needed(era5_file)
    ds = xr.open_dataset(era5_file, decode_times=True)

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
    pt = ds[[v_t2m, v_msr, v_mser]].sel(lat=LAT, lon=lon_use, method="nearest")

    df = pt.to_dataframe().reset_index().rename(columns={
        v_t2m: "t2m_K",
        v_msr: "snowfall_rate_kgm2s",
        v_mser: "snow_evap_rate_kgm2s",
    })
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month

    # ECMWF convention: downward flux positive; snow evaporation is negative for sublimation
    df["net_accum_rate_kgm2s"] = df["snowfall_rate_kgm2s"] + df["snow_evap_rate_kgm2s"]

    seconds_in_month = df["time"].dt.days_in_month.to_numpy(dtype=float) * 24.0 * 3600.0
    # 1 kg/m^2 = 1 mm w.e.
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
# USAP-DC: best-effort link scraping
# =========================

def usap_dir(data_dir: Path, uid: str) -> Path:
    d = data_dir / "usap_dc" / uid
    d.mkdir(parents=True, exist_ok=True)
    return d


def try_usap_scrape_download(uid: str, url: str, outdir: Path, max_files: int = 50) -> List[Path]:
    """
    Attempts to scrape direct file links from a USAP-DC dataset page HTML.
    If it fails (reCAPTCHA / blocked), returns [] and prints manual instructions.
    """
    if os.getenv("SKIP_USAP", "0").strip() in ("1", "true", "yes"):
        print(f"[usap] SKIP_USAP=1 -> skipping scrape for {uid}")
        return []

    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        print("[usap] beautifulsoup4 not installed; skipping scrape. Install: pip install beautifulsoup4")
        return []

    import requests

    print(f"[usap] fetching dataset page {uid} ...")
    try:
        html = requests.get(url, timeout=180).text
    except Exception as e:
        print(f"[usap] FAILED to fetch dataset page {uid}: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    exts = (".csv", ".txt", ".mat", ".xlsx", ".xls", ".zip", ".tar", ".tar.gz")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = (a.get_text() or "").strip()
        if any(href.lower().endswith(ext) for ext in exts) or any(text.lower().endswith(ext) for ext in exts):
            links.append(href)

    # Normalize / de-dup
    norm = []
    for href in links:
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://www.usap-dc.org" + href
        elif href.startswith("http"):
            pass
        else:
            href = url.rstrip("/") + "/" + href
        norm.append(href)
    norm = list(dict.fromkeys(norm))

    downloaded: List[Path] = []
    for href in norm[:max_files]:
        fn = href.split("/")[-1].split("?")[0]
        if not any(fn.lower().endswith(ext) for ext in exts):
            continue
        out = outdir / fn
        if out.exists() and out.stat().st_size > 0:
            downloaded.append(out)
            continue
        try:
            r = requests.get(href, stream=True, timeout=180)
            if r.status_code != 200:
                continue
            tmp = out.with_suffix(out.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=2**20):
                    if chunk:
                        f.write(chunk)
            tmp.replace(out)
            if out.stat().st_size > 0:
                downloaded.append(out)
        except Exception:
            continue

    if len(downloaded) == 0:
        print(f"[usap] Could not auto-download files for dataset {uid}.")
        print(f"       Manual: open {url}")
        print(f"       Download files into: {outdir}")
    else:
        print(f"[usap] downloaded {len(downloaded)} file(s) for {uid} -> {outdir}")

    return downloaded


# =========================
# NOAA loaders -> processed CSVs
# =========================

def load_sp19_depth_age(noaa_age_file: Path, proc_dir: Path) -> pd.DataFrame:
    df = read_noaa_template_table(noaa_age_file)
    depth_col = pick_col(df, ["depth"], 0)
    age_col = pick_col(df, ["age"], 1)

    out = df[[depth_col, age_col]].rename(columns={depth_col: "depth_m", age_col: "age"})
    out["depth_m"] = pd.to_numeric(out["depth_m"], errors="coerce")
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out = out.dropna().sort_values("depth_m").reset_index(drop=True)

    # If ages look like BP, add CE
    if out["age"].max() > 1000:
        out["age_yrBP"] = out["age"]
        out["year_CE"] = 1950.0 - out["age_yrBP"]

    out_csv = proc_dir / "sp19_depth_age.csv"
    out.to_csv(out_csv, index=False)
    print(f"[processed] wrote: {out_csv}")
    return out


def load_sp19_density(noaa_dens_file: Path, proc_dir: Path) -> pd.DataFrame:
    df = read_noaa_template_table(noaa_dens_file)
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

    print(f"[processed] wrote: {proc_dir/'buizert2021_spice_temp.csv'}")
    print(f"[processed] wrote: {proc_dir/'buizert2021_spice_accum.csv'}")
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
    out = fig_dir / "sp19_depth_age_density.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[figure] wrote: {out}")


def plot_climate(era5: pd.DataFrame, buT: pd.DataFrame, buA: pd.DataFrame, fig_dir: Path) -> None:
    fig, (axT, axA) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    if not era5.empty:
        axT.plot(era5["time"], era5["t2m_K"], label="ERA5 t2m (K)")
        axA.plot(era5["time"], era5["net_accum_m_iceeq_month"], label="ERA5 net (m ice eq / month)")
        axA.plot(era5["time"], era5["snowfall_m_iceeq_month"], "--", alpha=0.7, label="ERA5 snowfall")
        axA.plot(era5["time"], era5["sublimation_m_iceeq_month"], ":", alpha=0.7, label="ERA5 sublimation (mag)")

    # Buizert: show last 400 years (for visual comparison)
    if not buT.empty:
        bt = buT[buT["year_CE"] >= 1550].copy()
        if not bt.empty:
            axT.plot(pd.to_datetime(bt["year_CE"].astype(int).astype(str)) + pd.Timedelta(days=182),
                     bt["temp"], label="Buizert2021 SPICE temp (last 400y; units as provided)")
    if not buA.empty:
        ba = buA[buA["year_CE"] >= 1550].copy()
        if not ba.empty:
            axA.plot(pd.to_datetime(ba["year_CE"].astype(int).astype(str)) + pd.Timedelta(days=182),
                     ba["accum"], label="Buizert2021 SPICE accum (last 400y; units as provided)")

    axT.set_ylabel("Temperature")
    axT.legend(loc="best")
    axT.set_title("South Pole: climate quick-look (ERA5 + Buizert2021 subset)")

    axA.set_ylabel("Accumulation")
    axA.set_xlabel("Time")
    axA.legend(loc="best")

    fig.tight_layout()
    out = fig_dir / "south_pole_climate_quicklook.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[figure] wrote: {out}")


# =========================
# Main
# =========================

def main() -> None:
    root_dir, data_dir, proc_dir, fig_dir = get_repo_dirs()
    print(f"[site] {SITE_NAME}")
    print(f"[paths] root={root_dir}")
    print(f"        data={data_dir}  processed={proc_dir}  figures={fig_dir}")
    print(f"[coords] SPICEcore lat/lon={LAT:.2f}, {LON:.2f}; USP50 lat/lon={USP50_LAT:.2f}, {USP50_LON:.2f}")

    # --- Raw data locations in data/ ---
    noaa_dir = data_dir / "noaa_ncei"
    noaa_dir.mkdir(parents=True, exist_ok=True)

    sp19_age_file = noaa_dir / "sp19age.txt"
    sp19_dens_file = noaa_dir / "sp19dens.txt"
    bu_temp_file = noaa_dir / "buizert2021spice-temp-noaa.txt"
    bu_accum_file = noaa_dir / "buizert2021spice-accum-noaa.txt"

    # NOAA downloads
    download_file(NOAA_SP19_AGE_URL, sp19_age_file)
    download_file(NOAA_SP19_DENS_URL, sp19_dens_file)
    download_file(NOAA_BUIZERT_SPICE_TEMP_URL, bu_temp_file)
    download_file(NOAA_BUIZERT_SPICE_ACCUM_URL, bu_accum_file)

    # USAP-DC (best-effort scrape)
    if os.getenv("SKIP_USAP", "0").strip() not in ("1", "true", "yes"):
        for uid, (label, url) in USAP_DATASETS.items():
            outdir = usap_dir(data_dir, uid)
            try_usap_scrape_download(uid, url, outdir)

    # ERA5
    era5_dir = data_dir / "era5"
    era5_dir.mkdir(parents=True, exist_ok=True)
    era5_file = era5_dir / f"ERA5_monthly_south_pole_{SPLICE_YEAR_ERA5}_present.nc"
    era5 = load_era5_point(era5_file, proc_dir)

    # Process NOAA -> processed/
    depth_age = load_sp19_depth_age(sp19_age_file, proc_dir)
    density = load_sp19_density(sp19_dens_file, proc_dir)
    buT, buA = load_buizert_temp_accum(bu_temp_file, bu_accum_file, proc_dir)

    # Figures -> figures/
    plot_depth_age_density(depth_age, density, fig_dir)
    plot_climate(era5, buT, buA, fig_dir)

    # Guidance about manual USAP downloads
    print("\n[usap] manual download targets (if not present):")
    for uid, (label, url) in USAP_DATASETS.items():
        d = usap_dir(data_dir, uid)
        if len(list(d.glob("*"))) == 0:
            print(f"  - {uid}: {label}")
            print(f"    open: {url}")
            print(f"    save files into: {d}")

    print("\nDone.")

if __name__ == "__main__":
    main()
