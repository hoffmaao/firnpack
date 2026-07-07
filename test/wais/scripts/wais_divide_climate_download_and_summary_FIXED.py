#!/usr/bin/env python3
"""
WAIS Divide climate forcing builder + summary plot (robust parsing)

Fixes:
- Robust parsing for wais2008accum.txt (variable number of columns / malformed rows) by
  scanning numeric tokens per line instead of using pandas read_csv with a fixed column count.

Pre-1940:
  - Temperature option A: LMR Seasonal (tas_mean.nc) at WAIS Divide, rebased to 1950–1980,
    downscaled to monthly by repeating each seasonal anomaly across 3 months and adding
    ERA5 1950–1980 monthly climatology.
  - Temperature option B: Core isotope (WDC05A d18O annual) -> annual temperature anomaly
    via regression to ERA5 annual mean over overlap, then downscaled to monthly by adding
    an ERA5 climatological seasonal cycle (preserving the annual mean).
  - Accumulation: annual (layer/annual) accumulation from wais2008accum.txt, distributed
    to months using ERA5 1950–1980 monthly net-accum fractions.

Post-1940:
  - Temperature: ERA5 monthly t2m
  - Accumulation: ERA5 net = snowfall - sublimation = snowfall_rate + snow_evaporation_rate
    (ECMWF sign convention: downward flux positive; negative values indicate evaporation/sublimation)

Outputs:
  - Downloads (LMR, NOAA text files, ERA5)
  - Extracted CSVs
  - Merged monthly forcing CSV + NetCDF
  - 2-row PNG plot (temperature; accumulation)

Requirements:
  pip install numpy pandas xarray netcdf4 matplotlib requests cdsapi

ERA5:
  Requires ~/.cdsapirc (Copernicus CDS API key).
  Dataset: "ERA5 monthly averaged data on single levels from 1940 to present"
  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means

LMR Seasonal:
  https://atmos.uw.edu/~zilumeng/LMR_Seasonal/

NOAA files:
  - WDC05A d18O (Steig et al. 2013): https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/wdc05a2013d18o.txt
  - WAIS accumulation (2008): https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/wais2008accum.txt
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# =========================
# Config
# =========================

SITE_NAME = "WAIS Divide (WDC05A/WDC06A area)"

# WAIS Divide site (approx.)
LAT = -79.46
LON = -112.09  # degrees East (negative = West)

SPLICE_YEAR = 1940

# baseline for anomalies + climatologies
BASELINE_START = 1950
BASELINE_END = 1980

# calibration window for isotope->T regression (must be within isotope availability and ERA5)
CALIB_START = 1940
CALIB_END = 2005

OUTDIR = Path("../source")
DL_DIR = OUTDIR / "."
OUTDIR.mkdir(parents=True, exist_ok=True)
DL_DIR.mkdir(parents=True, exist_ok=True)

# ---- URLs ----
LMR_TAS_URL = "https://atmos.uw.edu/~zilumeng/LMR_Seasonal/data/mean/tas_mean.nc"
WDC05A_D18O_URL = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/wdc05a2013d18o.txt"
WAIS_ACCUM_URL = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/wais2008accum.txt"

# ---- Local download paths ----
LMR_FILE = DL_DIR / "LMR_Seasonal_tas_mean.nc"
D18O_FILE = DL_DIR / "wdc05a2013d18o.txt"
ACCUM_FILE = DL_DIR / "wais2008accum.txt"
ERA5_FILE = DL_DIR / "ERA5_monthly_wais_divide_1940_present.nc"

# ---- ERA5 ----
ERA5_DATASET = "reanalysis-era5-single-levels-monthly-means"
ERA5_VARS = [
    "2m_temperature",
    "mean_snowfall_rate",
    "mean_snow_evaporation_rate",
]

# ---- Outputs ----
CSV_LMR_POINT = OUTDIR / "wais_lmr_seasonal_point.csv"
CSV_D18O_ANNUAL = OUTDIR / "wais_d18o_annual.csv"
CSV_ACCUM_ANNUAL = OUTDIR / "wais_accum_annual.csv"
CSV_ERA5_POINT = OUTDIR / "wais_era5_monthly_point.csv"

CSV_MERGED = OUTDIR / "wais_monthly_forcing_preLMR+layers_postERA5.csv"
NC_MERGED = OUTDIR / "wais_monthly_forcing_preLMR+layers_postERA5.nc"
PLOT_FILE = OUTDIR / "wais_temperature_and_accumulation_summary.png"

# densities for m w.e. -> m ice eq
RHO_W = 1000.0
RHO_I = 917.0


# =========================
# Utilities
# =========================

def download_file(url: str, path: Path, overwrite: bool = False) -> None:
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        print(f"[download] exists: {path.name}")
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
    print(f"[download] wrote: {path.name}")

def lon_to_dataset(ds: xr.Dataset, lon: float) -> float:
    lon_vals = ds["lon"].values
    lon_min = float(np.nanmin(lon_vals))
    lon_max = float(np.nanmax(lon_vals))
    if lon_min >= 0.0 and lon_max > 180.0:
        return lon % 360.0
    return lon

def safe_timestamp(year: int, month: int, day: int = 15) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=int(month), day=int(day))

def month_quarter(season_month: int) -> list[int]:
    # assumes season_month is quarter-start month in {1,4,7,10}
    m = int(season_month)
    return [m, m + 1, m + 2]

_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?")

def _decode_bytes(raw: bytes) -> str:
    # Handle odd encodings gracefully
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")

def parse_noaa_numeric_lines(raw: bytes) -> list[list[float]]:
    """
    Return a list of numeric-token lists, one per non-comment line.
    This is robust to ragged rows / extra columns.
    """
    txt = _decode_bytes(raw)
    rows = []
    for ln in txt.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        # Keep only lines that contain at least one digit
        if not any(ch.isdigit() for ch in s):
            continue
        nums = _NUM_RE.findall(s)
        if len(nums) == 0:
            continue
        try:
            rows.append([float(x) for x in nums])
        except Exception:
            continue
    return rows

def pick_year_and_value(rows: list[list[float]],
                        year_range=(1000, 3000),
                        prefer_int=True) -> pd.DataFrame:
    """
    Heuristic: from each row, find a plausible year (CE) and take the next numeric as value.
    - year_range: (min,max) for year
    - prefer_int: require the year to be close to an integer
    """
    out_year = []
    out_val = []
    for r in rows:
        if len(r) < 2:
            continue

        year_idx = None
        year_val = None

        # First pass: look for integer-ish years in range
        for i, v in enumerate(r):
            if (year_range[0] <= v <= year_range[1]):
                if prefer_int:
                    if abs(v - round(v)) < 1e-6:
                        year_idx = i
                        year_val = int(round(v))
                        break
                else:
                    year_idx = i
                    year_val = int(round(v))
                    break

        if year_idx is None:
            # Second pass: accept the first value if it's plausible-ish
            v0 = r[0]
            if year_range[0] <= v0 <= year_range[1]:
                year_idx = 0
                year_val = int(round(v0))

        if year_idx is None:
            continue
        if year_idx + 1 >= len(r):
            continue

        val = r[year_idx + 1]

        # Filter obviously crazy years/values
        if year_val < year_range[0] or year_val > year_range[1]:
            continue
        if not np.isfinite(val):
            continue

        out_year.append(year_val)
        out_val.append(float(val))

    df = pd.DataFrame({"year": out_year, "value": out_val})
    df = df.dropna().sort_values("year").reset_index(drop=True)
    # Deduplicate by year (take mean if duplicates)
    if not df.empty:
        df = df.groupby("year", as_index=False)["value"].mean()
    return df


# =========================
# LMR Seasonal temperature (site point)
# =========================

def load_lmr_seasonal_point() -> pd.DataFrame:
    download_file(LMR_TAS_URL, LMR_FILE)
    ds = xr.open_dataset(LMR_FILE, decode_times=True)
    if "tas" not in ds:
        raise KeyError(f"LMR file missing 'tas'. Vars: {list(ds.data_vars)}")

    lon_use = lon_to_dataset(ds, LON)
    pt = ds[["tas"]].sel(lat=LAT, lon=lon_use, method="nearest")

    # cftime-safe extraction
    tvals = pt["time"].values
    years = np.array([int(tv.year) for tv in tvals], dtype=int)
    months = np.array([int(tv.month) for tv in tvals], dtype=int)

    # rebase to 1950–1980 mean (across seasonal samples)
    base_mask = (years >= BASELINE_START) & (years <= BASELINE_END)
    if not np.any(base_mask):
        raise ValueError("LMR baseline years not present; cannot rebase.")
    tas_base = float(pt["tas"].isel(time=np.where(base_mask)[0]).mean().values)

    tas_anom = pt["tas"].values.astype(float) - tas_base

    df = pd.DataFrame({"year": years, "season_month": months, "tas_anom_K": tas_anom})
    df = df[df["year"] < SPLICE_YEAR].copy().reset_index(drop=True)

    df.to_csv(CSV_LMR_POINT, index=False)
    print(f"[csv] wrote: {CSV_LMR_POINT}")
    return df


# =========================
# Ice-core d18O annual (WDC05A)
# =========================

def load_wdc05a_d18o_annual() -> pd.DataFrame:
    download_file(WDC05A_D18O_URL, D18O_FILE)
    raw = D18O_FILE.read_bytes()
    rows = parse_noaa_numeric_lines(raw)

    # Many NOAA isotope tables are "year  d18O" with optional extra columns.
    df = pick_year_and_value(rows, year_range=(0, 3000), prefer_int=True)
    if df.empty:
        raise ValueError("Failed to parse any rows from WDC05A isotope file.")

    df = df.rename(columns={"value": "d18O"})
    df = df[df["year"] < 3000].copy()
    df.to_csv(CSV_D18O_ANNUAL, index=False)
    print(f"[csv] wrote: {CSV_D18O_ANNUAL}")
    return df


# =========================
# Annual accumulation record (wais2008accum.txt) — robust parsing
# =========================

def load_wais_accum_annual() -> pd.DataFrame:
    download_file(WAIS_ACCUM_URL, ACCUM_FILE)
    raw = ACCUM_FILE.read_bytes()

    rows = parse_noaa_numeric_lines(raw)

    # Try strict "year CE" first (most likely)
    df = pick_year_and_value(rows, year_range=(1500, 2500), prefer_int=True)

    # If that fails, relax year range (some files use 0..2000)
    if df.empty:
        df = pick_year_and_value(rows, year_range=(0, 3000), prefer_int=True)

    if df.empty:
        # As a last resort, accept non-integer-ish years
        df = pick_year_and_value(rows, year_range=(0, 3000), prefer_int=False)

    if df.empty:
        raise ValueError(
            "Failed to parse accumulation file (wais2008accum.txt). "
            "If this persists, download the .xls version and export as CSV."
        )

    # The parsed "value" could be in kg/m^2/yr (or mm w.e./yr) or m w.e./yr.
    med = float(np.nanmedian(df["value"].values))
    if med > 10.0:
        # Treat as kg/m^2/yr or mm w.e./yr -> m w.e./yr
        accum_m_we_yr = df["value"] / 1000.0
        unit_note = "interpreted as kg m-2 yr-1 (or mm w.e./yr)"
    else:
        # Treat as m w.e./yr
        accum_m_we_yr = df["value"]
        unit_note = "interpreted as m w.e. yr-1"

    out = df.rename(columns={"value": "accum_raw"}).copy()
    out["accum_m_iceeq_yr"] = accum_m_we_yr * (RHO_W / RHO_I)
    out["unit_note"] = unit_note

    out = out[out["year"] < SPLICE_YEAR].copy().reset_index(drop=True)

    # Debug print
    print(f"[accum] parsed {len(out)} annual rows from wais2008accum.txt ({unit_note})")
    if len(out) > 0:
        print(f"[accum] year range: {int(out.year.min())} .. {int(out.year.max())} | median raw={med:g}")

    out.to_csv(CSV_ACCUM_ANNUAL, index=False)
    print(f"[csv] wrote: {CSV_ACCUM_ANNUAL}")
    return out


# =========================
# ERA5 monthly point series (1940+)
# =========================

def download_era5_monthly_if_needed() -> None:
    if ERA5_FILE.exists() and ERA5_FILE.stat().st_size > 0:
        print(f"[era5] exists: {ERA5_FILE.name}")
        return

    try:
        import cdsapi
    except ImportError as e:
        raise SystemExit(
            "Missing cdsapi. Install with:\n"
            "  pip install cdsapi\n"
            "Then configure ~/.cdsapirc (Copernicus CDS API key)."
        ) from e

    end_year = pd.Timestamp.today().year
    years = [str(y) for y in range(SPLICE_YEAR, end_year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]

    # small bbox around site (CDS expects [N,W,S,E])
    d = 0.5
    area = [LAT + d / 2, LON - d / 2, LAT - d / 2, LON + d / 2]

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
    c.retrieve(ERA5_DATASET, req, str(ERA5_FILE))
    print(f"[era5] wrote: {ERA5_FILE.name}")

def load_era5_monthly_point() -> pd.DataFrame:
    download_era5_monthly_if_needed()
    ds = xr.open_dataset(ERA5_FILE, decode_times=True)

    def pick(*cands):
        for c in cands:
            if c in ds.data_vars:
                return c
        return None

    v_t2m = pick("t2m", "2m_temperature")
    v_msr = pick("msr", "mean_snowfall_rate")
    v_mser = pick("mser", "mean_snow_evaporation_rate")
    if any(v is None for v in (v_t2m, v_msr, v_mser)):
        raise KeyError(f"ERA5 missing vars. Have: {list(ds.data_vars)}")

    lon_use = lon_to_dataset(ds, LON)
    pt = ds[[v_t2m, v_msr, v_mser]].sel(lat=LAT, lon=lon_use, method="nearest")

    df = pt.to_dataframe().reset_index().rename(columns={
        v_t2m: "t2m_K",
        v_msr: "snowfall_rate_kgm2s",
        v_mser: "snow_evap_rate_kgm2s",
    })
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month

    # net = snowfall - sublimation; snow_evap_rate is typically negative for sublimation
    df["net_accum_rate_kgm2s"] = df["snowfall_rate_kgm2s"] + df["snow_evap_rate_kgm2s"]

    seconds_in_month = df["time"].dt.days_in_month.to_numpy(dtype=float) * 24.0 * 3600.0
    df["snowfall_m_we_month"] = (df["snowfall_rate_kgm2s"].to_numpy() * seconds_in_month) / 1000.0
    df["snow_evap_m_we_month"] = (df["snow_evap_rate_kgm2s"].to_numpy() * seconds_in_month) / 1000.0
    df["net_accum_m_we_month"] = (df["net_accum_rate_kgm2s"].to_numpy() * seconds_in_month) / 1000.0

    df["snowfall_m_iceeq_month"] = df["snowfall_m_we_month"] * (RHO_W / RHO_I)
    df["snow_evap_m_iceeq_month"] = df["snow_evap_m_we_month"] * (RHO_W / RHO_I)
    df["net_accum_m_iceeq_month"] = df["net_accum_m_we_month"] * (RHO_W / RHO_I)

    df.to_csv(CSV_ERA5_POINT, index=False)
    print(f"[csv] wrote: {CSV_ERA5_POINT}")
    return df


# =========================
# Build pre-ERA monthly (two temperature options + annual accumulation)
# =========================

def build_preera_monthly(
    lmr_seasonal: pd.DataFrame,
    d18o_annual: pd.DataFrame,
    accum_annual: pd.DataFrame,
    era5_monthly: pd.DataFrame,
) -> pd.DataFrame:

    # ERA5 baseline monthly climatology for absolute temperature (1950–1980)
    base = era5_monthly[
        (era5_monthly["year"] >= BASELINE_START) & (era5_monthly["year"] <= BASELINE_END)
    ].copy()
    if base.empty:
        raise ValueError("ERA5 baseline (1950–1980) not present in ERA5 monthly data.")

    t_clim_month = base.groupby("month")["t2m_K"].mean().to_dict()
    t_clim_month_mean = float(np.mean(list(t_clim_month.values())))

    # ERA5 baseline monthly net-accum climatology for distributing annual accumulation
    a_clim = base.groupby("month")["net_accum_m_iceeq_month"].mean()
    a_pos = np.clip(a_clim.to_numpy(dtype=float), 0.0, None)
    frac = (a_pos / np.sum(a_pos)) if np.sum(a_pos) > 0 else np.ones(12) / 12.0
    frac_by_month = {m: float(frac[m - 1]) for m in range(1, 13)}

    # ---- Temperature option A: LMR seasonal anomaly -> monthly absolute T ----
    rows_lmr = []
    for _, r in lmr_seasonal.iterrows():
        y = int(r["year"])
        sm = int(r["season_month"])
        anom = float(r["tas_anom_K"])
        for m in month_quarter(sm):
            rows_lmr.append({
                "time": safe_timestamp(y, m, 15),
                "year": y,
                "month": m,
                "t2m_K_lmr": anom + float(t_clim_month[m]),
            })
    df_lmr_m = pd.DataFrame(rows_lmr).drop_duplicates(subset=["time"]).sort_values("time")

    # ---- Temperature option B: d18O annual -> annual T anomaly via regression to ERA5 annual ----
    era_ann = era5_monthly.groupby("year", as_index=False)["t2m_K"].mean().rename(columns={"t2m_K": "t2m_K_ann"})

    era_base_ann = era_ann[(era_ann["year"] >= BASELINE_START) & (era_ann["year"] <= BASELINE_END)]
    if era_base_ann.empty:
        raise ValueError("ERA5 annual baseline (1950–1980) is empty; cannot calibrate core temperature.")
    t0 = float(era_base_ann["t2m_K_ann"].mean())

    d18_base = d18o_annual[(d18o_annual["year"] >= BASELINE_START) & (d18o_annual["year"] <= BASELINE_END)]
    if d18_base.empty:
        raise ValueError("Core isotope baseline (1950–1980) is empty; cannot calibrate.")
    d0 = float(d18_base["d18O"].mean())

    overlap = d18o_annual.merge(era_ann, on="year", how="inner")
    overlap = overlap[(overlap["year"] >= CALIB_START) & (overlap["year"] <= CALIB_END)].copy()
    if overlap.empty:
        raise ValueError("No overlap for isotope->T calibration (check CALIB_START/CALIB_END).")

    x = (overlap["d18O"] - d0).to_numpy(dtype=float)
    y = (overlap["t2m_K_ann"] - t0).to_numpy(dtype=float)
    if np.nanvar(x) <= 0:
        raise ValueError("Cannot calibrate isotope->T: d18O variance in overlap is zero.")

    slope = np.nanmean((x - np.nanmean(x)) * (y - np.nanmean(y))) / np.nanvar(x)  # K per permil
    print(f"[calib] d18O→T slope = {slope:.4f} K per ‰ (ERA annual, {CALIB_START}-{CALIB_END})")

    d18_pre = d18o_annual[d18o_annual["year"] < SPLICE_YEAR].copy()
    d18_pre["t2m_K_ann_core"] = t0 + slope * (d18_pre["d18O"] - d0)

    rows_core = []
    for _, r in d18_pre.iterrows():
        y = int(r["year"])
        t_ann = float(r["t2m_K_ann_core"])
        for m in range(1, 13):
            # Add a climatological seasonal cycle but preserve annual mean t_ann
            t_month = t_ann + (float(t_clim_month[m]) - t_clim_month_mean)
            rows_core.append({
                "time": safe_timestamp(y, m, 15),
                "year": y,
                "month": m,
                "t2m_K_core": t_month,
            })
    df_core_m = pd.DataFrame(rows_core).sort_values("time")

    # ---- Annual accumulation -> monthly using ERA5 baseline fractions ----
    rows_acc = []
    for _, r in accum_annual.iterrows():
        y = int(r["year"])
        a_yr = float(r["accum_m_iceeq_yr"])
        for m in range(1, 13):
            rows_acc.append({
                "time": safe_timestamp(y, m, 15),
                "year": y,
                "month": m,
                "accum_m_iceeq_month_layers": a_yr * frac_by_month[m],
            })
    df_acc_m = pd.DataFrame(rows_acc).sort_values("time")

    # ---- Merge pre-ERA monthly ----
    pre = df_acc_m.merge(df_lmr_m, on=["time", "year", "month"], how="inner")
    pre = pre.merge(df_core_m, on=["time", "year", "month"], how="inner")

    if pre.empty:
        raise ValueError("Pre-ERA monthly merge is empty; check year coverage.")

    pre["source"] = "preERA_layers+LMR+core"
    return pre


# =========================
# Merge with ERA5 post-1940 and save/plot
# =========================

def build_merged(pre: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    post = era5[era5["year"] >= SPLICE_YEAR].copy()
    post["t2m_K_lmr"] = np.nan
    post["t2m_K_core"] = np.nan
    post["accum_m_iceeq_month_layers"] = np.nan
    post["sublimation_m_iceeq_month"] = -post["snow_evap_m_iceeq_month"]
    post["source"] = "ERA5_monthly"

    post = post[[
        "time", "year", "month",
        "t2m_K",
        "t2m_K_lmr", "t2m_K_core",
        "net_accum_m_iceeq_month", "snowfall_m_iceeq_month", "sublimation_m_iceeq_month",
        "accum_m_iceeq_month_layers",
        "source"
    ]].copy()

    pre2 = pre.copy()
    pre2["t2m_K"] = np.nan  # no ERA5 temp pre-1940 (use t2m_K_lmr or t2m_K_core)
    pre2["net_accum_m_iceeq_month"] = pre2["accum_m_iceeq_month_layers"]
    pre2["snowfall_m_iceeq_month"] = np.nan
    pre2["sublimation_m_iceeq_month"] = np.nan

    pre2 = pre2[[
        "time", "year", "month",
        "t2m_K",
        "t2m_K_lmr", "t2m_K_core",
        "net_accum_m_iceeq_month", "snowfall_m_iceeq_month", "sublimation_m_iceeq_month",
        "accum_m_iceeq_month_layers",
        "source"
    ]].copy()

    merged = pd.concat([pre2, post], ignore_index=True).sort_values("time").reset_index(drop=True)
    return merged


def save_outputs(merged: pd.DataFrame) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(CSV_MERGED, index=False)
    print(f"[csv] wrote: {CSV_MERGED}")

    ds = xr.Dataset(
        data_vars=dict(
            t2m_K_era5=("time", merged["t2m_K"].to_numpy(dtype=float)),
            t2m_K_lmr=("time", merged["t2m_K_lmr"].to_numpy(dtype=float)),
            t2m_K_core=("time", merged["t2m_K_core"].to_numpy(dtype=float)),
            net_accum_m_iceeq_month=("time", merged["net_accum_m_iceeq_month"].to_numpy(dtype=float)),
            snowfall_m_iceeq_month=("time", merged["snowfall_m_iceeq_month"].to_numpy(dtype=float)),
            sublimation_m_iceeq_month=("time", merged["sublimation_m_iceeq_month"].to_numpy(dtype=float)),
            accum_layers_m_iceeq_month=("time", merged["accum_m_iceeq_month_layers"].to_numpy(dtype=float)),
        ),
        coords=dict(time=pd.to_datetime(merged["time"]).to_numpy()),
        attrs=dict(
            site=SITE_NAME,
            lat=str(LAT),
            lon=str(LON),
            splice_year=str(SPLICE_YEAR),
            baseline=f"{BASELINE_START}-{BASELINE_END}",
            notes=(
                "Pre-1940: accumulation from wais2008accum distributed using ERA5 baseline fractions; "
                "temperature provided as two options: (a) LMR seasonal anomaly + ERA5 baseline climatology; "
                "(b) core d18O annual anomaly calibrated to ERA5 annual mean, then applied pre-1940 and downscaled with ERA5 climatology. "
                "Post-1940: ERA5 monthly t2m and net accumulation (snowfall + snow_evap_rate)."
            ),
        )
    )
    ds.to_netcdf(NC_MERGED)
    print(f"[nc] wrote: {NC_MERGED}")


def make_plot(merged: pd.DataFrame) -> None:
    pre = merged[merged["source"] == "preERA_layers+LMR+core"]
    post = merged[merged["source"] == "ERA5_monthly"]

    fig, (axT, axA) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)

    # Temperature panel
    axT.plot(pre["time"], pre["t2m_K_lmr"], label="Pre-1940: LMR seasonal + ERA5 climatology (K)")
    axT.plot(pre["time"], pre["t2m_K_core"], label="Pre-1940: core d18O-derived T + ERA5 seasonal cycle (K)")
    axT.plot(post["time"], post["t2m_K"], label="ERA5 monthly t2m (K)")
    axT.axvline(pd.Timestamp(SPLICE_YEAR, 1, 1), linestyle=":", linewidth=1)
    axT.set_ylabel("Temperature (K)")
    axT.set_title(f"{SITE_NAME} — Temperature and accumulation forcing (pre-1940 + ERA5)")
    axT.legend(loc="best")

    # Accumulation panel
    axA.plot(pre["time"], pre["accum_m_iceeq_month_layers"], label="Pre-1940: annual accum → monthly (m ice eq / month)")
    axA.plot(post["time"], post["net_accum_m_iceeq_month"], label="ERA5 net (snowfall − sublimation)")
    axA.plot(post["time"], post["snowfall_m_iceeq_month"], linestyle="--", alpha=0.8, label="ERA5 snowfall")
    axA.plot(post["time"], post["sublimation_m_iceeq_month"], linestyle=":", alpha=0.8, label="ERA5 sublimation (magnitude)")
    axA.axvline(pd.Timestamp(SPLICE_YEAR, 1, 1), linestyle=":", linewidth=1)
    axA.set_ylabel("Accumulation (m ice eq / month)")
    axA.set_xlabel("Time")
    axA.legend(loc="best")

    fig.savefig(PLOT_FILE, dpi=200)
    plt.close(fig)
    print(f"[plot] wrote: {PLOT_FILE}")


def main():
    print(f"[site] {SITE_NAME} | lat={LAT:.3f}, lon={LON:.3f}")

    lmr = load_lmr_seasonal_point()
    d18 = load_wdc05a_d18o_annual()
    acc = load_wais_accum_annual()
    era5 = load_era5_monthly_point()

    pre = build_preera_monthly(lmr, d18, acc, era5)
    merged = build_merged(pre, era5)

    save_outputs(merged)
    make_plot(merged)

    print("\nDone.")
    print(f"  Merged CSV : {CSV_MERGED}")
    print(f"  Merged NC  : {NC_MERGED}")
    print(f"  Plot PNG   : {PLOT_FILE}")


if __name__ == "__main__":
    main()
