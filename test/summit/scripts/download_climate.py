#!/usr/bin/env python3
"""
Summit Station climate forcing builder + summary plot

Pre-ERA (before SPLICE_YEAR=1940):
  - Temperature: LMR Seasonal 'tas_mean.nc' at Summit (seasonal), rebased to 1950–1980 and
                downscaled to monthly by repeating each seasonal anomaly across 3 months.
  - Accumulation: Summit annual layer accumulation (Osman et al. 2021 NOAA template),
                  distributed to months using ERA5 1950–1980 monthly net-accum fractions.

ERA period (>= 1940):
  - Temperature: ERA5 monthly mean 2m temperature
  - Accumulation: ERA5 net = snowfall - sublimation
                  using mean_snowfall_rate + mean_snow_evaporation_rate
                  (IFS convention: downward fluxes positive; negative means evaporation/sublimation)

Outputs:
  - downloads/ raw files
  - extracted point series CSVs
  - merged monthly forcing CSV + NetCDF
  - summary plot PNG (2 rows: temperature, accumulation)

Data sources:
  - LMR Seasonal dataset: tas_mean.nc (gridded 2m air temperature) https://atmos.uw.edu/~zilumeng/LMR_Seasonal/
  - Summit annual accumulation (NOAA template): https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/osman2021/summit2021accum.txt
  - ERA5 monthly means (1940–present): https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means

Requirements:
  pip install numpy pandas xarray netcdf4 matplotlib requests cdsapi
  ERA5 download also requires ~/.cdsapirc (Copernicus CDS API key).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# =========================
# Config
# =========================

SITE_NAME = "Summit Station, Greenland"

# Summit approx location: 72°36’N, 38°25’W
LAT = 72.6
LON = -38.4167  # degrees East (negative = West)

SPLICE_YEAR = 1940

# Baseline used to rebase LMR anomalies and to build ERA5 climatologies/fractions
BASELINE_START = 1950
BASELINE_END = 1980

# LMR Seasonal download (gridded ensemble mean tas)
LMR_TAS_URL = "https://atmos.uw.edu/~zilumeng/LMR_Seasonal/data/mean/tas_mean.nc"

# Summit annual layer accumulation (NOAA template file)
NOAA_ACCUM_URL = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/osman2021/summit2021accum.txt"

# ERA5 dataset name for cdsapi
ERA5_DATASET = "reanalysis-era5-single-levels-monthly-means"

# Request variables (CDS names)
ERA5_VARS = [
    "2m_temperature",
    "mean_snowfall_rate",
    "mean_snow_evaporation_rate",
]

# Output folder
OUTDIR = Path("../processed/")
DL_DIR = Path("../sources/")
OUTDIR.mkdir(parents=True, exist_ok=True)
DL_DIR.mkdir(parents=True, exist_ok=True)

LMR_FILE = DL_DIR / "LMR_Seasonal_tas_mean.nc"
NOAA_ACCUM_FILE = DL_DIR / "summit2021accum.txt"
ERA5_FILE = DL_DIR / "ERA5_monthly_summit_1940_present.nc"

CSV_LMR_POINT = OUTDIR / "summit_lmr_seasonal_point.csv"
CSV_NOAA_ACCUM = OUTDIR / "summit_layer_accum_annual.csv"
CSV_ERA5_POINT = OUTDIR / "summit_era5_monthly_point.csv"
CSV_MERGED = OUTDIR / "summit_monthly_forcing_preLMR+layer_postERA5.csv"
NC_MERGED = OUTDIR / "summit_monthly_forcing_preLMR+layer_postERA5.nc"
PLOT_FILE = OUTDIR / "summit_temperature_and_accumulation_summary.png"

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
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=2**20):
            if chunk:
                f.write(chunk)
    tmp.replace(path)
    print(f"[download] wrote: {path.name}")

def lon_to_dataset(ds: xr.Dataset, lon: float) -> float:
    """Convert lon to match ds lon convention (0..360 or -180..180)."""
    lon_vals = ds["lon"].values
    lon_min = float(np.nanmin(lon_vals))
    lon_max = float(np.nanmax(lon_vals))
    if lon_min >= 0.0 and lon_max > 180.0:
        return lon % 360.0
    return lon

def month_quarter(month: int) -> list[int]:
    """Map LMR 'season timestamp month' to the 3 months of that quarter."""
    # We assume LMR times are quarter starts: 1,4,7,10
    m = int(month)
    return [m, m + 1, m + 2]

def safe_timestamp(year: int, month: int, day: int = 15) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=int(month), day=int(day))

def parse_noaa_template_table(path: Path) -> pd.DataFrame:
    """NOAA template: comment lines begin with '#'; data lines do not."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in txt.splitlines() if ln.strip() and (not ln.lstrip().startswith("#"))]
    if len(lines) < 2:
        raise ValueError(f"No data lines found in {path}")
    buf = io.StringIO("\n".join(lines))
    df = pd.read_csv(buf, sep=r"\s+", engine="python")
    return df


# =========================
# LMR: seasonal temperature at Summit
# =========================

def load_lmr_seasonal_point() -> pd.DataFrame:
    download_file(LMR_TAS_URL, LMR_FILE)

    ds = xr.open_dataset(LMR_FILE, decode_times=True)
    if "tas" not in ds:
        raise KeyError(f"LMR file missing 'tas'. Data vars: {list(ds.data_vars)}")

    lon_use = lon_to_dataset(ds, LON)

    # nearest point
    pt = ds[["tas"]].sel(lat=LAT, lon=lon_use, method="nearest")

    # LMR time is often cftime objects; avoid pandas conversion for very early years.
    tvals = pt["time"].values
    years = np.array([int(tv.year) for tv in tvals], dtype=int)
    months = np.array([int(tv.month) for tv in tvals], dtype=int)

    # Build baseline mean over 1950–1980 (across all seasonal samples in those years)
    base_mask = (years >= BASELINE_START) & (years <= BASELINE_END)
    if not np.any(base_mask):
        raise ValueError("LMR baseline years not found in tas_mean.nc; cannot rebase anomalies.")
    tas_base = float(pt["tas"].isel(time=np.where(base_mask)[0]).mean().values)

    tas = pt["tas"].values.astype(float)
    tas_anom = tas - tas_base

    df = pd.DataFrame({
        "year": years,
        "season_month": months,
        "tas_anom_K": tas_anom,
    })

    # For pre-ERA monthly downscaling we only need years < SPLICE_YEAR
    df = df[df["year"] < SPLICE_YEAR].copy().reset_index(drop=True)
    df["source"] = "LMR_seasonal_point"

    df.to_csv(CSV_LMR_POINT, index=False)
    print(f"[csv] wrote: {CSV_LMR_POINT}")
    return df


# =========================
# NOAA: annual layer accumulation at Summit
# =========================

def load_noaa_layer_accum_annual() -> pd.DataFrame:
    download_file(NOAA_ACCUM_URL, NOAA_ACCUM_FILE)

    df = parse_noaa_template_table(NOAA_ACCUM_FILE)

    # Find a year column and an accumulation column robustly
    year_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("age_ce", "year_ce", "year"):
            year_col = c
            break
    if year_col is None:
        # fall back: first column
        year_col = df.columns[0]

    accum_col = None
    for c in df.columns:
        if "accum" in c.lower():
            accum_col = c
            break
    if accum_col is None:
        raise KeyError(f"Could not find accumulation column in NOAA file. Columns: {list(df.columns)}")

    out = df[[year_col, accum_col]].rename(columns={year_col: "year", accum_col: "accum_kgm2yr"}).copy()
    out["year"] = out["year"].astype(int)
    out["accum_kgm2yr"] = out["accum_kgm2yr"].astype(float)

    # pre-ERA only
    out = out[out["year"] < SPLICE_YEAR].copy()

    # convert to m ice eq / yr
    out["accum_m_iceeq_yr"] = (out["accum_kgm2yr"] / 1000.0) * (RHO_W / RHO_I)

    out.to_csv(CSV_NOAA_ACCUM, index=False)
    print(f"[csv] wrote: {CSV_NOAA_ACCUM}")
    return out


# =========================
# ERA5: monthly temperature + snowfall + snow evaporation
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
            "Also configure ~/.cdsapirc with your CDS API key."
        ) from e

    end_year = pd.Timestamp.today().year
    years = [str(y) for y in range(SPLICE_YEAR, end_year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]

    # small bbox around Summit (CDS expects [N,W,S,E])
    d = 0.25
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
    c.retrieve(ERA5_DATASET, req, str(ERA5_FILE))
    print(f"[era5] wrote: {ERA5_FILE.name}")

def load_era5_monthly_point() -> pd.DataFrame:
    download_era5_monthly_if_needed()

    ds = xr.open_dataset(ERA5_FILE, decode_times=True)

    # Try common variable names in output netcdf
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

    # Net accumulation rate: snowfall - sublimation.
    # IFS sign convention: downward flux positive; evaporation/sublimation typically negative.
    # So net = snowfall + snow_evap_rate (because snow_evap_rate < 0 for evaporation).
    df["net_accum_rate_kgm2s"] = df["snowfall_rate_kgm2s"] + df["snow_evap_rate_kgm2s"]

    # Convert rates to monthly totals (m ice eq / month)
    seconds_in_month = df["time"].dt.days_in_month.to_numpy(dtype=float) * 24.0 * 3600.0

    # 1 kg/m^2 = 1 mm w.e.
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
# Build pre-ERA monthly series (LMR temp + layer accum)
# =========================

def build_preera_monthly(
    lmr_seasonal: pd.DataFrame,
    layer_annual: pd.DataFrame,
    era5_monthly: pd.DataFrame,
) -> pd.DataFrame:

    # ERA5 baseline monthly climatology for temperature (1950–1980)
    base = era5_monthly[
        (era5_monthly["year"] >= BASELINE_START) & (era5_monthly["year"] <= BASELINE_END)
    ].copy()
    if base.empty:
        raise ValueError("ERA5 baseline (1950–1980) not present in ERA5 monthly data.")

    t_clim = base.groupby("month")["t2m_K"].mean().to_dict()  # month -> K

    # ERA5 baseline monthly net-accum climatology for distributing annual layer accumulation
    a_clim = base.groupby("month")["net_accum_m_iceeq_month"].mean()
    a_pos = np.clip(a_clim.to_numpy(dtype=float), 0.0, None)
    if np.sum(a_pos) <= 0:
        frac = np.ones(12) / 12.0
    else:
        frac = a_pos / np.sum(a_pos)
    frac_by_month = {m: float(frac[m-1]) for m in range(1, 13)}

    # Expand LMR seasonal anomalies to monthly anomalies
    rows_T = []
    for _, r in lmr_seasonal.iterrows():
        y = int(r["year"])
        sm = int(r["season_month"])
        anom = float(r["tas_anom_K"])
        for m in month_quarter(sm):
            if m > 12:
                # should not happen with sm in {1,4,7,10}
                continue
            rows_T.append({
                "time": safe_timestamp(y, m, 15),
                "year": y,
                "month": m,
                "tas_anom_K": anom,
            })
    lmr_monthly = pd.DataFrame(rows_T).drop_duplicates(subset=["time"]).sort_values("time")

    # Expand annual layer accumulation to monthly
    rows_A = []
    for _, r in layer_annual.iterrows():
        y = int(r["year"])
        a_yr = float(r["accum_m_iceeq_yr"])
        for m in range(1, 13):
            rows_A.append({
                "time": safe_timestamp(y, m, 15),
                "year": y,
                "month": m,
                "layer_accum_m_iceeq_month": a_yr * frac_by_month[m],
            })
    layer_monthly = pd.DataFrame(rows_A).sort_values("time")

    # Merge on time (intersection of availability)
    pre = pd.merge(lmr_monthly, layer_monthly, on=["time", "year", "month"], how="inner")
    if pre.empty:
        raise ValueError(
            "Pre-ERA merge is empty. This usually means the layer record years and LMR years "
            "do not overlap. (Note: summit2021accum starts at ~1743 CE.)"
        )

    # Absolute temperature by adding ERA5 baseline monthly climatology
    pre["t2m_K"] = pre["tas_anom_K"] + pre["month"].map(t_clim)

    # For plotting consistency with ERA5 naming:
    pre["net_accum_m_iceeq_month"] = pre["layer_accum_m_iceeq_month"]
    pre["snowfall_m_iceeq_month"] = np.nan
    pre["sublimation_m_iceeq_month"] = np.nan  # unknown pre-ERA
    pre["source"] = "preERA_LMRtemp+layeraccum"
    return pre[[
        "time","year","month","t2m_K","tas_anom_K",
        "net_accum_m_iceeq_month","snowfall_m_iceeq_month","sublimation_m_iceeq_month",
        "source"
    ]].sort_values("time").reset_index(drop=True)


# =========================
# Splice with ERA5 (post-1940)
# =========================

def build_merged_monthly(pre: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    post = era5[era5["year"] >= SPLICE_YEAR].copy()
    post["sublimation_m_iceeq_month"] = -post["snow_evap_m_iceeq_month"]  # magnitude
    post = post.rename(columns={
        "net_accum_m_iceeq_month": "net_accum_m_iceeq_month",
        "snowfall_m_iceeq_month": "snowfall_m_iceeq_month",
    })
    post["tas_anom_K"] = np.nan
    post["source"] = "ERA5_monthly"

    post = post[[
        "time","year","month","t2m_K","tas_anom_K",
        "net_accum_m_iceeq_month","snowfall_m_iceeq_month","sublimation_m_iceeq_month",
        "source"
    ]].copy()

    merged = pd.concat([pre, post], ignore_index=True).sort_values("time").reset_index(drop=True)
    return merged


# =========================
# Plot + Save
# =========================

def save_outputs(merged: pd.DataFrame) -> None:
    merged.to_csv(CSV_MERGED, index=False)
    print(f"[csv] wrote: {CSV_MERGED}")

    ds = xr.Dataset(
        data_vars=dict(
            t2m_K=("time", merged["t2m_K"].to_numpy(dtype=float)),
            net_accum_m_iceeq_month=("time", merged["net_accum_m_iceeq_month"].to_numpy(dtype=float)),
            snowfall_m_iceeq_month=("time", merged["snowfall_m_iceeq_month"].to_numpy(dtype=float)),
            sublimation_m_iceeq_month=("time", merged["sublimation_m_iceeq_month"].to_numpy(dtype=float)),
            tas_anom_K=("time", merged["tas_anom_K"].to_numpy(dtype=float)),
        ),
        coords=dict(time=pd.to_datetime(merged["time"]).to_numpy()),
        attrs=dict(
            site=SITE_NAME,
            lat=str(LAT),
            lon=str(LON),
            splice_year=str(SPLICE_YEAR),
            baseline=f"{BASELINE_START}-{BASELINE_END}",
            notes=(
                "Pre-ERA: temperature = ERA5 monthly climatology + LMR seasonal anomaly (rebased to baseline); "
                "accumulation = annual layer accumulation distributed by ERA5 baseline monthly net fractions. "
                "Post-ERA: ERA5 monthly t2m + snowfall + snow evaporation (net = snowfall + snow_evap_rate)."
            ),
        )
    )
    ds.to_netcdf(NC_MERGED)
    print(f"[nc] wrote: {NC_MERGED}")

def make_plot(merged: pd.DataFrame) -> None:
    pre = merged[merged["source"] == "preERA_LMRtemp+layeraccum"]
    post = merged[merged["source"] == "ERA5_monthly"]

    fig, (axT, axA) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)

    # Temperature panel
    axT.plot(pre["time"], pre["t2m_K"], label="Pre-ERA: ERA5 climatology + LMR seasonal anomaly")
    axT.plot(post["time"], post["t2m_K"], label="ERA5 monthly t2m (1940–present)")
    axT.axvline(pd.Timestamp(SPLICE_YEAR, 1, 1), linestyle=":", linewidth=1)
    axT.set_ylabel("Temperature (K)")
    axT.set_title(f"{SITE_NAME} — Temperature and accumulation forcing (pre-ERA + ERA5)")
    axT.legend(loc="best")

    # Accumulation panel (m ice eq / month)
    axA.plot(pre["time"], pre["net_accum_m_iceeq_month"], label="Pre-ERA layer accumulation (monthly-distributed)")
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


# =========================
# Main
# =========================

def main():
    print(f"[site] {SITE_NAME} | lat={LAT:.4f}, lon={LON:.4f}")

    lmr = load_lmr_seasonal_point()
    layer = load_noaa_layer_accum_annual()
    era5 = load_era5_monthly_point()

    pre = build_preera_monthly(lmr, layer, era5)
    merged = build_merged_monthly(pre, era5)

    save_outputs(merged)
    make_plot(merged)

    print("\nDone.")
    print(f"  Merged CSV : {CSV_MERGED}")
    print(f"  Merged NC  : {NC_MERGED}")
    print(f"  Plot PNG   : {PLOT_FILE}")

if __name__ == "__main__":
    main()
