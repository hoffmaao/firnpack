#!/usr/bin/env python3
"""
WAIS Divide climate forcing builder + summary plot

Goal:
  - Pre-1940:
      * Temperature from:
          (1) LMR Seasonal (tas_mean.nc) sampled at WAIS Divide, rebased to 1950–1980,
              and downscaled to monthly by repeating each seasonal anomaly across 3 months.
          (2) Ice-core water isotope (annual d18O at WDC05A) -> annual temperature anomaly
              via linear regression to ERA5 annual mean T over overlap, then applied pre-1940.
              Downscaled to monthly by adding ERA5 1950–1980 monthly climatology.
      * Accumulation from annual-layer/annual accumulation record (wais2008accum.*),
        distributed to months using ERA5 1950–1980 monthly net-accum fractions.
  - Post-1940:
      * Temperature: ERA5 monthly mean 2m temperature
      * Accumulation: ERA5 net = snowfall - sublimation
                      computed as snowfall_rate + snow_evaporation_rate
                      (ECMWF convention: downward flux positive; negative = sublimation/evaporation)

Outputs:
  - downloaded raw files
  - extracted point series CSVs
  - merged monthly forcing CSV + NetCDF (contains BOTH pre-ERA temperature options)
  - summary plot PNG (2 rows: temperature; accumulation)

Requirements:
  pip install numpy pandas xarray netcdf4 matplotlib requests cdsapi
  ERA5 download requires ~/.cdsapirc (Copernicus CDS API key)

References:
  - LMR Seasonal: https://atmos.uw.edu/~zilumeng/LMR_Seasonal/
  - WDC05A d18O (Steig 2013): https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/wdc05a2013d18o.txt
  - WAIS accumulation record (2008): https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/wais2008accum.txt
  - ERA5 monthly means: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# =========================
# Config
# =========================

SITE_NAME = "WAIS Divide (WDC05A/WDC06A area)"

# WAIS Divide site (consistent with WDC05A metadata)
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
    return [m, m+1, m+2]

def parse_text_table_remove_comments(raw_bytes: bytes) -> pd.DataFrame:
    """
    Robust-ish parser for NOAA-like template files:
      - decode with utf-8-sig, fall back to latin-1
      - drop lines starting with '#'
      - read whitespace-delimited table
    """
    try:
        txt = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        txt = raw_bytes.decode("latin-1", errors="replace")

    lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        raise ValueError("No data lines found after removing comment lines.")
    buf = io.StringIO("\n".join(lines))
    # try whitespace first
    df = pd.read_csv(buf, sep=r"\s+", engine="python")
    return df

def pick_column(df: pd.DataFrame, candidates_substrings: list[str], fallback_idx: int = 0) -> str:
    cols = list(df.columns)
    # exact or substring match (case-insensitive)
    for sub in candidates_substrings:
        for c in cols:
            if sub.lower() == str(c).lower():
                return c
    for sub in candidates_substrings:
        for c in cols:
            if sub.lower() in str(c).lower():
                return c
    return cols[fallback_idx]


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
# Ice-core d18O annual (WDC05A) -> annual temperature anomaly (calibrated to ERA5)
# =========================

def load_wdc05a_d18o_annual() -> pd.DataFrame:
    download_file(WDC05A_D18O_URL, D18O_FILE)
    raw = D18O_FILE.read_bytes()
    df = parse_text_table_remove_comments(raw)

    # Expect a year column (AD) and an isotope column
    year_col = pick_column(df, ["year", "yr", "time", "ad"], fallback_idx=0)
    d18_col = pick_column(df, ["d18", "d18o", "delta18o"], fallback_idx=1)

    out = df[[year_col, d18_col]].rename(columns={year_col: "year", d18_col: "d18O"}).copy()
    out["year"] = out["year"].astype(int)
    out["d18O"] = pd.to_numeric(out["d18O"], errors="coerce")
    out = out.dropna().sort_values("year").reset_index(drop=True)

    out.to_csv(CSV_D18O_ANNUAL, index=False)
    print(f"[csv] wrote: {CSV_D18O_ANNUAL}")
    return out


# =========================
# Annual accumulation record (pre-1940) (wais2008accum.*)
# =========================

def load_wais_accum_annual() -> pd.DataFrame:
    download_file(WAIS_ACCUM_URL, ACCUM_FILE)
    raw = ACCUM_FILE.read_bytes()

    df = parse_text_table_remove_comments(raw)

    year_col = pick_column(df, ["year", "yr", "ad", "age_ce"], fallback_idx=0)
    acc_col = pick_column(df, ["accum", "snow", "b", "precip"], fallback_idx=1)

    out = df[[year_col, acc_col]].rename(columns={year_col: "year", acc_col: "accum_raw"}).copy()
    out["year"] = out["year"].astype(int)
    out["accum_raw"] = pd.to_numeric(out["accum_raw"], errors="coerce")
    out = out.dropna().sort_values("year").reset_index(drop=True)

    # Infer units heuristically:
    med = float(np.nanmedian(out["accum_raw"].values))
    # If values look like kg m-2 yr-1 (or mm w.e./yr), convert: /1000 -> m w.e.
    if med > 10.0:
        accum_m_we_yr = out["accum_raw"] / 1000.0
        unit_note = "interpreted as kg m-2 yr-1 (or mm w.e./yr)"
    # If values look like meters per year, keep as m w.e.
    else:
        accum_m_we_yr = out["accum_raw"]
        unit_note = "interpreted as m w.e. yr-1"
    out["accum_m_iceeq_yr"] = accum_m_we_yr * (RHO_W / RHO_I)

    out = out[out["year"] < SPLICE_YEAR].copy()
    out["unit_note"] = unit_note

    out.to_csv(CSV_ACCUM_ANNUAL, index=False)
    print(f"[csv] wrote: {CSV_ACCUM_ANNUAL}  ({unit_note})")
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
# Build pre-ERA monthly (two temperature options + layer accumulation)
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

    # ERA5 baseline monthly net-accum climatology for distributing annual accumulation
    a_clim = base.groupby("month")["net_accum_m_iceeq_month"].mean()
    a_pos = np.clip(a_clim.to_numpy(dtype=float), 0.0, None)
    frac = (a_pos / np.sum(a_pos)) if np.sum(a_pos) > 0 else np.ones(12) / 12.0
    frac_by_month = {m: float(frac[m-1]) for m in range(1, 13)}

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
    # baseline annual mean (1950–1980)
    era_base_ann = era_ann[(era_ann["year"] >= BASELINE_START) & (era_ann["year"] <= BASELINE_END)]
    t0 = float(era_base_ann["t2m_K_ann"].mean())

    # d18O baseline (1950–1980)
    d18_base = d18o_annual[(d18o_annual["year"] >= BASELINE_START) & (d18o_annual["year"] <= BASELINE_END)]
    if d18_base.empty:
        raise ValueError("d18O baseline (1950–1980) not available in WDC05A series.")
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

    # annual->monthly by adding monthly climatology
    rows_core = []
    for _, r in d18_pre.iterrows():
        y = int(r["year"])
        t_ann = float(r["t2m_K_ann_core"])
        for m in range(1, 13):
            rows_core.append({
                "time": safe_timestamp(y, m, 15),
                "year": y,
                "month": m,
                "t2m_K_core": t_ann + (float(t_clim_month[m]) - np.mean(list(t_clim_month.values()))),
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
    pre = df_acc_m.merge(df_lmr_m, on=["time","year","month"], how="inner")
    pre = pre.merge(df_core_m, on=["time","year","month"], how="inner")

    if pre.empty:
        raise ValueError("Pre-ERA monthly merge is empty; check year coverage and formats.")

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
        "time","year","month",
        "t2m_K",
        "t2m_K_lmr","t2m_K_core",
        "net_accum_m_iceeq_month","snowfall_m_iceeq_month","sublimation_m_iceeq_month",
        "accum_m_iceeq_month_layers",
        "source"
    ]].copy()

    pre2 = pre.copy()
    pre2["t2m_K"] = np.nan  # no ERA5 temp pre-1940
    pre2["net_accum_m_iceeq_month"] = pre2["accum_m_iceeq_month_layers"]
    pre2["snowfall_m_iceeq_month"] = np.nan
    pre2["sublimation_m_iceeq_month"] = np.nan

    pre2 = pre2[[
        "time","year","month",
        "t2m_K",
        "t2m_K_lmr","t2m_K_core",
        "net_accum_m_iceeq_month","snowfall_m_iceeq_month","sublimation_m_iceeq_month",
        "accum_m_iceeq_month_layers",
        "source"
    ]].copy()

    merged = pd.concat([pre2, post], ignore_index=True).sort_values("time").reset_index(drop=True)
    return merged

def save_outputs(merged: pd.DataFrame) -> None:
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
                "Pre-1940: accumulation from annual-layer/annual accumulation record distributed using ERA5 baseline fractions; "
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
    axT.plot(pre["time"], pre["t2m_K_core"], label="Pre-1940: core d18O-derived T + ERA5 climatology (K)")
    axT.plot(post["time"], post["t2m_K"], label="ERA5 monthly t2m (K)")
    axT.axvline(pd.Timestamp(SPLICE_YEAR, 1, 1), linestyle=":", linewidth=1)
    axT.set_ylabel("Temperature (K)")
    axT.set_title(f"{SITE_NAME} — Temperature and accumulation forcing (pre-1940 + ERA5)")
    axT.legend(loc="best")

    # Accumulation panel
    axA.plot(pre["time"], pre["accum_m_iceeq_month_layers"], label="Pre-1940: annual layers → monthly (m ice eq / month)")
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
