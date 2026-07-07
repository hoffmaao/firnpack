#!/usr/bin/env python3
"""
southpole_preprocess_visualize_FIXED_v2.py

Fixes two issues you hit:

1) ERA5 not visible in the climate plot
   Your `processed/era5_monthly_point.csv` has interleaved rows where *every other*
   record is NaN for a given variable (e.g., t2m on 00:00, snowfall on 06:00).
   Matplotlib draws line segments only between consecutive finite points; if each
   finite value is separated by a NaN, you get no visible line.

   -> Solution: coalesce to a single monthly row (Period('M')) and take the first
      non-NaN (or mean) per variable, then plot.

2) USP50_firn_temperatures_170109-181224.csv is "wide" (timestamps are in the
   TIMESTAMP row), so there is no time column.
   -> Solution: parse the TIMESTAMP row into a DatetimeIndex, then plot a subset of
      thermistors (chosen across depth).

Directory layout expected:
southpole/
├── data
├── figures
├── processed
└── scripts

Run from southpole/scripts:
  python southpole_preprocess_visualize_FIXED_v2.py

Outputs (southpole/figures):
  - south_pole_climate_forcing.png
  - usp50_strain_rate.png
  - south_pole_temperatures.png
  - south_pole_grain_radius.png   (if MAT parsing succeeds)

Also prints column headers for each CSV loaded, and MAT keys.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ----------------------------
# Root/path discovery
# ----------------------------

def find_project_root(start: Path) -> Path:
    """Walk upward from `start` to find a directory containing data/, processed/, figures/."""
    start = start.resolve()
    for cand in [start] + list(start.parents):
        if (cand / "data").is_dir() and (cand / "processed").is_dir() and (cand / "figures").is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find project root above {start}. Expected folders: data/, processed/, figures/."
    )

def repo_dirs() -> Tuple[Path, Path, Path, Path]:
    here = Path(__file__).resolve().parent
    # Running from scripts/ or elsewhere: search upward
    root = find_project_root(here)
    data = root / "data"
    proc = root / "processed"
    figs = root / "figures"
    for d in (data, proc, figs):
        d.mkdir(parents=True, exist_ok=True)
    return root, data, proc, figs


# ----------------------------
# Common helpers
# ----------------------------

def print_csv_header(path: Path, df: pd.DataFrame, n: int = 3) -> None:
    print(f"\n[csv] {path}")
    print(f"  columns ({len(df.columns)}): {list(df.columns)[:20]}{' ...' if len(df.columns)>20 else ''}")
    try:
        print("  head:")
        print(df.head(n).to_string(index=False))
    except Exception:
        pass

def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    print_csv_header(path, df)
    return df

def yearce_to_py_datetimes(year_ce: np.ndarray, month: int = 7, day: int = 1) -> List[dt.datetime]:
    """Matplotlib-friendly datetimes, avoids pandas datetime64[ns] bounds."""
    out: List[dt.datetime] = []
    for y in np.asarray(year_ce, dtype=float):
        if not np.isfinite(y):
            continue
        yi = int(round(y))
        yi = max(1, min(9999, yi))
        out.append(dt.datetime(yi, month, day))
    return out


# ----------------------------
# ERA5 coalescing (fix NaN-separated lines)
# ----------------------------

def coalesce_era5_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the interleaved-NaN table into one row per month.

    We use month period keys, then for each column take the first non-NaN value.
    """
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out["ym"] = out["time"].dt.to_period("M")

    def first_valid(s: pd.Series):
        s2 = pd.to_numeric(s, errors="coerce")
        s2 = s2[np.isfinite(s2)]
        if len(s2) == 0:
            return np.nan
        return float(s2.iloc[0])

    # Aggregate numeric columns; keep year/month if present
    cols = [c for c in out.columns if c not in ("time", "ym")]
    agg = out.groupby("ym", sort=True)[cols].agg(first_valid).reset_index()

    # Use mid-month timestamps for plotting
    agg["time"] = agg["ym"].dt.to_timestamp(how="start") + pd.Timedelta(days=14)
    agg = agg.drop(columns=["ym"])
    # Keep a clean sort
    agg = agg.sort_values("time").reset_index(drop=True)
    return agg


# ----------------------------
# Climate forcing plot
# ----------------------------

def infer_temp_units_and_convert_to_C(series: pd.Series) -> Tuple[pd.Series, str]:
    x = pd.to_numeric(series, errors="coerce")
    med = float(np.nanmedian(x))
    if np.isfinite(med) and med > 150.0:
        return x - 273.15, "°C (converted from K)"
    return x, "°C (as provided)"

def climate_plot(proc_dir: Path, figs_dir: Path) -> None:
    era_path = proc_dir / "era5_monthly_point.csv"
    buT_path = proc_dir / "buizert2021_spice_temp.csv"
    buA_path = proc_dir / "buizert2021_spice_accum.csv"

    if not era_path.exists():
        raise FileNotFoundError(f"Missing {era_path}. Run the download script first.")
    if not buT_path.exists() or not buA_path.exists():
        raise FileNotFoundError("Missing Buizert processed CSVs. Run the download script first.")

    # --- ERA5: coalesce to monthly, then compute annual mean/total ---
    era_raw = safe_read_csv(era_path)
    era_m = coalesce_era5_monthly(era_raw)
    era_m["time"] = pd.to_datetime(era_m["time"])
    era_m["year"] = era_m["time"].dt.year

    # Temperature
    era_m["t2m_C"] = pd.to_numeric(era_m["t2m_K"], errors="coerce") - 273.15
    era_T_ann = era_m.groupby("year")["t2m_C"].mean()

    # Accumulation: annual total from monthly totals
    if "net_accum_m_iceeq_month" in era_m.columns:
        era_m["acc_m_month"] = pd.to_numeric(era_m["net_accum_m_iceeq_month"], errors="coerce")
        acc_units = "m ice eq"
    elif "net_accum_m_we_month" in era_m.columns:
        era_m["acc_m_month"] = pd.to_numeric(era_m["net_accum_m_we_month"], errors="coerce")
        acc_units = "m w.e."
    else:
        raise KeyError("ERA5 CSV missing expected net accumulation column.")

    era_A_ann = era_m.groupby("year")["acc_m_month"].sum()

    # Build datetime lists for annual series (matplotlib-friendly)
    era_years = era_T_ann.index.to_numpy(dtype=int)
    era_time_ann = [dt.datetime(int(y), 7, 1) for y in era_years]

    # --- Buizert: low-res spinup ---
    buT = safe_read_csv(buT_path)
    buA = safe_read_csv(buA_path)

    buT_sub = buT[buT["year_CE"] >= 1300].copy()
    buA_sub = buA[buA["year_CE"] >= 1300].copy()

    buT_time = yearce_to_py_datetimes(buT_sub["year_CE"].to_numpy())
    buA_time = yearce_to_py_datetimes(buA_sub["year_CE"].to_numpy())

    buT_C, buT_units = infer_temp_units_and_convert_to_C(buT_sub["temp"])
    buA_val = pd.to_numeric(buA_sub["accum"], errors="coerce")

    # --- overlap window mean (quick “do they match?” check) ---
    # overlap years based on available time
    ov0 = max(int(np.nanmin(era_years)), int(np.nanmin(buT_sub["year_CE"])))
    ov1 = min(int(np.nanmax(era_years)), int(np.nanmax(buT_sub["year_CE"])))
    ov_t0 = dt.datetime(ov0, 1, 1)
    ov_t1 = dt.datetime(ov1, 12, 31)

    era_T_mean = float(np.nanmean(era_T_ann.loc[ov0:ov1].to_numpy(dtype=float)))
    era_A_mean = float(np.nanmean(era_A_ann.loc[ov0:ov1].to_numpy(dtype=float)))

    buT_ov = buT_sub[(buT_sub["year_CE"] >= ov0) & (buT_sub["year_CE"] <= ov1)]
    buA_ov = buA_sub[(buA_sub["year_CE"] >= ov0) & (buA_sub["year_CE"] <= ov1)]

    bu_T_mean = float(np.nanmean(pd.to_numeric(buT_ov["temp"], errors="coerce")))
    if "converted from K" in buT_units:
        bu_T_mean -= 273.15
    bu_A_mean = float(np.nanmean(pd.to_numeric(buA_ov["accum"], errors="coerce")))

    # --- Plot ---
    fig, (axT, axA) = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True)

    # Temperature panel
    axT.plot(buT_time, buT_C, linewidth=2.0,
             label=f"Ice-core (Buizert2021) temperature {buT_units}")

    # FULL resolution (monthly) with transparency
    axT.plot(era_m["time"], era_m["t2m_C"], linewidth=0.7, alpha=0.18,
             label="ERA5 monthly 2m temperature (°C, α=0.18)")

    # Annual mean (bold)
    axT.plot(era_time_ann, era_T_ann.to_numpy(dtype=float), linewidth=2.2,
             label="ERA5 annual-mean 2m temperature (°C)")

    # Overlap means (horizontal segments)
    axT.plot([ov_t0, ov_t1], [bu_T_mean, bu_T_mean], linestyle="--", linewidth=2.2,
             label=f"Ice-core mean ({ov0}-{ov1}): {bu_T_mean:.2f} °C")
    axT.plot([ov_t0, ov_t1], [era_T_mean, era_T_mean], linestyle=":", linewidth=2.2,
             label=f"ERA5 mean ({ov0}-{ov1}): {era_T_mean:.2f} °C")

    axT.set_ylabel("Temperature (°C)")
    axT.legend(loc="best")
    axT.set_title("South Pole forcing: low-res ice-core spinup + high-res ERA5")

    # Accumulation panel
    axA.plot(buA_time, buA_val, linewidth=2.0,
             label="Ice-core accumulation (Buizert2021; units as provided)")

    # FULL resolution for accum: 12-mo running total (looks like “annual rate” but keeps month-to-month variability)
    era_roll_annual = era_m.set_index("time")["acc_m_month"].rolling(12, min_periods=6).sum()
    axA.plot(era_roll_annual.index, era_roll_annual.to_numpy(dtype=float),
             linewidth=0.7, alpha=0.18,
             label=f"ERA5 monthly (12-mo running total; {acc_units}/yr, α=0.18)")

    # Annual totals (bold)
    axA.plot(era_time_ann, era_A_ann.to_numpy(dtype=float), linewidth=2.2,
             label=f"ERA5 annual total ({acc_units}/yr)")

    # Overlap means (horizontal segments)
    axA.plot([ov_t0, ov_t1], [bu_A_mean, bu_A_mean], linestyle="--", linewidth=2.2,
             label=f"Ice-core mean ({ov0}-{ov1}): {bu_A_mean:.4f}")
    axA.plot([ov_t0, ov_t1], [era_A_mean, era_A_mean], linestyle=":", linewidth=2.2,
             label=f"ERA5 mean ({ov0}-{ov1}): {era_A_mean:.4f}")

    axA.set_ylabel("Accumulation")
    axA.set_xlabel("Time")
    axA.legend(loc="best")

    fig.tight_layout()
    out = figs_dir / "south_pole_climate_forcing.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n[figure] wrote: {out}")


# ----------------------------
# Strain / compaction (USP50)
# ----------------------------

def find_time_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = str(c).lower()
        if "time" in cl or "date" in cl or "datetime" in cl:
            return c
    return None

def plot_strain_rate(data_dir: Path, figs_dir: Path) -> None:
    bh = data_dir / "usap_dc" / "601680" / "usapdc_601680" / "USP50_borehole_lengths.csv"
    if not bh.exists():
        print(f"\n[strain] missing {bh}")
        return

    df = safe_read_csv(bh)
    tcol = find_time_column(df)
    if tcol is None:
        raise KeyError(f"Could not identify time column in {bh}. Columns={list(df.columns)}")

    t = pd.to_datetime(df[tcol], errors="coerce")
    t0 = t.iloc[0]
    t_years = (t - t0).dt.total_seconds().to_numpy() / (365.25 * 24 * 3600.0)

    numeric_cols: List[str] = []
    for c in df.columns:
        if c == tcol:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(v).sum() > 10:
            df[c] = v
            numeric_cols.append(c)

    if not numeric_cols:
        raise RuntimeError(f"No numeric length columns found in {bh}.")

    fig, (axL, axE) = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True)

    nlegend = 0
    for c in numeric_cols:
        L = df[c].to_numpy(dtype=float)
        mask = np.isfinite(L) & np.isfinite(t_years)
        if mask.sum() < 10:
            continue
        tt = t[mask]
        yy = t_years[mask]
        LL = L[mask]
        L0 = LL[0]
        axL.plot(tt, LL - L0, alpha=0.7, linewidth=0.9, label=c if nlegend < 10 else None)
        dLdt = np.gradient(LL, yy)
        eps = -(1.0 / max(L0, 1e-12)) * dLdt
        axE.plot(tt, eps, alpha=0.7, linewidth=0.9, label=c if nlegend < 10 else None)
        nlegend += 1

    axL.set_ylabel("Δ length (m)")
    axL.set_title("USP50 borehole shortening (compaction proxy)")
    if nlegend > 0:
        axL.legend(ncol=2, fontsize=8)

    axE.set_ylabel("Strain rate (1/yr)\n(-(1/L0)dL/dt)")
    axE.set_xlabel("Time")
    if nlegend > 0:
        axE.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    out = figs_dir / "usp50_strain_rate.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n[figure] wrote: {out}")


# ----------------------------
# Temperature datasets (USP50) — handle wide format
# ----------------------------

def parse_wide_timestamp_csv(path: Path) -> Tuple[pd.DatetimeIndex, pd.DataFrame, pd.Series]:
    """
    Parse USP50_firn_temperatures_170109-181224.csv.

    File structure:
      - Header row: unnamed + many numeric columns + 'depth'
      - Row with first col == 'TIMESTAMP' holds timestamps across columns
      - Rows with first col like 'Therm_temp*' hold temperature values
      - 'depth' column holds thermistor depths
    """
    df = safe_read_csv(path)
    idcol = df.columns[0]

    # Find TIMESTAMP row
    rn = df[idcol].astype(str)
    is_ts = rn.str.upper() == "TIMESTAMP"
    if is_ts.sum() != 1:
        raise RuntimeError(f"Expected exactly one TIMESTAMP row in {path}, found {is_ts.sum()}")

    ts_row = df.loc[is_ts].iloc[0]
    time_strings = ts_row.iloc[1:-1].astype(str)
    times = pd.to_datetime(time_strings, errors="coerce")

    valid = times.notna()
    time_cols = list(df.columns[1:-1][valid])

    # Sensor rows
    is_sensor = rn.str.contains("therm_temp", case=False, na=False)
    sensors = df.loc[is_sensor].copy()
    if len(sensors) == 0:
        raise RuntimeError(f"No Therm_temp rows found in {path}")

    depths = pd.to_numeric(sensors["depth"], errors="coerce")

    # Values matrix: (nsensors x ntimes) -> transpose to (ntimes x nsensors)
    vals = sensors[time_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    temp = pd.DataFrame(vals.T, index=pd.DatetimeIndex(times[valid]), columns=sensors[idcol].astype(str).to_list())

    return pd.DatetimeIndex(times[valid]), temp, depths.set_axis(sensors[idcol].astype(str).to_list())


def plot_temperatures(data_dir: Path, proc_dir: Path, figs_dir: Path) -> None:
    f_601525 = data_dir / "usap_dc" / "601525" / "USP50_firn_temperatures_170109-181224.csv"

    era_path = proc_dir / "era5_monthly_point.csv"
    era_raw = safe_read_csv(era_path) if era_path.exists() else pd.DataFrame()
    era_T_C = None
    era_t = None
    if not era_raw.empty and "t2m_K" in era_raw.columns:
        era = coalesce_era5_monthly(era_raw)
        era_t = era["time"]
        era_T_C = pd.to_numeric(era["t2m_K"], errors="coerce") - 273.15

    fig, axs = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=False)

    # Panel 1: surface temp context (ERA5)
    if era_T_C is not None:
        axs[0].plot(era_t, era_T_C, linewidth=1.2, alpha=0.85, label="ERA5 t2m (°C)")
        axs[0].set_ylabel("Temperature (°C)")
        axs[0].set_title("Surface temperature (ERA5) for context")
        axs[0].legend(loc="best")
    else:
        axs[0].text(0.02, 0.5, "ERA5 not available", transform=axs[0].transAxes)

    # Panel 2: borehole thermistors (USP50 601525)
    if f_601525.exists():
        tindex, temp_df, depth_by_sensor = parse_wide_timestamp_csv(f_601525)

        # Choose a subset of sensors across depth (avoid plotting dozens)
        pairs = [(name, float(depth_by_sensor.get(name, np.nan))) for name in temp_df.columns]
        pairs = [(n, d) for n, d in pairs if np.isfinite(d)]
        pairs.sort(key=lambda x: x[1])

        if len(pairs) == 0:
            # Fall back: first 10 channels
            sel = list(temp_df.columns[:10])
            labels = sel
        else:
            k = min(10, len(pairs))
            idx = np.linspace(0, len(pairs) - 1, k).round().astype(int)
            sel = [pairs[i][0] for i in idx]
            labels = [f"{pairs[i][0]} ({pairs[i][1]:.2f} m)" for i in idx]

        # Downsample to daily means to keep plots light
        td = temp_df[sel].copy()
        td.index = pd.to_datetime(td.index)
        td = td.resample("1D").mean()

        for c, lab in zip(sel, labels):
            axs[1].plot(td.index, td[c], linewidth=0.9, alpha=0.85, label=lab)

        axs[1].set_title("USP50 borehole thermistor temperatures (subset across depth; daily means)")
        axs[1].set_ylabel("Temperature (°C; as provided)")
        axs[1].set_xlabel("Time")
        axs[1].legend(ncol=2, fontsize=8, loc="best")

    else:
        axs[1].text(0.02, 0.5, f"Missing: {f_601525}", transform=axs[1].transAxes)

    fig.tight_layout()
    out = figs_dir / "south_pole_temperatures.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n[figure] wrote: {out}")


# ----------------------------
# Grain radius (MAT compilation, 601551)
# ----------------------------

def print_mat_keys(mat: Dict) -> None:
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f"  MAT keys ({len(keys)}): {keys}")
    for k in keys[:25]:
        v = mat[k]
        shp = getattr(v, "shape", None)
        if shp is not None:
            print(f"    - {k}: shape={shp} dtype={getattr(v,'dtype',None)}")

def find_key(mat: Dict, patterns: List[str]) -> Optional[str]:
    keys = [k for k in mat.keys() if not k.startswith("__")]
    kl = {k: k.lower() for k in keys}
    for pat in patterns:
        pat = pat.lower()
        for k, low in kl.items():
            if pat == low:
                return k
    for pat in patterns:
        pat = pat.lower()
        for k, low in kl.items():
            if pat in low:
                return k
    return None

def load_grain_radius_and_plot(data_dir: Path, figs_dir: Path) -> None:
    """
    Grain radius (MAT compilation, 601551).

    NOTE: Your current mrsldata_insitu_gradius.mat does not include South Pole,
    only Dome C, Vostok, and WAIS. So we (a) print keys/shapes, (b) write a
    placeholder figure and return without crashing.
    """
    mat_path = data_dir / "usap_dc" / "601551" / "insituData" / "mrsldata_insitu_gradius.mat"
    if not mat_path.exists():
        print(f"\n[grain] missing {mat_path}")
        return

    print(f"\n[mat] loading {mat_path}")
    mat = loadmat(mat_path)
    print_mat_keys(mat)

    keys = [k for k in mat.keys() if not k.startswith("__")]
    out = figs_dir / "south_pole_grain_radius.png"

    # Look for any SP-ish key naming (none in your file)
    sp_markers = ("south", "pole", "sp", "spice")
    sp_keys = [k for k in keys if any(m in k.lower() for m in sp_markers)]

    if not sp_keys:
        msg = (
            "No South Pole grain-radius series found in mrsldata_insitu_gradius.mat.\n\n"
            "Available series keys:\n  - " + "\n  - ".join(keys) + "\n\n"
            "This compilation includes Dome C, Vostok, and WAIS only.\n"
            "If we want South Pole grain size, we need a different dataset\n"
            "(e.g., SPICEcore microstructure/grain-size product).\n"
        )
        print("\n[grain] WARNING:", msg.replace("\n", " "))
        fig = plt.figure(figsize=(8, 3.6))
        plt.axis("off")
        plt.text(0.01, 0.98, msg, va="top", ha="left", family="monospace")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        print(f"[figure] wrote placeholder: {out}")
        return

    # If you ever get a South Pole key in the future, you can implement plotting here.


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    root, data, proc, figs = repo_dirs()
    print(f"[paths] root={root}")
    print(f"        data={data}\n        processed={proc}\n        figures={figs}")

    climate_plot(proc, figs)
    plot_strain_rate(data, figs)
    plot_temperatures(data, proc, figs)
    load_grain_radius_and_plot(data, figs)

    print("\nDone.")

if __name__ == "__main__":
    main()
