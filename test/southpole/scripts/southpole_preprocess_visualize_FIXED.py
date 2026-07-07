#!/usr/bin/env python3
"""
southpole/southpole_preprocess_visualize_FIXED.py

Fixes the path bug you hit where the script looked for:
  /home/.../test/processed/era5_monthly_point.csv
instead of:
  /home/.../test/southpole/processed/era5_monthly_point.csv

This happens when the script assumes it is inside southpole/scripts and sets:
  root = scripts_dir.parent
But if the script lives in southpole/ (as in your tree), scripts_dir == southpole/,
so root becomes the parent directory (test/).

This version auto-detects the project root:
  - If the script directory contains data/, processed/, figures/ -> use it
  - Else if the parent contains them -> use parent

It then produces:
  figures/south_pole_climate_forcing.png
  figures/usp50_strain_rate.png
  figures/south_pole_temperatures.png
  figures/south_pole_grain_radius.png  (if MAT parsing succeeds)

It also prints CSV headers for every CSV it loads and MAT keys for the grain-size MAT.

Run (from southpole/):
  python southpole_preprocess_visualize_FIXED.py
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
# Path helpers
# ----------------------------

def repo_dirs() -> Tuple[Path, Path, Path, Path]:
    here = Path(__file__).resolve().parent
    if (here / "data").is_dir() and (here / "processed").is_dir() and (here / "figures").is_dir():
        root = here
    elif (here.parent / "data").is_dir() and (here.parent / "processed").is_dir() and (here.parent / "figures").is_dir():
        root = here.parent
    else:
        raise FileNotFoundError(
            f"Could not find southpole root from {here}. Expected data/, processed/, figures/."
        )

    data = root / "data"
    proc = root / "processed"
    figs = root / "figures"
    for d in (data, proc, figs):
        d.mkdir(parents=True, exist_ok=True)
    return root, data, proc, figs


def print_csv_header(path: Path, df: pd.DataFrame) -> None:
    print(f"\n[csv] {path}")
    print(f"  columns ({len(df.columns)}): {list(df.columns)}")
    try:
        print("  head:")
        print(df.head(3).to_string(index=False))
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
# Climate forcing plot
# ----------------------------

def infer_temp_units_and_convert_to_C(series: pd.Series) -> Tuple[pd.Series, str]:
    x = pd.to_numeric(series, errors="coerce")
    med = float(np.nanmedian(x))
    if med > 150.0:
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

    era = safe_read_csv(era_path)
    era["time"] = pd.to_datetime(era["time"])
    era_T_C = pd.to_numeric(era["t2m_K"], errors="coerce") - 273.15

    if "net_accum_m_iceeq_month" in era.columns:
        era_acc_yr = pd.to_numeric(era["net_accum_m_iceeq_month"], errors="coerce") * 12.0
        era_acc_label = "m ice eq / yr (ERA5 annualized)"
    elif "net_accum_m_we_month" in era.columns:
        era_acc_yr = pd.to_numeric(era["net_accum_m_we_month"], errors="coerce") * 12.0
        era_acc_label = "m w.e. / yr (ERA5 annualized)"
    else:
        raise KeyError("ERA5 CSV missing expected net accumulation column.")

    buT = safe_read_csv(buT_path)
    buA = safe_read_csv(buA_path)

    if "year_CE" not in buT.columns or "temp" not in buT.columns:
        raise KeyError("Buizert temp CSV missing year_CE or temp columns.")
    if "year_CE" not in buA.columns or "accum" not in buA.columns:
        raise KeyError("Buizert accum CSV missing year_CE or accum columns.")

    buT_sub = buT[buT["year_CE"] >= 1500].copy()
    buA_sub = buA[buA["year_CE"] >= 1500].copy()

    buT_time = yearce_to_py_datetimes(buT_sub["year_CE"].to_numpy())
    buA_time = yearce_to_py_datetimes(buA_sub["year_CE"].to_numpy())

    buT_C, buT_units = infer_temp_units_and_convert_to_C(buT_sub["temp"])
    buA_val = pd.to_numeric(buA_sub["accum"], errors="coerce")

    fig, (axT, axA) = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True)

    axT.plot(buT_time, buT_C, label=f"Ice-core (Buizert2021) temperature {buT_units}", linewidth=1.5)
    axT.plot(era["time"], era_T_C, label="ERA5 2m temperature (°C)", linewidth=1.0, alpha=0.9)
    axT.set_ylabel("Temperature (°C)")
    axT.legend(loc="best")
    axT.set_title("South Pole forcing: low-res ice-core spinup + high-res ERA5")

    axA.plot(buA_time, buA_val, label="Ice-core accumulation (Buizert2021; units as provided)", linewidth=1.5)
    axA.plot(era["time"], era_acc_yr, label=era_acc_label, linewidth=1.0, alpha=0.9)
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
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    if not numeric_cols:
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
        L = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(L) & np.isfinite(t_years)
        if mask.sum() < 10:
            continue
        tt = t[mask]
        yy = t_years[mask]
        LL = L[mask]
        L0 = LL[0]
        axL.plot(tt, LL - L0, alpha=0.7, linewidth=0.9, label=c if nlegend < 8 else None)
        dLdt = np.gradient(LL, yy)
        eps = -(1.0 / max(L0, 1e-12)) * dLdt
        axE.plot(tt, eps, alpha=0.7, linewidth=0.9, label=c if nlegend < 8 else None)
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
# Temperature datasets (USP50)
# ----------------------------

def plot_temperatures(data_dir: Path, proc_dir: Path, figs_dir: Path) -> None:
    f_601525 = data_dir / "usap_dc" / "601525" / "USP50_firn_temperatures_170109-181224.csv"
    f_therm_table = data_dir / "usap_dc" / "601680" / "usapdc_601680" / "ThermistorTable_extrapolated.csv"

    era_path = proc_dir / "era5_monthly_point.csv"
    era = safe_read_csv(era_path) if era_path.exists() else pd.DataFrame()
    era_T_C = None
    if not era.empty and "t2m_K" in era.columns:
        era["time"] = pd.to_datetime(era["time"])
        era_T_C = pd.to_numeric(era["t2m_K"], errors="coerce") - 273.15

    fig, axs = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True)

    if f_601525.exists():
        df = safe_read_csv(f_601525)
        tcol = find_time_column(df)
        if tcol is None:
            raise KeyError(f"Could not identify time column in {f_601525}. Columns={list(df.columns)}")
        tt = pd.to_datetime(df[tcol], errors="coerce")

        plotted = 0
        for c in df.columns:
            if c == tcol:
                continue
            v = pd.to_numeric(df[c], errors="coerce")
            if np.isfinite(v).sum() < 10:
                continue
            axs[0].plot(tt, v, linewidth=0.8, alpha=0.8, label=c if plotted < 10 else None)
            plotted += 1
            if plotted >= 12:
                break
        axs[0].set_title("USP50 firn temperatures (601525) — subset of channels")
        axs[0].set_ylabel("Temperature (as provided)")
        if plotted > 0:
            axs[0].legend(ncol=2, fontsize=8)
    else:
        axs[0].text(0.02, 0.5, f"Missing: {f_601525.name}", transform=axs[0].transAxes)

    if f_therm_table.exists():
        df = safe_read_csv(f_therm_table)
        tcol = find_time_column(df)
        if tcol is None:
            axs[1].text(0.02, 0.5, "ThermistorTable_extrapolated.csv has no time column\n(see printed headers).",
                        transform=axs[1].transAxes)
        else:
            tt = pd.to_datetime(df[tcol], errors="coerce")
            plotted = 0
            for c in df.columns:
                if c == tcol:
                    continue
                v = pd.to_numeric(df[c], errors="coerce")
                if np.isfinite(v).sum() < 10:
                    continue
                axs[1].plot(tt, v, linewidth=0.8, alpha=0.8, label=c if plotted < 10 else None)
                plotted += 1
                if plotted >= 12:
                    break
            axs[1].set_title("USP50 thermistor table (601680) — subset of channels")
            axs[1].set_ylabel("Temperature (as provided)")
            if plotted > 0:
                axs[1].legend(ncol=2, fontsize=8)
    else:
        axs[1].text(0.02, 0.5, f"Missing: {f_therm_table.name}", transform=axs[1].transAxes)

    if era_T_C is not None:
        axs[0].plot(era["time"], era_T_C, color="k", linewidth=1.2, alpha=0.6, label="ERA5 t2m (°C)")
        axs[1].plot(era["time"], era_T_C, color="k", linewidth=1.2, alpha=0.6, label="ERA5 t2m (°C)")

    axs[1].set_xlabel("Time")
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
    mat_path = data_dir / "usap_dc" / "601551" / "insituData" / "mrsldata_insitu_gradius.mat"
    if not mat_path.exists():
        print(f"\n[grain] missing {mat_path}")
        return

    print(f"\n[mat] loading {mat_path}")
    mat = loadmat(mat_path)
    print_mat_keys(mat)

    k_lat = find_key(mat, ["lat", "latitude"])
    k_z = find_key(mat, ["depth", "z"])
    k_gr = find_key(mat, ["gradius", "grain", "radius"])

    if not (k_lat and k_z and k_gr):
        raise RuntimeError(
            "Could not locate expected arrays in gradius MAT file.\n"
            f"Found keys: {[k for k in mat.keys() if not k.startswith('__')]}\n"
            "Expected something like lat/depth/gradius.\n"
            "Use the printed MAT keys above to decide which to plot."
        )

    lat = np.asarray(mat[k_lat]).squeeze().reshape(-1)
    z = np.asarray(mat[k_z]).squeeze().reshape(-1)
    gr = np.asarray(mat[k_gr]).squeeze().reshape(-1)

    m = np.isfinite(lat) & np.isfinite(z) & np.isfinite(gr) & (lat < -88.0)
    if m.sum() == 0:
        print("[grain] No rows match lat<-88. Plotting all finite values instead.")
        m = np.isfinite(z) & np.isfinite(gr)

    fig = plt.figure(figsize=(5.0, 6.2))
    plt.scatter(gr[m], z[m], s=10, alpha=0.6)
    plt.gca().invert_yaxis()
    plt.xlabel(f"{k_gr} (units as provided)")
    plt.ylabel("Depth (m) (as provided)")
    plt.title("Grain radius compilation (subset near South Pole)")
    plt.tight_layout()
    out = figs_dir / "south_pole_grain_radius.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n[figure] wrote: {out}")


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
