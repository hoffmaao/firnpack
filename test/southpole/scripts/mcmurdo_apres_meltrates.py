"""McMurdo Ice Shelf ApRES strain + basal melt-rate processing.

Run (no arguments):
    python scripts/mcmurdo_apres_meltrates.py

Directory assumptions (your tree):

    <repo>/data/GA04/... .DAT/.dat
    <repo>/figures/
    <repo>/scripts/ (this file)

Outputs:

    <repo>/results/GA04/preprocessed/*.npz
    <repo>/results/GA04/pair_results.csv
    <repo>/figures/GA04/preprocess/*.png
    <repo>/figures/GA04/pairs/*.png
    <repo>/figures/GA04/timeseries.png

Notes
-----
- The code follows the MATLAB workflow you provided (func_preprocess +
  func_strain_rates + ct_fmcw_melt logic) but is implemented in pure Python.
- If your GA04 folder contains only a single profile (as in the provided zip),
  the script will preprocess it and stop (no strain/melt can be computed).
"""

from __future__ import annotations

import csv
import sys
from statistics import median
from pathlib import Path

import numpy as np

# Ensure we can import the local ./scripts/apres package when executed as a script
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from apres.config import ProcessingConfig
from apres.io import find_dat_files
from apres.preprocess import preprocess_file
from apres.plotting import (
    plot_pair_diagnostics,
    plot_preprocess_profile,
    plot_timeseries,
    plot_melt_method_diagnostics,
    basal_metrics_from_profile,
)
from apres.serialization import (
    load_preprocessed_npz,
    save_pair_diagnostics_npz,
    save_pair_results_csv,
    save_preprocessed_npz,
)
from apres.strain import strain_melt_between_profiles


# ------------------------------
# User-editable settings
# ------------------------------
STATION = "GA04"
RECOMPUTE_PREPROCESS = False  # set True if you changed config and want fresh preprocessing


def main() -> None:
    repo_root = HERE.parent

    data_dir = repo_root / "data" / STATION
    figures_dir = repo_root / "figures" / STATION
    results_dir = repo_root / "results" / STATION

    cfg = ProcessingConfig(station=STATION, data_dir=data_dir, figures_dir=figures_dir, results_dir=results_dir)
    cfg.validate()

    # Find raw files
    raw_files = find_dat_files(data_dir)
    if len(raw_files) == 0:
        raise SystemExit(f"No *.DAT/*.dat files found under {data_dir}")

    print(f"\n[{STATION}] Found {len(raw_files)} raw files under {data_dir}")

    # Preprocess (with caching)
    pre_dir = results_dir / "preprocessed"
    fig_pre_dir = figures_dir / "preprocess"

    profiles = []
    for i, f in enumerate(raw_files, start=1):
        cache = pre_dir / f"{f.stem}.npz"

        print(f"\n[{STATION}] ({i}/{len(raw_files)}) Preprocess: {f}")

        # Caching logic: load if possible, but recompute if
        #  - user requested RECOMPUTE_PREPROCESS
        #  - raw file is newer than cache
        #  - new diagnostics (e.g. sub-bands) are enabled but missing in cache
        p = None
        if cache.exists() and not RECOMPUTE_PREPROCESS:
            try:
                p_try = load_preprocessed_npz(cache)
                need_subbands = bool(getattr(cfg, "subbands_enabled", False))
                has_subbands = getattr(p_try, "spec_cor_subbands", None) is not None

                raw_newer = False
                try:
                    raw_newer = f.stat().st_mtime > cache.stat().st_mtime
                except Exception:
                    raw_newer = False

                if raw_newer or (need_subbands and not has_subbands):
                    p = None
                else:
                    p = p_try
            except Exception:
                p = None

        if p is None:
            p = preprocess_file(f, cfg, figures_dir=fig_pre_dir)
            save_preprocessed_npz(p, cache)

        # Save a quick diagnostic plot each time (cheap)
        out_png = fig_pre_dir / f"{p.name}_preprocess.png"
        plot_preprocess_profile(p, out_png, cfg)

        profiles.append(p)

    # Sort by timestamp
    profiles = sorted(profiles, key=lambda p: p.timestamp)

    # ------------------------------------------------------
    # Per-acquisition basal diagnostics (for accretion/QC)
    # ------------------------------------------------------
    # These diagnostics are *not* used in the solver; they are written to
    # CSV and plotted to help determine whether basal-return complexity
    # (multiple peaks, trailing energy) might be driving spurious melt-rate
    # behaviour.
    acq_metrics_csv = results_dir / "acquisition_metrics.csv"
    with acq_metrics_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "station",
                "timestamp",
                "profile",
                "gps_on",
                "latitude_deg",
                "longitude_deg",
                "bed_depth_m",
                "bed_peak_amp_db",
                "basal_window_mean_amp_db",
                "peak_ratio_db",
                "peak_separation_m",
                "peak1_depth_m",
                "peak2_depth_m",
                "peak1_amp_db",
                "peak2_amp_db",
            ],
        )
        w.writeheader()
        for p in profiles:
            m = basal_metrics_from_profile(p, cfg)
            w.writerow(
                {
                    "station": STATION,
                    "timestamp": p.timestamp.isoformat(),
                    "profile": p.name,
                    "gps_on": p.gps_on,
                    "latitude_deg": p.latitude_deg,
                    "longitude_deg": p.longitude_deg,
                    "bed_depth_m": m["bed_depth_m"],
                    "bed_peak_amp_db": m["bed_peak_amp_db"],
                    "basal_window_mean_amp_db": m["basal_window_mean_amp_db"],
                    "peak_ratio_db": m["peak_ratio_db"],
                    "peak_separation_m": m["peak_separation_m"],
                    "peak1_depth_m": m["peak1_depth_m"],
                    "peak2_depth_m": m["peak2_depth_m"],
                    "peak1_amp_db": m["peak1_amp_db"],
                    "peak2_amp_db": m["peak2_amp_db"],
                }
            )

    if len(profiles) < 2:
        print(
            f"\n[{STATION}] Only {len(profiles)} profile(s) found. "
            "Need at least 2 to compute strain and basal melt rates."
        )
        print(f"Preprocess figures written to: {fig_pre_dir}")
        print(f"Preprocessed caches written to: {pre_dir}")
        return

    # Pairwise strain + melt between consecutive profiles
    fig_pair_dir = figures_dir / "pairs"
    pair_results = []

    for k in range(len(profiles) - 1):
        p1 = profiles[k]
        p2 = profiles[k + 1]
        print(
            f"\n[{STATION}] Pair {k+1}/{len(profiles)-1}: {p1.name} ({p1.timestamp}) -> {p2.name} ({p2.timestamp})"
        )

        try:
            result, coarse, fine, fit, bed = strain_melt_between_profiles(p1, p2, cfg)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        pair_results.append(result)

        # Save full per-pair diagnostics arrays (optional but handy)
        pair_npz = results_dir / "pairs" / f"{p1.name}__{p2.name}.npz"
        save_pair_diagnostics_npz(result, coarse, fine, fit, bed, pair_npz)

        # Diagnostics figure
        out_png = fig_pair_dir / f"{p1.name}__{p2.name}_diagnostics.png"
        plot_pair_diagnostics(p1, p2, result, coarse, fine, fit, bed, out_png, cfg)

        print(
            f"  dt_days={result.dt_days:.2f}  vsr={result.vsr_per_year:.3e}/yr  "
            f"melt_rate={result.melt_rate_m_per_year:.3f} m/yr"
        )

    # ---------------------------------
    # Post-processing: median filter the time series (optional)
    # ---------------------------------
    def _rolling_median_nan(x: np.ndarray, window: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.size
        if window <= 1 or n == 0:
            return x.copy()
        half = window // 2
        out = np.full(n, np.nan, dtype=float)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            out[i] = float(np.nanmedian(x[lo:hi]))
        return out

    if pair_results and cfg.timeseries_median_filter_enabled and cfg.timeseries_median_filter_window > 1:
        w = int(cfg.timeseries_median_filter_window)
        melt_raw = np.array([r.melt_rate_m_per_year for r in pair_results], dtype=float)
        vsr_raw = np.array([r.vsr_per_year for r in pair_results], dtype=float)
        melt_f = _rolling_median_nan(melt_raw, w)
        vsr_f = _rolling_median_nan(vsr_raw, w)
        for r, mf, vf in zip(pair_results, melt_f.tolist(), vsr_f.tolist()):
            r.melt_rate_m_per_year_medfilt = None if (mf is None or not np.isfinite(mf)) else float(mf)
            r.vsr_per_year_medfilt = None if (vf is None or not np.isfinite(vf)) else float(vf)

    # Write CSV
    csv_path = results_dir / "pair_results.csv"
    save_pair_results_csv(pair_results, csv_path)

    # Write a simple site summary (median melt rate + best-effort lat/lon)
    summary_path = results_dir / "site_summary.csv"
    lat_vals = [p.latitude_deg for p in profiles if getattr(p, "latitude_deg", None) is not None]
    lon_vals = [p.longitude_deg for p in profiles if getattr(p, "longitude_deg", None) is not None]
    gps_vals = [p.gps_on for p in profiles if getattr(p, "gps_on", None) is not None]

    site_lat = median(lat_vals) if (lat_vals and lon_vals) else None
    site_lon = median(lon_vals) if (lat_vals and lon_vals) else None
    gps_on = max(gps_vals) if gps_vals else None

    melt_rates_raw = [r.melt_rate_m_per_year for r in pair_results if np.isfinite(r.melt_rate_m_per_year)]
    melt_rates_med = [
        float(r.melt_rate_m_per_year_medfilt)
        for r in pair_results
        if r.melt_rate_m_per_year_medfilt is not None and np.isfinite(float(r.melt_rate_m_per_year_medfilt))
    ]

    melt_median_raw = median(melt_rates_raw) if melt_rates_raw else None
    melt_mean_raw = (sum(melt_rates_raw) / len(melt_rates_raw)) if melt_rates_raw else None

    melt_median_medfilt = median(melt_rates_med) if melt_rates_med else None
    melt_mean_medfilt = (sum(melt_rates_med) / len(melt_rates_med)) if melt_rates_med else None

    # Convenience: prefer filtered values if enabled/available
    melt_median = melt_median_medfilt if melt_median_medfilt is not None else melt_median_raw
    melt_mean = melt_mean_medfilt if melt_mean_medfilt is not None else melt_mean_raw

    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "station",
                "latitude_deg",
                "longitude_deg",
                "gps_on",
                "melt_rate_m_per_year_median",
                "melt_rate_m_per_year_mean",
                "melt_rate_m_per_year_median_raw",
                "melt_rate_m_per_year_mean_raw",
                "melt_rate_m_per_year_median_medfilt",
                "melt_rate_m_per_year_mean_medfilt",
                "timeseries_median_filter_enabled",
                "timeseries_median_filter_window",
                "n_pairs",
                "pair_results_csv",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "station": STATION,
                "latitude_deg": site_lat,
                "longitude_deg": site_lon,
                "gps_on": gps_on,
                "melt_rate_m_per_year_median": melt_median,
                "melt_rate_m_per_year_mean": melt_mean,
                "melt_rate_m_per_year_median_raw": melt_median_raw,
                "melt_rate_m_per_year_mean_raw": melt_mean_raw,
                "melt_rate_m_per_year_median_medfilt": melt_median_medfilt,
                "melt_rate_m_per_year_mean_medfilt": melt_mean_medfilt,
                "timeseries_median_filter_enabled": int(bool(cfg.timeseries_median_filter_enabled)),
                "timeseries_median_filter_window": int(cfg.timeseries_median_filter_window),
                "n_pairs": len(pair_results),
                "pair_results_csv": str(csv_path),
            }
        )

    # Time series plot
    ts_png = figures_dir / "timeseries.png"
    plot_timeseries(pair_results, ts_png, cfg, profiles=profiles)

    # Melt-method diagnostics (phase vs amplitude, sub-band dependence)
    md_png = figures_dir / "melt_method_diagnostics.png"
    plot_melt_method_diagnostics(pair_results, md_png, cfg)

    print(f"\n[{STATION}] Done.")
    print(f"Pair results CSV: {csv_path}")
    print(f"Site summary CSV: {summary_path}")
    print(f"Acquisition metrics CSV: {acq_metrics_csv}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
