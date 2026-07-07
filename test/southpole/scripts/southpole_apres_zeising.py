"""southpole_apres_zeising.py

Process South Pole ApRES raw .DAT files through the full Zeising pipeline
to produce displacement profiles with proper phase-error-derived uncertainties.

Processing chain (per site):
    1. preprocess_file() — raw .DAT bursts → phase-sensitive range profiles
       (bad chirp culling, windowing, range-velocity transform, error estimation)
    2. align_coarse() — amplitude cross-correlation for bulk alignment
    3. align_fine() — complex cross-correlation per depth window →
       dh_m (displacement) and dhe_m (phase standard error × λ/(4π))
    4. fit_ice() — Menke WLS strain fit using dhe_m as weights,
       residual-based covariance for uncertainty

Input:
    Raw .DAT files from Hills et al. (2022), USAP-DC 601503
    Site-to-directory mapping from Acquisition_MetaData.txt

Output:
    test/southpole/processed/apres_zeising_processed.csv
    test/southpole/figures/apres_zeising_profiles.png

Run:
    cd test/southpole/scripts
    python southpole_apres_zeising.py
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the Zeising pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent))
from apres.config import ProcessingConfig
from apres.preprocess import preprocess_file
from apres.strain import (
    align_coarse,
    align_fine,
    fit_ice,
    StrainFit,
)

# ============================================================================
# Paths
# ============================================================================
_HERE = Path(__file__).parent
_SOUTHPOLE = _HERE.parent
RAW_DIR = _SOUTHPOLE / "data/usap_dc/601503/raw"
METADATA = RAW_DIR / "Acquisition_MetaData.txt"
OUT_CSV = _SOUTHPOLE / "processed" / "apres_zeising_processed.csv"
FIG_DIR = _SOUTHPOLE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DAYS_PER_YEAR = 365.25

# ============================================================================
# Read site metadata
# ============================================================================

def load_site_metadata() -> list[dict]:
    """Parse Acquisition_MetaData.txt for VV-polarization site pairs."""
    sites = []
    with open(METADATA) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["polarization"].strip() != "VV":
                continue
            d1 = row.get("2018-19_directory", "").strip()
            d2 = row.get("2019-20_directory", "").strip()
            if not d1 or not d2:
                continue
            sites.append({
                "name": row["Site"].strip(),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "dir1": d1,
                "dir2": d2,
            })
    return sites


def find_dat_file(directory: Path) -> Optional[Path]:
    """Find the first .DAT file in a directory (there's typically one per burst)."""
    dats = sorted(directory.glob("*.DAT")) + sorted(directory.glob("*.dat"))
    return dats[0] if dats else None


# ============================================================================
# Processing
# ============================================================================

def process_site(
    site: dict,
    cfg: ProcessingConfig,
) -> Optional[dict]:
    """Process one site through the full Zeising pipeline from raw .DAT files."""
    name = site["name"]
    dir1 = RAW_DIR / site["dir1"]
    dir2 = RAW_DIR / site["dir2"]

    dat1 = find_dat_file(dir1)
    dat2 = find_dat_file(dir2)
    if dat1 is None or dat2 is None:
        print(f"  {name}: SKIP — missing .DAT file(s)")
        return None

    print(f"  {name}: preprocessing epoch 1 ({dat1.name})...")
    try:
        p1 = preprocess_file(dat1, cfg)
    except Exception as e:
        print(f"    FAILED preprocessing ep1: {e}")
        return None

    print(f"    preprocessing epoch 2 ({dat2.name})...")
    try:
        p2 = preprocess_file(dat2, cfg)
    except Exception as e:
        print(f"    FAILED preprocessing ep2: {e}")
        return None

    dt_days = (p2.timestamp - p1.timestamp).total_seconds() / 86400.0
    dt_years = dt_days / DAYS_PER_YEAR
    print(f"    ep1: {p1.timestamp.strftime('%Y-%m-%d %H:%M')}  "
          f"ep2: {p2.timestamp.strftime('%Y-%m-%d %H:%M')}  "
          f"dt={dt_days:.1f} d")

    # --- Zeising pipeline ---
    print(f"    Coarse alignment...")
    try:
        coarse = align_coarse(p1, p2, cfg)
    except Exception as e:
        print(f"    FAILED coarse alignment: {e}")
        return None

    print(f"    Fine alignment...")
    try:
        fine = align_fine(p1, p2, cfg, coarse)
    except Exception as e:
        print(f"    FAILED fine alignment: {e}")
        return None

    print(f"    Menke WLS strain fit...")
    try:
        fit = fit_ice(p1, p2, cfg, fine, coarse, dt_days)
    except Exception as e:
        print(f"    FAILED strain fit: {e}")
        return None

    # Extract displacement profile
    range_m = fine.range_m
    dh_m = fine.dh_m
    dhe_m = fine.dhe_m
    coherence = np.abs(fine.coherence)

    # Convert displacement to rate (m/yr)
    dRdt_myr = dh_m * DAYS_PER_YEAR / dt_days
    dRdt_err_myr = dhe_m * DAYS_PER_YEAR / dt_days

    # Menke residual (using fit mask)
    used = fine.range_m >= cfg.firn_depth_m
    if used.any() and np.isfinite(dh_m[used]).sum() > 2:
        resid = dh_m[used] - (fit.m0 + fit.m1 * (range_m[used] - fit.mid_range_m))
        resid_std = float(np.std(resid[np.isfinite(resid)])) * 100
    else:
        resid_std = float("nan")

    print(f"    vsr = {fit.vsr_per_year:.4e} +/- {fit.vsr_err_per_year:.4e} 1/yr  "
          f"R2 = {fit.r2:.4f}  N_windows = {len(range_m)}  "
          f"resid_std = {resid_std:.2f} cm")

    return {
        "name": name,
        "dt_days": dt_days,
        "range_m": range_m,
        "dh_m": dh_m,
        "dhe_m": dhe_m,
        "dRdt_myr": dRdt_myr,
        "dRdt_err_myr": dRdt_err_myr,
        "coherence": coherence,
        "fit": fit,
        "p1_timestamp": p1.timestamp,
        "p2_timestamp": p2.timestamp,
    }


# ============================================================================
# Output
# ============================================================================

def save_csv(results: list[dict], out_path: Path):
    """Save processed results to CSV."""
    rows = []
    for r in results:
        n = len(r["range_m"])
        fit = r["fit"]
        for i in range(n):
            rows.append({
                "site": r["name"],
                "range_m": r["range_m"][i],
                "dh_m": r["dh_m"][i],
                "dhe_m": r["dhe_m"][i],
                "dRdt_myr": r["dRdt_myr"][i],
                "dRdt_err_myr": r["dRdt_err_myr"][i],
                "coherence": r["coherence"][i],
                "dt_days": r["dt_days"],
                "vsr_per_year": fit.vsr_per_year,
                "vsr_err_per_year": fit.vsr_err_per_year,
            })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df)} rows)")


def plot_profiles(results: list[dict], out_path: Path):
    """Plot dR/dt profiles with Zeising uncertainties for all sites."""
    n_sites = len(results)
    n_cols = min(n_sites, 5)
    n_rows = (n_sites + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows),
                              squeeze=False, sharey=True)
    cmap = plt.cm.tab10

    for i, r in enumerate(results):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        name = r["name"]
        rng = r["range_m"]
        dRdt = r["dRdt_myr"]
        dRdt_err = r["dRdt_err_myr"]
        coh = r["coherence"]
        fit = r["fit"]

        ok = np.isfinite(dRdt) & np.isfinite(rng) & (coh > 0.3) & (rng <= 200)

        ax.errorbar(dRdt[ok] * 100, rng[ok],
                    xerr=dRdt_err[ok] * 100,
                    fmt=".", ms=4, elinewidth=0.5, color=cmap(i % 10),
                    label="Zeising")

        # Show Menke fit line
        r_fit = np.linspace(float(rng[ok].min()), float(rng[ok].max()), 100)
        dh_fit = fit.m0 + fit.m1 * (r_fit - fit.mid_range_m)
        dRdt_fit = dh_fit * DAYS_PER_YEAR / r["dt_days"]
        ax.plot(dRdt_fit * 100, r_fit, "k--", lw=1, alpha=0.7, label="Menke fit")

        ax.set_title(f"{name}\nvsr={fit.vsr_per_year:.2e} 1/yr", fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="grey", lw=0.5)
        ax.grid(True, ls="--", alpha=0.3)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("Range (m)", fontsize=9)
        if row == n_rows - 1:
            ax.set_xlabel("dR/dt (cm/yr)", fontsize=8)
        if i == 0:
            ax.legend(fontsize=6)

    for i in range(n_sites, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        "South Pole ApRES: Zeising pipeline (raw .DAT → preprocess → align → Menke WLS)\n"
        "dR/dt with phase-error uncertainties (dhe from pse × λ/(4π))",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("South Pole ApRES: Zeising pipeline (raw .DAT files)")
    print("=" * 65)

    if not RAW_DIR.exists():
        print(f"Raw data not found at {RAW_DIR}")
        print("Extract raw.zip first: cd data/usap_dc/601503 && unzip raw.zip")
        raise SystemExit(1)

    # Configure for South Pole firn
    cfg = ProcessingConfig(
        station="SouthPole",
        max_range_m=1000.0,
        # Firn settings
        firn_depth_m=100.0,
        min_depth_m=10.0,
        # Coarse alignment: constrain for small firn displacements
        maxlag_bins=(-2, 2),
        coarse_chunk_width_m=20.0,
        coarse_step_m=4.0,
        # Fine alignment
        fine_chunk_width_m=6.0,
        fine_step_m=2.0,
        # Coherence thresholds
        min_cohere_coarse=0.5,
        min_cohere_fine=0.5,
        min_ampcor=0.5,
        min_ampcor_prom=0.02,
        # Fit method: Menke WLS
        fit_method="menke",
        do_melt_estimate=False,
        # Smart unwrap
        do_smart_unwrap=True,
        # Ice thickness (firn-only, set conservatively high)
        ice_thickness_method="use",
        ice_thickness_use_m=900.0,
        ice_thickness_min_m=50.0,
        bed_search_min_m=800.0,
        bed_search_max_m=1000.0,
        # Sub-bands off for firn
        subbands_enabled=False,
    )

    # Load site metadata
    sites = load_site_metadata()
    print(f"\nFound {len(sites)} VV-polarization site pairs in metadata")

    # Process each site
    results = []
    for site in sites:
        try:
            r = process_site(site, cfg)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  {site['name']}: FAILED — {e}")

    if not results:
        print("\nNo sites processed successfully!")
        raise SystemExit(1)

    # Save
    print(f"\n--- Saving results ---")
    save_csv(results, OUT_CSV)
    plot_profiles(results, FIG_DIR / "apres_zeising_profiles.png")

    # Summary
    print(f"\n--- Summary ---")
    print(f"{'site':>10s}  {'vsr (1/yr)':>12s}  {'vsr_err':>12s}  {'R2':>6s}  "
          f"{'N_fine':>6s}  {'dt (d)':>8s}")
    for r in results:
        f = r["fit"]
        n_ok = int(np.sum(np.isfinite(r["dh_m"]) & (r["coherence"] > 0.3)))
        print(f"{r['name']:>10s}  {f.vsr_per_year:12.4e}  {f.vsr_err_per_year:12.4e}  "
              f"{f.r2:6.4f}  {n_ok:6d}  {r['dt_days']:8.1f}")

    print("\nDone.")
