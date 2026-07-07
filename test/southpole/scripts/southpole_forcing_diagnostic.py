"""southpole_forcing_diagnostic.py

Visualise the climate forcing (temperature and accumulation) used to drive
the South Pole firn model in southpole_inversion_v1.py.

Two forcing segments are applied in that model:
  1. Spinup (1200 yr, time-varying):
       T, A from Buizert 2021 SPICEcore reconstruction (40-yr resolution,
       linearly interpolated to annual), covering 740–1940 CE.
  2. ERA5 transient run (1940–2015):
       T = ERA5 annual-mean 2-m temperature + bias correction (-4.71 K)
       A = ERA5 annual-total net accumulation (m ice eq/yr)

Output figures:
  diag_forcing.png          — ERA5 period detail (1940–2015) with reference lines
  diag_forcing_timeline.png — Full 740–2015 CE timeline: Buizert spinup + ERA5

No Firedrake required.

Run:
    cd test/southpole/scripts
    python southpole_forcing_diagnostic.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Paths
# =============================================================================
_HERE         = Path(__file__).parent
_SOUTHPOLE    = _HERE.parent
PROCESSED     = _SOUTHPOLE / "processed"
ERA5_CSV          = PROCESSED / "era5_monthly_point.csv"
SP19_DENS_CSV     = PROCESSED / "sp19_density.csv"
SP19_AGE_CSV      = PROCESSED / "sp19_depth_age.csv"
BUIZERT_TEMP_CSV  = PROCESSED / "buizert2021_spice_temp.csv"
BUIZERT_ACCUM_CSV = PROCESSED / "buizert2021_spice_accum.csv"
OUT_DIR       = _SOUTHPOLE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Constants (must stay in sync with southpole_inversion_v1.py)
# =============================================================================
ERA5_START_YEAR    = 1940
ERA5_END_YEAR      = 2015
SPINUP_YEARS       = 1200
SPINUP_T_K         = 221.87   # K    — Buizert 2021 pre-industrial South Pole
ERA5_T_OFFSET_K    = -4.71    # K    — ERA5 warm-bias correction at South Pole
ERA5_A_OFFSET_M_YR = +0.0163  # m/yr — ERA5 accumulation bias correction
                               #   offset = Buizert_1940 - ERA5_1940 = 0.09795 - 0.08169
RHO_I           = 917.0      # kg/m³
RHO_SURF_IC     = 350.0      # kg/m³ — surface density used in density IC
H0              = 130.0      # m  — model column height
CORE_YEAR       = 2015       # SP19 core year

# =============================================================================
# Data loading
# =============================================================================

def load_era5_annual(csv_path: Path):
    """Return (years, T_K_raw, A_m_iceeq_yr) from the ERA5 monthly point CSV."""
    df = pd.read_csv(csv_path)
    t_rows = df[df["t2m_K"].notna()][["year", "t2m_K"]].copy()
    a_rows = df[df["net_accum_m_iceeq_month"].notna()][
        ["year", "net_accum_m_iceeq_month"]
    ].copy()
    T_by_yr = t_rows.groupby("year")["t2m_K"].mean()
    A_by_yr = a_rows.groupby("year")["net_accum_m_iceeq_month"].sum()
    years = np.array(sorted(T_by_yr.index.intersection(A_by_yr.index)), dtype=int)
    return (
        years,
        T_by_yr.loc[years].values.astype(float),
        A_by_yr.loc[years].values.astype(float),
    )


def load_sp19_density(depth_max: float = H0):
    df = pd.read_csv(SP19_DENS_CSV)
    df = df[df["depth_m"] <= depth_max].copy()
    dep = df["depth_m"].values.astype(float)
    rho = df["rho_kgm3"].values.astype(float) * 1000.0   # g/cm³ → kg/m³
    return dep[np.argsort(dep)], rho[np.argsort(dep)]


def load_sp19_age(core_year: int = CORE_YEAR, depth_max: float = H0):
    df = pd.read_csv(SP19_AGE_CSV)
    mask = (
        (df["year_CE"] >= 0)
        & (df["year_CE"] <= core_year)
        & (df["depth_m"] <= depth_max)
    )
    df = df[mask].copy()
    dep    = df["depth_m"].values.astype(float)
    age_yr = float(core_year) - df["year_CE"].values.astype(float)
    idx    = np.argsort(dep)
    return dep[idx], age_yr[idx]


# =============================================================================
# SP19-implied long-term mean accumulation (same computation as inversion script)
# =============================================================================

def compute_sp19_accumulation(sp19_dep, sp19_rho, sp19_age_dep, sp19_age_yr):
    """Return (A_sp19, mass_col, z_max, age_max).

    In steady state: A = mass_above(z) / (rho_i × age(z))
    Uses the deepest SP19 annual-layer observation as the anchor depth.
    """
    z_max   = float(sp19_age_dep[-1])
    age_max = float(sp19_age_yr[-1])
    z_grid  = np.linspace(0.0, z_max, 2000)
    rho_grid = np.interp(
        z_grid,
        np.concatenate([[0.0], sp19_dep]),
        np.concatenate([[RHO_SURF_IC], sp19_rho]),
    )
    mass_col = float(np.trapz(rho_grid, z_grid))
    A_sp19   = mass_col / (age_max * RHO_I)
    return A_sp19, mass_col, z_max, age_max


# =============================================================================
# Buizert spinup reconstruction (same logic as inversion script)
# =============================================================================

def load_buizert_spinup(spinup_start_year_ce: float,
                        era5_start_year: float = ERA5_START_YEAR):
    """Return (years_ce, T_K, A_m_iceeq_yr) at annual resolution for the spinup.

    Linearly interpolates the Buizert 2021 SPICEcore reconstruction (40-year
    resolution) to annual steps over [spinup_start_year_ce, era5_start_year).
    """
    df_T = pd.read_csv(BUIZERT_TEMP_CSV).sort_values("year_CE")
    df_A = pd.read_csv(BUIZERT_ACCUM_CSV).sort_values("year_CE")
    years = np.arange(float(spinup_start_year_ce), float(era5_start_year), 1.0)
    T = np.interp(years, df_T["year_CE"].values, df_T["temp"].values + 273.15)
    A = np.interp(years, df_A["year_CE"].values, df_A["accum"].values)
    return years, T, A


# =============================================================================
# Helper: centred running mean via pandas rolling
# =============================================================================

def running_mean(x: np.ndarray, years: np.ndarray, window: int = 11) -> np.ndarray:
    """Centred running mean; edges use min window//2 observations."""
    s = pd.Series(x, index=years)
    return s.rolling(window, center=True, min_periods=window // 2).mean().values


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    print("Loading ERA5 forcing ...")
    era5_years, era5_T_raw, era5_A = load_era5_annual(ERA5_CSV)

    # Slice to model run window [ERA5_START_YEAR, ERA5_END_YEAR)
    run_mask  = (era5_years >= ERA5_START_YEAR) & (era5_years < ERA5_END_YEAR)
    run_years = era5_years[run_mask]
    run_T_raw = era5_T_raw[run_mask]
    run_T_cor = run_T_raw + ERA5_T_OFFSET_K    # bias-corrected temperature
    run_A_raw = era5_A[run_mask]
    run_A     = run_A_raw + ERA5_A_OFFSET_M_YR  # bias-corrected accumulation

    # ERA5 1940–1960 raw mean (historical reference; not bias-corrected)
    mask_40_60  = (era5_years >= ERA5_START_YEAR) & (era5_years <= 1960)
    A_spin_era5 = float(np.mean(era5_A[mask_40_60]))

    print("Loading Buizert 2021 SPICEcore spinup reconstruction ...")
    _spinup_start_ce = ERA5_START_YEAR - SPINUP_YEARS
    spin_years, spin_T, spin_A = load_buizert_spinup(_spinup_start_ce, ERA5_START_YEAR)
    print(f"  Spinup: {spin_years[0]:.0f}–{spin_years[-1]:.0f} CE  "
          f"T={spin_T.min()-273.15:.2f}–{spin_T.max()-273.15:.2f} °C  "
          f"A mean={np.mean(spin_A):.4f} m/yr")

    print("Loading SP19 density and age (for SP19-implied A) ...")
    sp19_dep, sp19_rho     = load_sp19_density()
    sp19_age_dep, sp19_age_yr = load_sp19_age()
    A_spin_sp19, mass_col, z_max, age_max = compute_sp19_accumulation(
        sp19_dep, sp19_rho, sp19_age_dep, sp19_age_yr
    )

    # --- Summary statistics ---
    T_era5_mean     = float(np.mean(run_T_cor)) - 273.15
    A_era5_mean     = float(np.mean(run_A))
    A_trend_per_dec = float(np.polyfit(run_years, run_A, 1)[0]) * 10.0
    A_buizert_mean  = float(np.mean(spin_A))

    print(f"\nForcing summary")
    print(f"  ERA5 run period          : {run_years[0]}–{run_years[-1]}  ({len(run_years)} yr)")
    print(f"  Spinup period            : {len(spin_years)} yr  "
          f"({spin_years[0]:.0f}–{spin_years[-1]:.0f} CE, Buizert 2021)")
    print(f"  T Buizert spinup range   : {spin_T.min()-273.15:.2f}–{spin_T.max()-273.15:.2f} °C  "
          f"(mean {np.mean(spin_T)-273.15:.2f} °C)")
    print(f"  T ERA5  1940–2015 mean   : {T_era5_mean:.2f} °C  "
          f"(ERA5 raw mean = {np.mean(run_T_raw)-273.15:.2f} °C; "
          f"bias = {ERA5_T_OFFSET_K:+.2f} K)")
    print(f"  A Buizert spinup mean    : {A_buizert_mean:.4f} m ice eq/yr  "
          f"(range {spin_A.min():.4f}–{spin_A.max():.4f})")
    print(f"  A SP19-implied           : {A_spin_sp19:.4f} m ice eq/yr  "
          f"[mass_col={mass_col:.0f} kg/m², age_max={age_max:.0f} yr at {z_max:.1f} m]")
    print(f"  A ERA5  1940–60 raw mean : {A_spin_era5:.4f} m ice eq/yr")
    print(f"  A ERA5  1940–2015 raw    : {float(np.mean(run_A_raw)):.4f} m ice eq/yr  "
          f"→ corrected (+{ERA5_A_OFFSET_M_YR:.4f}): {A_era5_mean:.4f} m ice eq/yr")
    print(f"  A ERA5  linear trend     : {A_trend_per_dec:+.4f} m ice eq/yr per decade")
    print(f"  A junction (Buizert→ERA5): {spin_A[-1]:.4f} → {run_A[0]:.4f} m/yr (corrected)  "
          f"(Δ = {run_A[0]-spin_A[-1]:+.4f})")

    # Running means
    T_rm = running_mean(run_T_cor - 273.15, run_years, window=11)
    A_rm = running_mean(run_A, run_years, window=11)

    # Cumulative accumulation anomalies (ERA5 period vs various reference rates)
    cum_anom_bui  = np.cumsum(run_A - A_buizert_mean)  # vs Buizert spinup mean
    cum_anom_sp19 = np.cumsum(run_A - A_spin_sp19)     # vs SP19-implied mean
    cum_anom_era5 = np.cumsum(run_A - A_spin_era5)     # vs ERA5 1940-60 mean

    print(f"\n  Cumulative A anomaly at {run_years[-1]} vs Buizert spinup mean"
          f" : {cum_anom_bui[-1]:+.3f} m ice eq")
    print(f"  Cumulative A anomaly at {run_years[-1]} vs SP19-implied mean"
          f"  : {cum_anom_sp19[-1]:+.3f} m ice eq")
    print(f"  Cumulative A anomaly at {run_years[-1]} vs ERA5 1940-60 mean"
          f" : {cum_anom_era5[-1]:+.3f} m ice eq")

    # ==========================================================================
    # Figure
    # ==========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.subplots_adjust(hspace=0.40)

    xl = (run_years[0] - 1, run_years[-1] + 1)

    # -------------------------------------------------------------------------
    # Panel 1: Temperature
    # -------------------------------------------------------------------------
    ax = axes[0]
    ax.plot(run_years, run_T_raw - 273.15, color="0.80", lw=0.8, zorder=1,
            label="ERA5 raw (uncorrected)")
    ax.plot(run_years, run_T_cor - 273.15, color="0.50", lw=0.9, zorder=2,
            label=f"ERA5 bias-corrected ({ERA5_T_OFFSET_K:+.2f} K)")
    ax.plot(run_years, T_rm, color="steelblue", lw=2.2, zorder=3,
            label="11-yr centred mean (corrected)")
    ax.axhline(float(np.mean(spin_T)) - 273.15, color="darkorange", ls="--", lw=1.6, zorder=4,
               label=f"Buizert spinup mean = {np.mean(spin_T)-273.15:.2f} °C")
    ax.axhline(T_era5_mean, color="green", ls=":", lw=1.2, zorder=4,
               label=f"ERA5 1940–2015 mean = {T_era5_mean:.2f} °C")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        "ERA5 2-m temperature at South Pole (annual mean)\n"
        f"ERA5 warm bias correction: {ERA5_T_OFFSET_K:+.2f} K "
        f"(corrected ERA5 → Buizert 1950 at 1940–1960 boundary)",
        fontsize=9,
    )
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlim(*xl)

    # -------------------------------------------------------------------------
    # Panel 2: Accumulation
    # -------------------------------------------------------------------------
    ax = axes[1]
    ax.bar(run_years, run_A_raw, width=0.8, color="0.85", zorder=1,
           label=f"ERA5 raw annual total")
    ax.bar(run_years, run_A, width=0.8, color="0.55", alpha=0.7, zorder=2,
           label=f"ERA5 bias-corrected (+{ERA5_A_OFFSET_M_YR:.4f} m/yr)")
    ax.plot(run_years, A_rm, color="steelblue", lw=2.2, zorder=3,
            label="11-yr centred mean (corrected)")
    ax.axhline(A_buizert_mean, color="red", ls="--", lw=1.8, zorder=4,
               label=f"Buizert spinup mean = {A_buizert_mean:.4f} m/yr  [spinup]")
    ax.axhline(A_spin_sp19, color="purple", ls=":", lw=1.2, zorder=4,
               label=f"SP19-implied mean = {A_spin_sp19:.4f} m/yr")
    ax.axhline(A_era5_mean, color="green", ls=":", lw=1.2, zorder=4,
               label=f"ERA5 corrected mean = {A_era5_mean:.4f} m/yr")
    ax.set_ylabel("Accumulation (m ice eq/yr)")
    ax.set_title(
        "ERA5 net accumulation at South Pole (annual total)\n"
        f"Bias correction: +{ERA5_A_OFFSET_M_YR:.4f} m/yr  "
        f"(Buizert 1940 = 0.0980 m/yr vs ERA5 1940 raw = {run_A_raw[0]:.4f} m/yr)",
        fontsize=9,
    )
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, ls="--", alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 3: Cumulative A anomaly
    # -------------------------------------------------------------------------
    ax = axes[2]
    ax.axhline(0, color="k", lw=0.8, zorder=5)
    ax.fill_between(run_years, cum_anom_bui, 0, alpha=0.15, color="steelblue", zorder=1)
    ax.plot(run_years, cum_anom_bui, color="steelblue", lw=2.2, zorder=4,
            label=f"vs Buizert spinup mean ({A_buizert_mean:.4f} m/yr)  [model reference]")
    ax.plot(run_years, cum_anom_sp19, color="red", lw=1.6, ls="--", zorder=3,
            label=f"vs SP19-implied mean ({A_spin_sp19:.4f} m/yr)")
    ax.plot(run_years, cum_anom_era5, color="darkorange", lw=1.4, ls=":", zorder=3,
            label=f"vs ERA5 1940–60 mean ({A_spin_era5:.4f} m/yr)")

    # Annotate final values
    for val, col, offset_y in [
        (cum_anom_bui[-1],  "steelblue",   15),
        (cum_anom_sp19[-1], "red",         -15),
        (cum_anom_era5[-1], "darkorange",  -35),
    ]:
        ax.annotate(
            f"{val:+.2f} m",
            xy=(run_years[-1], val),
            xytext=(-55, offset_y), textcoords="offset points",
            fontsize=8, color=col,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative anomaly (m ice eq)")
    ax.set_title(
        "Cumulative ERA5 accumulation anomaly (ERA5 − reference rate)\n"
        "Negative = ERA5 era has accumulated less than the spinup rate "
        "→ deep layers buried more slowly than spinup assumed",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, ls="--", alpha=0.3)

    _T_bui_mean_C = float(np.mean(spin_T)) - 273.15
    _T_bui_range  = f"{spin_T.min()-273.15:.1f}–{spin_T.max()-273.15:.1f} °C"
    fig.suptitle(
        f"South Pole climate forcing  |  "
        f"Spinup: Buizert 2021 (T mean {_T_bui_mean_C:.1f} °C, A mean {A_buizert_mean:.4f} m/yr)  |  "
        f"ERA5 transient: {ERA5_START_YEAR}–{ERA5_END_YEAR}",
        fontsize=10, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])

    out_path = OUT_DIR / "diag_forcing.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.close(fig)

    # ==========================================================================
    # Figure 2 — full 740–2015 CE timeline: Buizert spinup + ERA5 transient
    # ==========================================================================
    print("Building full-timeline figure ...")

    # Combine spinup (annual) and ERA5 arrays for continuous x-axis
    full_years = np.concatenate([spin_years, run_years.astype(float)])
    full_T_C   = np.concatenate([spin_T - 273.15,  run_T_cor - 273.15])
    full_A     = np.concatenate([spin_A,            run_A])       # both bias-corrected

    # 11-yr running mean across the full timeline
    full_T_rm = running_mean(full_T_C, full_years, window=11)
    full_A_rm = running_mean(full_A,   full_years, window=11)

    fig2, axes2 = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig2.subplots_adjust(hspace=0.35)
    xl2 = (full_years[0] - 5, full_years[-1] + 5)

    # Vertical line at spinup → ERA5 junction
    junction = float(ERA5_START_YEAR)

    # ------------------------------------------------------------------
    # Figure 2, Panel 1: Temperature
    # ------------------------------------------------------------------
    ax = axes2[0]
    # Spinup shading and data
    ax.axvspan(full_years[0], junction, alpha=0.06, color="darkorange", zorder=0)
    ax.plot(spin_years, spin_T - 273.15, color="darkorange", lw=0.9, alpha=0.7, zorder=2,
            label="Buizert 2021 (annual interp)")
    # ERA5 data
    ax.plot(run_years, run_T_cor - 273.15, color="0.60", lw=0.9, zorder=2,
            label="ERA5 bias-corrected")
    # Running mean across full timeline
    ax.plot(full_years, full_T_rm, color="steelblue", lw=2.2, zorder=3,
            label="11-yr centred mean")
    # Junction line
    ax.axvline(junction, color="k", ls="--", lw=1.0, zorder=4,
               label=f"{ERA5_START_YEAR} CE: spinup → ERA5")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        "South Pole temperature 740–2015 CE\n"
        "Orange: Buizert 2021 spinup  |  Grey: ERA5 (bias-corrected)  |  "
        f"Blue: 11-yr running mean",
        fontsize=9,
    )
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlim(*xl2)

    # ------------------------------------------------------------------
    # Figure 2, Panel 2: Accumulation
    # ------------------------------------------------------------------
    ax = axes2[1]
    ax.axvspan(full_years[0], junction, alpha=0.06, color="darkorange", zorder=0)
    ax.bar(spin_years, spin_A, width=0.8, color="darkorange", alpha=0.6, zorder=1,
           label="Buizert 2021 (annual interp)")
    ax.bar(run_years, run_A_raw, width=0.8, color="0.75", alpha=0.7, zorder=1,
           label=f"ERA5 raw annual total")
    ax.bar(run_years, run_A, width=0.8, color="0.45", alpha=0.7, zorder=2,
           label=f"ERA5 corrected (+{ERA5_A_OFFSET_M_YR:.4f} m/yr)")
    ax.plot(full_years, full_A_rm, color="steelblue", lw=2.2, zorder=3,
            label="11-yr centred mean (corrected)")
    ax.axhline(A_buizert_mean, color="red", ls="--", lw=1.6, zorder=4,
               label=f"Buizert spinup mean = {A_buizert_mean:.4f} m/yr")
    ax.axhline(A_spin_sp19, color="purple", ls=":", lw=1.4, zorder=4,
               label=f"SP19-implied mean = {A_spin_sp19:.4f} m/yr")
    ax.axvline(junction, color="k", ls="--", lw=1.0, zorder=4,
               label=f"{ERA5_START_YEAR} CE: spinup → ERA5")
    # Annotate junction: corrected ERA5 should match Buizert closely
    ax.annotate(
        f"Junction (corrected): {spin_A[-1]:.3f}→{run_A[0]:.3f} m/yr  Δ={run_A[0]-spin_A[-1]:+.4f}",
        xy=(junction, (spin_A[-1] + run_A[0]) / 2),
        xytext=(30, 20), textcoords="offset points",
        fontsize=8, color="k",
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
    )
    ax.set_xlabel("Year CE")
    ax.set_ylabel("Accumulation (m ice eq/yr)")
    ax.set_title(
        "South Pole accumulation 740–2015 CE\n"
        "Orange: Buizert 2021 spinup  |  Grey: ERA5  |  "
        f"Blue: 11-yr running mean",
        fontsize=9,
    )
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, ls="--", alpha=0.3)

    fig2.suptitle(
        f"South Pole climate forcing — full timeline 740–2015 CE  |  "
        f"Spinup: Buizert 2021 ({SPINUP_YEARS} yr)  |  "
        f"Transient: ERA5 {ERA5_START_YEAR}–{ERA5_END_YEAR}",
        fontsize=10, y=0.995,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.975])

    out_path2 = OUT_DIR / "diag_forcing_timeline.png"
    fig2.savefig(out_path2, dpi=150)
    print(f"Saved: {out_path2}")
    plt.close(fig2)

    print("Done.")
