"""southpole_preinversion_diagnostic.py

Standalone diagnostic: run the South Pole firn forward model at default
parameters and compare against ALL available density and age observations.
No tape, no optimisation.

Datasets (with observation years):
  SP19 / SPICEcore density (den_SouthPole_2014_16)  — core year 2015 CE
    same data as processed/sp19_density.csv
  USP50 NICL density                                — drilled austral summer 2016-17
  USP50 field density (126 + 106 cores combined)    — drilled austral summer 2016-17
  USP50 snow pit                                    — 2017 surface snow
  SP19 depth-age (SPICEcore annual layers)          — core year 2015 CE

Observation-year consistency:
  SP19 density and age observations → model run ends at 2015
  USP50 density observations        → model run ends at 2017

ERA5 bias correction: -4.71 K (see inversion script for derivation).
Spinup temperature: Buizert 2021 pre-industrial value, 221.87 K (-51.28°C).

Outputs (all saved to test/southpole/results/):
  diag_data_overview.png  – all density datasets + age data (no model)
  diag_profiles.png       – model vs all datasets at matched years
  diag_residuals.png      – normalised residuals and J by depth (SP19)
  diag_j_by_depth.png     – per-observation J contribution (SP19)

Run:
    cd test/southpole/scripts
    OMP_NUM_THREADS=1 python southpole_preinversion_diagnostic.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

import firedrake as fd
from firedrake.adjoint import stop_annotating

from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.constants import year as YEAR_S

# =============================================================================
# Paths
# =============================================================================
_HERE             = Path(__file__).parent
_SOUTHPOLE        = _HERE.parent
PROCESSED         = _SOUTHPOLE / "processed"
ERA5_CSV          = PROCESSED / "era5_monthly_point.csv"
SP19_DENS_CSV     = PROCESSED / "sp19_density.csv"
SP19_AGE_CSV      = PROCESSED / "sp19_depth_age.csv"
BUIZERT_TEMP_CSV  = PROCESSED / "buizert2021_spice_temp.csv"
BUIZERT_ACCUM_CSV = PROCESSED / "buizert2021_spice_accum.csv"

# USP50 density files (Stevens 2023, austral summer 2016-17)
USP50_DIR     = _SOUTHPOLE / "data/usap_dc/601680/usapdc_601680"
USP50_NICL    = USP50_DIR / "USP50_density_nicl.csv"
USP50_106     = USP50_DIR / "USP50_density_106.csv"
USP50_126     = USP50_DIR / "USP50_density_126.csv"
USP50_PIT     = USP50_DIR / "USP50_density_pit.csv"

# insituData .mat (den_SouthPole_2014_16 == sp19_density.csv, same data)
INSITU_MAT    = _SOUTHPOLE / "data/usap_dc/601551/insituData/mrsldata_insitu_dens.mat"

OUT_DIR       = _SOUTHPOLE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Observation-year constants
# =============================================================================
# SP19 (SPICEcore): surface at year_CE = 2015 in the depth-age file.
SP19_CORE_YEAR  = 2015

# USP50: instruments first log at 2017-01-09, cores drilled austral summer 2016-17.
# We use 2016 as the representative "end of accumulation year" for the density core.
USP50_CORE_YEAR = 2017

# =============================================================================
# Model settings (must stay in sync with southpole_inversion_v1.py)
# =============================================================================
H0            = 130.0
NZ            = 100
STRETCH_P     = 2.5
SURFACE_ID    = 2

SPINUP_YEARS    = 1200
SPINUP_DT_YEARS = 1.0
ERA5_START_YEAR = 1940
DT_YEARS        = 1.0

ERA5_T_OFFSET_K    = -4.71     # K — ERA5 warm bias correction at South Pole
ERA5_A_OFFSET_M_YR = +0.0163  # m/yr — ERA5 accumulation bias correction
                                #   offset = Buizert_1940 - ERA5_1940 = 0.09795 - 0.08169
SPINUP_T_K      = 221.87       # K — Buizert 2021 pre-industrial South Pole

RHO_M     = 550.0
RHO_SMOOTH = 20.0

# Default (initial) parameters — same as inversion script
PARAMS = dict(kc0=9.2e-9, kc1=3.7e-9, kg=1.3e-7, rho_surf=350.0, r2_surf=2.5e-7)

SIGMA_RHO_ABS    = 15.0
SIGMA_RHO_REL    = 0.03
SIGMA_AGE_ABS_S  = 1.0 * float(YEAR_S)
SIGMA_AGE_REL    = 0.02
USE_UNIFORM_SIGMA = True

# =============================================================================
# Data loading
# =============================================================================

def load_era5_annual(csv_path):
    df = pd.read_csv(csv_path)
    t_rows = df[df["t2m_K"].notna()][["year", "t2m_K"]].copy()
    a_rows = df[df["net_accum_m_iceeq_month"].notna()][["year", "net_accum_m_iceeq_month"]].copy()
    T_by_yr = t_rows.groupby("year")["t2m_K"].mean()
    A_by_yr = a_rows.groupby("year")["net_accum_m_iceeq_month"].sum()
    years = np.array(sorted(T_by_yr.index.intersection(A_by_yr.index)), dtype=int)
    return years, T_by_yr.loc[years].values.astype(float), A_by_yr.loc[years].values.astype(float)


def load_buizert_spinup(spinup_start_year_ce, era5_start_year=ERA5_START_YEAR):
    """Return (years_ce, T_K, A_m_iceeq_yr) at annual resolution for the spinup.

    Linearly interpolates the Buizert 2021 SPICEcore reconstruction (40-year
    resolution) to annual time steps over [spinup_start_year_ce, era5_start_year).
    """
    df_T = pd.read_csv(BUIZERT_TEMP_CSV).sort_values("year_CE")
    df_A = pd.read_csv(BUIZERT_ACCUM_CSV).sort_values("year_CE")
    years_annual = np.arange(float(spinup_start_year_ce), float(era5_start_year), 1.0)
    T_annual = np.interp(years_annual, df_T["year_CE"].values,
                         df_T["temp"].values + 273.15)
    A_annual = np.interp(years_annual, df_A["year_CE"].values,
                         df_A["accum"].values)
    return years_annual, T_annual, A_annual


def load_sp19_density(depth_max=H0):
    """SPICEcore / Stevens density (2014-16 collection, core year 2015)."""
    df  = pd.read_csv(SP19_DENS_CSV)
    df  = df[df["depth_m"] <= depth_max].copy()
    dep = df["depth_m"].values.astype(float)
    rho = df["rho_kgm3"].values.astype(float) * 1000.0   # stored in g/cm³
    idx = np.argsort(dep)
    return dep[idx], rho[idx]


def load_usp50_nicl(depth_max=H0):
    """USP50 NICL lab density (Stevens 2023, core drilled austral summer 2016-17)."""
    df  = pd.read_csv(USP50_NICL)
    df  = df[df["depth"] <= depth_max].copy()
    dep = df["depth"].values.astype(float)
    rho = df["Density"].values.astype(float)              # already in kg/m³
    idx = np.argsort(dep)
    return dep[idx], rho[idx]


def load_usp50_field(depth_max=H0):
    """USP50 field-measured density: 126-core (top 12m) + 106-core (12-105m).

    As in USP50_datarelease.py (Stevens 2023):
      USP50_density_126.csv — field, kg/m³, top ~12 m
      USP50_density_106.csv — field, g/cm³ (×1000), ~6-105 m
    Combined: 126 for the top, then 106 where dep > last depth in 126.
    """
    df126 = pd.read_csv(USP50_126)
    df106 = pd.read_csv(USP50_106)
    dep126 = df126["depth"].values.astype(float)
    rho126 = df126["Density"].values.astype(float)        # kg/m³
    dep106 = df106["Core St"].values.astype(float)
    rho106 = df106["rhoavg"].values.astype(float) * 1000.0  # g/cm³ → kg/m³

    # Combine: use 126 for top, 106 for remainder (deeper than last 126 point)
    ind    = dep106 > dep126[-1]
    dep_c  = np.concatenate([dep126, dep106[ind]])
    rho_c  = np.concatenate([rho126, rho106[ind]])
    mask   = dep_c <= depth_max
    idx    = np.argsort(dep_c[mask])
    return dep_c[mask][idx], rho_c[mask][idx]


def load_usp50_pit():
    """USP50 snow-pit density (surface, 0-2 m, 2017)."""
    df  = pd.read_csv(USP50_PIT)
    dep = df.iloc[:, 0].values.astype(float)
    rho = df.iloc[:, 1].values.astype(float)
    idx = np.argsort(dep)
    return dep[idx], rho[idx]


def load_sp19_age(core_year=SP19_CORE_YEAR, depth_max=H0):
    """SP19 annual-layer depth-age (SPICEcore, core year 2015)."""
    df   = pd.read_csv(SP19_AGE_CSV)
    mask = (df["year_CE"] >= 0) & (df["year_CE"] <= core_year) & (df["depth_m"] <= depth_max)
    df   = df[mask].copy()
    dep  = df["depth_m"].values.astype(float)
    age_yr = float(core_year) - df["year_CE"].values.astype(float)
    age_s  = age_yr * float(YEAR_S)
    idx    = np.argsort(dep)
    return dep[idx], age_s[idx]


# =============================================================================
# Mesh / model helpers
# =============================================================================

def build_stretched_mesh(H0, nz, p):
    mesh = fd.IntervalMesh(nz, 0.0, 1.0)
    xi   = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** p)
    cfs    = mesh.coordinates.function_space()
    mesh.coordinates.assign(fd.Function(cfs).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    return mesh, V


def mesh_depth_sort(mesh, H0):
    x_flat = (mesh.coordinates.dat.data_ro[:, 0]
              if mesh.coordinates.dat.data_ro.ndim == 2
              else mesh.coordinates.dat.data_ro)
    depth = H0 - x_flat
    idx   = np.argsort(depth)
    return x_flat, depth, idx


def make_real(R, val, name):
    f = fd.Function(R, name=name)
    f.dat.data[:] = float(val)
    return f


def w_surf_value(params, accum_m_yr, rho_surf):
    return -accum_m_yr * params.rho_i / rho_surf / float(YEAR_S)


# =============================================================================
# Forward run (no tape)
# =============================================================================

def run_forward(params, spin_T, spin_A, run_T, run_A,
                mesh, V, rho_ic, age_ic,
                extra_years=0, verbose=True):
    """Spinup + ERA5 forward run (no tape).

    spin_T, spin_A : 1-D arrays of length SPINUP_YEARS giving the annual
                     Buizert 2021 temperature (K) and accumulation (m ice eq/yr)
                     for the spinup period.
    extra_years    : continue running past ERA5 for this many additional years
                     (useful to compare against USP50 core year 2016/17).
    Returns sorted-by-depth arrays.
    """
    R = fd.FunctionSpace(mesh, "R", 0)

    H     = fd.Function(V, name="H")
    rho   = fd.Function(V, name="rho")
    w     = fd.Function(V, name="w")
    sigma = fd.Function(V, name="sigma")
    r2    = fd.Function(V, name="r2")
    age   = fd.Function(V, name="age")

    Ts    = make_real(R, T_spin,  "Ts")
    accum = make_real(R, A_spin,  "accum")
    Hs    = make_real(R, float(params.c_i) * (T_spin - float(params.T_ref)), "Hs")
    dt_s  = make_real(R, float(DT_YEARS * YEAR_S), "dt")

    kc0_fn = make_real(R, PARAMS["kc0"], "kc0")
    kc1_fn = make_real(R, PARAMS["kc1"], "kc1")

    rho_surf_bc_fn = fd.Function(V, name="rho_surf_bc")
    rho_surf_bc_fn.interpolate(fd.Constant(PARAMS["rho_surf"]))
    rho_surf_fn = make_real(R, PARAMS["rho_surf"], "rho_surf_fs")

    w_surf_fn = fd.Function(V, name="w_surf_bc")
    r2_surf_fn = fd.Function(V, name="r2_surf_bc")
    r2_surf_fn.interpolate(fd.Constant(PARAMS["r2_surf"]))

    bc_H     = fd.DirichletBC(V, Hs,            SURFACE_ID)
    bc_rho   = fd.DirichletBC(V, rho_surf_bc_fn, SURFACE_ID)
    bc_w     = fd.DirichletBC(V, w_surf_fn,      SURFACE_ID)
    bc_sigma = fd.DirichletBC(V, make_real(R, 0.0, "sig0"), SURFACE_ID)
    bc_r2    = fd.DirichletBC(V, r2_surf_fn,     SURFACE_ID)
    bc_age   = fd.DirichletBC(V, make_real(R, 0.0, "age0"), SURFACE_ID)
    bcs      = [bc_H, bc_rho, bc_w]

    rhoCoef = fd.Function(V, name="rhoCoef")
    model   = FirnModel(params)
    solver  = FirnColumnSolver(model)

    def _step(T_k, A_k):
        Ts.assign(float(T_k))
        Hs.assign(float(params.c_i) * (float(T_k) - float(params.T_ref)))
        accum.assign(float(A_k))
        w_surf_fn.interpolate(fd.Constant(w_surf_value(params, float(A_k), PARAMS["rho_surf"])))
        s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
        rhoCoef.interpolate((1.0 - s) * kc0_fn + s * kc1_fn)
        solver.prognostic_solve(
            enthalpy=H, density=rho, firn_velocity=w, dt=dt_s,
            accumulation=accum, surface_density=rho_surf_fn,
            boundary_conditions=bcs, surface_temperature=Ts,
            enthalpy_bc_constant=Hs,
            stress=sigma, grain_radius2=r2,
            stress_boundary_condition=bc_sigma,
            grain_radius2_boundary_condition=bc_r2,
            rhoCoef=rhoCoef,
            age=age, age_boundary_condition=bc_age,
        )

    with stop_annotating():
        rho.assign(rho_ic)
        H.assign(float(params.c_i) * (float(spin_T[0]) - float(params.T_ref)))
        sigma.assign(0.0)
        r2.assign(PARAMS["r2_surf"])
        w.assign(0.0)
        age.assign(age_ic)

    n_spin = int(round(SPINUP_YEARS / SPINUP_DT_YEARS))
    if verbose:
        print(f"    Spinup: {n_spin} steps  "
              f"T={spin_T.min()-273.15:.2f}–{spin_T.max()-273.15:.2f} °C  "
              f"A={spin_A.min():.4f}–{spin_A.max():.4f} m/yr (Buizert 2021)")
    with stop_annotating():
        for _k in range(n_spin):
            _step(float(spin_T[_k]), float(spin_A[_k]))

    if verbose:
        print(f"    ERA5 run: {len(run_T)} steps")
    with stop_annotating():
        for k in range(len(run_T)):
            _step(run_T[k], run_A[k])

    # Optional extra years beyond ERA5 window (to match later observation years)
    if extra_years > 0:
        if verbose:
            print(f"    Extra {extra_years} years at last ERA5 values")
        with stop_annotating():
            for _ in range(extra_years):
                _step(run_T[-1], run_A[-1])

    x_flat, depth, sort_idx = mesh_depth_sort(mesh, H0)
    P0_ref = FirnParameters()
    return {
        "depth"  : depth[sort_idx],
        "rho"    : rho.dat.data_ro[sort_idx].copy(),
        "age_yr" : age.dat.data_ro[sort_idx].copy() / float(YEAR_S),
        "T_C"    : H.dat.data_ro[sort_idx].copy() / float(P0_ref.c_i) + float(P0_ref.T_ref) - 273.15,
        "w"      : w.dat.data_ro[sort_idx].copy(),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_data_overview(sp19_rho, usp50_nicl, usp50_field, usp50_pit,
                       sp19_age, out_path):
    """Panel 1: all density datasets + age profile (no model)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 8))

    ax = axes[0]
    ax.set_title("South Pole — all density datasets", fontsize=10)

    # USP50 datasets first (they go deeper)
    pit_dep, pit_rho = usp50_pit
    ax.scatter(pit_rho, pit_dep, s=15, color="purple", zorder=5,
               label=f"USP50 pit (2017, N={len(pit_dep)})")

    nicl_dep, nicl_rho = usp50_nicl
    ax.plot(nicl_rho, nicl_dep, ".-", color="darkorange", ms=4, lw=1,
            label=f"USP50 NICL (2016-17, N={len(nicl_dep)})")

    field_dep, field_rho = usp50_field
    ax.plot(field_rho, field_dep, "s-", color="green", ms=3, lw=1, mfc="none",
            label=f"USP50 field 126+106 (2016-17, N={len(field_dep)})")

    sp19_dep, sp19_rho_arr = sp19_rho
    ax.plot(sp19_rho_arr, sp19_dep, "o", color="steelblue", ms=3, mfc="none",
            label=f"SP19 / SPICEcore (2015, N={len(sp19_dep)})")

    ax.set_xlabel("Density (kg/m³)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_xlim(150, 950)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, ls="--", alpha=0.4)

    ax = axes[1]
    age_dep, age_s = sp19_age
    ax.plot(age_s / float(YEAR_S), age_dep, ".", color="steelblue", ms=3,
            label=f"SP19 annual layers (2015, N={len(age_dep)})")
    ax.set_xlabel("Age (yr CE 2015 − year_CE)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title("SP19 depth-age", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)

    fig.suptitle("South Pole firn observations — overview (no model)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_model_vs_data(state_2015, state_2017,
                       sp19_rho, usp50_nicl, usp50_field, sp19_age,
                       sigma_rho, sigma_age,
                       J_rho, J_age, out_path):
    """Model profiles vs all observations — separate model lines for 2015 and 2017."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    # --- Density ---
    ax = axes[0]
    ax.plot(state_2015["rho"], state_2015["depth"], "b-",   lw=1.8, label="Model (end 2015)")
    ax.plot(state_2017["rho"], state_2017["depth"], "b--",  lw=1.2, label="Model (end 2017)")

    sp19_dep, sp19_rho_arr = sp19_rho
    ax.errorbar(sp19_rho_arr, sp19_dep, xerr=sigma_rho,
                fmt="o", color="steelblue", ms=3, elinewidth=0.4, alpha=0.7,
                label=f"SP19 2015 (N={len(sp19_dep)})")

    nicl_dep, nicl_rho = usp50_nicl
    ax.plot(nicl_rho, nicl_dep, ".-", color="darkorange", ms=4, lw=0.8,
            label=f"USP50 NICL 2016-17 (N={len(nicl_dep)})")

    field_dep, field_rho = usp50_field
    ax.plot(field_rho, field_dep, "s", color="green", ms=4, mfc="none",
            label=f"USP50 field 2016-17 (N={len(field_dep)})")

    ax.set_xlabel("Density (kg/m³)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_xlim(200, 950)
    ax.set_title(f"Density  (SP19 J_rho={J_rho:.2f})")
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", alpha=0.4)

    # --- Age ---
    ax = axes[1]
    ax.plot(state_2015["age_yr"], state_2015["depth"], "b-", lw=1.8, label="Model (end 2015)")

    age_dep, age_s = sp19_age
    ax.errorbar(age_s / float(YEAR_S), age_dep,
                xerr=sigma_age / float(YEAR_S),
                fmt=".", color="steelblue", ms=3, elinewidth=0.4, alpha=0.7,
                label=f"SP19 2015 (N={len(age_dep)})")
    ax.set_xlabel("Age (yr)")
    ax.invert_yaxis()
    ax.set_title(f"Age  (SP19 J_age={J_age:.2f})")
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", alpha=0.4)

    # --- Temperature ---
    ax = axes[2]
    ax.plot(state_2015["T_C"], state_2015["depth"], "g-",  lw=1.8, label="Model (end 2015)")
    ax.plot(state_2017["T_C"], state_2017["depth"], "g--", lw=1.2, label="Model (end 2017)")
    ax.axvline(-51.28, color="orange", ls="--", lw=0.9, label="Buizert 1950 (-51.3°C)")
    ax.set_xlabel("Temperature (°C)")
    ax.invert_yaxis()
    ax.set_title("Temperature")
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", alpha=0.4)

    fig.suptitle(
        f"South Pole forward model vs observations\n"
        f"Spinup {SPINUP_YEARS} yr Buizert 2021 + ERA5 {ERA5_START_YEAR}–2015/2017 | "
        f"T_spin {spin_T.min()-273.15:.1f}–{spin_T.max()-273.15:.1f} °C  "
        f"kc0={PARAMS['kc0']:.2e}  kc1={PARAMS['kc1']:.2e}  kg={PARAMS['kg']:.2e}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_residuals(sp19_dep, res_rho, rho_model_at_obs, sp19_rho_arr,
                   age_dep, res_age, age_model_yr, obs_age_yr,
                   out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(res_rho, sp19_dep, s=6, alpha=0.6, color="steelblue")
    ax.axvline(0, color="k", lw=0.8)
    for xv in (-2, 2): ax.axvline(xv, color="r", ls="--", lw=0.6)
    ax.set_xlabel("Normalised residual (σ)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_title("SP19 density residuals vs depth")
    ax.grid(True, ls="--", alpha=0.4)

    ax = axes[0, 1]
    v = [min(sp19_rho_arr.min(), rho_model_at_obs.min()),
         max(sp19_rho_arr.max(), rho_model_at_obs.max())]
    ax.scatter(sp19_rho_arr, rho_model_at_obs, s=6, alpha=0.6, color="steelblue")
    ax.plot(v, v, "k--", lw=0.8)
    rms = np.sqrt(np.mean((rho_model_at_obs - sp19_rho_arr)**2))
    bias = np.mean(rho_model_at_obs - sp19_rho_arr)
    ax.set_xlabel("SP19 density (kg/m³)"); ax.set_ylabel("Model density (kg/m³)")
    ax.set_title(f"Density model vs obs  RMS={rms:.1f} kg/m³  bias={bias:+.1f}")
    ax.grid(True, ls="--", alpha=0.4)

    ax = axes[1, 0]
    ax.scatter(res_age, age_dep, s=4, alpha=0.5, color="darkorange")
    ax.axvline(0, color="k", lw=0.8)
    for xv in (-2, 2): ax.axvline(xv, color="r", ls="--", lw=0.6)
    ax.set_xlabel("Normalised residual (σ)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_title("SP19 age residuals vs depth")
    ax.grid(True, ls="--", alpha=0.4)

    ax = axes[1, 1]
    v2 = [min(obs_age_yr.min(), age_model_yr.min()),
          max(obs_age_yr.max(), age_model_yr.max())]
    ax.scatter(obs_age_yr, age_model_yr, s=4, alpha=0.5, color="darkorange")
    ax.plot(v2, v2, "k--", lw=0.8)
    rms2  = np.sqrt(np.mean((age_model_yr - obs_age_yr)**2))
    bias2 = np.mean(age_model_yr - obs_age_yr)
    ax.set_xlabel("SP19 age (yr)"); ax.set_ylabel("Model age (yr)")
    ax.set_title(f"Age model vs obs  RMS={rms2:.1f} yr  bias={bias2:+.1f} yr")
    ax.grid(True, ls="--", alpha=0.4)

    fig.suptitle("SP19 — normalised residuals at initial parameters", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_j_by_depth(sp19_dep, res_rho, age_dep, res_age, out_path):
    j_rho_i = 0.5 * res_rho ** 2
    j_age_i = 0.5 * res_age ** 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))

    ax = axes[0]
    ax.barh(sp19_dep, j_rho_i, height=0.6, color="steelblue", alpha=0.7)
    ax.invert_yaxis()
    ax.axvline(0.5, color="r", ls="--", lw=0.8, label="1σ")
    ax.axvline(2.0, color="r", ls=":", lw=0.8, label="2σ")
    ax.set_xlabel("J contribution  0.5·(Δρ/σ)²"); ax.set_ylabel("Depth (m)")
    ax.set_title(f"Density J by depth  total={j_rho_i.sum():.1f}  mean={j_rho_i.mean():.2f}")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.4)

    ax = axes[1]
    ax.barh(age_dep, j_age_i, height=0.3, color="darkorange", alpha=0.7)
    ax.invert_yaxis()
    ax.axvline(0.5, color="r", ls="--", lw=0.8, label="1σ")
    ax.axvline(2.0, color="r", ls=":", lw=0.8, label="2σ")
    ax.set_xlabel("J contribution  0.5·(Δage/σ)²")
    ax.set_title(f"Age J by depth  total={j_age_i.sum():.1f}  mean={j_age_i.mean():.2f}")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.4)

    fig.suptitle("J contribution by observation depth (SP19, initial parameters)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("South Pole pre-inversion diagnostic")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Load all observations
    # -------------------------------------------------------------------------
    print("\nLoading ERA5 ...")
    era5_years, era5_T, era5_A = load_era5_annual(ERA5_CSV)
    run_mask_2015  = (era5_years >= ERA5_START_YEAR) & (era5_years < SP19_CORE_YEAR)
    run_years_2015 = era5_years[run_mask_2015]
    run_T_2015     = era5_T[run_mask_2015] + ERA5_T_OFFSET_K
    run_A_2015     = era5_A[run_mask_2015] + ERA5_A_OFFSET_M_YR

    # For USP50 comparison: extend ERA5 run to USP50_CORE_YEAR
    run_mask_usp   = (era5_years >= ERA5_START_YEAR) & (era5_years < USP50_CORE_YEAR)
    run_T_usp      = era5_T[run_mask_usp] + ERA5_T_OFFSET_K
    run_A_usp      = era5_A[run_mask_usp] + ERA5_A_OFFSET_M_YR

    print(f"  SP19 ERA5 window : {run_years_2015[0]}–{run_years_2015[-1]}  "
          f"({len(run_years_2015)} steps)")
    print(f"  USP50 ERA5 window: {ERA5_START_YEAR}–{USP50_CORE_YEAR-1}  "
          f"({len(run_T_usp)} steps)")

    print("\nLoading Buizert 2021 SPICEcore spinup reconstruction ...")
    _spin_start_ce = ERA5_START_YEAR - SPINUP_YEARS
    _spin_years, spin_T, spin_A = load_buizert_spinup(_spin_start_ce, ERA5_START_YEAR)
    print(f"  Spinup: {_spin_years[0]:.0f}–{_spin_years[-1]:.0f} CE  "
          f"T={spin_T.min()-273.15:.2f}–{spin_T.max()-273.15:.2f} °C  "
          f"A mean={np.mean(spin_A):.4f} m/yr")

    print("\nLoading density datasets ...")
    sp19_rho_dep, sp19_rho_val = load_sp19_density()
    print(f"  SP19 / SPICEcore (2015): N={len(sp19_rho_dep)}, "
          f"depth {sp19_rho_dep[0]:.2f}–{sp19_rho_dep[-1]:.2f} m")

    usp50_nicl_dep, usp50_nicl_val = load_usp50_nicl()
    print(f"  USP50 NICL (2016-17):  N={len(usp50_nicl_dep)}, "
          f"depth {usp50_nicl_dep[0]:.2f}–{usp50_nicl_dep[-1]:.2f} m")

    usp50_field_dep, usp50_field_val = load_usp50_field()
    print(f"  USP50 field (2016-17): N={len(usp50_field_dep)}, "
          f"depth {usp50_field_dep[0]:.2f}–{usp50_field_dep[-1]:.2f} m")

    usp50_pit_dep, usp50_pit_val = load_usp50_pit()
    print(f"  USP50 pit (2017):      N={len(usp50_pit_dep)}, "
          f"depth {usp50_pit_dep[0]:.2f}–{usp50_pit_dep[-1]:.2f} m")

    print("\nLoading SP19 age ...")
    sp19_age_dep, sp19_age_s = load_sp19_age()
    obs_age_yr = sp19_age_s / float(YEAR_S)
    print(f"  SP19 age (2015): N={len(sp19_age_dep)}, "
          f"depth {sp19_age_dep[0]:.2f}–{sp19_age_dep[-1]:.2f} m, "
          f"age {obs_age_yr.min():.1f}–{obs_age_yr.max():.1f} yr")

    # -------------------------------------------------------------------------
    # Uncertainties (same as inversion script)
    # -------------------------------------------------------------------------
    sigma_rho = SIGMA_RHO_ABS + SIGMA_RHO_REL * np.abs(sp19_rho_val)
    sigma_age = SIGMA_AGE_ABS_S + SIGMA_AGE_REL * np.maximum(sp19_age_s, 0.0)
    if USE_UNIFORM_SIGMA:
        sigma_rho = np.full_like(sigma_rho, float(np.median(sigma_rho)))
        sigma_age = np.full_like(sigma_age, float(np.median(sigma_age)))
    print(f"\n  sigma_rho = {np.median(sigma_rho):.1f} kg/m³ (uniform median)")
    print(f"  sigma_age = {np.median(sigma_age)/float(YEAR_S):.1f} yr (uniform median)")

    # -------------------------------------------------------------------------
    # Mesh + ICs
    # -------------------------------------------------------------------------
    print("\nBuilding mesh ...")
    mesh, V = build_stretched_mesh(H0, NZ, STRETCH_P)
    P0 = FirnParameters(kg=PARAMS["kg"])

    _ic_dep, _ic_rho = load_sp19_density(depth_max=2.0 * H0)
    _ic_dep = np.concatenate([[0.0], _ic_dep])
    _ic_rho = np.concatenate([[PARAMS["rho_surf"]], _ic_rho])
    _x_flat, _depth, _ = mesh_depth_sort(mesh, H0)
    _rho_nodes = np.clip(
        np.interp(_depth, _ic_dep, _ic_rho,
                  left=PARAMS["rho_surf"], right=float(_ic_rho[-1])),
        200.0, float(P0.rho_i) - 0.5,
    )
    rho_ic = fd.Function(V, name="rho_ic")
    rho_ic.dat.data[:] = _rho_nodes

    _sort_i    = np.argsort(_x_flat)
    _x_s       = _x_flat[_sort_i]
    _rho_s     = _rho_nodes[_sort_i]
    _cumint    = np.concatenate([[0.0], np.cumsum(0.5*(_rho_s[:-1]+_rho_s[1:])*np.diff(_x_s))])
    _age_nodes = np.empty_like(_rho_nodes)
    _age_nodes[_sort_i] = float(YEAR_S) / (float(np.mean(spin_A)) * float(P0.rho_i)) * (_cumint[-1] - _cumint)
    age_ic = fd.Function(V, name="age_ic")
    age_ic.dat.data[:] = _age_nodes
    print(f"  Age IC max = {_age_nodes.max()/float(YEAR_S):.0f} yr at bottom")

    # -------------------------------------------------------------------------
    # Plot 1: raw data overview (no model)
    # -------------------------------------------------------------------------
    print("\nPlotting data overview ...")
    plot_data_overview(
        (sp19_rho_dep, sp19_rho_val),
        (usp50_nicl_dep, usp50_nicl_val),
        (usp50_field_dep, usp50_field_val),
        (usp50_pit_dep, usp50_pit_val),
        (sp19_age_dep, sp19_age_s),
        OUT_DIR / "diag_data_overview.png",
    )

    # -------------------------------------------------------------------------
    # Run forward models at SP19 year (2015) and USP50 year (2017)
    # -------------------------------------------------------------------------
    print(f"\nRunning forward model to SP19 core year ({SP19_CORE_YEAR}) ...")
    state_2015 = run_forward(P0, spin_T, spin_A, run_T_2015, run_A_2015,
                             mesh, V, rho_ic, age_ic, extra_years=0, verbose=True)

    print(f"\nRunning forward model to USP50 core year ({USP50_CORE_YEAR}) ...")
    state_2017 = run_forward(P0, spin_T, spin_A, run_T_usp, run_A_usp,
                             mesh, V, rho_ic, age_ic, extra_years=0, verbose=True)

    # -------------------------------------------------------------------------
    # Compute SP19 J and residuals
    # -------------------------------------------------------------------------
    rho_m_at_sp19  = np.interp(sp19_rho_dep, state_2015["depth"], state_2015["rho"])
    age_m_at_sp19  = np.interp(sp19_age_dep, state_2015["depth"], state_2015["age_yr"])
    res_rho = (rho_m_at_sp19 - sp19_rho_val) / sigma_rho
    res_age = (age_m_at_sp19 - obs_age_yr) / (sigma_age / float(YEAR_S))

    J_rho = 0.5 * np.mean(res_rho ** 2)
    J_age = 0.5 * np.mean(res_age ** 2)

    print(f"\nObjective (SP19, initial params):")
    print(f"  J_rho  = {J_rho:.4f}   N_rho={len(sp19_rho_dep)}")
    print(f"  J_age  = {J_age:.4f}   N_age={len(sp19_age_dep)}")
    print(f"  J_total= {J_rho+J_age:.4f}")
    rms_rho = np.sqrt(np.mean((rho_m_at_sp19 - sp19_rho_val)**2))
    bias_rho = np.mean(rho_m_at_sp19 - sp19_rho_val)
    rms_age  = np.sqrt(np.mean((age_m_at_sp19 - obs_age_yr)**2))
    bias_age = np.mean(age_m_at_sp19 - obs_age_yr)
    print(f"\nResidue (SP19):")
    print(f"  density: RMS={rms_rho:.1f} kg/m³  bias={bias_rho:+.1f} kg/m³  "
          f"  {100*np.mean(np.abs(res_rho)>2):.0f}% > 2σ")
    print(f"  age:     RMS={rms_age:.1f} yr     bias={bias_age:+.1f} yr  "
          f"  {100*np.mean(np.abs(res_age)>2):.0f}% > 2σ")

    # Cross-compare USP50 to 2017 model
    rho_m_at_nicl  = np.interp(usp50_nicl_dep,  state_2017["depth"], state_2017["rho"])
    rho_m_at_field = np.interp(usp50_field_dep, state_2017["depth"], state_2017["rho"])
    rms_nicl  = np.sqrt(np.mean((rho_m_at_nicl  - usp50_nicl_val)**2))
    rms_field = np.sqrt(np.mean((rho_m_at_field - usp50_field_val)**2))
    print(f"\nResidual vs USP50 (model 2017):")
    print(f"  NICL  RMS = {rms_nicl:.1f} kg/m³  bias = {np.mean(rho_m_at_nicl-usp50_nicl_val):+.1f}")
    print(f"  field RMS = {rms_field:.1f} kg/m³  bias = {np.mean(rho_m_at_field-usp50_field_val):+.1f}")

    # -------------------------------------------------------------------------
    # Remaining plots
    # -------------------------------------------------------------------------
    print("\nGenerating plots ...")
    plot_model_vs_data(
        state_2015, state_2017,
        (sp19_rho_dep, sp19_rho_val),
        (usp50_nicl_dep, usp50_nicl_val),
        (usp50_field_dep, usp50_field_val),
        (sp19_age_dep, sp19_age_s),
        sigma_rho, sigma_age,
        J_rho, J_age,
        OUT_DIR / "diag_profiles.png",
    )
    plot_residuals(
        sp19_rho_dep, res_rho, rho_m_at_sp19, sp19_rho_val,
        sp19_age_dep, res_age, age_m_at_sp19, obs_age_yr,
        OUT_DIR / "diag_residuals.png",
    )
    plot_j_by_depth(
        sp19_rho_dep, res_rho,
        sp19_age_dep, res_age,
        OUT_DIR / "diag_j_by_depth.png",
    )

    print("\nDone.")
