"""spinup_resolution_sweep.py

Standalone spinup experiment to test sensitivity of South Pole firn
profiles to spatial resolution (NZ) and temporal resolution (dt).

Runs the Buizert 2021 spinup (740-1940 CE) + ERA5 (1940-2015) at
several resolution combinations and compares final density/age profiles
against SP19 observations.

Run:
    cd test/southpole/scripts
    OMP_NUM_THREADS=1 /home/andrew/venv-firedrake/bin/python spinup_resolution_sweep.py
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import firedrake as fd

from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.constants import year as YEAR_S

# =============================================================================
# Paths
# =============================================================================
_HERE = Path(__file__).parent
_SOUTHPOLE = _HERE.parent
PROCESSED_DIR = _SOUTHPOLE / "processed"
ERA5_CSV       = PROCESSED_DIR / "era5_monthly_point.csv"
SP19_DENS_CSV  = PROCESSED_DIR / "sp19_density.csv"
SP19_AGE_CSV   = PROCESSED_DIR / "sp19_depth_age.csv"
BUIZERT_TEMP_CSV  = PROCESSED_DIR / "buizert2021_spice_temp.csv"
BUIZERT_ACCUM_CSV = PROCESSED_DIR / "buizert2021_spice_accum.csv"
OUT_DIR = _SOUTHPOLE / "results"

# =============================================================================
# Fixed physical settings (matching v6 inversion)
# =============================================================================
H0 = 130.0
SURFACE_ID = 2
CORE_YEAR = 2015
ERA5_START_YEAR = 1940
SPINUP_START_CE = 740
ERA5_T_OFFSET_K = 0.0
ERA5_A_OFFSET_M_YR = +0.0163

P0 = FirnParameters()
RHO_M = float(getattr(P0, "rho_m", 550.0))
RHO_SMOOTH = 20.0
RHO_SURF = 350.0
R2_SURF = 2.5e-7

# =============================================================================
# Resolution cases to sweep
# =============================================================================
CASES = [
    # (label, NZ, STRETCH_P, spinup_dt_yr, era5_dt_yr)
    ("NZ100_dt10_dt0.5",   100, 2.5,  10.0, 0.5),   # baseline (v6 settings)
    ("NZ100_dt1_dt0.5",    100, 2.5,   1.0, 0.5),   # 10x finer spinup dt
    ("NZ100_dt1_dt0.25",   100, 2.5,   1.0, 0.25),  # finer ERA5 dt too
    ("NZ200_dt10_dt0.5",   200, 2.5,  10.0, 0.5),   # 2x spatial resolution
    ("NZ200_dt1_dt0.5",    200, 2.5,   1.0, 0.5),   # 2x spatial + fine spinup
    ("NZ50_dt10_dt0.5",     50, 2.5,  10.0, 0.5),   # coarser spatial
]


# =============================================================================
# Data loading (same as v6)
# =============================================================================
def load_era5_annual(csv_path):
    df = pd.read_csv(csv_path)
    t_rows = df[df["t2m_K"].notna()][["year", "t2m_K"]].copy()
    a_rows = df[df["net_accum_m_iceeq_month"].notna()][["year", "net_accum_m_iceeq_month"]].copy()
    T_by_yr = t_rows.groupby("year")["t2m_K"].mean()
    A_by_yr = a_rows.groupby("year")["net_accum_m_iceeq_month"].sum()
    years = np.array(sorted(T_by_yr.index.intersection(A_by_yr.index)), dtype=int)
    return years, T_by_yr.loc[years].values.astype(float), A_by_yr.loc[years].values.astype(float)


def load_sp19_density(csv_path, depth_max=130.0):
    df = pd.read_csv(csv_path)
    df = df[df["depth_m"] <= depth_max].copy()
    depth = df["depth_m"].values.astype(float)
    rho = df["rho_kgm3"].values.astype(float) * 1000.0
    idx = np.argsort(depth)
    return depth[idx], rho[idx]


def load_sp19_age(csv_path, core_year=CORE_YEAR, depth_max=130.0):
    df = pd.read_csv(csv_path)
    mask = (df["year_CE"] <= core_year) & (df["depth_m"] <= depth_max)
    df = df[mask].copy()
    depth = df["depth_m"].values.astype(float)
    age_yr = float(core_year) - df["year_CE"].values.astype(float)
    idx = np.argsort(depth)
    return depth[idx], age_yr[idx]


def build_stretched_mesh(H0, nz, p):
    mesh = fd.IntervalMesh(nz, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** p)
    coord_fs = mesh.coordinates.function_space()
    mesh.coordinates.assign(fd.Function(coord_fs).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    return mesh, V


def make_real(R, val, name):
    f = fd.Function(R, name=name)
    f.assign(float(val))
    return f


# =============================================================================
# Run one spinup + ERA5 forward pass
# =============================================================================
def run_case(label, nz, stretch_p, spinup_dt_yr, era5_dt_yr,
             spin_T, spin_A, era5_T, era5_A):
    """Run spinup + ERA5 forward and return depth, rho, age, T profiles."""

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Case: {label}")
    print(f"  NZ={nz}, stretch_p={stretch_p}")
    print(f"  spinup dt={spinup_dt_yr}yr, ERA5 dt={era5_dt_yr}yr")
    print(f"{'='*60}")

    mesh, V = build_stretched_mesh(H0, nz, stretch_p)
    R = fd.FunctionSpace(mesh, "R", 0)

    params = FirnParameters()
    model = FirnModel(params)
    solver = FirnColumnSolver(model, surface_id=SURFACE_ID)

    H     = fd.Function(V, name="H")
    rho   = fd.Function(V, name="rho")
    w     = fd.Function(V, name="w")
    sigma = fd.Function(V, name="sigma")
    r2    = fd.Function(V, name="r2")
    age   = fd.Function(V, name="age")

    # Initial guess
    T0 = float(spin_T[0])
    A0 = float(spin_A[0])
    H.assign(float(params.c_i) * (T0 - float(params.T_ref)))
    xi = fd.SpatialCoordinate(mesh)[0]
    depth = H0 - xi
    rho.interpolate(RHO_SURF + (float(params.rho_i) - RHO_SURF) * (1.0 - fd.exp(-depth / 20.0)))
    w.assign(-A0 * float(params.rho_i) / RHO_SURF / float(YEAR_S))
    sigma.assign(0.0)
    r2.assign(R2_SURF)
    age.assign(0.0)

    dt_fn   = make_real(R, spinup_dt_yr * YEAR_S, "dt")
    Ts_fn   = make_real(R, T0, "Ts")
    Hs_fn   = make_real(R, float(params.c_i) * (T0 - float(params.T_ref)), "Hs")
    accum_fn = make_real(R, A0, "accum")
    rho_s_fn = make_real(R, RHO_SURF, "rho_s")
    w_surf_fn = make_real(R, -A0 * float(params.rho_i) / RHO_SURF / float(YEAR_S), "ws")

    bc_H = fd.DirichletBC(V, Hs_fn, SURFACE_ID)
    bc_rho = fd.DirichletBC(V, rho_s_fn, SURFACE_ID)
    bc_w = fd.DirichletBC(V, w_surf_fn, SURFACE_ID)
    bc_sigma = fd.DirichletBC(V, make_real(R, 0.0, "s0"), SURFACE_ID)
    bc_r2 = fd.DirichletBC(V, make_real(R, R2_SURF, "r2s"), SURFACE_ID)
    bc_age = fd.DirichletBC(V, make_real(R, 0.0, "a0"), SURFACE_ID)

    rhoCoef = fd.Function(V, name="rhoCoef")

    # --- Resample Buizert spinup to requested dt ---
    blk = int(round(spinup_dt_yr))
    n_blk = len(spin_T) // blk
    spin_T_rs = np.array([spin_T[i*blk:(i+1)*blk].mean() for i in range(n_blk)])
    spin_A_rs = np.array([spin_A[i*blk:(i+1)*blk].mean() for i in range(n_blk)])
    n_spin = len(spin_T_rs)

    print(f"  Spinup: {n_spin} steps × dt={spinup_dt_yr}yr = {n_spin*spinup_dt_yr:.0f}yr")

    # --- Spinup ---
    dt_fn.assign(spinup_dt_yr * YEAR_S)
    for step in range(n_spin):
        Tk = float(spin_T_rs[step])
        Ak = float(spin_A_rs[step])
        Ts_fn.assign(Tk)
        Hs_fn.assign(float(params.c_i) * (Tk - float(params.T_ref)))
        accum_fn.assign(Ak)
        w_surf_fn.assign(-Ak * float(params.rho_i) / RHO_SURF / float(YEAR_S))

        s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
        rhoCoef.interpolate((1.0 - s) * float(params.kc0) + s * float(params.kc1))
        solver.prognostic_solve(
            enthalpy=H, density=rho, firn_velocity=w, dt=dt_fn,
            accumulation=accum_fn, surface_density=rho_s_fn,
            boundary_conditions=[bc_H, bc_rho, bc_w],
            surface_temperature=Ts_fn, enthalpy_bc_constant=Hs_fn,
            stress=sigma, grain_radius2=r2,
            stress_boundary_condition=bc_sigma,
            grain_radius2_boundary_condition=bc_r2,
            rhoCoef=rhoCoef,
            age=age, age_boundary_condition=bc_age,
        )

    rho_post_spinup = rho.dat.data_ro.copy()
    t_spinup = time.time() - t0
    print(f"  After spinup: rho=[{rho.dat.data_ro.min():.1f}, {rho.dat.data_ro.max():.1f}], "
          f"age_max={age.dat.data_ro.max()/YEAR_S:.0f}yr  ({t_spinup:.0f}s)")

    # --- ERA5 forward ---
    n_substeps = int(round(1.0 / era5_dt_yr))
    era5_T_sub = np.repeat(era5_T, n_substeps)
    era5_A_sub = np.repeat(era5_A, n_substeps)
    n_era5 = len(era5_T_sub)
    dt_fn.assign(era5_dt_yr * YEAR_S)
    print(f"  ERA5: {n_era5} steps × dt={era5_dt_yr}yr")

    for step in range(n_era5):
        Tk = float(era5_T_sub[step])
        Ak = float(era5_A_sub[step])
        Ts_fn.assign(Tk)
        Hs_fn.assign(float(params.c_i) * (Tk - float(params.T_ref)))
        accum_fn.assign(Ak)
        w_surf_fn.assign(-Ak * float(params.rho_i) / RHO_SURF / float(YEAR_S))

        s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
        rhoCoef.interpolate((1.0 - s) * float(params.kc0) + s * float(params.kc1))
        solver.prognostic_solve(
            enthalpy=H, density=rho, firn_velocity=w, dt=dt_fn,
            accumulation=accum_fn, surface_density=rho_s_fn,
            boundary_conditions=[bc_H, bc_rho, bc_w],
            surface_temperature=Ts_fn, enthalpy_bc_constant=Hs_fn,
            stress=sigma, grain_radius2=r2,
            stress_boundary_condition=bc_sigma,
            grain_radius2_boundary_condition=bc_r2,
            rhoCoef=rhoCoef,
            age=age, age_boundary_condition=bc_age,
        )

    t_total = time.time() - t0
    print(f"  Final: rho=[{rho.dat.data_ro.min():.1f}, {rho.dat.data_ro.max():.1f}], "
          f"age_max={age.dat.data_ro.max()/YEAR_S:.0f}yr  ({t_total:.0f}s)")

    # Extract profiles on sorted depth grid
    mesh_x = (mesh.coordinates.dat.data_ro[:, 0]
              if mesh.coordinates.dat.data_ro.ndim == 2
              else mesh.coordinates.dat.data_ro)
    depth_nodes = H0 - mesh_x
    idx = np.argsort(depth_nodes)

    T_profile = H.dat.data_ro[idx] / float(params.c_i) + float(params.T_ref) - 273.15

    return {
        "label": label,
        "depth": depth_nodes[idx],
        "rho": rho.dat.data_ro[idx].copy(),
        "age_yr": age.dat.data_ro[idx].copy() / float(YEAR_S),
        "T_C": T_profile,
        "w": w.dat.data_ro[idx].copy(),
        "time_s": t_total,
    }


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    era5_years, era5_T, era5_A = load_era5_annual(ERA5_CSV)
    mask = (era5_years >= ERA5_START_YEAR) & (era5_years < CORE_YEAR)
    era5_T_phase = era5_T[mask] + ERA5_T_OFFSET_K
    era5_A_phase = era5_A[mask] + ERA5_A_OFFSET_M_YR

    obs_rho_depth, obs_rho_val = load_sp19_density(SP19_DENS_CSV)
    obs_age_depth, obs_age_yr = load_sp19_age(SP19_AGE_CSV)

    # Load Buizert at 1yr resolution for flexible resampling
    buiz_T_df = pd.read_csv(BUIZERT_TEMP_CSV).sort_values("year_CE")
    buiz_A_df = pd.read_csv(BUIZERT_ACCUM_CSV).sort_values("year_CE")
    spin_years_1yr = np.arange(float(SPINUP_START_CE), float(ERA5_START_YEAR), 1.0)
    spin_T_1yr = np.interp(spin_years_1yr, buiz_T_df["year_CE"].values,
                           buiz_T_df["temp"].values + 273.15)
    spin_A_1yr = np.interp(spin_years_1yr, buiz_A_df["year_CE"].values,
                           buiz_A_df["accum"].values)

    print(f"  Buizert 1yr: {len(spin_T_1yr)} points, "
          f"T=[{spin_T_1yr.min()-273.15:.1f}, {spin_T_1yr.max()-273.15:.1f}]°C")
    print(f"  ERA5: {len(era5_T_phase)} years")
    print(f"  SP19 obs: {len(obs_rho_depth)} rho, {len(obs_age_depth)} age")

    # --- Run all cases ---
    results = []
    for label, nz, stretch_p, spinup_dt, era5_dt in CASES:
        try:
            res = run_case(label, nz, stretch_p, spinup_dt, era5_dt,
                           spin_T_1yr, spin_A_1yr, era5_T_phase, era5_A_phase)
            results.append(res)
        except Exception as e:
            print(f"  FAILED: {e}")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Density
    ax = axes[0]
    ax.plot(obs_rho_val, obs_rho_depth, "k.", ms=2, alpha=0.5, label="SP19", zorder=10)
    for res, c in zip(results, colors):
        ax.plot(res["rho"], res["depth"], "-", color=c, lw=1.2, label=res["label"])
    ax.set_xlabel("Density (kg/m³)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title("Density")
    ax.legend(fontsize=6, loc="lower left")

    # Age
    ax = axes[1]
    ax.plot(obs_age_yr, obs_age_depth, "k.", ms=2, alpha=0.5, label="SP19", zorder=10)
    for res, c in zip(results, colors):
        ax.plot(res["age_yr"], res["depth"], "-", color=c, lw=1.2, label=res["label"])
    ax.set_xlabel("Age (yr)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title("Age")
    ax.legend(fontsize=6, loc="lower right")

    # Temperature
    ax = axes[2]
    for res, c in zip(results, colors):
        ax.plot(res["T_C"], res["depth"], "-", color=c, lw=1.2, label=res["label"])
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title("Temperature")
    ax.legend(fontsize=6)

    fig.suptitle("South Pole spinup resolution sensitivity\n"
                 f"Buizert {SPINUP_START_CE}-{ERA5_START_YEAR} + ERA5 {ERA5_START_YEAR}-{CORE_YEAR}")
    fig.tight_layout()
    out_path = OUT_DIR / "spinup_resolution_sweep.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

    # Print timing summary
    print("\nTiming summary:")
    for res in results:
        print(f"  {res['label']:25s}  {res['time_s']:.0f}s  "
              f"rho=[{res['rho'].min():.0f},{res['rho'].max():.0f}]  "
              f"age_max={res['age_yr'].max():.0f}yr")

    print("\nDone.")
