"""forward_hl_ale.py

Forward H&L model with ALE mesh motion following um-fdm's update_height().

After each timestep, mesh nodes are repositioned so that each cell conserves
mass:  l_new = l_ini * rho_ini / rho_current.  The surface position drops as
firn compacts, matching the physical process.

Compares: fixed mesh (Eulerian) vs ALE mesh motion.

Run:
    OMP_NUM_THREADS=1 python forward_hl_ale.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import firedrake as fd

from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.solvers.firn_solver import FirnColumnSolver
from firnpack.physics.densification import herron_langway
from firnpack.constants import year as YEAR_S

# Paths
_HERE = Path(__file__).parent
_SOUTHPOLE = _HERE.parent.parent  # herron_langway/ -> scripts/ -> southpole/
PROCESSED_DIR = _SOUTHPOLE / "processed"
OUT_DIR = _SOUTHPOLE / "results"

# Settings
H0 = 130.0
NZ = 100
STRETCH_P = 2.5
SURFACE_ID = 2
RHO_SURF = 350.0
DT_YEARS = 10.0
SPINUP_START_CE = 740
SPINUP_END_CE = 1940


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


def update_mesh_ale(mesh, rho, rho_ini, l_ini, H0):
    """ALE mesh update following um-fdm's update_height().

    Each cell conserves mass:  l_new[i] = l_ini[i] * rho_ini[i] / rho_current[i].
    Nodes are repositioned by summing from the base upward.

    Returns the new surface height.
    """
    coords = mesh.coordinates.dat.data
    x_old = coords.copy().flatten()
    idx = np.argsort(x_old)  # base to surface

    # Get cell-averaged density (midpoint of each element)
    rho_arr = rho.dat.data_ro
    # For CG1 on an interval mesh with n elements, we have n+1 nodes
    # Cell i has nodes idx[i] and idx[i+1]
    n_cells = len(idx) - 1
    rho_cell = np.zeros(n_cells)
    for i in range(n_cells):
        rho_cell[i] = 0.5 * (rho_arr[idx[i]] + rho_arr[idx[i + 1]])

    # New cell lengths: mass conservation
    l_new = l_ini * rho_ini / np.maximum(rho_cell, 1.0)

    # Rebuild node positions from base upward
    x_new = np.zeros(len(idx))
    x_new[0] = 0.0  # base stays at x=0
    for i in range(n_cells):
        x_new[i + 1] = x_new[i] + l_new[i]

    # Map back to original node ordering
    x_result = np.zeros_like(x_old)
    for rank, orig_idx in enumerate(idx):
        x_result[orig_idx] = x_new[rank]

    if coords.ndim == 2:
        coords[:, 0] = x_result
    else:
        coords[:] = x_result
    new_surface = x_new[-1]
    return new_surface


def run_forward(use_ale, label, spin_T, spin_A):
    t0 = time.time()
    print(f"\n--- {label} ---")

    mesh, V = build_stretched_mesh(H0, NZ, STRETCH_P)
    R = fd.FunctionSpace(mesh, "R", 0)

    params = FirnParameters()
    model = FirnModel(params, densification_rate_fn=herron_langway)
    solver = FirnColumnSolver(model, surface_id=SURFACE_ID)

    H = fd.Function(V, name="H")
    rho = fd.Function(V, name="rho")
    w = fd.Function(V, name="w")
    age = fd.Function(V, name="age")

    T0, A0 = float(spin_T[0]), float(spin_A[0])
    xi = fd.SpatialCoordinate(mesh)[0]
    depth = H0 - xi

    H.assign(float(params.c_i) * (T0 - float(params.T_ref)))
    rho.interpolate(RHO_SURF + (float(params.rho_i) - RHO_SURF) * (1.0 - fd.exp(-depth / 20.0)))
    w.assign(-A0 * float(params.rho_i) / RHO_SURF / float(YEAR_S))
    age.assign(0.0)

    dt_fn = make_real(R, DT_YEARS * YEAR_S, "dt")
    Ts = make_real(R, T0, "Ts")
    Hs = make_real(R, float(params.c_i) * (T0 - float(params.T_ref)), "Hs")
    accum = make_real(R, A0, "accum")
    rho_s_fn = make_real(R, RHO_SURF, "rho_s")
    w_surf = make_real(R, -A0 * float(params.rho_i) / RHO_SURF / float(YEAR_S), "ws")

    bc_H = fd.DirichletBC(V, Hs, SURFACE_ID)
    bc_rho = fd.DirichletBC(V, rho_s_fn, SURFACE_ID)
    bc_w = fd.DirichletBC(V, w_surf, SURFACE_ID)
    bc_age = fd.DirichletBC(V, make_real(R, 0.0, "a0"), SURFACE_ID)

    # Store initial cell lengths and densities for ALE
    if use_ale:
        coords_ini = mesh.coordinates.dat.data_ro.copy().flatten()
        idx_sort = np.argsort(coords_ini)
        n_cells = len(idx_sort) - 1
        l_ini = np.diff(coords_ini[idx_sort])
        rho_arr = rho.dat.data_ro
        rho_ini = np.array([0.5 * (rho_arr[idx_sort[i]] + rho_arr[idx_sort[i+1]])
                            for i in range(n_cells)])

    surface_heights = []

    for step in range(len(spin_T)):
        Tk, Ak = float(spin_T[step]), float(spin_A[step])
        Ts.assign(Tk)
        Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
        accum.assign(Ak)
        w_surf.assign(-Ak * float(params.rho_i) / RHO_SURF / float(YEAR_S))

        solver.prognostic_solve(
            enthalpy=H, density=rho, firn_velocity=w, dt=dt_fn,
            accumulation=accum, surface_density=rho_s_fn,
            boundary_conditions=[bc_H, bc_rho, bc_w],
            surface_temperature=Ts, enthalpy_bc_constant=Hs,
            age=age, age_boundary_condition=bc_age,
        )

        if use_ale:
            new_surf = update_mesh_ale(mesh, rho, rho_ini, l_ini, H0)
            surface_heights.append(new_surf)

    t_total = time.time() - t0

    # Extract profiles as depth from CURRENT surface
    mesh_x = mesh.coordinates.dat.data_ro.flatten()
    if use_ale:
        current_surface = mesh_x.max()
    else:
        current_surface = H0
    depth_nodes = current_surface - mesh_x
    idx = np.argsort(depth_nodes)

    print(f"  Time: {t_total:.0f}s")
    print(f"  rho: [{rho.dat.data_ro.min():.0f}, {rho.dat.data_ro.max():.0f}] kg/m³")
    print(f"  age: [{age.dat.data_ro.min()/YEAR_S:.0f}, {age.dat.data_ro.max()/YEAR_S:.0f}] yr")
    if use_ale:
        print(f"  Surface: {H0:.1f} → {current_surface:.1f} m "
              f"(Δ = {current_surface - H0:.1f} m)")

    return {
        "label": label,
        "depth": depth_nodes[idx],
        "rho": rho.dat.data_ro[idx].copy(),
        "age_yr": age.dat.data_ro[idx].copy() / float(YEAR_S),
        "w_m_yr": w.dat.data_ro[idx].copy() * float(YEAR_S),
        "T_C": H.dat.data_ro[idx] / float(params.c_i) + float(params.T_ref) - 273.15,
        "surface_heights": surface_heights if use_ale else [],
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load forcing
    buiz_T = pd.read_csv(PROCESSED_DIR / "buizert2021_spice_temp.csv").sort_values("year_CE")
    buiz_A = pd.read_csv(PROCESSED_DIR / "buizert2021_spice_accum.csv").sort_values("year_CE")
    years = np.arange(float(SPINUP_START_CE), float(SPINUP_END_CE), 1.0)
    T_1yr = np.interp(years, buiz_T["year_CE"].values, buiz_T["temp"].values + 273.15)
    A_1yr = np.interp(years, buiz_A["year_CE"].values, buiz_A["accum"].values)
    blk = int(DT_YEARS)
    n_blk = len(T_1yr) // blk
    sT = np.array([T_1yr[i*blk:(i+1)*blk].mean() for i in range(n_blk)])
    sA = np.array([A_1yr[i*blk:(i+1)*blk].mean() for i in range(n_blk)])
    print(f"Forcing: {SPINUP_START_CE}-{SPINUP_END_CE} CE, {len(sT)} steps × dt={DT_YEARS}yr")

    # Load SP19 observations
    df_rho = pd.read_csv(PROCESSED_DIR / "sp19_density.csv")
    df_rho = df_rho[df_rho["depth_m"] <= H0]
    obs_rho_d, obs_rho = df_rho["depth_m"].values, df_rho["rho_kgm3"].values * 1000.0
    df_age = pd.read_csv(PROCESSED_DIR / "sp19_depth_age.csv")
    df_age = df_age[(df_age["year_CE"] <= 2015) & (df_age["depth_m"] <= H0)]
    obs_age_d, obs_age = df_age["depth_m"].values, 2015.0 - df_age["year_CE"].values

    # Run both
    res_fixed = run_forward(False, "Fixed mesh (Eulerian)", sT, sA)
    res_ale   = run_forward(True,  "ALE mesh (um-fdm style)", sT, sA)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    ax = axes[0]
    ax.plot(obs_rho, obs_rho_d, "k.", ms=2, alpha=0.4, label="SP19")
    ax.plot(res_fixed["rho"], res_fixed["depth"], "C0-", lw=2, label=res_fixed["label"])
    ax.plot(res_ale["rho"], res_ale["depth"], "C1--", lw=2, label=res_ale["label"])
    ax.set_xlabel("Density (kg/m³)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title("Density")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(obs_age[::5], obs_age_d[::5], "k.", ms=2, alpha=0.4, label="SP19")
    ax.plot(res_fixed["age_yr"], res_fixed["depth"], "C0-", lw=2, label=res_fixed["label"])
    ax.plot(res_ale["age_yr"], res_ale["depth"], "C1--", lw=2, label=res_ale["label"])
    ax.set_xlabel("Age (yr)")
    ax.invert_yaxis()
    ax.set_title("Age")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(res_fixed["w_m_yr"], res_fixed["depth"], "C0-", lw=2, label=res_fixed["label"])
    ax.plot(res_ale["w_m_yr"], res_ale["depth"], "C1--", lw=2, label=res_ale["label"])
    ax.set_xlabel("Velocity (m/yr)")
    ax.invert_yaxis()
    ax.set_title("Velocity")
    ax.legend(fontsize=8)

    fig.suptitle(f"H&L: Fixed Eulerian vs ALE mesh motion\n"
                 f"({SPINUP_START_CE}-{SPINUP_END_CE} CE, dt={DT_YEARS}yr)")
    fig.tight_layout()
    out = OUT_DIR / "forward_hl_ale_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
