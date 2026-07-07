"""forward_hl_freesurface.py

Validate the free-surface-on-static-mesh approach against ALE mesh motion.
Runs the same H&L forward model three ways:
  1. Fixed mesh (Eulerian, no surface tracking)
  2. ALE mesh motion (um-fdm style)
  3. Free surface on a static oversized mesh (masked physics)

Usage:
    OMP_NUM_THREADS=1 python forward_hl_freesurface.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import firedrake as fd

from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.physics.densification import herron_langway
from firn.constants import year as YEAR_S

_HERE = Path(__file__).parent
_SOUTHPOLE = _HERE.parent.parent
PROCESSED_DIR = _SOUTHPOLE / "processed"
OUT_DIR = _SOUTHPOLE / "results"

H0 = 130.0
H_MAX = 200.0   # oversized domain for free surface
NZ = 100
STRETCH_P = 2.5
SURFACE_ID = 2
BASE_ID = 1
RHO_SURF = 350.0
DT_YEARS = 10.0


def build_mesh(height, nz, stretch):
    mesh = fd.IntervalMesh(nz, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = height * (1.0 - (1.0 - xi) ** stretch)
    coord_fs = mesh.coordinates.function_space()
    mesh.coordinates.assign(fd.Function(coord_fs).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    return mesh, V


def make_real(R, val, name):
    f = fd.Function(R, name=name); f.assign(float(val)); return f


def run_free_surface(label, spin_T, spin_A):
    """Run H&L on an oversized static mesh with free surface tracking."""
    t0 = time.time()
    print(f"\n--- {label} ---")

    mesh, V = build_mesh(H_MAX, NZ, STRETCH_P)
    R = fd.FunctionSpace(mesh, "R", 0)
    dx = fd.dx(domain=mesh)

    params = FirnParameters()
    model = FirnModel(params, densification_rate_fn=herron_langway)
    solver = FirnColumnSolver(model, surface_id=SURFACE_ID)

    H = fd.Function(V); rho = fd.Function(V); w = fd.Function(V)
    age = fd.Function(V)

    T0, A0 = float(spin_T[0]), float(spin_A[0])
    xi = fd.SpatialCoordinate(mesh)[0]

    # Surface height as a tracked Constant
    h = fd.Constant(H0)
    mask_width = fd.Constant(float(params.mask_width))
    chi = fd.Function(V, name="chi")
    chi.interpolate(model.surface_mask(xi, h, mask_width))

    # IC: exponential density profile below h, rho_surf above
    H.assign(float(params.c_i) * (T0 - float(params.T_ref)))
    rho.interpolate(
        fd.conditional(
            fd.lt(xi, h),
            RHO_SURF + (float(params.rho_i) - RHO_SURF) * (1.0 - fd.exp(-(h - xi) / 20.0)),
            fd.Constant(RHO_SURF),
        )
    )
    w.assign(-A0 * float(params.rho_i) / RHO_SURF / float(YEAR_S))
    age.assign(0.0)

    dt_fn = make_real(R, DT_YEARS * YEAR_S, "dt")
    Ts = make_real(R, T0, "Ts")
    Hs = make_real(R, float(params.c_i) * (T0 - float(params.T_ref)), "Hs")
    accum = make_real(R, A0, "accum")
    rho_s = make_real(R, RHO_SURF, "rho_s")

    # Base velocity BC (w=0 at base for oversized domain)
    # The mass-flux approach: w_base balances accumulation in ice-eq
    w_base = make_real(R, -A0 / float(YEAR_S), "w_base")
    bc_H = fd.DirichletBC(V, Hs, SURFACE_ID)
    bc_rho = fd.DirichletBC(V, rho_s, SURFACE_ID)
    bc_w = fd.DirichletBC(V, w_base, BASE_ID)
    bc_age = fd.DirichletBC(V, make_real(R, 0.0, "a0"), SURFACE_ID)

    surface_heights = [H0]

    for step in range(len(spin_T)):
        Tk, Ak = float(spin_T[step]), float(spin_A[step])
        Ts.assign(Tk)
        Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
        accum.assign(Ak)
        w_base.assign(-Ak / float(YEAR_S))

        # Update mask from current h
        chi.interpolate(model.surface_mask(xi, h, mask_width))

        solver.prognostic_solve(
            enthalpy=H, density=rho, firn_velocity=w, dt=dt_fn,
            accumulation=accum, surface_density=rho_s,
            boundary_conditions=[bc_H, bc_rho, bc_w],
            surface_temperature=Ts, enthalpy_bc_constant=Hs,
            age=age, age_boundary_condition=bc_age,
        )

        # Compute drhodt at current state (for h update)
        T_field = model.temperature_from_enthalpy(H)
        bdot = fd.Constant(float(Ak) * float(params.rho_i) / float(params.spy))
        if model.densification_rate_fn is not None:
            drhodt_expr = model.densification_rate_fn(rho, T_field, params=params, bdot=bdot)
        else:
            drhodt_expr = model.densification_rate_arthern(rho, T_field, bdot)

        # Masked densification for h tendency
        drhodt_masked = drhodt_expr * chi

        # Update surface height: dh/dt = (a_snow - ∫ drhodt·χ dx) / ρ_surf
        dhdt = model.surface_tendency(drhodt_masked, accum, RHO_SURF, dx)
        h.assign(float(h) + float(dt_fn) * dhdt)
        surface_heights.append(float(h))

    t_total = time.time() - t0
    h_final = float(h)

    # Extract profiles: depth from PHYSICAL surface h
    mesh_x = mesh.coordinates.dat.data_ro.flatten()
    depth = h_final - mesh_x
    idx = np.argsort(depth)
    # Only keep points below the surface (depth >= 0)
    mask = depth[idx] >= 0
    d = depth[idx][mask]
    r = rho.dat.data_ro[idx][mask]
    a = age.dat.data_ro[idx][mask] / float(YEAR_S)

    print(f"  Time: {t_total:.0f}s")
    print(f"  Surface: {H0:.1f} → {h_final:.1f}m (Δ={h_final-H0:.1f}m)")
    print(f"  rho (below surface): [{r.min():.0f}, {r.max():.0f}] kg/m³")
    print(f"  age: [{a.min():.0f}, {a.max():.0f}] yr")

    return {
        "label": label, "depth": d, "rho": r, "age_yr": a,
        "h_history": np.array(surface_heights), "h_final": h_final,
    }


def run_ale(label, spin_T, spin_A):
    """Run H&L with ALE mesh motion (reference implementation)."""
    t0 = time.time()
    print(f"\n--- {label} ---")

    mesh, V = build_mesh(H0, NZ, STRETCH_P)
    R = fd.FunctionSpace(mesh, "R", 0)
    params = FirnParameters()
    model = FirnModel(params, densification_rate_fn=herron_langway)
    solver = FirnColumnSolver(model, surface_id=SURFACE_ID)

    H = fd.Function(V); rho = fd.Function(V); w = fd.Function(V)
    age = fd.Function(V)
    T0, A0 = float(spin_T[0]), float(spin_A[0])
    xi = fd.SpatialCoordinate(mesh)[0]
    H.assign(float(params.c_i) * (T0 - float(params.T_ref)))
    rho.interpolate(RHO_SURF + (float(params.rho_i) - RHO_SURF) * (1 - fd.exp(-(H0 - xi) / 20)))
    w.assign(-A0 * float(params.rho_i) / RHO_SURF / float(YEAR_S))
    age.assign(0.0)

    dt_fn = make_real(R, DT_YEARS * YEAR_S, "dt")
    Ts = make_real(R, T0, "Ts")
    Hs = make_real(R, float(params.c_i) * (T0 - float(params.T_ref)), "Hs")
    accum = make_real(R, A0, "accum")
    rho_s = make_real(R, RHO_SURF, "rho_s")
    ws = make_real(R, -A0 * float(params.rho_i) / RHO_SURF / float(YEAR_S), "ws")
    bc_H = fd.DirichletBC(V, Hs, SURFACE_ID)
    bc_rho = fd.DirichletBC(V, rho_s, SURFACE_ID)
    bc_w = fd.DirichletBC(V, ws, SURFACE_ID)
    bc_age = fd.DirichletBC(V, make_real(R, 0.0, "a0"), SURFACE_ID)

    coords = mesh.coordinates.dat.data_ro.copy().flatten()
    idx_s = np.argsort(coords); nc = len(idx_s) - 1
    l_ini = np.diff(coords[idx_s])
    ra = rho.dat.data_ro
    rho_ini = np.array([.5 * (ra[idx_s[i]] + ra[idx_s[i+1]]) for i in range(nc)])

    surface_heights = [H0]
    for step in range(len(spin_T)):
        Tk, Ak = float(spin_T[step]), float(spin_A[step])
        Ts.assign(Tk); Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
        accum.assign(Ak); ws.assign(-Ak * float(params.rho_i) / RHO_SURF / float(YEAR_S))
        solver.prognostic_solve(
            enthalpy=H, density=rho, firn_velocity=w, dt=dt_fn,
            accumulation=accum, surface_density=rho_s,
            boundary_conditions=[bc_H, bc_rho, bc_w],
            surface_temperature=Ts, enthalpy_bc_constant=Hs,
            age=age, age_boundary_condition=bc_age,
        )
        # ALE update
        ra = rho.dat.data_ro
        rc = np.array([.5 * (ra[idx_s[i]] + ra[idx_s[i+1]]) for i in range(nc)])
        ln = l_ini * rho_ini / np.maximum(rc, 1)
        xn = np.zeros(len(idx_s))
        for i in range(nc): xn[i+1] = xn[i] + ln[i]
        xr = np.zeros_like(coords)
        for rank, oi in enumerate(idx_s): xr[oi] = xn[rank]
        cd = mesh.coordinates.dat.data
        if cd.ndim == 2: cd[:, 0] = xr
        else: cd[:] = xr
        surface_heights.append(float(mesh.coordinates.dat.data_ro.max()))

    t_total = time.time() - t0
    h_final = float(mesh.coordinates.dat.data_ro.max())
    mx = mesh.coordinates.dat.data_ro.flatten()
    depth = h_final - mx
    idx = np.argsort(depth)
    d, r, a = depth[idx], rho.dat.data_ro[idx], age.dat.data_ro[idx] / float(YEAR_S)

    print(f"  Time: {t_total:.0f}s")
    print(f"  Surface: {H0:.1f} → {h_final:.1f}m (Δ={h_final-H0:.1f}m)")
    print(f"  rho: [{r.min():.0f}, {r.max():.0f}] kg/m³")
    print(f"  age: [{a.min():.0f}, {a.max():.0f}] yr")

    return {
        "label": label, "depth": d, "rho": r, "age_yr": a,
        "h_history": np.array(surface_heights), "h_final": h_final,
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    buiz_T = pd.read_csv(PROCESSED_DIR / "buizert2021_spice_temp.csv").sort_values("year_CE")
    buiz_A = pd.read_csv(PROCESSED_DIR / "buizert2021_spice_accum.csv").sort_values("year_CE")
    yr = np.arange(740, 1940, 1.0)
    sT1 = np.interp(yr, buiz_T["year_CE"].values, buiz_T["temp"].values + 273.15)
    sA1 = np.interp(yr, buiz_A["year_CE"].values, buiz_A["accum"].values)
    blk = int(DT_YEARS)
    sT = np.array([sT1[i*blk:(i+1)*blk].mean() for i in range(len(sT1)//blk)])
    sA = np.array([sA1[i*blk:(i+1)*blk].mean() for i in range(len(sA1)//blk)])
    print(f"Forcing: 740-1940 CE, {len(sT)} steps × dt={DT_YEARS}yr")

    df_rho = pd.read_csv(PROCESSED_DIR / "sp19_density.csv")
    df_rho = df_rho[df_rho["depth_m"] <= H0]
    obs_rho_d, obs_rho = df_rho["depth_m"].values, df_rho["rho_kgm3"].values * 1000

    res_ale = run_ale("ALE mesh motion", sT, sA)
    res_fs  = run_free_surface("Free surface (static mesh)", sT, sA)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    ax = axes[0]
    ax.plot(obs_rho, obs_rho_d, "k.", ms=2, alpha=0.4, label="SP19")
    ax.plot(res_ale["rho"], res_ale["depth"], "C0-", lw=2, label=res_ale["label"])
    ax.plot(res_fs["rho"], res_fs["depth"], "C1--", lw=2, label=res_fs["label"])
    ax.set_xlabel("Density (kg/m³)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_title("Density"); ax.legend(fontsize=8)
    ax.set_ylim(150, -5)

    ax = axes[1]
    ax.plot(res_ale["age_yr"], res_ale["depth"], "C0-", lw=2, label=res_ale["label"])
    ax.plot(res_fs["age_yr"], res_fs["depth"], "C1--", lw=2, label=res_fs["label"])
    ax.set_xlabel("Age (yr)"); ax.invert_yaxis()
    ax.set_title("Age"); ax.legend(fontsize=8)
    ax.set_ylim(150, -5)

    ax = axes[2]
    years = np.linspace(740, 1940, len(res_ale["h_history"]))
    ax.plot(years, res_ale["h_history"], "C0-", lw=2, label=res_ale["label"])
    years_fs = np.linspace(740, 1940, len(res_fs["h_history"]))
    ax.plot(years_fs, res_fs["h_history"], "C1--", lw=2, label=res_fs["label"])
    ax.set_xlabel("Year CE"); ax.set_ylabel("Surface height (m)")
    ax.set_title("Surface evolution"); ax.legend(fontsize=8)

    fig.suptitle("H&L: ALE vs Free Surface on Static Mesh")
    fig.tight_layout()
    out = OUT_DIR / "forward_hl_freesurface_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
