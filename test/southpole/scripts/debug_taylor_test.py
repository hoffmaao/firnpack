"""debug_taylor_test.py

Systematic diagnosis of the order-1 Taylor remainder in the firn inversion.

Tests each component of the objective independently:
  1. Short spinup (10 yr) vs full (1200 yr) — is it tape length?
  2. J_rho alone vs J_age alone — which component breaks?
  3. Single control at a time — which parameter breaks?
  4. Simplified dR/dt expression — does the new formula cause issues?

Run:
    cd test/southpole/scripts
    OMP_NUM_THREADS=1 python debug_taylor_test.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import firedrake as fd
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    pause_annotation,
    stop_annotating,
    get_working_tape,
)

from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.constants import year as YEAR_S

# =============================================================================
# Paths
# =============================================================================
_HERE = Path(__file__).parent
_SOUTHPOLE = _HERE.parent
PROCESSED_DIR = _SOUTHPOLE / "processed"
ERA5_CSV = PROCESSED_DIR / "era5_monthly_point.csv"
SP19_DENS_CSV = PROCESSED_DIR / "sp19_density.csv"
SP19_AGE_CSV = PROCESSED_DIR / "sp19_depth_age.csv"
BUIZERT_TEMP_CSV = PROCESSED_DIR / "buizert2021_spice_temp.csv"
BUIZERT_ACCUM_CSV = PROCESSED_DIR / "buizert2021_spice_accum.csv"

# =============================================================================
# Minimal settings
# =============================================================================
H0 = 130.0
NZ = 100
STRETCH_P = 2.5
SURFACE_ID = 2
ERA5_START_YEAR = 1940
CORE_YEAR = 2015
DT_YEARS = 1.0
ERA5_T_OFFSET_K = -4.71
ERA5_A_OFFSET_M_YR = +0.0163

P0 = FirnParameters()
RHO_M = float(getattr(P0, "rho_m", 550.0))
RHO_SMOOTH = 20.0
OBS_KERNEL_SIGMA_M = 0.5


def load_era5():
    df = pd.read_csv(ERA5_CSV)
    t_rows = df[df["t2m_K"].notna()][["year", "t2m_K"]]
    a_rows = df[df["net_accum_m_iceeq_month"].notna()][["year", "net_accum_m_iceeq_month"]]
    T_by_yr = t_rows.groupby("year")["t2m_K"].mean()
    A_by_yr = a_rows.groupby("year")["net_accum_m_iceeq_month"].sum()
    years = np.array(sorted(T_by_yr.index.intersection(A_by_yr.index)), dtype=int)
    return years, T_by_yr.loc[years].values, A_by_yr.loc[years].values


def load_buizert(spinup_start_ce):
    df_T = pd.read_csv(BUIZERT_TEMP_CSV).sort_values("year_CE")
    df_A = pd.read_csv(BUIZERT_ACCUM_CSV).sort_values("year_CE")
    years = np.arange(float(spinup_start_ce), float(ERA5_START_YEAR), 1.0)
    T = np.interp(years, df_T["year_CE"].values, df_T["temp"].values + 273.15)
    A = np.interp(years, df_A["year_CE"].values, df_A["accum"].values)
    return years, T, A


def load_sp19_density(depth_max=H0):
    df = pd.read_csv(SP19_DENS_CSV)
    df = df[df["depth_m"] <= depth_max]
    depth = df["depth_m"].values.astype(float)
    rho = df["rho_kgm3"].values.astype(float) * 1000.0
    idx = np.argsort(depth)
    return depth[idx], rho[idx]


def load_sp19_age(depth_max=H0):
    df = pd.read_csv(SP19_AGE_CSV)
    mask = (df["year_CE"] <= CORE_YEAR) & (df["depth_m"] <= depth_max)
    df = df[mask]
    depth = df["depth_m"].values.astype(float)
    age_s = (float(CORE_YEAR) - df["year_CE"].values.astype(float)) * float(YEAR_S)
    idx = np.argsort(depth)
    return depth[idx], age_s[idx]


def make_real(R, value, name):
    f = fd.Function(R, name=name)
    f.dat.data[:] = float(value)
    return f


def safe_vertex_coords(H0, depths):
    d = np.clip(np.asarray(depths, dtype=float), 0.0, H0)
    x = H0 - d
    eps = 1.0e-8 * max(H0, 1.0)
    x = np.where(x <= 0.0, eps, x)
    x = np.where(x >= H0, H0 - eps, x)
    return np.vstack([x]).T


def taylor_test_single(rf, controls, name="", seed=1, epsilons=None):
    """Run Taylor test and return the convergence orders."""
    if epsilons is None:
        epsilons = [1e-2, 5e-3, 2.5e-3, 1.25e-3]

    with stop_annotating():
        m0 = np.array([float(c.dat.data_ro[0]) for c in controls])
        for c, v in zip(controls, m0):
            c.assign(float(v))

    with stop_annotating():
        J0 = float(rf(controls))
        g_list = rf.derivative()

    with stop_annotating():
        rng = np.random.default_rng(seed)
        d = rng.standard_normal(len(m0))
        d = d / (np.linalg.norm(d) + 1e-30) * 1e-2 * (np.abs(m0) + 1.0)

        dx_ctrl = fd.dx(domain=controls[0].function_space().mesh())
        gdotd = 0.0
        for gf, dv in zip(g_list, d):
            tmp = fd.Function(gf.function_space())
            tmp.assign(float(dv))
            gdotd += float(fd.assemble(gf * tmp * dx_ctrl))

    orders = []
    prev_r, prev_eps = None, None
    for eps in epsilons:
        m1 = m0 + eps * d
        with stop_annotating():
            for c, v in zip(controls, m1):
                c.assign(float(v))
            J1 = float(rf(controls))
        r1 = abs(J1 - J0 - eps * gdotd)
        if prev_r is not None and r1 > 0 and prev_r > 0:
            order = math.log(prev_r / r1) / math.log(prev_eps / eps)
            orders.append(order)
        prev_r, prev_eps = r1, eps

    with stop_annotating():
        for c, v in zip(controls, m0):
            c.assign(float(v))

    mean_order = float(np.mean(orders)) if orders else 0.0
    status = "OK" if mean_order > 1.8 else "BROKEN"
    print(f"  {name:30s}  mean order = {mean_order:.2f}  [{status}]  orders={[f'{o:.2f}' for o in orders]}")
    return mean_order


def run_test(spinup_years, n_era5_steps, n_rho_obs, n_age_obs,
             controls_to_use, test_name):
    """Build a minimal forward model and run Taylor test."""

    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"  spinup={spinup_years}yr, era5={n_era5_steps}yr, "
          f"rho_obs={n_rho_obs}, age_obs={n_age_obs}, "
          f"controls={controls_to_use}")
    print(f"{'='*60}")

    # --- Data ---
    era5_years, era5_T, era5_A = load_era5()
    run_mask = (era5_years >= ERA5_START_YEAR) & (era5_years < CORE_YEAR)
    run_T = (era5_T[run_mask] + ERA5_T_OFFSET_K)[:n_era5_steps]
    run_A = (era5_A[run_mask] + ERA5_A_OFFSET_M_YR)[:n_era5_steps]

    spinup_start_ce = ERA5_START_YEAR - spinup_years
    spin_years, spin_T, spin_A = load_buizert(spinup_start_ce)

    obs_rho_depth, obs_rho_val = load_sp19_density()
    obs_age_depth, obs_age_s = load_sp19_age()

    # Subsample observations
    if n_rho_obs > 0 and n_rho_obs < len(obs_rho_depth):
        idx = np.linspace(0, len(obs_rho_depth)-1, n_rho_obs, dtype=int)
        obs_rho_depth = obs_rho_depth[idx]
        obs_rho_val = obs_rho_val[idx]
    if n_age_obs > 0 and n_age_obs < len(obs_age_depth):
        idx = np.linspace(0, len(obs_age_depth)-1, n_age_obs, dtype=int)
        obs_age_depth = obs_age_depth[idx]
        obs_age_s = obs_age_s[idx]

    sigma_rho = np.full_like(obs_rho_val, 30.0)
    sigma_age = np.full_like(obs_age_s, 20.0 * float(YEAR_S))

    # --- Mesh ---
    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
    coord_fs = mesh.coordinates.function_space()
    mesh.coordinates.assign(fd.Function(coord_fs).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)
    dx_domain = fd.dx(domain=mesh)

    # --- Density IC ---
    ic_depth_raw, ic_rho_raw = load_sp19_density(depth_max=2.0 * H0)
    ic_depth = np.concatenate([[0.0], ic_depth_raw])
    ic_rho = np.concatenate([[350.0], ic_rho_raw])
    mesh_x = (mesh.coordinates.dat.data_ro[:, 0]
              if mesh.coordinates.dat.data_ro.ndim == 2
              else mesh.coordinates.dat.data_ro)
    mesh_depth = H0 - mesh_x
    rho_at_nodes = np.interp(mesh_depth, ic_depth, ic_rho, left=350.0, right=ic_rho[-1])
    rho_at_nodes = np.clip(rho_at_nodes, 200.0, float(P0.rho_i) - 0.5)

    # --- Kernels ---
    _x, = fd.SpatialCoordinate(mesh)

    def _make_kernels(depths, sigma_m, tag):
        xy = safe_vertex_coords(H0, depths)
        kernels = []
        for j, xp in enumerate([float(xy_j[0]) for xy_j in xy]):
            wgt = fd.exp(-0.5 * ((_x - xp) / float(sigma_m)) ** 2)
            denom = float(fd.assemble(wgt * dx_domain)) + 1e-30
            phi = fd.Function(V, name=f"phi_{tag}_{j:04d}")
            phi.interpolate(wgt / denom)
            kernels.append(phi)
        return kernels

    # --- Tape ---
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()

    # Controls
    log_kc0 = make_real(R, math.log(9.2e-9), "log_kc0")
    log_kc1 = make_real(R, math.log(3.7e-9), "log_kc1")
    log_kg = make_real(R, math.log(1.3e-7), "log_kg")
    log_rho_s = make_real(R, math.log(350.0), "log_rho_s")
    log_r2_s = make_real(R, math.log(1.0e-9), "log_r2_s")

    all_controls = {
        "kc0": log_kc0,
        "kc1": log_kc1,
        "kg": log_kg,
        "rho_surf": log_rho_s,
        "r2_surf": log_r2_s,
    }

    kc0_expr = fd.exp(log_kc0)
    kc1_expr = fd.exp(log_kc1)
    kg_expr = fd.exp(log_kg)
    rho_surf_expr = fd.exp(log_rho_s)
    r2_surf_expr = fd.exp(log_r2_s)

    params = FirnParameters(kg=kg_expr)
    model = FirnModel(params)
    solver = FirnColumnSolver(model)

    Ts = make_real(R, run_T[0] if len(run_T) > 0 else 220.0, "Ts")
    accum = make_real(R, run_A[0] if len(run_A) > 0 else 0.08, "accum")
    Hs = make_real(R, float(params.c_i) * ((run_T[0] if len(run_T) > 0 else 220.0) - float(params.T_ref)), "Hs")
    dt_val = make_real(R, float(DT_YEARS * YEAR_S), "dt")

    rho_surf_fs = make_real(R, 350.0, "rho_surf_fs")
    rho_surf_fs.interpolate(rho_surf_expr)

    def w_surf_value(acc, rho_s):
        return -acc * params.rho_i / rho_s / float(YEAR_S)

    bc_H = fd.DirichletBC(V, Hs, SURFACE_ID)
    rho_surf_bc = fd.Function(V, name="rho_surf_bc")
    rho_surf_bc.interpolate(rho_surf_expr)
    bc_rho = fd.DirichletBC(V, rho_surf_bc, SURFACE_ID)
    w_surf_bc = fd.Function(V, name="w_surf_bc")
    w_surf_bc.interpolate(w_surf_value(accum, rho_surf_expr))
    bc_w = fd.DirichletBC(V, w_surf_bc, SURFACE_ID)
    bcs = [bc_H, bc_rho, bc_w]

    bc_sigma = fd.DirichletBC(V, make_real(R, 0.0, "sig0"), SURFACE_ID)
    r2_surf_bc = fd.Function(V, name="r2_surf_bc")
    r2_surf_bc.interpolate(r2_surf_expr)
    bc_r2 = fd.DirichletBC(V, r2_surf_bc, SURFACE_ID)
    bc_age = fd.DirichletBC(V, make_real(R, 0.0, "age0"), SURFACE_ID)
    rhoCoef = fd.Function(V, name="rhoCoef")

    # State
    H_fn = fd.Function(V, name="H")
    rho = fd.Function(V, name="rho")
    w = fd.Function(V, name="w")
    sigma_fn = fd.Function(V, name="sigma")
    r2 = fd.Function(V, name="r2")
    age = fd.Function(V, name="age")

    rho_ic = fd.Function(V, name="rho_ic")
    rho_ic.dat.data[:] = rho_at_nodes

    T_spin_mean = float(np.mean(spin_T)) if len(spin_T) > 0 else 220.0
    dr2dt = float(1.3e-7) * np.exp(-float(P0.Eg) / (float(P0.R) * T_spin_mean))
    w_surf_mean = float(np.mean(spin_A) if len(spin_A) > 0 else 0.08) * float(P0.rho_i) / 350.0 / float(YEAR_S)
    r2_ic_nodes = 1.0e-9 + (dr2dt / abs(w_surf_mean)) * mesh_depth
    r2_ic_nodes = np.clip(r2_ic_nodes, 1.0e-9, 2.5e-7)
    r2_ic = fd.Function(V, name="r2_ic")
    r2_ic.dat.data[:] = r2_ic_nodes

    with stop_annotating():
        domain_len = float(fd.assemble(fd.Constant(1.0) * dx_domain)) + 1e-30
    _prior_fn = fd.Function(V, name="prior_scratch")

    # Build kernels
    with stop_annotating():
        rho_kernels = _make_kernels(obs_rho_depth[:n_rho_obs], OBS_KERNEL_SIGMA_M, "rho") if n_rho_obs > 0 else []
        age_kernels = _make_kernels(obs_age_depth[:n_age_obs], OBS_KERNEL_SIGMA_M, "age") if n_age_obs > 0 else []

    rho_obs_vals = [float(v) for v in obs_rho_val[:n_rho_obs]]
    rho_sig_vals = [float(v) for v in sigma_rho[:n_rho_obs]]
    age_obs_vals = [float(v) for v in obs_age_s[:n_age_obs]]
    age_sig_vals = [float(v) for v in sigma_age[:n_age_obs]]

    # --- Forward mapping ---
    def forward_mapping():
        rho_surf_fs.interpolate(rho_surf_expr)
        rho_surf_bc.interpolate(rho_surf_expr)
        r2_surf_bc.interpolate(r2_surf_expr)

        rho.assign(rho_ic)
        H_fn.assign(float(params.c_i) * (T_spin_mean - float(params.T_ref)))
        sigma_fn.assign(0.0)
        r2.assign(r2_ic)
        w.assign(0.0)
        age.assign(0.0)

        # Spinup
        for k in range(spinup_years):
            T_sk = float(spin_T[k]) if k < len(spin_T) else T_spin_mean
            A_sk = float(spin_A[k]) if k < len(spin_A) else 0.08
            Ts.assign(T_sk)
            Hs.assign(float(params.c_i) * (T_sk - float(params.T_ref)))
            accum.assign(A_sk)
            w_surf_bc.interpolate(w_surf_value(accum, rho_surf_expr))
            s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
            rhoCoef.interpolate((1.0 - s) * kc0_expr + s * kc1_expr)
            solver.prognostic_solve(
                enthalpy=H_fn, density=rho, firn_velocity=w, dt=dt_val,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                stress=sigma_fn, grain_radius2=r2,
                stress_boundary_condition=bc_sigma,
                grain_radius2_boundary_condition=bc_r2,
                rhoCoef=rhoCoef,
                age=age, age_boundary_condition=bc_age,
            )

        # ERA5
        for k in range(n_era5_steps):
            T_k = float(run_T[k])
            A_k = float(run_A[k])
            Ts.assign(T_k)
            Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)
            w_surf_bc.interpolate(w_surf_value(accum, rho_surf_expr))
            s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
            rhoCoef.interpolate((1.0 - s) * kc0_expr + s * kc1_expr)
            solver.prognostic_solve(
                enthalpy=H_fn, density=rho, firn_velocity=w, dt=dt_val,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                stress=sigma_fn, grain_radius2=r2,
                stress_boundary_condition=bc_sigma,
                grain_radius2_boundary_condition=bc_r2,
                rhoCoef=rhoCoef,
                age=age, age_boundary_condition=bc_age,
            )

        # Objective
        J = 0.0

        if n_rho_obs > 0:
            for phi, obs_c, sig_c in zip(rho_kernels, rho_obs_vals, rho_sig_vals):
                pred = fd.assemble(rho * phi * dx_domain)
                r = (pred - obs_c) / sig_c
                J += 0.5 * r * r

        if n_age_obs > 0:
            for phi, obs_c, sig_c in zip(age_kernels, age_obs_vals, age_sig_vals):
                pred = fd.assemble(age * phi * dx_domain)
                r = (pred - obs_c) / sig_c
                J += 0.5 * r * r

        return J

    J = forward_mapping()
    print(f"  J = {float(J):.6e}")

    # Select controls
    controls = [all_controls[c] for c in controls_to_use]
    rf = ReducedFunctional(J, [Control(c) for c in controls])
    pause_annotation()

    # Taylor test
    taylor_test_single(rf, controls, name=test_name)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Taylor remainder diagnostic")
    print("=" * 60)

    # Test 1: Very short spinup, density only, single control
    run_test(spinup_years=5, n_era5_steps=5, n_rho_obs=10, n_age_obs=0,
             controls_to_use=["kc0"],
             test_name="5yr_rho_kc0_only")

    # Test 2: Short spinup, density only, all controls
    run_test(spinup_years=5, n_era5_steps=5, n_rho_obs=10, n_age_obs=0,
             controls_to_use=["kc0", "kc1", "kg", "rho_surf", "r2_surf"],
             test_name="5yr_rho_all_ctrl")

    # Test 3: Short spinup, age only, single control
    run_test(spinup_years=10, n_era5_steps=5, n_rho_obs=0, n_age_obs=10,
             controls_to_use=["kc0"],
             test_name="10yr_age_kc0_only")

    # Test 4: Short spinup, both density + age
    run_test(spinup_years=10, n_era5_steps=5, n_rho_obs=10, n_age_obs=10,
             controls_to_use=["kc0", "kc1", "kg"],
             test_name="10yr_rho_age_3ctrl")

    # Test 5: Medium spinup (100 yr), density only
    run_test(spinup_years=100, n_era5_steps=10, n_rho_obs=10, n_age_obs=0,
             controls_to_use=["kc0", "kc1"],
             test_name="100yr_rho_kc0_kc1")

    # Test 6: Medium spinup (100 yr), both
    run_test(spinup_years=100, n_era5_steps=10, n_rho_obs=10, n_age_obs=10,
             controls_to_use=["kc0", "kc1", "kg"],
             test_name="100yr_rho_age_3ctrl")

    # Test 7: Full spinup, density only (is tape length the issue?)
    run_test(spinup_years=1200, n_era5_steps=75, n_rho_obs=10, n_age_obs=0,
             controls_to_use=["kc0"],
             test_name="1200yr_rho_kc0_only")

    print("\n" + "=" * 60)
    print("Summary: if short-spinup tests pass but long-spinup fails,")
    print("the issue is numerical accumulation over the long tape.")
    print("If all tests fail, the issue is in the weak form / expression.")
    print("=" * 60)
