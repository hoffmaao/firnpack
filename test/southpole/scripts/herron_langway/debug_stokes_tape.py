"""Debug: check if tape replay matches direct forward for Stokes inversion."""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import firedrake as fd
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, pause_annotation,
    stop_annotating, get_working_tape,
)
from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.physics.densification import stokes_compressible
from firn.constants import year as YEAR_S

_HERE = Path(__file__).parent
_SP = _HERE.parent.parent
PROC = _SP / "processed"
OUT = _SP / "results"

H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF = 350.0
SPINUP_DT = 10.0
ERA5_DT = 0.5
P0 = FirnParameters()

INITIAL = dict(
    E_lin=5.0e9, Q_glen=60.0e3,
    c2_a=15.78652, c2_b=20.46489,
)
BOUNDS = dict(
    E_lin=(1.0e6, 1.0e13), Q_glen=(10.0e3, 200.0e3),
    c2_a=(5.0, 40.0), c2_b=(5.0, 40.0),
)
LOG = {"E_lin", "Q_glen"}


def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f


if __name__ == "__main__":
    # ---- Load forcing ----
    era5 = pd.read_csv(PROC / "era5_monthly_point.csv")
    tR = era5[era5["t2m_K"].notna()].groupby("year")["t2m_K"].mean()
    aR = era5[era5["net_accum_m_iceeq_month"].notna()].groupby("year")["net_accum_m_iceeq_month"].sum()
    yrs = sorted(tR.index.intersection(aR.index))
    m = (np.array(yrs) >= 1940) & (np.array(yrs) < 2015)
    eT = np.repeat(tR.loc[yrs].values[m], 2)
    eA = np.repeat((aR.loc[yrs].values[m] + 0.0163), 2)
    n_era5 = len(eT); dt_era5 = ERA5_DT * YEAR_S

    bT = pd.read_csv(PROC / "buizert2021_spice_temp.csv").sort_values("year_CE")
    bA = pd.read_csv(PROC / "buizert2021_spice_accum.csv").sort_values("year_CE")
    yr1 = np.arange(740, 1940, 1.0)
    sT1 = np.interp(yr1, bT["year_CE"].values, bT["temp"].values + 273.15)
    sA1 = np.interp(yr1, bA["year_CE"].values, bA["accum"].values)
    blk = int(SPINUP_DT)
    sT = np.array([sT1[i*blk:(i+1)*blk].mean() for i in range(len(sT1)//blk)])
    sA = np.array([sA1[i*blk:(i+1)*blk].mean() for i in range(len(sA1)//blk)])
    n_spin = len(sT); dt_spin = SPINUP_DT * YEAR_S

    # ---- Load observations ----
    df = pd.read_csv(PROC / "sp19_density.csv"); df = df[df["depth_m"] <= H0]
    obs_rho_d, obs_rho = df["depth_m"].values, df["rho_kgm3"].values * 1000
    sig_rho = np.full_like(obs_rho, np.median(15.0 + 0.03 * np.abs(obs_rho)))

    df = pd.read_csv(PROC / "sp19_depth_age.csv")
    df = df[(df["year_CE"] <= 2015) & (df["depth_m"] <= H0)]
    obs_age_d = df["depth_m"].values[::5]
    obs_age_yr = 2015.0 - df["year_CE"].values[::5]
    sig_age_yr = 1.0 + 0.005 * obs_age_yr

    N_rho, N_age = len(obs_rho_d), len(obs_age_d)

    # ---- Mesh ----
    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
    mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)
    dx = fd.dx(domain=mesh)

    # ---- VOMs ----
    eps_x = 1e-8 * H0
    def d2x(depths):
        x = H0 - np.clip(depths, 0, H0)
        return np.where(x <= 0, eps_x, np.where(x >= H0, H0 - eps_x, x))

    with stop_annotating():
        _prior = fd.Function(V, name="prior")
        dlen = float(fd.assemble(fd.Constant(1.0) * dx)) + 1e-30

        vom_rho = fd.VertexOnlyMesh(mesh, d2x(obs_rho_d).reshape(-1, 1))
        Pr = fd.FunctionSpace(vom_rho, "DG", 0)
        rho_obs_f = fd.Function(Pr); rho_obs_f.dat.data[:] = obs_rho
        rho_sig_f = fd.Function(Pr); rho_sig_f.dat.data[:] = sig_rho
        rho_pred_f = fd.Function(Pr)

        vom_age = fd.VertexOnlyMesh(mesh, d2x(obs_age_d).reshape(-1, 1))
        Pa = fd.FunctionSpace(vom_age, "DG", 0)
        age_obs_f = fd.Function(Pa); age_obs_f.dat.data[:] = obs_age_yr
        age_sig_f = fd.Function(Pa); age_sig_f.dat.data[:] = sig_age_yr
        age_pred_f = fd.Function(Pa)

    # ---- State ----
    H_f = fd.Function(V, name="H"); rho_f = fd.Function(V, name="rho")
    w_f = fd.Function(V, name="w"); age_f = fd.Function(V, name="age")
    depth = H0 - xi

    # ---- Controls ----
    log_E = mk(R, math.log(INITIAL["E_lin"]), "lE")
    log_Q = mk(R, math.log(INITIAL["Q_glen"]), "lQ")
    c2a = mk(R, INITIAL["c2_a"], "c2a")
    c2b = mk(R, INITIAL["c2_b"], "c2b")

    ctrls = [log_E, log_Q, c2a, c2b]
    names = ["E_lin", "Q_glen", "c2_a", "c2_b"]
    n_ctrl = len(ctrls)

    # ---- Model ----
    params = FirnParameters(
        stokes_n_glen=1.0,
        stokes_E_lin=fd.exp(log_E),
        stokes_Q_glen=fd.exp(log_Q),
        stokes_c2_a=c2a,
        stokes_c2_b=c2b,
    )
    model = FirnModel(params, densification_rate_fn=stokes_compressible)
    solver = FirnColumnSolver(model, surface_id=SID)

    Ts = mk(R, float(sT[0]), "Ts")
    Hs = mk(R, float(params.c_i) * (sT[0] - float(params.T_ref)), "Hs")
    ac = mk(R, float(sA[0]), "ac")
    rs = mk(R, RHO_SURF, "rs")
    ws = mk(R, -sA[0] * float(params.rho_i) / RHO_SURF / YEAR_S, "ws")
    dt = mk(R, dt_spin, "dt")

    bc_H = fd.DirichletBC(V, Hs, SID)
    bc_rho = fd.DirichletBC(V, rs, SID)
    bc_w = fd.DirichletBC(V, ws, SID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)

    def forward():
        T0, A0 = float(sT[0]), float(sA[0])
        H_f.assign(float(params.c_i) * (T0 - float(params.T_ref)))
        rho_f.interpolate(fd.Constant(RHO_SURF) +
            (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) *
            (1.0 - fd.exp(-depth / 20.0)))
        w_f.assign(-A0 * float(params.rho_i) / RHO_SURF / YEAR_S)
        age_f.assign(0.0)

        dt.assign(dt_spin)
        for k in range(n_spin):
            Tk, Ak = float(sT[k]), float(sA[k])
            Ts.assign(Tk); Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
            ac.assign(Ak); ws.assign(-Ak * float(params.rho_i) / RHO_SURF / YEAR_S)
            solver.prognostic_solve(
                enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt,
                accumulation=ac, surface_density=rs,
                boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age,
            )

        dt.assign(dt_era5)
        for k in range(n_era5):
            Tk, Ak = float(eT[k]), float(eA[k])
            Ts.assign(Tk); Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
            ac.assign(Ak); ws.assign(-Ak * float(params.rho_i) / RHO_SURF / YEAR_S)
            solver.prognostic_solve(
                enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt,
                accumulation=ac, surface_density=rs,
                boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age,
            )

        J = 0.0
        rho_pred_f.interpolate(rho_f)
        r = (rho_pred_f - rho_obs_f) / rho_sig_f
        J_rho = 0.5 / float(N_rho) * fd.assemble(r * r * fd.dx(domain=vom_rho))

        _age_yr = fd.Function(V, name="age_yr")
        _age_yr.interpolate(age_f / fd.Constant(YEAR_S))
        age_pred_f.interpolate(_age_yr)
        r = (age_pred_f - age_obs_f) / age_sig_f
        J_age = 0.5 / float(N_age) * fd.assemble(r * r * fd.dx(domain=vom_age))

        J_prior = 0.0
        _n_inv = 1.0 / n_ctrl
        for c, nm in zip(ctrls, names):
            if nm in LOG:
                _prior.interpolate((c - math.log(float(INITIAL[nm]))) / 0.5)
            else:
                _prior.interpolate((c - fd.Constant(float(INITIAL[nm]))) / fd.Constant(0.3))
            J_prior = J_prior + _n_inv * 0.5 * fd.assemble(_prior * _prior * dx) / dlen

        J = J_rho + J_age + J_prior
        return J, J_rho, J_age, J_prior

    # ---- Test 1: Direct forward ----
    tape = get_working_tape(); tape.clear_tape()
    continue_annotation()
    J_total, J_rho, J_age, J_prior = forward()
    pause_annotation()

    print("=" * 60)
    print("Direct forward() at initial params:")
    print(f"  J_total = {float(J_total):.4f}")
    print(f"  J_rho   = {float(J_rho):.4f}")
    print(f"  J_age   = {float(J_age):.4f}")
    print(f"  J_prior = {float(J_prior):.4f}")
    print(f"  rho: [{rho_f.dat.data_ro.min():.1f}, {rho_f.dat.data_ro.max():.1f}]")
    print(f"  age: [{age_f.dat.data_ro.min()/YEAR_S:.1f}, {age_f.dat.data_ro.max()/YEAR_S:.1f}] yr")
    rho_after_forward = rho_f.dat.data_ro.copy()

    # ---- Test 2: Tape replay at same controls ----
    # Build RF from the total J
    # Need to redo with a single J return
    tape.clear_tape()
    continue_annotation()

    # Reset controls to initial
    log_E.assign(math.log(INITIAL["E_lin"]))
    log_Q.assign(math.log(INITIAL["Q_glen"]))
    c2a.assign(INITIAL["c2_a"])
    c2b.assign(INITIAL["c2_b"])

    J_total2, J_rho2, J_age2, J_prior2 = forward()
    rf = ReducedFunctional(J_total2, [Control(c) for c in ctrls])
    pause_annotation()

    print(f"\nDirect forward() #2 (for tape):")
    print(f"  J_total = {float(J_total2):.4f}")
    print(f"  rho: [{rho_f.dat.data_ro.min():.1f}, {rho_f.dat.data_ro.max():.1f}]")

    # Now replay tape at same controls
    x0 = np.array([float(c.dat.data_ro[0]) for c in ctrls])
    with stop_annotating():
        for c, v in zip(ctrls, x0):
            c.assign(float(v))
        J_replay = float(rf(ctrls))

    print(f"\nTape replay rf(x0):")
    print(f"  J_replay = {J_replay:.4f}")
    print(f"  rho: [{rho_f.dat.data_ro.min():.1f}, {rho_f.dat.data_ro.max():.1f}]")
    rho_after_replay = rho_f.dat.data_ro.copy()

    # ---- Test 3: Second replay ----
    with stop_annotating():
        for c, v in zip(ctrls, x0):
            c.assign(float(v))
        J_replay2 = float(rf(ctrls))

    print(f"\nSecond tape replay rf(x0):")
    print(f"  J_replay2 = {J_replay2:.4f}")
    print(f"  rho: [{rho_f.dat.data_ro.min():.1f}, {rho_f.dat.data_ro.max():.1f}]")

    # ---- Compare ----
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  forward() #2:    J = {float(J_total2):.4f}")
    print(f"  rf(x0) replay 1: J = {J_replay:.4f}")
    print(f"  rf(x0) replay 2: J = {J_replay2:.4f}")
    print(f"  Match? {abs(float(J_total2) - J_replay) < 1.0}")
    if abs(float(J_total2) - J_replay) > 1.0:
        print(f"  *** MISMATCH: tape replay differs by {float(J_total2) - J_replay:.4f} ***")
        print(f"  rho max after forward: {rho_after_forward.max():.1f}")
        print(f"  rho max after replay:  {rho_after_replay.max():.1f}")
