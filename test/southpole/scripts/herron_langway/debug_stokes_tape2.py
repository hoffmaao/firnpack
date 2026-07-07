"""Debug: check age field replay specifically."""
from __future__ import annotations
import math
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

H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF = 350.0
SPINUP_DT = 10.0
ERA5_DT = 0.5

INITIAL = dict(E_lin=5.0e9, Q_glen=60.0e3, c2_a=15.78652, c2_b=20.46489)
LOG = {"E_lin", "Q_glen"}


def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f


if __name__ == "__main__":
    # Minimal forcing: just 10 spinup + 10 ERA5 steps for speed
    era5 = pd.read_csv(PROC / "era5_monthly_point.csv")
    tR = era5[era5["t2m_K"].notna()].groupby("year")["t2m_K"].mean()
    aR = era5[era5["net_accum_m_iceeq_month"].notna()].groupby("year")["net_accum_m_iceeq_month"].sum()
    yrs = sorted(tR.index.intersection(aR.index))
    m = (np.array(yrs) >= 1940) & (np.array(yrs) < 2015)
    eT = np.repeat(tR.loc[yrs].values[m], 2)[:10]
    eA = np.repeat((aR.loc[yrs].values[m] + 0.0163), 2)[:10]

    bT = pd.read_csv(PROC / "buizert2021_spice_temp.csv").sort_values("year_CE")
    bA = pd.read_csv(PROC / "buizert2021_spice_accum.csv").sort_values("year_CE")
    yr1 = np.arange(740, 1940, 1.0)
    sT1 = np.interp(yr1, bT["year_CE"].values, bT["temp"].values + 273.15)
    sA1 = np.interp(yr1, bA["year_CE"].values, bA["accum"].values)
    blk = int(SPINUP_DT)
    sT = np.array([sT1[i*blk:(i+1)*blk].mean() for i in range(len(sT1)//blk)])[:10]
    sA = np.array([sA1[i*blk:(i+1)*blk].mean() for i in range(len(sA1)//blk)])[:10]
    n_spin = len(sT); n_era5 = len(eT)
    dt_spin = SPINUP_DT * YEAR_S; dt_era5 = ERA5_DT * YEAR_S

    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
    mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)

    H_f = fd.Function(V, name="H"); rho_f = fd.Function(V, name="rho")
    w_f = fd.Function(V, name="w"); age_f = fd.Function(V, name="age")
    depth = H0 - xi

    log_E = mk(R, math.log(INITIAL["E_lin"]), "lE")
    log_Q = mk(R, math.log(INITIAL["Q_glen"]), "lQ")
    c2a = mk(R, INITIAL["c2_a"], "c2a")
    c2b = mk(R, INITIAL["c2_b"], "c2b")
    ctrls = [log_E, log_Q, c2a, c2b]

    params = FirnParameters(
        stokes_n_glen=1.0,
        stokes_E_lin=fd.exp(log_E),
        stokes_Q_glen=fd.exp(log_Q),
        stokes_c2_a=c2a, stokes_c2_b=c2b,
    )
    model = FirnModel(params, densification_rate_fn=stokes_compressible)
    solver = FirnColumnSolver(model, surface_id=SID)

    Ts = mk(R, float(sT[0]), "Ts")
    Hs = mk(R, float(params.c_i) * (sT[0] - float(params.T_ref)), "Hs")
    ac = mk(R, float(sA[0]), "ac"); rs = mk(R, RHO_SURF, "rs")
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
                age=age_f, age_boundary_condition=bc_age)

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
                age=age_f, age_boundary_condition=bc_age)

        # Just return a simple age-based cost
        _age_yr = fd.Function(V, name="age_yr")
        _age_yr.interpolate(age_f / fd.Constant(YEAR_S))
        return fd.assemble(_age_yr * _age_yr * fd.dx(domain=mesh))

    # ---- Build tape ----
    tape = get_working_tape(); tape.clear_tape()
    continue_annotation()
    J = forward()
    rf = ReducedFunctional(J, [Control(c) for c in ctrls])
    pause_annotation()

    coords = mesh.coordinates.dat.data_ro.flatten()
    idx = np.argsort(coords)
    d = H0 - coords[idx]

    print("After tape-building forward:")
    print(f"  J = {float(J):.4f}")
    print(f"  rho: [{rho_f.dat.data_ro.min():.1f}, {rho_f.dat.data_ro.max():.1f}]")
    age_yr_fwd = age_f.dat.data_ro[idx] / YEAR_S
    print(f"  age: [{age_yr_fwd.min():.1f}, {age_yr_fwd.max():.1f}] yr")
    print(f"  age at 5 depths: {np.interp([0, 25, 50, 75, 100], d, age_yr_fwd)}")

    # ---- Replay ----
    x0 = np.array([float(c.dat.data_ro[0]) for c in ctrls])
    with stop_annotating():
        for c, v in zip(ctrls, x0): c.assign(float(v))
        J_replay = float(rf(ctrls))

    print(f"\nAfter tape replay:")
    print(f"  J = {J_replay:.4f}")
    print(f"  rho: [{rho_f.dat.data_ro.min():.1f}, {rho_f.dat.data_ro.max():.1f}]")
    age_yr_rep = age_f.dat.data_ro[idx] / YEAR_S
    print(f"  age: [{age_yr_rep.min():.1f}, {age_yr_rep.max():.1f}] yr")
    print(f"  age at 5 depths: {np.interp([0, 25, 50, 75, 100], d, age_yr_rep)}")

    print(f"\nAge difference (replay - forward):")
    diff = age_yr_rep - age_yr_fwd
    print(f"  [{diff.min():.2f}, {diff.max():.2f}] yr")
    print(f"  at 5 depths: {np.interp([0, 25, 50, 75, 100], d, diff)}")

    # ---- Second replay ----
    with stop_annotating():
        for c, v in zip(ctrls, x0): c.assign(float(v))
        J_replay2 = float(rf(ctrls))

    age_yr_rep2 = age_f.dat.data_ro[idx] / YEAR_S
    print(f"\nAfter second replay:")
    print(f"  J = {J_replay2:.4f}")
    print(f"  age: [{age_yr_rep2.min():.1f}, {age_yr_rep2.max():.1f}] yr")
    diff2 = age_yr_rep2 - age_yr_rep
    print(f"  Age diff from first replay: [{diff2.min():.4f}, {diff2.max():.4f}]")

    print(f"\n{'='*60}")
    if abs(float(J) - J_replay) > 0.01:
        print("*** TAPE REPLAY MISMATCH CONFIRMED ***")
        print(f"  forward J = {float(J):.4f}, replay J = {J_replay:.4f}")
    else:
        print("Tape replay matches forward.")
