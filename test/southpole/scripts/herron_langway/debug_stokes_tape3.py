"""Debug: isolate the assemble mismatch — try different cost formulations."""
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
from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.solvers.firn_solver import FirnColumnSolver
from firnpack.physics.densification import stokes_compressible
from firnpack.constants import year as YEAR_S

_HERE = Path(__file__).parent
_SP = _HERE.parent.parent
PROC = _SP / "processed"

H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF = 350.0
SPINUP_DT = 10.0

INITIAL = dict(E_lin=5.0e9, Q_glen=60.0e3, c2_a=15.78652, c2_b=20.46489)


def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f


if __name__ == "__main__":
    # Minimal forcing: 5 spinup steps
    bT = pd.read_csv(PROC / "buizert2021_spice_temp.csv").sort_values("year_CE")
    bA = pd.read_csv(PROC / "buizert2021_spice_accum.csv").sort_values("year_CE")
    yr1 = np.arange(740, 1940, 1.0)
    sT1 = np.interp(yr1, bT["year_CE"].values, bT["temp"].values + 273.15)
    sA1 = np.interp(yr1, bA["year_CE"].values, bA["accum"].values)
    blk = int(SPINUP_DT)
    sT = np.array([sT1[i*blk:(i+1)*blk].mean() for i in range(len(sT1)//blk)])[:5]
    sA = np.array([sA1[i*blk:(i+1)*blk].mean() for i in range(len(sA1)//blk)])[:5]
    n_spin = len(sT)
    dt_spin = SPINUP_DT * YEAR_S

    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
    mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)
    dx = fd.dx(domain=mesh)

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
    dt_f = mk(R, dt_spin, "dt")
    bc_H = fd.DirichletBC(V, Hs, SID)
    bc_rho = fd.DirichletBC(V, rs, SID)
    bc_w = fd.DirichletBC(V, ws, SID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)

    # ---- Test each cost formulation ----
    def run_test(label, cost_fn):
        print(f"\n{'='*60}")
        print(f"Test: {label}")

        def forward():
            T0, A0 = float(sT[0]), float(sA[0])
            H_f.assign(float(params.c_i) * (T0 - float(params.T_ref)))
            rho_f.interpolate(fd.Constant(RHO_SURF) +
                (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) *
                (1.0 - fd.exp(-depth / 20.0)))
            w_f.assign(-A0 * float(params.rho_i) / RHO_SURF / YEAR_S)
            age_f.assign(0.0)

            dt_f.assign(dt_spin)
            for k in range(n_spin):
                Tk, Ak = float(sT[k]), float(sA[k])
                Ts.assign(Tk); Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
                ac.assign(Ak); ws.assign(-Ak * float(params.rho_i) / RHO_SURF / YEAR_S)
                solver.prognostic_solve(
                    enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_f,
                    accumulation=ac, surface_density=rs,
                    boundary_conditions=[bc_H, bc_rho, bc_w],
                    surface_temperature=Ts, enthalpy_bc_constant=Hs,
                    age=age_f, age_boundary_condition=bc_age)
            return cost_fn()

        tape = get_working_tape(); tape.clear_tape()
        # Reset controls
        log_E.assign(math.log(INITIAL["E_lin"]))
        log_Q.assign(math.log(INITIAL["Q_glen"]))
        c2a.assign(INITIAL["c2_a"]); c2b.assign(INITIAL["c2_b"])

        continue_annotation()
        J = forward()
        rf = ReducedFunctional(J, [Control(c) for c in ctrls])
        pause_annotation()

        J_fwd = float(J)

        x0 = np.array([float(c.dat.data_ro[0]) for c in ctrls])
        with stop_annotating():
            for c, v in zip(ctrls, x0): c.assign(float(v))
            J_rep = float(rf(ctrls))

        match = abs(J_fwd - J_rep) < 0.01 * max(abs(J_fwd), 1.0)
        print(f"  forward: {J_fwd:.6f}")
        print(f"  replay:  {J_rep:.6f}")
        print(f"  diff:    {J_fwd - J_rep:.6f}")
        print(f"  {'OK' if match else '*** MISMATCH ***'}")

    # Test 1: cost = assemble(rho * dx)
    run_test("rho integral", lambda: fd.assemble(rho_f * dx))

    # Test 2: cost = assemble(rho^2 * dx)
    run_test("rho^2 integral", lambda: fd.assemble(rho_f * rho_f * dx))

    # Test 3: cost = assemble(age * dx)
    run_test("age integral", lambda: fd.assemble(age_f * dx))

    # Test 4: cost via CG1 intermediate (age_yr)
    def cost_age_cg1():
        _f = fd.Function(V, name="age_yr_test")
        _f.interpolate(age_f / fd.Constant(YEAR_S))
        return fd.assemble(_f * _f * dx)
    run_test("age via CG1 intermediate", cost_age_cg1)

    # Test 5: cost = assemble(H * dx) (enthalpy)
    run_test("H integral", lambda: fd.assemble(H_f * dx))

    # Test 6: mixed rho + age
    def cost_mixed():
        return fd.assemble(rho_f * dx) + fd.assemble(age_f * dx)
    run_test("rho + age mixed", cost_mixed)

    print(f"\n{'='*60}")
    print("Done.")
