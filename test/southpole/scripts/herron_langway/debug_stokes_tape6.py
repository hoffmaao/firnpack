"""Debug: does varying forcing cause the tape mismatch?"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import firedrake as fd
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, pause_annotation,
    stop_annotating, get_working_tape,
)
from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.solvers.firn_solver import FirnColumnSolver
from firnpack.physics.densification import stokes_compressible
from firnpack.constants import year as YEAR_S

H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF = 350.0
INITIAL = dict(E_lin=5.0e9, Q_glen=60.0e3, c2_a=15.78652, c2_b=20.46489)


def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f


if __name__ == "__main__":
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

    T_base = 222.0; A_base = 0.08
    Ts = mk(R, T_base, "Ts")
    Hs = mk(R, float(params.c_i) * (T_base - float(params.T_ref)), "Hs")
    ac = mk(R, A_base, "ac"); rs = mk(R, RHO_SURF, "rs")
    ws = mk(R, -A_base * float(params.rho_i) / RHO_SURF / YEAR_S, "ws")
    dt_f = mk(R, 10.0 * YEAR_S, "dt")
    bc_H = fd.DirichletBC(V, Hs, SID)
    bc_rho = fd.DirichletBC(V, rs, SID)
    bc_w = fd.DirichletBC(V, ws, SID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)

    n_steps = 5

    # Varying forcing arrays
    T_arr = [222.0, 223.0, 222.5, 221.0, 222.0]
    A_arr = [0.08, 0.09, 0.07, 0.085, 0.08]

    # ---- Test A: constant forcing ----
    print("Test A: CONSTANT forcing, 5 steps")
    solver_A = FirnColumnSolver(model, surface_id=SID)

    def forward_A():
        H_f.assign(float(params.c_i) * (T_base - float(params.T_ref)))
        rho_f.interpolate(fd.Constant(RHO_SURF) +
            (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) *
            (1.0 - fd.exp(-depth / 20.0)))
        w_f.assign(-A_base * float(params.rho_i) / RHO_SURF / YEAR_S)
        age_f.assign(0.0)
        for _ in range(n_steps):
            solver_A.prognostic_solve(
                enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_f,
                accumulation=ac, surface_density=rs,
                boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age)
        return fd.assemble(rho_f * rho_f * dx)

    tape = get_working_tape(); tape.clear_tape()
    continue_annotation()
    J_A = forward_A()
    rf_A = ReducedFunctional(J_A, [Control(c) for c in ctrls])
    pause_annotation()
    J_fwd_A = float(J_A)
    x0 = np.array([float(c.dat.data_ro[0]) for c in ctrls])
    with stop_annotating():
        for c, v in zip(ctrls, x0): c.assign(float(v))
        J_rep_A = float(rf_A(ctrls))
    print(f"  forward={J_fwd_A:.4f}, replay={J_rep_A:.4f}, diff={J_fwd_A-J_rep_A:.6f}")

    # ---- Test B: VARYING forcing ----
    print("\nTest B: VARYING forcing, 5 steps")
    solver_B = FirnColumnSolver(model, surface_id=SID)

    def forward_B():
        H_f.assign(float(params.c_i) * (T_base - float(params.T_ref)))
        rho_f.interpolate(fd.Constant(RHO_SURF) +
            (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) *
            (1.0 - fd.exp(-depth / 20.0)))
        w_f.assign(-A_base * float(params.rho_i) / RHO_SURF / YEAR_S)
        age_f.assign(0.0)
        for k in range(n_steps):
            Tk, Ak = T_arr[k], A_arr[k]
            Ts.assign(Tk)
            Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
            ac.assign(Ak)
            ws.assign(-Ak * float(params.rho_i) / RHO_SURF / YEAR_S)
            solver_B.prognostic_solve(
                enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_f,
                accumulation=ac, surface_density=rs,
                boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age)
        return fd.assemble(rho_f * rho_f * dx)

    tape.clear_tape()
    continue_annotation()
    J_B = forward_B()
    rf_B = ReducedFunctional(J_B, [Control(c) for c in ctrls])
    pause_annotation()
    J_fwd_B = float(J_B)
    with stop_annotating():
        for c, v in zip(ctrls, x0): c.assign(float(v))
        J_rep_B = float(rf_B(ctrls))
    print(f"  forward={J_fwd_B:.4f}, replay={J_rep_B:.4f}, diff={J_fwd_B-J_rep_B:.6f}")

    # ---- Test C: VARYING forcing + control-dep zero hack ----
    print("\nTest C: VARYING forcing + control-dependent assign hack, 5 steps")
    solver_C = FirnColumnSolver(model, surface_id=SID)
    zero_ctrl = fd.Constant(0.0) * fd.exp(log_E)

    def forward_C():
        H_f.assign(float(params.c_i) * (T_base - float(params.T_ref)))
        rho_f.interpolate(fd.Constant(RHO_SURF) +
            (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) *
            (1.0 - fd.exp(-depth / 20.0)))
        w_f.assign(-A_base * float(params.rho_i) / RHO_SURF / YEAR_S)
        age_f.assign(0.0)
        for k in range(n_steps):
            Tk, Ak = T_arr[k], A_arr[k]
            # Make forcing assigns control-dependent via trivial zero_ctrl
            Ts.interpolate(fd.Constant(Tk) + zero_ctrl)
            Hs.interpolate(fd.Constant(float(params.c_i) * (Tk - float(params.T_ref))) + zero_ctrl)
            ac.interpolate(fd.Constant(Ak) + zero_ctrl)
            ws.interpolate(fd.Constant(-Ak * float(params.rho_i) / RHO_SURF / YEAR_S) + zero_ctrl)
            solver_C.prognostic_solve(
                enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_f,
                accumulation=ac, surface_density=rs,
                boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age)
        return fd.assemble(rho_f * rho_f * dx)

    tape.clear_tape()
    continue_annotation()
    J_C = forward_C()
    rf_C = ReducedFunctional(J_C, [Control(c) for c in ctrls])
    pause_annotation()
    J_fwd_C = float(J_C)
    with stop_annotating():
        for c, v in zip(ctrls, x0): c.assign(float(v))
        J_rep_C = float(rf_C(ctrls))
    print(f"  forward={J_fwd_C:.4f}, replay={J_rep_C:.4f}, diff={J_fwd_C-J_rep_C:.6f}")
