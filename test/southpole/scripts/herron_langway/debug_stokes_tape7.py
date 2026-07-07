"""Debug: test eval_cb_pre approach to fix forcing replay."""
from __future__ import annotations
import math
import numpy as np
import firedrake as fd
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, pause_annotation,
    stop_annotating, get_working_tape,
)
from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.physics.densification import stokes_compressible
from firn.constants import year as YEAR_S

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
    T_arr = [222.0, 223.0, 222.5, 221.0, 222.0]
    A_arr = [0.08, 0.09, 0.07, 0.085, 0.08]

    # ---- Approach: use stop_annotating for forcing, apply via tape blocks ----
    # Store forcing as tape-block-indexed arrays. Each time step k gets its own
    # set of "annotated" assign blocks.
    #
    # DIFFERENT approach: put forcing updates OUTSIDE annotation.
    # The solver still sees the R-space values. On tape replay, the solver
    # blocks are replayed using whatever Ts/ac/ws values are current.
    # We use eval_cb_pre to set them before each replay.

    print("Test: varying forcing with stop_annotating + eval_cb_pre")
    solver = FirnColumnSolver(model, surface_id=SID)

    def forward():
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
            solver.prognostic_solve(
                enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_f,
                accumulation=ac, surface_density=rs,
                boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age)
        return fd.assemble(rho_f * rho_f * dx)

    tape = get_working_tape(); tape.clear_tape()
    continue_annotation()
    J = forward()
    rf = ReducedFunctional(J, [Control(c) for c in ctrls])
    pause_annotation()
    J_fwd = float(J)

    # Standard replay (broken)
    x0 = np.array([float(c.dat.data_ro[0]) for c in ctrls])
    with stop_annotating():
        for c, v in zip(ctrls, x0): c.assign(float(v))
        J_rep = float(rf(ctrls))
    print(f"  forward={J_fwd:.4f}")
    print(f"  replay (standard)={J_rep:.4f}, diff={J_fwd-J_rep:.6f}")

    # Now try: clear tape, re-annotate, re-run forward, rebuild RF
    # This is the "no-replay" approach
    tape.clear_tape()
    for c, v in zip(ctrls, x0): c.assign(float(v))
    continue_annotation()
    J2 = forward()
    rf2 = ReducedFunctional(J2, [Control(c) for c in ctrls])
    pause_annotation()
    J_fwd2 = float(J2)

    with stop_annotating():
        for c, v in zip(ctrls, x0): c.assign(float(v))
        J_rep2 = float(rf2(ctrls))
    print(f"  forward2={J_fwd2:.4f}")
    print(f"  replay2 (fresh tape)={J_rep2:.4f}, diff={J_fwd2-J_rep2:.6f}")

    # Test: Taylor test on the second RF (should work if adjoint is correct)
    print("\n  Taylor test on fresh-tape RF...")
    with stop_annotating():
        for c, v in zip(ctrls, x0): c.assign(float(v))
        J0 = float(rf2(ctrls))
        g_list = rf2.derivative()
    g = np.array([float(gi.dat.data_ro[0]) for gi in g_list])
    rng = np.random.default_rng(42)
    d = rng.standard_normal(len(ctrls)) * 0.01
    gd = g @ d
    print(f"  J0={J0:.4f}, <dJ,d>={gd:.6e}")
    prev = None
    for eps in [0.1, 0.05, 0.025, 0.0125]:
        with stop_annotating():
            for c, v in zip(ctrls, x0 + eps*d): c.assign(float(v))
            J1 = float(rf2(ctrls))
        r = abs(J1 - J0 - eps * gd)
        o = f"{math.log(prev / r) / math.log(2):.2f}" if prev and r > 0 else ""
        print(f"  {eps:.4f}  {abs(J1 - J0):.6e}  {r:.6e}  {o}")
        prev = r
