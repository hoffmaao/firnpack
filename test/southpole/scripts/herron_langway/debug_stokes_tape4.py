"""Minimal tape test: single time step."""
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

    T0 = 222.0  # K
    A0 = 0.08   # m/yr ice eq

    Ts = mk(R, T0, "Ts")
    Hs = mk(R, float(params.c_i) * (T0 - float(params.T_ref)), "Hs")
    ac = mk(R, A0, "ac"); rs = mk(R, RHO_SURF, "rs")
    ws = mk(R, -A0 * float(params.rho_i) / RHO_SURF / YEAR_S, "ws")
    dt_f = mk(R, 10.0 * YEAR_S, "dt")
    bc_H = fd.DirichletBC(V, Hs, SID)
    bc_rho = fd.DirichletBC(V, rs, SID)
    bc_w = fd.DirichletBC(V, ws, SID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)

    for n_steps in [1, 2, 5]:
        # Fresh solver each time
        solver = FirnColumnSolver(model, surface_id=SID)

        def forward():
            H_f.assign(float(params.c_i) * (T0 - float(params.T_ref)))
            rho_f.interpolate(fd.Constant(RHO_SURF) +
                (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) *
                (1.0 - fd.exp(-depth / 20.0)))
            w_f.assign(-A0 * float(params.rho_i) / RHO_SURF / YEAR_S)
            age_f.assign(0.0)

            for _ in range(n_steps):
                solver.prognostic_solve(
                    enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_f,
                    accumulation=ac, surface_density=rs,
                    boundary_conditions=[bc_H, bc_rho, bc_w],
                    surface_temperature=Ts, enthalpy_bc_constant=Hs,
                    age=age_f, age_boundary_condition=bc_age)

            return fd.assemble(rho_f * rho_f * dx) + fd.assemble(age_f * age_f * dx)

        tape = get_working_tape(); tape.clear_tape()
        continue_annotation()
        J = forward()
        rf = ReducedFunctional(J, [Control(c) for c in ctrls])
        pause_annotation()
        J_fwd = float(J)

        x0 = np.array([float(c.dat.data_ro[0]) for c in ctrls])
        with stop_annotating():
            for c, v in zip(ctrls, x0): c.assign(float(v))
            J_rep = float(rf(ctrls))

        diff_pct = 100 * abs(J_fwd - J_rep) / max(abs(J_fwd), 1)
        print(f"n_steps={n_steps}: forward={J_fwd:.4f}, replay={J_rep:.4f}, diff={J_fwd-J_rep:.6f} ({diff_pct:.6f}%)")
