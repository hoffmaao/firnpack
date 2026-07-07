"""clean_stepper_taylor.py

Isolate the forward-adjoint problem.  The FirnColumnSolver adds operations that
Evan Cummings' original did NOT have and that are prime adjoint-corruption
suspects: it interpolates drhodt into a CG1 Function, runs an unused
overburden_stress BVP solve, and interpolates a strain-rate `eps` — all on tape,
every step.

Here we build a LEAN (H, rho, w, age) stepper using the verified model weak
forms DIRECTLY (drhodt as a UFL expression inside velocity_delta, Evan's way),
with cached solvers and none of the extra operations.  Then run a Taylor test.

If this gives order 2.0, the FirnColumnSolver extras were the culprit and this
lean stepper becomes the clean forward driver.
"""
from __future__ import annotations
import functools, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import firedrake as fd
from firedrake.adjoint import (Control, continue_annotation, pause_annotation,
                               stop_annotating, get_working_tape)
from pyadjoint import compute_gradient
from firn.models.firn import FirnParameters, FirnModel
from firn.physics.densification import herron_langway as _hl
from firn.constants import year as YEAR_S
sys.stdout.reconfigure(line_buffering=True)
herron_langway = functools.partial(_hl, smooth=True)

PROC = Path(__file__).parent.parent.parent / "processed"
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, T_SURF_C, BDOT = 350.0, -45.5, 0.085
DT_YEARS = 5.0
SPIN_YEARS = float(sys.argv[1]) if len(sys.argv) > 1 else 800.0
n_steps = int(SPIN_YEARS / DT_YEARS)

P0 = FirnParameters()
c_i, T_ref, rho_i, spy = float(P0.c_i), float(P0.T_ref), float(P0.rho_i), float(P0.spy)

mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0 * (1.0 - (1.0 - fd.SpatialCoordinate(mesh)[0]) ** STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space())
                        .interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1)
R = fd.FunctionSpace(mesh, "R", 0)
dx = fd.dx(domain=mesh)
depth = H0 - fd.SpatialCoordinate(mesh)[0]
psi = fd.TestFunction(V)

def mk(v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f

# controls
log_k0 = mk(math.log(11.0), "lk0"); log_k1 = mk(math.log(575.0), "lk1")
log_Ea1 = mk(math.log(10160.0), "lEa1"); log_Ea2 = mk(math.log(21400.0), "lEa2")
ctrls = [log_k0, log_k1, log_Ea1, log_Ea2]
names = ["k0", "k1", "Ea1", "Ea2"]
params = FirnParameters(hl_k0_prefactor=fd.exp(log_k0), hl_k1_prefactor=fd.exp(log_k1),
                        hl_Ea_stage1=fd.exp(log_Ea1), hl_Ea_stage2=fd.exp(log_Ea2))
model = FirnModel(params, densification_rate_fn=herron_langway)

# state + olds
T_surf_K = T_SURF_C + 273.15
H_f = fd.Function(V, name="H"); rho_f = fd.Function(V, name="rho")
w_f = fd.Function(V, name="w"); age_f = fd.Function(V, name="age")
H_o = fd.Function(V); rho_o = fd.Function(V); w_o = fd.Function(V); age_o = fd.Function(V)

ws0 = -BDOT * rho_i / RHO_SURF / spy
ac = mk(BDOT, "ac"); rs = mk(RHO_SURF, "rs"); ws = mk(ws0, "ws")
Hs = mk(c_i * (T_surf_K - T_ref), "Hs"); dt_r = mk(DT_YEARS * YEAR_S, "dt")
bc_H = fd.DirichletBC(V, Hs, SID); bc_rho = fd.DirichletBC(V, rs, SID)
bc_w = fd.DirichletBC(V, ws, SID); bc_age = fd.DirichletBC(V, mk(0.0, "a0"), SID)
bdot_mass = ac * rho_i / spy   # kg/m2/s

# ---- cached solvers (forms reference persistent Functions) ----
sp_lin = {"ksp_type": "preonly", "pc_type": "lu"}
Ht = fd.TrialFunction(V); rt = fd.TrialFunction(V)
wt = fd.TrialFunction(V); at = fd.TrialFunction(V)

# enthalpy (linear in H): use model.enthalpy_form with trial
F_H = model.enthalpy_form(Ht, H_o, rho_f, w_f, psi, dt_r)
solv_H = fd.LinearVariationalSolver(
    fd.LinearVariationalProblem(fd.lhs(F_H), fd.rhs(F_H), H_f, bcs=[bc_H],
                                constant_jacobian=False), solver_parameters=sp_lin)

T_expr = model.temperature_from_enthalpy(H_f)
F_rho, _ = model.density_form(rt, rho_o, T_expr, w_f, None, bdot_mass, psi, dt_r)
solv_rho = fd.LinearVariationalSolver(
    fd.LinearVariationalProblem(fd.lhs(F_rho), fd.rhs(F_rho), rho_f, bcs=[bc_rho],
                                constant_jacobian=False), solver_parameters=sp_lin)

# velocity: drhodt as a UFL EXPRESSION (Evan's way -- no interpolation)
drhodt_expr = herron_langway(rho_f, T_expr, params=params, bdot=bdot_mass)
delta_w = model.velocity_delta(wt, w_o, rho_f, drhodt_expr, psi, regularization=1e-3)
solv_w = fd.LinearVariationalSolver(
    fd.LinearVariationalProblem(fd.lhs(delta_w), fd.rhs(delta_w), w_f, bcs=[bc_w],
                                constant_jacobian=False), solver_parameters=sp_lin)

F_age = model.age_form(at, age_o, w_f, w_o, psi, dt_r)
solv_age = fd.LinearVariationalSolver(
    fd.LinearVariationalProblem(fd.lhs(F_age), fd.rhs(F_age), age_f, bcs=[bc_age],
                                constant_jacobian=False), solver_parameters=sp_lin)

# ---- density obs kernels (off tape) ----
xc = fd.SpatialCoordinate(mesh)[0]
dens = pd.read_csv(PROC / "sp19_density.csv"); dens = dens[dens.depth_m <= H0]
dens = dens.iloc[::4]
obs_d = dens.depth_m.values; obs_rho = dens.rho_kgm3.values * 1000.0
sig = 15.0 + 0.03 * obs_rho
kers = []
with stop_annotating():
    for d in obs_d:
        s = float(np.clip(0.04 * d + 0.5, 0.5, 3.0))
        phi = fd.Function(V).interpolate(fd.exp(-0.5 * ((xc - (H0 - d)) / s) ** 2))
        phi.dat.data[:] /= float(fd.assemble(phi * dx)) + 1e-30
        kers.append(phi)
N = len(kers)

def forward():
    H_f.assign(c_i * (T_surf_K - T_ref)); H_o.assign(H_f)
    rho_f.interpolate(RHO_SURF + (820.0 - RHO_SURF) * (1.0 - fd.exp(-depth / 25.0)))
    rho_o.assign(rho_f); w_f.assign(ws0); w_o.assign(w_f)
    age_f.assign(0.0); age_o.assign(0.0)
    for _ in range(n_steps):
        solv_H.solve()
        solv_rho.solve()
        solv_w.solve()
        solv_age.solve()
        H_o.assign(H_f); rho_o.assign(rho_f); w_o.assign(w_f); age_o.assign(age_f)
    J = 0.0
    for phi, o, s in zip(kers, obs_rho, sig):
        pred = fd.assemble(rho_f * phi * dx)
        r = (pred - float(o)) / float(s)
        J = J + 0.5 / N * r * r
    return J

tape = get_working_tape()
print(f"clean stepper Taylor: {n_steps} steps, {N} density obs")
tape.clear_tape(); continue_annotation()
J0f = forward()
dJ = compute_gradient(J0f, [Control(c) for c in ctrls])
pause_annotation()
J0 = float(J0f)
dlen = float(fd.assemble(fd.Constant(1.0) * dx))
g = np.array([float(gi.dat.data_ro[0]) * dlen for gi in dJ])
print(f"  J0={J0:.6f}  rho_max={float(rho_f.dat.data_ro.max()):.0f}")
print("  dJ: " + " ".join(f"{n}={g[i]:.3e}" for i, n in enumerate(names)))
rng = np.random.default_rng(0); d = rng.standard_normal(4) * 0.01; gd = g @ d
prev = None
for eps in [0.1, 0.05, 0.025, 0.0125, 0.00625]:
    tape.clear_tape(); continue_annotation()
    x0 = np.array([float(c.dat.data_ro[0]) for c in ctrls])
    for c, v in zip(ctrls, x0 + eps * d): c.assign(float(v))
    J1 = float(forward()); pause_annotation()
    for c, v in zip(ctrls, x0): c.assign(float(v))  # restore
    r = abs(J1 - J0 - eps * gd)
    o = f"order={math.log(prev/r)/math.log(2):.2f}" if prev and r > 0 else ""
    print(f"  eps={eps:.5f} rem={r:.3e} {o}")
    prev = r
print("Done.")
