"""Minimal adjoint diagnostic: test each control individually.

Runs one forward eval + gradient, then finite-differences each control
to compare adjoint vs FD gradient components.
"""
from __future__ import annotations
import math, os, sys, numpy as np
# Import everything from the main script up to control setup
# but override: skip optimization, just do diagnostic

os.environ.setdefault("OMP_NUM_THREADS", "1")

import firedrake as fd
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, pause_annotation,
    stop_annotating, get_working_tape,
)
from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.constants import year as YEAR_S

# Use a VERY short run: 10-year spinup, 5-year ERA5
SPINUP_YEARS_DIAG = 10
ERA5_YEARS_DIAG = 5

# Import settings from the main script
sys.path.insert(0, os.path.dirname(__file__))
from southpole_inversion_v5_sp19 import (
    H0, NZ, STRETCH_P, SURFACE_ID, BASE_ID,
    INITIAL, BOUNDS, TRUTH, PENALTY_STRENGTH,
    SIGMA_RHO_ABS, SIGMA_AGE_ABS_S, USE_PRIOR,
    PRIOR_SIGMA_LOG, RHO_SURF_PRIOR_SIGMA_LOG,
    R2_SURF_PRIOR_SIGMA_LOG, DIV_H_PRIOR_SIGMA,
    RHO_M, RHO_SMOOTH, SEED,
    build_stretched_mesh, make_bcs, w_base_value, make_real,
)

print("=== Adjoint Diagnostic ===\n")

mesh, V = build_stretched_mesh(H0, NZ, STRETCH_P)
R_space = fd.FunctionSpace(mesh, "R", 0)
dx_domain = fd.dx(domain=mesh)
domain_len = fd.assemble(fd.Constant(1.0) * dx_domain)

# State fields
H     = fd.Function(V, name="H")
rho   = fd.Function(V, name="rho")
sigma = fd.Function(V, name="sigma")
r2    = fd.Function(V, name="r2")
w     = fd.Function(V, name="w")
age   = fd.Function(V, name="age")

# Simple initial conditions
rho_init = 500.0  # uniform density
rho.assign(rho_init)

# Controls
log_kc0   = make_real(R_space, math.log(INITIAL["kc0"]),      "log_kc0")
log_kc1   = make_real(R_space, math.log(INITIAL["kc1"]),      "log_kc1")
log_kg    = make_real(R_space, math.log(INITIAL["kg"]),       "log_kg")
log_rho_s = make_real(R_space, math.log(INITIAL["rho_surf"]), "log_rho_s")
log_r2_s  = make_real(R_space, math.log(INITIAL["r2_surf"]),  "log_r2_s")
div_h_ctrl = make_real(R_space, INITIAL["div_h"],             "div_h")

controls = [log_kc0, log_kc1, log_kg, log_rho_s, log_r2_s, div_h_ctrl]
names    = ["kc0", "kc1", "kg", "rho_surf", "r2_surf", "div_h"]

kc0_expr      = fd.exp(log_kc0)
kc1_expr      = fd.exp(log_kc1)
kg_expr       = fd.exp(log_kg)
rho_surf_expr = fd.exp(log_rho_s)
r2_surf_expr  = fd.exp(log_r2_s)

params = FirnParameters(kg=kg_expr)
model  = FirnModel(params)
solver = FirnColumnSolver(model, surface_id=SURFACE_ID,
                           horizontal_divergence=div_h_ctrl)

Ts    = make_real(R_space, 222.0, "Ts")
accum = make_real(R_space, 0.08, "accum")
Hs    = make_real(R_space, float(params.c_i) * (222.0 - float(params.T_ref)), "Hs")
dt    = make_real(R_space, 1.0 * float(YEAR_S), "dt")

rho_surf_fs = make_real(R_space, INITIAL["rho_surf"], "rho_surf_fs")
bcs, bc_vals = make_bcs(V, params, accum, Hs)
w_base_fn = bc_vals["w_base_fn"]
bc_w_base = bc_vals["bc_w_base"]
bc_sigma = fd.DirichletBC(V, make_real(R_space, 0.0, "sig0"), SURFACE_ID)
bc_age   = fd.DirichletBC(V, make_real(R_space, 0.0, "age0"), SURFACE_ID)
rhoCoef  = fd.Function(V, name="rhoCoef")

penalty_bcs = {
    "rho": (rho_surf_expr, PENALTY_STRENGTH),
    "r2":  (r2_surf_expr,  PENALTY_STRENGTH),
}

_prior_fn = fd.Function(V, name="_prior_fn")

def forward_short():
    """Short forward model: a few timesteps."""
    rho_surf_fs.interpolate(rho_surf_expr)
    rho.assign(rho_init)
    H.assign(float(params.c_i) * (222.0 - float(params.T_ref)))
    sigma.assign(0.0)
    r2.assign(INITIAL["r2_surf"])
    w.assign(0.0)
    age.assign(0.0)

    n_steps = SPINUP_YEARS_DIAG + ERA5_YEARS_DIAG
    for k in range(n_steps):
        accum.assign(0.08)
        w_base_fn.interpolate(w_base_value(accum))

        s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
        rhoCoef.interpolate((1.0 - s) * kc0_expr + s * kc1_expr)

        solver.prognostic_solve(
            enthalpy=H, density=rho, firn_velocity=w, dt=dt,
            accumulation=accum, surface_density=rho_surf_fs,
            boundary_conditions=bcs, surface_temperature=Ts,
            enthalpy_bc_constant=Hs,
            stress=sigma, grain_radius2=r2,
            stress_boundary_condition=bc_sigma,
            grain_radius2_boundary_condition=None,
            penalty_bcs=penalty_bcs,
            mass_flux_bc=bc_w_base,
            rhoCoef=rhoCoef,
            age=age, age_boundary_condition=bc_age,
        )

    # Simple objective: just density L2 norm (to test tape)
    J = 0.5 * fd.assemble(rho * rho * dx_domain) / domain_len

    # Add prior terms (these should definitely have correct derivatives)
    if USE_PRIOR:
        for log_ctrl, truth_val, sig_log in [
            (log_kc0,   TRUTH["kc0"],      PRIOR_SIGMA_LOG),
            (log_kc1,   TRUTH["kc1"],      PRIOR_SIGMA_LOG),
            (log_kg,    TRUTH["kg"],        PRIOR_SIGMA_LOG),
            (log_rho_s, TRUTH["rho_surf"],  RHO_SURF_PRIOR_SIGMA_LOG),
            (log_r2_s,  TRUTH["r2_surf"],   R2_SURF_PRIOR_SIGMA_LOG),
        ]:
            _prior_fn.interpolate((log_ctrl - math.log(float(truth_val))) / sig_log)
            J = J + 0.5 * fd.assemble(_prior_fn * _prior_fn * dx_domain) / domain_len
        _prior_fn.interpolate(
            (div_h_ctrl - fd.Constant(float(TRUTH["div_h"]))) / fd.Constant(DIV_H_PRIOR_SIGMA)
        )
        J = J + 0.5 * fd.assemble(_prior_fn * _prior_fn * dx_domain) / domain_len

    return J

# --- Build tape ---
tape = get_working_tape()
tape.clear_tape()
continue_annotation()

J_init = forward_short()
rf = ReducedFunctional(J_init, [Control(c) for c in controls])
pause_annotation()

J0 = float(rf(controls))
print(f"J(m0) = {J0:.6e}")
print(f"Tape blocks: {len(tape.get_blocks())}")

# --- Adjoint gradient ---
with stop_annotating():
    g_list = rf.derivative()
g_adj = np.array([float(g.dat.data_ro[0]) * domain_len for g in g_list])
print(f"\nAdjoint gradient:")
for nm, gi in zip(names, g_adj):
    print(f"  dJ/d({nm:10s}) = {gi:+.6e}")

# --- Per-control finite differences ---
print(f"\nPer-control FD test (relative eps=1e-4):")

x0 = np.array([float(c.dat.data_ro[0]) for c in controls])
for i, nm in enumerate(names):
    # Use relative perturbation scaled to the control value.
    # For tiny controls like div_h (-1.6e-12), use 1% of the value.
    if abs(x0[i]) > 1e-30:
        eps_fd = 1e-2 * abs(x0[i])  # 1% relative
    else:
        eps_fd = 1e-4
    x_pert = x0.copy()
    x_pert[i] += eps_fd

    with stop_annotating():
        for c, v in zip(controls, x_pert):
            c.assign(float(v))
        J1 = float(rf(controls))

    fd_grad = (J1 - J0) / eps_fd
    adj_grad = g_adj[i]
    ratio = fd_grad / adj_grad if abs(adj_grad) > 1e-30 else float('inf')

    print(f"  {nm:10s}:  FD={fd_grad:+.6e}  adj={adj_grad:+.6e}  ratio={ratio:.3f}  |J1-J0|={abs(J1-J0):.3e}")

    # Reset
    with stop_annotating():
        for c, v in zip(controls, x0):
            c.assign(float(v))

# --- Also test replay consistency ---
print(f"\nReplay consistency:")
with stop_annotating():
    for c, v in zip(controls, x0):
        c.assign(float(v))
    J_replay = float(rf(controls))
print(f"  J0 (original) = {J0:.10e}")
print(f"  J0 (replay)   = {J_replay:.10e}")
print(f"  Difference     = {abs(J_replay - J0):.3e}")

print("\nDone.")
