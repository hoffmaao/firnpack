"""test_adjoint.py

Twin experiment: infer the firn densification parameter *kg* (grain-growth factor)
from synthetic observations of vertical velocity w(t, z).

This is intentionally patterned after the G-ADOPT adjoint tutorial (adjoint.py):
- clear tape
- time loop with cumulative misfit
- ReducedFunctional
- Taylor test
- minimise

Requirements
------------
- Firedrake
- firedrake-adjoint (pyadjoint)
- Your firn package (firnpack.models.firn, firnpack.solvers.firn_solver, firnpack.constants)

Notes
-----
1) Your FirnModel uses kg in the Arthern/Ligtenberg densification law as a divisor.
   That makes it a good first scalar control parameter.
2) DO NOT use w at the surface as an observation here, because w(surface) is
   imposed as a Dirichlet BC derived from accumulation and rho_s.
   We instead use an L2 misfit over the *interior* profile (or optionally base only).

Run
---
python test_adjoint.py

Then inspect the printed recovered kg and (optionally) the written VTK files.
"""

from __future__ import annotations

import importlib
import numpy as np

import firedrake as fd

# Importing the adjoint module turns on pyadjoint overloading/taping.
# (Same idea as `from gadopt.inverse import *` in the demo.)
# Prefer the non-deprecated import path.
try:
    from firedrake.adjoint import (
        Control,
        ReducedFunctional,
        get_working_tape,
        pause_annotation,
        stop_annotating,
        taylor_test,
        minimize,
    )
except Exception:  # pragma: no cover
    from firedrake_adjoint import (  # type: ignore
        Control,
        ReducedFunctional,
        get_working_tape,
        pause_annotation,
        stop_annotating,
        taylor_test,
        minimize,
    )

from firnpack.constants import year
from firnpack.models.firn import FirnModel, FirnParameters
from firnpack.solvers.firn_solver import FirnColumnSolver


# -----------------------------------------------------------------------------
# User-tunable experiment settings
# -----------------------------------------------------------------------------

# Geometry / discretisation
H0 = 40.0
nz = 80
mesh_stretch_p = 3.0

# Forcing assumptions (held fixed during inversion)
accum_m_iceeq_per_yr = 0.3
rho_surface = 300.0
Ts_K = 248.0

# Time stepping
# Keep this short for a first inverse test: you will run many forward replays.
dt_days = 10.0
nsteps = 60

# Synthetic "truth" and initial guess for inversion
kg_true = 2.0e-7
kg_initial_guess = 1.0e-7

# Bounds for kg in optimisation
kg_lower = 1.0e-8
kg_upper = 1.0e-6

# Observation operator choice
#   "profile" : L2 misfit over entire depth profile each timestep
#   "base"    : misfit of basal velocity only (boundary integral)
OBS_OPERATOR = "profile"

# Synthetic noise level (set to 0.0 to start)
NOISE_REL = 0.0  # e.g. 0.02 = 2% relative Gaussian noise

# Suppress the very verbose diagnostics prints inside firn_solver.py
SUPPRESS_SOLVER_PRINTS = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_stretched_mesh(H0: float, nz: int, p: float) -> fd.Mesh:
    """Interval mesh on [0, H0] with vertical clustering near the surface."""
    mesh = fd.IntervalMesh(nz, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    z_expr = H0 * (1.0 - (1.0 - xi) ** p)

    coord_fs = mesh.coordinates.function_space()
    new_coords = fd.Function(coord_fs).interpolate(fd.as_vector([z_expr]))
    mesh.coordinates.assign(new_coords)
    return mesh


def initial_conditions(V: fd.FunctionSpace, params: FirnParameters, Ts: fd.Constant):
    """Create and initialise H, rho, w for a forward run."""
    H = fd.Function(V, name="enthalpy")
    rho = fd.Function(V, name="density")
    w = fd.Function(V, name="firn_velocity")

    H_init = params.c_i * (float(Ts) - params.T_ref)

    H.assign(H_init)
    rho.assign(300.0)
    w.assign(0.0)

    return H, rho, w, float(H_init)


def make_bcs(V: fd.FunctionSpace, params: FirnParameters, accum: fd.Constant, rho_s: fd.Constant,
             Hs_bc: fd.Constant, surface_id: int = 2):
    """Dirichlet BCs matching test.py: H, rho, and w imposed at the surface."""
    w_surf = -accum * params.rho_i / rho_s / year

    bc_H = fd.DirichletBC(V, Hs_bc, surface_id)
    bc_rho = fd.DirichletBC(V, rho_s, surface_id)
    bc_w = fd.DirichletBC(V, w_surf, surface_id)

    return [bc_H, bc_rho, bc_w]


def apply_relative_noise(f: fd.Function, rel: float, rng: np.random.Generator) -> fd.Function:
    """Return a deepcopy of f with optional relative Gaussian noise added."""
    g = f.copy(deepcopy=True)
    if rel <= 0.0:
        return g

    data = g.dat.data
    scale = rel * np.max(np.abs(data))
    if scale == 0.0:
        return g

    data[:] = data + scale * rng.standard_normal(size=data.shape)
    return g


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Optionally silence prints inside the FirnColumnSolver implementation.
    if SUPPRESS_SOLVER_PRINTS:
        try:
            mod = importlib.import_module("firnpack.solvers.firn_solver")
            mod.print = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        except Exception:
            # If the module path differs in your install, you can comment this out
            # and just edit firn_solver.py directly.
            pass

    # ------------------------------------------------------------------
    # Mesh / spaces / fixed forcings
    # ------------------------------------------------------------------
    mesh = build_stretched_mesh(H0, nz, mesh_stretch_p)
    V = fd.FunctionSpace(mesh, "CG", 1)

    dx = fd.dx(domain=mesh)
    ds = fd.ds(domain=mesh)

    surface_id = 2
    base_id = 1

    accum = fd.Constant(accum_m_iceeq_per_yr,domain=mesh)
    rho_s = fd.Constant(rho_surface,domain=mesh)
    Ts = fd.Constant(Ts_K,domain=mesh)
    dt = fd.Constant(dt_days * 86400.0,domain=mesh)

    # ------------------------------------------------------------------
    # 1) Reference twin: generate synthetic observations w_obs[k]
    # ------------------------------------------------------------------
    rng = np.random.default_rng(1234)

    with stop_annotating():
        params_ref = FirnParameters(kg=kg_true)
        model_ref = FirnModel(params_ref)
        solver_ref = FirnColumnSolver(model_ref)

        H_ref, rho_ref, w_ref, H_init_ref = initial_conditions(V, params_ref, Ts)
        Hs_bc_ref = fd.Constant(H_init_ref,domain=mesh)
        bcs_ref = make_bcs(V, params_ref, accum, rho_s, Hs_bc_ref, surface_id=surface_id)

        w_obs: list[fd.Function] = []

        for k in range(nsteps):
            H_ref, rho_ref, w_ref = solver_ref.prognostic_solve(
                enthalpy=H_ref,
                density=rho_ref,
                firn_velocity=w_ref,
                dt=dt,
                accumulation=accum,
                surface_density=rho_s,
                boundary_conditions=bcs_ref,
                surface_temperature=Ts,
                enthalpy_bc_constant=Hs_bc_ref,
            )

            w_obs.append(apply_relative_noise(w_ref, NOISE_REL, rng))

    # ------------------------------------------------------------------
    # 2) Clear tape (as in the G-ADOPT adjoint demo)
    # ------------------------------------------------------------------
    tape = get_working_tape()
    tape.clear_tape()

    # ------------------------------------------------------------------
    # 3) Set up the inversion problem (kg as the Control)
    # ------------------------------------------------------------------
    kg = fd.Constant(kg_initial_guess,domain=mesh)

    params = FirnParameters(kg=kg)
    model = FirnModel(params)
    solver = FirnColumnSolver(model)

    H, rho, w, H_init = initial_conditions(V, params, Ts)
    Hs_bc = fd.Constant(H_init,domain=mesh)
    bcs = make_bcs(V, params, accum, rho_s, Hs_bc, surface_id=surface_id)

    control = Control(kg)

    # ------------------------------------------------------------------
    # 4) Time-dependent objective: accumulate w-misfit over timesteps
    # ------------------------------------------------------------------
    J = 0.0

    for k in range(nsteps):
        H, rho, w = solver.prognostic_solve(
            enthalpy=H,
            density=rho,
            firn_velocity=w,
            dt=dt,
            accumulation=accum,
            surface_density=rho_s,
            boundary_conditions=bcs,
            surface_temperature=Ts,
            enthalpy_bc_constant=Hs_bc,
        )

        if OBS_OPERATOR == "profile":
            # L2 misfit of full depth profile
            J += 0.5 * fd.assemble((w - w_obs[k]) ** 2 * dx)
        elif OBS_OPERATOR == "base":
            # boundary misfit at the base only (single scalar per timestep)
            J += 0.5 * fd.assemble((w - w_obs[k]) ** 2 * ds(base_id))
        else:
            raise ValueError(f"Unknown OBS_OPERATOR={OBS_OPERATOR!r}")

    # Optional weak regularisation / prior on kg (helps if data are noisy)
    # (Keep alpha small; start with 0.0)
    alpha = 0.0
    kg_prior = 1.3e-7
    if alpha > 0.0:
        J += 0.5 * alpha * (kg - kg_prior) ** 2

    # ------------------------------------------------------------------
    # 5) Reduced functional + pause annotation
    # ------------------------------------------------------------------
    rf = ReducedFunctional(J, control)

    # Stop any further taping (same pattern as the demo)
    pause_annotation()

    # ------------------------------------------------------------------
    # 6) Gradient check (Taylor test)
    # ------------------------------------------------------------------
    # For a scalar Constant control, a simple direction is fine.
    dkg = fd.Constant(1.0,domain=mesh)
    print("Running Taylor test for dJ/dkg ...")
    rate = taylor_test(rf, kg, dkg)
    print(f"  Taylor test convergence rate: {rate}")

    # ------------------------------------------------------------------
    # 7) Optimisation
    # ------------------------------------------------------------------
    J_hist: list[float] = []
    kg_hist: list[float] = []

    def record_eval(Jval, mval):
        # Jval is usually an AdjFloat, but casts to float.
        J_hist.append(float(Jval))
        kg_hist.append(float(mval))
        print(f"  J = {float(Jval):.6e} ; kg = {float(mval):.6e}")

    rf.eval_cb_post = record_eval

    kg_lb = fd.Constant(kg_lower,domain=mesh)
    kg_ub = fd.Constant(kg_upper,domain=mesh)

    print("\nStarting optimisation...")
    print(f"  true kg         = {kg_true:.6e}")
    print(f"  initial guess kg = {kg_initial_guess:.6e}")
    print(f"  bounds           = [{kg_lower:.2e}, {kg_upper:.2e}]")

    # L-BFGS-B is usually the most robust default for 1-parameter problems.
    kg_opt = minimize(
        rf,
        method="L-BFGS-B",
        bounds=(kg_lb, kg_ub),
        options={"maxiter": 25},
    )

    print("\nDone.")
    print(f"Recovered kg: {float(kg_opt):.6e}")

    # ------------------------------------------------------------------
    # 8) (Optional) write out the time history of J and kg
    # ------------------------------------------------------------------
    with open("kg_inversion_history.txt", "w") as f:
        f.write("iter,J,kg\n")
        for i, (Jv, kv) in enumerate(zip(J_hist, kg_hist)):
            f.write(f"{i},{Jv},{kv}\n")


if __name__ == "__main__":
    main()
