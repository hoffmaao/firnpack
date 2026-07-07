"""test_adjoint_sparse_obs.py

Twin experiment: infer the firn densification parameter kg (grain-growth factor)
from *sparse* synthetic observations of vertical velocity w(t, z).

This script adapts the "sparse data" pattern used in the Icepack tutorial
"Assimilating sparse data" (VertexOnlyMesh + DG0 space on the point cloud)
to the 1D firn column setting.

Key idea
--------
Instead of comparing full profiles w(z) each timestep, we define a *point-set*
mesh at a handful of observation depths and assemble the misfit as a sum over
those points.

Run
---
python test_adjoint_sparse_obs.py

Outputs
-------
- Prints an objective sanity check and recovers kg in a bounded optimisation.
- Writes kg_inversion_history.txt

Notes
-----
- Uses firedrake.adjoint (preferred) with a fallback to firedrake_adjoint.
- Uses a log-parameter control m = log(kg) to enforce positivity.
- Normalises the time-integrated misfit by an observation norm so J ~ O(1).
"""

from __future__ import annotations

import importlib
import math
from typing import Sequence

import numpy as np

import firedrake as fd

# Prefer the non-deprecated import path.
try:
    from firedrake.adjoint import (
        Control,
        ReducedFunctional,
        continue_annotation,
        get_working_tape,
        minimize,
        pause_annotation,
        stop_annotating,
        taylor_test,
    )
except Exception:  # pragma: no cover
    from firedrake_adjoint import (  # type: ignore
        Control,
        ReducedFunctional,
        continue_annotation,
        get_working_tape,
        minimize,
        pause_annotation,
        stop_annotating,
        taylor_test,
    )

from firn.constants import year
from firn.models.firn import FirnModel, FirnParameters
from firn.solvers.firn_solver import FirnColumnSolver


# ----------------------------------------------------------------------------
# User settings
# ----------------------------------------------------------------------------

# Geometry / discretisation
H0 = 40.0
nz = 80
mesh_stretch_p = 3.0

# Forcing assumptions (held fixed during inversion)
accum_m_iceeq_per_yr = 0.3
rho_surface = 300.0
Ts_K = 248.0

# Time stepping
dt_days = 10.0
nsteps = 120
skip_steps_in_objective = 10

# Sparse observation depths (m below surface). Avoid 0 (surface BC for w).
OBS_DEPTHS_M = [2.0, 5.0, 10.0, 20.0, 30.0]

# Synthetic noise level (relative, applied on the point values)
NOISE_REL = 0.0

# Parameter truth / initial guess
kg_true = 2.0e-7
kg_initial_guess = 1.0e-7
kg_lower = 1.0e-8
kg_upper = 1.0e-6

# Suppress the verbose diagnostics prints inside firn_solver.py
SUPPRESS_SOLVER_PRINTS = True


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def build_stretched_mesh(H0: float, nz: int, p: float) -> fd.Mesh:
    """Interval mesh on [0, H0] with clustering near the surface."""
    mesh = fd.IntervalMesh(nz, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    z_expr = H0 * (1.0 - (1.0 - xi) ** p)
    coord_fs = mesh.coordinates.function_space()
    new_coords = fd.Function(coord_fs)
    new_coords.interpolate(fd.as_vector([z_expr]))
    mesh.coordinates.assign(new_coords)
    return mesh


def make_real(R: fd.FunctionSpace, value: float, name: str) -> fd.Function:
    f = fd.Function(R, name=name)
    f.assign(float(value))
    return f


def real_value(f: fd.Function) -> float:
    return float(f.dat.data_ro[0])


def initial_conditions(
    V: fd.FunctionSpace,
    params: FirnParameters,
    Ts: fd.Function,
    rho_surf_val: float,
    accum_val: float,
    H0: float,
    density_e_folding: float = 15.0,
):
    """Initialise H, rho, w with a simple realistic-ish profile."""
    mesh = V.mesh()
    x = fd.SpatialCoordinate(mesh)[0]
    depth = H0 - x

    H = fd.Function(V, name="enthalpy")
    rho = fd.Function(V, name="density")
    w = fd.Function(V, name="firn_velocity")

    Ts_val = real_value(Ts)
    H_init = params.c_i * (Ts_val - params.T_ref)
    H.assign(H_init)

    rho_expr = rho_surf_val + (params.rho_i - rho_surf_val) * (1.0 - fd.exp(-depth / density_e_folding))
    rho.interpolate(rho_expr)

    w_surf_val = -accum_val * params.rho_i / rho_surf_val / float(year)
    w.assign(w_surf_val)

    return H, rho, w, float(H_init)


def make_bcs(
    V: fd.FunctionSpace,
    params: FirnParameters,
    accum: fd.Function,
    rho_s: fd.Function,
    Hs_bc: fd.Function,
    surface_id: int = 2,
):
    w_surf = -accum * params.rho_i / rho_s / year
    bc_H = fd.DirichletBC(V, Hs_bc, surface_id)
    bc_rho = fd.DirichletBC(V, rho_s, surface_id)
    bc_w = fd.DirichletBC(V, w_surf, surface_id)
    return [bc_H, bc_rho, bc_w]


def build_observation_point_set(
    background_mesh: fd.Mesh,
    H0: float,
    depths_m: Sequence[float],
):
    """Create a VertexOnlyMesh at given depths (below surface) and DG0 space."""
    depths = np.array(depths_m, dtype=float)
    if np.any(depths <= 0.0):
        raise ValueError("All OBS_DEPTHS_M must be > 0 to avoid the surface BC.")
    if np.any(depths >= H0):
        raise ValueError("All OBS_DEPTHS_M must be < H0.")

    z_pts = (H0 - depths).reshape((-1, 1))  # Nx1 coordinate array
    point_mesh = fd.VertexOnlyMesh(background_mesh, z_pts, missing_points_behaviour="error")
    Qobs = fd.FunctionSpace(point_mesh, "DG", 0)
    dx_obs = fd.dx(domain=point_mesh)
    return point_mesh, Qobs, dx_obs


def apply_relative_noise_point_values(f: fd.Function, rel: float, rng: np.random.Generator) -> None:
    """In-place relative Gaussian noise on a DG0 point-set function."""
    if rel <= 0.0:
        return
    data = f.dat.data
    scale = rel * np.max(np.abs(data))
    if scale == 0.0:
        return
    data[:] = data + scale * rng.standard_normal(size=data.shape)


def scalar_from_checkpoint(control_fn: fd.Function) -> float:
    """Safely retrieve the last checkpointed scalar value of a Real-space control."""
    try:
        chk = control_fn.block_variable.checkpoint  # type: ignore[attr-defined]
        if hasattr(chk, "dat"):
            return float(chk.dat.data_ro[0])
        return float(chk)
    except Exception:
        return real_value(control_fn)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    if SUPPRESS_SOLVER_PRINTS:
        try:
            mod = importlib.import_module("firn.solvers.firn_solver")
            mod.print = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        except Exception:
            pass

    mesh = build_stretched_mesh(H0, nz, mesh_stretch_p)
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)

    surface_id = 2

    # Build sparse observation mesh/space
    obs_mesh, Qobs, dx_obs = build_observation_point_set(mesh, H0, OBS_DEPTHS_M)
    N_obs = len(OBS_DEPTHS_M)

    dt_seconds = float(dt_days * 86400.0)

    # ------------------------------------------------------------------
    # 1) Reference twin: generate sparse synthetic observations
    # ------------------------------------------------------------------
    rng = np.random.default_rng(1234)

    accum_ref = make_real(R, accum_m_iceeq_per_yr, "accum_ref")
    rho_s_ref = make_real(R, rho_surface, "rho_s_ref")
    Ts_ref = make_real(R, Ts_K, "Ts_ref")
    dt_ref = make_real(R, dt_seconds, "dt_ref")
    kg_ref = make_real(R, kg_true, "kg_ref")

    with stop_annotating():
        params_ref = FirnParameters(kg=kg_ref)
        model_ref = FirnModel(params_ref)
        solver_ref = FirnColumnSolver(model_ref)

        H_ref, rho_ref, w_ref, H_init_ref = initial_conditions(
            V,
            params_ref,
            Ts_ref,
            rho_surf_val=float(rho_surface),
            accum_val=float(accum_m_iceeq_per_yr),
            H0=H0,
        )
        Hs_bc_ref = make_real(R, H_init_ref, "Hs_bc_ref")
        bcs_ref = make_bcs(V, params_ref, accum_ref, rho_s_ref, Hs_bc_ref, surface_id=surface_id)

        # Reusable point-space work function
        w_pts_ref = fd.Function(Qobs, name="w_pts_ref")

        w_obs: list[fd.Function] = []
        for k in range(nsteps):
            H_ref, rho_ref, w_ref = solver_ref.prognostic_solve(
                enthalpy=H_ref,
                density=rho_ref,
                firn_velocity=w_ref,
                dt=dt_ref,
                accumulation=accum_ref,
                surface_density=rho_s_ref,
                boundary_conditions=bcs_ref,
                surface_temperature=Ts_ref,
                enthalpy_bc_constant=Hs_bc_ref,
            )

            # Interpolate model velocity onto the point cloud
            w_pts_ref.interpolate(w_ref)
            wk = w_pts_ref.copy(deepcopy=True)
            apply_relative_noise_point_values(wk, NOISE_REL, rng)
            w_obs.append(wk)

        # Observation norm for scaling
        obs_norm = 0.0
        for k in range(skip_steps_in_objective, nsteps):
            obs_norm += float(fd.assemble((w_obs[k] ** 2) * dx_obs))
        obs_norm = float(obs_norm) + 1.0e-30

    # ------------------------------------------------------------------
    # 2) Clear tape and resume annotation
    # ------------------------------------------------------------------
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()

    # ------------------------------------------------------------------
    # 3) Inversion: control m = log(kg)
    # ------------------------------------------------------------------
    accum = make_real(R, accum_m_iceeq_per_yr, "accum")
    rho_s = make_real(R, rho_surface, "rho_s")
    Ts = make_real(R, Ts_K, "Ts")
    dt = make_real(R, dt_seconds, "dt")

    m = make_real(R, math.log(kg_initial_guess), "logkg")
    kg_expr = fd.exp(m)

    params = FirnParameters(kg=kg_expr)
    model = FirnModel(params)
    solver = FirnColumnSolver(model)

    H, rho, w, H_init = initial_conditions(
        V,
        params,
        Ts,
        rho_surf_val=float(rho_surface),
        accum_val=float(accum_m_iceeq_per_yr),
        H0=H0,
    )
    Hs_bc = make_real(R, H_init, "Hs_bc")
    bcs = make_bcs(V, params, accum, rho_s, Hs_bc, surface_id=surface_id)

    control = Control(m)

    # Reusable interpolant onto point-set
    w_pts = fd.Function(Qobs, name="w_pts")

    # ------------------------------------------------------------------
    # 4) Time-dependent objective (sparse, scaled)
    # ------------------------------------------------------------------
    J_raw = 0.0
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

        if k < skip_steps_in_objective:
            continue

        w_pts.interpolate(w)
        # Pointwise misfit summed over the point cloud
        J_raw += 0.5 * fd.assemble((w_pts - w_obs[k]) ** 2 * dx_obs) / float(N_obs)

    J = J_raw / obs_norm
    rf = ReducedFunctional(J, control)
    pause_annotation()

    # ------------------------------------------------------------------
    # 5) Sanity checks
    # ------------------------------------------------------------------
    m_true = make_real(R, math.log(kg_true), "m_true")
    m_lo = make_real(R, math.log(kg_lower), "m_lo")
    m_hi = make_real(R, math.log(kg_upper), "m_hi")

    print("\nObjective sanity check (smaller is better):")
    print(f"  J(initial guess) = {float(rf(m)):.6e}  at kg={kg_initial_guess:.3e}")
    print(f"  J(true kg)       = {float(rf(m_true)):.6e}  at kg={kg_true:.3e}")
    print(f"  J(lower bound)   = {float(rf(m_lo)):.6e}  at kg={kg_lower:.3e}")
    print(f"  J(upper bound)   = {float(rf(m_hi)):.6e}  at kg={kg_upper:.3e}\n")

    m.assign(math.log(kg_initial_guess))
    g = rf.derivative()
    g0 = float(g.dat.data_ro[0]) if hasattr(g, "dat") else float(g)
    print(f"dJ/d(log kg) at initial guess: {g0:.6e}")

    # Taylor test: use a small log-perturbation direction
    dm = make_real(R, 0.01, "dm")
    print("\nRunning Taylor test for dJ/d(log kg) ...")
    rate = taylor_test(rf, m, dm)
    print(f"  Taylor test convergence rate: {rate}")

    # ------------------------------------------------------------------
    # 6) Optimisation
    # ------------------------------------------------------------------
    J_hist: list[float] = []
    kg_hist: list[float] = []

    def record_eval(Jval, _mval):
        Jf = float(Jval)
        m_now = scalar_from_checkpoint(m)
        kg_now = float(math.exp(m_now))
        J_hist.append(Jf)
        kg_hist.append(kg_now)
        print(f"  J = {Jf:.6e} ; kg = {kg_now:.6e}")

    rf.eval_cb_post = record_eval

    m_lb = make_real(R, math.log(kg_lower), "m_lb")
    m_ub = make_real(R, math.log(kg_upper), "m_ub")

    print("\nStarting optimisation...")
    print(f"  true kg          = {kg_true:.6e}")
    print(f"  initial guess kg = {kg_initial_guess:.6e}")
    print(f"  bounds           = [{kg_lower:.2e}, {kg_upper:.2e}]")
    print(f"  obs depths (m)   = {list(OBS_DEPTHS_M)}")

    m_opt = minimize(
        rf,
        method="L-BFGS-B",
        bounds=(m_lb, m_ub),
        options={"maxiter": 50, "gtol": 1.0e-12},
    )

    if hasattr(m_opt, "dat"):
        kg_opt = float(math.exp(float(m_opt.dat.data_ro[0])))
    else:
        kg_opt = float(math.exp(float(m_opt)))

    print("\nDone.")
    print(f"Recovered kg: {kg_opt:.6e}")

    with open("kg_inversion_history.txt", "w") as f:
        f.write("iter,J,kg\n")
        for i, (Jv, kv) in enumerate(zip(J_hist, kg_hist)):
            f.write(f"{i},{Jv},{kv}\n")


if __name__ == "__main__":
    main()
