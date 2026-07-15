"""Adjoint correctness and kg recovery from *sparse* point observations.

The dense counterpart lives in ``test_adjoint.py``. What is distinct here, and
what these tests exist to verify, is the extra machinery between the model and
the misfit:

* observations enter through a ``VertexOnlyMesh`` point cloud, so the adjoint
  must also be correct through the point interpolation operator;
* the control is reparametrised as ``m = log(kg)`` to enforce positivity, so
  the gradient additionally traverses the ``exp`` chain rule.

The pattern follows icepack's "Assimilating sparse data" demo, adapted to the
1D firn column.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import firedrake as fd
except Exception:  # pragma: no cover
    fd = None

if fd is not None:
    from firedrake.adjoint import (
        Control,
        ReducedFunctional,
        continue_annotation,
        minimize,
        pause_annotation,
        stop_annotating,
        taylor_test,
    )

    from firnpack.constants import year
    from firnpack.models.firn import FirnModel, FirnParameters
    from firnpack.solvers.firn_solver import FirnColumnSolver

from conftest import build_stretched_mesh, make_bcs, make_real, real_value

H0 = 40.0
MESH_STRETCH_P = 3.0

ACCUM_M_ICEEQ_PER_YR = 0.3
RHO_SURFACE = 300.0
TS_K = 248.0
DT_DAYS = 10.0

KG_TRUE = 2.0e-7
KG_INITIAL_GUESS = 1.0e-7
KG_LOWER = 1.0e-8
KG_UPPER = 1.0e-6

OBS_DEPTHS_M = [2.0, 5.0, 10.0, 20.0, 30.0]

# Fast tier. skip_steps scales with nsteps: it excludes the initial transient
# from the misfit, so it must stay well below nsteps or nothing is measured.
FAST_NZ = 20
FAST_NSTEPS = 12
FAST_SKIP_STEPS = 2

# Slow tier: the original script's fidelity, where kg is identifiable.
FULL_NZ = 80
FULL_NSTEPS = 120
FULL_SKIP_STEPS = 10


def _initial_conditions(V, params, Ts, density_e_folding: float = 15.0):
    """H, rho, w with a realistic exponential density profile and nonzero w."""
    x = fd.SpatialCoordinate(V.mesh())[0]
    depth = H0 - x

    H = fd.Function(V, name="enthalpy")
    rho = fd.Function(V, name="density")
    w = fd.Function(V, name="firn_velocity")

    H_init = params.c_i * (real_value(Ts) - params.T_ref)
    H.assign(H_init)

    rho.interpolate(
        RHO_SURFACE
        + (params.rho_i - RHO_SURFACE) * (1.0 - fd.exp(-depth / density_e_folding))
    )
    w.assign(-ACCUM_M_ICEEQ_PER_YR * params.rho_i / RHO_SURFACE / float(year))

    return H, rho, w, float(H_init)


def _build_observation_point_set(background_mesh):
    """A VertexOnlyMesh at the observation depths, plus its DG0 space."""
    depths = np.array(OBS_DEPTHS_M, dtype=float)
    if np.any(depths <= 0.0):
        raise ValueError("All OBS_DEPTHS_M must be > 0 to avoid the surface BC.")
    if np.any(depths >= H0):
        raise ValueError("All OBS_DEPTHS_M must be < H0.")

    z_pts = (H0 - depths).reshape((-1, 1))
    point_mesh = fd.VertexOnlyMesh(background_mesh, z_pts, missing_points_behaviour="error")
    return fd.FunctionSpace(point_mesh, "DG", 0), fd.dx(domain=point_mesh)


def _build_reduced_functional(nz: int, nsteps: int, skip_steps: int):
    """Tape the log(kg)->sparse-misfit map against a synthetic twin."""
    mesh = build_stretched_mesh(H0, nz, MESH_STRETCH_P)
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)

    Qobs, dx_obs = _build_observation_point_set(mesh)
    n_obs = len(OBS_DEPTHS_M)
    surface_id = 2

    accum = make_real(R, ACCUM_M_ICEEQ_PER_YR, "accum")
    rho_s = make_real(R, RHO_SURFACE, "rho_s")
    Ts = make_real(R, TS_K, "Ts")
    dt = make_real(R, DT_DAYS * 86400.0, "dt")

    with stop_annotating():
        params_ref = FirnParameters(kg=make_real(R, KG_TRUE, "kg_ref"))
        solver_ref = FirnColumnSolver(FirnModel(params_ref))

        H_ref, rho_ref, w_ref, H_init_ref = _initial_conditions(V, params_ref, Ts)
        Hs_bc_ref = make_real(R, H_init_ref, "Hs_bc_ref")
        bcs_ref = make_bcs(V, params_ref, accum, rho_s, Hs_bc_ref, surface_id)

        w_pts_ref = fd.Function(Qobs, name="w_pts_ref")
        w_obs = []
        for _ in range(nsteps):
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
            w_pts_ref.interpolate(w_ref)
            w_obs.append(w_pts_ref.copy(deepcopy=True))

        obs_norm = sum(
            float(fd.assemble((w_obs[k] ** 2) * dx_obs)) for k in range(skip_steps, nsteps)
        ) + 1.0e-30

    continue_annotation()

    m = make_real(R, math.log(KG_INITIAL_GUESS), "logkg")
    params = FirnParameters(kg=fd.exp(m))
    solver = FirnColumnSolver(FirnModel(params))

    H, rho, w, H_init = _initial_conditions(V, params, Ts)
    Hs_bc = make_real(R, H_init, "Hs_bc")
    bcs = make_bcs(V, params, accum, rho_s, Hs_bc, surface_id)

    control = Control(m)
    w_pts = fd.Function(Qobs, name="w_pts")

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
        if k < skip_steps:
            continue

        w_pts.interpolate(w)
        J_raw += 0.5 * fd.assemble((w_pts - w_obs[k]) ** 2 * dx_obs) / float(n_obs)

    rf = ReducedFunctional(J_raw / obs_norm, control)
    pause_annotation()

    return rf, m, R


def test_gradient_through_point_interpolation_passes_taylor_test(
    firedrake, adjoint_tape, quiet_solver
):
    """dJ/d(log kg) is correct through both VertexOnlyMesh and the exp chain.

    The direction is an absolute 0.01 in log space -- a 1% multiplicative step
    on kg. Unlike the dense case it must not be scaled by the control, which is
    O(-16) here, so a control-scaled direction would be enormous.
    """
    rf, m, R = _build_reduced_functional(FAST_NZ, FAST_NSTEPS, FAST_SKIP_STEPS)

    rate = taylor_test(rf, m, make_real(R, 0.01, "dm"))

    assert rate > 1.9


def test_objective_is_minimised_at_the_true_kg(firedrake, adjoint_tape, quiet_solver):
    """J is smallest at the truth and ~0 there, and larger at both bounds.

    Each rf(...) call re-runs the forward and leaves the tape's control at the
    value last passed, so nothing here may assume the control is still at the
    initial guess afterwards.
    """
    rf, m, R = _build_reduced_functional(FAST_NZ, FAST_NSTEPS, FAST_SKIP_STEPS)

    J_guess = float(rf(make_real(R, math.log(KG_INITIAL_GUESS), "m_guess")))
    J_true = float(rf(make_real(R, math.log(KG_TRUE), "m_true")))
    J_lower = float(rf(make_real(R, math.log(KG_LOWER), "m_lo")))
    J_upper = float(rf(make_real(R, math.log(KG_UPPER), "m_hi")))

    assert J_true == pytest.approx(0.0, abs=1e-12)
    assert J_true < J_guess
    assert J_true < J_lower
    assert J_true < J_upper


@pytest.mark.slow
def test_kg_is_recovered_from_five_point_observations(
    firedrake, adjoint_tape, quiet_solver
):
    """L-BFGS-B recovers kg_true from only five point measurements."""
    rf, m, R = _build_reduced_functional(FULL_NZ, FULL_NSTEPS, FULL_SKIP_STEPS)

    m_opt = minimize(
        rf,
        method="L-BFGS-B",
        bounds=(
            make_real(R, math.log(KG_LOWER), "m_lb"),
            make_real(R, math.log(KG_UPPER), "m_ub"),
        ),
        options={"maxiter": 50, "gtol": 1.0e-12},
    )

    m_val = float(m_opt.dat.data_ro[0]) if hasattr(m_opt, "dat") else float(m_opt)
    assert math.exp(m_val) == pytest.approx(KG_TRUE, rel=5e-2)
