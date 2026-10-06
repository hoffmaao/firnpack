"""Adjoint correctness and kg recovery from dense velocity observations.

A twin experiment: generate synthetic w(t, z) from a known ``kg``, then check
that (a) the pyadjoint gradient through the time loop is correct, and (b)
L-BFGS-B recovers the true ``kg``.

The two claims cost very different amounts. Gradient correctness is
*resolution-independent* -- a Taylor test on a coarse mesh over a few steps
exercises exactly the same adjoint code path as a fine one -- so it runs in the
default tier. Parameter recovery genuinely needs the long integration to make
``kg`` identifiable, so it is marked slow.

Surface w is not usable as an observation: it is a Dirichlet BC derived from
accumulation and rho_s, so the misfit is taken over the interior profile.
"""

from __future__ import annotations

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

# Fast tier: enough to exercise the adjoint, small enough to run in seconds.
FAST_NZ = 20
FAST_NSTEPS = 8

# Slow tier: the original script's fidelity, where kg is identifiable.
FULL_NZ = 80
FULL_NSTEPS = 60


def _initial_conditions(V, params, Ts):
    """Create and initialise H, rho, w for a forward run."""
    H = fd.Function(V, name="enthalpy")
    rho = fd.Function(V, name="density")
    w = fd.Function(V, name="firn_velocity")

    H_init = params.c_i * (real_value(Ts) - params.T_ref)

    H.assign(H_init)
    rho.assign(RHO_SURFACE)
    w.assign(0.0)

    return H, rho, w, float(H_init)


def _build_reduced_functional(nz: int, nsteps: int):
    """Tape the kg->misfit map against a synthetic twin.

    Returns the reduced functional and its control. The reference twin runs
    under ``stop_annotating`` so only the second forward lands on the tape.
    """
    mesh = build_stretched_mesh(H0, nz, MESH_STRETCH_P)
    V = fd.FunctionSpace(mesh, "CG", 1)
    # pyadjoint needs mesh-attached scalars: Real-space Functions, not Constants.
    R = fd.FunctionSpace(mesh, "R", 0)
    dx = fd.dx(domain=mesh)

    surface_id = 2

    accum = make_real(R, ACCUM_M_ICEEQ_PER_YR, "accum")
    rho_s = make_real(R, RHO_SURFACE, "rho_s")
    Ts = make_real(R, TS_K, "Ts")
    dt = make_real(R, DT_DAYS * 86400.0, "dt")

    with stop_annotating():
        params_ref = FirnParameters(kg=KG_TRUE)
        solver_ref = FirnColumnSolver(FirnModel(params_ref))

        H_ref, rho_ref, w_ref, H_init_ref = _initial_conditions(V, params_ref, Ts)
        Hs_bc_ref = make_real(R, H_init_ref, "Hs_bc_ref")
        bcs_ref = make_bcs(V, params_ref, accum, rho_s, Hs_bc_ref, surface_id)

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
            w_obs.append(w_ref.copy(deepcopy=True))

        # w is O(1e-8) m/s, so the raw misfit is O(1e-12). L-BFGS-B tests ftol
        # against max(|J|, 1), so an unscaled J looks converged at step one.
        # Normalising to O(1) is what makes the optimiser actually move.
        obs_norm = sum(float(fd.assemble((wk**2) * dx)) for wk in w_obs) + 1.0e-30

    continue_annotation()

    kg = make_real(R, KG_INITIAL_GUESS, "kg")
    params = FirnParameters(kg=kg)
    solver = FirnColumnSolver(FirnModel(params))

    H, rho, w, H_init = _initial_conditions(V, params, Ts)
    Hs_bc = make_real(R, H_init, "Hs_bc")
    bcs = make_bcs(V, params, accum, rho_s, Hs_bc, surface_id)

    control = Control(kg)

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
        J += 0.5 * fd.assemble((w - w_obs[k]) ** 2 * dx)

    rf = ReducedFunctional(J / obs_norm, control)
    pause_annotation()

    return rf, kg, R


def test_gradient_passes_taylor_test(firedrake, adjoint_tape):
    """dJ/dkg from the tape matches finite differences at second order.

    pyadjoint's first-order Taylor remainder converges at rate 2 when the
    gradient is right, and at rate 1 when it is wrong, so the 1.9 threshold
    cleanly separates the two.
    """
    rf, kg, R = _build_reduced_functional(FAST_NZ, FAST_NSTEPS)

    # The direction must be scaled to the control. taylor_test perturbs by
    # eps*dkg with eps=0.01, so a unit direction would drive kg to ~0.01 and
    # diverge the solve; scaling by kg makes it a 1% relative step.
    dkg = make_real(R, KG_INITIAL_GUESS, "dkg")
    rate = taylor_test(rf, kg, dkg)

    assert rate > 1.9


def test_true_kg_beats_the_initial_guess(firedrake, adjoint_tape):
    """The objective actually prefers the truth: J(kg_true) < J(kg_guess).

    Without noise the twin is exact, so J at the truth should be ~0. If this
    fails, recovery cannot work and the misfit itself is misdefined.
    """
    rf, kg, R = _build_reduced_functional(FAST_NZ, FAST_NSTEPS)

    J_guess = float(rf(make_real(R, KG_INITIAL_GUESS, "kg_guess")))
    J_true = float(rf(make_real(R, KG_TRUE, "kg_true")))

    assert J_true < J_guess
    assert J_true == pytest.approx(0.0, abs=1e-12)


@pytest.mark.slow
def test_kg_is_recovered_from_dense_velocity_observations(
    firedrake, adjoint_tape
):
    """L-BFGS-B recovers kg_true from noise-free dense w observations."""
    rf, kg, R = _build_reduced_functional(FULL_NZ, FULL_NSTEPS)

    # kg ~ 1e-7 makes dJ/dkg ~ 1e7, and the default ftol/gtol are absolute, so
    # the defaults declare convergence before taking a real step.
    kg_opt = minimize(
        rf,
        method="L-BFGS-B",
        bounds=(make_real(R, KG_LOWER, "kg_lb"), make_real(R, KG_UPPER, "kg_ub")),
        options={"maxiter": 25, "ftol": 1.0e-14, "gtol": 1.0e-12},
    )

    kg_recovered = float(kg_opt.dat.data_ro[0]) if hasattr(kg_opt, "dat") else float(kg_opt)
    assert kg_recovered == pytest.approx(KG_TRUE, rel=5e-2)
