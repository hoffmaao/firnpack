"""Verification of the enthalpy solver against exact diffusion behaviour.

Three claims, each with an analytic reference:

1. ``enthalpy_from_T`` and ``temperature_from_enthalpy`` are exact inverses.
2. With ``w = 0`` and constant kappa, the linear profile is the exact steady
   state of pure diffusion, so one step must leave it invariant.
3. A ``sin(pi z / H0)`` perturbation of that steady state is the first
   eigenmode of the Laplacian on [0, H0] with homogeneous Dirichlet ends, so
   it must decay as ``exp(-kappa (pi/H0)^2 t)``.

Claim 3 is the sharp one: it pins the decay *rate*, so it fails if the
diffusivity is not kappa or the operator is not the Laplacian. The former
script only printed "should decrease", which passes for any decaying operator
regardless of rate.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import firedrake as fd
except Exception:  # pragma: no cover
    fd = None

if fd is not None:
    from firnpack.models.firn import FirnModel, FirnParameters


# Column and forcing shared by the diffusion tests.
H0 = 100.0  # m
NZ = 200
KAPPA = 1.0e-6  # m^2/s, order of ice diffusivity
TS_TOP = 250.0  # K
TS_BOT = 260.0  # K
DT_SECONDS = 30.0 * 86400.0
NSTEPS = 24  # 2 years at 30-day steps


def enthalpy_from_T(params, T_K: float) -> float:
    """H = c_i (T - T_ref), in J/kg."""
    return float(params.c_i) * (float(T_K) - float(params.T_ref))


def _constant_thermal_diffusivity(model, kappa_const) -> None:
    """Override model.thermal_diffusivity to return a constant kappa."""
    model.thermal_diffusivity = lambda _H, _rho: kappa_const


def _build_diffusion_solver():
    """A pure-diffusion column (w=0, constant kappa) started from steady state.

    Returns the solver plus the fields and geometry the tests measure against.
    """
    params = FirnParameters()
    model = FirnModel(params)

    mesh = fd.IntervalMesh(NZ, 0.0, H0)
    V = fd.FunctionSpace(mesh, "CG", 1)

    w = fd.Function(V, name="w").assign(0.0)
    rho = fd.Function(V, name="rho").assign(500.0)

    H_top = enthalpy_from_T(params, TS_TOP)
    H_bot = enthalpy_from_T(params, TS_BOT)

    R = fd.FunctionSpace(mesh, "R", 0)
    kappa_const = fd.Function(R, name="kappa").assign(KAPPA)
    _constant_thermal_diffusivity(model, kappa_const)

    x = fd.SpatialCoordinate(mesh)[0]
    H = fd.Function(V, name="H")
    H_old = fd.Function(V, name="H_old")

    # Exact steady state of pure diffusion with Dirichlet ends: linear in z.
    H.interpolate(H_bot + (H_top - H_bot) * (x / H0))
    H_old.assign(H)

    dt = fd.Constant(DT_SECONDS)
    psi = fd.TestFunction(V)
    F_H = model.enthalpy_form(H, H_old, rho, w, psi, dt)

    problem = fd.NonlinearVariationalProblem(
        F_H, H, bcs=[fd.DirichletBC(V, fd.Constant(H_bot), 1),
                     fd.DirichletBC(V, fd.Constant(H_top), 2)],
        J=fd.derivative(F_H, H),
    )
    solver = fd.NonlinearVariationalSolver(
        problem,
        solver_parameters={
            "snes_type": "newtonls",
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    return solver, model, params, V, x, H, H_old, H_top, H_bot


@pytest.mark.parametrize("T_test", [230.0, 250.0, 273.15])
def test_enthalpy_temperature_roundtrip_is_exact(T_test):
    """T -> H -> T recovers the input to round-off."""
    if fd is None:
        pytest.skip("firedrake not available")

    params = FirnParameters()
    model = FirnModel(params)

    H_test = enthalpy_from_T(params, T_test)
    T_roundtrip = float(model.temperature_from_enthalpy(H_test))

    assert T_roundtrip == pytest.approx(T_test, rel=1e-12)


def test_linear_profile_is_invariant_under_one_step():
    """The linear profile is the exact steady state, so a step must not move it.

    A failure here is O(H_top - H_bot) ~ 2e4 J/kg, so the tolerance sits ten
    orders of magnitude below any real defect while staying clear of the LU
    solve's round-off.
    """
    if fd is None:
        pytest.skip("firedrake not available")

    solver, _model, _params, V, _x, H, H_old, _H_top, _H_bot = _build_diffusion_solver()

    solver.solve()

    err_inf = float(np.abs(fd.Function(V).interpolate(H - H_old).dat.data_ro).max())
    assert err_inf < 1.0e-6


def test_first_eigenmode_decays_at_the_analytic_rate():
    """sin(pi z/H0) must decay as exp(-kappa (pi/H0)^2 t).

    The rate is recovered by a log-linear fit over every step rather than from
    the endpoint ratio. Over this window the mode decays only ~6%, so an
    endpoint ratio is nearly blind: a 10% error in kappa moves it by 0.6%,
    which no usable tolerance would catch. The fitted slope, by contrast, is
    directly proportional to kappa, so the same error shows up as a 10% miss.

    Crank-Nicolson (theta_H=0.5) is second order in dt and CG1 at dz=0.5 m
    resolves the first mode to ~1e-5 relative, so both discretisation errors
    sit far inside the 2% band.
    """
    if fd is None:
        pytest.skip("firedrake not available")

    solver, model, _params, V, x, H, H_old, H_top, H_bot = _build_diffusion_solver()

    solver.solve()  # settle onto the discrete steady state
    H_old.assign(H)

    # Perturb by the first eigenmode. It vanishes at both ends, so it is
    # compatible with the Dirichlet BCs and excites exactly one mode.
    H.interpolate(H + 0.5 * (H_top - H_bot) * fd.sin(np.pi * x / H0))
    H_old.assign(H)

    baseline = fd.Function(V).interpolate(TS_BOT + (TS_TOP - TS_BOT) * (x / H0))

    amp = []
    for _ in range(NSTEPS):
        solver.solve()
        T = model.temperature_from_enthalpy(H)
        amp.append(float(fd.norm(T - baseline, norm_type="L2")))
        H_old.assign(H)

    amp = np.array(amp)

    assert np.all(np.diff(amp) < 0.0), "perturbation must decay monotonically"

    # amp[k] = A0 exp(-rate t_k), so log(amp) is linear in t with slope -rate.
    t = DT_SECONDS * np.arange(1, NSTEPS + 1)  # amp[0] is already one step in
    slope = np.polyfit(t, np.log(amp), 1)[0]
    fitted_rate = -slope

    expected_rate = KAPPA * (np.pi / H0) ** 2  # s^-1
    assert fitted_rate == pytest.approx(expected_rate, rel=2.0e-2)
