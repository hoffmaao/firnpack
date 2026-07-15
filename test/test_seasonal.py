"""Seasonal surface forcing of the firn column.

The physical content is thermal damping: a sinusoidal surface temperature
propagates downward as a decaying wave. For pure conduction the amplitude falls
as ``exp(-z/d)`` with skin depth ``d = sqrt(2 kappa / omega)``, so the tests
below check that the model damps exponentially, at a depth scale consistent
with its own diffusivity.

The former script only wrote plots and called ``plt.show()`` twice at module
level, so importing it blocked the test session forever.
"""

from __future__ import annotations

from math import pi, sin

import numpy as np
import pytest

try:
    import firedrake as fd
except Exception:  # pragma: no cover
    fd = None

if fd is not None:
    from firnpack.constants import year
    from firnpack.models.firn import FirnModel, FirnParameters
    from firnpack.solvers.firn_solver import FirnColumnSolver

from conftest import build_stretched_mesh, depth_coordinates

H0 = 40.0
NZ = 80
T_MEAN = 268.0
T_AMP = 7.0
RHO_SURFACE = 300.0
DT_SECONDS = 10.0 * 86400.0
NSTEPS = 110  # ~3 years: two to shed the initial transient, one to measure

# Depth window for the exponential fit: below the first metre, where the
# near-surface nonlinearity lives, and above the depth at which the signal
# reaches round-off.
FIT_Z_MIN = 5.0
FIT_Z_MAX = 20.0


def _surface_T(t_seconds: float) -> float:
    return T_MEAN + T_AMP * sin(2.0 * pi * (t_seconds / float(year)))


def _run_seasonal_column():
    """Force a column with an annual cycle; return depth, T history, rho, model."""
    mesh = build_stretched_mesh(H0, NZ, 3.0)
    V = fd.FunctionSpace(mesh, "CG", 1)

    H = fd.Function(V, name="enthalpy")
    rho = fd.Function(V, name="density")
    w = fd.Function(V, name="firn_velocity")

    params = FirnParameters()
    model = FirnModel(params)
    solver = FirnColumnSolver(model)

    Ts = fd.Constant(_surface_T(0.0))
    H_init = params.c_i * (float(Ts) - params.T_ref)

    H.project(fd.Constant(H_init))
    rho.project(fd.Constant(RHO_SURFACE))
    w.project(fd.Constant(0.0))

    accum = fd.Constant(0.3)
    rho_s = fd.Constant(RHO_SURFACE)
    Hs_bc = fd.Constant(H_init)

    surface_id = 2
    bcs = [
        fd.DirichletBC(V, Hs_bc, surface_id),
        fd.DirichletBC(V, rho_s, surface_id),
        fd.DirichletBC(V, -accum * params.rho_i / rho_s / year, surface_id),
    ]

    dt = fd.Constant(DT_SECONDS)
    t = 0.0

    depth = depth_coordinates(V, H0)
    order = np.argsort(depth)

    T_history = []
    time_history = []
    for _ in range(NSTEPS):
        Ts.assign(_surface_T(t))
        Hs_bc.assign(params.c_i * (float(Ts) - params.T_ref))

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
        t += float(dt)

        T_now = fd.Function(V).interpolate(model.temperature_from_enthalpy(H))
        T_history.append(T_now.dat.data_ro.copy()[order])
        time_history.append(t / float(year))

    return {
        "depth": depth[order],
        "T": np.array(T_history),
        "time": np.array(time_history),
        "rho": rho.dat.data_ro.copy()[order],
        "model": model,
        "params": params,
        "V": V,
    }


def _seasonal_amplitude(run) -> np.ndarray:
    """Half peak-to-peak temperature range over the final year, per depth."""
    final_year = run["time"] >= run["time"][-1] - 1.0
    T = run["T"][final_year]
    return 0.5 * (T.max(axis=0) - T.min(axis=0))


@pytest.fixture(scope="module")
def seasonal_column(firedrake):
    """One column run shared by every assertion in this module."""
    return _run_seasonal_column()


def test_surface_temperature_tracks_the_forcing(seasonal_column):
    """The surface enthalpy BC reproduces the imposed annual amplitude."""
    amp = _seasonal_amplitude(seasonal_column)
    assert amp[0] == pytest.approx(T_AMP, rel=2e-2)


def test_seasonal_signal_damps_monotonically_with_depth(seasonal_column):
    """A conducted thermal wave loses amplitude at every step downward."""
    amp = _seasonal_amplitude(seasonal_column)
    assert np.all(np.diff(amp) <= 1e-9)


def test_seasonal_signal_is_negligible_at_depth(seasonal_column):
    """By 20 m the annual cycle is well under 1% of the surface amplitude."""
    run = seasonal_column
    amp = _seasonal_amplitude(run)

    i20 = int(np.argmin(np.abs(run["depth"] - 20.0)))
    assert amp[i20] / amp[0] < 1e-2


def test_damping_is_exponential_with_a_physical_skin_depth(seasonal_column):
    """Amplitude decays as exp(-z/d), with d consistent with the model's kappa.

    Pure conduction gives d = sqrt(2 kappa / omega). Downward advection carries
    the wave deeper, so the measured skin depth is expected to *exceed* the
    conduction value rather than match it -- the bounds below are one-sided in
    that sense and are wide enough to absorb kappa varying with density across
    the fit window, while still failing on an order-of-magnitude error.
    """
    run = seasonal_column
    amp = _seasonal_amplitude(run)
    z = run["depth"]

    window = (z >= FIT_Z_MIN) & (z <= FIT_Z_MAX)
    assert window.sum() >= 5, "fit window must contain enough dofs"

    # log(amp) = log(A0) - z/d, so the slope is -1/d.
    slope, intercept = np.polyfit(z[window], np.log(amp[window]), 1)
    fitted_d = -1.0 / slope
    assert fitted_d > 0.0

    # The decay really is exponential: the log-linear fit must be tight.
    predicted = intercept + slope * z[window]
    residual = np.log(amp[window]) - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((np.log(amp[window]) - np.log(amp[window]).mean()) ** 2))
    assert 1.0 - ss_res / ss_tot > 0.99

    # Conduction skin depth from the model's own diffusivity at the mean
    # density over the same window.
    V = run["V"]
    rho_mean = float(run["rho"][window].mean())
    H_ref = fd.Function(V).assign(run["params"].c_i * (T_MEAN - run["params"].T_ref))
    rho_ref = fd.Function(V).assign(rho_mean)
    kappa = float(
        fd.Function(V).interpolate(run["model"].thermal_diffusivity(H_ref, rho_ref)).dat.data_ro[0]
    )
    omega = 2.0 * np.pi / float(year)
    conduction_d = np.sqrt(2.0 * kappa / omega)

    assert 0.8 * conduction_d <= fitted_d <= 2.0 * conduction_d


def test_density_respects_its_surface_bc_and_physical_bounds(seasonal_column):
    """Density starts at the deposited value and never exceeds solid ice."""
    run = seasonal_column
    rho = run["rho"]
    rho_i = float(run["params"].rho_i)

    assert np.all(np.isfinite(rho))
    assert rho[0] == pytest.approx(RHO_SURFACE, rel=1e-6)
    assert rho.min() >= RHO_SURFACE - 1.0
    assert rho.max() <= rho_i + 1e-6
