"""Age tracing in the firn column.

The age equation is a pure material derivative, D(age)/Dt = 1: a particle ages
at one second per second. Starting the whole column at age 0 turns that into an
exact, mesh-independent check -- material deep enough not to have been advected
out must carry ``age == elapsed``, to round-off, at any resolution.

The former script only wrote a plot and called ``plt.show()`` at module level,
so importing it blocked the test session forever.
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
# nz=80 is the original script's resolution. The surface Galerkin overshoot
# documented in test_age_is_bounded_by_the_elapsed_time falls off steeply with
# refinement (1.4e-2 yr at nz=40, 5.3e-4 at nz=80, 6.5e-10 at nz=160), so this
# is the cheapest resolution at which the bound below is comfortably tight.
NZ = 80
T_MEAN = 268.0
T_AMP = 7.0
DT_SECONDS = 60.0 * 86400.0
NSTEPS = 61  # ~10 years


def _surface_T(t_seconds: float) -> float:
    return T_MEAN + T_AMP * sin(2.0 * pi * (t_seconds / float(year)))


def _run_age_column(nz: int = NZ, nsteps: int = NSTEPS):
    """Advance a firn column with age tracing; return depth, age (yr), elapsed."""
    mesh = build_stretched_mesh(H0, nz, 3.0)
    V = fd.FunctionSpace(mesh, "CG", 1)

    H = fd.Function(V, name="enthalpy")
    rho = fd.Function(V, name="density")
    w = fd.Function(V, name="firn_velocity")
    age = fd.Function(V, name="age")

    params = FirnParameters()
    solver = FirnColumnSolver(FirnModel(params))

    Ts = fd.Constant(_surface_T(0.0))
    H_init = params.c_i * (float(Ts) - params.T_ref)

    H.project(fd.Constant(H_init))
    rho.project(fd.Constant(300.0))
    w.project(fd.Constant(0.0))
    age.project(fd.Constant(0.0))

    accum = fd.Constant(0.3)
    rho_s = fd.Constant(300.0)
    Hs_bc = fd.Constant(H_init)

    surface_id = 2
    bcs = [
        fd.DirichletBC(V, Hs_bc, surface_id),
        fd.DirichletBC(V, rho_s, surface_id),
        fd.DirichletBC(V, -accum * params.rho_i / rho_s / year, surface_id),
    ]
    bc_age = fd.DirichletBC(V, fd.Constant(0.0), surface_id)

    dt = fd.Constant(DT_SECONDS)
    t = 0.0
    for _ in range(nsteps):
        Ts.assign(_surface_T(t))
        Hs_bc.assign(params.c_i * (float(Ts) - params.T_ref))

        H, rho, w, age = solver.prognostic_solve(
            enthalpy=H,
            density=rho,
            firn_velocity=w,
            dt=dt,
            accumulation=accum,
            surface_density=rho_s,
            boundary_conditions=bcs,
            age=age,
            age_boundary_condition=bc_age,
        )
        t += float(dt)

    depth = depth_coordinates(V, H0)
    age_yr = age.dat.data_ro.copy() / float(year)
    order = np.argsort(depth)

    return depth[order], age_yr[order], t / float(year)


@pytest.fixture(scope="module")
def age_column(firedrake):
    """One column run shared by every assertion in this module."""
    return _run_age_column()


def test_surface_age_is_zero(age_column):
    """The surface Dirichlet BC pins freshly deposited firn at age zero."""
    _depth, age_yr, _elapsed = age_column
    assert age_yr[0] == pytest.approx(0.0, abs=1e-12)


def test_deep_age_equals_elapsed_time(age_column):
    """D(age)/Dt = 1, so undisturbed deep material is exactly as old as the run.

    Nothing enters through the base and the column starts at age 0, so the
    deepest firn has simply aged for the whole integration. This is exact and
    independent of mesh resolution.
    """
    _depth, age_yr, elapsed = age_column
    assert age_yr[-1] == pytest.approx(elapsed, rel=1e-6)


def test_age_is_bounded_by_the_elapsed_time(age_column):
    """Age is non-negative and cannot exceed the run length.

    The upper bound carries a small allowance: unstabilised CG1 advection
    overshoots slightly where the age gradient is steepest near the surface.
    It is a discretisation artifact, not a modelling error -- it converges away
    under refinement (1.4e-2 yr at nz=40, 5.3e-4 at nz=80, 6.5e-10 at nz=160),
    while the deep-age law above stays exact at every resolution.
    """
    _depth, age_yr, elapsed = age_column

    assert np.all(np.isfinite(age_yr))
    assert age_yr.min() >= -1e-9
    assert age_yr.max() <= elapsed * (1.0 + 1e-3)


def test_age_increases_with_depth(age_column):
    """Deeper firn is older, up to the same near-surface overshoot."""
    _depth, age_yr, elapsed = age_column
    assert np.all(np.diff(age_yr) > -1e-3 * elapsed)
