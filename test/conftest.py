"""Shared fixtures and helpers for the firnpack test suite.

Two things here are load-bearing beyond mere convenience:

* ``adjoint_tape`` guarantees ``pause_annotation()`` runs even when a test
  fails. The pyadjoint tape is process-global state, so a test that dies with
  annotation left on silently taps every later test in the session.
* ``quiet_solver`` replaces the hand-rolled monkeypatch the adjoint scripts
  used, which overwrote ``print`` on the solver module and never restored it.

Test modules import firedrake behind a try/except and skip inside the test
rather than at module scope, so collection stays cheap and a missing firedrake
reports as a skip instead of a collection error.
"""

from __future__ import annotations

import importlib

import matplotlib
import numpy as np
import pytest

# Force a non-interactive backend before anything imports pyplot. With DISPLAY
# set, matplotlib picks tkagg, and a plt.show() anywhere in a collected module
# opens a window and blocks the run forever waiting to be closed. This is a
# backstop, not the fix: no test should call show() at all.
matplotlib.use("Agg")

try:
    import firedrake as fd
except Exception:  # pragma: no cover - exercised only without firedrake
    fd = None

if fd is not None:
    from firnpack.constants import year
else:  # pragma: no cover
    year = None


# -----------------------------------------------------------------------------
# Column construction helpers (extracted from the former scripts, which each
# carried a near-identical private copy)
# -----------------------------------------------------------------------------

def build_stretched_mesh(H0: float, nz: int, p: float = 3.0):
    """Interval mesh on [0, H0] with vertical clustering near the surface."""
    mesh = fd.IntervalMesh(nz, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    z_expr = H0 * (1.0 - (1.0 - xi) ** p)

    coord_fs = mesh.coordinates.function_space()
    new_coords = fd.Function(coord_fs).interpolate(fd.as_vector([z_expr]))
    mesh.coordinates.assign(new_coords)
    return mesh


def make_real(R, value: float, name: str):
    f = fd.Function(R, name=name)
    f.assign(float(value))
    return f


def real_value(f) -> float:
    return float(f.dat.data_ro[0])


def make_bcs(V, params, accum, rho_s, Hs_bc, surface_id: int = 2):
    """Dirichlet BCs imposing H, rho and w at the surface."""
    w_surf = -accum * params.rho_i / rho_s / year

    return [
        fd.DirichletBC(V, Hs_bc, surface_id),
        fd.DirichletBC(V, rho_s, surface_id),
        fd.DirichletBC(V, w_surf, surface_id),
    ]


def depth_coordinates(V, H0: float) -> np.ndarray:
    """Depth below surface at each dof, ascending from the surface."""
    x = fd.SpatialCoordinate(V.mesh())[0]
    return fd.Function(V).interpolate(H0 - x).dat.data_ro.copy()


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def firedrake():
    """The firedrake module, or skip the test if it is unavailable."""
    if fd is None:
        pytest.skip("firedrake not available")
    return fd


@pytest.fixture
def adjoint_tape(firedrake):
    """A cleared pyadjoint tape, with annotation guaranteed off afterwards.

    The teardown runs even if the test raises. Without it, a failure inside a
    taped forward leaves annotation on and every subsequent test in the session
    silently records onto a shared tape.
    """
    from firedrake.adjoint import get_working_tape, pause_annotation

    tape = get_working_tape()
    tape.clear_tape()
    try:
        yield tape
    finally:
        pause_annotation()
        tape.clear_tape()


@pytest.fixture
def quiet_solver(monkeypatch):
    """Silence the solver's per-step diagnostic prints, restoring them after."""
    mod = importlib.import_module("firnpack.solvers.firn_solver")
    monkeypatch.setattr(mod, "print", lambda *a, **k: None, raising=False)
