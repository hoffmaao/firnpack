"""Coupled firn + Richards aquifer column (1D, confined)."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import firedrake as fd
except Exception:  # pragma: no cover
    fd = None

pytestmark = pytest.mark.skipif(fd is None, reason="firedrake not available")

from firnpack.aquifer import AQUIFER_SITES, AquiferSite, run_aquifer_column


SHORT = dict(H0=40.0, NZ=60, spinup_years=10.0, run_years=1.0,
             dt_days=6.0, save_every_days=30.0, verbose=False)


@pytest.fixture(scope="module")
def base_run():
    return run_aquifer_column(site=AQUIFER_SITES[0], **SHORT)


def test_melt_actually_enters_the_column(base_run):
    """Regression: the surface flux must not be frozen at its initial value.

    The Richards variational problem is assembled once and holds the objects it
    was given, so a plain float boundary flux is baked in permanently. Runs
    start in midwinter, when melt is zero, so that silently produced a column
    no water ever entered - and a confident, wrong "no aquifer" answer. The
    driver now passes an ``fd.Constant`` and assigns into it.
    """
    assert base_run["melt_cum_kg_m2"][-1] > 0.0
    gained = base_run["storage_kg_m2"][-1] - base_run["storage0_kg_m2"]
    refroze = base_run["refreeze_cum_kg_m2"][-1]
    assert gained + refroze > 0.5 * base_run["melt_cum_kg_m2"][-1]


def test_water_budget_closes(base_run):
    """melt in = refrozen + change in storage + elastic storage.

    Tight because the phase change is a sink inside the Richards residual, not
    a post-solve state edit: the mixed form accounts for it exactly. The
    earlier state-edit version leaked a few percent, which is what this bound
    would have caught.
    """
    melt = base_run["melt_cum_kg_m2"][-1]
    assert melt > 0.0
    assert abs(base_run["budget_residual_kg_m2"]) < 1e-4 * melt


def test_enthalpy_projection_between_cg_and_dg_is_negligible(base_run):
    """The CG1/DG1 round trip must not leak energy worth caring about."""
    refroze = base_run["refreeze_cum_kg_m2"][-1]
    latent = 3.34e5 * max(refroze, 1e-9)
    assert base_run["energy_projection_slip_J_m2"] < 1e-3 * latent


def test_no_melt_means_no_liquid_water(base_run):
    """A site with zero melt must stay dry - the null case."""
    dry_site = AquiferSite("nomelt", T_mean_C=-7.0, T_amp_C=11.0,
                           accum_m_ie_yr=1.5, melt_m_we_yr=0.0)
    res = run_aquifer_column(site=dry_site, **SHORT)
    assert res["melt_cum_kg_m2"][-1] == 0.0
    # only the trace the retention curve cannot dry below
    assert res["storage_kg_m2"][-1] <= res["storage0_kg_m2"] * 1.05
    assert not res["persists"]


def test_melt_season_shape_integrates_to_the_annual_total():
    """The prescribed melt curve must deliver exactly the stated total."""
    site = AQUIFER_SITES[0]
    ts = np.linspace(0.0, 1.0, 20001)
    rates = np.array([site.melt_flux_m_s(t) for t in ts])
    total_m = np.trapezoid(rates, ts) * 365.0 * 24 * 3600
    assert total_m == pytest.approx(site.melt_m_we_yr, rel=1e-3)


def test_melt_is_independent_of_mean_temperature():
    """Cold and base cases must get the same melt, or the contrast is confounded.

    Tying melt to positive degree days couples the two controls: dropping the
    mean temperature to test cold content also destroys the melt season, and
    below about -11 C here the air never reaches freezing and the melt vanishes
    entirely - answering "what if there were no melt" instead.
    """
    base = next(s for s in AQUIFER_SITES if s.name == "se_greenland")
    cold = next(s for s in AQUIFER_SITES if s.name == "cold")
    assert cold.melt_m_we_yr == base.melt_m_we_yr
    ts = np.linspace(0.0, 1.0, 2001)
    a = np.array([base.melt_flux_m_s(t) for t in ts])
    b = np.array([cold.melt_flux_m_s(t) for t in ts])
    assert np.allclose(a, b)
    assert cold.T_mean_C < base.T_mean_C


def test_refreezing_fills_pores_instead_of_thickening_the_column():
    """Refrozen meltwater is matrix ice: it raises density, not the column.

    Meyer & Hewitt's ice equation is D(rho)/Dt + rho dw/dz = m. The source
    was originally applied in the velocity integration only, with the
    compaction rate standing in for D(rho)/Dt, which solves
    rho dw/dz = m - compaction: every refrozen kilogram *stretched* the column
    and none of it densified the firn. Under the ERA5 forcing that put
    close-off at 55 m against an observed aquifer base of 27.7 m, so there was
    no floor for water to pond on. Here one step with a uniform source m must
    (a) add exactly int(m) dt of mass to the column and (b) leave the velocity
    field where the dry step put it.
    """
    from firnpack.constants import year
    from firnpack.models.firn import FirnModel, FirnParameters
    from firnpack.solvers.firn_solver import FirnColumnSolver

    H0, NZ = 40.0, 80
    mesh = fd.IntervalMesh(NZ, 0.0, H0)
    V = fd.FunctionSpace(mesh, "CG", 1)
    params = FirnParameters()
    z = fd.SpatialCoordinate(mesh)[0]
    depth = H0 - z
    H_val = params.c_i * (263.0 - params.T_ref)
    rho_init = 350.0 + (params.rho_i - 350.0) * (1.0 - fd.exp(-depth / 25.0))

    def step(m):
        H = fd.Function(V).interpolate(fd.Constant(H_val))
        rho = fd.Function(V).interpolate(rho_init)
        w = fd.Function(V).interpolate(fd.Constant(0.0))
        accum, rho_s, Hs = fd.Constant(1.0), fd.Constant(350.0), fd.Constant(H_val)
        bcs = [fd.DirichletBC(V, Hs, 2), fd.DirichletBC(V, rho_s, 2),
               fd.DirichletBC(V, -accum * params.rho_i / rho_s / year, 2)]
        solver = FirnColumnSolver(FirnModel(params))
        dt = fd.Constant(5.0 * 86400.0)
        H, rho, w = solver.prognostic_solve(
            enthalpy=H, density=rho, firn_velocity=w, dt=dt,
            accumulation=accum, surface_density=rho_s,
            boundary_conditions=bcs, refreezing=m)
        return rho, w, float(dt)

    m = 2.0e-6                                   # kg m^-3 s^-1, ~63 kg/m^3/yr
    rho_dry, w_dry, dt = step(None)
    rho_wet, w_wet, _ = step(fd.Function(V).interpolate(fd.Constant(m)))

    gained = fd.assemble((rho_wet - rho_dry) * fd.dx)
    expected = m * dt * H0
    # advection sees the slightly denser field, so tolerance is a few 1e-3
    assert abs(gained - expected) < 5e-3 * expected

    # The velocity is the same integration of -compaction/rho from the
    # surface; refreezing must not have shown up in it. What the old
    # formulation added was int(m / rho) dz of upward stretching - about a
    # fifth of the column velocity here. What remains is the compaction rate
    # re-evaluated on a density m*dt (~0.9 kg/m^3) higher, a few 1e-3.
    wd, ww = w_dry.dat.data_ro, w_wet.dat.data_ro
    stretch = fd.assemble(fd.Constant(m) / rho_dry * fd.dx)
    assert stretch > 0.1 * np.max(np.abs(wd))          # the bug was not subtle
    assert np.max(np.abs(ww - wd)) < 0.05 * stretch
    assert np.max(np.abs(ww - wd)) < 1e-2 * np.max(np.abs(wd))
