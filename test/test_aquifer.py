"""Coupled firn + Richards aquifer column (1D, confined)."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import firedrake as fd
except Exception:  # pragma: no cover
    fd = None

pytestmark = pytest.mark.skipif(fd is None, reason="firedrake not available")

from firnpack.aquifer import (
    AQUIFER_SITES, AquiferSite, ReanalysisSite, run_aquifer_column, wet_layer,
)


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


def test_a_sealed_base_keeps_every_kilogram_in_the_column(base_run):
    """The alternative outlet: with no base condition, nothing may leave.

    ``basal_drainage=False`` replaces the advective outlet with a no-flow
    base. It is not what the aquifer experiments use - they let water ride out
    with the compacting firn - but it is the closed-column contrast, and its
    budget has to close without a drainage term at all.
    """
    res = run_aquifer_column(site=AQUIFER_SITES[0], basal_drainage=False,
                             **SHORT)
    melt = res["melt_cum_kg_m2"][-1]
    assert melt > 0.0
    assert res["drained_kg_m2"] == 0.0
    assert abs(res["budget_residual_kg_m2"]) < 1e-4 * melt
    # and the water the open base let go is still here - as liquid or as ice.
    # Comparing liquid storage alone would not show it: the retained water
    # sits against the cold base and most of it refreezes, so the sealed
    # column ends with slightly *less* standing water than the open one while
    # holding more total mass. Water plus refrozen ice is the quantity the
    # sealed base actually conserves, and the elastic (specific-storage) term
    # belongs with them because the sealed column is the one that pressurises.
    # The tolerance absorbs the two runs' budget residuals; folding
    # budget_residual_kg_m2 into the sum instead would make this unfalsifiable,
    # since the driver defines it as melt - refreeze - drain - (storage -
    # storage0) - elastic, and substituting it back leaves melt + storage0 -
    # drain, which the two runs share by construction whatever the water did.
    assert base_run["drained_kg_m2"] > 0.0

    def held(r):
        return (r["storage_kg_m2"][-1] + r["refreeze_cum_kg_m2"][-1]
                + r["elastic_storage_kg_m2"])

    assert held(res) - held(base_run) == pytest.approx(
        base_run["drained_kg_m2"], rel=0.02)


def test_wet_layer_reports_the_thickest_contiguous_zone():
    """Two disjoint wet zones are not one layer spanning both.

    During a melt season a column with a deep aquifer carries a near-surface
    wet layer and the aquifer itself, with dry firn between them. Reporting
    the first and last wet node spans the dry gap as well, and the recorded
    ``wet_top_m``/``wet_bottom_m`` then describe a layer that is not there.
    """
    depth = np.arange(10.0)
    theta = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0, 0.0])
    assert wet_layer(depth, theta, 0.05) == (5.0, 7.0)
    # a single wet node is still a zone, and a dry column has none
    assert wet_layer(depth, np.eye(10)[3], 0.05) == (3.0, 3.0)
    assert all(np.isnan(v) for v in wet_layer(depth, np.zeros(10), 0.05))


def _constant_forcing_csv(path, days, snow_per_day):
    """A minimal ERA5-shaped record: steady weather, steady snowfall."""
    import pandas as pd
    n = int(days)
    hours = 24.0
    pd.DataFrame({
        "time": pd.date_range("2000-01-01", periods=n, freq="D"),
        "sample_hours": np.full(n, hours),
        "t2m": np.full(n, 263.0),
        "ssrd": np.full(n, 100.0 * hours * 3600.0),
        "strd": np.full(n, 250.0 * hours * 3600.0),
        "sp": np.full(n, 8.2e4),
        "d2m": np.full(n, 260.0),
        "u10": np.full(n, 3.0),
        "v10": np.full(n, 0.0),
        "sf": np.full(n, snow_per_day),
    }).to_csv(path, index=False)
    return path


def test_the_trailing_accumulation_window_wraps_instead_of_truncating(tmp_path):
    """Burial must not collapse every time the forcing record repeats.

    Runs longer than the record cycle it, and the trailing-year accumulation
    window straddles the wrap point once per lap. Clamping that window at the
    start of the record returns a partial total over a full-year denominator:
    a month past the wrap it reported a twelfth of the true burial rate, which
    the driver feeds straight into the surface velocity and the densification
    forcing. Snowfall is constant here, so the trailing-year rate is the same
    everywhere and any dip is the truncation.
    """
    csv = _constant_forcing_csv(tmp_path / "forcing.csv", days=3 * 365,
                                snow_per_day=0.004)
    site = ReanalysisSite(csv, 2000, 2002)
    assert site.span_years == pytest.approx(3.0, rel=0.01)

    mid = site.accum_m_ie_yr_at(1.5)
    assert mid == pytest.approx(site.accum_m_ie_yr, rel=1e-6)
    for t in (0.02, 0.5, site.span_years + 0.02, 2.0 * site.span_years + 0.4):
        assert site.accum_m_ie_yr_at(t) == pytest.approx(mid, rel=1e-6)

    # and a window longer than the record still averages the whole of it
    assert site.accum_m_ie_yr_at(0.3, window_yr=2.0 * site.span_years) \
        == pytest.approx(mid, rel=1e-6)


def _synthetic_forcing_csv(path, *, sw_peak):
    """A year of 3-hourly forcing: fixed snowfall, melt set by the shortwave.

    Self-contained rather than reading the tutorial's ERA5 extract, so the
    suite stays independent of case-study data. Only the shortwave differs
    between the two columns the test compares, which is what changes the melt
    while leaving the snowfall identical.
    """
    import pandas as pd

    block_h = 3.0
    block_s = block_h * 3600.0
    t = pd.date_range("2010-01-01", "2010-12-31 21:00", freq="3h")
    day = t.dayofyear.values
    season = np.maximum(np.cos(2.0 * np.pi * (day - 195) / 365.0), 0.0)
    diurnal = np.maximum(np.cos(2.0 * np.pi * (t.hour.values - 13) / 24.0), 0.0)
    sw = sw_peak * season * diurnal                       # W/m2
    df = pd.DataFrame({
        "time": t,
        "sf": 1.4 / len(t),                               # m w.e. per block, ~1.4 m/yr
        "tp": 1.4 / len(t),
        "t2m": 273.15 - 14.0 + 17.0 * season,             # peaks at +3 C
        "d2m": 273.15 - 17.0 + 17.0 * season,
        "sp": 84000.0,
        "ssrd": sw * block_s,                             # accumulated over the block
        "strd": 260.0 * block_s,
        "smlt": 0.0,
        "u10": 4.0,
        "v10": 0.0,
        "fal": 0.6,
        "sample_hours": block_h,
    })
    df.to_csv(path, index=False)
    return path


def test_melt_does_not_add_mass_to_the_column(tmp_path):
    """Total surface mass input is the snowfall, whatever the melt rate.

    Melting adds no mass and removes none: it converts snow that is already
    at the surface into water, which Richards then carries down. So the
    matrix influx (the surface velocity boundary condition, accum * rho_i)
    plus the water influx (melt * rho_w) must equal the gross snowfall,
    independent of how much melts.

    The regression: `accum_m_ie_yr` was built from ERA5 gross snowfall while
    the melt derived from that same snowfall was injected on top of it, so
    the melted fraction entered the column twice and the total mass input
    rose with the melt rate. At the aquifer forcing that was 30-50% more mass
    than the climate delivers, which inflates every aquifer result. Two
    columns differing only in shortwave, and therefore only in melt, pin it:
    their melt totals must differ and their mass inputs must not.
    """
    from firnpack.aquifer import ReanalysisSite
    from firnpack.constants import ice_density as rho_i, water_density as rho_w

    cold = ReanalysisSite(_synthetic_forcing_csv(tmp_path / "cold.csv",
                                                 sw_peak=150.0), 2010, 2010,
                          albedo="era5")
    warm = ReanalysisSite(_synthetic_forcing_csv(tmp_path / "warm.csv",
                                                 sw_peak=400.0), 2010, 2010,
                          albedo="era5")

    # the two really do differ in melt, or the test proves nothing
    assert warm.melt_m_we_yr > cold.melt_m_we_yr + 0.05

    def mass_in(site):
        return site.accum_m_ie_yr * rho_i + site.melt_m_we_yr * rho_w

    assert cold.snowfall_m_ie_yr == pytest.approx(warm.snowfall_m_ie_yr, rel=1e-12)
    snowfall_mass = cold.snowfall_m_ie_yr * rho_i
    assert mass_in(cold) == pytest.approx(snowfall_mass, rel=1e-9)
    assert mass_in(warm) == pytest.approx(snowfall_mass, rel=1e-9)


def test_overburden_never_exceeds_the_weight_of_the_firn_above_it():
    """sigma(z) cannot be larger than integral(rho g dz) above z.

    Mode B carries the overburden prognostically, integrating
    ``dsigma/dt + w dsigma/dz = bdot g`` from sigma = 0 at the surface, so
    along a particle path sigma = bdot g age. That equals the true overburden
    only when ``bdot`` is the same matrix influx that drives the velocity
    field, and hence the age, that it multiplies. Whatever bdot is, though,
    sigma may never exceed the weight actually present above the parcel, and
    that one-sided bound is what this pins.

    The regression: the driver was changed to load the densification with
    GROSS snowfall while the surface velocity used the net matrix influx.
    Every kilogram of melt does stay in the column, so the mass argument was
    right, but sigma = bdot g age then asserts all of it has already refrozen
    ABOVE every parcel - true only below the whole refreezing zone and badly
    wrong near the surface, where densification is fastest. The
    low-accumulation case reached bubble close-off at 1.7 m with 77% of the
    column saturated before the Richards solve failed at year 10.5, and the
    ERA5 columns pressurised the same way. Nothing caught it: the water
    budget still closed exactly, and no test compared the density profile
    with the load that produced it.

    The site must melt, or the test is vacuous - with no melt gross and net
    are the same number and the bug is invisible. Refreezing then puts mass
    at depth that the proxy cannot represent, so the correct (net) loading
    UNDERSTATES sigma and satisfies the bound, while the gross loading
    overshoots it by the ratio of the two rates.
    """
    site = AquiferSite(name="loading_check", T_mean_C=-12.0, T_amp_C=10.0,
                       accum_m_ie_yr=1.0, melt_m_we_yr=0.5,
                       label="melting: overburden must not exceed the weight above")
    r = run_aquifer_column(site=site, H0=40.0, NZ=60, spinup_years=20.0,
                           run_years=1.0, dt_days=6.0, save_every_days=30.0,
                           verbose=False)

    d = np.asarray(r["depth_m"])                  # ascending from the surface
    rho = np.asarray(r["rho_final"])
    sigma = np.asarray(r["sigma_final_Pa"])
    g = 9.81

    weight = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1])
                                              * np.diff(d))]) * g

    deep = d >= 5.0                               # skip the surface layer
    assert weight[deep].max() > 1e4                # there is a real load to compare
    over = sigma[deep] / weight[deep]
    assert over.max() < 1.05, (
        f"overburden exceeds the weight above it by up to "
        f"{100 * (over.max() - 1):.0f}%")

    # and the column must not seal within the first few metres
    sealed = d[rho >= 830.0]
    assert not sealed.size or sealed.min() > 10.0
