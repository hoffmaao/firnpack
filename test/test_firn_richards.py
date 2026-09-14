"""Mixed-form (head-based) Richards percolation in firn.

These are the checks that distinguish a correct variably-saturated solver from
one that merely runs. In particular ``test_hydrostatic_equilibrium_is_exact``
is the sharpest single test of the gravity sign convention and the facet terms:
a column in hydrostatic balance has zero flux everywhere, so any sign error in
the gravity flux, the upwinding, or the SIPG terms shows up immediately as
drift, while a plain "does water go down" test would not notice.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import firedrake as fd
except Exception:  # pragma: no cover
    fd = None

pytestmark = pytest.mark.skipif(fd is None, reason="firedrake not available")

from firnpack.models.firn_richards import (
    FirnRichardsModel, FirnRichardsParameters,
)
from firnpack.solvers.firn_richards_solver import FirnRichardsSolver


H0 = 20.0
NZ = 80
C_I = 2009.0
L_F = 3.34e5
NO_FLOW = {1: {"flux": 0.0}, 2: {"flux": 0.0}}


def _column(rho_val=400.0, T_C=0.0, h_val=-5.0, **kw):
    mesh = fd.IntervalMesh(NZ, 0.0, H0)
    V = fd.FunctionSpace(mesh, "DG", 1)
    model = FirnRichardsModel(FirnRichardsParameters(**kw))
    rho = fd.Function(V).interpolate(fd.Constant(rho_val))
    curves = model.curves(V, rho)
    solver = FirnRichardsSolver(model)
    head = fd.Function(V, name="head").interpolate(fd.Constant(h_val))
    H = fd.Function(V, name="enthalpy").interpolate(fd.Constant(C_I * T_C))
    # One dict, holding Constants: the solver builds its variational problem
    # once and keeps these objects, so a fresh dict per step would be ignored.
    # It refuses one rather than accepting it silently.
    q_top = fd.Constant(0.0)
    bcs = {1: {"flux": fd.Constant(0.0)}, 2: {"flux": q_top}}
    return dict(mesh=mesh, V=V, model=model, rho=rho, curves=curves,
                solver=solver, head=head, H=H, bcs=bcs, q_top=q_top)


def _dxq(mesh):
    """The form's own quadrature.

    Conservation is exact in the rule the residual is assembled with; measuring
    it with the default rule instead reports a spurious ~1e-3 error that is
    quadrature, not physics.
    """
    return fd.dx(domain=mesh, metadata={"quadrature_degree": 3})


def _run(st, *, dt=3600.0, nsteps=100, top_flux=0.0, phase_change=False):
    frozen = 0.0
    dxq = _dxq(st["mesh"])
    st["q_top"].assign(top_flux)
    for _ in range(nsteps):
        st["head"], m = st["solver"].step(
            head=st["head"], enthalpy=st["H"], density=st["rho"],
            curves=st["curves"], dt=dt, bcs=st["bcs"],
            phase_change=phase_change,
        )
        if m is not None:
            frozen += float(fd.assemble(m * dxq)) * dt
    return frozen


# ----------------------------------------------------------------------
def test_hydrostatic_equilibrium_is_exact():
    """h = z_wt - z has zero total head gradient, so nothing may move."""
    st = _column()
    x = fd.SpatialCoordinate(st["mesh"])[0]
    st["head"].interpolate(fd.Constant(5.0) - x)
    before = st["head"].dat.data_ro.copy()

    _run(st, nsteps=20)

    assert np.abs(st["head"].dat.data_ro - before).max() < 1e-9


def test_the_gravity_facet_flux_takes_the_donor_cells_conductivity():
    """Gravity flows down, so the facet flux must carry the upper cell's K.

    Regression. The gravity flux vector is ``-K e_z``, and the numerical flux
    was built from the positive part of ``+K e_z . n``, which picks the cell
    the water arrives in rather than the one it leaves. Every other test here
    runs with ``K`` continuous across facets - uniform density, continuous
    head - where donor and receiver coincide and the error is invisible, so
    this one puts a hundred-fold jump in ``K_s`` on the interior facet.

    Probe: a saturated column at uniform head. The storage terms and every
    diffusive term vanish (``grad h = 0``, ``jump(h) = 0``), the boundaries
    carry zero flux, and summing the residual over a cell's own basis
    functions tests it against the cell indicator, so the gravity volume term
    integrates away too. What is left in each cell is the net gravity flux
    across the interior facet, and its magnitude is whichever conductivity the
    numerical flux chose. Downwinded, it reads 1; donor-cell upwinded, 100.
    """
    K_lower, K_upper = 1.0, 100.0
    mesh = fd.IntervalMesh(2, 0.0, 2.0)
    V = fd.FunctionSpace(mesh, "DG", 1)
    model = FirnRichardsModel(FirnRichardsParameters(K_min=0.0))
    rho = fd.Function(V).interpolate(fd.Constant(400.0))
    curves = model.curves(V, rho)
    x = fd.SpatialCoordinate(mesh)[0]
    # via DG0, so each cell gets one constant: interpolating the jump straight
    # into DG1 would evaluate it at the shared vertex and tilt the lower cell
    step = fd.Function(fd.FunctionSpace(mesh, "DG", 0)).interpolate(
        fd.conditional(x < 1.0, K_lower, K_upper))
    curves.K_s.interpolate(step)

    head = fd.Function(V).interpolate(fd.Constant(0.0))     # saturated, k_r = 1
    theta_old = fd.Function(V).interpolate(curves.moisture_content(head))
    F = model.residual(
        head=head, head_old=head, curves=curves, dt=fd.Constant(1.0),
        bcs={1: {"flux": fd.Constant(0.0)}, 2: {"flux": fd.Constant(0.0)}},
        theta_old=theta_old)
    r = fd.assemble(F).dat.data_ro
    xs = fd.Function(V).interpolate(x).dat.data_ro
    cells = sorted(V.cell_node_map().values, key=lambda c: xs[c].mean())
    lower, upper = (float(r[c].sum()) for c in cells)

    # the lower cell gains water at the upper (donor) cell's conductivity
    assert lower == pytest.approx(-K_upper, rel=1e-10)
    assert upper == pytest.approx(K_upper, rel=1e-10)


def test_a_saturated_head_boundary_admits_water_through_the_surface():
    """A Dirichlet head boundary lets water in, and the amount is bounded.

    What this does NOT pin, despite an earlier docstring here claiming it:
    the donor-cell convention of the boundary's gravity term. Two reasons.
    Placed at the base, as it originally was, the outward normal is -e_z, the
    gravity flux -K e_z points into the domain, and the exterior-conductivity
    term 0.5*(fn_g - |fn_g|) is identically zero for every state the test can
    reach, so the branch was never executed at all. Moved to the surface it
    does execute, but a saturated boundary against firn at h = -5 m drives a
    Nitsche diffusive influx about twelve times the gravity influx, so
    inverting the gravity upwinding would barely move the total and the test
    could not tell. The donor-cell convention is pinned instead by
    test_the_gravity_facet_flux_takes_the_donor_cells_conductivity, which
    assembles the term directly.

    What remains here is still worth keeping: the boundary admits water at
    all, it is not throttled to the dry interior conductivity, and it cannot
    admit more than the pore space can hold. The column is throttled by
    perm_scale so the influx is resolvable; at full firn conductivity a
    saturated surface floods the top cell within two steps and Newton
    refuses it at any step size, which is physical rather than a defect.
    """
    st = _column(h_val=-5.0, perm_scale=1.0e-4)
    st["bcs"] = {1: {"flux": fd.Constant(0.0)}, 2: {"h": fd.Constant(0.0)}}
    st["solver"] = FirnRichardsSolver(st["model"])
    curves, dxq = st["curves"], _dxq(st["mesh"])
    theta0 = fd.assemble(curves.moisture_content(st["head"]) * dxq)

    dt, n = 1.0, 20
    for _ in range(n):
        st["head"], _ = st["solver"].step(
            head=st["head"], enthalpy=st["H"], density=st["rho"],
            curves=curves, dt=dt, bcs=st["bcs"], phase_change=False)

    gained = fd.assemble(curves.moisture_content(st["head"]) * dxq) - theta0
    K_dry = fd.Function(st["V"]).interpolate(
        curves.relative_conductivity(fd.Constant(-5.0))).dat.data_ro.max()
    pore = fd.assemble(curves.theta_s * dxq)
    assert gained > 100.0 * K_dry * dt * n    # not throttled to the dry interior
    assert gained < pore                      # and bounded by the pore space


def test_mass_is_conserved_under_infiltration():
    """Stored water equals what was let in, to round-off.

    Storage is retention *plus* the elastic (specific-storage) water. Leaving
    the elastic part out is what made this budget look like it leaked at the
    1e-3 level when it did not.
    """
    st = _column()
    dxq = _dxq(st["mesh"])
    theta0 = fd.assemble(st["curves"].moisture_content(st["head"]) * dxq)

    q, dt, n = 1.0e-6, 3600.0, 100
    _run(st, dt=dt, nsteps=n, top_flux=q)

    theta1 = fd.assemble(st["curves"].moisture_content(st["head"]) * dxq)
    stored = (theta1 - theta0) + st["solver"].elastic_storage_m
    assert stored == pytest.approx(q * dt * n, rel=1e-6)


def test_a_water_table_forms_and_is_hydrostatic_below_it():
    """The aquifer signature: h = 0 at an interior surface, h = z below it.

    The moisture form cannot produce this - once S hits 1 it has no variable
    left. Recovering hydrostatic head under the table is the statement that
    the saturated zone is a real saturated zone and not a clamp.
    """
    st = _column()
    _run(st, nsteps=300, top_flux=5.0e-6)

    Vc = fd.FunctionSpace(st["mesh"], "CG", 1)
    x = fd.SpatialCoordinate(st["mesh"])[0]
    z = fd.Function(Vc).interpolate(x).dat.data_ro
    hc = fd.Function(Vc).project(st["head"]).dat.data_ro

    saturated = z[hc >= 0.0]
    assert saturated.size > 0, "no saturated zone formed"
    table = saturated.max()
    assert 5.0 < table < H0 - 1.0, f"water table at {table:.2f} m is not interior"
    # below the table the head must be hydrostatic: h(base) = height of water
    assert st["head"].dat.data_ro.max() == pytest.approx(table, abs=0.3)


def test_refreezing_conserves_energy_and_cannot_superheat():
    """Latent heat released equals L times the mass frozen, exactly."""
    st = _column(T_C=-10.0)
    dxq = _dxq(st["mesh"])
    E0 = fd.assemble(st["rho"] * st["H"] * dxq)

    frozen = _run(st, nsteps=100, top_flux=2.0e-6, phase_change=True)

    E1 = fd.assemble(st["rho"] * st["H"] * dxq)
    assert frozen > 0.0
    assert (E1 - E0) == pytest.approx(L_F * frozen, rel=1e-10)
    # H = c_i (T - T_m), so temperate is the ceiling
    assert st["H"].dat.data_ro.max() <= 1e-9


def test_refreezing_closes_the_water_budget():
    """In = frozen + stored, to round-off.

    This is the property the phase change was rewritten for. Editing the state
    after the solve - computing theta, subtracting the frozen water, inverting
    the retention curve - is exact only at nodes, while the budget is an
    integral over cell interiors, and it leaked a few parts per thousand.
    Carrying the freezing as a sink inside the residual makes the mixed form
    account for it exactly instead.
    """
    st = _column(T_C=-10.0)
    dxq = _dxq(st["mesh"])
    theta0 = fd.assemble(st["curves"].moisture_content(st["head"]) * dxq)

    q, dt, n = 2.0e-6, 3600.0, 100
    frozen = _run(st, dt=dt, nsteps=n, top_flux=q, phase_change=True)

    theta1 = fd.assemble(st["curves"].moisture_content(st["head"]) * dxq)
    rho_w = st["model"].params.rho_w
    injected = q * dt * n * rho_w
    stored = ((theta1 - theta0) + st["solver"].elastic_storage_m) * rho_w
    assert frozen > 0.0
    assert injected == pytest.approx(frozen + stored, rel=1e-6)


def test_conservation_holds_across_the_forcing_envelope():
    """Mass and energy close over the range these columns are actually run at.

    Beyond it - about eight times peak Greenland melt onto firn at -20 C or
    colder - the nonlinear solve stops converging at any sub-step. That is a
    robustness limit, not a conservation one, and it is well outside the
    forcing in firnpack.aquifer.
    """
    rho_w, L_f = 1000.0, 3.34e5
    for T_C, q in ((-2.0, 1.5e-7), (-10.0, 5.0e-7), (-20.0, 4.3e-7)):
        st = _column(T_C=T_C)
        dxq = _dxq(st["mesh"])
        theta0 = fd.assemble(st["curves"].moisture_content(st["head"]) * dxq)
        E0 = fd.assemble(st["rho"] * st["H"] * dxq)

        frozen = _run(st, dt=3600.0, nsteps=60, top_flux=q, phase_change=True)

        theta1 = fd.assemble(st["curves"].moisture_content(st["head"]) * dxq)
        E1 = fd.assemble(st["rho"] * st["H"] * dxq)
        injected = q * 3600.0 * 60 * rho_w
        stored = ((theta1 - theta0) + st["solver"].elastic_storage_m) * rho_w
        assert injected == pytest.approx(frozen + stored, rel=1e-4), f"T={T_C}"
        assert (E1 - E0) == pytest.approx(L_f * frozen, rel=1e-10), f"T={T_C}"


def test_very_dry_cold_firn_freezes_at_most_its_trace_water():
    """How dry "dry" can be in a head formulation, and what that costs.

    With ``theta_r = 0`` there is no perfectly dry state at any finite head:
    every head holds some water, and a uniform head is not even an equilibrium
    (``h + z`` still has a gradient), so the trace drains downward and freezes.
    The honest statement is therefore not "nothing freezes" but "nothing beyond
    the trace the curve cannot represent away", which is what is asserted here.

    The trace is a real artifact of the floor: at ``h_min = -100 m`` it is a
    few tenths of a percent of the pore space. Anything that needs a genuinely
    dry column - the South Pole and Summit inversions - uses the dry
    :class:`~firnpack.models.firn.FirnModel` path, where the refreezing source
    is absent entirely rather than merely small.
    """
    params = FirnRichardsParameters()
    st = _column(T_C=-30.0, h_val=params.h_min)
    dxq = _dxq(st["mesh"])
    rho_w = st["model"].params.rho_w

    held = float(fd.assemble(
        st["curves"].moisture_content(st["head"]) * dxq)) * rho_w   # kg m^-2
    pore = float(fd.assemble(st["curves"].theta_s * dxq)) * rho_w

    assert held < 0.005 * pore, (
        f"floor holds {held:.1f} kg/m^2, {100*held/pore:.2f}% of pore space")

    frozen = _run(st, nsteps=10, phase_change=True)
    assert 0.0 <= frozen <= held, f"froze {frozen:.3f} of {held:.3f} kg/m^2 held"


# ----------------------------------------------------------------------
# Constitutive checks
# ----------------------------------------------------------------------
def test_close_off_cutoff_makes_dense_firn_impermeable():
    """Calonne's fit alone never reaches zero; an aquifer needs a floor to sit on."""
    st = _column()
    model = st["model"]
    dx = fd.dx(domain=st["mesh"])

    def k(rho_val, cutoff=True):
        m = FirnRichardsModel(FirnRichardsParameters(closeoff_cutoff=cutoff))
        return float(fd.assemble(m.permeability(fd.Constant(rho_val)) * dx))

    # untouched over the range the fit was calibrated on
    for rho_val in (350.0, 550.0, 800.0):
        assert k(rho_val) == pytest.approx(k(rho_val, cutoff=False), rel=1e-10)
    assert k(917.0) == 0.0
    assert k(900.0) < 0.01 * k(830.0)
    del model


def test_permeability_follows_prognostic_grain_size():
    """Coarser grains percolate faster - the Mode B coupling."""
    st = _column()
    dx = fd.dx(domain=st["mesh"])
    fine = float(fd.assemble(
        st["model"].permeability(fd.Constant(450.0), fd.Constant(1e-7)) * dx))
    coarse = float(fd.assemble(
        st["model"].permeability(fd.Constant(450.0), fd.Constant(4e-7)) * dx))
    assert coarse == pytest.approx(4.0 * fine, rel=1e-10)


def test_retention_inversion_round_trips():
    """head_from_moisture must invert moisture_content on the wet branch."""
    st = _column()
    V, curves, model = st["V"], st["curves"], st["model"]
    for h_val in (-0.5, -2.0, -10.0):
        h = fd.Function(V).interpolate(fd.Constant(h_val))
        theta = fd.Function(V).interpolate(curves.moisture_content(h))
        back = fd.Function(V).interpolate(model.head_from_moisture(curves, theta))
        assert back.dat.data_ro.max() == pytest.approx(h_val, rel=1e-6)


def test_aquifer_perches_on_a_low_permeability_layer():
    """A saturated zone forms above an ice layer - the aquifer geometry.

    Controlled against an identical column with no lens, because the base is
    impermeable and the bottom cell saturates on whatever gets past regardless.
    This is the case the moisture form cannot do (see the xfail of the same
    name in test_hydrology.py): here the water table above the lens is a real
    free surface at h = 0.
    """
    def run(with_lens):
        st = _column(T_C=0.0)
        x = fd.SpatialCoordinate(st["mesh"])[0]
        if with_lens:
            st["rho"].interpolate(
                fd.conditional(
                    fd.And(fd.ge(x, 0.30 * H0), fd.le(x, 0.40 * H0)),
                    fd.Constant(900.0), fd.Constant(400.0),
                )
            )
            st["model"].refresh_curves(st["curves"], st["rho"])
        _run(st, dt=3600.0, nsteps=200, top_flux=1.0e-6)
        Vc = fd.FunctionSpace(st["mesh"], "CG", 1)
        z = fd.Function(Vc).interpolate(x).dat.data_ro
        hc = fd.Function(Vc).project(st["head"]).dat.data_ro
        band = (z > 0.40 * H0) & (z < 0.60 * H0)
        return hc[band].max()

    h_lens = run(True)
    h_open = run(False)
    assert h_lens > h_open + 0.05, (
        f"lens did not perch water above it: {h_lens:.3f} vs {h_open:.3f} m")

    # Note: this asserts perching (a raised head above the lens), not a fully
    # saturated zone. Driving it to h >= 0 needs more water than a closed
    # column can take - with an impermeable base the head runs away instead of
    # settling, and the solve fails. In the aquifer driver the base is open and
    # the outlet is the advective flux theta*w, not a drainage law.


def test_free_drainage_lets_a_wet_column_drain_and_closes_the_budget():
    """A 'free' base outlet drains under gravity at K(h) and no faster.

    Without an outlet a column with sustained net recharge has no steady state:
    it fills and then pressurises (3102 m of head was measured in a 60 m
    column). Free drainage is the 1D stand-in for aquifer water reaching the
    bed. Two things are asserted: water actually leaves, and the budget still
    closes once the outflow is counted.

    The column carries a real density profile so the base is dense and the
    outflow throttled, which is the physical situation. A uniform 400 kg/m^3
    column started fully saturated drains 11 m of water in about eight minutes
    at K_s = 2e-2 m/s - a waterfall that Newton refuses at any step size, and
    not a state a firn aquifer is ever in.
    """
    st = _column(T_C=0.0, h_val=-0.3)
    x = fd.SpatialCoordinate(st["mesh"])[0]
    st["rho"].interpolate(350.0 + (917.0 - 350.0) * (1.0 - fd.exp(-(H0 - x) / 6.0)))
    st["model"].refresh_curves(st["curves"], st["rho"])
    st["bcs"] = {1: {"free": None}, 2: {"flux": fd.Constant(0.0)}}
    st["solver"] = FirnRichardsSolver(st["model"])   # rebuild against new bcs
    dxq = _dxq(st["mesh"])
    theta0 = fd.assemble(st["curves"].moisture_content(st["head"]) * dxq)
    ds_base = fd.ds(1, domain=st["mesh"], metadata={"quadrature_degree": 3})

    dt, n, drained = 3600.0, 60, 0.0
    for _ in range(n):
        st["head"], _ = st["solver"].step(
            head=st["head"], enthalpy=st["H"], density=st["rho"],
            curves=st["curves"], dt=dt, bcs=st["bcs"], phase_change=False)
        drained += float(fd.assemble(
            st["curves"].relative_conductivity(st["head"]) * ds_base)) * dt

    theta1 = fd.assemble(st["curves"].moisture_content(st["head"]) * dxq)
    lost = (theta0 - theta1) - st["solver"].elastic_storage_m
    assert drained > 0.0
    assert lost == pytest.approx(drained, rel=1e-6)


def test_sealed_base_does_not_drain():
    """The default 'flux' = 0 base must still hold water: control for the above."""
    st = _column(T_C=0.0, h_val=2.0)
    dxq = _dxq(st["mesh"])
    theta0 = fd.assemble(st["curves"].moisture_content(st["head"]) * dxq)
    _run(st, dt=3600.0, nsteps=60)
    theta1 = fd.assemble(st["curves"].moisture_content(st["head"]) * dxq)
    stored = (theta1 - theta0) + st["solver"].elastic_storage_m
    assert abs(stored) < 1e-9


def test_advective_outflow_at_the_base_is_theta_times_w_and_closes_the_budget():
    """Water leaves the base with the compacting firn, at theta * w, and no
    faster. No drainage law: the rate is fixed by the imposed velocity and the
    pore water at the base, and it caps at porosity * |w|."""
    st = _column(T_C=0.0, h_val=-0.3)
    V = st["V"]
    w = fd.Function(V).interpolate(fd.Constant(-1.0e-7))     # ~3 m/yr downward
    st["bcs"] = {1: {"advect": None}, 2: {"flux": fd.Constant(0.0)}}
    st["solver"] = FirnRichardsSolver(st["model"])
    dxq = _dxq(st["mesh"])
    theta0 = fd.assemble(st["curves"].moisture_content(st["head"]) * dxq)
    x = fd.SpatialCoordinate(st["mesh"])
    e_z = fd.grad(x[0])
    n = fd.FacetNormal(st["mesh"])
    ds_base = fd.ds(1, domain=st["mesh"], metadata={"quadrature_degree": 3})

    dt, nsteps, drained = 3600.0, 60, 0.0
    for _ in range(nsteps):
        st["head"], _ = st["solver"].step(
            head=st["head"], enthalpy=st["H"], density=st["rho"],
            curves=st["curves"], dt=dt, bcs=st["bcs"], firn_velocity=w,
            phase_change=False)
        theta_b = st["curves"].moisture_content(st["head"])
        drained += float(fd.assemble(theta_b * w * fd.dot(e_z, n) * ds_base)) * dt

    theta1 = fd.assemble(st["curves"].moisture_content(st["head"]) * dxq)
    lost = (theta0 - theta1) - st["solver"].elastic_storage_m
    assert drained > 0.0
    # 1e-5, not 1e-6: the solve removes water at every sub-step's head, while
    # this diagnostic samples only the head at the end of each full step, so
    # the two differ by the sub-stepping granularity (~1 ppm here). The
    # driver's own budget, which integrates the same term, closes to 1e-4.
    assert lost == pytest.approx(drained, rel=1e-5)
    # and it is bounded by porosity * |w| * time, the physical ceiling
    por = 1.0 - 400.0 / 917.0
    assert drained <= por * 1.0e-7 * dt * nsteps * (1.0 + 1e-9)


def test_deep_permeability_correction_applies_only_where_it_was_measured():
    """perm_scale_deep multiplies k in dense firn and leaves fresh firn alone.

    The Helheim conductivity was measured inside the aquifer (~550-650
    kg/m^3), where Calonne's snow-based fit runs about 10x high. Scaling the
    whole column by that correction slowed percolation through the seasonal
    cold-wave zone so much that the column refroze 89-97% of its melt and no
    aquifer could form. The correction is therefore a smooth ramp in density:
    exactly 1 below perm_rho_lo, exactly perm_scale_deep above perm_rho_hi.
    """
    mesh = fd.IntervalMesh(4, 0.0, 1.0)
    V = fd.FunctionSpace(mesh, "CG", 1)
    base = FirnRichardsParameters()
    deep = FirnRichardsParameters(perm_scale_deep=0.1, perm_rho_lo=450.0,
                                  perm_rho_hi=600.0)

    def k(rho, p):
        r = fd.Function(V).interpolate(fd.Constant(rho))
        f = fd.Function(V).interpolate(FirnRichardsModel(p).permeability(r))
        return float(f.dat.data_ro[0])

    for rho in (350.0, 450.0):
        assert k(rho, deep) == pytest.approx(k(rho, base), rel=1e-12)
    for rho in (600.0, 700.0, 800.0):
        assert k(rho, deep) == pytest.approx(0.1 * k(rho, base), rel=1e-12)
    mid = k(525.0, deep) / k(525.0, base)          # ramp midpoint
    assert mid == pytest.approx(0.55, rel=1e-12)
    # monotone through the ramp
    ratios = [k(r, deep) / k(r, base) for r in np.linspace(440.0, 610.0, 18)]
    assert np.all(np.diff(ratios) <= 1e-12)


def test_freezing_cannot_overdraw_a_cell_the_water_leaves_within_the_step():
    """The sink is limited by what a cell still holds when the sub-step ends.

    A wetting front passing through cold firn is the case: the cell is
    saturated when the step is staged, so the staged sink is sized to the cold
    content (here 0.04 of the pore space, several times the water a cell at
    h = -20 m holds), but the water drains on through the cell within the
    same step. A sink fixed at staging then keeps taking
    water that is no longer there, theta falls below theta(h_min), and the
    head runs to -1000 m to satisfy storage. That cell, next to a saturated
    one at +23 m under a forming ice lens, is what stopped Newton in the
    ERA5-forced column. With the sink implicit in the head, no cell can end a
    step below the driest state the curve expresses, and the water budget
    still closes on the freezing that actually occurred.
    """
    st = _column(rho_val=500.0, T_C=-5.0, h_val=-20.0)
    x = fd.SpatialCoordinate(st["mesh"])[0]
    # a saturated slab in the top 2 m of a cold, dry, sealed column: the
    # water has to pass through the cold cells below to go anywhere
    st["head"].interpolate(fd.conditional(x > H0 - 2.0, 0.0, -20.0))
    dxq = _dxq(st["mesh"])
    curves = st["curves"]
    theta_min = float(fd.assemble(
        st["model"].driest_moisture(curves) * fd.dx) / H0)
    theta0 = float(fd.assemble(curves.moisture_content(st["head"]) * dxq))

    frozen = 0.0
    dt = 6.0 * 3600.0
    for _ in range(12):
        st["head"], m = st["solver"].step(
            head=st["head"], enthalpy=st["H"], density=st["rho"],
            curves=curves, dt=dt, bcs=st["bcs"], phase_change=True)
        frozen += float(fd.assemble(m * dxq)) * dt / 1000.0

    theta = fd.Function(st["V"]).interpolate(curves.moisture_content(st["head"]))
    assert theta.dat.data_ro.min() >= theta_min * (1.0 - 1e-6)
    assert st["head"].dat.data_ro.min() >= st["model"].params.h_min * (1.0 + 1e-3)
    assert frozen > 0.0
    theta1 = float(fd.assemble(curves.moisture_content(st["head"]) * dxq))
    lost = (theta0 - theta1) - st["solver"].elastic_storage_m
    assert lost == pytest.approx(frozen, rel=1e-6)


def test_dry_cells_keep_a_bounded_head():
    """Below h_min the retention curve is linear, so a trace loss is bounded.

    With theta_r = 0 van Genuchten's tail has a capacity that vanishes as
    1/h^2: a cell at the retention floor that loses a trace of water sees a
    divergent head change, and heads of -200 to -2400 m appeared beside
    saturated cells in the ERA5-forced column, where Newton then failed. A
    dry, cold column that drains its trace through the K_min film
    conductivity for a season must stay within a couple of h_min, keep a
    positive water content, and remain solvable.
    """
    st = _column(rho_val=500.0, T_C=-2.0, h_val=-90.0)
    st["bcs"] = {1: {"free": None}, 2: {"flux": fd.Constant(0.0)}}
    st["solver"] = FirnRichardsSolver(st["model"])
    curves = st["curves"]
    h_min = st["model"].params.h_min
    dt = 6.0 * 86400.0

    def advance(n):
        for _ in range(n):
            st["head"], _ = st["solver"].step(
                head=st["head"], enthalpy=st["H"], density=st["rho"],
                curves=curves, dt=dt, bcs=st["bcs"], phase_change=True)
        return float(st["head"].dat.data_ro.min())

    h0 = float(st["head"].dat.data_ro.min())
    h1 = advance(30)                                 # half a year
    h2 = advance(30)                                 # and another
    h = st["head"].dat.data_ro
    theta = fd.Function(st["V"]).interpolate(curves.moisture_content(st["head"]))
    assert h1 < h_min                                # it did dry past the join
    # bounded: the floor tapers off below h_min, so the drift slows and the
    # second half-year moves the driest cell less than a third as far as the
    # first; the old curve went to -2000 m and beyond, accelerating
    assert h2 > 5.0 * h_min
    assert (h1 - h2) < (h0 - h1) / 3.0
    assert theta.dat.data_ro.min() >= 0.0
    # and the linear branch inverts exactly where it still holds water (a
    # cell dried to exactly zero sits on the flat part and has no inverse)
    back = fd.Function(st["V"]).interpolate(
        st["model"].head_from_moisture(curves, theta))
    dry = (h < h_min - 1.0) & (theta.dat.data_ro > 1e-6)
    if dry.any():
        assert np.allclose(back.dat.data_ro[dry], h[dry], rtol=1e-8, atol=1e-6)
