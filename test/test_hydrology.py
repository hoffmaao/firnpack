"""Percolation, refreezing, drainage and aquifer formation.

This file used to be parked: it was the original script, it asserted nothing,
and its physics was unresolved - ``W_surf`` was pinned at 0 so no water was ever
injected, and the melt-flux routine that would have driven it was defined but
never called (and referenced ``FirnParameters`` fields that do not exist). Any
assertion written against that behaviour would have passed on a completely
broken solver.

The melt path is now resolved, so the file is a test. The assertions below are
budget and bound statements rather than numeric targets, so they pin the
physics without freezing a particular discretisation.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import firedrake as fd
except Exception:  # pragma: no cover
    fd = None

pytestmark = pytest.mark.skipif(fd is None, reason="firedrake not available")

from firnpack.mesh import column_mesh
from firnpack.models.firn import FirnModel, FirnParameters
from firnpack.models.hydrology import HydrologyModel, HydrologyParameters
from firnpack.solvers.hydrology_solver import HydrologySolver


H0 = 20.0
NZ = 60


def _column(rho_val=400.0, T_C=-5.0, W_val=0.0, **hyd_kw):
    """A uniform column plus the model objects that act on it."""
    mesh = column_mesh(H0, NZ, stretch=1.0)
    V = fd.FunctionSpace(mesh, "CG", 1)

    fp = FirnParameters()
    firn = FirnModel(fp)
    hp = HydrologyParameters(theta_W=1.0, theta_H=1.0, **hyd_kw)
    hyd = HydrologyModel(hp)
    solver = HydrologySolver(hyd, firn)

    def const(v):
        return fd.Function(V).interpolate(fd.Constant(v))

    state = dict(
        mesh=mesh, V=V, fp=fp, hp=hp, firn=firn, hyd=hyd, solver=solver,
        rho=const(rho_val),
        W=const(W_val),
        w=const(0.0),
        # H = c_i (T - T_m) is the cold content
        H=const(fp.c_i * T_C),
    )
    return state


def _run(st, *, dt=600.0, nsteps=100, melt=None, basal=None, drainage=True):
    tot_frozen = 0.0
    for _ in range(nsteps):
        st["H"], st["W"], m = st["solver"].prognostic_solve(
            enthalpy=st["H"], water=st["W"], density=st["rho"],
            firn_velocity=st["w"], dt=dt,
            surface_melt_flux=melt, basal_flux=basal,
            include_drainage=drainage,
        )
        tot_frozen += fd.assemble(m * fd.dx) * dt
    return tot_frozen


# ----------------------------------------------------------------------
# The melt path
# ----------------------------------------------------------------------
def test_surface_melt_enters_the_column():
    """The original defect: with no boundary term, no water could ever enter.

    Integrating the flux divergence by parts and dropping the surface integral
    imposes zero water flux at *both* ends, so the column was sealed. This is
    the end-to-end statement that it no longer is.
    """
    st = _column()
    assert fd.assemble(st["W"] * fd.dx) == 0.0

    _run(st, melt=2.0e-4, nsteps=50)

    assert fd.assemble(st["W"] * fd.dx) > 0.0
    # water must appear near the surface first, not uniformly
    z = st["V"].mesh().coordinates.dat.data_ro.ravel()
    Wd = st["W"].dat.data_ro
    assert Wd[z > 0.75 * H0].max() > Wd[z < 0.25 * H0].max()


def test_no_melt_flux_leaves_the_column_sealed():
    """Passing no surface flux must still mean no water: the default is closed."""
    st = _column()
    _run(st, melt=None, nsteps=25)
    assert fd.assemble(st["W"] * fd.dx) == 0.0


# ----------------------------------------------------------------------
# Conservation
# ----------------------------------------------------------------------
def test_mass_budget_closes():
    """Injected = frozen + stored, to round-off.

    This is the assertion that catches the failure mode the finite-element
    phase-change sink had: the increment is capped by the water at a node, but
    the mass matrix spreads that cap onto its neighbours, so nodes went
    negative and clipping them re-created water.
    """
    st = _column()
    dt, nsteps, melt = 600.0, 100, 2.0e-4
    frozen = _run(st, dt=dt, nsteps=nsteps, melt=melt)

    injected = melt * dt * nsteps
    stored = fd.assemble(st["W"] * fd.dx)
    assert abs(injected - frozen - stored) < 1e-9 * injected


def test_energy_budget_closes():
    """Matrix enthalpy gain equals the latent heat of the mass frozen."""
    st = _column()
    E0 = fd.assemble(st["rho"] * st["H"] * fd.dx)
    frozen = _run(st, melt=2.0e-4, nsteps=100)
    E1 = fd.assemble(st["rho"] * st["H"] * fd.dx)

    assert abs((E1 - E0) - st["hp"].L * frozen) < 1e-9 * abs(E1 - E0)


def test_closed_temperate_column_conserves_water():
    """With no source, no drainage and no phase change, water is conserved."""
    st = _column(T_C=0.0, W_val=20.0)   # temperate: no cold content to freeze into
    W0 = fd.assemble(st["W"] * fd.dx)
    _run(st, nsteps=100, melt=None, drainage=False)
    assert fd.assemble(st["W"] * fd.dx) == pytest.approx(W0, rel=1e-10)


# ----------------------------------------------------------------------
# Bounds
# ----------------------------------------------------------------------
def test_phase_change_respects_bounds_under_extreme_forcing():
    """W must stay >= 0 and H must never overshoot the melting point."""
    st = _column(T_C=-20.0)
    _run(st, dt=3600.0, nsteps=100, melt=5.0e-3)   # far more melt than can refreeze

    assert st["W"].dat.data_ro.min() >= -1e-12
    assert st["H"].dat.data_ro.max() <= 1e-9      # H = c_i (T - T_m) <= 0


def test_dry_cold_firn_is_an_exact_fixed_point():
    """No water and cold firn => zero phase change, bit-for-bit.

    This is what lets the dry inversions (South Pole, Summit) stay unchanged.
    """
    st = _column(W_val=0.0, T_C=-30.0)
    H_before = st["H"].dat.data_ro.copy()

    frozen = _run(st, nsteps=10, melt=None)

    assert frozen == 0.0
    assert st["W"].dat.data_ro.max() == 0.0
    assert np.array_equal(st["H"].dat.data_ro, H_before)


# ----------------------------------------------------------------------
# Constitutive relations
# ----------------------------------------------------------------------
def test_gravity_drives_water_downward():
    """With x measured upward, the gravitational Darcy flux must be negative."""
    st = _column(W_val=50.0, T_C=0.0)
    q = fd.Function(st["V"]).interpolate(
        st["hyd"].darcy_flux(st["W"], st["rho"])
    )
    assert q.dat.data_ro.max() < 0.0


def test_capillary_flux_is_diffusive():
    """Water must move from wet toward dry firn, not the reverse.

    ``Psi = -psi0 (1 - S)`` makes the matric potential most negative where the
    firn is driest. The previous ``+psi0 (1 - S)`` had the opposite sign, which
    made the capillary term anti-diffusive.
    """
    st = _column(T_C=0.0)
    x = fd.SpatialCoordinate(st["mesh"])[0]
    # saturation increasing upward
    # S must stay above S_res, or k_rel is legitimately zero and the flux
    # vanishes for a reason that has nothing to do with the capillary sign.
    st["W"].interpolate(fd.Constant(100.0) * x / H0 + fd.Constant(50.0))

    hyd, hp = st["hyd"], st["hp"]
    S = hyd.saturation(st["W"], st["rho"])
    K = hyd.hydraulic_conductivity(st["rho"])
    cap = fd.Function(st["V"]).interpolate(
        -hp.rho_w * K * hyd.k_rel(S) * hyd.capillary_potential_prime(S) * S.dx(0)
    )
    # dS/dx > 0, so the capillary flux must be downward (away from the wet top)
    assert cap.dat.data_ro.max() < 0.0


def test_calonne_permeability_matches_the_published_form():
    """k = 3 r^2 exp(-0.013 rho), and it must fall as firn densifies."""
    st = _column()
    hyd = st["hyd"]
    dx = fd.dx(domain=st["mesh"])
    for rho_val, r2 in ((400.0, 2.5e-7), (700.0, 1.0e-6)):
        rho_c, r2_c = fd.Constant(rho_val), fd.Constant(r2)
        got = float(fd.assemble(hyd.permeability(rho_c, r2_c) * dx)) / H0
        want = 3.0 * r2 * np.exp(-0.013 * rho_val)
        assert got == pytest.approx(want, rel=1e-10)

    k_loose = float(fd.assemble(hyd.permeability(fd.Constant(400.0)) * dx))
    k_dense = float(fd.assemble(hyd.permeability(fd.Constant(830.0)) * dx))
    assert k_dense < k_loose


def test_permeability_follows_prognostic_grain_size():
    """Coarser grains must percolate faster - the Mode B coupling."""
    st = _column()
    dx = fd.dx(domain=st["mesh"])
    fine = float(fd.assemble(
        st["hyd"].permeability(fd.Constant(450.0), fd.Constant(1e-7)) * dx))
    coarse = float(fd.assemble(
        st["hyd"].permeability(fd.Constant(450.0), fd.Constant(4e-7)) * dx))
    assert coarse == pytest.approx(4.0 * fine, rel=1e-10)


# ----------------------------------------------------------------------
# Drainage and the water table
# ----------------------------------------------------------------------
def test_lateral_drainage_removes_saturated_excess():
    """Above S_drain, water is lost on the drainage timescale.

    Started from an already-wet temperate column so the drainage term is tested
    directly, rather than waiting on a wetting front to arrive and happening to
    cross the threshold.
    """
    # phi = 1 - 400/917 = 0.564, so S = 0.3 is W = 169 kg/m^3
    st = _column(T_C=0.0, W_val=300.0, S_drain=0.3, tau_drain=5.0e5)
    st2 = _column(T_C=0.0, W_val=300.0)        # tau_drain None -> closed column

    W0 = fd.assemble(st["W"] * fd.dx)
    _run(st, dt=3600.0, nsteps=50, melt=None)
    _run(st2, dt=3600.0, nsteps=50, melt=None)

    W_drained = fd.assemble(st["W"] * fd.dx)
    W_closed = fd.assemble(st2["W"] * fd.dx)
    assert W_closed == pytest.approx(W0, rel=1e-10)   # closed column loses nothing
    assert W_drained < 0.99 * W_closed


def test_drainage_is_off_by_default():
    st = _column(T_C=0.0)
    assert st["hyd"].drainage_rate(st["W"], st["rho"]) is None


@pytest.mark.xfail(
    reason="the moisture form cannot hold a perched saturated zone; this is "
           "what test_firn_richards.py exists for",
    strict=False,
)
def test_aquifer_perches_on_a_low_permeability_layer():
    """Perching, which this formulation gets only partly right.

    Water does reach and slow at the lens, but it does not pond into a
    saturated zone above it the way it should. The reason is structural rather
    than a tuning failure: with ``W`` as the unknown, ``S = W/(rho_w phi)``
    saturates and the flux stops responding, so there is no variable left to
    carry a water table, and the pore-pressure penalty that stands in for one
    is a stiffness rather than a free surface.

    Left here, and xfailed rather than deleted, because it is the concrete
    evidence for moving to the mixed head form in
    :mod:`firnpack.models.firn_richards`, where the same configuration does
    perch.
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
        _run(st, dt=7200.0, nsteps=250, melt=1.0e-3, drainage=False)
        z = st["mesh"].coordinates.dat.data_ro.ravel()
        S = fd.Function(st["V"]).interpolate(
            st["hyd"].saturation(st["W"], st["rho"])
        ).dat.data_ro
        return S[(z > 0.40 * H0) & (z < 0.60 * H0)].max()

    assert run(True) > 3.0 * run(False)


def test_close_off_cutoff_makes_dense_firn_impermeable():
    """Permeability must vanish at close-off, and be untouched below it."""
    st = _column()
    hyd = st["hyd"]
    dx = fd.dx(domain=st["mesh"])

    def k(rho_val):
        return float(fd.assemble(hyd.permeability(fd.Constant(rho_val)) * dx))

    # below close-off the Calonne fit is unmodified (connectivity == 1)
    from firnpack.models.hydrology import HydrologyParameters, HydrologyModel
    raw = HydrologyModel(HydrologyParameters(closeoff_cutoff=False))
    for rho_val in (350.0, 550.0, 800.0):
        assert k(rho_val) == pytest.approx(
            float(fd.assemble(raw.permeability(fd.Constant(rho_val)) * dx)), rel=1e-10
        )
    # at and above solid ice it is exactly zero, and falls steeply in between
    assert k(917.0) == 0.0
    assert k(900.0) < 0.01 * k(830.0)


def test_refreezing_enters_the_base_to_surface_velocity_integration():
    """Refreezing adds ice mass, so it must appear in the continuity solve.

    The column velocity is obtained by integrating

        rho dw/dz = m - drho/dt

    from the base to the surface. That integration is unchanged by the
    hydrology except for the source ``m`` (Meyer & Hewitt's ice equation), so
    this checks two things at once: that ``m`` shifts the residual by exactly
    ``-m`` per unit test function, and that the dry path is untouched when no
    source is passed.
    """
    st = _column()
    V, firn = st["V"], st["firn"]
    psi = fd.TestFunction(V)
    w_trial = fd.TrialFunction(V)
    w_old = fd.Function(V).interpolate(fd.Constant(0.0))
    drhodt = fd.Function(V).interpolate(fd.Constant(1.0e-6))
    m = fd.Function(V).interpolate(fd.Constant(3.0e-7))

    def rhs(refreezing):
        delta = firn.velocity_delta(
            w_trial=w_trial, w_old=w_old, rho=st["rho"], drhodt=drhodt,
            test=psi, refreezing=refreezing,
        )
        return fd.assemble(fd.rhs(delta)).dat.data_ro.copy()

    dry = rhs(None)
    # passing an explicit zero must be identical to passing nothing
    assert np.allclose(rhs(fd.Function(V).interpolate(fd.Constant(0.0))), dry,
                       rtol=0, atol=0)

    wet = rhs(m)
    # rhs = -(residual without w); adding +m to the balance shifts it by int(m psi)
    shift = fd.assemble(m * psi * fd.dx).dat.data_ro
    assert np.allclose(wet - dry, shift, rtol=1e-12, atol=1e-18)


def test_velocity_integration_treats_drhodt_as_the_total_tendency():
    """velocity_delta solves rho dw/dz = m - drho/dt for the *total* drho/dt.

    Held at a fixed total tendency, a source m must reduce the downward
    velocity - that is what the residual says. It is not a statement that
    refreezing thickens a column: in the solver the same m also enters the
    density equation, so the total tendency rises by m and the two cancel,
    leaving dw/dz = -compaction/rho. That cancellation is checked at the
    solver level in test_aquifer.py; this pins the sign convention of the
    piece it relies on.
    """
    st = _column()
    V, firn = st["V"], st["firn"]
    psi = fd.TestFunction(V)
    w_trial = fd.TrialFunction(V)
    w_old = fd.Function(V).interpolate(fd.Constant(0.0))
    drhodt = fd.Function(V).interpolate(fd.Constant(1.0e-6))

    def solve_w(refreezing):
        delta = firn.velocity_delta(
            w_trial=w_trial, w_old=w_old, rho=st["rho"], drhodt=drhodt,
            test=psi, refreezing=refreezing,
        )
        w = fd.Function(V)
        bc = fd.DirichletBC(V, fd.Constant(0.0), 1)   # base
        fd.solve(fd.lhs(delta) == fd.rhs(delta), w, bcs=[bc])
        return w

    w_dry = solve_w(None)
    w_wet = solve_w(fd.Function(V).interpolate(fd.Constant(3.0e-7)))
    # both are downward (negative) above the base; refreezing makes it less so
    assert w_wet.dat.data_ro[-1] > w_dry.dat.data_ro[-1]


def test_water_stays_non_negative_at_a_sharp_front():
    """Positivity, which the scheme guarantees rather than repairs.

    Gravity drainage is a nonlinear advection, so the wetting front is close to
    a shock. Unstabilised CG1 rang across it and drove W to -22% of peak; the
    tail decayed by a factor -0.27 per node, the signature of the consistent
    mass matrix rather than of the flux. Lumping the mass and upwinding the
    flux makes the operator an M-matrix, so W simply cannot go negative -
    which matters because the aquifer question is a water budget, and the
    phase-change closure deliberately refuses to launder a negative W.
    """
    st = _column(T_C=0.0)
    _run(st, dt=3600.0, nsteps=120, melt=2.0e-3, drainage=False)
    # Ahead of the front the scheme leaves an exponentially small positive
    # residue, not a negative one: the bound is >= 0, not == 0.
    assert st["W"].dat.data_ro.min() >= 0.0


def test_stabilisation_does_not_leak_mass():
    """Upwinding and lumping change accuracy, never the budget."""
    st = _column(T_C=0.0)
    dt, nsteps, melt = 3600.0, 60, 1.0e-3
    _run(st, dt=dt, nsteps=nsteps, melt=melt, drainage=False)
    injected = melt * dt * nsteps
    assert fd.assemble(st["W"] * fd.dx) == pytest.approx(injected, rel=1e-12)
