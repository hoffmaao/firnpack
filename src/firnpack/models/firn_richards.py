"""Firn meltwater percolation in the mixed (head-based) Richards form.

Why this exists alongside :mod:`firnpack.models.hydrology`
----------------------------------------------------------
``hydrology.py`` writes the water balance in the **moisture form**, with the
bulk water content ``W`` as the unknown. That form conserves mass for free, but
it has a well-known defect: it runs out of a variable once the pore space
fills, because ``S = W/(rho_w phi)`` saturates and the flux stops responding to
further water. Getting a water table out of it required a pore-pressure penalty
on ``S > 1`` - a numerical stiffness, not a material property.

The **mixed form** of Celia, Bouloutas & Zarba (1990) - pressure head ``h`` as
the unknown, storage written as the ``theta``-difference
``(theta(h) - theta(h_0))/dt`` rather than ``C(h) dh/dt`` - keeps the exact mass
conservation of the moisture form *and* stays well posed through saturation,
because ``h`` remains meaningful where ``theta`` does not. That is the right
formulation for an aquifer, whose defining feature is a water table: the
surface ``h = 0``, an interior free surface the solution finds on its own
rather than a value clamped by hand.

Relation to Hewitt and to Cummings
----------------------------------
Hewitt's percolation model is posed non-dimensionally with idealised
constitutive laws, and the finite-element extension in Evan Cummings' firn code
keeps those laws on a fixed medium. Both are moisture-form models and inherit
its saturation defect, so neither can carry an unsaturated percolation zone and
a saturated aquifer in the same column. The mixed form can, which is what makes
a firn-aquifer simulation possible rather than a perched-water approximation.

Numerical treatment
-------------------
The discretisation follows the standard practice for variably-saturated flow,
and specifically the choices proven in the author's ``aquipack`` Richards solver
for coastal aquifers, reimplemented here so firnpack stays self-contained:

* van Genuchten (1980) retention with Mualem conductivity;
* an absolute conductivity floor ``K_min`` - physically film and vapour
  transport - so cells ahead of a drying front stay coupled to their
  neighbours instead of decoupling and stalling the front;
* a linear continuation of the retention curve below ``h_min``, so the
  driest cells keep a bounded head and a constant capacity (see
  ``VanGenuchtenCurves.h_lin``);
* a C1 blend of ``k_r`` across ``h = 0``, because Mualem's curve has a
  derivative jump there and Newton otherwise stalls on exactly the cells that
  straddle the water table;
* SIPG for the diffusive flux and donor-cell upwinding for the gravity flux, so
  the wetting front is monotone by construction rather than by tuned artificial
  diffusion.

What firn adds to groundwater Richards
--------------------------------------
1. **The medium moves and changes.** ``theta_s`` and ``K_s`` are fields derived
   from the prognostic density and grain radius, re-evaluated as the column
   compacts, not constants.
2. **Matrix advection.** Water is carried down with the compacting ice at the
   firn velocity ``w``, on top of its Darcy motion relative to the matrix. This
   burial is what Kuipers Munneke et al. (2014) identify as what protects an
   aquifer from the winter cold wave.
3. **Phase change.** Meltwater refreezes against the cold content of the firn.
   The freezing is a volumetric sink *inside* the Richards residual, implicit
   in the head and switched off smoothly as a cell approaches the driest state
   the retention curve expresses, so the water budget closes exactly and no
   cell can be overdrawn by water that left it within the step. The latent
   heat goes to the matrix enthalpy and the refrozen mass to the density
   equation (Meyer & Hewitt's ice equation), where it fills pores in place.
"""
from __future__ import annotations

from dataclasses import dataclass

import firedrake as fd

from firnpack.constants import (
    ice_density, water_density, latent_heat, water_viscosity, gravity,
    closeoff_density,
)


# --------------------------------------------------------------------------- #
# Constitutive curves
# --------------------------------------------------------------------------- #
@dataclass
class VanGenuchtenCurves:
    r"""van Genuchten (1980) retention with Mualem conductivity.

    .. math::
       S_e(h) = \left(1 + |\alpha h|^n\right)^{-m}, \quad m = 1 - 1/n

    ``theta_r``, ``theta_s`` and ``K_s`` may be Firedrake Functions, which is
    how firn's depth-varying porosity and permeability enter: nothing here
    assumes they are scalars.

    ``K_min`` [m/s] is an absolute floor on the conductivity. At a steepening
    front Mualem's ``k_r`` and the moisture capacity vanish together and the
    Newton rows for those cells go singular; a small floor (film and vapour
    transport) keeps them coupled without measurably changing wet-range
    fluxes. It is absolute, not a fraction of ``K_s``: firn's saturated
    conductivity is 1e-2 to 1e-1 m/s and varies over orders of magnitude with
    density, so a relative floor is a different physical quantity in every
    cell. Below ``h_floor`` it tapers to zero (see there), so a cell at the
    retention floor stops draining through it.

    ``blend`` [m] restores C1 continuity of ``k_r`` at ``h = 0``. Mualem's curve
    has a derivative jump there for every ``n``, so any cell whose head crosses
    the water table has a kinked residual and Newton stalls or cycles on it -
    the failure mode of columns whose cells are thicker than the capillary
    fringe, which in firn they always are. On ``[-blend, 0]`` the floored curve
    is replaced by the cubic Hermite interpolant matching value and slope at
    ``-blend`` and reaching 1 with zero slope at 0.
    """

    theta_r: object
    theta_s: object
    K_s: object
    alpha: float
    n: float
    K_min: float = 0.0
    blend: float = 0.0
    # Head around which K_min tapers to zero (logistic, half of K_min there,
    # width |h_floor|/4). None keeps the floor constant. The floor exists to
    # keep cells ahead of a front coupled; a cell already at the retention
    # floor must not keep draining through it, or its head falls without
    # bound. The taper stalled Newton when the retention curve's capacity
    # vanished in the same cells; with the linear tail (h_lin) it does not.
    h_floor: float | None = None
    # Head below which the retention curve continues linearly (C1 join) with
    # the slope it has there. None keeps van Genuchten's tail. With
    # theta_r = 0 that tail has a capacity C = dtheta/dh that vanishes as
    # 1/h^2: a cell that loses a trace of water sees a divergent head change
    # and gives Newton no leverage, which is how heads of -200 to -2400 m
    # appeared in dry cells of the ERA5-forced column. Linear below h_lin, a
    # trace loss costs a bounded head change and the row keeps a constant
    # capacity. Water below the join is a numerical trace, not physics.
    h_lin: float | None = None

    @property
    def m(self):
        return 1.0 - 1.0 / self.n

    def _linear_tail(self):
        """(Se, dSe/dh) at ``h_lin`` as floats."""
        a = abs(self.alpha * self.h_lin)
        g = 1.0 + a ** self.n
        Se_c = g ** (-self.m)
        # d/dh of (1 + (alpha|h|)^n)^-m for h < 0 is positive
        dSe_c = self.m * self.n * self.alpha * a ** (self.n - 1.0) * g ** (-self.m - 1.0)
        return Se_c, dSe_c

    def saturation(self, h):
        Se = (1.0 + abs(self.alpha * h) ** self.n) ** (-self.m)
        if self.h_lin is not None:
            Se_c, dSe_c = self._linear_tail()
            Se_lin = fd.max_value(Se_c + dSe_c * (h - self.h_lin), 0.0)
            Se = fd.conditional(h < self.h_lin, Se_lin, Se)
        return fd.conditional(h <= 0, Se, 1.0)

    def moisture_content(self, h):
        return self.theta_r + (self.theta_s - self.theta_r) * self.saturation(h)

    def _mualem(self, a):
        """Floorless Mualem ``k_r`` and ``dk_r/da`` for ``a = alpha|h|``.

        Written to work on UFL expressions and plain floats alike, because the
        C1 blend needs the float values at the join.
        """
        n, m = self.n, self.m
        g = 1.0 + a ** n
        t1 = 1.0 - a ** (n - 1.0) * g ** (-m)
        t2 = g ** (m / 2.0)
        dt1 = (-(n - 1.0) * a ** (n - 2.0) * g ** (-m)
               + m * n * a ** (2.0 * n - 2.0) * g ** (-m - 1.0))
        dt2 = (m / 2.0) * n * a ** (n - 1.0) * g ** (m / 2.0 - 1.0)
        k = t1 ** 2 / t2
        dk = (2.0 * t1 * dt1 * t2 - t1 ** 2 * dt2) / t2 ** 2
        return k, dk

    def relative_conductivity(self, h):
        """``K_s k_r(h) + K_min`` with the C1 blend applied.

        One conductivity for both the diffusive and the gravity flux. Floor
        only one of them and hydrostatic equilibrium, ``h = z_wt - z``,
        stops being a solution: a spurious flux appears in every dry cell.
        """
        a = abs(self.alpha * h)
        k_r = self._mualem(a)[0]
        if self.blend > 0.0:
            d = self.blend
            k0, dk0 = self._mualem(self.alpha * d)      # floats at h = -d
            slope0 = -self.alpha * dk0                  # dk_r/dh at h = -d, > 0
            s = (h + d) / d                             # 0 at -d, 1 at 0
            hermite = (k0 + (1.0 - k0) * (3.0 * s ** 2 - 2.0 * s ** 3)
                       + slope0 * d * (s - 2.0 * s ** 2 + s ** 3))
            k_r = fd.conditional(h <= -d, k_r, hermite)
        floor = self.K_min
        if self.h_floor is not None:
            w = abs(self.h_floor) / 4.0
            x = fd.max_value(fd.min_value((h - self.h_floor) / w, 40.0), -40.0)
            floor = self.K_min / (1.0 + fd.exp(-x))
        return fd.conditional(h <= 0, self.K_s * k_r, self.K_s) + floor


# --------------------------------------------------------------------------- #
@dataclass
class FirnRichardsParameters:
    """Hydraulic and thermal parameters for firn percolation."""

    rho_i: float = ice_density
    rho_w: float = water_density
    L: float = latent_heat
    g: float = gravity
    mu_w: float = water_viscosity

    # --- retention ---------------------------------------------------------
    # Firn is coarse and weakly capillary next to soil: a large alpha (small
    # air-entry head) and moderate n give the short capillary fringe and sharp
    # wetting fronts that firn actually shows.
    # n = 3 with this alpha is not usable: Mualem's conductivity then climbs
    # four orders of magnitude over 0.8 m of head, and Newton fails as soon as
    # the surface approaches the water table, at any step size and regardless
    # of ``blend`` (which widens the C1 join but cannot soften the curve
    # itself). n = 2 runs the same column for 300 steps and settles to a
    # hydrostatic water table. These are numerical defaults, not calibrated
    # firn retention: they want fitting to firn water-retention measurements
    # (e.g. Yamaguchi et al.) before any quantitative claim rests on them.
    vg_alpha: float = 4.0     # [1/m]
    vg_n: float = 2.0
    # theta_r is left at zero on purpose. Water is immobilised by Mualem's
    # k_r going to zero as the firn dries, not by removing it from the
    # retention curve - and if theta_r > 0 then refreezing, which can consume
    # even tightly-held water, drives theta below theta_r where the curve has
    # no head at all (the inversion returned -2.5e11 m and Newton died). With
    # theta_r = 0 every water content the column can reach has a finite head.
    S_res: float = 0.0        # irreducible saturation as a fraction of phi
    # Driest representable head. Bounds the retention inversion and, through
    # it, how much water one step is allowed to freeze.
    h_min: float = -100.0     # [m]
    # Absolute conductivity floor [m/s], tapered off below h_min. 1e-7 is
    # what the 1e-6 relative floor this began with amounted to in near-
    # surface firn; smaller values (1e-8 to 1e-10) leave the cells ahead of
    # a refreezing front decoupled and Newton fails on the freezing tests,
    # larger ones drain dry cells too fast for the taper to catch.
    K_min: float = 1.0e-7
    blend: float = 0.05       # [m]

    # --- permeability (Calonne 2012 + close-off cutoff) ---------------------
    perm_c: float = 3.0
    perm_b: float = -0.013
    perm_scale: float = 1.0
    # Density-dependent correction, applied on top of perm_scale. The Helheim
    # hydraulic-conductivity measurement (2.7e-4 m/s) was made in the
    # saturated aquifer at ~550-650 kg/m^3, where Calonne's fit runs ~10x
    # high. It says nothing about fresh near-surface firn, where Calonne's fit
    # is grounded in direct snow measurements. Scaling the whole column by the
    # deep correction slows percolation through the cold-wave zone and makes
    # the column refreeze 89-97% of its melt, which is what suppressed the
    # aquifer. perm_scale_deep multiplies k only above perm_rho_lo, ramping to
    # its full value by perm_rho_hi and holding it above: the band is exactly
    # the 550-650 kg/m^3 the measurement was made in. The default of 1.0
    # changes nothing.
    perm_scale_deep: float = 1.0
    perm_rho_lo: float = 550.0
    perm_rho_hi: float = 650.0
    r2_default: float = 2.5e-7
    rho_co: float = closeoff_density
    conn_exp: float = 3.0
    closeoff_cutoff: bool = True

    # --- saturated storage --------------------------------------------------
    # Below saturation the storage is d(theta)/dh, the retention curve's own
    # capacity. Above it theta is pinned at theta_s, so that derivative is zero
    # and the mixed-form mass term contributes nothing to the diagonal: the
    # Jacobian goes singular exactly where a water table forms, and Newton
    # fails at any step size. A specific storage term restores it,
    #
    #     S_s S(h) (h - h_0)/dt,
    #
    # and is real physics rather than a regulariser: it is the water and
    # skeleton compressibility, the same S_e = S_sk/(rho g) + beta_w phi/(1+phi)
    # used in the poroelastic groundwater model in ../aquifer. Water alone gives
    # only ~2e-6 1/m; firn's skeleton is far softer than rock, and 1e-4 1/m
    # corresponds to a bulk modulus of order 1e8 Pa.
    specific_storage: float = 1.0e-4   # [1/m]


    # Fraction of the drainable water one step may freeze. Freezing is now a
    # sink inside the solve rather than a state edit, so taking *all* of a
    # cell's water forces the head onto its floor and the Newton problem
    # becomes extremely stiff - hard forcing (-20 C with heavy melt) failed at
    # any sub-step. Leaving a margin costs nothing physically: the closure is
    # an equilibrium applied every step, so what is not frozen now is frozen
    # next step, and whatever *is* frozen is still removed exactly.
    freeze_safety: float = 0.9
    # Width, in volumetric water content, of the ramp that turns the freezing
    # sink off as a cell approaches theta(h_min). Cells holding more mobile
    # water than this freeze at the full cold-content demand; below it the
    # sink is proportional to what remains, which treated implicitly can
    # never take the cell past the driest state the curve expresses. A few
    # tenths of a percent: comparable to theta(h_min) itself.
    freeze_ramp_theta: float = 2.0e-3

    phi_min: float = 1.0e-4
    penalty_alpha: float = 2.0   # SIPG penalty constant (Hillewaert 2013)


class FirnRichardsModel:
    """Mixed-form Richards for a compacting, freezing firn column."""

    def __init__(self, params: FirnRichardsParameters | None = None):
        self.params = params or FirnRichardsParameters()

    # ------------------------------------------------------------------
    # Firn-derived hydraulic properties
    # ------------------------------------------------------------------
    def porosity(self, rho):
        p = self.params
        return fd.max_value(1.0 - rho / p.rho_i, p.phi_min)

    def pore_connectivity(self, rho):
        """Connected-pore fraction: 1 at ``rho_co``, 0 at solid ice.

        The ramp deliberately runs all the way to solid ice rather than closing
        at the nominal close-off density, so some mobility survives below
        close-off. That is a modelling choice, not an oversight, and it is worth
        being explicit about because the numbers look surprising: at
        rho = 830 kg/m^3 this returns 0.85, giving K = 5.3e-5 m/s - about 350
        times the peak melt supply these columns see. Permeability only falls to
        the melt rate near rho = 900.

        The justification is that close-off is a gradual transition rather than
        a surface, and wet firn retains some deep mobility past the nominal
        density. The cost is that a saturated zone does not get a hard floor at
        the firn-ice transition: in the 12-year runs the aquifer fills to the
        base of the domain, so its floor is partly the no-flux boundary rather
        than sealed ice, and the modelled water table sits deeper than observed
        aquifers partly for that reason (the missing lateral drainage is the
        other part). Closing the ramp near 840 instead would seal the base
        around 51 m in the SE Greenland case and raise the water table.
        """
        p = self.params
        f = (p.rho_i - rho) / (p.rho_i - p.rho_co)
        return fd.max_value(fd.min_value(f, 1.0), 0.0)

    def permeability(self, rho, grain_radius2=None):
        """Calonne et al. (2012) ``k = 3 r^2 exp(-0.013 rho)`` [m^2].

        Multiplied by the connected-pore fraction so it actually reaches zero
        at close-off. The bare fit never does - at 900 kg/m^3 it still gives
        K = 3.4e-5 m/s, so nothing can perch on an ice layer, and an aquifer
        has no floor to sit on.
        """
        p = self.params
        r2 = (fd.Constant(p.r2_default) if grain_radius2 is None
              else fd.max_value(grain_radius2, 1e-12))
        k = fd.Constant(p.perm_c) * r2 * fd.exp(fd.Constant(p.perm_b) * rho)
        if p.closeoff_cutoff:
            k = k * self.pore_connectivity(rho) ** p.conn_exp
        if p.perm_scale_deep != 1.0:
            # smoothstep ramp in density from 1 at rho_lo to perm_scale_deep
            # at rho_hi, so the calibrated correction applies where measured
            x = (rho - p.perm_rho_lo) / (p.perm_rho_hi - p.perm_rho_lo)
            x = fd.max_value(fd.min_value(x, 1.0), 0.0)
            ramp = x * x * (3.0 - 2.0 * x)
            k = k * (1.0 + (p.perm_scale_deep - 1.0) * ramp)
        return fd.Constant(p.perm_scale) * k

    def conductivity(self, rho, grain_radius2=None):
        """Saturated hydraulic conductivity [m s^-1]."""
        p = self.params
        return self.permeability(rho, grain_radius2) * p.rho_w * p.g / p.mu_w

    # ------------------------------------------------------------------
    def curves(self, V, rho, grain_radius2=None):
        """van Genuchten curves whose parameters are firn fields."""
        p = self.params
        theta_s = fd.Function(V, name="theta_s")
        theta_r = fd.Function(V, name="theta_r")
        K_s = fd.Function(V, name="K_s")
        # The conductivity floor is absolute (m/s). Film and vapour transport
        # do not scale with the pore-scale permeability, and a floor relative
        # to K_s was 0.3-5 m/yr of gravity drainage from every dry cell in
        # firn. A floor this small used to leave the driest rows singular;
        # the linear retention tail (h_lin) now gives those rows a constant
        # capacity instead, which is what makes the small floor usable.
        curves = VanGenuchtenCurves(
            theta_r=theta_r, theta_s=theta_s, K_s=K_s,
            alpha=p.vg_alpha, n=p.vg_n,
            K_min=p.K_min,
            blend=p.blend, h_lin=p.h_min, h_floor=p.h_min,
        )
        self.refresh_curves(curves, rho, grain_radius2)
        return curves

    def refresh_curves(self, curves, rho, grain_radius2=None):
        """Re-interpolate the field parameters in place.

        In place because the solver caches its variational problem against
        these Function objects; replacing them would silently leave the solver
        using the old medium.
        """
        p = self.params
        phi = self.porosity(rho)
        curves.theta_s.interpolate(phi)
        curves.theta_r.interpolate(fd.Constant(p.S_res) * phi)
        curves.K_s.interpolate(self.conductivity(rho, grain_radius2))

    # ------------------------------------------------------------------
    # Retention inversion (a public inverse of the retention curve, used for
    # diagnostics and tests). The phase change used to need it, to convert an
    # edited water content back into a head; that was replaced by carrying the
    # freezing sink inside the Richards residual, which conserves mass exactly
    # (see firnpack.solvers.firn_richards_solver).
    # ------------------------------------------------------------------
    def head_from_moisture(self, curves, theta):
        r"""Invert van Genuchten for the head holding a given water content.

        .. math::
           h = -\frac{1}{\alpha}\left(S_e^{-1/m} - 1\right)^{1/n}

        Saturated cells return ``h = 0``; only the unsaturated branch has to be
        right, since the mixed form carries any excess as positive head.
        """
        p = self.params
        m = 1.0 - 1.0 / p.vg_n
        Se = (theta - curves.theta_r) / (curves.theta_s - curves.theta_r)
        Se_min = (1.0 + abs(p.vg_alpha * p.h_min) ** p.vg_n) ** (-m)
        Se_vg = fd.max_value(fd.min_value(Se, 1.0 - 1e-12), Se_min)
        h = -(1.0 / p.vg_alpha) * (Se_vg ** (-1.0 / m) - 1.0) ** (1.0 / p.vg_n)
        if curves.h_lin is not None:
            # the linear tail inverts linearly
            Se_c, dSe_c = curves._linear_tail()
            h = fd.conditional(Se < Se_c, curves.h_lin + (Se - Se_c) / dSe_c, h)
        return fd.conditional(fd.ge(theta, curves.theta_s), fd.Constant(0.0), h)

    def driest_moisture(self, curves):
        """Water content at ``h_min``: the driest state the curve can express."""
        return curves.moisture_content(fd.Constant(self.params.h_min))

    # ------------------------------------------------------------------
    # Phase change
    # ------------------------------------------------------------------
    def freeze_increment(self, H, theta, rho, theta_min=0.0):
        """Water frozen per unit bulk volume [kg m^-3]; positive freezes.

        The same local-equilibrium closure as the moisture-form model: bounded
        by the cold content on one side and the available water on the other,
        so one step can neither overshoot ``H = 0`` nor drive ``theta`` below
        what the retention curve can express. Dry cold firn is a fixed point.

        ``theta_min`` is the water content at ``h_min``. Freezing stops there
        rather than at zero, so the head stays finite. The water left behind is
        a small fraction of a percent of the pore space, but it does mean this
        model slightly under-refreezes the most tightly held water.
        """
        p = self.params
        target = -rho * H / p.L
        avail = fd.Constant(p.freeze_safety) * fd.max_value(
            p.rho_w * (theta - theta_min), 0.0)
        return fd.max_value(fd.min_value(target, avail), -rho)

    def enthalpy_increment(self, f, rho):
        """Matrix enthalpy change [J kg^-1] from freezing ``f`` [kg m^-3]."""
        p = self.params
        return p.L * f / fd.max_value(rho, 1.0)

    # ------------------------------------------------------------------
    # Weak form
    # ------------------------------------------------------------------
    def _penalty(self, V):
        """SIPG penalty after Hillewaert (2013)."""
        deg = V.ufl_element().degree()
        if not isinstance(deg, int):
            deg = max(deg)
        return self.params.penalty_alpha * (deg + 1) * (deg + 1)

    @staticmethod
    def _e_z(mesh):
        """Unit vector along the mesh's last coordinate.

        firnpack's column coordinate is height above the base, so this points
        up and gravity drives water along ``-e_z``.
        """
        x = fd.SpatialCoordinate(mesh)
        return fd.grad(x[mesh.topological_dimension - 1])

    def residual(self, *, head, head_old, curves, dt, bcs, theta_old=None,
                 firn_velocity=None, source=None):
        """Full weak residual ``F(h; phi) = 0`` for one backward-Euler step.

            (theta(h) - theta_old)/dt + div(q) = source,
            q = -K(h) grad(h + z) + theta w

        The storage term is the Celia mixed form: differencing ``theta`` rather
        than writing ``C(h) dh/dt`` is what makes the scheme conserve mass
        exactly under Newton iteration.

        ``theta_old`` must be the water content **measured with the curves in
        force at the start of the step**, not recomputed from ``head_old`` with
        the current ones. In firn the medium compacts under the water, so
        ``theta_s`` shrinks between steps; evaluating both ends of the
        difference with the new curve makes ``theta(h_0)`` jump with no water
        having moved, and that water quietly disappears from the budget. It
        cost about half a percent of the melt total in a coupled run. Passing
        ``None`` recovers the fixed-medium behaviour.
        """
        h, h0 = head, head_old
        p_ = self.params
        V = h.function_space()
        mesh = V.mesh()
        phi = fd.TestFunction(V)
        n = fd.FacetNormal(mesh)
        deg = V.ufl_element().degree()
        if not isinstance(deg, int):
            deg = max(deg)
        md = {"quadrature_degree": 2 * deg + 1}
        dx = fd.dx(domain=mesh, metadata=md)
        dS = fd.dS(domain=mesh, metadata=md)
        ds = fd.ds(domain=mesh, metadata=md)

        K = curves.relative_conductivity(h)
        e_z = self._e_z(mesh)

        # --- storage (Celia mixed form + saturated specific storage) ---
        th_old = curves.moisture_content(h0) if theta_old is None else theta_old
        F = (curves.moisture_content(h) - th_old) / dt * phi * dx
        F += (fd.Constant(p_.specific_storage) * curves.saturation(h)
              * (h - h0) / dt * phi * dx)

        if source is not None:
            F += -source * phi * dx

        # --- interior flux: diffusion + gravity ---
        F += fd.inner(K * fd.grad(h), fd.grad(phi)) * dx
        F += fd.inner(K * e_z, fd.grad(phi)) * dx

        # --- matrix advection: water carried by the compacting firn ---
        if firn_velocity is not None:
            theta = curves.moisture_content(h)
            F += -fd.inner(theta * firn_velocity * e_z, fd.grad(phi)) * dx

        # --- facet terms: SIPG for diffusion, donor-cell upwind for gravity ---
        sigma = self._penalty(V) * fd.avg(fd.FacetArea(mesh) / fd.CellVolume(mesh))
        F += sigma * fd.inner(fd.jump(phi, n), fd.avg(K) * fd.jump(h, n)) * dS
        F += -fd.inner(fd.avg(K * fd.grad(phi)), fd.jump(h, n)) * dS
        F += -fd.inner(fd.jump(phi, n), fd.avg(K * fd.grad(h))) * dS

        # The gravity flux vector is -K e_z (gravity drives water down), so the
        # donor cell is the one that flux leaves, i.e. the one whose outward
        # normal it points along. Taking the positive part of dot(K*e_z, n)
        # instead picks the receiving cell, and a wetting front then advances
        # at the dry cell's conductivity rather than its own.
        qn = 0.5 * (fd.dot(-K * e_z, n) + abs(fd.dot(-K * e_z, n)))
        F += fd.jump(phi) * (qn("+") - qn("-")) * dS

        if firn_velocity is not None:
            theta = curves.moisture_content(h)
            fadv = theta * firn_velocity * e_z
            an = 0.5 * (fd.dot(fadv, n) + abs(fd.dot(fadv, n)))
            F += fd.jump(phi) * (an("+") - an("-")) * dS

        # --- boundaries ---
        sigma_ext = self._penalty(V) * fd.FacetArea(mesh) / fd.CellVolume(mesh)
        for bc_id, spec in bcs.items():
          # A boundary may carry more than one condition, and they are additive.
          for kind, value in spec.items():
            if kind == "flux":
                # Volumetric flux positive INTO the domain [m/s]. Pass an
                # fd.Constant if it varies in time: the form is assembled once
                # and a float is baked in permanently.
                F += -value * phi * ds(bc_id)
            elif kind == "free":
                # Free drainage: zero pressure-head gradient, water leaves
                # under gravity at the local conductivity, q.n = K(h). An
                # alternative outlet, retained and tested; the aquifer
                # experiments use "advect" instead. Written against the
                # gravity flux -K e_z so it carries the boundary's own
                # orientation rather than assuming an outlet at the base.
                F += fd.dot(-K * e_z, n) * phi * ds(bc_id)
            elif kind == "advect":
                # Natural advective outflow, Darcy flux zero. The firn below
                # the aquifer keeps compacting and moving down at the velocity
                # the continuity integration already imposes, and the water in
                # its pores moves with it, so what crosses the base is simply
                # theta * w - no drainage law and nothing to tune. This is the
                # boundary term the matrix-advection integration by parts
                # leaves behind, evaluated with the interior (donor-cell)
                # value, which is the upwind choice at an outflow boundary.
                #
                # It also caps the outflow physically at porosity * |w| at the
                # base: recharge above that must fill the column, and a run
                # that floods under this condition floods for a physical
                # reason rather than a boundary-condition artefact.
                if firn_velocity is None:
                    raise ValueError(
                        f"boundary {bc_id}: 'advect' needs firn_velocity")
                theta_b = curves.moisture_content(h)
                F += theta_b * firn_velocity * fd.dot(e_z, n) * phi * ds(bc_id)
            elif kind == "h":
                diff = h - value
                Fb = 2.0 * sigma_ext * phi * K * diff
                Fb -= fd.inner(K * fd.grad(phi), n) * diff
                Fb -= fd.inner(phi * n, K * fd.grad(h))
                # Upwinded like the interior gravity flux, and against the
                # same vector -K e_z: the interior conductivity where the
                # flux leaves, the exterior (Dirichlet) one where it enters.
                fn_i = fd.dot(-K * e_z, n)
                fn_g = fd.dot(-curves.relative_conductivity(value) * e_z, n)
                Fb += phi * (0.5 * (fn_i + abs(fn_i))
                             + 0.5 * (fn_g - abs(fn_g)))
                F += Fb * ds(bc_id)
            else:
                raise ValueError(
                    f"boundary {bc_id}: unknown type {kind!r} "
                    "(use 'h', 'flux', 'free' or 'advect')")
        return F
