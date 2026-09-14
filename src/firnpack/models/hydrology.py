"""Meltwater percolation, storage and refreezing in compacting firn.

.. note::

   **Superseded for aquifer work by** :mod:`firnpack.models.firn_richards`.

   This module writes the water balance in the *moisture form*, with the bulk
   water content ``W`` as the unknown. That form conserves mass for free, but
   it has no variable left once the pore space fills: ``S = W/(rho_w phi)``
   saturates, the flux stops responding, and there is nothing to carry a water
   table. The pore-pressure penalty on ``S > 1`` below is a numerical stiffness
   standing in for a free surface, not a material property.

   ``firn_richards`` uses the mixed head form of Celia et al. (1990), which
   keeps the exact mass conservation *and* stays well posed through saturation,
   so the water table is an interior free surface the solution finds on its
   own. Use this module for unsaturated percolation and refreezing in a column
   that never saturates; use ``firn_richards`` for anything with an aquifer in
   it. ``test_hydrology.py`` carries an xfailed perching test that is the
   concrete evidence for the split.

The formulation follows the continuum mixture model of Meyer & Hewitt (2017,
*The Cryosphere* 11, 2799-2813), specialised to the 1D column and to the field
set this package already carries.

Coordinates and field conventions (shared with :mod:`firnpack.models.firn`)
--------------------------------------------------------------------------
``x`` is **height above the column base**, increasing upward: the base is
boundary id 1 (``x = 0``) and the surface is boundary id 2 (``x = H``). Gravity
therefore drives water in the ``-x`` direction. This matters for every sign
below and is the convention ``FirnModel.overburden_stress`` already assumes.

* ``rho``  - bulk density of the **ice matrix** [kg m^-3], so the ice volume
  fraction is ``rho/rho_i`` and the porosity is ``phi = 1 - rho/rho_i``.
* ``W``    - mass of liquid water per unit **bulk** volume [kg m^-3]. The pore
  saturation is ``S = W / (rho_w phi)``.
* ``H``    - specific enthalpy of the matrix [J kg^-1]. ``FirnParameters`` sets
  ``T_ref = T_m``, so ``H = c_i (T - T_m)`` is the cold content: ``H < 0`` is
  cold firn, ``H = 0`` is temperate firn at the melting point.

Correspondence with Meyer & Hewitt
----------------------------------
Their ice and water mass balances, with a phase-change exchange term ``m``,

    d/dt[rho_i(1-phi)] + d/dz[rho_i(1-phi) w_i] = +m        (ice)
    d/dt[rho_w phi S]  + d/dz[rho_w phi S  w_w] = -m        (water)

become, in the variables above,

    D(rho)/Dt + rho dw/dx = m                               (see FirnModel)
    dW/dt + d/dx[W w + q] = -m                              (:meth:`water_form`)

where ``q`` is the Darcy mass flux **relative to the moving ice matrix** and
``w`` is the ice velocity recovered by integrating compaction from the base to
the surface. Keeping ``q`` relative to the matrix is what lets the existing
base-to-surface velocity integration stand unchanged apart from the ``m``
source: see :meth:`firnpack.models.firn.FirnModel.velocity_delta`.

Where this departs from Meyer & Hewitt
--------------------------------------
1. **Permeability** is tied to the model's own prognostic grain size via
   Calonne et al. (2012) rather than to porosity alone; see
   :meth:`permeability`.
2. Phase change uses the same local equilibrium closure, but written as a
   rate limited by both the available water and the available cold content so
   that a single backward-Euler step can never overshoot ``H = 0`` or drive
   ``W < 0``; see :meth:`phase_change_rate`.
"""

from dataclasses import dataclass

import firedrake as fd

try:  # optional: only needed for the lumped-mass measure
    from finat.point_set import PointSet
    from finat.quadrature import QuadratureRule
except Exception:  # pragma: no cover
    PointSet = QuadratureRule = None

from firnpack.constants import (
    ice_density, water_density, latent_heat,
    water_viscosity, gravity, heat_capacity, melting_temperature,
    closeoff_density,
)


@dataclass
class HydrologyParameters:
    """Parameters for percolation, storage and refreezing."""

    # --- physical constants -------------------------------------------------
    rho_i: float = ice_density
    rho_w: float = water_density
    L: float = latent_heat
    g: float = gravity
    mu_w: float = water_viscosity
    c_i: float = heat_capacity
    T_m: float = melting_temperature

    # --- regularization -----------------------------------------------------
    phi_min: float = 1e-4     # floor on porosity, keeps S finite as rho -> rho_i
    S_min: float = 0.0        # floor on saturation (0 keeps dry firn exactly dry)

    # --- retention and relative permeability --------------------------------
    S_res: float = 0.05       # irreducible saturation held against gravity
    m_S: float = 3.0          # relative-permeability exponent, k_rel = S_eff^m_S
    # Floor on the *capillary* mobility only (see darcy_flux). Below S_res the
    # gravity mobility is exactly zero, which leaves cells ahead of a wetting
    # front completely uncoupled: the front can then only advance once a node
    # fills past S_res on its own, and in a lumped scheme it stalls outright.
    # aquipack's Richards curves take the same remedy for the same reason -
    # a small relative-conductivity floor, physically film and vapour
    # transport, that keeps dried cells coupled to their neighbours. Applying
    # it to the capillary term alone is what keeps dry firn an exact fixed
    # point: a constant floor on the gravity term would drain water that is
    # not there.
    k_cap_min: float = 1.0e-4

    # --- permeability -------------------------------------------------------
    # "calonne2012": k = perm_c * r^2 * exp(perm_b * rho)   (grain-size based)
    # "porosity":    k = K0 * phi**m_phi                    (legacy phi-only law)
    permeability_law: str = "calonne2012"
    perm_c: float = 3.0       # Calonne et al. (2012) prefactor [-]
    perm_b: float = -0.013    # Calonne et al. (2012) density exponent [m^3 kg^-1]
    perm_scale: float = 1.0   # calibration multiplier on k (inversion control)
    r2_default: float = 2.5e-7   # fallback grain radius^2 [m^2] (r = 0.5 mm)
    K0: float = 1e-7          # legacy hydraulic conductivity scale [m s^-1]
    m_phi: float = 3.0        # legacy porosity exponent
    # Pore-connectivity cutoff at close-off. Calonne's fit is calibrated on
    # snow and firn and never actually reaches zero: at 900 kg/m^3 it still
    # returns K = 3.4e-5 m/s, so an "ice lens" stays four times more permeable
    # than a heavy melt supply and nothing can perch on it. Above close-off the
    # pores are by definition isolated and the permeability is zero, so the fit
    # is multiplied by the connected-porosity fraction
    #     f = clamp((rho_i - rho)/(rho_i - rho_co), 0, 1)
    # raised to ``conn_exp``. f = 1 for all rho <= rho_co, so the Calonne law is
    # untouched throughout the range it was calibrated on.
    closeoff_cutoff: bool = True
    rho_co: float = closeoff_density   # 815 kg m^-3
    conn_exp: float = 3.0

    # --- capillarity and pore pressure --------------------------------------
    # Below saturation the matric potential is suction,
    # Psi = -psi0 (1 - S): zero when saturated, most negative when dry, so
    # water is drawn from wet toward dry firn.
    #
    # At and above saturation the pore space cannot take more water, and the
    # potential becomes a positive pore pressure that stiffens sharply. This is
    # what produces a water table: a slight over-saturation builds a head
    # gradient that balances gravity, so the saturated zone stops accepting
    # inflow instead of draining at full conductivity forever. Without it a
    # column with an impermeable base has no equilibrium at all - W at the
    # bottom node grows without bound and the nonlinear solve fails at any step
    # size, which is exactly what it did.
    psi0: float = 0.5         # capillary head scale [m]
    # psi_sat is a stiffness, not a material property: it enforces S <= 1 by
    # penalty. In hydrogeological terms it is an effective specific storage
    # S_s = phi/psi_sat, so 1000 m corresponds to S_s ~ 5.6e-4 1/m - a few
    # hundred times more compressible than a real aquifer skeleton (1e-5 to
    # 1e-6 1/m, cf. the Biot formulation in ../aquifer), but stiff enough to
    # hold the over-saturation to under half a percent while staying tractable
    # for Newton. Measured peak over-saturation: 65% at 10 m, 17% at 50,
    # 4.5% at 200, 0.4% at 1000, at equal cost.
    psi_sat: float = 1000.0   # saturated pore-pressure stiffness [m]
    sat_width: float = 0.02   # smoothing width in saturation for the transition
    S_pot_max: float = 2.0    # cap on S used for the potential (keeps it finite)

    # --- numerics -----------------------------------------------------------
    # Gravity drainage is a nonlinear advection (k_rel ~ S^m), so wetting fronts
    # are near-shocks. Unstabilised CG1 oscillates across them and drives W
    # negative - 15-22% of peak at the resolutions this column runs at, and it
    # barely improves with a smaller step because the error is spatial. Upwind
    # (streamline-diffusion) stabilisation restores monotonicity at the cost of
    # first-order accuracy across the front. ``stab_scale = 0`` disables it.
    stabilization: bool = True
    stab_scale: float = 1.0
    # Lump the mass matrix in the water equation. Upwinding alone does not make
    # the scheme monotone: the CG1 consistent mass matrix (h/6)[1,4,1] rings
    # across a sharp front on its own, which showed up as a decaying
    # alternating tail below the wetting front (-7.0, +1.9, -0.5, +0.1, ratio
    # -0.27 per node - the consistent-mass signature, not a flux error).
    # Lumped mass + upwind flux gives an M-matrix, hence no undershoot.
    lumped_mass: bool = True

    # --- time weighting -----------------------------------------------------
    theta_W: float = 0.5
    theta_H: float = 0.5


class HydrologyModel:
    """Weak forms and closures for water in compacting firn."""

    def __init__(self, params: HydrologyParameters | None = None):
        self.params = params or HydrologyParameters()

    # ------------------------------------------------------------------
    # Measures
    # ------------------------------------------------------------------
    def _mass_measure(self, mesh):
        """``dx`` for mass/reaction terms: vertex quadrature when lumping.

        Vertex (trapezoidal) quadrature on a CG1 interval puts all the weight on
        the nodes, which is exactly mass lumping. Falls back to the consistent
        measure when lumping is off or the quadrature machinery is unavailable.
        """
        dx = fd.dx(domain=mesh)
        p = self.params
        if not p.lumped_mass or QuadratureRule is None:
            return dx
        if str(mesh.ufl_cell()) != "interval":
            return dx
        rule = QuadratureRule(PointSet([[0.0], [1.0]]), [0.5, 0.5])
        return fd.dx(domain=mesh, scheme=rule)

    # ------------------------------------------------------------------
    # Pore structure
    # ------------------------------------------------------------------
    def porosity(self, rho):
        """Bulk porosity ``phi = 1 - rho/rho_i``, floored at ``phi_min``."""
        p = self.params
        return fd.max_value(1.0 - rho / p.rho_i, p.phi_min)

    def saturation_raw(self, W, rho):
        """Unclamped pore saturation, floored at zero.

        ``S > 1`` is not physical as a fraction, but it is the natural measure
        of how much the pore space is over-filled, and the pore-pressure branch
        of the potential needs it: clamping here is what removed the water
        table.
        """
        p = self.params
        phi = self.porosity(rho)
        return fd.max_value(W / (p.rho_w * phi), 0.0)

    def saturation(self, W, rho):
        """Pore saturation ``S = W/(rho_w phi)``, clipped to ``[S_min, 1]``.

        The clip bounds only the *constitutive* use of ``S`` (relative
        permeability, capillarity). ``W`` itself remains the conserved
        variable, so water that arrives once ``S = 1`` still accumulates and
        forms a saturated zone; it simply stops increasing the local
        conductivity.
        """
        p = self.params
        phi = self.porosity(rho)
        S_raw = W / (p.rho_w * phi)
        return fd.max_value(fd.min_value(S_raw, 1.0), p.S_min)

    # ------------------------------------------------------------------
    # Hydraulic properties
    # ------------------------------------------------------------------
    def permeability(self, rho, grain_radius2=None):
        """Intrinsic permeability ``k`` [m^2].

        Default is Calonne et al. (2012), ``k = 3 r^2 exp(-0.013 rho)``, fitted
        to 3D tomography of snow and firn. It is preferred over a porosity-only
        law because this package already carries ``r^2`` as a prognostic in
        Mode B, so percolation inherits the model's own grain-growth history -
        the coupling Meyer & Hewitt's continuum framework calls for but leaves
        to a closure.

        Falls back to the legacy ``k ~ phi**m_phi`` form when
        ``permeability_law == "porosity"``, or to ``r2_default`` when no grain
        size is supplied.
        """
        p = self.params
        if p.permeability_law == "porosity":
            # Legacy law: K0 is a hydraulic conductivity, convert to a
            # permeability so callers see consistent units.
            phi = self.porosity(rho)
            K_hyd = fd.Constant(p.K0) * phi ** p.m_phi
            return K_hyd * p.mu_w / (p.rho_w * p.g)

        r2 = self.r2_or_default(grain_radius2)
        k = fd.Constant(p.perm_c) * r2 * fd.exp(fd.Constant(p.perm_b) * rho)
        if p.closeoff_cutoff:
            k = k * self.pore_connectivity(rho) ** p.conn_exp
        return fd.Constant(p.perm_scale) * k

    def pore_connectivity(self, rho):
        """Connected-pore fraction, 1 below close-off and 0 at solid ice."""
        p = self.params
        f = (p.rho_i - rho) / (p.rho_i - p.rho_co)
        return fd.max_value(fd.min_value(f, 1.0), 0.0)

    def r2_or_default(self, grain_radius2):
        """Grain radius squared, defaulting to ``params.r2_default``."""
        p = self.params
        if grain_radius2 is None:
            return fd.Constant(p.r2_default)
        return fd.max_value(grain_radius2, 1e-12)

    def hydraulic_conductivity(self, rho, grain_radius2=None):
        """Saturated hydraulic conductivity ``K = k rho_w g / mu`` [m s^-1]."""
        p = self.params
        k = self.permeability(rho, grain_radius2)
        return k * p.rho_w * p.g / p.mu_w

    def k_rel(self, S):
        """Relative permeability ``k_rel = ((S - S_res)/(1 - S_res))^m_S``."""
        p = self.params
        S_eff = fd.max_value(S - fd.Constant(p.S_res), 0.0) / (1.0 - p.S_res)
        return S_eff ** p.m_S

    def capillary_potential_prime(self, S_raw):
        """``dPsi/dS``: capillary suction below saturation, pore pressure above.

        Positive throughout: the potential *rises* toward zero as the firn wets,
        so water moves from wet toward dry and the term is diffusive. (The
        original ``Psi = +psi0 (1 - S)`` had the opposite sign, making the
        capillary flux anti-diffusive and sharpening fronts without bound.)

        The slope switches smoothly from ``psi0`` to the much larger ``psi_sat``
        as ``S`` passes 1. In the saturated zone that steep slope is a pore
        pressure: it lets ``dPsi/dx`` reach ``-1`` and cancel gravity, which is
        the hydrostatic balance that holds a water table up.
        """
        p = self.params
        ramp = 0.5 * (1.0 + fd.tanh((S_raw - 1.0) / fd.Constant(p.sat_width)))
        return fd.Constant(p.psi0) + fd.Constant(p.psi_sat - p.psi0) * ramp

    def darcy_flux(self, W, rho, grain_radius2=None):
        """Darcy mass flux of water **relative to the ice matrix** [kg m^-2 s^-1].

        With ``x`` measured upward the hydraulic head is ``h = Psi + x``, so

            q = -rho_w K k_rel (dPsi/dx + 1)

        and the gravitational part is negative, i.e. downward, as it must be.
        """
        p = self.params
        K = self.hydraulic_conductivity(rho, grain_radius2)
        # Mobility uses the clamped saturation: a fully saturated pore cannot
        # conduct better than fully saturated.
        krel = self.k_rel(self.saturation(W, rho))
        # The potential uses the unclamped saturation, capped only to keep it
        # finite, so the saturated branch can build pressure.
        S_pot = fd.min_value(self.saturation_raw(W, rho), p.S_pot_max)
        dPsidx = self.capillary_potential_prime(S_pot) * S_pot.dx(0)

        # Gravity and capillarity carry different mobilities on purpose: only
        # the capillary term gets the floor, so a dry column has exactly zero
        # flux (dS/dx = 0 there) while a wetting front still seeps forward.
        q_grav = -p.rho_w * K * krel
        q_cap = -p.rho_w * K * (krel + fd.Constant(p.k_cap_min)) * dPsidx
        return q_grav + q_cap

    def wave_speed(self, W, rho, w, grain_radius2=None):
        """Characteristic speed ``dF/dW`` of the water flux [m s^-1].

        The flux is ``F = W w + q(W)``, so the speed at which a disturbance in
        ``W`` travels is

            dF/dW = w - (K/phi) dk_rel/dS

        the second term being the Buckley-Leverett speed of gravity drainage.
        It is what sets both the stabilisation coefficient and the step size a
        front can be resolved at.
        """
        p = self.params
        S = self.saturation(W, rho)
        phi = self.porosity(rho)
        K = self.hydraulic_conductivity(rho, grain_radius2)
        S_eff = fd.max_value(S - fd.Constant(p.S_res), 0.0) / (1.0 - p.S_res)
        dkrel = p.m_S * S_eff ** (p.m_S - 1.0) / (1.0 - p.S_res)
        return w - K * dkrel / phi

    # ------------------------------------------------------------------
    # Phase change (Meyer & Hewitt local equilibrium)
    # ------------------------------------------------------------------
    def phase_change_increment(self, H, W, rho):
        """Mass of water that freezes in this step, per unit bulk volume [kg m^-3].

        Positive is **freezing** (water -> ice, latent heat released into the
        matrix); negative is melting.

        Meyer & Hewitt close the system by requiring local thermodynamic
        equilibrium: liquid water and cold firn cannot coexist, so the matrix
        relaxes to ``H = 0`` wherever water remains. Equilibrium is
        instantaneous, so the increment carries no timescale:

            f = clip( -rho H / L,  -rho,  W )

        The three pieces are, in order: the freezing that consumes exactly the
        local cold content (``rho c_i (T_m - T) = -rho H`` joules per cubic
        metre, each kilogram of ice released contributing ``L``); the limit that
        melting cannot consume more ice than is present; and the limit that
        freezing cannot consume more water than is present.

        Applied pointwise as ``W -= f``, ``H += L f / rho`` (see
        :meth:`firnpack.solvers.hydrology_solver.HydrologySolver.apply_phase_change`)
        this is bounded by construction: ``W`` cannot go negative because
        ``f <= W``, and ``H`` cannot overshoot zero from either side because
        ``f`` is capped by exactly the cold content.

        Dry firn is a fixed point: with ``W = 0`` and ``H < 0`` the target
        ``-rho H / L`` is positive, ``min`` selects ``W = 0``, and ``f`` is
        identically zero. The dry model is therefore bit-for-bit unchanged.

        The available water is clamped at zero deliberately. Using ``W`` raw
        would make this routine quietly repair a negative ``W`` produced
        upstream by transport: with ``H = 0`` it would pick ``f = W < 0``,
        "melt" that deficit away and leave ``W = 0`` exactly, hiding a mass
        error inside a physically-named term. Whether ``W`` stays non-negative
        is the transport scheme's responsibility, and it should be visible when
        it fails - see ``water_positivity_error``.
        """
        p = self.params
        target = -rho * H / p.L                       # kg m^-3 to bring H to 0
        avail = fd.max_value(W, 0.0)                  # see note on negative W
        capped = fd.min_value(target, avail)          # cannot freeze absent water
        return fd.max_value(capped, -rho)             # cannot melt absent ice

    def enthalpy_increment(self, f, rho):
        """Matrix enthalpy change ``L f / rho`` [J kg^-1] from freezing ``f``.

        The firn enthalpy is carried per unit mass, so a volumetric latent
        release ``L f`` [J m^-3] enters divided by density, exactly as the basal
        heat flux does in :meth:`firnpack.models.firn.FirnModel.enthalpy_form`.
        """
        p = self.params
        return p.L * f / fd.max_value(rho, 1.0)

    # ------------------------------------------------------------------
    # Weak forms
    # ------------------------------------------------------------------
    def water_form(self, W, W_old, rho, w, test, dt, *,
                   grain_radius2=None,
                   surface_flux=None, surface_id=2,
                   basal_flux=None, base_id=1):
        """Weak form for bulk water content ``W``.

            dW/dt + d/dx[ W w + q ] = 0

        Phase change is **not** here. Applying the freezing sink through the
        FEM mass matrix spreads a pointwise bound across neighbouring nodes, so
        a node can be driven negative even though the increment was capped by
        its own water content; clipping that away then creates mass. Phase
        change is instead applied pointwise after transport, where it is
        exactly conservative and exactly bounded - see
        :meth:`phase_change_increment`.

        Boundary terms
        --------------
        Integrating the flux divergence by parts leaves the surface integral
        ``(F.n) psi ds``, which the previous version dropped. Dropping it is not
        a neutral simplification: it silently imposes **zero water flux at both
        ends**, so no melt can ever enter at the surface and nothing can drain
        at the base. That is why ``W`` stayed identically zero regardless of
        forcing.

        ``surface_flux`` is the downward meltwater mass flux arriving at the
        surface [kg m^-2 s^-1], positive into the column. ``basal_flux`` is the
        downward loss at the base, positive out of the column; pass ``None`` for
        an impermeable bed.
        """
        p = self.params
        psi = test
        mesh = W.function_space().mesh()
        dx = fd.dx(domain=mesh)
        dx_m = self._mass_measure(mesh)
        ds = fd.ds(domain=mesh)

        thetaW = fd.Constant(p.theta_W)
        W_mid = thetaW * W + (1.0 - thetaW) * W_old

        q_mid = self.darcy_flux(W_mid, rho, grain_radius2)
        flux = w * W_mid + q_mid

        F = (W - W_old) / dt * psi * dx_m - flux * psi.dx(0) * dx

        # Upwind stabilisation: artificial diffusion nu = scale * h |a| / 2,
        # the streamline-diffusion coefficient that makes the advection
        # operator monotone. Without it the wetting front rings and W goes
        # negative; see HydrologyParameters.stabilization.
        if p.stabilization and p.stab_scale > 0.0:
            h = fd.CellDiameter(mesh)
            a = self.wave_speed(W_mid, rho, w, grain_radius2)
            nu = fd.Constant(0.5 * p.stab_scale) * h * fd.sqrt(a * a + 1e-30)
            F += nu * W_mid.dx(0) * psi.dx(0) * dx

        # --- surface influx: outward normal is +x, inflow is -F.n ---
        if surface_flux is not None:
            if isinstance(surface_flux, (int, float)):
                surface_flux = fd.Constant(float(surface_flux))
            F += -surface_flux * psi * ds(int(surface_id))

        # --- basal outflow: outward normal is -x, so F.n = -(downward flux) ---
        if basal_flux is not None:
            if isinstance(basal_flux, (int, float)):
                basal_flux = fd.Constant(float(basal_flux))
            F += basal_flux * psi * ds(int(base_id))

        return F

    def enthalpy_form(self, H, H_old, rho, w, K_func, test, dt):
        """Weak form for **matrix** enthalpy with refreezing.

            dH/dt + w dH/dx - d/dx(K dH/dx) = 0

        Latent heating from refreezing is applied pointwise alongside the
        matching water sink, for the reason given in :meth:`water_form`.

        The advective and diffusive terms match
        :meth:`firnpack.models.firn.FirnModel.enthalpy_form` term for term, so
        the wet model reduces to the dry one exactly.

        Why there is no ``-d(L q)/dx`` latent-transport term
        ----------------------------------------------------
        Meyer & Hewitt evolve a *mixture* enthalpy that already contains the
        liquid, so in their formulation the water flux carries a latent flux
        ``L q`` that must appear explicitly. Here ``H`` is the enthalpy of the
        ice matrix alone and the liquid is carried by its own conserved field
        ``W``, which is what keeps the dry limit of this model bit-for-bit
        identical to :class:`~firnpack.models.firn.FirnModel`.

        In that two-field split the latent energy travelling with the water is
        *already* accounted for by transporting ``W``: every kilogram of liquid
        is at ``T_m`` and carries exactly ``L`` relative to ice at ``T_m``, and
        that energy enters the matrix only when the water actually freezes,
        through ``m``. Including ``-d(L q)/dx`` as well double counts it, which
        deposits latent heat into the matrix without any phase change, drives
        ``H`` above zero, and then melts ice that was never there - the model
        creates water. The two formulations are equivalent; mixing them is not.

        Notes
        -----
        The previous version formed ``W_mid = thetaW*W + (1 - thetaW)*W``, which
        is just ``W``: the time weighting was a no-op. That expression, and the
        ``W``/``T_func`` arguments it needed, are gone with the latent-transport
        term they belonged to.
        """
        p = self.params
        psi = test
        mesh = H.function_space().mesh()
        dx = fd.dx(domain=mesh)

        thetaH = fd.Constant(p.theta_H)
        H_mid = thetaH * H + (1.0 - thetaH) * H_old

        # Lag the diffusivity on H_old, as the dry form does, so a nonlinear
        # conductivity law cannot put the trial function inside an exp().
        K_mid = K_func(H_old, rho)

        F = (
            (H - H_old) / dt * psi * dx
            + w * H_mid.dx(0) * psi * dx
            + K_mid * H_mid.dx(0) * psi.dx(0) * dx
        )

        return F
