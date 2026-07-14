from dataclasses import dataclass
import firedrake as fd

from firnpack.constants import (
    year,
    ice_density,
    gravity,
    heat_capacity,
    thermal_diffusivity,
    melting_temperature,
    gas_constant,
    Ec,
    Eg,
    kg,
    kcHh,
    kcLw,
)


@dataclass
class FirnParameters:
    # physical constants from icepack-style constants
    rho_i: float = ice_density
    c_i: float = heat_capacity
    K_cold: float = thermal_diffusivity  # κ = k/(ρ c)
    T_m: float = melting_temperature
    T_ref: float = melting_temperature
    spy: float = year
    g: float = gravity
    phi_min: float = 1e-8

    # densification “knobs”
    c0: float = 1.0e-13
    c_mult_high: float = 3.0
    rho_m: float = 550.0
    kc0: float = kcHh              # 9.2e-9 m^3 s kg^-1 (low-density)
    kc1: float = kcLw              # 3.7e-9 m^3 s kg^-1 (high-density)
    Ec: float = Ec                 # 60e3 J mol^-1
    Eg: float = Eg                 # 42.4e3 J mol^-1
    kg: float = kg                 # 1.3e-7 m^2 s^-1
    R: float = gas_constant        # 8.3144621 J mol^-1 K^-1
    Tavg: float = 248.0  

    # time discretization parameters
    theta_H: float = 0.5
    theta_rho: float = 0.878
    theta_w: float = 0.878

    # diffusivity in temperate ice
    K_temp_factor: float = thermal_diffusivity
    # full-density / grain-size prognostic options
    # These are used when solving for (rho, sigma, r^2) instead of the simplified Arthern density law.
    theta_liniger: float = 0.5     # Liniger theta-scheme for advection-reaction (Cummings-style)
    eps_r2: float = 1.0e-12        # small regularization added to r^2 in denominators (keeps forms smooth)
    r2_surf: float = 2.5e-7        # surface grain-size-squared (m^2); ~ (0.5 mm)^2 by default
    r2_f: float = 1.0e-2           # saturation grain-size-squared (m^2) (Kingslake et al. 2022)
    use_r2_saturation: bool = False

    # ------------------------------------------------------------------
    # Basal thermal boundary condition (dry firn)
    # ------------------------------------------------------------------
    # Total *upward* heat flux into the firn at the base (W m^-2).
    #
    # This can represent:
    #   - geothermal heat flux from the underlying ice/bed
    #   - frictional/shear heating at the base (e.g. tau_b * u_b)
    #
    # The default is 0.0 (insulating base) to preserve legacy behaviour.
    basal_heat_flux_W_m2: float = 0.0

    # Thermal conductivity of pure ice [W m^-1 K^-1].
    # Used in thermal_diffusivity(): k = k_ice * (rho/rho_i)^2.
    # Default 2.1 from Evan's model.
    k_ice: float = 2.1

    # Conductivity law selector for thermal_diffusivity().
    #   "quadratic"  (default): k = k_ice * (rho/rho_i)^2  (legacy behaviour)
    #   "calonne2019": Calonne et al. 2019 (GRL) snow/firn effective k(rho, T)
    #       with a smooth logistic transition at calonne_rho_t between the
    #       snow regression (Calonne 2011) and the firn/porous-ice regression,
    #       each multiplied by its own calibration scale (1.0 = as published).
    #       T-dependence via the Yen-1981 ice-conductivity ratio, reference
    #       -3 C. NOTE: verify constants against Calonne et al. 2019 before
    #       publication (air-conductivity correction neglected: small at firn
    #       densities and near-constant at a quasi-isothermal site).
    conductivity_law: str = "quadratic"
    k_snow_scale: float = 1.0      # multiplies the snow branch (rho < ~rho_t)
    k_firn_scale: float = 1.0      # multiplies the firn/ice branch
    calonne_rho_t: float = 450.0   # transition density [kg m^-3]
    calonne_beta: float = 0.02     # transition sharpness [m^3 kg^-1]

    # Firedrake boundary id for the *base* of an IntervalMesh.
    # By convention: 1 = left boundary (x=0), 2 = right boundary (x=H).
    base_boundary_id: int = 1

    # ------------------------------------------------------------------
    # Free surface tracking
    # ------------------------------------------------------------------
    # Smoothing width for the surface mask χ(x,h) = 0.5*(1+tanh((h-x)/δ)).
    # Should be ~0.5 * minimum cell size near the surface.
    mask_width: float = 0.5     # meters

    # ------------------------------------------------------------------
    # Herron & Langway (1980) empirical densification parameters.
    # Used when model.densification_rate_fn = herron_langway.
    # Convention: activation energies in J/mol, used with SI R=8.314
    # (matches CFM / Kingslake 2022 implementations; see densification.py).
    # ------------------------------------------------------------------
    hl_k0_prefactor: float = 11.0       # stage 1 prefactor  (1/yr)
    hl_k1_prefactor: float = 575.0      # stage 2 prefactor  ((m ice)⁻½/yr)
    hl_Ea_stage1: float = 10160.0       # stage 1 activation energy (J/mol)
    hl_Ea_stage2: float = 21400.0       # stage 2 activation energy (J/mol)
    hl_stage2_shape: float = 1.0        # stage 2 (rho_i-rho) shape exponent; 1 = H&L
    # Optional smooth cutoff of the H&L stage-2 rate approaching pore close-off
    # (rate *= 0.5*(1 - tanh((rho - cutoff)/8))). None = off (legacy). Used by
    # deep-column configurations so advecting parcels cannot ride into the
    # rho_i barrier kink (density_form's max_value(rho-rho_i, 0)); bubble
    # compression beyond close-off is not modelled.
    hl_deep_cutoff_rho: float | None = None

    # ------------------------------------------------------------------
    # Compressible Stokes densification (Gagliardini & Meyssonnier 1997).
    # Used when model.densification_rate_fn = stokes_compressible.
    # ------------------------------------------------------------------
    stokes_n_glen: float = 1.0          # Glen exponent (1=linear, 3=standard)
    stokes_A_glen: float = 6.8e-26      # Glen rate factor [Pa^-n s^-1] (~-25°C)
    stokes_Q_glen: float = 60.0e3       # activation energy for A(T) [J/mol]
    stokes_A_ref_T: float = 248.15      # reference temperature for A_glen [K]
    stokes_rhohat_thres: float = 0.81   # stage 1/2 threshold (relative density)
    stokes_c1_a: float = 13.22240       # stage 1 a-coefficient
    stokes_c2_a: float = 15.78652       # stage 1 a-coefficient
    stokes_c1_b: float = 15.09371       # stage 1 b-coefficient
    stokes_c2_b: float = 20.46489       # stage 1 b-coefficient
    stokes_E_lin: float = 5.0e9         # enhancement factor for n=1


class FirnModel:
    """Icepack-style firn column model for (H, rho, w).

    The densification rate law is pluggable via `densification_rate_fn`.
    Any callable with signature
        fn(rho, T, *, params, bdot, sigma=None, grain_radius2=None, rhoCoef=None, **kwargs)
    returning a UFL expression for dρ/dt [kg m⁻³ s⁻¹] may be supplied.
    See `firnpack.physics.densification` for built-in options.
    """

    def __init__(self, params: FirnParameters | None = None,
                 densification_rate_fn=None):
        self.params = params or FirnParameters()
        self.densification_rate_fn = densification_rate_fn

    # --- free surface tracking ---

    @staticmethod
    def surface_mask(x, h, delta):
        """Smooth indicator: 1 below the surface, 0 above.

        Parameters
        ----------
        x : UFL expression
            Spatial coordinate (from SpatialCoordinate(mesh)[0]).
        h : fd.Constant or UFL scalar
            Current surface height.
        delta : float or fd.Constant
            Smoothing width (meters).

        Returns
        -------
        UFL expression in [0, 1].
        """
        return 0.5 * (1.0 + fd.tanh((h - x) / delta))

    def surface_tendency(self, drhodt_masked, accumulation, rho_surf, dx):
        """Compute dh/dt from the global mass balance.

        dh/dt = (ȧ_snow − ∫ dρ/dt · χ dx) / ρ_surf

        where the integral is over the masked (active) domain.
        All operations are pyadjoint-safe (assemble + Constant arithmetic).

        Parameters
        ----------
        drhodt_masked : UFL expression
            Densification rate multiplied by the surface mask (χ · dρ/dt).
            Integrate this over the full domain to get the total mass
            change below the surface.
        accumulation : float or fd.Constant
            Surface accumulation rate in m ice-equivalent per year.
        rho_surf : float or fd.Constant
            Surface density (kg/m³).
        dx : UFL measure
            Integration measure (fd.dx(domain=mesh)).

        Returns
        -------
        dhdt : float
            Surface height tendency in m/s.
        """
        p = self.params
        # Mass flux onto the surface [kg m⁻² s⁻¹]
        a_snow = float(p.rho_i) * float(accumulation) / float(p.spy)
        # Total densification below the surface [kg m⁻² s⁻¹]
        mass_change = float(fd.assemble(drhodt_masked * dx))
        return (a_snow - mass_change) / float(rho_surf)

    # --- helper relations ---

    def porosity(self, rho):
        """Bulk porosity φ = 1 - ρ/ρ_i, regularized away from 0."""
        p = self.params
        phi = 1.0 - rho / p.rho_i
        return fd.max_value(phi, p.phi_min)

    def temperature_from_enthalpy(self, H):
        p = self.params
        return H / p.c_i + p.T_ref

    def _temperature(self, *, H=None, T=None):
        """Return a temperature UFL expression.

        Option B: prefer temperature derived from enthalpy H.
        If T is provided, it is used directly (backward compatibility).
        """
        if T is not None:
            return T
        if H is None:
            raise ValueError("Provide either enthalpy H (preferred) or explicit temperature T.")
        return self.temperature_from_enthalpy(H)

    def thermal_diffusivity(self, H, rho):
        """Effective thermal diffusivity κ = k_eff/(ρ c).

        Law selected by params.conductivity_law:
          "quadratic" (legacy): k = k_ice * (rho/rho_i)^2
          "calonne2019": k(ρ,T) = F(T) * [(1-θ)·s_s·k_snow(ρ) + θ·s_f·k_firn(ρ)]
              k_snow = 0.024 - 1.23e-4 ρ + 2.5e-6 ρ²   (Calonne 2011, -3 C)
              k_firn = 2.107 + 3.618e-3 (ρ - ρ_i)      (Calonne 2019, -3 C)
              θ(ρ)  = 1/(1 + exp(-2β(ρ - ρ_t)))        (ρ_t=450, β=0.02)
              F(T)  = exp(-5.7e-3 (T - 270.15))        (Yen-1981 k_ice ratio)
              s_s, s_f = k_snow_scale, k_firn_scale    (calibration controls)
        """
        p = self.params
        law = getattr(p, "conductivity_law", "quadratic")
        if law == "calonne2019":
            T = self._temperature(H=H)
            k_snow = 0.024 - 1.23e-4 * rho + 2.5e-6 * rho**2
            k_firn = 2.107 + 3.618e-3 * (rho - p.rho_i)
            theta = 1.0 / (1.0 + fd.exp(-2.0 * p.calonne_beta * (rho - p.calonne_rho_t)))
            FT = fd.exp(-5.7e-3 * (T - 270.15))
            ki = FT * ((1.0 - theta) * p.k_snow_scale * k_snow
                       + theta * p.k_firn_scale * k_firn)
        else:
            ki = p.k_ice * (rho / p.rho_i)**2
        ci = p.c_i
        return ki / (rho * ci)

    # ------------------------------------------------------------------
    # Overburden stress
    # ------------------------------------------------------------------

    def overburden_stress(self, rho, mesh, surface_id=2):
        """Compute overburden stress σ(x) = g ∫_x^H ρ dx'.

        Solves  dσ/dx = −g ρ,  σ(surface)=0  via a least-squares CG1
        variational problem:

            ∫ σ' v' dx = −g ∫ ρ v' dx,   σ(surface) = 0

        This minimises ‖σ' + gρ‖² in the derivative L² sense and uses the
        standard stiffness matrix, which is SPD with the Dirichlet BC.

        All operations are Firedrake-native (solve), so this is fully on
        the pyadjoint tape and the adjoint can differentiate through the
        stress computation.

        Returns a CG1 Function (Pa, positive in compression).
        """
        p = self.params
        V = fd.FunctionSpace(mesh, "CG", 1)

        # Lazily cache the sigma Function so the tape re-uses the same object
        if not hasattr(self, "_sigma_cache"):
            self._sigma_cache = fd.Function(V, name="overburden")
        sigma = self._sigma_cache

        trial = fd.TrialFunction(V)
        test = fd.TestFunction(V)

        # Least-squares weak form: ∫ σ' v' dx = −g ∫ ρ v' dx
        a = trial.dx(0) * test.dx(0) * fd.dx
        L = -p.g * rho * test.dx(0) * fd.dx

        bc = fd.DirichletBC(V, 0.0, surface_id)
        problem = fd.LinearVariationalProblem(a, L, sigma, bcs=[bc])
        solver = fd.LinearVariationalSolver(
            problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"}
        )
        solver.solve()

        return sigma

    # ------------------------------------------------------------------
    # Continuity-derived diagnostics
    # ------------------------------------------------------------------

    def vertical_strain_rate_from_continuity(self, rho, drhodt):
        """Vertical strain rate ε = ∂w/∂x implied by mass continuity.

        The velocity solve in this code uses the 1D continuity statement

            rho * (dw/dx) + drhodt = 0,

        where x is the *vertical* coordinate used in the mesh.

        Therefore

            ε = dw/dx = -drhodt / rho.

        Notes
        -----
        * This is often a more robust quantity to compare to ApRES-derived
          compaction strain rates than differentiating a (possibly DG) velocity
          field.
        * We guard the division with a small lower bound on rho.
        """
        return -drhodt / fd.max_value(rho, 1.0)

    def densification_rate_simple(self, rho, T):
        """
        Simple Arthern-style law just to get things moving.
        You can replace this with the full HL law later.
        """
        p = self.params
        c_lo = p.c0
        c_hi = p.c_mult_high * p.c0
        c = fd.conditional(fd.le(rho, p.rho_m), c_lo, c_hi)
        return c * (p.rho_i - rho)

    def densification_rate_arthern(self, rho, T, bdot):
        """
        Arthern et al. (2010) densification law with Ligtenberg (2011)
        accumulation dependence, exactly as in Evan's model.

        rho   : density field (kg m^-3)
        T     : temperature field (K)
        bdot  : mass accumulation (kg m^-2 s^-1)
        """
        p = self.params

        # Ligtenberg et al. (2011) accumulation-dependent corrections.
        # M0 applies to low-density regime (rho <= 550), M1 to high-density.
        # Reference: Cummings et al. (2013), um-fdm, Ligtenberg (2011).
        M0 = 1.435 - 0.151 * fd.ln(bdot * 1.0e3)
        M1 = 2.366 - 0.293 * fd.ln(bdot * 1.0e3)

        exp_factor = fd.exp(
            -p.Ec / (p.R * T) + p.Eg / (p.R * p.Tavg)
        )

        c0 = M0 * bdot * p.g * p.kc0 / p.kg * exp_factor
        c1 = M1 * bdot * p.g * p.kc1 / p.kg * exp_factor

        c = fd.conditional(fd.le(rho, p.rho_m), c0, c1)

        return c * (p.rho_i - rho)


    # ------------------------------------------------------------------
    # Full-density (Cummings/Kingslake-style) extensions
    # ------------------------------------------------------------------

    def grain_growth_rate_r2(self, T, r2=None):
        """Return grain-size-squared growth rate dr2/dt [m^2/s].

        By default we use the Arthern-style grain growth law:
            dr2/dt = k_g * exp(-E_g / (R T))

        If params.use_r2_saturation is True and r2 is provided, we apply a
        Kingslake-style saturation in a unit-consistent way:
            dr2/dt = k_g * exp(-E_g/(R T)) * (1 - r2/r2_f)

        Here k_g has units [m^2/s] (Arthern's k_a).
        """
        p = self.params
        base = p.kg * fd.exp(-p.Eg / (p.R * T))
        if p.use_r2_saturation and (r2 is not None):
            return base * (1.0 - r2 / p.r2_f)
        return base

    def densification_rate_full_density(self, rho, sigma, grain_radius2, rhoCoef, *, H=None, T=None):
        """Full-density densification rate dρ/dt [kg m^-3 s^-1].

        This is the stress- and grain-size-dependent form used in the
        (rho, sigma, r^2) system. A simple and smooth version is:

            dρ/dt = rhoCoef * exp(-E_c/(R T)) * (ρ_i - ρ) * σ / (r^2 + eps)

        where rhoCoef has units [m^3 s kg^-1] (like kc in Arthern).
        """
        p = self.params
        T = self._temperature(H=H, T=T)
        r_eff = grain_radius2 + p.eps_r2
        return rhoCoef * fd.exp(-p.Ec / (p.R * T)) * (p.rho_i - rho) * sigma / r_eff

    def stress_form(self, sigma, sigma_old, w, accumulation, test, dt):
        """Weak form for vertical stress evolution (simple burial loading + advection).

        We evolve sigma with:
            ∂σ/∂t + w ∂σ/∂z = bdot g
        using a Liniger theta-scheme for the advective term.
        """
        p = self.params
        psi = test
        mesh = sigma.function_space().mesh()
        dx = fd.dx(domain=mesh)

        theta = fd.Constant(p.theta_liniger)
        sig_mid = theta * sigma + (1.0 - theta) * sigma_old

        bdot = p.rho_i * accumulation / p.spy  # kg m^-2 s^-1
        dsigdt = bdot * p.g                    # Pa s^-1

        F = (sigma - sigma_old) / dt * psi * dx - dsigdt * psi * dx + w * sig_mid.dx(0) * psi * dx
        return F

    def grain_radius2_form(self, grain_radius2, grain_radius2_old, w, *, H=None, T=None, test, dt):
        """Weak form for grain-size-squared evolution (advection + Arrhenius growth)."""
        p = self.params
        T = self._temperature(H=H, T=T)
        xi = test
        mesh = grain_radius2.function_space().mesh()
        dx = fd.dx(domain=mesh)

        theta = fd.Constant(p.theta_liniger)
        r_mid = theta * grain_radius2 + (1.0 - theta) * grain_radius2_old

        dr2dt = self.grain_growth_rate_r2(T, r2=r_mid)

        F = (grain_radius2 - grain_radius2_old) / dt * xi * dx - dr2dt * xi * dx + w * r_mid.dx(0) * xi * dx
        return F, dr2dt

    def grain_radius2_form_dg(self, grain_radius2, grain_radius2_old, w, *, H=None, T=None, test, dt,
                               inflow_value=None):
        """DG weak form for grain-size-squared evolution with upwind advection.

        Uses explicit (old-time-level) upwind selection to keep the form linear
        in the trial function. At inflow boundaries uses inflow_value if given.
        """
        p = self.params
        T = self._temperature(H=H, T=T)
        xi = test
        mesh = grain_radius2.function_space().mesh()
        dx = fd.dx(domain=mesh)
        dS = fd.dS(domain=mesh)
        ds = fd.ds(domain=mesh)

        theta = fd.Constant(p.theta_liniger)
        r_mid = theta * grain_radius2 + (1.0 - theta) * grain_radius2_old

        dr2dt = self.grain_growth_rate_r2(T, r2=r_mid)

        n = fd.FacetNormal(mesh)
        w_n = w * n[0]

        # Upwind flux using Lax-Friedrichs-style splitting (linear in trial):
        #   flux = avg(w * r_mid) + 0.5 * |w_n| * jump(r_mid)
        # which is equivalent to selecting the upwind value.
        from ufl import algebra
        abs_w_n = fd.max_value(w_n, -w_n)

        # Temporal + source
        F = (grain_radius2 - grain_radius2_old) / dt * xi * dx
        F -= dr2dt * xi * dx

        # DG advection: integration by parts + upwind flux
        # IBP gives conservative ∂(wr)/∂z; subtract w_z·r for advective w·∂r/∂z.
        F -= r_mid * w * xi.dx(0) * dx                                    # volume (IBP)
        F -= w.dx(0) * grain_radius2_old * xi * dx                        # conservative → advective (explicit)
        F += fd.avg(w * r_mid) * fd.jump(xi, n[0]) * dS                   # central flux
        F += 0.5 * abs_w_n('+') * fd.jump(r_mid, n[0]) * fd.jump(xi, n[0]) * dS  # upwind correction

        # Exterior facets: split into outflow (interior value) and inflow (prescribed)
        # Use 0.5*(1 + sign(w_n)) as a smooth switch that doesn't involve the trial function
        if inflow_value is not None:
            # Outflow contribution (w_n > 0): use r_mid
            F += fd.max_value(w_n, 0) * r_mid * xi * ds
            # Inflow contribution (w_n < 0): use inflow_value (not trial-dependent)
            F += fd.min_value(w_n, 0) * inflow_value * xi * ds
        else:
            F += r_mid * w_n * xi * ds

        return F, dr2dt

    def age_form_dg(self, a, a_old, w, w_old, test, dt, inflow_value=None, source_mask=None):
        """DG weak form for age equation with upwind advection.

        Replaces Taylor-Galerkin stabilization with proper DG upwinding.
        At inflow boundaries (w*n < 0), uses inflow_value (default 0).
        """
        xi = test
        mesh = a.function_space().mesh()
        dx = fd.dx(domain=mesh)
        dS = fd.dS(domain=mesh)
        ds = fd.ds(domain=mesh)
        theta = fd.Constant(0.5)

        a_mid = theta * a + (1 - theta) * a_old

        n = fd.FacetNormal(mesh)
        w_n = w * n[0]
        a_up = fd.conditional(fd.gt(w_n('+'), 0), a_mid('+'), a_mid('-'))

        a_inflow = fd.Constant(0.0) if inflow_value is None else inflow_value

        # Temporal + source (age grows at rate 1; masked for free surface)
        _age_source = 1.0 if source_mask is None else source_mask
        F = (a - a_old) / dt * xi * dx
        F -= _age_source * xi * dx

        # DG advection: integration by parts + LF upwind flux
        # The IBP gives the conservative form ∂(wa)/∂z.  Subtract w_z·a
        # to recover the advective form w·∂a/∂z (the correct age equation).
        from ufl import algebra
        abs_w_n = fd.max_value(w_n, -w_n)
        F -= a_mid * w * xi.dx(0) * dx                                       # volume (IBP)
        F -= w.dx(0) * a_old * xi * dx                                       # conservative → advective (explicit)
        F += fd.avg(w * a_mid) * fd.jump(xi, n[0]) * dS                      # central flux
        F += 0.5 * abs_w_n('+') * fd.jump(a_mid, n[0]) * fd.jump(xi, n[0]) * dS  # upwind correction

        # Exterior facets: outflow uses interior, inflow uses prescribed value
        F += fd.max_value(w_n, 0) * a_mid * xi * ds
        F += fd.min_value(w_n, 0) * a_inflow * xi * ds

        return F

    def density_form_full_density(
        self,
        rho, rho_old,
        sigma, sigma_old,
        grain_radius2, grain_radius2_old,
        w, rhoCoef,
        *,
        H=None,
        T=None,
        test,
        dt,
    ):
        """Weak form for density using stress- and grain-size-dependent densification."""
        p = self.params
        phi = test
        T = self._temperature(H=H, T=T)
        mesh = rho.function_space().mesh()
        dx = fd.dx(domain=mesh)

        theta = fd.Constant(p.theta_liniger)
        rho_mid = theta * rho + (1.0 - theta) * rho_old
        sig_mid = theta * sigma + (1.0 - theta) * sigma_old
        r_mid = theta * grain_radius2 + (1.0 - theta) * grain_radius2_old

        r_eff = r_mid + p.eps_r2

        drhodt = rhoCoef * fd.exp(-p.Ec / (p.R * T)) * (p.rho_i - rho_mid) * sig_mid / r_eff

        F_rho = (rho - rho_old) / dt * phi * dx - drhodt * phi * dx + w * rho_mid.dx(0) * phi * dx

        # Soft barrier: penalize density above ρ_i to prevent unphysical
        # advection overshoots.  Uses rho_old (explicit) so the form
        # stays linear in the trial function.
        _rho_barrier_strength = fd.Constant(1.0 / float(dt))
        F_rho += _rho_barrier_strength * fd.max_value(rho_old - p.rho_i, 0) * phi * dx

        return F_rho, drhodt

    def density_form_full_density_dg(
        self,
        rho, rho_old,
        sigma, sigma_old,
        grain_radius2, grain_radius2_old,
        w, rhoCoef,
        *,
        H=None,
        T=None,
        test,
        dt,
        inflow_value=None,
    ):
        """DG weak form for density using stress- and grain-size-dependent densification."""
        p = self.params
        phi = test
        T = self._temperature(H=H, T=T)
        mesh = rho.function_space().mesh()
        dx = fd.dx(domain=mesh)
        dS = fd.dS(domain=mesh)
        ds = fd.ds(domain=mesh)

        theta = fd.Constant(p.theta_liniger)
        rho_mid = theta * rho + (1.0 - theta) * rho_old
        sig_mid = theta * sigma + (1.0 - theta) * sigma_old
        r_mid = theta * grain_radius2 + (1.0 - theta) * grain_radius2_old

        r_eff = r_mid + p.eps_r2

        drhodt = rhoCoef * fd.exp(-p.Ec / (p.R * T)) * (p.rho_i - rho_mid) * sig_mid / r_eff

        n = fd.FacetNormal(mesh)
        w_n = w * n[0]
        abs_w_n = fd.max_value(w_n, -w_n)

        # Temporal + source
        F = (rho - rho_old) / dt * phi * dx
        F -= drhodt * phi * dx

        # DG advection: IBP + Lax-Friedrichs upwind
        # IBP gives conservative ∂(wρ)/∂z; subtract w_z·ρ for advective w·∂ρ/∂z.
        F -= rho_mid * w * phi.dx(0) * dx
        F -= w.dx(0) * rho_old * phi * dx                                 # conservative → advective (explicit)
        F += fd.avg(w * rho_mid) * fd.jump(phi, n[0]) * dS
        F += 0.5 * abs_w_n('+') * fd.jump(rho_mid, n[0]) * fd.jump(phi, n[0]) * dS

        # Exterior facets: outflow uses interior, inflow uses prescribed value
        if inflow_value is not None:
            F += fd.max_value(w_n, 0) * rho_mid * phi * ds
            F += fd.min_value(w_n, 0) * inflow_value * phi * ds
        else:
            F += rho_mid * w_n * phi * ds

        # Soft barrier: penalize density above ρ_i (explicit in rho_old)
        _rho_barrier_strength = fd.Constant(1.0 / float(dt))
        F += _rho_barrier_strength * fd.max_value(rho_old - p.rho_i, 0) * phi * dx

        return F, drhodt

    def full_density_form(
        self,
        H,
        rho,
        sigma,
        grain_radius2,
        rho_old,
        sigma_old,
        grain_radius2_old,
        rhoCoef,
        w,
        accumulation,
        dt,
        phi,
        psi,
        xi,
    ):
        """Mixed residual for (rho, sigma, r^2) following a Cummings/Kingslake-style system.

        This is provided for convenience, but in the column solver we often
        solve (sigma) and (r^2) first, then (rho), since sigma and r^2 do not
        depend on rho in the simple burial + Arrhenius growth closure.
        """
        p = self.params
        mesh = rho.function_space().mesh()
        dx = fd.dx(domain=mesh)

        theta = fd.Constant(p.theta_liniger)
        rho_mid = theta * rho + (1.0 - theta) * rho_old
        sig_mid = theta * sigma + (1.0 - theta) * sigma_old
        r_mid = theta * grain_radius2 + (1.0 - theta) * grain_radius2_old

        # Temperature field from enthalpy.
        T = self.temperature_from_enthalpy(H)

        # Mass flux of accumulation in kg/m^2/s.
        bdot = p.rho_i * accumulation / p.spy

        # Densification rate
        r_eff = r_mid + p.eps_r2
        drhodt = rhoCoef * fd.exp(-p.Ec / (p.R * T)) * (p.rho_i - rho_mid) * sig_mid / r_eff

        # Stress evolution (simple burial loading)
        dsigdt = bdot * p.g

        # Grain growth (r^2)
        dr2dt = self.grain_growth_rate_r2(T, r2=r_mid)

        d_rho = (rho - rho_old) / dt * phi * dx - drhodt * phi * dx + w * rho_mid.dx(0) * phi * dx
        d_sigma = (sigma - sigma_old) / dt * psi * dx - dsigdt * psi * dx + w * sig_mid.dx(0) * psi * dx
        d_r = (grain_radius2 - grain_radius2_old) / dt * xi * dx - dr2dt * xi * dx + w * r_mid.dx(0) * xi * dx

        return d_rho + d_sigma + d_r


    # ------------------------------------------------------------------
    # 1) Enthalpy form (dry firn): advection + diffusion + optional basal flux
    # ------------------------------------------------------------------
    def enthalpy_form(
        self,
        H,
        H_old,
        rho,
        w,
        test,
        dt,
        *,
        basal_heat_flux_W_m2=None,
        base_boundary_id=None,
    ):
        """Weak form for dry enthalpy evolution.

        We evolve enthalpy ``H`` (J/kg) in a fixed 1D vertical coordinate ``x``
        consistent with the rest of this firn column model:

            (H^{n+1} - H^n)/dt + w * dH_mid/dx - d/dx( K * dH_mid/dx ) = 0

        where ``K`` is an effective thermal diffusivity (m^2/s) and ``w`` is the
        firn vertical velocity (m/s).

        **Basal heat flux**
        -------------------
        Optionally apply a total *upward* basal heat flux into the firn (W/m^2)
        at the base boundary. This can represent geothermal heat flux and/or
        basal frictional heating.

        In temperature form the boundary condition is:

            -k dT/dx = q_basal,   q_basal > 0 upward into the firn

        In enthalpy form (H = c_i (T - T_ref)), this is imposed as a Neumann BC:

            (K dH/dx) · n = q_basal / rho

        Parameters
        ----------
        basal_heat_flux_W_m2:
            Upward heat flux into the firn at the base in W/m^2. If ``None``,
            uses ``params.basal_heat_flux_W_m2``.
        base_boundary_id:
            Firedrake boundary id for the base. If ``None``, uses
            ``params.base_boundary_id`` (defaults to 1 for an IntervalMesh).
        """

        p = self.params
        psi = test
        mesh = H.function_space().mesh()
        dx = fd.dx(domain=mesh)
        ds = fd.ds(domain=mesh)

        # Semi-implicit theta scheme for the advective + diffusive terms.
        theta = fd.Constant(getattr(p, "theta_H", 1.0))
        H_mid = theta * H + (1.0 - theta) * H_old

        # Effective thermal diffusivity κ = k/(ρ c), with any temperature
        # dependence evaluated at the PREVIOUS step's enthalpy (lagged
        # linearization): K(H_mid) would put the trial function inside a
        # nonlinear expression (e.g. the Calonne-2019 exp() T-scaling) and
        # break the lhs/rhs split of this linear form. For the legacy
        # quadratic law (H-independent) this is symbolically identical.
        K = self.thermal_diffusivity(H_old, rho)

        F = (
            (H - H_old) / dt * psi * dx
            + w * H_mid.dx(0) * psi * dx
            + K * H_mid.dx(0) * psi.dx(0) * dx
        )

        # Optional basal heat flux term (Neumann BC).
        if basal_heat_flux_W_m2 is None:
            basal_heat_flux_W_m2 = getattr(p, "basal_heat_flux_W_m2", 0.0)

        if base_boundary_id is None:
            base_boundary_id = getattr(p, "base_boundary_id", 1)

        # Convert python numbers into UFL Constants (leave Functions/Constants alone).
        if isinstance(basal_heat_flux_W_m2, (int, float)):
            basal_heat_flux_W_m2 = fd.Constant(float(basal_heat_flux_W_m2))

        # Guard against division by (near) zero density.
        rho_eff = fd.max_value(rho, 1.0)
        F += -(basal_heat_flux_W_m2 / rho_eff) * psi * ds(int(base_boundary_id))

        return F

    # ------------------------------------------------------------------
    # 2) Density form: matches Evan's simple Density (not FullDensity)
    # ------------------------------------------------------------------
    def density_form_dg(self, rho, rho_old, T, w, bdot, *, test, dt,
                        inflow_value=None, source_mask=None, sigma=None):
        """DG weak form for density using the Arthern/Ligtenberg rate (Mode A).

        Mirrors `density_form_full_density_dg` but uses the basic Arthern rate
        instead of the Kingslake stress/grain-size form.

        ∂ρ/∂t + w ∂ρ/∂z = drhodt_arthern(ρ_old, T, bdot)
        """
        p = self.params
        phi = test
        mesh = rho.function_space().mesh()
        dx = fd.dx(domain=mesh)
        dS = fd.dS(domain=mesh)
        ds = fd.ds(domain=mesh)

        theta = fd.Constant(p.theta_rho)
        rho_mid = theta * rho + (1.0 - theta) * rho_old

        # Densification rate evaluated at previous density (explicit) to keep
        # the form linear in the trial function.  Use the pluggable rate_fn
        # if set on the model; otherwise fall back to the legacy Arthern form.
        if self.densification_rate_fn is not None:
            drhodt = self.densification_rate_fn(
                rho_old, T, params=p, bdot=bdot, sigma=sigma)
        else:
            drhodt = self.densification_rate_arthern(rho_old, T, bdot)

        # Optional source mask (free surface: zero out densification above h)
        if source_mask is not None:
            drhodt = drhodt * source_mask

        n = fd.FacetNormal(mesh)
        w_n = w * n[0]
        abs_w_n = fd.max_value(w_n, -w_n)

        # Temporal + source
        F = (rho - rho_old) / dt * phi * dx
        F -= drhodt * phi * dx

        # DG advection: IBP + Lax-Friedrichs upwind
        F -= rho_mid * w * phi.dx(0) * dx
        F -= w.dx(0) * rho_old * phi * dx   # conservative -> advective (explicit)
        F += fd.avg(w * rho_mid) * fd.jump(phi, n[0]) * dS
        F += 0.5 * abs_w_n('+') * fd.jump(rho_mid, n[0]) * fd.jump(phi, n[0]) * dS

        # Exterior facets: inflow = prescribed value, outflow = interior
        if inflow_value is not None:
            F += fd.max_value(w_n, 0) * rho_mid * phi * ds
            F += fd.min_value(w_n, 0) * inflow_value * phi * ds
        else:
            F += rho_mid * w_n * phi * ds

        # Soft barrier: penalize density above ρ_i (explicit in rho_old)
        _rho_barrier_strength = fd.Constant(1.0 / float(dt))
        F += _rho_barrier_strength * fd.max_value(rho_old - p.rho_i, 0) * phi * dx

        return F, drhodt

    def density_form(self, rho, rho_old, T, w, rhoCoef, bdot, test, dt,
                     sigma=None):
        """
        Weak form for density with densification evaluated explicitly in time
        (at rho_old):

            (rho^{n+1} - rho^n)/dt
          + w^n * (rho_mid,z)
          = drhodt(rho^n, T^n, bdot),

        where rho_mid is a theta-average used only in the advection term.

        Parameters
        ----------
        sigma : Function, optional
            Overburden stress [Pa]. Passed to densification functions that
            require it (e.g. stokes_compressible, kingslake).
        """
        phi = test
        mesh = rho.function_space().mesh()
        dx = fd.dx(domain=mesh)
        p = self.params

        theta = fd.Constant(p.theta_rho)
        rho_mid = theta * rho + (1 - theta) * rho_old

        # densification rate evaluated at previous time level (explicit)
        if self.densification_rate_fn is not None:
            drhodt_old = self.densification_rate_fn(
                rho_old, T, params=p, bdot=bdot, sigma=sigma)
        else:
            drhodt_old = self.densification_rate_arthern(rho_old, T, bdot)

        F_rho = (
            (rho - rho_old) / dt * phi * dx
            + w * rho_mid.dx(0) * phi * dx
            - drhodt_old * phi * dx
        )

        # Soft barrier: penalize density above ρ_i (explicit in rho_old)
        _rho_barrier_strength = fd.Constant(1.0 / float(dt))
        F_rho += _rho_barrier_strength * fd.max_value(rho_old - p.rho_i, 0) * phi * dx

        return F_rho, drhodt_old

    # ------------------------------------------------------------------
    # 3) Velocity form: matches Evan's Velocity class
    # ------------------------------------------------------------------
    # in firnpack/models/firn.py

        # ------------------------------------------------------------------
    # 3) Velocity form: match Evan's Velocity class as closely as possible
    # ------------------------------------------------------------------
    def velocity_delta(
        self,
        w_trial,
        w_old,
        rho,
        drhodt,
        test,
        regularization=0.0,
        horizontal_divergence=0.0,
    ):
        """
        Build the residual 'delta' for the velocity equation, matching
        Evan's Velocity class:

            rho * w_mid,z * eta + drhodt * eta + rho*div_h*eta = 0

        where ``div_h`` is the prescribed horizontal divergence (∂u/∂x + ∂v/∂y).
        In 1D column mode with laterally-uniform density this comes from the
        3D mass conservation law:

            dρ/dt + ρ (∂w/∂z + div_h) = 0

        and therefore modifies the vertical-velocity gradient as

            ∂w/∂z = -(1/ρ) dρ/dt - div_h.

        with w_mid = theta_w * w + (1 - theta_w) * w_old.

        Optionally add a tiny "mass" regularization term:
            + epsilon * w_trial * eta
        to stabilize the linear solve.
        """
        dx = fd.dx
        p = self.params

        theta = fd.Constant(p.theta_w)  # 0.878
        eta   = test

        # semi-implicit mid-point
        w_mid = theta * w_trial + (1.0 - theta) * w_old

        delta = rho * w_mid.dx(0) * eta * dx + drhodt * eta * dx

        # Optional horizontal-divergence (basal strain) contribution.
        # Keep this as a simple additive term so the default behaviour
        # (horizontal_divergence=0) is unchanged.
        if horizontal_divergence not in (0, 0.0, None):
            # Pass through Firedrake Functions / Constants / UFL expressions
            # directly (preserving the pyadjoint tape).  Only wrap plain
            # Python numbers in fd.Constant.
            if isinstance(horizontal_divergence, (int, float)):
                div_h = fd.Constant(horizontal_divergence)
            else:
                div_h = horizontal_divergence
            delta += rho * div_h * eta * dx

        if regularization != 0.0:
            eps = fd.Constant(regularization)
            delta += eps * w_trial * eta * dx

        return delta

    def mass_flux_delta(
        self,
        q_trial,
        rho,
        rho_old,
        dt,
        test,
        horizontal_divergence=0.0,
        regularization=0.0,
    ):
        """
        Residual for mass flux q = ρw from Eulerian continuity:

            ∂q/∂z + (ρ − ρ_old)/Δt + ρ · div_h = 0

        Apply a Dirichlet BC for q at either the surface or base, then
        recover the velocity field via  w = q / ρ.

        Compared with ``velocity_delta``, which solves for w directly:

        * The unknown is q = ρw (mass flux, kg m⁻² s⁻¹), not w.
        * The source is the *total* density tendency (ρ − ρ_old)/Δt from
          the just-solved density equation, not the densification rate
          dρ/dt alone.
        * The surface boundary value  q_s = −ḃ ρ_i / spy  is independent
          of the surface density ρ_s — the key advantage for inversions
          with ρ_s as a control.
        """
        dx = fd.dx
        eta = test

        delta = q_trial.dx(0) * eta * dx + (rho - rho_old) / dt * eta * dx

        if horizontal_divergence not in (0, 0.0, None):
            try:
                div_h = fd.Constant(float(horizontal_divergence))
            except Exception:
                div_h = horizontal_divergence
            delta += rho * div_h * eta * dx

        if regularization != 0.0:
            eps = fd.Constant(regularization)
            delta += eps * q_trial * eta * dx

        return delta

    def age_form(self, a, a_old, w, w_old, test, dt):
        """
        Evan-style Taylor–Galerkin age equation in 1D:
            a_t = 1 - w * a_z
        (see Cummings paper Age section; matches CSLVR form)
        """
        xi = test
        mesh = a.function_space().mesh()
        dx = fd.dx(domain=mesh)
        theta = fd.Constant(0.5)  # Evan uses 0.5 for age scheme

        a_mid = theta * a + (1 - theta) * a_old

        F = (a - a_old) / dt * xi * dx
        F += -1.0 * xi * dx
        F += w * a_mid.dx(0) * xi * dx
        F += -0.5 * (w - w_old) * a_mid.dx(0) * xi * dx
        F += (w**2) * dt / 2.0 * a_mid.dx(0) * xi.dx(0) * dx
        F += -(w * w.dx(0)) * dt / 2.0 * a_mid.dx(0) * xi * dx

        return F


    def thickness_tendency(self, rho, w, accumulation):
        """
        Column thickness tendency dh/dt from a 1D firn column.

        Parameters
        ----------
        rho : Function (kg m^-3)
            Density profile on the 1D mesh.
        w   : Function (m s^-1)
            Vertical firn velocity (same sign convention as in the model).
        accumulation : Constant or Function (m ice eq / yr)
            Surface accumulation rate (positive for gain).

        Returns
        -------
        float
            dh/dt in m s^-1.
        """
        p = self.params
        mesh = rho.function_space().mesh()
        dx = fd.dx(domain=mesh)

        # Geometric mesh thickness (in meters)
        coords = mesh.coordinates.dat.data_ro
        z_min = coords.min()
        z_max = coords.max()
        H_coord = z_max - z_min

        # Depth-mean density ρ̄ = (1/H) ∫ ρ dz
        column_mass = fd.assemble(rho * dx)   # kg m^-2
        rho_bar = column_mass / H_coord       # kg m^-3

        # Base density and velocity (first DOF is at the base for CG1 on IntervalMesh)
        rho_arr = rho.dat.data_ro
        w_arr   = w.dat.data_ro
        rho_b = float(rho_arr[0])
        w_b   = float(w_arr[0])

        # Surface mass flux from accumulation (kg m^-2 s^-1)
        bdot = float(accumulation)

        # Thickness tendency (m s^-1), following Evan's dh/dt ≈ (ρ_s a - ρ_b w_b)/ρ̄
        # Here we use ρ_i in bdot (ice-density equivalent); if you want to
        # use the actual surface density instead, replace p.rho_i with ρ_s.
        dhdt = -(bdot * ice_density/year + rho_b * w_b) / rho_bar

        return dhdt

