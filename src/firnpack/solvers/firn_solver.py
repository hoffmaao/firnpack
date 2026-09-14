import firedrake as fd
import numpy as np


def _scalar_value(x):
    """Best-effort extraction of a Python float from Firedrake scalars."""
    try:
        return float(x)
    except Exception:
        if hasattr(x, "dat"):
            return float(x.dat.data_ro[0])
        raise


class FirnColumnSolver:
    """
    1D firn-column time stepper.

    Supports two "modes":
      (A) Basic (H, rho, w) Arthern/Ligtenberg densification (legacy)
      (B) Full-density (H, rho, sigma, r^2, w) with stress + grain growth

    Option B (temperature-from-enthalpy) is respected by passing enthalpy H into
    the full-density closures (grain growth + full-density densification).

    Notes
    -----
    - Boundary conditions are expected to be a list/tuple with at least:
        bcs[0] : BC for enthalpy H at the surface
        bcs[1] : BC for density rho at the surface
        bcs[2] : BC for velocity w at the surface
      Additional BCs for sigma and r^2 are passed separately.
    """

    def __init__(
        self,
        model,
        solver_parameters=None,
        *,
        horizontal_divergence: float = 0.0,
        surface_id: int = 2,
    ):
        self.model = model
        self.solver_parameters = solver_parameters or {
            "snes_type": "newtonls",
            "snes_rtol": 1e-5,
            "snes_atol": 1e-7,
            "snes_max_it": 30,
            "ksp_type": "gmres",
            "pc_type": "lu",
        }
        # Horizontal divergence (1/s). This provides a simple way to include a
        # basal-strain/ice-sheet kinematics constraint in the 1-D continuity
        # equation:  rho*w_z + drhodt + rho*div_h = 0.
        if isinstance(horizontal_divergence, (int, float, np.floating, np.integer)):
            self.horizontal_divergence = fd.Constant(float(horizontal_divergence))
        else:
            # Allow users to pass an existing Firedrake Constant/Expression.
            self.horizontal_divergence = horizontal_divergence
        self._surface_id = surface_id
        self._fields = {}

    @property
    def fields(self):
        return self._fields

    def _setup(
        self,
        *,
        enthalpy,
        density,
        firn_velocity,
        dt,
        accumulation,
        surface_density,
        boundary_conditions,
    ):
        V = enthalpy.function_space()
        self._fields["V"] = V
        self._fields["test"] = fd.TestFunction(V)

        # Current fields (references)
        self._fields["H"] = enthalpy
        self._fields["rho"] = density
        self._fields["w"] = firn_velocity

        # Old copies - use Function() + assign() so the assignment is on
        # the pyadjoint tape (copy(deepcopy=True) may not be taped).
        self._fields["H_old"] = fd.Function(V, name="H_old")
        self._fields["H_old"].assign(enthalpy)
        self._fields["rho_old"] = fd.Function(V, name="rho_old")
        self._fields["rho_old"].assign(density)
        self._fields["w_old"] = fd.Function(V, name="w_old")
        self._fields["w_old"].assign(firn_velocity)

        # Optional old copies (created lazily)
        self._fields["age_old"] = None
        self._fields["sigma_old"] = None
        self._fields["r2_old"] = None

        # Scalars / forcing (usually Real-space Functions)
        self._fields["dt"] = dt
        self._fields["accumulation"] = accumulation
        self._fields["surface_density"] = surface_density
        self._fields["bcs"] = boundary_conditions

    @staticmethod
    def _add_refreezing_source(F_rho, refreezing, rho_test):
        """Refrozen meltwater is matrix ice: D(rho)/Dt = compaction + m.

        Meyer & Hewitt (2017) ice equation. Refreezing fills pore space in
        place, so it belongs in the density equation and nowhere else; the
        velocity integration then sees it only through the total tendency.
        """
        if refreezing is None:
            return F_rho
        if isinstance(refreezing, (int, float)):
            refreezing = fd.Constant(float(refreezing))
        mesh = rho_test.function_space().mesh()
        return F_rho - refreezing * rho_test * fd.dx(domain=mesh)

    def prognostic_solve(
        self,
        *,
        enthalpy,
        density,
        firn_velocity,
        dt,
        accumulation,
        surface_density,
        boundary_conditions=None,
        # Optional convenience forcing for setting enthalpy BC
        surface_temperature=None,
        enthalpy_bc_constant=None,
        # Optional age
        age=None,
        age_boundary_condition=None,
        # Optional full-density unknowns
        stress=None,
        grain_radius2=None,
        stress_boundary_condition=None,
        grain_radius2_boundary_condition=None,
        rhoCoef=None,
        # Refreezing mass source [kg m^-3 s^-1] from the hydrology solver.
        # Enters the continuity integration as an ice-mass source; None (the
        # default) leaves the dry residual untouched.
        refreezing=None,
        # Penalty BCs: adjoint-safe alternative to Dirichlet for control-
        # dependent surface values.  Dict mapping field name to
        # (target_expr, penalty_strength) where target_expr is a UFL expression
        # for the desired surface value and penalty_strength is a large constant
        # (e.g. 1e4).  Supported keys: "rho", "r2".
        # When a field has a penalty BC, the corresponding Dirichlet BC in
        # boundary_conditions / grain_radius2_boundary_condition is ignored
        # (pass None or a dummy).
        penalty_bcs=None,
        # Inflow values for DG fields at inflow boundaries (surface).
        # Dict mapping field name to a UFL expression for the inflow value.
        # Supported keys: "rho", "r2".  For DG fields, this sets the upwind
        # flux at the surface - the physically correct way to impose a
        # boundary condition without artificial penalty terms.
        inflow_values=None,
        # Basal velocity BC: pass a DirichletBC for w at the base instead
        # of the surface velocity BC in boundary_conditions[2].
        # The natural value is w_base = -bdot/spy (independent of ρ_surf).
        # The surface velocity emerges from integrating the continuity
        # equation through the column.  boundary_conditions[2] is ignored.
        mass_flux_bc=None,
        # Diagnostics
        verbose=False,
    ):
        if boundary_conditions is None:
            boundary_conditions = []

        # One-time setup
        if "H_old" not in self._fields:
            self._setup(
                enthalpy=enthalpy,
                density=density,
                firn_velocity=firn_velocity,
                dt=dt,
                accumulation=accumulation,
                surface_density=surface_density,
                boundary_conditions=boundary_conditions,
            )
        else:
            # Update scalar inputs
            self._fields["dt"].assign(dt)
            self._fields["accumulation"].assign(accumulation)
            self._fields["surface_density"].assign(surface_density)

            # Update "old" copies from provided current fields
            self._fields["H_old"].assign(enthalpy)
            self._fields["rho_old"].assign(density)
            self._fields["w_old"].assign(firn_velocity)

        if len(boundary_conditions) < 2:
            raise ValueError("boundary_conditions must contain at least [bc_H, bc_rho]. "
                             "Use None for rho if using penalty_bcs={'rho': ...}.")
        _has_w_bc = len(boundary_conditions) >= 3 and boundary_conditions[2] is not None
        if mass_flux_bc is None and not _has_w_bc:
            _w_penalty = penalty_bcs.get("w") if penalty_bcs else None
            if _w_penalty is None:
                raise ValueError(
                    "Either boundary_conditions[2] (velocity BC) or "
                    "mass_flux_bc (mass-flux BC) must be provided."
                )

        V = self._fields["V"]
        psi = self._fields["test"]

        H = enthalpy
        rho = density
        w = firn_velocity

        H_old = self._fields["H_old"]
        rho_old = self._fields["rho_old"]
        w_old = self._fields["w_old"]

        dt_val = self._fields["dt"]

        model = self.model
        p = model.params

        # --- update enthalpy BC from surface_temperature if provided ---
        if (surface_temperature is not None) and (enthalpy_bc_constant is not None):
            Ts = _scalar_value(surface_temperature)
            T_ref = getattr(p, "T_ref", 273.15)
            Hs = p.c_i * (Ts - T_ref)
            enthalpy_bc_constant.assign(Hs)

        # ------------------------------------------------------------------
        # 1) ENTHALPY SOLVE (nonlinear)
        # ------------------------------------------------------------------
        if verbose:
            print(f"  [pre-enthalpy] rho: min={rho.dat.data_ro.min():.3e}, max={rho.dat.data_ro.max():.3e}")
            print(f"  [pre-enthalpy] w:   min={w.dat.data_ro.min():.3e}, max={w.dat.data_ro.max():.3e}")
        F_H = model.enthalpy_form(H, H_old, rho, w, psi, dt_val)
        dH = fd.TrialFunction(V)
        J_H = fd.derivative(F_H, H, dH)

        problem_H = fd.NonlinearVariationalProblem(F_H, H, bcs=[boundary_conditions[0]], J=J_H)
        solver_H = fd.NonlinearVariationalSolver(problem_H, solver_parameters=self.solver_parameters)
        try:
            solver_H.solve()
        except Exception:
            _diag = {
                "H":     (H.dat.data_ro.min(),     H.dat.data_ro.max()),
                "H_old": (H_old.dat.data_ro.min(), H_old.dat.data_ro.max()),
                "rho":   (rho.dat.data_ro.min(),   rho.dat.data_ro.max()),
                "w":     (w.dat.data_ro.min(),      w.dat.data_ro.max()),
            }
            _nan_fields = [k for k, (mn, mx) in _diag.items()
                           if np.isnan(mn) or np.isnan(mx) or np.isinf(mn) or np.isinf(mx)]
            print("\n[firn_solver] ENTHALPY SOLVE FAILED:")
            for k, (mn, mx) in _diag.items():
                flag = " ← NaN/Inf" if k in _nan_fields else ""
                print(f"  {k:6s}: min={mn:.4e}, max={mx:.4e}{flag}")
            raise

        # ------------------------------------------------------------------
        # 2) DENSITY (and optional stress + grain-size) SOLVES
        # ------------------------------------------------------------------
        full_density = (stress is not None) and (grain_radius2 is not None)

        if full_density:
            _r2_has_penalty = penalty_bcs is not None and "r2" in penalty_bcs
            _r2_has_inflow = inflow_values is not None and "r2" in inflow_values
            if stress_boundary_condition is None:
                raise ValueError(
                    "When passing stress you must also pass stress_boundary_condition."
                )
            if grain_radius2_boundary_condition is None and not _r2_has_penalty and not _r2_has_inflow:
                raise ValueError(
                    "When passing grain_radius2 you must also pass "
                    "grain_radius2_boundary_condition, penalty_bcs={'r2': ...}, "
                    "or inflow_values={'r2': ...}."
                )

            sigma = stress
            r2 = grain_radius2

            # Lazily create old copies - use Function + assign so it's on tape
            if self._fields["sigma_old"] is None:
                self._fields["sigma_old"] = fd.Function(sigma.function_space(), name="sigma_old")
            self._fields["sigma_old"].assign(sigma)

            if self._fields["r2_old"] is None:
                self._fields["r2_old"] = fd.Function(r2.function_space(), name="r2_old")
            self._fields["r2_old"].assign(r2)

            sigma_old = self._fields["sigma_old"]
            r2_old = self._fields["r2_old"]

            # Two-stage kc (explicit in rho_old) unless a rhoCoef field is passed in
            if rhoCoef is None:
                rhoCoef_eff = fd.conditional(fd.le(rho_old, p.rho_m), p.kc0, p.kc1)
            else:
                rhoCoef_eff = rhoCoef

            # (a) Stress update
            sigma_trial = fd.TrialFunction(V)
            F_sigma = model.stress_form(
                sigma=sigma_trial,
                sigma_old=sigma_old,
                w=w,
                accumulation=accumulation,
                test=psi,
                dt=dt_val,
            )
            sig_problem = fd.LinearVariationalProblem(
                fd.lhs(F_sigma), fd.rhs(F_sigma), sigma, bcs=[stress_boundary_condition]
            )
            fd.LinearVariationalSolver(
                sig_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"}
            ).solve()

            # (b) Grain-size-squared update (Option B: temperature from enthalpy H)
            V_r2 = r2.function_space()
            _r2_is_dg = V_r2.ufl_element().family() == "Discontinuous Lagrange"
            r2_trial = fd.TrialFunction(V_r2)
            r2_test = fd.TestFunction(V_r2) if _r2_is_dg else psi

            if _r2_is_dg:
                # Determine inflow value: prefer inflow_values, fall back to penalty target
                _r2_inflow = None
                if inflow_values and "r2" in inflow_values:
                    _r2_inflow = inflow_values["r2"]
                elif penalty_bcs and "r2" in penalty_bcs:
                    _r2_inflow = penalty_bcs["r2"][0]
                F_r2, _ = model.grain_radius2_form_dg(
                    grain_radius2=r2_trial,
                    grain_radius2_old=r2_old,
                    w=w,
                    H=H,
                    test=r2_test,
                    dt=dt_val,
                    inflow_value=_r2_inflow,
                )
            else:
                F_r2, _ = model.grain_radius2_form(
                    grain_radius2=r2_trial,
                    grain_radius2_old=r2_old,
                    w=w,
                    H=H,
                    test=r2_test,
                    dt=dt_val,
                )
            # Penalty BC for r2: add penalty term to weak form instead of Dirichlet
            _r2_penalty = penalty_bcs.get("r2") if penalty_bcs else None
            if _r2_penalty is not None:
                _r2_target, _r2_pen = _r2_penalty
                _ds_surf = fd.ds(self._surface_id)
                F_r2 += fd.Constant(_r2_pen) * (r2_trial - _r2_target) * r2_test * _ds_surf
                r2_problem = fd.LinearVariationalProblem(
                    fd.lhs(F_r2), fd.rhs(F_r2), r2)
            else:
                r2_bcs = [] if _r2_is_dg else [grain_radius2_boundary_condition]
                r2_problem = fd.LinearVariationalProblem(
                    fd.lhs(F_r2), fd.rhs(F_r2), r2, bcs=r2_bcs)
            fd.LinearVariationalSolver(
                r2_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"}
            ).solve()

            # (c) Density update (Option B: temperature from enthalpy H)
            V_rho = rho.function_space()
            _rho_is_dg = V_rho.ufl_element().family() == "Discontinuous Lagrange"
            rho_trial = fd.TrialFunction(V_rho)
            rho_test = fd.TestFunction(V_rho) if _rho_is_dg else psi

            if _rho_is_dg:
                # Determine inflow value: prefer inflow_values, fall back to penalty target
                _rho_inflow = None
                if inflow_values and "rho" in inflow_values:
                    _rho_inflow = inflow_values["rho"]
                elif penalty_bcs and "rho" in penalty_bcs:
                    _rho_inflow = penalty_bcs["rho"][0]
                F_rho, _ = model.density_form_full_density_dg(
                    rho=rho_trial,
                    rho_old=rho_old,
                    sigma=sigma,
                    sigma_old=sigma_old,
                    grain_radius2=r2,
                    grain_radius2_old=r2_old,
                    w=w,
                    rhoCoef=rhoCoef_eff,
                    H=H,
                    test=rho_test,
                    dt=dt_val,
                    inflow_value=_rho_inflow,
                )
            else:
                F_rho, _ = model.density_form_full_density(
                    rho=rho_trial,
                    rho_old=rho_old,
                    sigma=sigma,
                    sigma_old=sigma_old,
                    grain_radius2=r2,
                    grain_radius2_old=r2_old,
                    w=w,
                    rhoCoef=rhoCoef_eff,
                    H=H,
                    test=rho_test,
                    dt=dt_val,
                )
            F_rho = self._add_refreezing_source(F_rho, refreezing, rho_test)
            # Penalty BC for rho: add penalty term to weak form instead of Dirichlet
            _rho_penalty = penalty_bcs.get("rho") if penalty_bcs else None
            if _rho_penalty is not None:
                _rho_target, _rho_pen = _rho_penalty
                _ds_surf = fd.ds(self._surface_id)
                F_rho += fd.Constant(_rho_pen) * (rho_trial - _rho_target) * rho_test * _ds_surf
                rho_problem = fd.LinearVariationalProblem(
                    fd.lhs(F_rho), fd.rhs(F_rho), rho)
            else:
                rho_bcs = [] if _rho_is_dg else [boundary_conditions[1]]
                rho_problem = fd.LinearVariationalProblem(
                    fd.lhs(F_rho), fd.rhs(F_rho), rho, bcs=rho_bcs)
            fd.LinearVariationalSolver(
                rho_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"}
            ).solve()

            # Densification rate for the velocity solve (use updated state)
            drhodt_expr = model.densification_rate_full_density(
                rho=rho,
                sigma=sigma,
                grain_radius2=r2,
                rhoCoef=rhoCoef_eff,
                H=H,
            )

        else:
            # Basic Arthern/Ligtenberg density update
            T = model.temperature_from_enthalpy(H)
            V_rho = rho.function_space()
            _rho_is_dg = V_rho.ufl_element().family() == "Discontinuous Lagrange"
            rho_trial = fd.TrialFunction(V_rho)
            rho_test = fd.TestFunction(V_rho) if _rho_is_dg else psi

            # Compute overburden stress from rho_old (needed by Stokes-type laws)
            _mesh = rho.function_space().mesh()
            _sigma = model.overburden_stress(
                rho_old, _mesh, surface_id=self._surface_id
            )

            if _rho_is_dg:
                # Determine inflow value: prefer inflow_values, fall back to penalty target
                _rho_inflow = None
                if inflow_values and "rho" in inflow_values:
                    _rho_inflow = inflow_values["rho"]
                elif penalty_bcs and "rho" in penalty_bcs:
                    _rho_inflow = penalty_bcs["rho"][0]
                F_rho, _ = model.density_form_dg(
                    rho=rho_trial,
                    rho_old=rho_old,
                    T=T,
                    w=w,
                    bdot=accumulation * p.rho_i / p.spy,
                    test=rho_test,
                    dt=dt_val,
                    inflow_value=_rho_inflow,
                    sigma=_sigma,
                )
            else:
                F_rho, _ = model.density_form(
                    rho=rho_trial,
                    rho_old=rho_old,
                    T=T,
                    w=w,
                    rhoCoef=None,
                    bdot=accumulation * p.rho_i / p.spy,
                    test=rho_test,
                    dt=dt_val,
                    sigma=_sigma,
                )
            F_rho = self._add_refreezing_source(F_rho, refreezing, rho_test)

            # Penalty BC for rho: add penalty term to weak form instead of Dirichlet
            _rho_penalty = penalty_bcs.get("rho") if penalty_bcs else None
            if _rho_penalty is not None:
                _rho_target, _rho_pen = _rho_penalty
                _ds_surf = fd.ds(self._surface_id)
                F_rho += fd.Constant(_rho_pen) * (rho_trial - _rho_target) * rho_test * _ds_surf
                rho_problem = fd.LinearVariationalProblem(
                    fd.lhs(F_rho), fd.rhs(F_rho), rho)
            else:
                rho_bcs = [] if _rho_is_dg else [boundary_conditions[1]]
                rho_problem = fd.LinearVariationalProblem(
                    fd.lhs(F_rho), fd.rhs(F_rho), rho, bcs=rho_bcs)
            fd.LinearVariationalSolver(
                rho_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"}
            ).solve()

            if model.densification_rate_fn is not None:
                drhodt_expr = model.densification_rate_fn(
                    rho, T, params=p, bdot=accumulation * p.rho_i / p.spy,
                    sigma=_sigma,
                )
            else:
                drhodt_expr = model.densification_rate_arthern(
                    rho, T, accumulation * p.rho_i / p.spy,
                )

        # ------------------------------------------------------------------
        # Continuity-derived diagnostics (more robust than w.dx for DG spaces)
        # ------------------------------------------------------------------
        # Evan's original implementation uses the UFL expression `drhodt_expr`
        # directly in the velocity weak form. For convenience/diagnostics (and
        # to enable stable strain-rate observation operators), we also
        # *interpolate* drhodt and eps = -drhodt/rho into V.

        if self._fields.get("drhodt") is None or self._fields["drhodt"].function_space() != V:
            self._fields["drhodt"] = fd.Function(V, name="drhodt")
        drhodt = self._fields["drhodt"]
        drhodt.interpolate(drhodt_expr)
        # Continuity needs the total density tendency. `drhodt` is the
        # compaction rate alone (kept that way: the strain-rate diagnostic
        # below is dw/dz, which refreezing does not change - it fills pores,
        # it does not thicken the column). With the refreezing source m in the
        # density equation, D(rho)/Dt = compaction + m, and velocity_delta's
        # rho dw/dz = m - D(rho)/Dt then reduces to -compaction, as it must.
        # Before this the compaction rate was passed as the total and the
        # column was stretched by every kilogram it refroze.
        #
        # So the two refreezing terms this solver hands velocity_delta -
        # `drhodt=drhodt_total`, which contains m, and `refreezing=m` itself,
        # which velocity_delta subtracts - cancel exactly, and that is the
        # point. velocity_delta states Meyer & Hewitt's ice equation
        # rho dw/dz = m - D(rho)/Dt in full, so it stays correct for a caller
        # that passes a total tendency without refrozen mass in it; here the
        # cancellation *is* the physics, not a redundant argument to drop.
        drhodt_total = drhodt if refreezing is None else drhodt + refreezing

        if self._fields.get("eps") is None or self._fields["eps"].function_space() != V:
            self._fields["eps"] = fd.Function(V, name="eps")
        eps = self._fields["eps"]
        eps.interpolate(model.vertical_strain_rate_from_continuity(rho, drhodt))

        if verbose:
            rho_arr = rho.dat.data_ro
            drhodt_arr = drhodt.dat.data_ro
            print("  [velocity diagnostics]")
            print(f"    rho:    min={rho_arr.min():.3e}, max={rho_arr.max():.3e}")
            print(f"    drhodt: min={drhodt_arr.min():.3e}, max={drhodt_arr.max():.3e}")
            print(f"    any NaNs? rho={np.isnan(rho_arr).any()}, drhodt={np.isnan(drhodt_arr).any()}")
            if full_density:
                sig_arr = stress.dat.data_ro
                r2_arr = grain_radius2.dat.data_ro
                print(f"    sigma:  min={sig_arr.min():.3e}, max={sig_arr.max():.3e}")
                print(f"    r2:     min={r2_arr.min():.3e}, max={r2_arr.max():.3e}")

        # ------------------------------------------------------------------
        # 3) VELOCITY SOLVE (linear)
        # ------------------------------------------------------------------
        if mass_flux_bc is not None:
            # --- Velocity solve with basal Dirichlet BC ---
            # Same velocity_delta (dρ/dt densification), but with the BC
            # at the base instead of the surface.  mass_flux_bc is a
            # DirichletBC for w at the base:  w_base = -bdot/spy.
            # This is independent of ρ_surf (the key advantage).
            # The surface velocity emerges from integrating the continuity
            # equation through the column.
            w_trial = fd.TrialFunction(V)

            delta_w = model.velocity_delta(
                w_trial=w_trial,
                w_old=w_old,
                rho=rho,
                drhodt=drhodt_total,
                test=psi,
                regularization=1.0e-3,
                horizontal_divergence=self.horizontal_divergence,
                refreezing=refreezing,
            )

            w_problem = fd.LinearVariationalProblem(
                fd.lhs(delta_w), fd.rhs(delta_w), w, bcs=[mass_flux_bc])
            fd.LinearVariationalSolver(
                w_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"}
            ).solve()
        else:
            # --- Original velocity formulation ---
            w_trial = fd.TrialFunction(V)

            delta_w = model.velocity_delta(
                w_trial=w_trial,
                w_old=w_old,
                rho=rho,
                drhodt=drhodt_total,
                test=psi,
                regularization=1.0e-3,
                horizontal_divergence=self.horizontal_divergence,
                refreezing=refreezing,
            )

            # Penalty BC for velocity (adjoint-safe alternative to Dirichlet)
            _w_penalty = penalty_bcs.get("w") if penalty_bcs else None
            if _w_penalty is not None:
                _w_target = _w_penalty[0]
                _w_pen = _w_penalty[1]
                _w_pen_id = _w_penalty[2] if len(_w_penalty) > 2 else self._surface_id
                _ds_w = fd.ds(_w_pen_id)
                delta_w += fd.Constant(_w_pen) * (w_trial - _w_target) * psi * _ds_w
                w_problem = fd.LinearVariationalProblem(
                    fd.lhs(delta_w), fd.rhs(delta_w), w)
            else:
                w_problem = fd.LinearVariationalProblem(
                    fd.lhs(delta_w), fd.rhs(delta_w), w, bcs=[boundary_conditions[2]])
            fd.LinearVariationalSolver(
                w_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"}
            ).solve()

        # ------------------------------------------------------------------
        # 4) AGE SOLVE (optional, after velocity is updated)
        # ------------------------------------------------------------------
        if age is not None:
            if self._fields["age_old"] is None:
                self._fields["age_old"] = fd.Function(age.function_space(), name="age_old")
            if self._fields["age_old"] is not None:
                self._fields["age_old"].assign(age)
            age_old = self._fields["age_old"]

            V_age = age.function_space()
            _age_is_dg = V_age.ufl_element().family() == "Discontinuous Lagrange"
            a_trial = fd.TrialFunction(V_age)
            a_test = fd.TestFunction(V_age) if _age_is_dg else psi

            # CG requires a Dirichlet BC; DG uses inflow value (default 0)
            if (not _age_is_dg) and age_boundary_condition is None:
                raise ValueError("age_boundary_condition must be provided for CG age")

            if _age_is_dg:
                F_age = model.age_form_dg(
                    a=a_trial,
                    a_old=age_old,
                    w=w,
                    w_old=w_old,
                    test=a_test,
                    dt=dt_val,
                )
            else:
                F_age = model.age_form(
                    a=a_trial,
                    a_old=age_old,
                    w=w,
                    w_old=w_old,
                    test=a_test,
                    dt=dt_val,
                )

            age_bcs = [] if _age_is_dg else [age_boundary_condition]
            age_problem = fd.LinearVariationalProblem(
                fd.lhs(F_age), fd.rhs(F_age), age, bcs=age_bcs
            )
            fd.LinearVariationalSolver(
                age_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"}
            ).solve()

            age_old.assign(age)

        # ------------------------------------------------------------------
        # 5) Update old fields for next step
        # ------------------------------------------------------------------
        H_old.assign(H)
        rho_old.assign(rho)
        w_old.assign(w)
        if full_density:
            self._fields["sigma_old"].assign(stress)
            self._fields["r2_old"].assign(grain_radius2)

        if age is None:
            return H, rho, w
        return H, rho, w, age
