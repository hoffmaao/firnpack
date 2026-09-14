"""Time stepping for the coupled water + enthalpy system.

One step is operator-split into transport followed by local phase change, the
order Meyer & Hewitt's local-equilibrium closure implies:

1. Advance ``W`` with percolation, the surface melt influx, basal outflow and
   lateral drainage.
2. Advance ``H`` with advection and conduction.
3. Exchange mass and latent heat between the two **pointwise**.

Step 3 is deliberately not folded into the weak forms of steps 1-2. The
freezing increment is capped by the water present *at a node*, but a finite
element sink spreads that cap over neighbouring nodes through the mass matrix,
so a node can still be driven negative; clipping the result then creates water.
Applied pointwise, the same increment conserves mass and energy to round-off
and cannot drive ``W < 0`` or push ``H`` past zero from either side. See
:meth:`firnpack.models.hydrology.HydrologyModel.phase_change_increment`.

The freezing rate is returned as a field so the caller can pass it straight to
``FirnModel.velocity_delta`` as the ice-mass source, keeping the firn and
hydrology halves consistent.
"""

import firedrake as fd
from firedrake.exceptions import ConvergenceError


class HydrologySolver:
    def __init__(self, hydrology_model, firn_model, solver_parameters=None):
        self.hydrology_model = hydrology_model
        self.firn_model = firn_model
        self.solver_parameters = solver_parameters or {
            "snes_type": "newtonls",
            # Backtracking: k_rel ~ S^m and the clamps on S make the residual
            # strongly nonlinear, and an undamped Newton step overshoots into
            # regions where the wave speed is meaningless.
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 50,
            "ksp_type": "gmres",
            "pc_type": "lu",
        }
        self._fields = {}

    @property
    def fields(self):
        return self._fields

    # ------------------------------------------------------------------
    def _setup(self, *, enthalpy, water, dt):
        V = enthalpy.function_space()
        self._fields["V"] = V
        self._fields["H_old"] = enthalpy.copy(deepcopy=True)
        self._fields["W_old"] = water.copy(deepcopy=True)
        self._fields["m_phase"] = fd.Function(V, name="refreezing_rate")
        self._fields["W_sub_old"] = water.copy(deepcopy=True)
        self._fields["W_entry"] = water.copy(deepcopy=True)
        self._fields["dt"] = dt if isinstance(dt, fd.Constant) else fd.Constant(dt)
        self._fields["test"] = fd.TestFunction(V)

    # ------------------------------------------------------------------
    def _solve_water(self, W, W_old, dt_c, **kw):
        """One water transport solve at the step currently held by ``dt_c``."""
        V = self._fields["V"]
        F = self.hydrology_model.water_form(
            W, W_old, kw["rho"], kw["w"], self._fields["test"], dt_c,
            grain_radius2=kw["grain_radius2"],
            surface_flux=kw["surface_flux"], surface_id=kw["surface_id"],
            basal_flux=kw["basal_flux"], base_id=kw["base_id"],
            include_drainage=kw["include_drainage"],
        )
        problem = fd.NonlinearVariationalProblem(
            F, W, bcs=kw["bcs"], J=fd.derivative(F, W, fd.TrialFunction(V))
        )
        fd.NonlinearVariationalSolver(
            problem, solver_parameters=self.solver_parameters
        ).solve()

    def _water_transport(self, W, W_old, dt_total, max_halvings=8, **kw):
        """Transport water over ``dt_total``, sub-stepping if Newton fails.

        Percolation is far faster than the firn it moves through - a wetting
        front crosses a cell in minutes while the column evolves over years -
        so a step sized for the firn can be far beyond what the nonlinear water
        problem will take. Rather than force the caller to pick a step that
        works for the worst moment of the melt season, halve on failure and
        sub-step. Retrying costs nothing when the step was fine, which is most
        of the year.
        """
        dt_c = self._fields["dt"]
        sub_old = self._fields["W_sub_old"]
        W_entry = self._fields["W_entry"]
        W_entry.assign(W)

        n_sub = 1
        for _ in range(max_halvings + 1):
            dt_c.assign(dt_total / n_sub)
            W.assign(W_entry)
            sub_old.assign(W_entry)
            try:
                for _k in range(n_sub):
                    self._solve_water(W, sub_old, dt_c, **kw)
                    sub_old.assign(W)
                W_old.assign(W_entry)
                dt_c.assign(dt_total)
                return n_sub
            except ConvergenceError:
                n_sub *= 2
        raise ConvergenceError(
            f"water transport failed to converge even at dt/{n_sub // 2}"
        )

    # ------------------------------------------------------------------
    def apply_phase_change(self, *, enthalpy, water, density, dt):
        """Exchange mass and latent heat between ``water`` and ``enthalpy``.

        Updates both fields in place and returns the freezing rate
        ``m = f/dt`` [kg m^-3 s^-1], positive for freezing.

        Everything is done with ``interpolate`` on UFL expressions, which is
        pointwise at CG nodes and stays on the pyadjoint tape - unlike a direct
        ``.dat.data`` update, which would silently break differentiability.
        """
        hydro = self.hydrology_model
        V = enthalpy.function_space()

        f = fd.Function(V, name="freeze_increment")
        f.interpolate(hydro.phase_change_increment(enthalpy, water, density))

        water.interpolate(water - f)
        enthalpy.interpolate(enthalpy + hydro.enthalpy_increment(f, density))

        m = self._fields.get("m_phase")
        if m is None or m.function_space() != V:
            m = fd.Function(V, name="refreezing_rate")
            self._fields["m_phase"] = m
        m.interpolate(f / dt)
        return m

    # ------------------------------------------------------------------
    def prognostic_solve(
        self,
        *,
        enthalpy,
        water,
        density,
        firn_velocity,
        dt,
        grain_radius2=None,
        surface_melt_flux=None,
        basal_flux=None,
        surface_id=2,
        base_id=1,
        water_bcs=None,
        update_enthalpy=True,
        include_drainage=True,
    ):
        """Advance ``(W, H)`` by one step.

        Parameters
        ----------
        surface_melt_flux:
            Downward meltwater mass flux into the surface [kg m^-2 s^-1],
            positive into the column. This is the melt path: leaving it at
            ``None`` reproduces the old closed-column behaviour in which no
            water could ever enter.
        basal_flux:
            Downward water loss at the base [kg m^-2 s^-1], positive out of the
            column. ``None`` is an impermeable bed.
        grain_radius2:
            Prognostic grain radius squared [m^2], used by the Calonne (2012)
            permeability. ``None`` falls back to ``params.r2_default``.

        Returns
        -------
        (H, W, m_phase)
        """
        if water_bcs is None:
            water_bcs = []

        if "H_old" not in self._fields:
            self._setup(enthalpy=enthalpy, water=water, dt=dt)
        else:
            self._fields["H_old"].assign(enthalpy)
            self._fields["W_old"].assign(water)
            self._fields["dt"].assign(
                float(dt) if not isinstance(dt, fd.Constant) else dt
            )

        V = self._fields["V"]
        psi = self._fields["test"]
        H, W, rho, w = enthalpy, water, density, firn_velocity
        H_old = self._fields["H_old"]
        W_old = self._fields["W_old"]
        dt_c = self._fields["dt"]

        hydro = self.hydrology_model
        firn = self.firn_model

        # 1) water transport (sub-stepped if the nonlinear solve needs it)
        self._n_sub_last = self._water_transport(
            W, W_old, float(dt) if not isinstance(dt, fd.Constant) else float(dt),
            rho=rho, w=w, grain_radius2=grain_radius2,
            surface_flux=surface_melt_flux, surface_id=surface_id,
            basal_flux=basal_flux, base_id=base_id,
            include_drainage=include_drainage, bcs=water_bcs,
        )

        # 2) enthalpy transport
        if update_enthalpy:
            F_H = hydro.enthalpy_form(
                H, H_old, rho, w,
                K_func=firn.thermal_diffusivity, test=psi, dt=dt_c,
            )
            problem_H = fd.NonlinearVariationalProblem(
                F_H, H, J=fd.derivative(F_H, H, fd.TrialFunction(V))
            )
            fd.NonlinearVariationalSolver(
                problem_H, solver_parameters=self.solver_parameters
            ).solve()

        # 3) local phase change
        m_phase = self.apply_phase_change(
            enthalpy=H, water=W, density=rho, dt=dt_c
        )

        return H, W, m_phase
