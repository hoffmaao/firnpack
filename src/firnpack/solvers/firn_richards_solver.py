"""Time stepping for mixed-form Richards percolation in firn.

One step is:

1. Refresh the retention and conductivity curves against the current density
   and grain radius - the medium moves and compacts under the water.
2. Evaluate the freezing increment from the state at the start of the step.
3. One backward-Euler Richards solve for the head, **with that freezing as a
   volumetric sink in the equation**.
4. Release the matching latent heat into the matrix enthalpy.

Why the sink is in the equation
-------------------------------
The obvious alternative - solve, then edit the state - does not conserve mass.
Editing means computing ``theta``, subtracting the frozen water, and inverting
the retention curve to get a head back. The inversion is exact, but only *at
nodes*: interpolating a nonlinear expression into DG1 reproduces it at the
nodes and a straight line in between, while the water budget is an integral
over cell interiors. The two disagree, and the run leaks a few percent of the
melt total. It also silently discards water in saturated cells, where
``theta`` is pinned at ``theta_s`` and the excess is carried as head instead.

Putting the sink in the residual removes both problems at once. The mixed form
integrates to

    d/dt int theta  =  boundary flux + int source,

exactly, in the quadrature the residual is assembled with, so whatever the sink
removes is exactly what the budget records - saturated or not. The enthalpy
release uses the same field, so mass and energy stay consistent by
construction rather than by cancellation.

The increment is still evaluated explicitly, from the start-of-step state, and
still bounded by both the cold content and the available water; that is what
keeps a single step from overshooting ``H = 0`` or over-drawing a cell.
"""

import firedrake as fd
from firedrake.exceptions import ConvergenceError


DIRECT = {
    "snes_type": "newtonls",
    # L2, not backtracking. Once a real water table forms the column spans
    # roughly -40 m to +8 m of head and the relative conductivity between
    # neighbouring cells varies over many orders of magnitude. Backtracking
    # enforces sufficient decrease on the merit function and stalls there; the
    # L2 search minimises the residual norm along the direction and gets
    # through. Measured on a 12-year aquifer column that fails at year 6.1 -
    # the moment the aquifer actually forms - under every other option tried:
    # "bt", "cp", trust region ("newtontr") and "bt" with a five-fold
    # iteration cap all fail after nine successively halved step sizes, while
    # "l2" completes the run. It is not a compressibility problem either:
    # raising the specific storage 10x and 100x changes nothing.
    "snes_linesearch_type": "l2",
    "snes_rtol": 1e-10,
    "snes_atol": 1e-12,
    "snes_max_it": 60,
    "ksp_type": "preonly",
    "pc_type": "lu",
}


class FirnRichardsSolver:
    def __init__(self, model, solver_parameters=None):
        self.model = model
        self.solver_parameters = solver_parameters or dict(DIRECT)
        self._fields = {}
        self._solver = None

    @property
    def fields(self):
        return self._fields

    # ------------------------------------------------------------------
    def _setup(self, *, head, curves, bcs, firn_velocity, source):
        V = head.function_space()
        self._fields["V"] = V
        self._fields["head"] = head
        self._fields["head_old"] = head.copy(deepcopy=True)
        self._fields["head_entry"] = head.copy(deepcopy=True)
        # Water content at the start of the step, measured with the curves that
        # were in force then. Carried as its own field because the medium
        # compacts: re-deriving it from head_old with the refreshed curves
        # would lose the water squeezed out of the closing pore space.
        self._fields["theta_old"] = fd.Function(V, name="theta_old")
        self._fields["theta_entry"] = fd.Function(V, name="theta_entry")
        # Preallocated. Allocating this inside the sub-step loop meant a fresh
        # Function per sub-step across thousands of steps, which is what ran
        # the machine out of memory on a 12-year run.
        self._fields["head_prev"] = fd.Function(V, name="head_prev")
        self._fields["dt"] = fd.Constant(1.0)
        self._fields["curves"] = curves
        self._fields["bcs"] = bcs
        self._fields["firn_velocity"] = firn_velocity
        self._fields["m_phase"] = fd.Function(V, name="refreezing_rate")
        # freeze: what actually froze over the step [kg m^-3], accumulated
        # across sub-steps from the sink the solve applied. freeze_demand is
        # the staged upper bound from the cold content.
        self._fields["freeze"] = fd.Function(V, name="freeze_increment")
        self._fields["freeze_demand"] = fd.Function(V, name="freeze_demand")
        self._fields["freeze_attempt"] = fd.Function(V, name="freeze_attempt")
        self._fields["freeze_sub"] = fd.Function(V, name="freeze_sub")
        # Volumetric water sink demand [1/s] carrying the phase change. Held
        # as a Function and assigned into, because the residual is assembled
        # once: anything passed by value would be frozen at its initial state.
        self._fields["phase_sink"] = fd.Function(V, name="phase_sink")
        self._fields["user_source"] = source
        # Water held elastically by the specific-storage term rather than in
        # the retention curve. It is real storage and has to be counted, or a
        # budget written as "in = frozen + change in theta" misses it and
        # reports a leak that is not there.
        self._fields["elastic_storage_m"] = 0.0

        # The sink is implicit in the head. Sizing it from the entry state
        # alone overdrew cells that a wetting front passed through and left
        # within one step - the staged sink kept taking water that had already
        # drained on, theta fell below theta(h_min), and the head ran to
        # -1000 m to satisfy storage. One such cell under a forming ice lens,
        # next to a saturated one at +23 m, is what stopped Newton in the
        # ERA5-forced column.
        #
        # The demand is multiplied by a ramp in the water still present,
        # linear over the last ``freeze_ramp_theta`` of it. Treated
        # implicitly that is a linear decay toward theta_min, so
        # theta - theta_min = (theta_old - theta_min) / (1 + dt demand / delta)
        # stays positive by construction. A min(demand, available/dt) limiter
        # guarantees the same bound but puts the Jacobian's switch exactly
        # where every water-limited cell sits; Newton chattered across it and
        # failed at every sub-step size, smoothed or not.
        p = self.model.params
        theta_min = self.model.driest_moisture(curves)
        remaining = fd.max_value(curves.moisture_content(head) - theta_min, 0.0)
        ramp = fd.min_value(remaining / fd.Constant(p.freeze_ramp_theta), 1.0)
        sink_rate = -self._fields["phase_sink"] * ramp
        self._fields["sink_rate"] = sink_rate
        total_source = -sink_rate
        if source is not None:
            total_source = total_source + source
        # realised freezing over one sub-step, projected with the residual's
        # own quadrature so the reported mass is the mass the solve removed
        md = {"quadrature_degree": 3}
        self._fields["freeze_projector"] = fd.Projector(
            fd.Constant(p.rho_w) * sink_rate * self._fields["dt"],
            self._fields["freeze_sub"], form_compiler_parameters=md)

        # Boundary outflow over one sub-step [m], as a form assembled after
        # each sub-step solve. Evaluating it once on the end-of-step head
        # was exact only when the step was not sub-stepped; with sub-steps
        # the outflow the residual actually applied is the sum over them.
        mesh = V.mesh()
        outflow_form = None
        xs = fd.SpatialCoordinate(mesh)
        e_z = fd.grad(xs[mesh.topological_dimension - 1])
        nrm = fd.FacetNormal(mesh)
        for bc_id, spec in bcs.items():
            if "advect" in spec and firn_velocity is not None:
                term = (curves.moisture_content(head) * firn_velocity
                        * fd.dot(e_z, nrm) * self._fields["dt"]
                        * fd.ds(bc_id, domain=mesh, metadata=md))
                outflow_form = term if outflow_form is None else outflow_form + term
        self._fields["outflow_form"] = outflow_form
        self.last_outflow_m = 0.0

        self._fields["theta_old"].interpolate(curves.moisture_content(head))
        self._fields["theta_entry"].assign(self._fields["theta_old"])
        F = self.model.residual(
            head=head, head_old=self._fields["head_old"], curves=curves,
            dt=self._fields["dt"], bcs=bcs,
            theta_old=self._fields["theta_old"],
            firn_velocity=firn_velocity, source=total_source,
        )
        problem = fd.NonlinearVariationalProblem(F, head)
        self._solver = fd.NonlinearVariationalSolver(
            problem, solver_parameters=self.solver_parameters)

    # ------------------------------------------------------------------
    def _advance_head(self, dt_total, max_halvings=8):
        """One Richards step, sub-stepped if Newton will not take it whole.

        Percolation is far faster than the firn it moves through, so a step
        sized for the column can be well beyond what the nonlinear problem
        accepts during a melt event. Halving on failure costs nothing for the
        rest of the year, when the full step converges first try.
        """
        h = self._fields["head"]
        h_old = self._fields["head_old"]
        h_entry = self._fields["head_entry"]
        dt_c = self._fields["dt"]
        h_entry.assign(h)
        theta_old = self._fields["theta_old"]
        theta_entry = self._fields["theta_entry"]
        curves = self._fields["curves"]

        n_sub = 1
        for _ in range(max_halvings + 1):
            h.assign(h_entry)
            h_old.assign(h_entry)
            theta_old.assign(theta_entry)
            dt_c.assign(dt_total / n_sub)
            # Accumulated per attempt and only committed on success. An
            # attempt that fails part way through is rolled back and retried,
            # so adding straight to the running total would count the
            # discarded sub-steps as well and inflate the elastic storage the
            # water budget is closed against.
            elastic_attempt = 0.0
            outflow_attempt = 0.0
            freeze_attempt = self._fields["freeze_attempt"]
            freeze_attempt.assign(0.0)
            h_prev = self._fields["head_prev"]
            outflow_form = self._fields["outflow_form"]
            try:
                for _k in range(n_sub):
                    h_prev.assign(h_old)
                    self._solver.solve()
                    elastic_attempt += self._elastic_storage_increment(h, h_prev)
                    # what the solve actually applied in this sub-step
                    self._fields["freeze_projector"].project()
                    freeze_attempt.assign(freeze_attempt + self._fields["freeze_sub"])
                    if outflow_form is not None:
                        outflow_attempt += float(fd.assemble(outflow_form))
                    h_old.assign(h)
                    # within the step the medium is fixed, so later sub-steps
                    # may take theta_old straight off the curve
                    theta_old.interpolate(curves.moisture_content(h))
                dt_c.assign(dt_total)
                self._fields["elastic_storage_m"] += elastic_attempt
                self._fields["freeze"].assign(freeze_attempt)
                self.last_outflow_m = outflow_attempt
                return n_sub
            except ConvergenceError:
                n_sub *= 2
        raise ConvergenceError(
            f"Richards step failed to converge; tried {max_halvings + 1} step "
            f"sizes down to dt/{n_sub // 2}")

    def _elastic_storage_increment(self, h, h_prev):
        """Water the specific-storage term took up over one sub-step [m].

        The form is built once and re-assembled against the same Functions;
        rebuilding it per sub-step is what made this the hot spot.
        """
        form = self._fields.get("elastic_form")
        if form is None:
            model = self.model
            curves = self._fields["curves"]
            mesh = h.function_space().mesh()
            deg = h.function_space().ufl_element().degree()
            if not isinstance(deg, int):
                deg = max(deg)
            dxq = fd.dx(domain=mesh,
                        metadata={"quadrature_degree": 2 * deg + 1})
            Ss = fd.Constant(model.params.specific_storage)
            form = Ss * curves.saturation(h) * (h - h_prev) * dxq
            self._fields["elastic_form"] = form
        return float(fd.assemble(form))

    @property
    def elastic_storage_m(self):
        """Cumulative elastic storage since setup, as metres of water."""
        return self._fields.get("elastic_storage_m", 0.0)

    # ------------------------------------------------------------------
    def stage_phase_change(self, *, head, enthalpy, density, curves, dt):
        """Stage the freezing demand for the coming step from the current state.

        Positive ``f`` freezes. Bounded by the cold content on one side and by
        the water the retention curve can actually give up on the other, so one
        step cannot overshoot ``H = 0``. It is a demand, not the outcome: the
        residual limits the sink to the water each cell still holds at the end
        of the sub-step, and ``freeze`` records what was actually removed.
        """
        model = self.model
        p = model.params
        f = self._fields["freeze_demand"]
        theta = curves.moisture_content(head)
        f.interpolate(model.freeze_increment(
            enthalpy, theta, density, theta_min=model.driest_moisture(curves)))
        # water leaves as a volume fraction per second; the residual caps
        # this further by what each cell still holds when the sub-step ends
        self._fields["phase_sink"].interpolate(-f / (p.rho_w * dt))
        return f

    def release_latent_heat(self, *, enthalpy, density, dt):
        """Add the latent heat of the freezing that occurred; report its rate."""
        model = self.model
        f = self._fields["freeze"]
        enthalpy.interpolate(enthalpy + model.enthalpy_increment(f, density))
        m = self._fields["m_phase"]
        m.interpolate(f / dt)
        return m

    # ------------------------------------------------------------------
    def step(self, *, head, enthalpy, density, curves, dt,
             bcs, firn_velocity=None, grain_radius2=None, source=None,
             phase_change=True):
        """Advance the head by ``dt`` [s] and exchange phase.

        Returns ``(head, m_phase)``; ``m_phase`` is ``None`` when
        ``phase_change`` is off.
        """
        if self._solver is None:
            self._setup(head=head, curves=curves, bcs=bcs,
                        firn_velocity=firn_velocity, source=source)
        elif head is not self._fields["head"]:
            raise ValueError(
                "step() was called with a different head Function than the "
                "first call; assign into the original instead")
        elif bcs is not self._fields["bcs"]:
            raise ValueError(
                "step() was called with a different bcs dict than the first "
                "call. The variational problem is built once and holds the "
                "boundary values it was given, so a new dict is silently "
                "ignored: pass fd.Constant values once and assign into them. "
                "A plain float here freezes the boundary flux at its initial "
                "value - which, for a melt flux starting in midwinter, means "
                "no melt ever enters.")

        # Measure the water BEFORE refreshing: theta_s shrinks as the column
        # compacts, and the difference has to straddle that change or the water
        # squeezed out of the closing pore space is lost.
        self._fields["theta_entry"].interpolate(curves.moisture_content(head))

        # the medium has moved since the last step
        self.model.refresh_curves(curves, density, grain_radius2)

        dt_c = self._fields["dt"]
        dt_c.assign(float(dt))
        if phase_change:
            self.stage_phase_change(head=head, enthalpy=enthalpy,
                                    density=density, curves=curves, dt=dt_c)
        else:
            self._fields["phase_sink"].assign(0.0)

        self._n_sub_last = self._advance_head(float(dt))   # sets "freeze"

        # last_outflow_m was accumulated over the sub-steps by _advance_head,
        # on the heads each solve produced.

        m = None
        if phase_change:
            m = self.release_latent_heat(enthalpy=enthalpy, density=density,
                                         dt=self._fields["dt"])
        return head, m
