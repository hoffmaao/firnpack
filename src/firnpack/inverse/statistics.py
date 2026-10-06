"""firnpack.inverse.statistics - icepack-style data assimilation problem/solver.

Modelled on icepack's ``StatisticsProblem`` + ``MaximumProbabilityEstimator``
(``icepack/statistics.py``): a passive problem object holds the four pieces of an
inverse problem, and an estimator maximises the posterior probability (minimises
the negative log-posterior) over the controls.

    problem = StatisticsProblem(simulation, loss_functional, regularization, controls)
    est     = MaximumProbabilityEstimator(problem, max_iterations=80)
    m_map   = est.solve()

Two deliberate divergences from icepack, both forced by firn:

* **Backend is scipy L-BFGS-B, not ROL.** firnpack optimises Real-space
  (mesh-attached scalar) controls under box bounds, with a per-control scaling
  that the objective's absolute ftol test depends on. That machinery is proven
  (it reproduces the frozen South Pole MAP), so the estimator keeps it and wraps
  it in the icepack object shape rather than adopting ROL.

* **loss/regularization return objective scalars, not forms.** icepack's
  observable is a field and its loss is ``∫(u-u_obs)² dx`` - one assemblable
  form. firn observables are kernel-averaged point predictions, so the misfit is
  a sum of scalar residuals²; ``loss_functional(state)`` and
  ``regularization(controls)`` therefore return the on-tape objective
  contribution (an ``AdjFloat``) directly, and ``solve`` sums them into J.

``simulation`` stays a plain callable, so callers evaluate the forward at any
controls for diagnostics or truth-synthesis without touching the optimiser -
exactly as icepack's tests and notebooks call ``simulation(θ)`` directly.
"""
from __future__ import annotations

import collections.abc

import numpy as np
from firedrake.adjoint import (
    Control,
    compute_derivative,
    continue_annotation,
    pause_annotation,
)
from scipy.optimize import minimize as _sp_minimize


def _as_list(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes)):
        return list(x)
    return [x]


class StatisticsProblem:
    """A data-assimilation problem: simulation, loss, regularization, controls.

    Parameters
    ----------
    simulation
        ``controls -> state``. Runs the forward model and returns whatever the
        loss needs (for firn, the settled column fields). Called by the
        estimator, and callable directly for forward-only diagnostics.
    loss_functional
        ``state -> objective scalar`` (negative log-likelihood / model-data
        misfit). A single callable or an iterable of them; summed.
    regularization
        ``controls -> objective scalar`` (negative log-prior). A single callable
        or an iterable of them; summed. Pass ``[]`` for no prior.
    controls
        The control(s): a single Firedrake ``Function``/``Constant`` or an
        iterable of them.
    copy_controls
        If True (icepack default), the controls are deep-copied on store so
        ``solve`` cannot mutate the caller's initial guess. firnpack's
        ``simulation`` reads the controls through deep closure capture (params →
        model → forms), so the forward model and the stored controls must be the
        *same* objects - the firn factory passes ``copy_controls=False``.
    bounds
        Optional box bounds aligned with ``controls`` as a list of ``(lo, hi)``
        in the controls' internal coordinates (log-space where the control is
        logged). firnpack extension: the L-BFGS-B backend needs them.
    scale
        Optional per-control scale array (typically each control's prior sigma).
        firnpack extension: the optimiser steps and the absolute ftol test are
        taken in scaled coordinates.
    """

    def __init__(self, simulation, loss_functional, regularization, controls,
                 bounds=None, scale=None, copy_controls=True, reset=None,
                 grad_scale=None):
        self._simulation = simulation
        self._loss_functionals = _as_list(loss_functional)
        self._regularizations = _as_list(regularization)
        ctrls = _as_list(controls)
        self._controls = [c.copy(deepcopy=True) for c in ctrls] if copy_controls else ctrls
        self._bounds = list(bounds) if bounds is not None else None
        self._scale = np.asarray(scale, float) if scale is not None else None
        self._reset = reset
        self._grad_scale = float(grad_scale) if grad_scale is not None else 1.0

    @property
    def simulation(self):
        return self._simulation

    @property
    def loss_functional(self):
        return self._loss_functionals

    @property
    def regularization(self):
        return self._regularizations

    @property
    def controls(self):
        return self._controls

    @property
    def bounds(self):
        return self._bounds

    @property
    def scale(self):
        return self._scale

    @property
    def reset(self):
        """Optional per-evaluation reset (backends that rebuild solvers each step).

        Called at the start of every objective evaluation, inside annotation,
        before the controls are assigned. firnpack uses it to clear the pyadjoint
        tape and rebuild the NaN-robust solvers, matching the historical engine.
        """
        return self._reset

    @property
    def grad_scale(self):
        """Scalar correcting pyadjoint's raw control gradient to true dJ/dx.

        1.0 for ordinary Function controls. For firn's Real-space (mesh-attached
        scalar) controls the raw gradient is integral-normalised, so the true
        derivative is ``raw * domain_length`` - the factor the historical engine
        applied as ``*dlen``.
        """
        return self._grad_scale


class MaximumProbabilityEstimator:
    """Maximise the posterior probability of the controls (MAP estimate).

    Mirrors icepack's estimator: build the reduced functional
    ``J = Σ loss + Σ reg`` over ``Control``s of the problem's controls, minimise
    it, and return the optimised controls. The backend here is scipy L-BFGS-B on
    the Real-space controls in scaled coordinates (see the module docstring).

    Parameters
    ----------
    problem
        A :class:`StatisticsProblem`.
    method
        Only ``"L-BFGS-B"`` is implemented (the proven firn backend).
    max_iterations, ftol, gtol, maxls
        scipy L-BFGS-B options. Defaults match the historical engine so the
        South Pole MAP reproduces.
    verbose
        Print per-iteration J / rms lines.
    callback
        Optional ``(n_eval, J, gradient, controls) -> None`` invoked after each
        objective evaluation, for progress logging.
    """

    def __init__(self, problem, method="L-BFGS-B", max_iterations=80,
                 ftol=1e-8, gtol=1e-7, maxls=30, verbose=True, callback=None):
        if method != "L-BFGS-B":
            raise NotImplementedError("Only the L-BFGS-B backend is implemented.")
        self._problem = problem
        self._method = method
        self._max_iterations = int(max_iterations)
        self._ftol, self._gtol, self._maxls = float(ftol), float(gtol), int(maxls)
        self._verbose = verbose
        self._callback = callback
        self._controls = None
        self._state = None
        self._J_hist = []

    @property
    def problem(self):
        return self._problem

    @property
    def controls(self):
        return self._controls

    @property
    def state(self):
        return self._state

    @property
    def J_hist(self):
        return self._J_hist

    def _objective(self, controls):
        """J = Σ loss(state) + Σ reg(controls) on the active tape."""
        self._state = self._problem.simulation(controls)
        J = 0.0
        for loss in self._problem.loss_functional:
            J = J + loss(self._state)
        for reg in self._problem.regularization:
            J = J + reg(controls)
        return J

    def forward_result(self):
        """Evaluate the objective once at the current controls; return J.

        A convenience for forward-only diagnostics (icepack calls
        ``simulation(θ)`` directly for the same purpose). Runs the problem's
        reset (tape clear + rebuild) so it is safe to call standalone.
        """
        if self._problem.reset is not None:
            self._problem.reset()
        else:
            continue_annotation()
        J = float(self._objective(self._problem.controls))
        pause_annotation()
        return J

    def solve(self):
        """Run the optimisation; return the controls at the MAP.

        The controls are optimised in place on the problem's own control
        Functions and returned (a single Function, or a list matching the
        input), as icepack does.
        """
        problem = self._problem
        controls = problem.controls
        self._controls = controls

        n = len(controls)
        x0 = np.array([float(c.dat.data_ro[0]) for c in controls])
        scale = problem.scale if problem.scale is not None else np.ones(n)
        gscale = problem.grad_scale
        if problem.bounds is not None:
            lb = np.array([b[0] for b in problem.bounds], float)
            ub = np.array([b[1] for b in problem.bounds], float)
        else:
            lb = np.full(n, -np.inf)
            ub = np.full(n, np.inf)

        n_eval = [0]
        self._J_hist = []

        def eval_J_and_grad(xs):
            x = np.clip(x0 + xs * scale, lb, ub)
            # Per-eval reset: clear the tape and rebuild solvers (firn's
            # NaN-robust protocol), inside annotation, before assigning controls.
            # Controls are wrapped fresh each eval because clear_tape invalidates
            # the previous block variables.
            if problem.reset is not None:
                problem.reset()
            else:
                continue_annotation()
            for c, v in zip(controls, x):
                c.assign(float(v))
            try:
                J = self._objective(controls)
                dJ = compute_derivative(J, [Control(c) for c in controls], apply_riesz=True)
                Jv = float(J)
                g = np.array([float(gi.dat.data_ro[0]) for gi in dJ]) * gscale * scale
            except Exception as exc:  # forward blew up; steer the optimiser back
                if self._verbose:
                    print(f"  *** {exc}")
                Jv, g = 1e8, xs * 100.0
            pause_annotation()
            n_eval[0] += 1
            self._J_hist.append(Jv)
            if self._callback is not None:
                self._callback(n_eval[0], Jv, g, controls)
            return Jv, g

        lb_s = (lb - x0) / scale
        ub_s = (ub - x0) / scale
        res = _sp_minimize(
            eval_J_and_grad, np.zeros(n), jac=True, method="L-BFGS-B",
            bounds=list(zip(lb_s, ub_s)),
            options={"maxiter": self._max_iterations, "ftol": self._ftol,
                     "gtol": self._gtol, "maxls": self._maxls},
        )
        x_map = x0 + res.x * scale
        for c, v in zip(controls, x_map):
            c.assign(float(v))

        self.result = res
        return controls if n > 1 else controls[0]
