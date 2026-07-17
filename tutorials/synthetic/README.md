# Synthetic OSSE — method verification & identifiability

The credibility anchor for the paper: an Observing System Simulation Experiment
that proves the inverse framework recovers **known truth**, validates the
adjoint gradient, and characterizes what the observations can and cannot
constrain.

## Design

1. **Truth.** Fix a known `(densification params, surface-T history, accumulation
   history, conductivity)` — a plausible cold-firn scenario.
2. **Synthetic observations.** Run the forward model at truth, sample density,
   depth–age, borehole temperature (and optionally ApRES velocity) at realistic
   depths, add realistic correlated noise.
3. **Recover.** Run the *same* assimilation engine used for South Pole / Summit
   and check the MAP recovers truth within the Laplace posterior bars.
4. **Adjoint verification.** Taylor test (order → 2) and finite-difference
   gradient check on the reduced functional.
5. **Identifiability.** Demonstrate the empirically-found limits from the real
   sites in a controlled setting — e.g. the isothermal (k, Ea) rate degeneracy,
   and the d(age)/dz null space at sub-decadal accumulation scales (why b(t) is
   only recoverable at multidecadal resolution).

## Build notes

Build on the **current** H&L + thermal + accumulation physics (`firn.models.firn`,
`firn.physics.densification`). The archived `test_gadopt_age_layers_*` OSSE is a
useful *structural* template (truth dict, off-tape truth forward, synthetic
dated-layer obs, multi-control inversion + UQ) but uses the older grain-based
Kingslake controls — reuse its shape, not its model.

Seeds:
- `archive/southpole/scripts/diagnostics/clean_stepper_taylor.py` — the lean
  adjoint stepper + Taylor test matching the production assimilation architecture.
- `test/test_adjoint.py`, `test/test_adjoint_sparse_obs.py` — single-parameter
  twin experiments (dense / sparse obs) that recover a known `kg`.

Status: built — see `run.py`, `plot.py` (figures), and the `output/` scripts here.
