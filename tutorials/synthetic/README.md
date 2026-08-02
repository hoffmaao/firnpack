# Synthetic OSSE - method verification & identifiability

The credibility anchor for the paper: an Observing System Simulation Experiment
that proves the inverse framework recovers **known truth**, validates the
adjoint gradient, and characterizes what the observations can and cannot
constrain.

## Design

1. **Truth.** Fix a known `(densification params, surface-T history, accumulation
   history, conductivity)` - a cold-firn scenario carrying both a large old-time
   excursion and an exponential recent trend, so the experiment probes the
   thermal null space and the recent-trend recovery at once.
2. **Synthetic observations.** Run the forward model at truth and sample
   density, the depth–age *gradient*, borehole temperature, compaction rate,
   and ApRES-like differenced velocity at realistic depths, adding noise at the
   **recalibrated, data-derived sigma levels** the real sites now use, so
   χ²/N ≈ 1 at truth by construction.
3. **Recover.** Run the *same* assimilation engine used for South Pole / Summit,
   from the South Pole knot layouts rather than truth's, and check the MAP
   recovers truth within the Laplace posterior bars.
4. **Adjoint verification.** Taylor test (order → 2) and finite-difference
   gradient check on the reduced functional.
5. **Identifiability.** Demonstrate the empirically-found limits from the real
   sites in a controlled setting - e.g. the isothermal (k, Ea) rate degeneracy,
   and the d(age)/dz null space at sub-decadal accumulation scales (why b(t) is
   only recoverable at multidecadal resolution).

The exact truth signals, observation blocks, sigmas, knot layouts, and the
env flags that vary them (including the DEEP mode that extends the domain into
the deep ice) are documented where they are defined, in `run.py`'s module
docstring and inline comments. Read that, not a second copy here.

## Build notes

Build on the **current** H&L + thermal + accumulation physics
(`firnpack.models.firn`, `firnpack.physics.densification`). The archived `test_gadopt_age_layers_*` OSSE is a
useful *structural* template (truth dict, off-tape truth forward, synthetic
dated-layer obs, multi-control inversion + UQ) but uses the older grain-based
Kingslake controls - reuse its shape, not its model.

Seeds:
- `archive/southpole/scripts/diagnostics/clean_stepper_taylor.py` (untracked
  archive; not present in a fresh clone) - the lean adjoint stepper + Taylor
  test matching the production assimilation architecture.
- `test/test_adjoint.py`, `test/test_adjoint_sparse_obs.py` - single-parameter
  twin experiments (dense / sparse obs) that recover a known `kg`.

Status: built - see `run.py` (the OSSE itself), `plot.py` (figures, a pure
reader of `output/`), and `diagnostics/` for the observability, thermal
degeneracy, deep-observability, and reanalysis experiments.
