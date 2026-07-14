# Documentation

Paper-facing documentation for FirnGrain: the methods notes, derivations, and
conventions that the code assumes but does not state.

This directory is for prose that outlives any single run. Things that belong
here:

- **Methods** - the forward model (enthalpy, densification, grain growth), the
  inverse formulation (objective, controls, priors), and the discretisation.
- **Derivations** - anything a reader would otherwise have to reverse-engineer
  from the weak forms in `src/firnpack/models/`.
- **Conventions and gotchas** - datum definitions, unit conventions, and the
  pyadjoint patterns the engine is built around (control-dependent expressions
  must be interpolated into CG1 before assembly; no control-dependent Dirichlet
  BCs; time-varying control forcing needs fresh literal `Constant`s per step).
- **Data provenance** - the per-site staging story. Note that the authoritative,
  machine-readable provenance for each site's observations stays next to the
  data itself, in `tutorials/<site>/data/README.md`.

## What lives elsewhere

| Looking for | Go to |
|---|---|
| Roadmap, phase status, publication plan | `PUBLICATION_PLAN.md` |
| How to run a case study | `tutorials/<site>/run.py` and the root `README.md` |
| Observation provenance for a site | `tutorials/<site>/data/README.md` |
| South Pole inversion history + adjoint rules | `test/southpole/` |
| Repo layout | root `README.md` |
