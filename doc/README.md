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
| How to run a case study | `tutorials/<site>/run.py` and the root `README.md` |
| Observation provenance for a site | `tutorials/<site>/data/README.md` |
| South Pole inversion history (r5→r8/r10) | `southpole_inversion_handoff.md` |
| South Pole adjoint rules and gotchas | `southpole_adjoint_notes.md` |
| Superseded scripts, logs, and raw data | `archive/` (untracked; see root `README.md`) |
| Repo layout | root `README.md` |

## Contents

- `firn_hydrology.md` - percolation, refreezing and the aquifer column: the
  mixed-form Richards model on a compacting medium, the implicit phase-change
  sink, the ice-mass source in the density equation, the surface energy
  balance, and the four failure modes found on the way to a SE Greenland
  aquifer (each with its regression test).

- `southpole_inversion_handoff.md` - the South Pole inversion history and the
  pyadjoint rules the engine depends on. Rescued from the pre-engine script
  tree when that history moved to `archive/`.
- `southpole_adjoint_notes.md` - South Pole working notes and adjoint gotchas,
  rescued from the same place.

Both describe the flagship case study and predate the shared engine, so read
them alongside `src/firnpack/inverse/` rather than as a description of it.
