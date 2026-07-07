# FirnGrain

A 1D firn column model — densification, heat/enthalpy transport, and accumulation
— with an **adjoint-based inverse framework** (Firedrake + pyadjoint) for
assimilating ice-core and geophysical observations to recover firn physics and
climate history.

> **Status: research code being prepared for publication.** The canonical model
> is stable; the inverse framework and the three case studies below are being
> consolidated into a clean, reproducible layout. See `PUBLICATION_PLAN.md` for
> the roadmap and current state.

## What it does

Given depth-resolved observations at a site — density, ice-core depth–age,
borehole temperature, and (where available) phase-sensitive radar (ApRES)
vertical velocity — FirnGrain solves an inverse problem for:

- the **densification law** (Herron–Langway two-stage rates, with an optional
  deep-firn shape exponent),
- **firn thermal conductivity** (Calonne 2019 or a ρ² law),
- the surface **temperature history**, and
- the surface **accumulation history**,

then uses the calibrated model as a **firn reanalysis** (attribution of
firn-air-content and surface-height change to climate forcing) with Laplace
uncertainty quantification.

## The three test cases (paper framework)

| Test | Purpose | Regime |
|------|---------|--------|
| **Synthetic OSSE** | recover known truth; validate the adjoint & identifiability | controlled |
| **South Pole** | multi-observable real-data assimilation (flagship) | cold, low-accumulation |
| **Summit, Greenland** | transferability to a contrasting site | warm, high-accumulation |

Each is a thin configuration over one shared inverse engine (see
`PUBLICATION_PLAN.md`, Phase 2).

## Install

FirnGrain runs inside a [Firedrake](https://www.firedrakeproject.org) environment.

```bash
# 1. Install Firedrake (provides PETSc, pyadjoint) — see the Firedrake docs.
# 2. Into that environment:
pip install -e .
pip install -r requirements.txt   # numpy/scipy/pandas/matplotlib (pinned)
```

Runs are single-threaded and use PETSc; always set `OMP_NUM_THREADS=1` and run
script files (PETSc intercepts `python -c`):

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/southpole/run.py
```

## Layout

```
src/firn/          canonical model: constants, mesh, physics/densification,
                   models/firn (FirnParameters, FirnModel), solvers/firn_solver
                   inverse/       shared assimilation engine (in progress)
experiments/
  synthetic/       OSSE: truth -> synthetic obs -> recover
  southpole/       South Pole assimilation + reanalysis
  summit/          Summit Greenland assimilation
tests/             pytest unit tests (adjoint FD/Taylor, conservation, MMS)
PUBLICATION_PLAN.md  roadmap + repo-cleanup status
```

## Physics references

Herron & Langway (1980); Arthern et al. (2010) / Ligtenberg et al. (2011);
Kingslake et al. (2022); Calonne et al. (2019, conductivity); Gagliardini &
Meyssonnier (1997, compressible Stokes). Per-site observation provenance is in
each `experiments/<site>/data/README.md`.

## License

_TBD_ — see `PUBLICATION_PLAN.md` (decision pending).
