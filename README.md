# FirnGrain

A 1D firn column model — densification, heat/enthalpy transport, and accumulation
— with an **adjoint-based inverse framework** (Firedrake + pyadjoint) for
assimilating ice-core and geophysical observations to recover firn physics and
climate history.

> **Status: research code.** The canonical model is stable; the inverse
> framework and the three case studies below are being consolidated into a
> clean, reproducible layout.

## What it does

Given depth-resolved observations at a site — density, ice-core depth–age (as a
layer-gradient constraint), borehole temperature, and (where available)
phase-sensitive radar (ApRES) vertical velocity or borehole compaction-coil
shortening rates — FirnGrain solves an inverse problem for:

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

Each is a thin configuration over one shared inverse engine
(`src/firnpack/inverse/`).

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
PYTHONPATH=src OMP_NUM_THREADS=1 python tutorials/southpole/run.py
```

## Layout

The repository has four top-level parts: the **package**, the **tests**, the
**tutorials** (worked case studies), and the **documentation**.

```
src/firnpack/      the package - everything importable
  constants.py     physical constants (SI)
  mesh.py          column mesh construction
  physics/         pluggable rate laws (densification, grain growth)
  models/          FirnModel: PDE weak forms, closures, enthalpy<->T
  solvers/         FirnColumnSolver: time-stepping, BCs, mesh motion
  data/            climate histories, observation containers
  inverse/         shared config-driven assimilation engine
                   (SiteConfig + assimilate); the through-line for all
                   three tutorials

tutorials/         the three case studies, each a thin config over the engine
  synthetic/       OSSE: truth -> synthetic obs -> recover (validates adjoint)
  southpole/       South Pole assimilation + reanalysis (flagship)
  summit/          Summit, Greenland assimilation (transferability)

  Each tutorial has the same shape:
    run.py         the inversion itself
    config.py      build_cfg() -> the SiteConfig, incl. the error model and
                   its legacy-reproduction env guards; run.py and every
                   diagnostics/ script build the SAME config from it
                   (southpole/ and summit/; synthetic/ builds its own inline)
    plot.py        figures from output/ - pure reader, no Firedrake/solve
    diagnostics/   diagnostic + experiment scripts                  [tracked]
    data/          curated observation CSVs + provenance README   [tracked]
    output/        MAP JSONs (+ .h5) from long runs - expensive     [ignored]
    figures/       rebuilt any time from output/ by plot.py         [ignored]

  Two exceptions are tracked inside southpole/output/: the frozen reference
  MAPs sp_joint_r8.json and usp50_k_snow_fit.json. They are results by
  provenance but inputs by use - run.py warm-starts from r8 and the plot
  scripts compare against it - so they are version-controlled deliberately.

test/              the pytest suites (adjoint FD/Taylor, conservation, MMS),
                   each self-contained, plus three parked scripts that assert
                   nothing and are not collected - see CLAUDE.md

doc/               methods notes, derivations, and paper-facing documentation

archive/           parking lot, untracked - superseded outputs, staged raw
                   data, and the South Pole script/log history that predates
                   the shared engine. Nothing on the critical path; every
                   file is still recoverable from git history.

```

`output/` and `figures/` are both regenerable and untracked, but they are not
equally cheap: `output/` holds MAP JSONs that cost hours of compute, while
`figures/` is one `plot.py` invocation away. Keeping them apart is why
figures are safe to delete wholesale.

## Physics references

Herron & Langway (1980); Arthern et al. (2010) / Ligtenberg et al. (2011);
Kingslake et al. (2022); Calonne et al. (2019, conductivity); Gagliardini &
Meyssonnier (1997, compressible Stokes). Per-site observation provenance is in
each `tutorials/<site>/data/README.md`.

## License

Copyright (C) 2026 Andrew Hoffman <ah301@rice.edu>, Rice University.

FirnGrain is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. The full text is in [LICENSE](LICENSE), or at
<https://www.gnu.org/licenses/>.

This matches [icepack](https://github.com/icepack/icepack), which FirnGrain
builds on.
