# Firnpack

A 1D firn column model - densification, heat/enthalpy transport, and
accumulation - with an **adjoint-based inverse framework** (Firedrake +
pyadjoint) for assimilating ice-core and geophysical observations to recover
firn physics and climate history. This code is in development and may change
as use cases mature.

## Solver

Given depth-resolved observations like density, dated layers, borehole temperature,
and phase-sensitive radar (ApRES) vertical velocity or borehole compaction rates
Firnpack solves equations for firn densification that can be used with time-dependent
data assimilative schemes to better constrain/parameterize:

- the **densification law** (Herron–Langway two-stage rates, with an optional
  deep-firn shape exponent),
- **firn thermal conductivity** (Calonne 2019 or a ρ² law),
- the surface **temperature history**, and
- the surface **accumulation history**,

these calibrated models can then be used to initialize **firn reanalysis**
and attribute firn-air-content and surface-height change to climate forcing with Laplace
uncertainty quantification.

## Install

Firnpack runs inside a [Firedrake](https://www.firedrakeproject.org) environment.

```bash
# 1. Install Firedrake (provides PETSc, pyadjoint) - see the Firedrake docs.
# 2. Into that environment:
pip install -e .
pip install -r requirements.txt   # numpy/scipy/pandas/matplotlib (pinned)
```

Runs are single-threaded and use PETSc; always set `OMP_NUM_THREADS=1` and run
script files (PETSc intercepts `python -c`):

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python tutorials/southpole/run.py
```

The test suite runs in two tiers. `pytest` alone runs the fast tier (the
`slow` marker is deselected by default in `pyproject.toml`); the long-running
end-to-end tests opt in explicitly:

```bash
pytest              # fast tier
pytest -m slow      # long-running end-to-end tests
```


## Physics references

Herron & Langway (1980); Arthern et al. (2010) / Ligtenberg et al. (2011);
Kingslake et al. (2022); Calonne et al. (2019, conductivity); Gagliardini &
Meyssonnier (1997, compressible Stokes). Per-site observation provenance is in
each `tutorials/<site>/data/README.md`.

## License

Copyright (C) 2026 Andrew Hoffman <ah301@rice.edu>, Rice University.

Firnpack is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. The full text is in [LICENSE](LICENSE), or at
<https://www.gnu.org/licenses/>.

This matches [icepack](https://github.com/icepack/icepack), which Firnpack
builds on.
