# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FirnGrain is a 1D firn column model implemented in Python using the **Firedrake** finite element library. It models firn (compacting snow) densification dynamics with support for grain growth, stress evolution, and inverse/adjoint modeling.

## Commands

Runs need the Firedrake environment, single-threaded, from script files
(PETSc intercepts `python -c`):

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 /home/andrew/venv-firedrake-2026/bin/python <script>
```

```bash
# Install in development mode
pip install -e .

# Run all tests (testpaths = test/)
pytest test/

# Run a specific test
pytest test/test_firnmice_checkpoint.py

# Run a case study, then rebuild its figures from results/
PYTHONPATH=src OMP_NUM_THREADS=1 python tutorials/southpole/run.py
PYTHONPATH=src OMP_NUM_THREADS=1 python tutorials/southpole/plot_misfits.py

# Run scripts directly (many are scripts, not pytest-style)
PYTHONPATH=src OMP_NUM_THREADS=1 python tutorials/summit/forward_summit.py
```

Some test scripts use environment variables for control:
```bash
FIRNMICE_RUN_FULL=1 pytest test/test_firnmice_checkpoint.py
```

## Repository structure

Four top-level parts (see the root `README.md` for the full tree):

- **`src/firnpack/`** - the package; everything importable (`import firnpack`)
- **`test/`** - pytest tests, plus per-site script history, run logs, staged data
- **`tutorials/`** - the three case studies (synthetic, southpole, summit), each a
  thin config over `firnpack.inverse`. Each holds `run.py`, `plot_*.py`,
  `data/` (tracked), `results/` (MAP JSONs, ignored), `figures/` (ignored)
- **`doc/`** - methods notes, derivations, paper-facing documentation

Figures are never committed: `plot_*.py` rebuilds them from `results/` into the
tutorial's `figures/`. `results/` is the expensive artifact; `figures/` is not.

## Architecture

The codebase follows a **model-solver separation pattern**:

- **`src/firnpack/models/`** — Define PDE weak forms and physical relationships (using Firedrake's UFL)
- **`src/firnpack/solvers/`** — Time-stepping logic, boundary conditions, Firedrake solver configuration
- **`src/firnpack/inverse/`** — Config-driven adjoint assimilation engine (`SiteConfig` + `assimilate`)
- **`src/firnpack/constants.py`** — Physical constants in SI units

### Core Classes

**`FirnParameters`** (dataclass in solver files): Physical and numerical parameters including theta-scheme weights (`theta_H=0.5`, `theta_rho=0.878`), densification tuning constants (`c0`, `kg`, `Ec`, `Eg`), and grain growth parameters.

**`FirnModel`** (`src/firnpack/models/firn.py`): Defines all physical closures (densification rates, grain growth, stress), temperature-enthalpy conversions, and UFL weak forms for each prognostic variable.

**`FirnColumnSolver`** (`src/firnpack/solvers/firn_solver.py`): Time-stepper with two modes:
- **Mode A** — Basic `(H, ρ, w)`: enthalpy, density, velocity (Arthern/Ligtenberg densification)
- **Mode B** — Full-density `(H, ρ, σ, r², w)`: adds deviatoric stress and grain-radius-squared

### Key Technologies

- **Firedrake** — Finite element discretization and solver (Newton-Krylov, GMRES with LU preconditioning)
- **`firedrake.adjoint`** — Automatic differentiation for inverse/parameter inference problems
- **`firedrake.CheckpointFile`** (HDF5) — State persistence and checkpointing
- **UFL** (Unified Form Language) — Symbolic PDE specification within Firedrake

### Test Directory

`test/` contains both pytest-style tests (`test_*.py`) and standalone experimental scripts (`forward_*.py`, `*_inversion.py`, `*_checkpoint.py`). Real field data for Summit (Greenland), WAIS Divide, and South Pole are staged under `test/summit/`, `test/WAIS/`, and `test/SouthPole/`.
