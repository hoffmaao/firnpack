# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FirnGrain is a 1D firn column model implemented in Python using the **Firedrake** finite element library. It models firn (compacting snow) densification dynamics with support for grain growth, stress evolution, and inverse/adjoint modeling.

## Commands

```bash
# Install in development mode
pip install -e .

# Run all tests
pytest test/

# Run a specific test
pytest test/test_firnmice_checkpoint.py

# Run test scripts directly (many are scripts, not pytest-style)
python test/forward_summit_v36.py
```

Some test scripts use environment variables for control:
```bash
FIRNMICE_RUN_FULL=1 pytest test/test_firnmice_checkpoint.py
```

## Architecture

The codebase follows a **model-solver separation pattern**:

- **`src/firn/models/`** — Define PDE weak forms and physical relationships (using Firedrake's UFL)
- **`src/firn/solvers/`** — Time-stepping logic, boundary conditions, Firedrake solver configuration
- **`src/firn/constants.py`** — Physical constants in SI units

### Core Classes

**`FirnParameters`** (dataclass in solver files): Physical and numerical parameters including theta-scheme weights (`theta_H=0.5`, `theta_rho=0.878`), densification tuning constants (`c0`, `kg`, `Ec`, `Eg`), and grain growth parameters.

**`FirnModel`** (`src/firn/models/firn.py`): Defines all physical closures (densification rates, grain growth, stress), temperature-enthalpy conversions, and UFL weak forms for each prognostic variable.

**`FirnColumnSolver`** (`src/firn/solvers/firn_solver.py`): Time-stepper with two modes:
- **Mode A** — Basic `(H, ρ, w)`: enthalpy, density, velocity (Arthern/Ligtenberg densification)
- **Mode B** — Full-density `(H, ρ, σ, r², w)`: adds deviatoric stress and grain-radius-squared

### Multiple Versions

Both models and solvers have multiple numbered versions (`firn_v2.py` through `firn_v4.py`, `firn_solver_v2.py` through `firn_solver_v5.py`). These represent iterative experimental refinements. The unnumbered files (`firn.py`, `firn_solver.py`) are the primary versions.

### Key Technologies

- **Firedrake** — Finite element discretization and solver (Newton-Krylov, GMRES with LU preconditioning)
- **`firedrake.adjoint`** — Automatic differentiation for inverse/parameter inference problems
- **`firedrake.CheckpointFile`** (HDF5) — State persistence and checkpointing
- **UFL** (Unified Form Language) — Symbolic PDE specification within Firedrake

### Test Directory

`test/` contains both pytest-style tests (`test_*.py`) and standalone experimental scripts (`forward_*.py`, `*_inversion.py`, `*_checkpoint.py`). Real field data for Summit (Greenland), WAIS Divide, and South Pole are staged under `test/summit/`, `test/WAIS/`, and `test/SouthPole/`.
