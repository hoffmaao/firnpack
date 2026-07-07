"""FirnGrain: A 1D firn column model with pluggable physics and adjoint support.

The user-facing API follows the icepack/hydropack pattern:

    import firn

    model = firn.FirnModel(densification=firn.physics.herron_langway)
    mesh  = firn.column_mesh(height=130, n=100, stretch=2.5)
    ...

Submodules:
    firn.physics      -- Pluggable rate laws (densification, grain growth)
    firn.data         -- Climate histories, observations, boundary data
    firn.models       -- FirnModel (PDE weak forms)
    firn.solvers      -- FirnColumnSolver (time-stepping, BCs, mesh motion)
    firn.statistics   -- Inverse problems, MAP estimation, UQ
    firn.constants    -- Physical constants in SI units
"""

# Core classes lifted to top level for convenience
from firn.models.firn import FirnModel, FirnParameters
from firn.solvers.firn_solver import FirnColumnSolver
from firn.mesh import column_mesh
from firn.constants import year

# Submodule namespaces
import firn.physics
import firn.data
import firn.models
import firn.solvers

__all__ = [
    "FirnModel",
    "FirnParameters",
    "FirnColumnSolver",
    "column_mesh",
    "year",
    "physics",
    "data",
    "models",
    "solvers",
]
