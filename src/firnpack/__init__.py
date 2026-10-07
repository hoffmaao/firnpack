"""Firnpack: A 1D firn column model with pluggable physics and adjoint support.

The user-facing API follows the icepack/hydropack pattern:

    import firnpack

    model = firnpack.FirnModel(densification=firnpack.physics.herron_langway)
    mesh  = firnpack.column_mesh(height=130, n=100, stretch=2.5)
    ...

Submodules:
    firnpack.physics   -- Pluggable rate laws (densification, grain growth)
    firnpack.data      -- Climate histories, observations, boundary data
    firnpack.models    -- FirnModel (PDE weak forms)
    firnpack.solvers   -- FirnColumnSolver (time-stepping, BCs, mesh motion)
    firnpack.inverse   -- Config-driven adjoint assimilation (SiteConfig, assimilate)
    firnpack.mesh      -- Column mesh construction
    firnpack.constants -- Physical constants in SI units
"""

# Core classes lifted to top level for convenience
from firnpack.models.firn import FirnModel, FirnParameters
from firnpack.solvers.firn_solver import FirnColumnSolver
from firnpack.mesh import column_mesh
from firnpack.constants import year

# Submodule namespaces
import firnpack.physics
import firnpack.data
import firnpack.models
import firnpack.solvers

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
