"""Mesh utilities for 1D firn column models.

Provides convenience functions for creating interval meshes with
optional power-law stretching, and ALE (Arbitrary Lagrangian-Eulerian)
mesh motion that conserves mass as firn compacts.
"""
import firedrake as fd


def column_mesh(height: float, n: int, stretch: float = 1.0):
    """Create a 1D column mesh for firn modeling.

    Parameters
    ----------
    height : float
        Column height in meters (x=0 at base, x=height at surface).
    n : int
        Number of elements.
    stretch : float, optional
        Power-law stretching exponent. Values > 1 concentrate elements
        near the surface (x=height). Default 1.0 (uniform).

    Returns
    -------
    mesh : firedrake.Mesh
        The interval mesh with physical coordinates.
    """
    mesh = fd.IntervalMesh(n, 0.0, 1.0)
    if stretch != 1.0:
        xi = fd.SpatialCoordinate(mesh)[0]
        x_phys = height * (1.0 - (1.0 - xi) ** stretch)
        coord_fs = mesh.coordinates.function_space()
        mesh.coordinates.assign(
            fd.Function(coord_fs).interpolate(fd.as_vector([x_phys]))
        )
    else:
        # Scale uniform mesh to physical height
        mesh.coordinates.dat.data[:] *= height
    return mesh


class ALEMeshMover:
    """ALE mesh motion for a 1D firn column.

    Moves mesh coordinates so that each cell conserves mass:
        ρ(x_new) dx_new = ρ_ref(x_ref) dx_ref

    The new coordinates are found by solving:
        dX/dξ = (dx_ref/dξ) * (ρ_ref / ρ)

    where ξ is a reference coordinate.  This is a linear variational
    problem that Firedrake can differentiate through via pyadjoint.

    Parameters
    ----------
    mesh : firedrake.Mesh
        The 1D interval mesh to move.
    rho_ref : firedrake.Function
        Reference density field (snapshot at setup time).
    base_id : int
        Boundary ID for the base (x=0). Default 1.
    """

    def __init__(self, mesh, rho_ref, base_id=1):
        self.mesh = mesh
        self.base_id = base_id

        # Store the reference state (coordinates and density at setup)
        self._x_ref = mesh.coordinates.copy(deepcopy=True)
        self._rho_ref = rho_ref.copy(deepcopy=True)

        # Build the solver for the coordinate update ODE:
        #   dX/dξ * φ dξ = (dx_ref/dξ) * (ρ_ref/ρ) * φ dξ
        # with X(base) = 0.
        # Scalar CG1 space for solving the coordinate ODE
        self._V_scalar = fd.FunctionSpace(mesh, "CG", 1)
        self._X_scalar = fd.Function(self._V_scalar, name="X_ale_scalar")
        # Store reference x as a scalar CG1 function
        self._x_ref_scalar = fd.Function(self._V_scalar, name="x_ref_scalar")
        self._x_ref_scalar.dat.data[:] = mesh.coordinates.dat.data_ro.flatten()

    def update(self, rho):
        """Move the mesh coordinates to conserve mass given current density.

        Parameters
        ----------
        rho : firedrake.Function
            Current density field (CG1 on the mesh).
        """
        mesh = self.mesh
        rho_ref = self._rho_ref

        # Density ratio, clamped to prevent extreme distortion
        rho_safe = fd.max_value(rho, fd.Constant(1.0))
        ratio = rho_ref / rho_safe

        # Solve for new coordinates: dX/dx = ρ_ref/ρ, X(base)=0
        # In weak form: ∫ X' φ' dx = ∫ (ρ_ref/ρ) φ' dx
        # (integration by parts of ∫ X'' φ dx with natural BCs at surface)
        V_s = self._V_scalar
        X_trial = fd.TrialFunction(V_s)
        phi = fd.TestFunction(V_s)
        dx_m = fd.dx(domain=mesh)

        a = X_trial.dx(0) * phi.dx(0) * dx_m
        L = ratio * phi.dx(0) * dx_m

        bc_base = fd.DirichletBC(V_s, 0.0, self.base_id)
        fd.solve(a == L, self._X_scalar, bcs=[bc_base])

        # Assign to mesh coordinates (pyadjoint-tracked)
        mesh.coordinates.interpolate(fd.as_vector([self._X_scalar]))
