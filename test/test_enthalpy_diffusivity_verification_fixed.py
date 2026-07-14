"""test_enthalpy_diffusivity_verification.py

Small verification harness for the firn enthalpy/temperature solver.

Goal
----
1) Verify that enthalpy_from_T and temperature_from_enthalpy are consistent.
2) Verify that the enthalpy weak form in FirnModel.enthalpy_form gives the expected
   diffusion behaviour for a simple 1D column when w=0 (pure diffusion).
3) Demonstrate how to force a constant thermal diffusivity κ (m^2/s) via the same
   monkey-patch pattern used in the adjoint scripts.

This test is *not* tied to ApRES/ERA5 – it is purely a physics/unit check.

Run
---
python test_enthalpy_diffusivity_verification.py
"""

from __future__ import annotations

import numpy as np
import firedrake as fd

from firnpack.models.firn import FirnModel, FirnParameters


def enthalpy_from_T(params: FirnParameters, T_K: float) -> float:
    """H = c_i (T - T_ref) (units: J/kg)."""
    return float(params.c_i) * (float(T_K) - float(params.T_ref))


def apply_constant_thermal_diffusivity(model: FirnModel, kappa_const: fd.Function) -> None:
    """Override model.thermal_diffusivity(H, rho) to return constant κ (m^2/s)."""

    def _kappa(_H, _rho):
        return kappa_const

    model.thermal_diffusivity = _kappa


def main():
    # -------------------------------------------------------------------------
    # Basic parameter sanity
    # -------------------------------------------------------------------------
    p = FirnParameters()
    model = FirnModel(p)

    T_test = 250.0
    H_test = enthalpy_from_T(p, T_test)
    T_roundtrip = float(model.temperature_from_enthalpy(H_test))
    print("Roundtrip check:")
    print(f"  T_in  = {T_test:.3f} K")
    print(f"  H     = {H_test:.3f} J/kg")
    print(f"  T_out = {T_roundtrip:.3f} K  (should match T_in)\n")

    # -------------------------------------------------------------------------
    # Diffusion-only enthalpy solve on a simple column
    # -------------------------------------------------------------------------
    H0 = 100.0  # m
    nz = 200
    mesh = fd.IntervalMesh(nz, 0.0, H0)
    V = fd.FunctionSpace(mesh, "CG", 1)

    # No advection
    w = fd.Function(V, name="w").assign(0.0)

    # Fixed density just to provide rho to thermal_diffusivity (not used if κ const)
    rho = fd.Function(V, name="rho").assign(500.0)
    rho_old = fd.Function(V, name="rho_old").assign(500.0)

    # Boundary temperatures (Dirichlet top+bottom so steady-state is linear)
    Ts_top = 250.0
    Ts_bot = 260.0
    H_top = enthalpy_from_T(p, Ts_top)
    H_bot = enthalpy_from_T(p, Ts_bot)

    # Constant κ (roughly ice diffusivity order ~1e-6 m^2/s)
    kappa_val = 1.0e-6
    R = fd.FunctionSpace(mesh, "R", 0)
    kappa_const = fd.Function(R, name="kappa").assign(kappa_val)
    apply_constant_thermal_diffusivity(model, kappa_const)

    # Initial condition: exact steady-state linear enthalpy profile
    x = fd.SpatialCoordinate(mesh)[0]
    H = fd.Function(V, name="H")
    H_old = fd.Function(V, name="H_old")
    H.interpolate(H_bot + (H_top - H_bot) * (x / H0))
    H_old.assign(H)

    # Time step (seconds)
    dt = fd.Constant(30.0 * 86400.0)  # 30 days

    psi = fd.TestFunction(V)
    F_H = model.enthalpy_form(H, H_old, rho, w, psi, dt)
    J_H = fd.derivative(F_H, H)

    bcs = [
        fd.DirichletBC(V, fd.Constant(H_bot), 1),
        fd.DirichletBC(V, fd.Constant(H_top), 2),
    ]

    prob = fd.NonlinearVariationalProblem(F_H, H, bcs=bcs, J=J_H)
    solver = fd.NonlinearVariationalSolver(
        prob,
        solver_parameters={
            "snes_type": "newtonls",
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    # One step should preserve the steady-state solution (within solver tolerance)
    solver.solve()
    err_inf = float(fd.norm(H - H_old, norm_type="linf"))
    print("Diffusion steady-state check (w=0, κ const, linear IC):")
    print(f"  ||H - H_old||_inf = {err_inf:.3e}  (should be ~0)\n")

    # Now perturb and show decay of the perturbation
    H_old.assign(H)
    H.interpolate(H + 0.5 * (H_top - H_bot) * fd.sin(np.pi * x / H0))
    H_old.assign(H)

    nsteps = 24  # 2 years at 30-day steps
    amp = []
    for _ in range(nsteps):
        solver.solve()
        # measure amplitude of deviation from the linear baseline (rough proxy)
        T = model.temperature_from_enthalpy(H)
        amp.append(float(fd.norm(T - fd.Function(V).interpolate(Ts_bot + (Ts_top - Ts_bot) * (x / H0)), norm_type="L2")))
        H_old.assign(H)

    print("Perturbation decay (L2 norm of T deviation from linear baseline):")
    print(f"  first = {amp[0]:.3e}, last = {amp[-1]:.3e} (should decrease)\n")

    print("Done.")


if __name__ == "__main__":
    main()
