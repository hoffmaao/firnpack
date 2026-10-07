"""test_kingslake_full_density_verification.py

PARKED, not a test: this file asserts nothing, and in its current form it
cannot run. The ``main()`` pipeline sits behind a ``__main__`` guard, so
importing or collecting this module runs nothing.

It is not converted because its parameter names never matched the model. The
``FirnParameters(...)`` call below passes ``firn_alpha``, ``dt``, ``rho_s``,
``kcHh``, ``kcLw``, ``K`` and ``grain``; none of them are fields of
``FirnParameters`` (see ``src/firnpack/models/firn.py``), so construction
raises ``TypeError`` immediately. ``firn_alpha`` appears exactly once in the
whole repo -- on that line, since the initial commit -- so there is no prior
art to recover the intent from. Repairing this needs a human decision on what
``firn_alpha`` and ``K`` were meant to be; in a file whose whole purpose is
catching sign and factor bugs, a wrong guess is worse than no test at all.

What it is meant to do
----------------------
1) Runs a 1D firn column to (quasi) steady state under constant forcing
   (isothermal Ts, constant accumulation b), using the FULL-DENSITY solver
   with prognostic (rho, sigma, r^2).
2) Runs the reduced ODE model (Eq. 24 / Eq. 25) in Kingslake et al. (2022)
   for the same nondimensional parameters.
3) Plots and saves comparison figures (porosity and vertical velocity).

Why it's useful
---------------
If the Firedrake column cannot reproduce the reduced ODE behaviour in the limit
of constant grain size, the issue is very likely in one of:
  * drho/dt constitutive form (missing (1-phi) factor -> spurious 1/rho
    amplification near the surface)
  * stress sign convention / use of |sigma|
  * depth coordinate orientation / sign of w

Run
---
Not runnable as-is; fix the parameter names above first. Once repaired it is
intended to be run directly, writing PNG figures to ./kingslake_verify_outputs/:

    PYTHONPATH=src OMP_NUM_THREADS=1 python test/test_kingslake_full_density_verification.py

You can tweak beta_list, dt_years, and spinup_years at the bottom.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

try:
    import firedrake as fd
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires Firedrake. Activate your firedrake env and rerun."
    ) from exc

# Local patched modules (drop these into your repo, or adjust imports to your package)
from firnpack.models.firn import FirnModel, FirnParameters
from firnpack.solvers.firn_solver import FirnColumnSolver


YEAR_S = 365.25 * 24 * 3600


def solve_reduced_ode_eq24(
    alpha: float,
    beta: float,
    rs2: float,
    phi_s: float,
    n: float = 1.0,
    m: float = 1.0,
    npts: int = 2001,
):
    """Solve Kingslake Eq. (24) with sigma=-z and constant r^2=rs2.

    Returns arrays (z, phi, w) on z in [0,1].
    """

    z = np.linspace(0.0, 1.0, npts)
    dz = z[1] - z[0]

    phi = np.empty_like(z)
    w = np.empty_like(z)

    phi[0] = phi_s
    w[0] = beta / (1.0 - phi_s)

    for k in range(npts - 1):
        zk = z[k]
        # Forward Euler is fine for smooth verification curves.
        dphi_dz = -(zk**n) * (phi[k] ** m) * (1.0 - phi[k]) / (alpha * w[k] * rs2)
        dw_dz = -(zk**n) * (phi[k] ** m) / (alpha * rs2)

        phi[k + 1] = np.clip(phi[k] + dz * dphi_dz, 0.0, 1.0)
        w[k + 1] = w[k] + dz * dw_dz

    return z, phi, w


def run_firedrake_column(
    beta: float,
    *,
    outdir: str,
    H0_m: float = 100.0,
    nz: int = 200,
    spinup_years: float = 1200.0,
    dt_years: float = 1.0,
    Ts_K: float = 253.15,
    b0_m_per_yr: float = 0.1,
    phi_s: float = 0.5,
    rs2_nd: float = 0.029,
    alpha: float = 0.082,
    kg_m2_s: float = 0.0,
    use_r2_saturation: bool = False,
):
    """Run the Firedrake full-density model under constant forcing."""

    # ------------------------------------------------------------------
    # Mesh (x increases upward: 0=bed, H0=surface)
    # ------------------------------------------------------------------
    mesh = fd.IntervalMesh(nz, 0.0, H0_m)
    V = fd.FunctionSpace(mesh, "CG", 1)

    # ------------------------------------------------------------------
    # Parameters: set up a Kingslake-intermediate-like case.
    # We use a single kc (via rhoCoef field) so kc0/kc1 aren't used.
    # ------------------------------------------------------------------
    rho_i = 917.0
    rho_s = rho_i * (1.0 - phi_s)

    # Convert nondim surface grain size rs2_nd to dimensional r_s^2 using r0^2.
    # From paper Eq. (16): r0^2 = ka*z0*exp(-Eg/RTs)/b0.
    # We keep the exact numerical value used in the paper (Table 3, intermediate):
    r0_sq = 8.8e-6  # [m^2]
    r_s_sq = rs2_nd * r0_sq

    # Compute dimensional kc from alpha definition (paper Eq. 13):
    #   alpha = r0^2 / (kc * t0 * sigma0^n * exp(-Ec/RTs))
    # where t0=z0/b0 and sigma0=rho_i g z0.
    g = 9.81
    sigma0 = rho_i * g * H0_m
    t0_s = (H0_m / (b0_m_per_yr / YEAR_S))
    Ec = 60e3
    Rgas = 8.314
    kc = r0_sq / (alpha * t0_s * (sigma0**1.0) * np.exp(-Ec / (Rgas * Ts_K)))

    # Base parameter object; we set full_density=True and provide defaults.
    params = FirnParameters(
        firn_alpha=0.0,
        dt=dt_years * YEAR_S,
        rho_i=rho_i,
        rho_s=rho_s,
        kcHh=kc,
        kcLw=kc,
        rho_m=rho_i,  # unused since we pass rhoCoef explicitly
        Ec=Ec,
        g=g,
        R=Rgas,
        spy=YEAR_S,
        theta_liniger=1.0,
        K=0.0,  # isothermal
        c_i=2009.0,
        T_ref=273.15,
        grain=True,
        ka=kg_m2_s,
        Eg=42.4e3,
        r2_f=1e-2,
        use_r2_saturation=use_r2_saturation,
        eps_r2=1e-12,
        full_density=True,
        # Kingslake-style exponents
        n=1.0,
        m=1.0,
        phi_min=1e-8,
        sigma_min=0.0,
    )

    model = FirnModel(params)
    solver = FirnColumnSolver(model)

    # Controls / coefficients
    accumulation = fd.Constant(beta * b0_m_per_yr)
    dt = fd.Constant(dt_years * YEAR_S)

    Ts = fd.Constant(Ts_K)

    # Fields
    H = fd.Function(V, name="enthalpy")
    rho = fd.Function(V, name="rho")
    w = fd.Function(V, name="w")
    sigma = fd.Function(V, name="sigma")
    r2 = fd.Function(V, name="r2")
    age = fd.Function(V, name="age")

    # Init: uniform fields
    H.assign(params.c_i * (Ts_K - params.T_ref))
    rho.assign(rho_s)
    w.assign(-float(accumulation) * rho_i / rho_s / YEAR_S)
    sigma.assign(0.0)
    r2.assign(r_s_sq)
    age.assign(0.0)

    # rhoCoef field (kc). We keep it constant for verification.
    rhoCoef = fd.Function(V, name="rhoCoef")
    rhoCoef.assign(kc)

    # Boundary conditions
    bc_H = fd.DirichletBC(V, H, 2)
    bc_rho = fd.DirichletBC(V, rho_s, 2)
    bc_w = fd.DirichletBC(V, w, 2)
    bc_sigma = fd.DirichletBC(V, 0.0, 2)
    bc_r2 = fd.DirichletBC(V, r_s_sq, 2)
    bc_age = fd.DirichletBC(V, 0.0, 2)

    bcs = [bc_H, bc_rho, bc_w, bc_sigma, bc_r2, bc_age]

    nsteps = int(spinup_years / dt_years)

    for step in range(nsteps):
        solver.prognostic_solve(
            H,
            rho,
            w,
            dt,
            accumulation,
            surface_density=rho_s,
            boundary_conditions=bcs,
            surface_temperature=Ts,
            enthalpy_bc_constant=True,
            stress=sigma,
            grain_radius2=r2,
            rhoCoef=rhoCoef,
            age=age,
            verbose=False,
        )

    # Extract profiles at vertices (CG1 -> dofs on vertices)
    x = np.asarray(mesh.coordinates.dat.data_ro).reshape(-1)
    depth = H0_m - x

    # Sort shallow->deep
    idx = np.argsort(depth)
    depth = depth[idx]
    rho_arr = rho.dat.data_ro[idx]
    w_arr = w.dat.data_ro[idx]

    phi_arr = 1.0 - rho_arr / rho_i

    # Convert w to nondimensional (divide by b0)
    w_down = -w_arr * YEAR_S  # m/yr
    w_nd = w_down / b0_m_per_yr

    # Convert depth to nondim z
    z_nd = depth / H0_m

    return {
        "kc": kc,
        "r_s_sq": r_s_sq,
        "r0_sq": r0_sq,
        "z_nd": z_nd,
        "phi": phi_arr,
        "w_nd": w_nd,
    }


def main():
    outdir = "kingslake_verify_outputs"
    os.makedirs(outdir, exist_ok=True)

    alpha = 0.082
    rs2 = 0.029
    phi_s = 0.5

    beta_list = [0.5, 1.0, 2.0, 5.0]

    for beta in beta_list:
        print(f"Running beta={beta} ...")
        fd_res = run_firedrake_column(
            beta,
            outdir=outdir,
            alpha=alpha,
            rs2_nd=rs2,
            phi_s=phi_s,
            spinup_years=1200.0,
            dt_years=1.0,
        )

        z, phi_ode, w_ode = solve_reduced_ode_eq24(alpha, beta, rs2, phi_s)

        # Plot porosity
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.plot(phi_ode, z, label="ODE Eq(24)")
        ax.plot(fd_res["phi"], fd_res["z_nd"], label="Firedrake")
        ax.invert_yaxis()
        ax.set_xlabel("Porosity φ")
        ax.set_ylabel("z / z0")
        ax.set_title(f"Kingslake verification: φ(z), β={beta}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"phi_beta{beta:.2f}.png"), dpi=200)
        plt.close(fig)

        # Plot velocity
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.plot(w_ode, z, label="ODE Eq(24)")
        ax.plot(fd_res["w_nd"], fd_res["z_nd"], label="Firedrake")
        ax.invert_yaxis()
        ax.set_xlabel("w / b0")
        ax.set_ylabel("z / z0")
        ax.set_title(f"Kingslake verification: w(z), β={beta}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"w_beta{beta:.2f}.png"), dpi=200)
        plt.close(fig)

    print(f"Done. Figures in {outdir}/")


if __name__ == "__main__":
    main()
