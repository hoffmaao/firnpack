"""forward_densification_comparison.py

Demonstration of the pluggable densification rate interface.
Runs the same transient forward model at South Pole conditions with
different densification laws (Arthern/Ligtenberg vs Herron & Langway)
and plots the resulting density, age, and velocity profiles.

Usage:
    OMP_NUM_THREADS=1 python forward_densification_comparison.py
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import firedrake as fd

from firn.models.firn import FirnParameters
from firn.constants import year as YEAR_S
from firn.physics.densification import arthern_ligtenberg, herron_langway

# =============================================================================
# Paths
# =============================================================================
_HERE = Path(__file__).parent
_SOUTHPOLE = _HERE.parent
PROCESSED_DIR = _SOUTHPOLE / "processed"
BUIZERT_TEMP_CSV  = PROCESSED_DIR / "buizert2021_spice_temp.csv"
BUIZERT_ACCUM_CSV = PROCESSED_DIR / "buizert2021_spice_accum.csv"
SP19_DENS_CSV  = PROCESSED_DIR / "sp19_density.csv"
SP19_AGE_CSV   = PROCESSED_DIR / "sp19_depth_age.csv"
OUT_DIR = _SOUTHPOLE / "results"

# =============================================================================
# Settings
# =============================================================================
H0 = 130.0
NZ = 100
STRETCH_P = 2.5
SURFACE_ID = 2
SPINUP_START_CE = 740
SPINUP_END_CE = 2015  # use full Buizert record
DT_YEARS = 10.0       # spinup timestep
RHO_SURF = 350.0
R2_SURF = 2.5e-7
RHO_M = 550.0


# =============================================================================
# Helpers
# =============================================================================
def build_stretched_mesh(H0, nz, p):
    mesh = fd.IntervalMesh(nz, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** p)
    coord_fs = mesh.coordinates.function_space()
    mesh.coordinates.assign(fd.Function(coord_fs).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    return mesh, V


def make_real(R, val, name):
    f = fd.Function(R, name=name)
    f.assign(float(val))
    return f


# =============================================================================
# Simple transient forward model using pluggable densification rate
# =============================================================================
def run_forward(rate_fn, label, spin_T, spin_A, dt_yr=DT_YEARS):
    """Run a transient forward model with the given densification rate function.

    rate_fn : callable (rho, T, *, params, bdot, ...) -> UFL expression
    Returns depth, rho, age_yr, T_C profiles.
    """
    t0 = time.time()
    print(f"\n--- Forward run: {label} ---")

    mesh, V = build_stretched_mesh(H0, NZ, STRETCH_P)
    R = fd.FunctionSpace(mesh, "R", 0)

    params = FirnParameters()

    # State fields (CG1)
    H = fd.Function(V, name="H")
    rho = fd.Function(V, name="rho")
    w = fd.Function(V, name="w")
    age = fd.Function(V, name="age")

    T0 = float(spin_T[0])
    A0 = float(spin_A[0])
    xi = fd.SpatialCoordinate(mesh)[0]
    depth = H0 - xi

    # Initial condition (exponential density profile)
    H.assign(float(params.c_i) * (T0 - float(params.T_ref)))
    rho.interpolate(
        RHO_SURF + (float(params.rho_i) - RHO_SURF) * (1.0 - fd.exp(-depth / 20.0))
    )
    w.assign(-A0 * float(params.rho_i) / RHO_SURF / float(YEAR_S))
    age.assign(0.0)

    # Boundary value functions
    Ts_fn    = make_real(R, T0, "Ts")
    Hs_fn    = make_real(R, float(params.c_i) * (T0 - float(params.T_ref)), "Hs")
    accum_fn = make_real(R, A0, "accum")
    rho_s_fn = make_real(R, RHO_SURF, "rho_s")
    w_surf_fn = make_real(R, -A0 * float(params.rho_i) / RHO_SURF / float(YEAR_S), "ws")

    bc_H = fd.DirichletBC(V, Hs_fn, SURFACE_ID)
    bc_rho = fd.DirichletBC(V, rho_s_fn, SURFACE_ID)
    bc_w = fd.DirichletBC(V, w_surf_fn, SURFACE_ID)
    bc_age = fd.DirichletBC(V, make_real(R, 0.0, "a0"), SURFACE_ID)

    # Time-stepping loop
    dt_s = float(dt_yr * YEAR_S)
    dt_fn = make_real(R, dt_s, "dt")
    theta_rho = fd.Constant(params.theta_rho)
    theta_age = fd.Constant(0.5)

    rho_old = fd.Function(V, name="rho_old")
    age_old = fd.Function(V, name="age_old")
    H_old   = fd.Function(V, name="H_old")
    w_old   = fd.Function(V, name="w_old")

    n_steps = len(spin_T)
    for step in range(n_steps):
        Tk = float(spin_T[step])
        Ak = float(spin_A[step])
        Ts_fn.assign(Tk)
        Hs_fn.assign(float(params.c_i) * (Tk - float(params.T_ref)))
        accum_fn.assign(Ak)
        w_surf_fn.assign(-Ak * float(params.rho_i) / RHO_SURF / float(YEAR_S))

        rho_old.assign(rho)
        age_old.assign(age)
        H_old.assign(H)
        w_old.assign(w)

        # === Enthalpy (simple diffusion, uses temperature from previous rho/w) ===
        T_expr = H / float(params.c_i) + float(params.T_ref)
        H_trial = fd.TrialFunction(V)
        psi = fd.TestFunction(V)
        kappa = fd.Constant(float(params.K_cold))
        F_H = (
            (H_trial - H_old) / dt_fn * psi * fd.dx
            + w_old * H_trial.dx(0) * psi * fd.dx
            + kappa * H_trial.dx(0) * psi.dx(0) * fd.dx
        )
        fd.solve(fd.lhs(F_H) == fd.rhs(F_H), H, bcs=[bc_H])

        # === Density via pluggable rate function (advective form) ===
        # Use H&L / Arthern rate computed at rho_old (explicit)
        T = H / float(params.c_i) + float(params.T_ref)
        bdot_kgm2s = accum_fn * float(params.rho_i) / float(params.spy)

        drhodt_old = rate_fn(
            rho_old, T,
            params=params,
            bdot=bdot_kgm2s,
            smooth=True,
        )

        rho_trial = fd.TrialFunction(V)
        rho_mid = theta_rho * rho_trial + (1.0 - theta_rho) * rho_old
        F_rho = (
            (rho_trial - rho_old) / dt_fn * psi * fd.dx
            + w_old * rho_mid.dx(0) * psi * fd.dx
            - drhodt_old * psi * fd.dx
        )
        fd.solve(fd.lhs(F_rho) == fd.rhs(F_rho), rho, bcs=[bc_rho])

        # === Velocity from continuity: rho * dw/dz + drhodt = 0 ===
        drhodt_new = rate_fn(
            rho, T,
            params=params,
            bdot=bdot_kgm2s,
            smooth=True,
        )
        w_trial = fd.TrialFunction(V)
        F_w = (
            rho * w_trial.dx(0) * psi * fd.dx
            + drhodt_new * psi * fd.dx
            + fd.Constant(1.0e-3) * w_trial * psi * fd.dx  # tiny regularization
        )
        fd.solve(fd.lhs(F_w) == fd.rhs(F_w), w, bcs=[bc_w])

        # === Age (simple advection + unit source) ===
        a_trial = fd.TrialFunction(V)
        a_mid = theta_age * a_trial + (1.0 - theta_age) * age_old
        F_age = (
            (a_trial - age_old) / dt_fn * psi * fd.dx
            + w * a_mid.dx(0) * psi * fd.dx
            - fd.Constant(1.0) * psi * fd.dx
        )
        fd.solve(fd.lhs(F_age) == fd.rhs(F_age), age, bcs=[bc_age])

    t_total = time.time() - t0
    print(f"  Finished {n_steps} steps in {t_total:.0f}s")
    print(f"  rho = [{rho.dat.data_ro.min():.1f}, {rho.dat.data_ro.max():.1f}] kg/m³")
    print(f"  age = [{age.dat.data_ro.min()/YEAR_S:.0f}, {age.dat.data_ro.max()/YEAR_S:.0f}] yr")

    mesh_x = (mesh.coordinates.dat.data_ro[:, 0]
              if mesh.coordinates.dat.data_ro.ndim == 2
              else mesh.coordinates.dat.data_ro)
    depth_nodes = H0 - mesh_x
    idx = np.argsort(depth_nodes)

    return {
        "label": label,
        "depth": depth_nodes[idx],
        "rho": rho.dat.data_ro[idx].copy(),
        "age_yr": age.dat.data_ro[idx].copy() / float(YEAR_S),
        "w_m_yr": w.dat.data_ro[idx].copy() * float(YEAR_S),
        "T_C": H.dat.data_ro[idx] / float(params.c_i) + float(params.T_ref) - 273.15,
        "time_s": t_total,
    }


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Buizert 2021 forcing ...")
    buiz_T_df = pd.read_csv(BUIZERT_TEMP_CSV).sort_values("year_CE")
    buiz_A_df = pd.read_csv(BUIZERT_ACCUM_CSV).sort_values("year_CE")
    years_1yr = np.arange(float(SPINUP_START_CE), float(SPINUP_END_CE), 1.0)
    T_1yr = np.interp(years_1yr, buiz_T_df["year_CE"].values,
                      buiz_T_df["temp"].values + 273.15)
    A_1yr = np.interp(years_1yr, buiz_A_df["year_CE"].values,
                      buiz_A_df["accum"].values)

    # Block-average to DT_YEARS resolution
    blk = int(round(DT_YEARS))
    n_blk = len(T_1yr) // blk
    spin_T = np.array([T_1yr[i*blk:(i+1)*blk].mean() for i in range(n_blk)])
    spin_A = np.array([A_1yr[i*blk:(i+1)*blk].mean() for i in range(n_blk)])
    print(f"  Forcing: {SPINUP_START_CE}-{SPINUP_END_CE} CE, "
          f"{len(spin_T)} steps × dt={DT_YEARS}yr")
    print(f"  T range: {spin_T.min()-273.15:.1f} to {spin_T.max()-273.15:.1f}°C")
    print(f"  A range: {spin_A.min():.4f} to {spin_A.max():.4f} m ice/yr")

    # Load SP19 observations for comparison
    df_rho = pd.read_csv(SP19_DENS_CSV)
    df_rho = df_rho[df_rho["depth_m"] <= H0]
    obs_rho_depth = df_rho["depth_m"].values
    obs_rho = df_rho["rho_kgm3"].values * 1000.0  # g/cm³ → kg/m³

    df_age = pd.read_csv(SP19_AGE_CSV)
    df_age = df_age[(df_age["year_CE"] <= 2015) & (df_age["depth_m"] <= H0)]
    obs_age_depth = df_age["depth_m"].values
    obs_age = 2015.0 - df_age["year_CE"].values

    # Run both models with identical forcing and IC
    results = []
    for label, rate_fn in [
        ("Arthern/Ligtenberg", arthern_ligtenberg),
        ("Herron & Langway",   herron_langway),
    ]:
        try:
            results.append(run_forward(rate_fn, label, spin_T, spin_A))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 7))
    colors = ["C0", "C1"]

    axes[0].plot(obs_rho, obs_rho_depth, "k.", ms=2, alpha=0.4, label="SP19")
    for res, c in zip(results, colors):
        axes[0].plot(res["rho"], res["depth"], "-", color=c, lw=1.5, label=res["label"])
    axes[0].set_xlabel("Density (kg/m³)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()
    axes[0].set_title("Density")
    axes[0].axvline(RHO_M, color="gray", ls="--", lw=0.5)
    axes[0].legend(fontsize=8)

    axes[1].plot(obs_age, obs_age_depth, "k.", ms=2, alpha=0.4, label="SP19")
    for res, c in zip(results, colors):
        axes[1].plot(res["age_yr"], res["depth"], "-", color=c, lw=1.5, label=res["label"])
    axes[1].set_xlabel("Age (yr)")
    axes[1].invert_yaxis()
    axes[1].set_title("Age")
    axes[1].legend(fontsize=8)

    for res, c in zip(results, colors):
        axes[2].plot(res["w_m_yr"], res["depth"], "-", color=c, lw=1.5, label=res["label"])
    axes[2].set_xlabel("Velocity (m/yr)")
    axes[2].invert_yaxis()
    axes[2].set_title("Vertical velocity")
    axes[2].legend(fontsize=8)

    for res, c in zip(results, colors):
        axes[3].plot(res["T_C"], res["depth"], "-", color=c, lw=1.5, label=res["label"])
    axes[3].set_xlabel("Temperature (°C)")
    axes[3].invert_yaxis()
    axes[3].set_title("Temperature")
    axes[3].legend(fontsize=8)

    fig.suptitle(
        f"South Pole: pluggable densification — {SPINUP_START_CE}-{SPINUP_END_CE} CE, "
        f"dt={DT_YEARS:.0f}yr"
    )
    fig.tight_layout()
    out_path = OUT_DIR / "forward_densification_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

    # Timing summary
    print("\nTiming:")
    for res in results:
        print(f"  {res['label']:25s}  {res['time_s']:.0f}s  "
              f"rho_max={res['rho'].max():.0f}  age_max={res['age_yr'].max():.0f}yr")

    print("\nDone.")
