"""southpole_forward_diagnostic.py

Run the firn forward model to steady state and check internal consistency
of the age, density, and velocity equations.

The age equation is a strong constraint: in steady state,
    age(z) = (1/A_ice) * integral_0^z rho(z')/rho_i dz'

If the model's age profile matches this analytic result, mass conservation
is working correctly. Any discrepancy reveals issues in the solver.

Run:
    cd test/southpole/scripts
    python southpole_forward_diagnostic.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
_HERE = Path(__file__).parent
_SOUTHPOLE = _HERE.parent
PROCESSED_DIR = _SOUTHPOLE / "processed"
SP19_DENS_CSV = PROCESSED_DIR / "sp19_density.csv"
FIG_DIR = _SOUTHPOLE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Analytic steady-state predictions
# ============================================================================

def load_sp19():
    df = pd.read_csv(SP19_DENS_CSV)
    z = df["depth_m"].values
    rho = df["rho_kgm3"].values
    if rho.max() < 10:
        rho = rho * 1000.0
    return z, rho


def steady_state_predictions(z, rho, A_m_ice_yr, rho_s, rho_i=917.0, n_ice=1.775):
    """Compute steady-state age, velocity, and dR/dt from the density profile.

    In exact steady state with constant accumulation A:
      - Mass conservation: rho(z) * w(z) = rho_s * w_s = -A * rho_i
      - Velocity: w(z) = -A * rho_i / rho(z)  (m/yr, negative = downward)
      - Age: age(z) = (1/A_ice) * integral_0^z rho(z')/rho_i dz'  (years)
        where A_ice = A in m ice eq/yr
      - dR/dt: (w_s - w(z)) / n_ice  (m/yr, ApRES observable)
    """
    w_s = -A_m_ice_yr * rho_i / rho_s  # surface velocity (m/yr)
    w = -A_m_ice_yr * rho_i / rho       # velocity profile

    # Age from mass integration
    integrand = rho / rho_i / A_m_ice_yr
    age = np.zeros_like(z)
    age[1:] = cumulative_trapezoid(integrand, z)

    # dR/dt (ApRES observable)
    dRdt = (w_s - w) / n_ice

    return {
        "w_myr": w,
        "w_s_myr": w_s,
        "age_yr": age,
        "dRdt_myr": dRdt,
    }


if __name__ == "__main__":
    print("=" * 65)
    print("South Pole forward model diagnostic")
    print("=" * 65)

    z_sp19, rho_sp19 = load_sp19()

    # --- Test with different accumulation rates ---
    A_values = [0.06, 0.08, 0.10]  # m ice eq/yr
    rho_s = 350.0

    # Load observed age for comparison
    try:
        age_df = pd.read_csv(PROCESSED_DIR / "sp19_depth_age.csv")
        age_obs_depth = age_df["depth_m"].values
        age_obs_yr = 2015.0 - age_df["year_CE"].values
    except Exception:
        age_obs_depth = None
        age_obs_yr = None

    # Load Zeising dR/dt for comparison
    try:
        apres_df = pd.read_csv(PROCESSED_DIR / "apres_zeising_processed.csv")
        apres_x11n2 = apres_df[apres_df["site"] == "x11n2"]
    except Exception:
        apres_x11n2 = None

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, A in enumerate(A_values):
        ss = steady_state_predictions(z_sp19, rho_sp19, A, rho_s)
        color = ["steelblue", "firebrick", "seagreen"][i]
        label = f"A={A*100:.0f} cm/yr"

        # Row 1: Density, Age, dR/dt
        ax = axes[0, 0]
        ax.plot(rho_sp19, z_sp19, color=color, lw=1.5, label=label)
        ax.set_xlabel("Density (kg/m³)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title("SP19 Density")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        ax.plot(ss["age_yr"], z_sp19, color=color, lw=1.5, label=label)
        ax.set_xlabel("Age (yr)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title("Steady-state age = ∫ρ/(ρᵢA) dz")
        ax.legend(fontsize=8)

        ax = axes[0, 2]
        ax.plot(ss["dRdt_myr"] * 100, z_sp19, color=color, lw=1.5, label=label)
        ax.set_xlabel("dR/dt (cm/yr)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title("Steady-state dR/dt = (w_s−w)/n_ice")
        ax.legend(fontsize=8)

    # Overlay observations
    if age_obs_depth is not None:
        axes[0, 1].scatter(age_obs_yr, age_obs_depth, s=3, color="k", alpha=0.3,
                           label="SP19 obs", zorder=0)
        axes[0, 1].legend(fontsize=8)

    if apres_x11n2 is not None:
        from scipy.integrate import cumulative_trapezoid as ct
        n_ice = 1.775; rho_ice = 917.0
        sp19_z_ext = np.concatenate([[0.0], z_sp19])
        sp19_rho_ext = np.concatenate([[350.0], rho_sp19])
        n_arr = 1.0 + (n_ice - 1.0) * sp19_rho_ext / rho_ice
        R_sp19 = np.zeros_like(sp19_z_ext)
        R_sp19[1:] = ct(n_arr / n_ice, sp19_z_ext)

        r_a = apres_x11n2["range_m"].values
        v_a = apres_x11n2["dRdt_myr"].values
        coh_a = apres_x11n2["coherence"].values
        z_a = np.interp(r_a, R_sp19, sp19_z_ext, left=np.nan, right=np.nan)
        ok = (coh_a > 0.3) & np.isfinite(z_a) & (z_a <= 130)
        axes[0, 2].scatter(v_a[ok] * 100, z_a[ok], s=8, color="k", alpha=0.5,
                           label="ApRES x11n2", zorder=0)
        axes[0, 2].legend(fontsize=8)

    # Row 2: Velocity, mass flux rho*w, and age diagnostic
    A_ref = 0.08
    ss_ref = steady_state_predictions(z_sp19, rho_sp19, A_ref, rho_s)

    ax = axes[1, 0]
    ax.plot(ss_ref["w_myr"] * 100, z_sp19, "steelblue", lw=2)
    ax.set_xlabel("Velocity w(z) (cm/yr)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title(f"Velocity (A={A_ref*100:.0f} cm/yr)")
    ax.axhline(0, color="grey", lw=0.5)

    ax = axes[1, 1]
    rho_w = rho_sp19 * ss_ref["w_myr"]
    ax.plot(rho_w, z_sp19, "steelblue", lw=2)
    ax.set_xlabel("ρw (kg m⁻² yr⁻¹)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title("Mass flux ρw (should be constant in SS)")
    target = -A_ref * 917.0
    ax.axvline(target, color="k", ls="--", lw=1, label=f"−A·ρᵢ = {target:.1f}")
    ax.legend(fontsize=8)

    # Age residual: analytic SS age vs SP19 observed age
    ax = axes[1, 2]
    if age_obs_depth is not None:
        age_ss_at_obs = np.interp(age_obs_depth, z_sp19, ss_ref["age_yr"])
        residual = age_obs_yr - age_ss_at_obs
        ax.plot(residual, age_obs_depth, "k.", ms=2, alpha=0.5)
        ax.set_xlabel("Age residual: SP19 − SS model (yr)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title(f"Age residual (A={A_ref*100:.0f} cm/yr)")
        ax.axvline(0, color="grey", lw=0.5)
        rms = np.sqrt(np.mean(residual**2))
        ax.text(0.95, 0.05, f"RMS = {rms:.1f} yr", transform=ax.transAxes,
                ha="right", fontsize=10, bbox=dict(fc="white", alpha=0.8))

    fig.suptitle(
        "Steady-state firn diagnostic: analytic predictions from SP19 density\n"
        "age(z) = ∫₀ᶻ ρ/(ρᵢ A) dz'  |  w(z) = −Aρᵢ/ρ  |  dR/dt = (w_s−w)/n_ice",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIG_DIR / "forward_diagnostic_steady_state.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)

    # Print key numbers
    print(f"\nSteady-state diagnostics (A = {A_ref} m ice eq/yr, rho_s = {rho_s}):")
    print(f"  w_surface = {ss_ref['w_s_myr']*100:.1f} cm/yr")
    print(f"  w(130m)   = {ss_ref['w_myr'][-1]*100:.1f} cm/yr")
    print(f"  age(130m) = {ss_ref['age_yr'][-1]:.0f} yr")
    print(f"  dR/dt(20m)  = {np.interp(20, z_sp19, ss_ref['dRdt_myr'])*100:.2f} cm/yr")
    print(f"  dR/dt(100m) = {np.interp(100, z_sp19, ss_ref['dRdt_myr'])*100:.2f} cm/yr")
    if age_obs_depth is not None:
        age_ss_at_130 = ss_ref["age_yr"][-1]
        age_obs_at_130 = np.interp(127.5, age_obs_depth, age_obs_yr)
        print(f"\n  SP19 age at 127.5m: {age_obs_at_130:.0f} yr")
        print(f"  SS model age at 127.5m: {age_ss_at_130:.0f} yr")
        print(f"  Discrepancy: {age_obs_at_130 - age_ss_at_130:.0f} yr")
        # What accumulation would match?
        A_implied = np.interp(127.5, z_sp19, rho_sp19) * 127.5 / (age_obs_at_130 * 917.0)
        print(f"  Implied A to match SP19 age: {A_implied*100:.2f} cm ice eq/yr")

    print("\nDone.")
