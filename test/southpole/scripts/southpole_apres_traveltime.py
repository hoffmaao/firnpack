"""southpole_apres_traveltime.py

Compute ApRES observables in travel-time (apparent range) coordinates and
build the forward-model operator that maps firn-model density/velocity fields
to predicted apparent-range change rate dR/dt.

Physics (Elizabeth Case et al. 2020; see also Kingslake et al. 2014):
=======================================================================

ApRES measures changes in two-way travel time, reported as apparent range:

    R(z_R) = integral_0^{z_R}  n(rho(z)) / n_ice  dz

where n(rho) is the CRIM refractive index:

    n(rho) = 1 + (n_ice - 1) * rho / rho_ice

The "ice-equivalent range" divides by n_ice so that R = z for pure ice.

The rate of change of apparent range to a reflector at depth z_R has TWO
contributions (this is the key insight from Case et al.):

    dR/dt(z_R) = n(rho(z_R))/n_ice * w(z_R)
                 + (n_ice - 1)/(n_ice * rho_ice) * integral_0^{z_R} drho/dt(z) dz

Term 1:  Physical layer displacement * local refractive index ratio
Term 2:  Density changes in the overlying column alter the optical path length

For firn, Term 2 is significant and non-negligible: densification above a
reflector shortens the travel time (range decreases) even if the reflector
hasn't moved.  This is the "compaction correction" that must be accounted for
when comparing model predictions with ApRES vertical velocities.

The ApRES "vertical velocity" v_VV(R) is just dR/dt expressed per unit time,
but reported on the apparent-range axis R rather than the depth axis z.

Coordinate convention:
    - ApRES data are on a range-bin axis R (ice-equivalent metres)
    - The firn model is on a depth axis z (true metres from surface)
    - This script provides functions to:
      (a) Convert model depth z -> apparent range R given rho(z)
      (b) Compute model-predicted dR/dt(R) for comparison with ApRES

Run standalone:
    python southpole_apres_traveltime.py

This produces a diagnostic figure showing the depth-to-range mapping,
the two terms of dR/dt, and a comparison with processed ApRES data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# Physical constants
# ============================================================================
C_LIGHT  = 299_792_458.0     # speed of light in vacuum (m/s)
N_ICE    = 1.775              # refractive index of pure ice (CRIM)
RHO_ICE  = 917.0             # density of pure ice (kg/m^3)
F_C      = 300.0e6           # ApRES centre frequency (Hz)
LAM_C    = C_LIGHT / F_C     # free-space centre wavelength (m)
LAM_EFF  = LAM_C / N_ICE     # ice-equivalent wavelength (m)
SPY      = 365.25 * 86400.0  # seconds per year

# CRIM prefactor:  (n_ice - 1) / (n_ice * rho_ice)
CRIM_FACTOR = (N_ICE - 1.0) / (N_ICE * RHO_ICE)

# ============================================================================
# Paths
# ============================================================================
_HERE      = Path(__file__).parent
_SOUTHPOLE = _HERE.parent
OUT_DIR    = _SOUTHPOLE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_VV_CSV = _SOUTHPOLE / "processed" / "apres_vertical_velocity_processed.csv"
SP19_DENS_CSV    = _SOUTHPOLE / "processed" / "sp19_density.csv"

# ============================================================================
# CRIM functions
# ============================================================================

def crim_refractive_index(rho: np.ndarray) -> np.ndarray:
    """CRIM refractive index: n(rho) = 1 + (n_ice - 1) * rho / rho_ice."""
    return 1.0 + (N_ICE - 1.0) * np.asarray(rho) / RHO_ICE


def crim_permittivity(rho: np.ndarray) -> np.ndarray:
    """CRIM relative permittivity: eps = n(rho)^2."""
    n = crim_refractive_index(rho)
    return n ** 2


def depth_to_range(z: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Convert true depth z to apparent range R (ice-equivalent metres).

    R(z_R) = integral_0^{z_R}  n(rho(z)) / n_ice  dz

    Parameters
    ----------
    z : (N,) depth from surface (m), must be monotonically increasing
    rho : (N,) density profile (kg/m^3)

    Returns
    -------
    R : (N,) apparent range (ice-equivalent m)
    """
    z = np.asarray(z)
    rho = np.asarray(rho)
    n = crim_refractive_index(rho)
    integrand = n / N_ICE  # n(rho)/n_ice; =1 for pure ice

    # Cumulative integral using trapezoidal rule
    R = np.zeros_like(z)
    R[1:] = cumulative_trapezoid(integrand, z)
    return R


def range_to_depth(R_target: np.ndarray, z: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Invert the depth-to-range mapping: given R, find z.

    Parameters
    ----------
    R_target : (M,) apparent range values to invert
    z : (N,) depth grid
    rho : (N,) density profile on z grid

    Returns
    -------
    z_target : (M,) true depths corresponding to R_target
    """
    R = depth_to_range(z, rho)
    return np.interp(R_target, R, z, left=np.nan, right=np.nan)


# ============================================================================
# Forward model: predicted dR/dt from model fields
# ============================================================================

def predicted_dRdt(
    z: np.ndarray,
    rho: np.ndarray,
    w: np.ndarray,
    drhodt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute model-predicted apparent-range change rate dR/dt.

    dR/dt(z_R) = [n(rho(z_R))/n_ice] * w(z_R)
                 + CRIM_FACTOR * integral_0^{z_R} drho/dt(z) dz

    Parameters
    ----------
    z : (N,) depth from surface (m), monotonically increasing
    rho : (N,) density (kg/m^3)
    w : (N,) vertical velocity (m/s), negative = downward (compaction)
    drhodt : (N,) densification rate (kg/m^3/s)

    Returns
    -------
    R : (N,) apparent range axis (ice-equivalent m)
    dRdt : (N,) total dR/dt (m/s) on the range axis
    term1 : (N,) displacement term
    term2 : (N,) density-change path-length term
    """
    z = np.asarray(z)
    rho = np.asarray(rho)
    w = np.asarray(w)
    drhodt = np.asarray(drhodt)

    R = depth_to_range(z, rho)

    # Term 1: physical displacement scaled by local refractive index ratio
    n_local = crim_refractive_index(rho)
    term1 = (n_local / N_ICE) * w

    # Term 2: cumulative density-change path-length correction
    int_drhodt = np.zeros_like(z)
    int_drhodt[1:] = cumulative_trapezoid(drhodt, z)
    term2 = CRIM_FACTOR * int_drhodt

    dRdt = term1 + term2

    return R, dRdt, term1, term2


def predicted_dRdt_on_range_axis(
    z: np.ndarray,
    rho: np.ndarray,
    w: np.ndarray,
    drhodt: np.ndarray,
    R_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute model-predicted dR/dt interpolated to observed range bins.

    Parameters
    ----------
    z, rho, w, drhodt : model fields on depth axis (see predicted_dRdt)
    R_obs : (M,) observed range bins (ice-equivalent m)

    Returns
    -------
    dRdt_obs : (M,) predicted dR/dt at observed range bins
    term1_obs : (M,) displacement contribution
    term2_obs : (M,) density-change contribution
    """
    R, dRdt, term1, term2 = predicted_dRdt(z, rho, w, drhodt)

    dRdt_obs = np.interp(R_obs, R, dRdt, left=np.nan, right=np.nan)
    term1_obs = np.interp(R_obs, R, term1, left=np.nan, right=np.nan)
    term2_obs = np.interp(R_obs, R, term2, left=np.nan, right=np.nan)

    return dRdt_obs, term1_obs, term2_obs


# ============================================================================
# Synthetic density profile for standalone diagnostic
# ============================================================================

def herron_langway_profile(z: np.ndarray, rho_s: float = 350.0,
                           z_e: float = 30.0) -> np.ndarray:
    """Simple exponential density profile for illustration."""
    return RHO_ICE - (RHO_ICE - rho_s) * np.exp(-z / z_e)


def eulerian_drhodt_steady(z: np.ndarray, rho: np.ndarray,
                           w: np.ndarray) -> np.ndarray:
    """Eulerian ∂ρ/∂t from mass continuity.

    ∂ρ/∂t = -∂(ρw)/∂z

    In strict steady state this is zero.  Returns drho/dt in kg/m^3/s.
    """
    rho_w = rho * w
    drhodt = -np.gradient(rho_w, z)
    return drhodt


def eulerian_drhodt_perturbed(z: np.ndarray, rho: np.ndarray,
                              w: np.ndarray,
                              accum_perturbation: float = 0.20) -> np.ndarray:
    """Transient Eulerian ∂ρ/∂t from a step increase in accumulation.

    After a sudden increase in accumulation, the near-surface density
    decreases (more fresh snow) while the velocity field hasn't fully
    adjusted, producing a non-zero ∂ρ/∂t.  This illustrates Term 2
    in the dR/dt decomposition.

    Parameters
    ----------
    accum_perturbation : fractional increase in accumulation (0.20 = 20%)
    """
    # Lagrangian rate: Dρ/Dt = w * ∂ρ/∂z (how fast a layer densifies)
    lagrangian = w * np.gradient(rho, z)

    # If accumulation increases, the Lagrangian rate stays roughly the same
    # but the advection term increases → net ∂ρ/∂t < 0 (density decreasing
    # at each fixed depth because lighter snow arrives faster)
    drhodt = -accum_perturbation * lagrangian
    return drhodt


def steady_state_velocity(z: np.ndarray, rho: np.ndarray,
                          accum_m_ice_yr: float = 0.022) -> np.ndarray:
    """Steady-state vertical velocity from mass continuity.

    w(z) = -accum * rho_ice / rho(z) * (1/spy)
    adjusted for densification: w(z) = w_surf * rho_s / rho(z)
    """
    rho_s = rho[0]
    w_surf = -accum_m_ice_yr * RHO_ICE / rho_s / SPY  # m/s, negative
    w = w_surf * rho_s / rho  # continuity: rho*w = const in steady state
    return w


# ============================================================================
# Load observed ApRES data
# ============================================================================

def load_processed_vv(site: str = "x11n2"):
    """Load processed ApRES vertical velocities from CSV."""
    import pandas as pd
    df = pd.read_csv(PROCESSED_VV_CSV)
    ds = df[df["site"] == site].copy()
    return {
        "range_m": ds["range_m"].values,
        "v_m_yr": ds["v_m_yr"].values,
        "v_smooth_m_yr": ds["v_smooth_m_yr"].values,
        "coherence": ds["coherence"].values,
        "strain_rate_yr": ds["strain_rate_yr"].values,
        "dt_days": ds["dt_days"].values[0],
    }


def load_sp19_density():
    """Load SP19 ice-core density profile."""
    import pandas as pd
    df = pd.read_csv(SP19_DENS_CSV)
    return df["depth_m"].values, df["rho_kgm3"].values


# ============================================================================
# Plotting
# ============================================================================

def plot_traveltime_diagnostic(out_path: Path):
    """Multi-panel diagnostic showing depth-to-range mapping and dR/dt decomposition."""

    # --- Synthetic model profile ---
    z = np.linspace(0, 130, 1000)
    rho = herron_langway_profile(z, rho_s=350.0, z_e=30.0)
    w = steady_state_velocity(z, rho, accum_m_ice_yr=0.022)
    # Use perturbed Eulerian rate to show non-zero Term 2
    drhodt = eulerian_drhodt_perturbed(z, rho, w, accum_perturbation=0.20)

    # Overlay SP19 density if available
    sp19_z, sp19_rho = None, None
    if SP19_DENS_CSV.exists():
        sp19_z, sp19_rho = load_sp19_density()

    # Compute model-predicted dR/dt
    R, dRdt, term1, term2 = predicted_dRdt(z, rho, w, drhodt)

    # Convert to m/yr
    dRdt_yr = dRdt * SPY
    term1_yr = term1 * SPY
    term2_yr = term2 * SPY

    # Load ApRES data for comparison
    apres = None
    if PROCESSED_VV_CSV.exists():
        try:
            apres = load_processed_vv("x11n2")
        except Exception:
            pass

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ---- Panel (0,0): Density profile ----
    ax = axes[0, 0]
    ax.plot(rho, z, "steelblue", lw=2, label="Synthetic (HL-like)")
    if sp19_z is not None:
        ax.scatter(sp19_rho, sp19_z, s=15, color="red", zorder=3,
                   label="SP19 ice core")
    ax.set_xlabel("Density (kg/m3)")
    ax.set_ylabel("Depth z (m)")
    ax.invert_yaxis()
    ax.set_title("Density profile")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    # ---- Panel (0,1): Depth vs apparent range ----
    ax = axes[0, 1]
    ax.plot(z, z, "k--", lw=1, label="True depth z")
    ax.plot(R, z, "steelblue", lw=2, label="Apparent range R(z)")
    ax.fill_betweenx(z, z, R, alpha=0.15, color="steelblue")
    delta = R - z
    ax.set_xlabel("Depth / Range (m)")
    ax.set_ylabel("True depth z (m)")
    ax.invert_yaxis()
    ax.set_title(f"Depth -> Range mapping\n(max offset {delta.max():.1f} m at surface)")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    # ---- Panel (0,2): n(rho)/n_ice ratio ----
    ax = axes[0, 2]
    n_ratio = crim_refractive_index(rho) / N_ICE
    ax.plot(n_ratio, z, "steelblue", lw=2)
    ax.axvline(1.0, color="k", ls="--", lw=0.8, label="Pure ice (n/n_ice = 1)")
    ax.set_xlabel("n(rho) / n_ice")
    ax.set_ylabel("Depth z (m)")
    ax.invert_yaxis()
    ax.set_title("Local refractive index ratio\n(travel-time vs depth scaling)")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    # ---- Panel (1,0): dR/dt decomposition on depth axis ----
    ax = axes[1, 0]
    ax.plot(dRdt_yr * 100, z, "k", lw=2, label="Total dR/dt")
    ax.plot(term1_yr * 100, z, "steelblue", lw=1.5, ls="--",
            label="Term 1: displacement")
    ax.plot(term2_yr * 100, z, "firebrick", lw=1.5, ls=":",
            label="Term 2: density change")
    ax.set_xlabel("dR/dt (cm/yr)")
    ax.set_ylabel("Depth z (m)")
    ax.invert_yaxis()
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_title("dR/dt decomposition\n(depth axis)")
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", alpha=0.3)

    # ---- Panel (1,1): dR/dt on RANGE axis (what ApRES sees) ----
    ax = axes[1, 1]
    firn_mask = R <= 150
    ax.plot(dRdt_yr[firn_mask] * 100, R[firn_mask], "k", lw=2,
            label="Model dR/dt")
    ax.plot(term1_yr[firn_mask] * 100, R[firn_mask], "steelblue",
            lw=1.2, ls="--", alpha=0.7, label="Displacement")
    ax.plot(term2_yr[firn_mask] * 100, R[firn_mask], "firebrick",
            lw=1.2, ls=":", alpha=0.7, label="Density change")
    if apres is not None:
        amask = (apres["range_m"] <= 150) & (apres["coherence"] > 0.5)
        ax.plot(apres["v_smooth_m_yr"][amask] * 100,
                apres["range_m"][amask],
                color="green", lw=1.5, alpha=0.8,
                label="ApRES x11n2 VV")
    ax.set_xlabel("dR/dt (cm/yr)")
    ax.set_ylabel("Apparent range R (m)")
    ax.invert_yaxis()
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_title("dR/dt on RANGE axis\n(ApRES coordinate)")
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", alpha=0.3)

    # ---- Panel (1,2): Strain rate comparison ----
    ax = axes[1, 2]
    # Model strain rate on range axis: d(dR/dt)/dR
    dRdt_smooth = gaussian_filter1d(dRdt_yr, sigma=5)
    strain_model = np.gradient(dRdt_smooth, R) * 100  # %/yr
    ax.plot(strain_model[firn_mask], R[firn_mask], "k", lw=2,
            label="Model d(dR/dt)/dR")
    if apres is not None:
        amask = (apres["range_m"] <= 150) & (apres["coherence"] > 0.5)
        ax.plot(apres["strain_rate_yr"][amask] * 100,
                apres["range_m"][amask],
                color="green", lw=1.5, alpha=0.8,
                label="ApRES x11n2 strain")
    ax.set_xlabel("Apparent strain rate (%/yr)")
    ax.set_ylabel("Apparent range R (m)")
    ax.invert_yaxis()
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_title("Apparent strain rate\non RANGE axis")
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", alpha=0.3)

    fig.suptitle(
        "ApRES travel-time coordinate framework for firn model comparison\n"
        "R(z) = integral n(rho)/n_ice dz  |  "
        "dR/dt = n/n_ice * w + CRIM_factor * integral drho/dt dz\n"
        "(Case et al. 2020; Kingslake et al. 2014)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_multisite_comparison(out_path: Path):
    """Compare synthetic model dR/dt with all ApRES sites."""
    import pandas as pd

    if not PROCESSED_VV_CSV.exists():
        print("  No processed VV data found, skipping multisite plot")
        return

    df = pd.read_csv(PROCESSED_VV_CSV)
    sites = sorted(df["site"].unique())

    # Synthetic model
    z = np.linspace(0, 130, 1000)
    rho = herron_langway_profile(z, rho_s=350.0, z_e=30.0)
    w = steady_state_velocity(z, rho, accum_m_ice_yr=0.022)
    drhodt = eulerian_drhodt_perturbed(z, rho, w, accum_perturbation=0.20)
    R, dRdt, _, _ = predicted_dRdt(z, rho, w, drhodt)
    dRdt_yr = dRdt * SPY

    n_sites = len(sites)
    n_cols = min(n_sites, 5)
    n_rows = (n_sites + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             squeeze=False)
    cmap = plt.cm.tab10

    for i, site in enumerate(sites):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        ds = df[df["site"] == site]
        r_obs = ds["range_m"].values
        v_obs = ds["v_smooth_m_yr"].values
        coh = ds["coherence"].values

        mask = (r_obs <= 150) & (coh > 0.5)
        color = cmap(i / max(n_sites - 1, 1))

        ax.plot(v_obs[mask] * 100, r_obs[mask],
                lw=1.2, color=color, label="ApRES VV")

        # Model prediction on ApRES range bins
        firn_mask = R <= 150
        ax.plot(dRdt_yr[firn_mask] * 100, R[firn_mask],
                "k--", lw=1.2, alpha=0.6, label="Model dR/dt")

        ax.set_title(site, fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        ax.axvline(0, color="grey", lw=0.5)
        ax.grid(True, ls="--", alpha=0.3)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("Range R (m)", fontsize=8)
        if row == n_rows - 1:
            ax.set_xlabel("dR/dt (cm/yr)", fontsize=8)
        if i == 0:
            ax.legend(fontsize=6)

    # Hide unused axes
    for i in range(n_sites, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        "Model dR/dt vs ApRES VV (all sites) in travel-time coordinates\n"
        "Synthetic HL profile (rho_s=350, z_e=30m, bdot=22mm/yr, T=222K)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("ApRES travel-time coordinate framework")
    print("=" * 65)

    # --- Diagnostic: depth-to-range mapping and dR/dt decomposition ---
    print("\n--- Travel-time coordinate diagnostic ---")
    z = np.linspace(0, 130, 500)
    rho = herron_langway_profile(z, rho_s=350.0, z_e=30.0)
    R = depth_to_range(z, rho)
    print(f"  Max range offset (R - z) at surface: {(R[1]-z[1]):.3f} m")
    print(f"  Max range offset at 130 m:           {(R[-1]-z[-1]):.1f} m")

    # CRIM sensitivity
    print(f"\n  CRIM factor: {CRIM_FACTOR:.6e} m / (kg/m^3)")
    print(f"  lambda_eff (ice-equivalent): {LAM_EFF:.4f} m")
    print(f"  n_ice: {N_ICE}")

    # dR/dt decomposition — steady state (∂ρ/∂t = 0)
    w = steady_state_velocity(z, rho, accum_m_ice_yr=0.022)
    drhodt_ss = eulerian_drhodt_steady(z, rho, w)
    R, dRdt_ss, term1_ss, term2_ss = predicted_dRdt(z, rho, w, drhodt_ss)

    print("\n  dR/dt decomposition — STEADY STATE (∂ρ/∂t ≈ 0):")
    print(f"  {'Depth':>6s} {'Range':>6s} {'Total':>10s} {'Displace':>10s} {'DensChg':>10s}")
    for zi in [10, 30, 50, 80, 110]:
        idx = np.argmin(np.abs(z - zi))
        total = dRdt_ss[idx] * SPY * 100  # cm/yr
        t1 = term1_ss[idx] * SPY * 100
        t2 = term2_ss[idx] * SPY * 100
        print(f"  {z[idx]:6.1f} {R[idx]:6.1f} {total:+10.4f} {t1:+10.4f} {t2:+10.4f}  cm/yr")

    firn = z <= 100
    print(f"\n  Note: Term 2 ≈ 0 in steady state — this is correct physics!")
    print(f"  The Eulerian ∂ρ/∂t at fixed depth z is zero when the density")
    print(f"  profile doesn't change. Term 2 becomes non-zero during transients.")

    # dR/dt decomposition — 20% accumulation perturbation
    drhodt_pert = eulerian_drhodt_perturbed(z, rho, w, accum_perturbation=0.20)
    R, dRdt_p, term1_p, term2_p = predicted_dRdt(z, rho, w, drhodt_pert)

    print(f"\n  dR/dt decomposition — 20% ACCUMULATION INCREASE (transient):")
    print(f"  {'Depth':>6s} {'Range':>6s} {'Total':>10s} {'Displace':>10s} {'DensChg':>10s}")
    for zi in [10, 30, 50, 80, 110]:
        idx = np.argmin(np.abs(z - zi))
        total = dRdt_p[idx] * SPY * 100  # cm/yr
        t1 = term1_p[idx] * SPY * 100
        t2 = term2_p[idx] * SPY * 100
        print(f"  {z[idx]:6.1f} {R[idx]:6.1f} {total:+10.4f} {t1:+10.4f} {t2:+10.4f}  cm/yr")

    t2_frac = np.abs(term2_p[firn]).mean() / (np.abs(dRdt_p[firn]).mean() + 1e-30)
    print(f"\n  Density-change term is {t2_frac*100:.1f}% of total dR/dt (0-100 m)")

    # --- Plots ---
    print("\n--- Generating plots ---")
    plot_traveltime_diagnostic(OUT_DIR / "diag_apres_traveltime.png")
    plot_multisite_comparison(OUT_DIR / "diag_apres_traveltime_multisite.png")

    print("\nDone.")
