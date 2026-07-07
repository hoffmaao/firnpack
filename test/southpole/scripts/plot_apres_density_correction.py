"""
Plot ApRES vertical velocities with and without density correction.

Hills et al. compute VV using the ice-equivalent wavelength:
    v_hills = phase * c_ice / (4*pi*f_c*dt)

This assumes the radar wave travels at c_ice everywhere. In firn (rho < rho_ice),
the wave travels faster (n(rho) < n_ice), so the same phase change corresponds to
a larger physical displacement:

    w_true = v_hills * n_ice / n(rho(z))

where n(rho) = 1 + (n_ice - 1)*rho/rho_ice  (CRIM formula).

We use the SP19 ice-core density to compute the correction.

Additionally, the Hills data is on a range axis R (ice-equivalent metres).
The true depth z differs from R in firn. We map R -> z using the density profile.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# ---- Paths ----
SOUTHPOLE = Path(__file__).parent.parent
VV_DIR = SOUTHPOLE / "data/usap_dc/601503/vertical_velocities/vertical_velocities"
SP19_CSV = SOUTHPOLE / "processed/sp19_density.csv"
OUT_DIR = SOUTHPOLE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Constants ----
N_ICE = 1.775
RHO_ICE = 917.0


def crim_n(rho):
    """CRIM refractive index."""
    return 1.0 + (N_ICE - 1.0) * np.asarray(rho) / RHO_ICE


def load_sp19():
    """Load SP19 density, convert to kg/m3, and extend to 300 m with ice density."""
    df = pd.read_csv(SP19_CSV)
    z = df["depth_m"].values
    rho = df["rho_kgm3"].values

    # Data is actually in g/cm3 (values 0.36-0.84), convert to kg/m3
    if rho.max() < 10:
        rho = rho * 1000.0

    # Extend to 300 m with ice density so range mapping covers all ApRES bins
    z_ext = np.linspace(z[-1] + 0.5, 300, 200)
    rho_ext = np.full_like(z_ext, RHO_ICE)
    z = np.concatenate([z, z_ext])
    rho = np.concatenate([rho, rho_ext])

    return z, rho


def depth_to_range(z, rho):
    """Convert true depth z to apparent range R (ice-equivalent m)."""
    n = crim_n(rho)
    integrand = n / N_ICE
    R = np.zeros_like(z)
    R[1:] = cumulative_trapezoid(integrand, z)
    return R


def range_to_depth_interp(R_target, z, rho):
    """Map apparent range R -> true depth z."""
    R = depth_to_range(z, rho)
    return np.interp(R_target, R, z, left=np.nan, right=np.nan)


def density_at_range(R_target, z, rho):
    """Get density at given range bins by mapping R -> z -> rho."""
    z_target = range_to_depth_interp(R_target, z, rho)
    return np.interp(z_target, z, rho, left=np.nan, right=np.nan)


def load_hills_vv(path):
    """Load a Hills VV file."""
    d = np.loadtxt(path, skiprows=1)
    return {
        "range_m": d[:, 0],
        "coherence": d[:, 1],
        "phase_offset_rad": d[:, 2],
        "v_m_yr": d[:, 3],
        "v_unc_m_yr": d[:, 4],
    }


# ---- Main ----
if __name__ == "__main__":
    z_sp19, rho_sp19 = load_sp19()

    # Load all sites
    sites = sorted(VV_DIR.glob("*_VV.txt"))
    site_data = {}
    for f in sites:
        name = f.stem.replace("_VV", "")
        site_data[name] = load_hills_vv(f)

    site_names = list(site_data.keys())
    n = len(site_names)

    # ---- Figure 1: All sites, raw vs density-corrected, 0-200 m ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    cmap = plt.cm.tab10

    for i, name in enumerate(site_names):
        d = site_data[name]
        r = d["range_m"]
        v = d["v_m_yr"]
        coh = d["coherence"]

        # Map range -> depth and get local density
        z_at_r = range_to_depth_interp(r, z_sp19, rho_sp19)
        rho_at_r = density_at_range(r, z_sp19, rho_sp19)
        n_at_r = crim_n(rho_at_r)

        # Density-corrected velocity: w_true = v_hills * n_ice / n(rho)
        # This corrects for the refractive index difference in firn
        v_corrected = v * N_ICE / n_at_r

        mask = (r >= 10) & (r <= 200) & (coh > 0.5)
        color = cmap(i % 10)

        ax1.plot(v[mask] * 100, r[mask], color=color, lw=1.0, label=name)
        ax2.plot(v_corrected[mask] * 100, r[mask], color=color, lw=1.0, label=name)

        # Correction factor (how much the velocity changes)
        correction = (v_corrected[mask] - v[mask]) * 100
        ax3.plot(correction, r[mask], color=color, lw=1.0, label=name)

    ax1.set_ylabel("Apparent range R (m)", fontsize=11)
    ax1.set_xlabel("dR/dt  (cm yr$^{-1}$)", fontsize=11)
    ax1.set_title("Hills et al. raw VV\n(ice refractive index assumed)", fontsize=10)
    ax1.invert_yaxis()
    ax1.legend(fontsize=7, loc="lower left")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("w corrected  (cm yr$^{-1}$)", fontsize=11)
    ax2.set_title("Density-corrected velocity\n$w = v_{Hills} \\times n_{ice} / n(\\rho)$",
                   fontsize=10)
    ax2.invert_yaxis()
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel("Correction  (cm yr$^{-1}$)", fontsize=11)
    ax3.set_title("Difference: corrected $-$ raw\n(refractive index effect only)",
                   fontsize=10)
    ax3.invert_yaxis()
    ax3.axvline(0, color="grey", lw=0.5)
    ax3.legend(fontsize=7, loc="lower left")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        "ApRES vertical velocity: effect of firn density on velocity derivation\n"
        "Density correction using SP19 ice-core profile via CRIM refractive index",
        fontsize=12, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out1 = OUT_DIR / "apres_density_correction_comparison.png"
    fig.savefig(out1, dpi=150)
    print(f"Saved: {out1}")
    plt.close(fig)

    # ---- Figure 2: Per-site panels on true depth axis ----
    n_cols = min(n, 5)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                              squeeze=False, sharey=True)

    for i, name in enumerate(site_names):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        d = site_data[name]
        r = d["range_m"]
        v = d["v_m_yr"]
        coh = d["coherence"]

        z_at_r = range_to_depth_interp(r, z_sp19, rho_sp19)
        rho_at_r = density_at_range(r, z_sp19, rho_sp19)
        n_at_r = crim_n(rho_at_r)
        v_corrected = v * N_ICE / n_at_r

        mask = (r >= 10) & (r <= 200) & (coh > 0.5) & np.isfinite(z_at_r)

        ax.plot(v[mask] * 100, z_at_r[mask], "steelblue", lw=1.2,
                label="Raw (ice $n$)")
        ax.plot(v_corrected[mask] * 100, z_at_r[mask], "firebrick", lw=1.2,
                label="Corrected")

        # Show the n(rho)/n_ice ratio at a few depths
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        ax.axvline(0, color="grey", lw=0.5)
        ax.grid(True, ls="--", alpha=0.3)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("True depth z (m)", fontsize=8)
        if row == n_rows - 1:
            ax.set_xlabel("Velocity (cm/yr)", fontsize=8)
        if i == 0:
            ax.legend(fontsize=6, loc="lower left")

    for i in range(n, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        "Per-site comparison: raw vs density-corrected vertical velocity\n"
        "Mapped to true depth using SP19 density",
        fontsize=11,
    )
    fig.tight_layout()
    out2 = OUT_DIR / "apres_density_correction_persite.png"
    fig.savefig(out2, dpi=150)
    print(f"Saved: {out2}")
    plt.close(fig)

    # ---- Print summary statistics ----
    print("\nDensity correction summary (0-200 m range):")
    print(f"{'Site':>10s}  {'n/n_ice at sfc':>14s}  {'Correction sfc':>14s}  "
          f"{'Correction 100m':>15s}  {'Correction 200m':>15s}")
    for name in site_names:
        d = site_data[name]
        r = d["range_m"]
        v = d["v_m_yr"]
        rho_at_r = density_at_range(r, z_sp19, rho_sp19)
        n_at_r = crim_n(rho_at_r)
        ratio = n_at_r / N_ICE

        for target_r in [10, 100, 200]:
            idx = np.argmin(np.abs(r - target_r))
            corr_pct = (N_ICE / n_at_r[idx] - 1) * 100
            if target_r == 10:
                print(f"{name:>10s}  {ratio[idx]:14.4f}  {corr_pct:+14.1f}%", end="")
            elif target_r == 100:
                print(f"  {corr_pct:+15.1f}%", end="")
            else:
                print(f"  {corr_pct:+15.1f}%")
