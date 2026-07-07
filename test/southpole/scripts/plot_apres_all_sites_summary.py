"""
Comprehensive summary figure of all 10 ApRES sites at South Pole.
Produces a 2x2 panel figure for deciding which site(s) to use in the
firn densification inversion.

Panels:
  (a) Raw v_smooth vs depth for all sites (firn zone 0-120 m)
  (b) Detrended v_smooth vs depth (deep-ice linear trend removed)
  (c) Multi-site mean ± 1 std of detrended signal, plus individual sites
  (d) Multi-site standard deviation vs depth (the depth-dependent σ_apres)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ICE = 1.775          # refractive index of ice (CRIM)
RHO_ICE = 917.0        # kg m-3
RHO_SURF = 350.0       # surface density assumption, kg m-3

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
base = Path(__file__).resolve().parent.parent
apres = pd.read_csv(base / "processed" / "apres_vertical_velocity_processed.csv")
sp19 = pd.read_csv(base / "processed" / "sp19_density.csv")

# Fix density units: file has g/cm3 (values 0.36-0.84), need kg/m3
sp19["rho_kgm3"] = sp19["rho_kgm3"] * 1000.0

# ---------------------------------------------------------------------------
# Build range-to-depth conversion via CRIM model on SP19 density profile
# ---------------------------------------------------------------------------
# Extend SP19 to surface
depths_sp19 = np.concatenate([[0.0], sp19["depth_m"].values])
rho_sp19 = np.concatenate([[RHO_SURF], sp19["rho_kgm3"].values])

# Also extend deeper with pure ice density for deep-range conversion
max_range_needed = apres["range_m"].max() + 100.0
# Extend to 1500 m depth (well beyond any range we need)
ext_depths = np.arange(depths_sp19[-1] + 1.0, 1500.0, 1.0)
depths_ext = np.concatenate([depths_sp19, ext_depths])
rho_ext = np.concatenate([rho_sp19, np.full(len(ext_depths), RHO_ICE)])

# Refractive index at each depth
n_rho = 1.0 + (N_ICE - 1.0) * rho_ext / RHO_ICE

# Range = cumulative integral of n(rho)/n_ice * dz  (depth -> range)
# Use trapezoidal integration
dz = np.diff(depths_ext)
n_mid = 0.5 * (n_rho[:-1] + n_rho[1:])
dr = n_mid / N_ICE * dz
range_from_depth = np.concatenate([[0.0], np.cumsum(dr)])

# Build interpolator:  range -> depth
range_to_depth_interp = interp1d(range_from_depth, depths_ext,
                                  kind="linear", fill_value="extrapolate")

# ---------------------------------------------------------------------------
# Process each site
# ---------------------------------------------------------------------------
sites = sorted(apres["site"].unique())
n_sites = len(sites)

# Distinct colors for 10 sites
cmap = plt.cm.tab10
colors = {s: cmap(i) for i, s in enumerate(sites)}

# Storage for interpolated profiles on common depth grid
depth_grid = np.linspace(0, 120, 500)  # common depth grid for firn zone
detrended_profiles = {}  # site -> (depth_array, detrended_cm_yr)
raw_profiles = {}        # site -> (depth_array, raw_cm_yr)

# Deep-ice fit results
fit_results = {}

for site in sites:
    df = apres[apres["site"] == site].copy()

    # Filter by coherence
    df = df[df["coherence"] > 0.5].copy()
    df = df.sort_values("range_m")

    # Convert range to depth
    df["depth_m"] = range_to_depth_interp(df["range_m"].values)

    # Convert velocity to cm/yr
    df["v_cm_yr"] = df["v_smooth_m_yr"] * 100.0

    # ---- Deep-ice linear fit (200-800 m RANGE) ----
    deep = df[(df["range_m"] >= 200) & (df["range_m"] <= 800)]
    if len(deep) < 10:
        print(f"WARNING: {site} has only {len(deep)} deep-ice points, skipping.")
        continue

    coeffs = np.polyfit(deep["range_m"].values, deep["v_cm_yr"].values, 1)
    slope, intercept = coeffs  # cm/yr per m-range, cm/yr

    # Store fit
    fit_results[site] = {"slope": slope, "intercept": intercept}

    # Detrend: subtract the linear fit evaluated at range
    df["v_detrended_cm_yr"] = df["v_cm_yr"] - np.polyval(coeffs, df["range_m"].values)

    # ---- Raw profile in firn zone ----
    firn = df[df["depth_m"] <= 120].copy()
    raw_profiles[site] = (firn["depth_m"].values, firn["v_cm_yr"].values)

    # ---- Detrended profile in firn zone ----
    detrended_profiles[site] = (firn["depth_m"].values, firn["v_detrended_cm_yr"].values)

# ---------------------------------------------------------------------------
# Interpolate detrended profiles to common grid for mean/std
# ---------------------------------------------------------------------------
interp_detrended = {}
for site in sites:
    if site not in detrended_profiles:
        continue
    d, v = detrended_profiles[site]
    if len(d) < 5:
        continue
    f = interp1d(d, v, kind="linear", bounds_error=False, fill_value=np.nan)
    interp_detrended[site] = f(depth_grid)

# Stack for statistics
stack = np.array(list(interp_detrended.values()))  # (n_sites, n_depths)
with np.errstate(all="ignore"):
    mean_detrended = np.nanmean(stack, axis=0)
    std_detrended = np.nanstd(stack, axis=0)
    count_valid = np.sum(~np.isnan(stack), axis=0)

# Mask where fewer than 3 sites contribute
mask = count_valid >= 3
mean_detrended[~mask] = np.nan
std_detrended[~mask] = np.nan

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax_a, ax_b, ax_c, ax_d = axes.flat

# ---- Panel (a): Raw v_smooth vs depth ----
for site in sites:
    if site not in raw_profiles:
        continue
    d, v = raw_profiles[site]
    ax_a.plot(v, d, color=colors[site], linewidth=0.8, alpha=0.85, label=site)
ax_a.set_xlabel("v_smooth (cm yr$^{-1}$)")
ax_a.set_ylabel("Depth (m)")
ax_a.set_ylim(120, 0)
ax_a.set_title("(a) Raw compaction velocity")
ax_a.legend(fontsize=7, ncol=2, loc="lower left")
ax_a.grid(True, alpha=0.3)

# ---- Panel (b): Detrended v_smooth vs depth ----
for site in sites:
    if site not in detrended_profiles:
        continue
    d, v = detrended_profiles[site]
    ax_b.plot(v, d, color=colors[site], linewidth=0.8, alpha=0.85, label=site)
ax_b.axvline(0, color="k", linewidth=0.5, linestyle="--")
ax_b.set_xlabel("Detrended v_smooth (cm yr$^{-1}$)")
ax_b.set_ylabel("Depth (m)")
ax_b.set_ylim(120, 0)
ax_b.set_title("(b) Detrended compaction velocity")
ax_b.legend(fontsize=7, ncol=2, loc="lower left")
ax_b.grid(True, alpha=0.3)

# ---- Panel (c): Mean ± std of detrended ----
for site in sites:
    if site not in interp_detrended:
        continue
    ax_c.plot(interp_detrended[site], depth_grid,
              color=colors[site], linewidth=0.5, alpha=0.4)

ax_c.fill_betweenx(depth_grid, mean_detrended - std_detrended,
                    mean_detrended + std_detrended,
                    color="gray", alpha=0.35, label="$\\pm$1 std")
ax_c.plot(mean_detrended, depth_grid, "k-", linewidth=2.0, label="Mean")
ax_c.axvline(0, color="k", linewidth=0.5, linestyle="--")
ax_c.set_xlabel("Detrended v_smooth (cm yr$^{-1}$)")
ax_c.set_ylabel("Depth (m)")
ax_c.set_ylim(120, 0)
ax_c.set_title("(c) Consensus densification signal")
ax_c.legend(fontsize=8, loc="lower left")
ax_c.grid(True, alpha=0.3)

# ---- Panel (d): Std vs depth ----
ax_d.plot(std_detrended, depth_grid, "k-", linewidth=1.5, label="$\\sigma_{\\mathrm{apres}}$")
ax_d.axvline(0.5, color="r", linewidth=1.0, linestyle="--", label="Floor = 0.5 cm yr$^{-1}$")
ax_d.set_xlabel("$\\sigma_{\\mathrm{apres}}$ (cm yr$^{-1}$)")
ax_d.set_ylabel("Depth (m)")
ax_d.set_ylim(120, 0)
ax_d.set_title("(d) Inter-site variability")
ax_d.legend(fontsize=8, loc="lower left")
ax_d.grid(True, alpha=0.3)

fig.suptitle("South Pole ApRES dR/dt — all 10 sites", fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = base / "results" / "apres_all_sites_summary.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nFigure saved to: {out_path}")
plt.close()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print(f"{'Site':<10} {'Slope':>12} {'Intercept':>12} {'Mean dR/dt':>14} {'Anomalous?':>12}")
print(f"{'':10} {'(cm/yr/m)':>12} {'(cm/yr)':>12} {'firn (cm/yr)':>14} {'':>12}")
print("-" * 90)

# Compute overall statistics of the mean detrended value for anomaly detection
mean_vals = {}
for site in sites:
    if site not in interp_detrended:
        continue
    firn_vals = interp_detrended[site]
    mean_vals[site] = np.nanmean(firn_vals)

overall_mean = np.mean(list(mean_vals.values()))
overall_std = np.std(list(mean_vals.values()))

for site in sites:
    if site not in fit_results:
        print(f"{site:<10} {'N/A':>12} {'N/A':>12} {'N/A':>14} {'SKIP':>12}")
        continue

    sl = fit_results[site]["slope"]
    ic = fit_results[site]["intercept"]
    mv = mean_vals.get(site, np.nan)

    # Flag as anomalous if > 2 std from the group mean
    anomalous = "YES" if abs(mv - overall_mean) > 2.0 * overall_std else ""

    print(f"{site:<10} {sl:>12.6f} {ic:>12.4f} {mv:>14.4f} {anomalous:>12}")

print("-" * 90)
print(f"{'Group':10} {'':>12} {'':>12} {overall_mean:>14.4f} {'mean':>12}")
print(f"{'':10} {'':>12} {'':>12} {overall_std:>14.4f} {'std':>12}")
print("=" * 90)
print("\nAnomalous = mean detrended dR/dt in firn zone > 2σ from group mean.")
