"""southpole_apres_preprocess.py

Preprocess ApRES stacked-profile data from Hills et al. (2022), USAP-DC 601503.
Derives vertical velocity and strain rate from pairs of complex radar returns
(2018-19 vs 2019-20 seasons), then compares with Hills' published vertical
velocity products to confirm self-consistency.

The stacked profiles are already range-processed, coherently averaged complex
spectra. To derive vertical velocity we compute, at each range bin:

    coherence   = |<s1 * conj(s2)>| / sqrt(<|s1|^2> * <|s2|^2>)
    phase_shift = angle( conj(s1) * s2 )
    v_vertical  = phase_shift * c_ice / (4 * pi * f_c * dt)

where c_ice = c0/sqrt(eps_ice), f_c is the ApRES center frequency, and dt is
the time between the two stacked profiles.

Run:
    cd test/southpole/scripts
    python southpole_apres_preprocess.py

Outputs (saved to test/southpole/results/):
    diag_apres_preprocess_comparison.png   — Our vs Hills comparison
    apres_vertical_velocity_processed.csv  — Processed VV for all sites
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# Paths
# ============================================================================
_HERE      = Path(__file__).parent
_SOUTHPOLE = _HERE.parent
OUT_DIR    = _SOUTHPOLE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

APRES_DIR = _SOUTHPOLE / "data/usap_dc/601503"
STACKED_DIR = APRES_DIR / "stacked_profiles" / "stacked_profiles"
VV_DIR = APRES_DIR / "vertical_velocities" / "vertical_velocities"

# ============================================================================
# Physical constants
# ============================================================================
C0 = 299_792_458.0          # speed of light in vacuum (m/s)
EPS_ICE = 3.18              # relative permittivity of ice
C_ICE = C0 / np.sqrt(EPS_ICE)  # speed of light in ice (m/s)

# ApRES radar parameters (standard BAS configuration)
F0_HZ = 2.0e8               # start frequency (Hz)
F1_HZ = 4.0e8               # stop  frequency (Hz)
FC_HZ = 0.5 * (F0_HZ + F1_HZ)  # center frequency (Hz)
LAMBDA_C = C_ICE / FC_HZ    # center wavelength in ice (m)

DAYS_PER_YEAR = 365.25


# ============================================================================
# I/O helpers
# ============================================================================

def matlab2datetime(datenum: float) -> datetime:
    """Convert MATLAB datenum to Python datetime."""
    return (datetime.fromordinal(int(datenum))
            + timedelta(days=datenum % 1)
            - timedelta(days=366))


@dataclass
class StackedProfile:
    """A single stacked ApRES profile (complex spectrum vs range)."""
    name: str
    timestamp: datetime
    datenum: float
    range_m: np.ndarray       # (N,)
    spectrum: np.ndarray      # (N,) complex
    uncertainty: np.ndarray   # (N,) real


def read_stacked_profile(path: Path) -> StackedProfile:
    """Read a stacked profile file.

    Format:
        Line 0:  DecimalDay <matlab_datenum>
        Line 1:  Range(m)  ComplexReturn  ComplexUncertainty
        Lines 2+: (range+0j)  (re+im_j)  (unc+0j)
    """
    path = Path(path)
    with open(path) as f:
        header = f.readline().strip()
        f.readline()  # column header line
        data_lines = f.readlines()

    datenum = float(header.split()[1])
    timestamp = matlab2datetime(datenum)

    ranges = []
    spectra = []
    uncertainties = []

    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        # Split on double-space between complex values
        parts = line.split("  ")
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) != 3:
            continue
        r = complex(parts[0]).real
        s = complex(parts[1])
        u = complex(parts[2]).real
        ranges.append(r)
        spectra.append(s)
        uncertainties.append(u)

    return StackedProfile(
        name=path.stem,
        timestamp=timestamp,
        datenum=datenum,
        range_m=np.array(ranges),
        spectrum=np.array(spectra, dtype=complex),
        uncertainty=np.array(uncertainties),
    )


def load_hills_vv(path: Path):
    """Load Hills' published vertical velocity file.

    Columns: Range(m) Coherence PhaseOffset(rad) VerticalVelocity(m/yr) VelocityUncertainty(m/yr)
    """
    d = np.loadtxt(path, skiprows=1)
    return {
        "range_m": d[:, 0],
        "coherence": d[:, 1],
        "phase_offset_rad": d[:, 2],
        "v_m_yr": d[:, 3],
        "v_unc_m_yr": d[:, 4],
    }


# ============================================================================
# Velocity derivation from stacked profiles
# ============================================================================

def compute_vertical_velocity(
    p1: StackedProfile,
    p2: StackedProfile,
    window_bins: int = 20,
    smooth_m: float = 0.0,
) -> dict:
    """Compute vertical velocity from two stacked profiles.

    Uses windowed cross-correlation (matching Hills et al. methodology):
    the range axis is divided into non-overlapping windows of `window_bins`
    bins. Within each window, coherence and phase offset are computed from
    the mean cross-spectrum, then converted to vertical velocity.

    Parameters
    ----------
    p1, p2 : StackedProfile
        Profiles from season 1 and season 2.
    window_bins : int
        Number of range bins per averaging window (default 20, matching
        the 20x downsampling in Hills' published vertical velocities).
    smooth_m : float
        Gaussian smoothing sigma in metres applied after windowing (0 = none).

    Returns
    -------
    dict with keys: range_m, coherence, phase_offset_rad,
                    v_m_yr, v_unc_m_yr, strain_rate_yr
    """
    assert p1.range_m.shape == p2.range_m.shape, "Range grids differ"
    assert np.allclose(p1.range_m, p2.range_m, atol=0.01), "Range grids differ"

    s1 = p1.spectrum
    s2 = p2.spectrum
    u1 = p1.uncertainty
    u2 = p2.uncertainty
    r_full = p1.range_m

    # Time difference
    dt_days = (p2.timestamp - p1.timestamp).total_seconds() / 86400.0
    dt_years = dt_days / DAYS_PER_YEAR

    # Windowed cross-correlation
    n_full = len(r_full)
    n_out = n_full // window_bins

    range_m = np.zeros(n_out)
    coherence = np.zeros(n_out)
    phase_offset = np.zeros(n_out)
    v_unc = np.zeros(n_out)

    for i in range(n_out):
        sl = slice(i * window_bins, (i + 1) * window_bins)
        seg1 = s1[sl]
        seg2 = s2[sl]
        unc1 = u1[sl]
        unc2 = u2[sl]

        range_m[i] = np.mean(r_full[sl])

        # Mean cross-spectrum over window
        cross = np.mean(np.conj(seg1) * seg2)
        pow1 = np.mean(np.abs(seg1) ** 2)
        pow2 = np.mean(np.abs(seg2) ** 2)
        denom = np.sqrt(pow1 * pow2)

        coherence[i] = np.abs(cross) / denom if denom > 0 else 0.0
        phase_offset[i] = np.angle(cross)

        # Uncertainty: propagated through mean cross-spectrum
        cross_amp = np.abs(cross)
        if cross_amp > 0:
            unc_rms = np.sqrt(np.mean(unc1**2 + unc2**2)) / np.sqrt(window_bins)
            v_unc[i] = unc_rms / cross_amp
        else:
            v_unc[i] = np.nan

    # Vertical velocity: v = phi * lambda_c / (4*pi*dt_years)
    v_m_yr = phase_offset * LAMBDA_C / (4.0 * np.pi * dt_years)

    # Velocity uncertainty
    v_unc_m_yr = v_unc * np.abs(LAMBDA_C / (4.0 * np.pi * dt_years))

    # Optional Gaussian smoothing
    if smooth_m > 0 and len(range_m) > 10:
        dr = float(np.median(np.diff(range_m)))
        if dr > 0:
            sigma_bins = smooth_m / dr
            mask_bad = ~np.isfinite(v_m_yr) | (coherence < 0.3)
            fill_val = np.nanmedian(v_m_yr)
            v_filled = np.where(mask_bad, fill_val, v_m_yr)
            v_smooth = gaussian_filter1d(v_filled, sigma=sigma_bins)
            v_smooth[mask_bad] = np.nan
        else:
            v_smooth = v_m_yr.copy()
    else:
        v_smooth = v_m_yr.copy()

    # Strain rate from smoothed velocity: dv/dz
    strain_rate = np.gradient(v_smooth, range_m)

    return {
        "range_m": range_m,
        "coherence": coherence,
        "phase_offset_rad": phase_offset,
        "v_m_yr": v_m_yr,
        "v_smooth_m_yr": v_smooth,
        "v_unc_m_yr": v_unc_m_yr,
        "strain_rate_yr": strain_rate,
        "dt_days": dt_days,
    }


# ============================================================================
# Processing pipeline
# ============================================================================

def find_matching_sites() -> list[tuple[str, Path, Path]]:
    """Find VV sites that exist in both seasons of stacked profiles."""
    s1_dir = STACKED_DIR / "2018_19"
    s2_dir = STACKED_DIR / "2019_20"

    if not s1_dir.exists() or not s2_dir.exists():
        raise FileNotFoundError(f"Stacked profile dirs not found: {s1_dir}, {s2_dir}")

    sites = []
    for f1 in sorted(s1_dir.glob("*_VV.txt")):
        f2 = s2_dir / f1.name
        if f2.exists():
            name = f1.stem.replace("_VV", "")
            sites.append((name, f1, f2))
    return sites


def process_all_sites(smooth_m: float = 25.0) -> list[dict]:
    """Process all matching sites and return results."""
    sites = find_matching_sites()
    results = []

    for name, f1, f2 in sites:
        print(f"  Processing {name}...")
        p1 = read_stacked_profile(f1)
        p2 = read_stacked_profile(f2)

        vv = compute_vertical_velocity(p1, p2, smooth_m=smooth_m)
        vv["name"] = name
        vv["t1"] = p1.timestamp
        vv["t2"] = p2.timestamp

        results.append(vv)

    return results


# ============================================================================
# Comparison with Hills et al. published velocities
# ============================================================================

def load_hills_all() -> dict[str, dict]:
    """Load all Hills vertical velocity files."""
    if not VV_DIR.exists():
        raise FileNotFoundError(f"Vertical velocity dir not found: {VV_DIR}")

    sites = {}
    for f in sorted(VV_DIR.glob("*_VV.txt")):
        name = f.stem.replace("_VV", "")
        sites[name] = load_hills_vv(f)
    return sites


def compare_with_hills(our_results: list[dict], hills: dict[str, dict],
                       firn_max_m: float = 150.0):
    """Compute comparison statistics between our processing and Hills'."""
    stats = []
    for r in our_results:
        name = r["name"]
        if name not in hills:
            continue
        h = hills[name]

        # Interpolate our results onto Hills' range grid
        mask_firn = h["range_m"] <= firn_max_m
        h_range = h["range_m"][mask_firn]
        h_v = h["v_m_yr"][mask_firn]

        our_v_interp = np.interp(h_range, r["range_m"], r["v_m_yr"],
                                  left=np.nan, right=np.nan)

        # Only compare where both are valid
        valid = np.isfinite(our_v_interp) & np.isfinite(h_v) & (h["coherence"][mask_firn] > 0.5)
        if np.sum(valid) < 5:
            continue

        residual = our_v_interp[valid] - h_v[valid]
        rmse = float(np.sqrt(np.mean(residual**2)))
        bias = float(np.mean(residual))
        corr = float(np.corrcoef(our_v_interp[valid], h_v[valid])[0, 1])

        stats.append({
            "name": name,
            "n_points": int(np.sum(valid)),
            "rmse_m_yr": rmse,
            "bias_m_yr": bias,
            "correlation": corr,
            "our_v_mean": float(np.nanmean(our_v_interp[valid])),
            "hills_v_mean": float(np.nanmean(h_v[valid])),
        })

    return stats


# ============================================================================
# Plotting
# ============================================================================

def plot_comparison(our_results: list[dict], hills: dict[str, dict],
                    stats: list[dict], out_path: Path,
                    firn_max_m: float = 150.0):
    """Multi-panel comparison figure showing all sites."""
    n_sites = len(our_results)
    n_cols = min(n_sites, 10)
    n_rows = 2  # row 1: velocity profiles, row 2: strain rate

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 10))
    if n_cols == 1:
        axes = axes[:, None]

    cmap = plt.cm.tab10

    for i, r in enumerate(our_results):
        if i >= n_cols:
            break
        name = r["name"]
        h = hills.get(name)
        stat = next((s for s in stats if s["name"] == name), None)
        color = cmap(i / max(n_cols - 1, 1))
        matched = stat is not None and stat["correlation"] > 0.99

        # ---- Row 1: velocity profiles ----
        ax = axes[0, i]
        mask = r["range_m"] <= firn_max_m
        ax.plot(r["v_m_yr"][mask], r["range_m"][mask],
                lw=1.2, color=color, label="Ours")
        if h is not None:
            hmask = h["range_m"] <= firn_max_m
            ax.plot(h["v_m_yr"][hmask], h["range_m"][hmask],
                    lw=1.2, color="k", ls="--", alpha=0.7, label="Hills")

        title_color = "green" if matched else "red"
        r_str = f"r={stat['correlation']:.3f}" if stat else "N/A"
        ax.set_title(f"{name}\n{r_str}", fontsize=8, fontweight="bold",
                     color=title_color)
        ax.invert_yaxis()
        ax.axvline(0, color="grey", lw=0.5)
        ax.grid(True, ls="--", alpha=0.3)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.set_ylabel("Depth (m)", fontsize=8)
            ax.legend(fontsize=5, loc="lower left")
        ax.set_xlabel("VV (m/yr)", fontsize=6)

        # ---- Row 2: strain rate comparison ----
        ax = axes[1, i]
        sr = r["strain_rate_yr"][mask] * 100  # to %/yr
        ax.plot(sr, r["range_m"][mask],
                lw=1.2, color=color, label="Ours")
        if h is not None:
            hmask = h["range_m"] <= firn_max_m
            h_v = h["v_m_yr"][hmask].copy()
            h_r = h["range_m"][hmask]
            bad = ~np.isfinite(h_v) | (h["coherence"][hmask] < 0.3)
            h_v[bad] = np.nanmedian(h_v)
            h_smooth = gaussian_filter1d(h_v, sigma=25.0 / np.median(np.diff(h_r)))
            h_sr = np.gradient(h_smooth, h_r) * 100
            ax.plot(h_sr, h_r, lw=1.2, color="k", ls="--", alpha=0.7, label="Hills")
        ax.invert_yaxis()
        ax.axvline(0, color="grey", lw=0.5)
        ax.grid(True, ls="--", alpha=0.3)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.set_ylabel("Depth (m)", fontsize=8)
            ax.legend(fontsize=5)
        ax.set_xlabel("dVV/dz (%/yr)", fontsize=6)

    n_matched = sum(1 for s in stats if s["correlation"] > 0.99)
    fig.suptitle(
        "ApRES stacked-profile processing: Our derivation vs Hills et al. (2022)\n"
        f"South Pole, firn column 0–{firn_max_m:.0f} m  |  "
        f"Green = matched (r>0.99, {n_matched}/{len(stats)} sites), "
        f"Red = raw data differs",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_summary_scatter(our_results: list[dict], hills: dict[str, dict],
                         stats: list[dict], out_path: Path,
                         firn_max_m: float = 150.0):
    """Summary scatter and residual figure separating matched/unmatched sites."""
    # Classify sites
    matched_names = {s["name"] for s in stats if s["correlation"] > 0.99}

    matched_ours, matched_hills = [], []
    unmatched_ours, unmatched_hills = [], []

    for r in our_results:
        name = r["name"]
        if name not in hills:
            continue
        h = hills[name]
        hmask = (h["range_m"] <= firn_max_m) & (h["coherence"] > 0.5)
        h_v = h["v_m_yr"][hmask]
        our_v = np.interp(h["range_m"][hmask], r["range_m"], r["v_m_yr"],
                           left=np.nan, right=np.nan)
        valid = np.isfinite(our_v) & np.isfinite(h_v)
        if name in matched_names:
            matched_ours.extend(our_v[valid].tolist())
            matched_hills.extend(h_v[valid].tolist())
        else:
            unmatched_ours.extend(our_v[valid].tolist())
            unmatched_hills.extend(h_v[valid].tolist())

    matched_ours = np.array(matched_ours)
    matched_hills = np.array(matched_hills)
    unmatched_ours = np.array(unmatched_ours)
    unmatched_hills = np.array(unmatched_hills)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: 1:1 scatter (all)
    ax = axes[0]
    ax.scatter(matched_hills, matched_ours, s=8, alpha=0.6,
               color="green", label=f"Matched ({len(matched_names)} sites)", zorder=3)
    ax.scatter(unmatched_hills, unmatched_ours, s=8, alpha=0.4,
               color="red", label=f"Different stacking ({len(stats)-len(matched_names)} sites)",
               zorder=2)
    all_v = np.concatenate([matched_hills, matched_ours, unmatched_hills, unmatched_ours])
    lims = [np.nanmin(all_v), np.nanmax(all_v)]
    ax.plot(lims, lims, "k--", lw=0.8, label="1:1")
    ax.set_xlabel("Hills et al. VV (m/yr)")
    ax.set_ylabel("Our derived VV (m/yr)")
    ax.set_title("All sites", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_aspect("equal")

    # Panel B: matched-only scatter
    ax = axes[1]
    ax.scatter(matched_hills, matched_ours, s=8, alpha=0.6, color="green",
               zorder=3)
    lims_m = [matched_hills.min(), matched_hills.max()]
    ax.plot(lims_m, lims_m, "k--", lw=0.8)
    rmse_m = np.sqrt(np.mean((matched_ours - matched_hills)**2))
    bias_m = np.mean(matched_ours - matched_hills)
    corr_m = np.corrcoef(matched_ours, matched_hills)[0, 1]
    ax.text(0.05, 0.95,
            f"N = {len(matched_ours)}\n"
            f"RMSE = {rmse_m:.6f} m/yr\n"
            f"Bias = {bias_m:.6f} m/yr\n"
            f"r = {corr_m:.6f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.8))
    ax.set_xlabel("Hills et al. VV (m/yr)")
    ax.set_ylabel("Our derived VV (m/yr)")
    ax.set_title("Matched sites only (r > 0.99)", fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_aspect("equal")

    # Panel C: residual histogram (matched only)
    ax = axes[2]
    resid = matched_ours - matched_hills
    ax.hist(resid, bins=60, color="green", edgecolor="darkgreen", alpha=0.7)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(bias_m, color="red", lw=1.0, ls="--",
               label=f"bias = {bias_m:.6f} m/yr")
    ax.set_xlabel("Residual: Ours - Hills (m/yr)")
    ax.set_ylabel("Count")
    ax.set_title("Residual (matched sites)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    fig.suptitle(
        "ApRES vertical velocity validation: our stacked-profile derivation vs Hills et al. (2022)\n"
        f"Firn column 0–{firn_max_m:.0f} m, coherence > 0.5  |  "
        f"{len(matched_names)}/{len(stats)} sites match perfectly (same raw stacking)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def save_processed_csv(our_results: list[dict], out_path: Path):
    """Save processed velocities to CSV for downstream use."""
    rows = []
    for r in our_results:
        for i in range(len(r["range_m"])):
            rows.append({
                "site": r["name"],
                "range_m": r["range_m"][i],
                "coherence": r["coherence"][i],
                "phase_offset_rad": r["phase_offset_rad"][i],
                "v_m_yr": r["v_m_yr"][i],
                "v_smooth_m_yr": r["v_smooth_m_yr"][i],
                "v_unc_m_yr": r["v_unc_m_yr"][i],
                "strain_rate_yr": r["strain_rate_yr"][i],
                "dt_days": r["dt_days"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({len(df)} rows)")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("South Pole ApRES stacked-profile preprocessing")
    print("=" * 65)

    FIRN_MAX = 150.0
    SMOOTH_M = 25.0  # Gaussian smoothing sigma (m)

    # ------------------------------------------------------------------
    # 1. Process stacked profiles
    # ------------------------------------------------------------------
    print(f"\n--- Processing stacked profiles (smooth σ={SMOOTH_M} m) ---")
    our_results = process_all_sites(smooth_m=SMOOTH_M)
    print(f"  Processed {len(our_results)} sites")

    for r in our_results:
        mask = r["range_m"] <= FIRN_MAX
        v_firn = r["v_m_yr"][mask]
        v_valid = v_firn[np.isfinite(v_firn)]
        sr_mean = np.nanmean(r["strain_rate_yr"][mask]) * 100
        print(f"    {r['name']:12s}  dt={r['dt_days']:.1f}d  "
              f"v_mean={np.nanmean(v_valid):+.4f} m/yr  "
              f"sr_mean={sr_mean:+.4f} %/yr")

    # ------------------------------------------------------------------
    # 2. Load Hills' published velocities
    # ------------------------------------------------------------------
    print("\n--- Loading Hills et al. published velocities ---")
    hills = load_hills_all()
    print(f"  Loaded {len(hills)} sites")

    # ------------------------------------------------------------------
    # 3. Compare
    # ------------------------------------------------------------------
    print("\n--- Comparison statistics (firn 0–150 m, coherence > 0.5) ---")
    stats = compare_with_hills(our_results, hills, firn_max_m=FIRN_MAX)

    for s in stats:
        print(f"    {s['name']:12s}  "
              f"RMSE={s['rmse_m_yr']:.5f}  "
              f"bias={s['bias_m_yr']:+.5f}  "
              f"r={s['correlation']:.4f}  "
              f"n={s['n_points']}")

    if stats:
        mean_rmse = np.mean([s["rmse_m_yr"] for s in stats])
        mean_bias = np.mean([s["bias_m_yr"] for s in stats])
        mean_corr = np.mean([s["correlation"] for s in stats])
        print(f"\n  Overall:  RMSE={mean_rmse:.5f}  bias={mean_bias:+.5f}  r={mean_corr:.4f}")

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    print("\n--- Generating plots ---")
    plot_comparison(our_results, hills, stats,
                    OUT_DIR / "diag_apres_preprocess_comparison.png",
                    firn_max_m=FIRN_MAX)

    plot_summary_scatter(our_results, hills, stats,
                         OUT_DIR / "diag_apres_preprocess_scatter.png",
                         firn_max_m=FIRN_MAX)

    # ------------------------------------------------------------------
    # 5. Save processed data
    # ------------------------------------------------------------------
    print("\n--- Saving processed data ---")
    save_processed_csv(our_results,
                       _SOUTHPOLE / "processed" / "apres_vertical_velocity_processed.csv")

    print("\nDone.")
