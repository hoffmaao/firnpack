"""southpole_strain_apres_diagnostic.py

Diagnostic script that plots:
  1. USP50 string-potentiometer compaction / strain measurements
     (Stevens et al. 2023, USAP-DC 601680, 50 km upstream of South Pole)
     12 boreholes from 4.4 m to 106 m depth, Jan 2017 – Dec 2018

  2. USP50 in-situ firn temperatures
     (Stevens et al. 2023, USAP-DC 601525, same site, 0–40 m)

  3. ApRES vertical velocities — Hills et al. (2022), USAP-DC 601503
     10 sites 5–17 km from South Pole.  Files: x{dist}n/s{offset}_VV.txt
     Columns: Range(m), Coherence, PhaseOffset(rad),
              VerticalVelocity(m/yr), VelocityUncertainty(m/yr)
     Range 2–3998 m (full ice column).  Firn portion = 0–130 m.
     Strain rate = ∂w/∂z computed by finite-difference on VerticalVelocity.

Run:
    cd test/southpole/scripts
    python southpole_strain_apres_diagnostic.py

Outputs (saved to test/southpole/results/):
    diag_strain.png           USP50 compaction and strain-rate vs depth
    diag_firn_temperature.png USP50 firn temperature profile
    diag_apres.png            ApRES vertical velocity + strain-rate profiles,
                              overlaid with USP50 depth-integrated rates
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# Paths
# =============================================================================
_HERE      = Path(__file__).parent
_SOUTHPOLE = _HERE.parent
OUT_DIR    = _SOUTHPOLE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# USP50 compaction and density data (USAP-DC 601680)
USP50_DIR     = _SOUTHPOLE / "data/usap_dc/601680/usapdc_601680"
HOLE_LENGTHS  = USP50_DIR / "USP50_borehole_lengths.csv"
HOLE_SPECS    = USP50_DIR / "USP50_instrument_hole_specs.csv"

# USP50 firn temperatures (USAP-DC 601525)
FIRN_TEMP_DIR = _SOUTHPOLE / "data/usap_dc/601525"
FIRN_TEMP_CSV = FIRN_TEMP_DIR / "USP50_firn_temperatures_170109-181224.csv"

# ApRES vertical velocities (USAP-DC 601503, Hills et al. 2022)
# -------------------------------------------------------------------
# DOWNLOAD INSTRUCTIONS (requires manual step, site uses reCAPTCHA):
#   1. Go to https://www.usap-dc.org/view/dataset/601503
#   2. Download "vertical_velocities.zip"  (~513 kB)
#   3. Unzip into the directory below
# -------------------------------------------------------------------
APRES_DIR = _SOUTHPOLE / "data/usap_dc/601503"

# ERA5 forcing (for model temperature comparison)
ERA5_CSV   = _SOUTHPOLE / "processed/era5_monthly_point.csv"

# =============================================================================
# Colours (consistent with other diagnostic plots)
# =============================================================================
_DEPTH_CMAP = plt.cm.plasma

# =============================================================================
# Helpers
# =============================================================================

SPY = 365.25 * 24 * 3600   # seconds per year


def _depth_color(depth_m: float, depth_max: float = 110.0):
    return _DEPTH_CMAP(min(depth_m / depth_max, 1.0))


# =============================================================================
# 1. USP50 compaction / strain
# =============================================================================

def load_usp50_strain() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load borehole lengths and compute cumulative compaction, log-strain.

    Returns
    -------
    specs : pd.DataFrame  hole specifications (one row per instrument)
    compaction : pd.DataFrame  cumulative compaction (m) indexed by timestamp
    log_strain  : pd.DataFrame  log-strain (dimensionless) indexed by timestamp
    """
    specs = pd.read_csv(HOLE_SPECS)
    hole_df = pd.read_csv(HOLE_LENGTHS, index_col="TIMESTAMP", parse_dates=True)
    hole_df.replace(-9999, np.nan, inplace=True)

    init_lengths = specs["init_borehole_length"].values  # m

    compaction_df = pd.DataFrame(index=hole_df.index)
    logstrain_df  = pd.DataFrame(index=hole_df.index)

    for ii in range(len(specs)):
        pot   = ii + 1
        col   = f"hole_length{pot}"
        L     = hole_df[col].copy()
        L0    = float(init_lengths[ii])

        # Pot 10 had no data for most of 2017; treat transparently as NaN
        if pot == 10:
            # Find first valid index
            i_st = L.first_valid_index()
            if i_st is not None:
                L0_10 = float(L[i_st])
                comp  = L0_10 - L
                comp[comp.index < i_st] = np.nan
                strain = np.log(L / L0_10)
                strain[strain.index < i_st] = np.nan
            else:
                comp   = L * np.nan
                strain = L * np.nan
        else:
            comp   = L0 - L          # positive = shortening
            strain = np.log(L / L0)  # negative = compression (log strain)

        compaction_df[f"hole{pot}"] = comp
        logstrain_df[f"hole{pot}"]  = strain

    return specs, compaction_df, logstrain_df


def compute_annual_strain_rate(specs: pd.DataFrame,
                               logstrain_df: pd.DataFrame) -> np.ndarray:
    """Mean log-strain rate (yr⁻¹) over the full record for each borehole."""
    rates = np.full(len(specs), np.nan)
    for ii in range(len(specs)):
        pot = ii + 1
        col = f"hole{pot}"
        s   = logstrain_df[col].dropna()
        if len(s) < 10:
            continue
        dt_yr = (s.index[-1] - s.index[0]).total_seconds() / SPY
        if dt_yr > 0:
            rates[ii] = (s.iloc[-1] - s.iloc[0]) / dt_yr
    return rates


def plot_strain(specs, compaction_df, logstrain_df, out_path):
    """Three-panel strain diagnostic figure."""
    depths = specs["init_borehole_bottom"].values.astype(float)
    n_holes = len(specs)
    annual_rates = compute_annual_strain_rate(specs, logstrain_df)

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    # --- Panel A: cumulative compaction vs time ---
    ax = axes[0]
    for ii in range(n_holes):
        pot  = ii + 1
        col  = f"hole{pot}"
        data = compaction_df[col].dropna()
        if len(data) < 2:
            continue
        color = _depth_color(depths[ii])
        ax.plot(data.index, data.values * 1e3,   # → mm
                lw=0.9, color=color,
                label=f"Hole {pot} ({depths[ii]:.0f} m)")

    sm = plt.cm.ScalarMappable(cmap=_DEPTH_CMAP,
                               norm=plt.Normalize(0, depths.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Borehole depth (m)", fraction=0.04, pad=0.02)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative compaction (mm)")
    ax.set_title("USP50 cumulative compaction", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(True, ls="--", alpha=0.35)

    # --- Panel B: annual log-strain rate vs borehole bottom depth ---
    ax = axes[1]
    # Strain rate in units of 1/yr  (negative = compressive)
    valid = ~np.isnan(annual_rates)
    ax.scatter(annual_rates[valid] * 100,   # → %/yr for readability
               depths[valid],
               c=depths[valid], cmap=_DEPTH_CMAP,
               vmin=0, vmax=depths.max(),
               s=60, zorder=3)

    for ii in np.where(valid)[0]:
        ax.annotate(f"{ii+1}", (annual_rates[ii] * 100, depths[ii]),
                    fontsize=6, xytext=(3, 0), textcoords="offset points")

    ax.set_xlabel("Mean log-strain rate (% yr⁻¹)")
    ax.set_ylabel("Borehole bottom depth (m)")
    ax.invert_yaxis()
    ax.set_title("Depth-mean strain rate", fontsize=9)
    ax.axvline(0, color="k", lw=0.6)
    ax.grid(True, ls="--", alpha=0.35)

    # --- Panel C: seasonal compaction rate (monthly means across both years) ---
    ax = axes[2]
    # Use the two deepest holes for a clear seasonal signal
    deep_holes = np.argsort(depths)[-3:][::-1]   # 3 deepest
    for ii in deep_holes:
        pot = ii + 1
        col = f"hole{pot}"
        s   = logstrain_df[col].dropna()
        if len(s) < 30:
            continue
        # Monthly strain rate (1/yr) via finite difference, then average by month
        dt_s    = s.index.to_series().diff().dt.total_seconds()
        ds_dt   = s.diff() / dt_s * SPY      # strain rate, yr⁻¹
        ds_dt   = ds_dt.clip(-1e-2, 0)       # remove artifacts
        monthly = ds_dt.resample("ME").mean()
        # Average over the two years by day-of-year
        monthly.index = monthly.index.map(lambda t: t.month)
        avg_by_month = monthly.groupby(monthly.index).mean()

        color = _depth_color(depths[ii])
        ax.plot(avg_by_month.index,
                avg_by_month.values * 100,   # %/yr
                "o-", lw=1.2, ms=5, color=color,
                label=f"Hole {pot} ({depths[ii]:.0f} m)")

    ax.set_xlabel("Month")
    ax.set_ylabel("Strain rate (% yr⁻¹)")
    ax.set_title("Seasonal compaction cycle\n(3 deepest boreholes)", fontsize=9)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J",
                        "J", "A", "S", "O", "N", "D"])
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", alpha=0.35)

    fig.suptitle(
        "USP50 firn compaction measurements (Stevens et al. 2023)\n"
        "50 km upstream of South Pole, Jan 2017 – Dec 2018",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


# =============================================================================
# 2. USP50 firn temperatures
# =============================================================================

def load_usp50_firn_temperatures():
    """Return (depths_m, timestamps, T_data) from USP50_firn_temperatures CSV.

    The CSV is transposed: rows = thermistors, columns = timestamps.
    The last column 'depth' gives the thermistor installation depth (m).
    Returns
    -------
    depths  : (N,) float  thermistor depths in m
    times   : DatetimeIndex
    T_data  : (N, T) float  temperature in °C  (NaN for missing)
    """
    df = pd.read_csv(FIRN_TEMP_CSV, index_col=0)

    # Extract depth column (one entry per thermistor row)
    depths = df["depth"].values.astype(float)

    # The TIMESTAMP row holds the actual timestamps; drop it from data
    timestamps = pd.to_datetime(df.loc["TIMESTAMP"].drop("depth").values)

    therm_df = df.drop("TIMESTAMP")
    T_data   = therm_df.drop(columns=["depth"]).values.astype(float)
    T_data[T_data == -9999] = np.nan

    # depths[0] is -9999 (TIMESTAMP row placeholder); skip it
    depths  = depths[1:].astype(float)
    return depths, timestamps, T_data


def load_era5_for_temperature():
    """Return mean ERA5 annual T (K) at South Pole for spinup + 1940-2018."""
    if not ERA5_CSV.exists():
        return None
    df     = pd.read_csv(ERA5_CSV)
    t_rows = df[df["t2m_K"].notna()][["year", "t2m_K"]].copy()
    T_yr   = t_rows.groupby("year")["t2m_K"].mean()
    return T_yr


def plot_firn_temperatures(depths, timestamps, T_data, out_path):
    """Two-panel firn temperature figure."""
    # Time-mean and seasonal envelope
    T_mean = np.nanmean(T_data, axis=1)
    T_p10  = np.nanpercentile(T_data, 10, axis=1)
    T_p90  = np.nanpercentile(T_data, 90, axis=1)

    # Annual mean (12-month average centred on each month)
    era5_T = load_era5_for_temperature()

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    # --- Panel A: temperature-depth profile ---
    ax = axes[0]

    ax.fill_betweenx(depths, T_p10, T_p90,
                     color="steelblue", alpha=0.2,
                     label="P10–P90 range (seasonal)")
    ax.plot(T_mean, depths, "b-o", ms=4, lw=1.5,
            label="2-year mean (USP50, 2017–2018)")

    # Reference lines
    ax.axvline(-51.28, color="darkorange", ls="--", lw=1.0,
               label="Buizert 2021 pre-industrial (−51.28°C)")
    if era5_T is not None:
        era5_corrected_mean = float(era5_T.loc[era5_T.index >= 1940].mean()) - 273.15 - 4.71
        ax.axvline(era5_corrected_mean, color="green", ls=":", lw=1.0,
                   label=f"ERA5 1940–end mean (bias-corrected): {era5_corrected_mean:.1f}°C")

    # Label the deepest observed T
    T_deep = T_mean[depths >= 10].mean()
    ax.text(T_deep + 0.3, depths[depths >= 10].mean(),
            f"Mean ≥10 m: {T_deep:.1f}°C",
            fontsize=7, color="navy", va="center")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title("USP50 firn temperature profile\n(50 km upstream of South Pole, 2017–2018)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, ls="--", alpha=0.35)

    # --- Panel B: time series at selected depths ---
    ax = axes[1]
    selected_depths = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    colors = plt.cm.cool(np.linspace(0, 1, len(selected_depths)))

    for target_d, color in zip(selected_depths, colors):
        idx = np.argmin(np.abs(depths - target_d))
        actual_d = depths[idx]
        T_ts = T_data[idx, :]
        # 30-day rolling mean for clarity
        ts_series = pd.Series(T_ts, index=timestamps)
        ts_smooth = ts_series.rolling(window=120, center=True, min_periods=30).mean()
        ax.plot(timestamps, ts_smooth, lw=1.0, color=color,
                label=f"{actual_d:.1f} m")

    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Firn temperature time series\n(30-day smoothed)", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.legend(title="Depth", fontsize=7, loc="lower right")
    ax.grid(True, ls="--", alpha=0.35)

    fig.suptitle(
        "USP50 in-situ firn temperatures (Stevens et al. 2023, USAP-DC 601525)\n"
        "50 km upstream of South Pole, 0–40 m depth",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


# =============================================================================
# 3. ApRES vertical velocities (Hills et al. 2022, USAP-DC 601503)
# =============================================================================

# Coherence threshold: discard samples with low phase coherence
APRES_COH_MIN = 0.5

# Depth range to show for firn comparison (m)
APRES_FIRN_MAX = 150.0

# Gaussian smoothing sigma for VV profiles (metres)
APRES_SMOOTH_M = 25.0

# Site colour map
_SITE_CMAP = plt.cm.tab10


def load_all_apres_vv(vv_dir: Path,
                      coh_min: float = APRES_COH_MIN,
                      smooth_m: float = APRES_SMOOTH_M):
    """Load all *_VV.txt files.  Returns list of dicts with keys:
       name, range_m, v_m_yr, v_smooth, v_unc, coherence, strain_rate
    Samples with coherence < coh_min are masked to NaN before smoothing.
    Smoothing uses a Gaussian filter (sigma = smooth_m) applied after
    filling NaN gaps with the local median; this reveals the underlying
    firn-compaction signal that is hidden in individual 4-m range bins.
    """
    sites = []
    for f in sorted(vv_dir.glob("*_VV.txt")):
        d = np.loadtxt(f, skiprows=1)
        r   = d[:, 0]   # Range / depth (m)
        coh = d[:, 1]   # Coherence
        v   = d[:, 3].copy()   # VerticalVelocity (m/yr)
        unc = d[:, 4]   # Uncertainty

        # Mask low-coherence samples
        bad = coh < coh_min
        v[bad]   = np.nan
        unc[bad] = np.nan

        # Gaussian-smooth the velocity profile to expose the firn signal.
        # NaN bins are filled with the running median before smoothing so
        # they don't corrupt neighbours, then the NaN mask is restored.
        dr = float(r[1] - r[0]) if len(r) > 1 else 4.2
        sigma_bins = smooth_m / dr
        fill_val   = np.nanmedian(v)
        v_filled   = np.where(np.isnan(v), fill_val, v)
        v_smooth   = gaussian_filter1d(v_filled, sigma=sigma_bins)
        v_smooth[bad] = np.nan         # re-mask genuinely incoherent bins

        # Strain rate ∂w/∂z from the SMOOTHED profile (1/yr)
        sr = np.gradient(v_smooth, r)

        sites.append(dict(
            name        = f.stem,          # e.g. "x11n2_VV"
            range_m     = r,
            v_m_yr      = v,               # raw (individual 4-m bins)
            v_smooth    = v_smooth,        # Gaussian-smoothed
            v_unc       = unc,
            coherence   = coh,
            strain_rate = sr,              # from smoothed v, 1/yr
        ))
    return sites


def usp50_depth_integrated_rates(specs, logstrain_df):
    """Return (bottom_depth_m, strain_rate_yr, velocity_m_yr) for USP50 boreholes.

    The velocity at each borehole bottom is the total compaction rate of the
    borehole column (in m/yr):
        VV(z_bottom) ≈ strain_rate × init_borehole_length
    This is the rate at which the borehole bottom moves toward the surface,
    equivalent to the apparent-range change rate measured by ApRES.
    """
    depths  = specs["init_borehole_bottom"].values.astype(float)
    lengths = specs["init_borehole_length"].values.astype(float)
    rates   = compute_annual_strain_rate(specs, logstrain_df)
    valid   = ~np.isnan(rates)
    # velocity = strain_rate × length (negative = compaction, range decreasing)
    velocities = rates * lengths
    return depths[valid], rates[valid], velocities[valid]


def plot_apres(sites: list, specs, logstrain_df, out_path: Path):
    """Four-panel ApRES figure.

    Panel A: full-column vertical velocity (raw, all sites)
    Panel B: firn-column Gaussian-smoothed VV + USP50 velocity overlay
    Panel C: firn-column smoothed strain rate ∂VV/∂z + USP50 strain-rate overlay
    Panel D: coherence vs depth (firn column)
    """
    colors = [_SITE_CMAP(i / max(len(sites) - 1, 1)) for i in range(len(sites))]

    fig, axes = plt.subplots(1, 4, figsize=(20, 9))

    # USP50 comparison
    usp_depth, usp_rate, usp_vel = usp50_depth_integrated_rates(specs, logstrain_df)

    # --- Panel A: full column velocity (raw) ---
    ax = axes[0]
    for site, col in zip(sites, colors):
        ax.plot(site["v_m_yr"], site["range_m"], lw=0.6, color=col,
                label=site["name"].replace("_VV", ""))
    ax.set_xlabel("Vertical velocity (m yr⁻¹)")
    ax.set_ylabel("Range / Depth (m)")
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title("Full column\nvertical velocity (raw)", fontsize=9)
    ax.legend(fontsize=6, loc="lower left")
    ax.grid(True, ls="--", alpha=0.3)

    # --- Panel B: firn column — smoothed VV + USP50 velocity ---
    ax = axes[1]
    for site, col in zip(sites, colors):
        mask = site["range_m"] <= APRES_FIRN_MAX
        # Show raw as faint background
        ax.plot(site["v_m_yr"][mask], site["range_m"][mask],
                lw=0.5, color=col, alpha=0.25)
        # Show Gaussian-smoothed as bold foreground
        ax.plot(site["v_smooth"][mask], site["range_m"][mask],
                lw=1.8, color=col,
                label=site["name"].replace("_VV", ""))

    # USP50 velocity overlay: VV_borehole = strain_rate × init_length
    # (compaction of each borehole column, comparable to ApRES apparent-range rate)
    ax.scatter(usp_vel, usp_depth,
               s=70, zorder=5, color="k", marker="D",
               label="USP50 boreholes\n(ε̇ × L₀)")
    for d, v in zip(usp_depth, usp_vel):
        ax.annotate(f"{d:.0f} m", (v, d),
                    fontsize=5.5, xytext=(3, 0), textcoords="offset points")

    ax.set_xlabel("Vertical velocity (m yr⁻¹)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"Firn column (0–{APRES_FIRN_MAX:.0f} m)\n"
                 f"smoothed VV  (σ = {APRES_SMOOTH_M:.0f} m Gaussian)", fontsize=9)
    ax.legend(fontsize=6, loc="lower left")
    ax.grid(True, ls="--", alpha=0.3)

    # --- Panel C: firn strain rate from smoothed VV + USP50 ---
    ax = axes[2]
    for site, col in zip(sites, colors):
        mask = site["range_m"] <= APRES_FIRN_MAX
        sr_pct = site["strain_rate"][mask] * 100   # % yr⁻¹
        ax.plot(sr_pct, site["range_m"][mask],
                lw=1.4, color=col,
                label=site["name"].replace("_VV", ""))

    # USP50 borehole depth-averaged strain rates (scatter)
    ax.scatter(usp_rate * 100, usp_depth,
               s=70, zorder=5, color="k", marker="D",
               label="USP50 boreholes\n(depth-averaged ε̇)")
    for d, r in zip(usp_depth, usp_rate):
        ax.annotate(f"{d:.0f} m", (r * 100, d),
                    fontsize=5.5, xytext=(3, 0), textcoords="offset points")

    ax.set_xlabel("Strain rate (% yr⁻¹)  [∂VV/∂z]")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title("Firn strain rate ∂VV/∂z\n(smoothed ApRES + USP50)", fontsize=9)
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(True, ls="--", alpha=0.3)

    # --- Panel D: coherence ---
    ax = axes[3]
    for site, col in zip(sites, colors):
        mask = site["range_m"] <= APRES_FIRN_MAX
        ax.plot(site["coherence"][mask], site["range_m"][mask],
                lw=0.8, color=col,
                label=site["name"].replace("_VV", ""))
    ax.axvline(APRES_COH_MIN, color="r", ls="--", lw=0.8,
               label=f"threshold ({APRES_COH_MIN})")
    ax.set_xlabel("Coherence")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_title("Phase coherence\n(firn column)", fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(True, ls="--", alpha=0.3)

    fig.suptitle(
        "ApRES vertical velocities — Hills et al. (2022), USAP-DC 601503\n"
        f"10 sites, 5–17 km from South Pole  |  Dec 2018 – Dec 2019"
        f"  |  Firn VV smoothed with {APRES_SMOOTH_M:.0f}-m Gaussian\n"
        "USP50 (Stevens et al. 2023) overlaid: borehole VV (panel B) and ε̇ (panel C)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("South Pole strain + temperature + ApRES diagnostic")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. USP50 compaction / strain
    # ------------------------------------------------------------------
    print("\n--- USP50 compaction (USAP-DC 601680) ---")
    if HOLE_LENGTHS.exists() and HOLE_SPECS.exists():
        specs, compaction_df, logstrain_df = load_usp50_strain()
        depths = specs["init_borehole_bottom"].values.astype(float)
        annual_rates = compute_annual_strain_rate(specs, logstrain_df)
        print(f"  Loaded {len(specs)} boreholes,  "
              f"depths {depths.min():.1f}–{depths.max():.1f} m")
        print(f"  Date range: {compaction_df.index[0].date()} – "
              f"{compaction_df.index[-1].date()}")
        print("  Annual log-strain rates (pot : depth : rate %/yr):")
        for ii in range(len(specs)):
            r = annual_rates[ii]
            d = depths[ii]
            if not np.isnan(r):
                print(f"    Hole {ii+1:2d}: {d:6.1f} m  →  {r*100:+.4f} %/yr")
        plot_strain(specs, compaction_df, logstrain_df,
                    OUT_DIR / "diag_strain.png")
    else:
        print(f"  Data not found at {HOLE_LENGTHS}  —  skipping.")

    # ------------------------------------------------------------------
    # 2. USP50 firn temperatures
    # ------------------------------------------------------------------
    print("\n--- USP50 firn temperatures (USAP-DC 601525) ---")
    if FIRN_TEMP_CSV.exists():
        depths_T, timestamps_T, T_data = load_usp50_firn_temperatures()
        T_deep_mean = float(np.nanmean(T_data[depths_T >= 10, :]))
        print(f"  Loaded {len(depths_T)} thermistors, "
              f"depths {depths_T.min():.2f}–{depths_T.max():.1f} m")
        print(f"  Time range: {timestamps_T[0].date()} – {timestamps_T[-1].date()}")
        print(f"  Mean annual T (depth ≥ 10 m) = {T_deep_mean:.2f}°C")
        print(f"  (Buizert 2021 pre-industrial = -51.28°C)")
        plot_firn_temperatures(depths_T, timestamps_T, T_data,
                               OUT_DIR / "diag_firn_temperature.png")
    else:
        print(f"  Data not found at {FIRN_TEMP_CSV}  —  skipping.")

    # ------------------------------------------------------------------
    # 3. ApRES vertical velocities
    # ------------------------------------------------------------------
    print("\n--- ApRES vertical velocities (USAP-DC 601503) ---")
    vv_dir = APRES_DIR / "vertical_velocities" / "vertical_velocities"
    if not vv_dir.exists():
        vv_dir = APRES_DIR / "vertical_velocities"   # alternate unzip layout

    vv_files = sorted(vv_dir.glob("*_VV.txt")) if vv_dir.exists() else []

    if vv_files:
        sites = load_all_apres_vv(vv_dir)
        print(f"  Loaded {len(sites)} sites from {vv_dir}")
        for s in sites:
            r   = s["range_m"]
            v   = s["v_m_yr"]
            mask = r <= APRES_FIRN_MAX
            v_surf = np.nanmean(v[mask][:3])
            sr_mean = np.nanmean(s["strain_rate"][mask]) * 100
            print(f"    {s['name']:20s}  v_surf≈{v_surf:+.3f} m/yr  "
                  f"mean_sr(0–{APRES_FIRN_MAX:.0f}m)≈{sr_mean:+.4f} %/yr")
        plot_apres(sites, specs, logstrain_df, OUT_DIR / "diag_apres.png")
    else:
        print(f"  No *_VV.txt files found under {APRES_DIR}")
        print("  Unzip vertical_velocities.zip from USAP-DC 601503 there and re-run.")

    print("\nDone.")
