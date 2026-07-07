"""southpole_inversion_v2.py

MAP inversion for South Pole firn densification parameters using real observations
and time-varying ERA5 forcing.

Forward model : 1D firn column (FirnColumnSolver full-density mode)
Spinup        : 1200 years at Buizert 2021 reconstruction, RUN ON TAPE
Taped run     : ERA5 annual temperature (bias-corrected -4.71 K) + accumulation, 1940 → 2015
Temperature   : ERA5 has a ~5°C warm bias at South Pole (underestimated winter inversions).
                Corrected ERA5 and Buizert spinup are continuous at the 1940 boundary.

Key design choice — on-tape spinup:
  Without this, the deep density misfit (20–100m) dominates J but has near-zero gradient
  because density at those depths is set by centuries of compaction history that a 75-year
  taped run cannot change.  Putting the spinup on tape gives ∂J/∂kc meaningful signal.

Controls (log-space):
    kc0      – densification rate coefficient, low-density regime
    kc1      – densification rate coefficient, high-density regime
    kg       – grain-growth rate coefficient
    rho_surf – surface density boundary condition (kg/m³)
    r2_surf  – surface grain-radius² boundary condition (m²)

Observations (snapshots at ERA5_END_YEAR):
    rho(z)   – SP19 ice-core depth-density profile (sp19_density.csv)
    age(z)   – SP19 ice-core depth-age profile, firn layers (sp19_depth_age.csv)
    dR/dt(R) – ApRES apparent-range change rate from Hills et al. (2022, USAP-DC 601503);
               vertical velocity in travel-time coordinates, converted to model depth axis.

Changes from v1:
    - Replaces USP50 borehole strain rates (disabled, STRAIN_WEIGHT=0) with
      ApRES apparent-range change rate (dR/dt) as an active observation type.
    - ApRES forward operator: dR/dt = (n/n_ice)*w + CRIM_FACTOR*(ρw - ρ_s·w_s)
      where Term 2 is the density-change path-length correction derived from
      the continuity equation (Case et al. 2020; Kingslake et al. 2014).
    - Observation coordinates: ApRES data is on a range axis R; converted to
      model depth axis z using SP19 density profile (fixed, off-tape).

Output:
    <OUT_DIR>/map_v2.json  – MAP parameter estimates

Run:
    cd test/southpole/scripts
    OMP_NUM_THREADS=1 python southpole_inversion_v2.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
import firedrake as fd
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    pause_annotation,
    stop_annotating,
    get_working_tape,
)
from gadopt.inverse import MinimizationProblem, LinMoreOptimiser, minimisation_parameters

from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.constants import year as YEAR_S

# =============================================================================
# Paths
# =============================================================================
_HERE = Path(__file__).parent
_SOUTHPOLE = _HERE.parent
PROCESSED_DIR = _SOUTHPOLE / "processed"
ERA5_CSV       = PROCESSED_DIR / "era5_monthly_point.csv"
SP19_DENS_CSV  = PROCESSED_DIR / "sp19_density.csv"
SP19_AGE_CSV   = PROCESSED_DIR / "sp19_depth_age.csv"

# ApRES processed vertical velocities (Hills et al. 2022, USAP-DC 601503)
APRES_VV_CSV   = PROCESSED_DIR / "apres_vertical_velocity_processed.csv"

BUIZERT_TEMP_CSV  = PROCESSED_DIR / "buizert2021_spice_temp.csv"
BUIZERT_ACCUM_CSV = PROCESSED_DIR / "buizert2021_spice_accum.csv"

OUT_DIR = _SOUTHPOLE / "results"
OUT_PREFIX = "southpole_map_v2"

# =============================================================================
# Settings
# =============================================================================

# Mesh / geometry
H0 = 130.0        # column height (m) — covers all SP19 density data (0.51–127.53 m)
NZ = 100          # number of elements (slightly more to maintain resolution with deeper column)
STRETCH_P = 2.5   # power-law stretching exponent (refines near surface)
SURFACE_ID = 2    # Firedrake boundary marker for x = H0

# Time
# Spinup is run ON TAPE (controlled parameters, inside forward_mapping) so the
# adjoint propagates through the full densification / age history.
#
# Age constraint: the gradient ∂J_age/∂kc at depth z is non-zero ONLY if the
# age at z was computed on tape.  The SP19 observations reach ~1197 yr at 130 m
# (year_CE = 818; age = 2015 − 818 = 1197 yr in the model's convention).
# With a short spinup the age IC (computed off-tape from the SS formula) carries
# zero gradient, so the optimizer cannot improve the age fit below ~42 m.
# Setting age.assign(0.0) at the IC start and running ≥ max_observed_age years
# on tape ensures the full age profile — including the deep portion — is
# reachable by ∂J/∂kc.  In steady state the age at depth z converges to
#   mass_above(z) / (A_spin · rho_i)  ≈ 1197 yr at 130 m (SP19-implied A_spin)
# independent of how long the spinup runs, so SPINUP_YEARS = 1200 yr is sufficient.
SPINUP_YEARS    = 1200
SPINUP_DT_YEARS = 1.0
ERA5_START_YEAR = 1940    # first year included in ERA5 forcing
ERA5_END_YEAR   = 2015    # SP19 core year (final time for taped run)
DT_YEARS        = 1.0

# ERA5 temperature bias correction
# ERA5 has a well-documented ~5°C warm bias at South Pole due to underestimated
# boundary-layer inversions during polar night.  Three independent constraints:
#   Buizert 2021 borehole reconstruction at 1950 CE: -51.28°C
#   USP50 in-situ firn thermistors (2017-2018 mean): -50.1°C
#   ERA5 1940-1960 mean: -46.57°C  |  ERA5 2017-2018 mean: -44.3°C
# We apply a constant additive correction so that corrected ERA5 at 1940-1960
# matches the Buizert pre-industrial value, ensuring continuity with the spinup.
#   offset = Buizert_1950 - ERA5_1940_1960_mean = -51.28 - (-46.57) = -4.71°C
ERA5_T_OFFSET_K = -4.71   # K  (applied to every ERA5 annual-mean T value)

# ERA5 accumulation bias correction
# ERA5 underestimates net accumulation at South Pole (boundary-layer and wind
# redistribution effects).  We apply a constant additive correction so that the
# corrected ERA5 first year (1940) matches the Buizert 2021 reconstruction at
# the same year, ensuring continuity with the spinup:
#   Buizert (1940 CE, interpolated) : 0.09795 m ice eq/yr
#   ERA5 (1940 annual)              : 0.08169 m ice eq/yr
#   offset = Buizert_1940 - ERA5_1940 = 0.09795 - 0.08169 = +0.0163 m/yr
ERA5_A_OFFSET_M_YR = +0.0163   # m ice eq/yr (applied to every ERA5 annual A value)

# Spinup temperature: Buizert 2021 pre-industrial value at South Pole (1950 CE).
# Using ERA5 temperatures for the spinup would be ~5°C too warm.
SPINUP_T_K = 221.87        # K  (-51.28°C; Buizert 2021 reconstruction at 1950 CE)

# SP19 core year (2015 ice core)
CORE_YEAR = ERA5_END_YEAR

# Observation depth limits
# Use the full column for density (SP19 data goes to 127.53 m, well within H0=130 m).
RHO_OBS_DEPTH_MAX = H0

# For age we use ALL layers within the model column (depth ≤ H0).
AGE_OBS_YEAR_MIN = 0   # effectively no year lower-bound; depth_max=H0 is the only filter

# Observation uncertainties
SIGMA_RHO_ABS   = 15.0   # kg/m³ absolute
SIGMA_RHO_REL   = 0.03   # relative (fraction of local density)
SIGMA_AGE_ABS_S = 1.0 * float(YEAR_S)  # 1-year absolute uncertainty (in seconds)
SIGMA_AGE_REL   = 0.02   # relative age uncertainty

# Use depth-uniform sigma (avoids unintentional depth weighting)
USE_UNIFORM_OBS_SIGMA = True
OBS_SIGMA_SCALE = 1.0

# Observation operator: "kernel" (Gaussian integral, adjoint-friendly) or
# "vertex" (VertexOnlyMesh).
# NOTE: "vertex" is explicitly flagged in the synthetic twin-experiment as the
# cause of 1st-order Taylor remainder tests (broken gradients).  Always use
# "kernel" unless specifically debugging the vertex path.
OBS_MISFIT_METHOD = os.environ.get("FIRN_OBS_MISFIT", "kernel").lower()

# Half-width of the normalised Gaussian kernel (m) used in the "kernel" obs operator.
# ~2-4x the mesh node spacing near the surface so each kernel covers 2-4 elements.
OBS_KERNEL_SIGMA_M = 0.5

# ---- ApRES settings ----
# CRIM (Complex Refractive Index Model) constants for travel-time coordinates.
N_ICE       = 1.775       # refractive index of pure ice
RHO_ICE_CRIM = 917.0      # density of pure ice (kg/m³)
CRIM_FACTOR = (N_ICE - 1.0) / (N_ICE * RHO_ICE_CRIM)   # m³/kg

# ApRES site selection and filtering
APRES_SITE         = "x11n2"   # representative site near South Pole
APRES_DEPTH_MIN    = 8.0       # m — exclude near-surface bins (antenna coupling artifacts)
APRES_DEPTH_MAX    = 115.0     # m — upper firn portion
APRES_COHERENCE_MIN = 0.5      # minimum radar coherence for valid data
APRES_N_OBS        = 25        # number of uniformly-spaced observation depths in firn zone

# ApRES detrending: fit linear trend to deep-ice data to remove ice-sheet dynamics
# The firn model only predicts the densification anomaly, not the background
# ice-sheet vertical strain.  Detrending isolates the densification signal.
APRES_FIT_RANGE_MIN = 200.0    # m — start of deep-ice fitting window (range axis)
APRES_FIT_RANGE_MAX = 800.0    # m — end of deep-ice fitting window
APRES_BASE_DEPTH    = 115.0    # m — reference depth for model anomaly subtraction

# No manual weight — relative contribution is determined by observation
# uncertainties σ and observation count, as required for a proper MAP/Bayesian
# formulation where the Hessian of J is the inverse posterior covariance.

# ApRES observation uncertainties: multi-site standard deviation (depth-dependent).
# Computed from the spread of detrended dR/dt across all available sites.
# A floor prevents over-fitting in depth ranges where sites happen to agree closely.
SIGMA_APRES_FLOOR = 0.005  # m/yr floor (~0.5 cm/yr)

# Smooth density-coefficient switch (tanh blending around rho_m)
P0 = FirnParameters()
RHO_M = float(getattr(P0, "rho_m", 550.0))
RHO_SMOOTH = 20.0  # kg/m³ half-width for tanh

# Controls — starting point and bounds
# NOTE: For real data there is no "truth"; the TRUTH dict is used only to
# centre the log-space priors (weak regularisation).
TRUTH = dict(
    kc0     = float(getattr(P0, "kc0",    9.2e-9)),
    kc1     = float(getattr(P0, "kc1",    3.7e-9)),
    kg      = float(getattr(P0, "kg",     1.3e-7)),
    rho_surf= 350.0,   # kg/m³ — South Pole surface density (denser than generic 315)
    r2_surf = 1.0e-9,  # m² — South Pole fresh snow, r ≈ 32 μm (much smaller than
                       # generic 2.5e-7; small r² drives large near-surface ε̇ that
                       # decays with depth as grains coarsen via grain growth)
)

INITIAL = dict(TRUTH)   # start optimisation from FirnParameters defaults

BOUNDS = dict(
    kc0     = (1.0e-11, 1.0e-5),  # upper widened: SP19 may need much larger kc to reach 870 kg/m³
    kc1     = (1.0e-11, 1.0e-5),
    kg      = (1.0e-13, 1.0e-5),  # widened both ends
    rho_surf= (250.0,   450.0),
    # Lower bound for r2_surf is limited by numerical stability of the density solve.
    r2_surf = (1.0e-9,  1.0e-6),
)

# Priors (weak, centred on TRUTH)
USE_PRIOR            = True
PRIOR_SIGMA_LOG      = 1.5   # log-space std dev for kc0, kc1, kg
RHO_SURF_PRIOR_SIGMA_LOG = 0.25   # log-space std dev for rho_surf (~28% variation)
R2_SURF_PRIOR_SIGMA_LOG  = 0.8

# Optimiser
MAX_ITER = 60
GRAD_TOL = 1.0e-10

SEED = 0

# =============================================================================
# CRIM helper functions (numpy, off-tape coordinate conversion)
# =============================================================================

def crim_refractive_index(rho: np.ndarray) -> np.ndarray:
    """CRIM refractive index: n(rho) = 1 + (n_ice - 1) * rho / rho_ice."""
    return 1.0 + (N_ICE - 1.0) * np.asarray(rho) / RHO_ICE_CRIM


def depth_to_range(z: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Convert true depth z to apparent range R (ice-equivalent metres).

    R(z_R) = integral_0^{z_R}  n(rho(z)) / n_ice  dz
    """
    z = np.asarray(z)
    rho = np.asarray(rho)
    n = crim_refractive_index(rho)
    integrand = n / N_ICE
    R = np.zeros_like(z)
    R[1:] = cumulative_trapezoid(integrand, z)
    return R


def range_to_depth(R_target: np.ndarray, z: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Invert depth-to-range mapping: given R, find z."""
    R = depth_to_range(z, rho)
    return np.interp(R_target, R, z, left=np.nan, right=np.nan)


# =============================================================================
# Data loading
# =============================================================================

def load_era5_annual(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (years, T_annual_K, accum_annual_m_iceeq_yr) from ERA5 monthly CSV.

    The CSV stores temperature and accumulation on alternating rows (different
    6-hourly offsets within the same month). We identify each type by which
    column is non-null.
    """
    df = pd.read_csv(csv_path)

    t_rows  = df[df["t2m_K"].notna()][["year", "t2m_K"]].copy()
    a_rows  = df[df["net_accum_m_iceeq_month"].notna()][["year", "net_accum_m_iceeq_month"]].copy()

    T_by_yr = t_rows.groupby("year")["t2m_K"].mean()
    A_by_yr = a_rows.groupby("year")["net_accum_m_iceeq_month"].sum()   # m ice eq / yr

    years = np.array(sorted(T_by_yr.index.intersection(A_by_yr.index)), dtype=int)
    return years, T_by_yr.loc[years].values.astype(float), A_by_yr.loc[years].values.astype(float)


def load_sp19_density(
    csv_path: Path,
    depth_max: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (depth_m, rho_kgm3) from the processed SP19 density file.

    NOTE: The column 'rho_kgm3' is stored in g/cm³ (values ~0.30–0.83).
    Values are multiplied by 1000 here to convert to kg/m³.
    """
    df = pd.read_csv(csv_path)
    df = df[df["depth_m"] <= depth_max].copy()
    depth = df["depth_m"].values.astype(float)
    rho   = df["rho_kgm3"].values.astype(float) * 1000.0   # g/cm³ → kg/m³
    idx   = np.argsort(depth)
    return depth[idx], rho[idx]


def load_sp19_age_firn(
    csv_path: Path,
    core_year: int = CORE_YEAR,
    year_min:  int = AGE_OBS_YEAR_MIN,
    depth_max: float = H0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (depth_m, age_s) for firn layers deposited between year_min and core_year.

    The SP19 depth-age file contains annual-layer picks for the full SPICEcore
    (0–1750 m). We keep layers from year_min to core_year within depth_max.
    With the default year_min = AGE_OBS_YEAR_MIN = 0 and depth_max = H0 = 130 m,
    this covers all firn layers (ages 0–1197 yr, depths 0–130 m, N ≈ 1198 pts).

    age_s is the time since deposition at core_year, in seconds (matches the
    model's age field which has RHS source term = 1.0 s/s).
    """
    df = pd.read_csv(csv_path)
    # Keep layers within the temporal window AND within the model column depth
    mask = (df["year_CE"] >= year_min) & (df["year_CE"] <= core_year) & (df["depth_m"] <= depth_max)
    df   = df[mask].copy()

    depth  = df["depth_m"].values.astype(float)
    age_yr = float(core_year) - df["year_CE"].values.astype(float)
    age_s  = age_yr * float(YEAR_S)

    idx = np.argsort(depth)
    return depth[idx], age_s[idx]


def detrend_apres_multisite(
    csv_path: Path,
    target_site: str = APRES_SITE,
    depth_min: float = APRES_DEPTH_MIN,
    depth_max: float = APRES_DEPTH_MAX,
    n_obs: int = APRES_N_OBS,
    coherence_min: float = APRES_COHERENCE_MIN,
    fit_range_min: float = APRES_FIT_RANGE_MIN,
    fit_range_max: float = APRES_FIT_RANGE_MAX,
    sigma_floor: float = SIGMA_APRES_FLOOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ApRES data, detrend using deep-ice linear fit, return firn residuals.

    For each site:
      1. Fit linear trend to deep-ice dR/dt (range > fit_range_min)
      2. Subtract trend from firn-zone observations
      3. Interpolate onto a common depth grid

    Returns
    -------
    depth_grid : (n_obs,) uniformly-spaced depth values in the firn zone
    obs_detrended : (n_obs,) detrended dR/dt for target_site (m/yr)
    sigma_multisite : (n_obs,) multi-site standard deviation (m/yr), floored
    trend_at_obs : (n_obs,) deep-ice linear trend at observation ranges (m/yr)
    """
    df = pd.read_csv(csv_path)
    sites = sorted(df["site"].unique())

    # SP19 density for range → depth conversion
    sp19_depth, sp19_rho = load_sp19_density(SP19_DENS_CSV, depth_max=2.0 * H0)
    sp19_depth = np.concatenate([[0.0], sp19_depth])
    sp19_rho   = np.concatenate([[350.0], sp19_rho])

    # Common depth grid
    depth_grid = np.linspace(depth_min, depth_max, n_obs)

    # Detrend each site and interpolate onto the common grid
    all_detrended = {}
    all_trend_coeffs = {}

    for site in sites:
        ds = df[df["site"] == site]
        r_s = ds["range_m"].values.astype(float)
        v_s = ds["v_smooth_m_yr"].values.astype(float)
        c_s = ds["coherence"].values.astype(float)

        # Deep-ice linear fit (in range coordinates)
        deep = (r_s >= fit_range_min) & (r_s <= fit_range_max) & (c_s >= coherence_min)
        if deep.sum() < 10:
            continue
        coeffs = np.polyfit(r_s[deep], v_s[deep], 1)  # [slope, intercept]
        all_trend_coeffs[site] = coeffs

        v_detrended = v_s - np.polyval(coeffs, r_s)

        # Convert range → depth, interpolate onto common grid
        z_s = range_to_depth(r_s, sp19_depth, sp19_rho)
        ok = np.isfinite(z_s) & (z_s >= depth_min - 5) & (z_s <= depth_max + 5) & (c_s >= coherence_min)
        if ok.sum() > 5:
            v_interp = np.interp(depth_grid, z_s[ok], v_detrended[ok])
            all_detrended[site] = v_interp

    if target_site not in all_detrended:
        raise ValueError(f"Target site {target_site} not found or had insufficient data")

    # Multi-site standard deviation as depth-dependent uncertainty
    all_v = np.array(list(all_detrended.values()))  # (n_sites, n_obs)
    sigma_ms = np.std(all_v, axis=0)
    sigma_ms = np.maximum(sigma_ms, sigma_floor)

    # Target site's detrended signal
    obs_detrended = all_detrended[target_site]

    # Deep-ice trend evaluated at observation range values (for diagnostics)
    # Convert depth_grid → range using SP19
    R_grid = np.interp(depth_grid, sp19_depth, depth_to_range(sp19_depth, sp19_rho))
    trend_at_obs = np.polyval(all_trend_coeffs[target_site], R_grid)

    n_sites_used = len(all_detrended)
    print(f"  Detrending: {n_sites_used} sites used, fit range {fit_range_min}–{fit_range_max} m")
    print(f"  Target site {target_site} trend: slope={all_trend_coeffs[target_site][0]:.2e} m/yr/m, "
          f"intercept={all_trend_coeffs[target_site][1]*100:.2f} cm/yr")
    print(f"  Multi-site sigma: {sigma_ms.min()*100:.2f}–{sigma_ms.max()*100:.2f} cm/yr "
          f"(floor={sigma_floor*100:.2f} cm/yr)")

    return depth_grid, obs_detrended, sigma_ms, trend_at_obs


def compute_spinup_forcing(
    era5_years: np.ndarray,
    era5_A: np.ndarray,
    avg_start: int = ERA5_START_YEAR,
    avg_end:   int = 1960,
) -> tuple[float, float]:
    """Return (T_K, accum_era5_m_iceeq_yr) for reference only.

    Both values are diagnostic references — the actual spinup now uses the
    Buizert 2021 time-varying reconstruction (see load_buizert_spinup).
    """
    mask = (era5_years >= avg_start) & (era5_years <= avg_end)
    T_sp = float(SPINUP_T_K)
    A_sp = float(np.mean(era5_A[mask]))
    return T_sp, A_sp


def load_buizert_spinup(
    temp_csv: Path,
    accum_csv: Path,
    spinup_start_year_ce: float,
    era5_start_year: float = ERA5_START_YEAR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (years_ce, T_K, A_m_iceeq_yr) at annual resolution for the spinup.

    Loads the Buizert 2021 SPICEcore temperature and accumulation reconstructions
    (40-year resolution) and linearly interpolates to annual time steps covering
    [spinup_start_year_ce, era5_start_year).

    Temperature is converted from °C to K.
    Accumulation is in m ice equivalent per year (as archived).
    """
    df_T = pd.read_csv(temp_csv).sort_values("year_CE")
    df_A = pd.read_csv(accum_csv).sort_values("year_CE")

    years_annual = np.arange(float(spinup_start_year_ce), float(era5_start_year), 1.0)
    T_annual = np.interp(years_annual, df_T["year_CE"].values,
                         df_T["temp"].values + 273.15)   # °C → K
    A_annual = np.interp(years_annual, df_A["year_CE"].values,
                         df_A["accum"].values)            # m ice eq/yr

    return years_annual, T_annual, A_annual

# =============================================================================
# Firedrake / mesh helpers
# =============================================================================

def make_real(R: fd.FunctionSpace, value: float, name: str) -> fd.Function:
    f = fd.Function(R, name=name)
    f.dat.data[:] = float(value)
    return f


def control_scalar_value(f) -> float:
    """Read a scalar Real-space Function, preferring the ROL checkpoint if present."""
    if hasattr(f, "block_variable") and getattr(f.block_variable, "checkpoint", None) is not None:
        f = f.block_variable.checkpoint
    if hasattr(f, "dat"):
        return float(f.dat.data_ro[0])
    return float(f)


def _as_scalar(x) -> float:
    if hasattr(x, "block_variable") and getattr(x.block_variable, "checkpoint", None) is not None:
        x = x.block_variable.checkpoint
    if hasattr(x, "dat"):
        return float(x.dat.data_ro[0])
    return float(x)


def commit_checkpoint(f) -> None:
    if hasattr(f, "block_variable") and getattr(f.block_variable, "checkpoint", None) is not None:
        f.assign(f.block_variable.checkpoint)


def get_control_values(controls: list) -> np.ndarray:
    return np.array([control_scalar_value(c) for c in controls], dtype=float)


def set_control_values(controls: list, values) -> None:
    for c, v in zip(controls, values):
        c.assign(float(v))
        # Also update pyadjoint checkpoint so control_scalar_value reads the
        # correct value (it prefers block_variable.checkpoint over .dat).
        if hasattr(c, "block_variable") and getattr(c.block_variable, "checkpoint", None) is not None:
            c.block_variable.checkpoint.assign(float(v))


def build_stretched_mesh(H0: float, nz: int, p: float):
    mesh = fd.IntervalMesh(nz, 0.0, 1.0)
    xi   = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** p)
    coord_fs = mesh.coordinates.function_space()
    mesh.coordinates.assign(fd.Function(coord_fs).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    return mesh, V


def safe_vertex_coords(H0: float, depths_m: np.ndarray) -> np.ndarray:
    """Convert observation depths (from surface) to mesh x-coordinates."""
    d   = np.clip(np.asarray(depths_m, dtype=float), 0.0, H0)
    x   = H0 - d
    eps = 1.0e-8 * max(H0, 1.0)
    x   = np.where(x <= 0.0, eps,      x)
    x   = np.where(x >= H0,  H0 - eps, x)
    return np.vstack([x]).T


def w_surf_value(params, accum_m_iceeq_yr, rho_s):
    """Surface vertical velocity (m/s, negative = downward).

    Parameters
    ----------
    accum_m_iceeq_yr : m ice equivalent per year  (same units as synthetic test)
    rho_s            : surface density (kg/m³), may be a UFL expression
    """
    return -accum_m_iceeq_yr * params.rho_i / rho_s / float(YEAR_S)


def make_bcs(V, params, accum_fn, rho_s_expr, Hs_fn, surface_id=SURFACE_ID):
    """Build Dirichlet BCs for H, rho, w.

    rho_s_expr may be a UFL expression (e.g. exp(log_rho_surf) for the taped
    run) or a plain Real-space Function (for the off-tape spinup/diagnostics).
    Interpolating into a V-space Function makes the BC adjoint-safe.
    """
    bc_H = fd.DirichletBC(V, Hs_fn, surface_id)

    rho_surf_bc = fd.Function(V, name="rho_surf_bc")
    rho_surf_bc.interpolate(rho_s_expr)
    bc_rho = fd.DirichletBC(V, rho_surf_bc, surface_id)

    w_surf_bc = fd.Function(V, name="w_surf_bc")
    w_surf_bc.interpolate(w_surf_value(params, accum_fn, rho_s_expr))
    bc_w = fd.DirichletBC(V, w_surf_bc, surface_id)

    return [bc_H, bc_rho, bc_w], {"rho_surf_bc": rho_surf_bc, "w_surf_bc": w_surf_bc}

# =============================================================================
# Taylor remainder test (gradient check)
# =============================================================================

def taylor_test_multicontrol(rf, controls, bounds=None, seed=1,
                              epsilons=None):
    if epsilons is None:
        epsilons = [1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3]

    with stop_annotating():
        m0 = np.array([control_scalar_value(c) for c in controls], dtype=float)
        set_control_values(controls, m0)

    # Wrap rf calls in stop_annotating() to prevent pyadjoint from appending new
    # blocks to the tape during replay.
    with stop_annotating():
        J0 = float(rf(controls))
        g_list = rf.derivative()

    with stop_annotating():
        rng = np.random.default_rng(seed)
        d   = rng.standard_normal(len(m0))

        if bounds is not None:
            lb_ub = [(_as_scalar(lb), _as_scalar(ub)) for lb, ub in bounds]
            span  = np.array([max(u - l, 1e-14) for l, u in lb_ub])
            d     = d / (np.linalg.norm(d) + 1e-30)
            d     = d * 0.05 * span
            for i, (lbv, ubv) in enumerate(lb_ub):
                d[i] = min(d[i], 0.9 * (ubv - m0[i]) / (max(epsilons) + 1e-30))
                d[i] = max(d[i], 0.9 * (lbv - m0[i]) / (max(epsilons) + 1e-30))
        else:
            d = d / (np.linalg.norm(d) + 1e-30) * 1e-2 * (np.abs(m0) + 1.0)

        # L2 inner product for Real-space controls
        dx_ctrl = fd.dx(domain=controls[0].function_space().mesh())
        gdotd   = 0.0
        for gf, dv in zip(g_list, d):
            tmp = fd.Function(gf.function_space())
            tmp.assign(float(dv))
            gdotd += float(fd.assemble(gf * tmp * dx_ctrl))

    print("\nTaylor remainder test (multi-control):")
    print(f"  J(m)      = {J0:.6e}")
    print(f"  <∇J, d>   = {gdotd:.6e}")
    print("  eps        |J(m+eps d)-J(m)|     |remainder|          order")

    prev_r, prev_eps = None, None
    for eps in epsilons:
        m1 = m0 + eps * d
        if bounds is not None:
            for i, (lb, ub) in enumerate(bounds):
                m1[i] = min(max(m1[i], _as_scalar(lb)), _as_scalar(ub))
        with stop_annotating():
            set_control_values(controls, m1)
            J1 = float(rf(controls))
        r1 = abs(J1 - J0 - eps * gdotd)
        order = ""
        if prev_r is not None and r1 > 0 and prev_r > 0:
            order = f"{math.log(prev_r / r1) / math.log(prev_eps / eps):.2f}"
        print(f"  {eps:9.3e}  {abs(J1-J0):20.6e}  {r1:20.6e}  {order}")
        prev_r, prev_eps = r1, eps

    with stop_annotating():
        set_control_values(controls, m0)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("Loading ERA5 forcing ...")
    era5_years, era5_T, era5_A = load_era5_annual(ERA5_CSV)

    # Slice to the taped-run window [ERA5_START_YEAR, ERA5_END_YEAR)
    run_mask = (era5_years >= ERA5_START_YEAR) & (era5_years < ERA5_END_YEAR)
    run_years = era5_years[run_mask]
    run_T_raw = era5_T[run_mask]                     # ERA5 annual-mean T (K), uncorrected
    run_T     = run_T_raw + ERA5_T_OFFSET_K          # bias-corrected T (K)
    run_A_raw = era5_A[run_mask]                     # ERA5 annual-total A (m ice eq/yr), uncorrected
    run_A     = run_A_raw + ERA5_A_OFFSET_M_YR       # bias-corrected A
    n_steps   = len(run_years)
    dt_s      = float(DT_YEARS * YEAR_S)

    print(f"  ERA5 run window : {run_years[0]}–{run_years[-1]}  ({n_steps} steps)")
    print(f"  T range (raw)   : {run_T_raw.min():.2f}–{run_T_raw.max():.2f} K")
    print(f"  T range (corrected, offset={ERA5_T_OFFSET_K:+.2f} K): {run_T.min():.2f}–{run_T.max():.2f} K")
    print(f"  A range (raw)   : {run_A_raw.min():.4f}–{run_A_raw.max():.4f} m ice eq/yr")
    print(f"  A range (corrected, offset={ERA5_A_OFFSET_M_YR:+.4f} m/yr): {run_A.min():.4f}–{run_A.max():.4f} m ice eq/yr")

    _, A_spin_era5 = compute_spinup_forcing(era5_years, era5_A)

    print("Loading Buizert 2021 SPICEcore spinup reconstruction ...")
    _spinup_start_ce = ERA5_START_YEAR - SPINUP_YEARS
    spin_years, spin_T, spin_A = load_buizert_spinup(
        BUIZERT_TEMP_CSV, BUIZERT_ACCUM_CSV,
        spinup_start_year_ce=_spinup_start_ce,
        era5_start_year=ERA5_START_YEAR,
    )
    print(f"  Spinup period  : {spin_years[0]:.0f}–{spin_years[-1]:.0f} CE  "
          f"({len(spin_years)} annual steps)")
    print(f"  T range        : {spin_T.min()-273.15:.2f}–{spin_T.max()-273.15:.2f} °C"
          f"  (mean {np.mean(spin_T)-273.15:.2f} °C)")
    print(f"  A range        : {spin_A.min():.4f}–{spin_A.max():.4f} m ice eq/yr"
          f"  (mean {np.mean(spin_A):.4f} m ice eq/yr)")
    print(f"  A at ERA5 junction ({ERA5_START_YEAR} CE) : {spin_A[-1]:.4f} m ice eq/yr"
          f"  →  ERA5 1940–1960 mean : {A_spin_era5:.4f} m ice eq/yr"
          f"  (junction Δ = {spin_A[-1]-A_spin_era5:+.4f} m/yr)")

    print("Loading SP19 density observations ...")
    obs_rho_depth, obs_rho_val = load_sp19_density(SP19_DENS_CSV, RHO_OBS_DEPTH_MAX)
    print(f"  N_rho = {len(obs_rho_depth)}, depth range {obs_rho_depth[0]:.2f}–{obs_rho_depth[-1]:.2f} m")
    print(f"  rho range {obs_rho_val.min():.0f}–{obs_rho_val.max():.0f} kg/m³")

    print("Loading SP19 age observations (firn portion) ...")
    obs_age_depth, obs_age_s = load_sp19_age_firn(SP19_AGE_CSV)
    print(f"  N_age = {len(obs_age_depth)}, depth range {obs_age_depth[0]:.2f}–{obs_age_depth[-1]:.2f} m")
    print(f"  age range {obs_age_s.min()/float(YEAR_S):.1f}–{obs_age_s.max()/float(YEAR_S):.1f} yr")

    print(f"Loading and detrending ApRES vertical velocities (site={APRES_SITE}) ...")
    apres_depth, apres_detrended_myr, sigma_apres_myr, apres_trend_myr = \
        detrend_apres_multisite(APRES_VV_CSV)
    N_apres_raw = len(apres_depth)
    print(f"  N_apres = {N_apres_raw}, depth range {apres_depth[0]:.1f}–{apres_depth[-1]:.1f} m")
    print(f"  Detrended dR/dt range {apres_detrended_myr.min()*100:.3f}–"
          f"{apres_detrended_myr.max()*100:.3f} cm/yr")

    # Convert detrended observations to SI (m/s)
    apres_detrended_mps = apres_detrended_myr / float(YEAR_S)

    # -------------------------------------------------------------------------
    # Mesh + spaces
    # -------------------------------------------------------------------------
    mesh, V = build_stretched_mesh(H0, NZ, STRETCH_P)
    R       = fd.FunctionSpace(mesh, "R", 0)
    dx_domain = fd.dx(domain=mesh)

    # -------------------------------------------------------------------------
    # Observation uncertainties
    # -------------------------------------------------------------------------
    sigma_rho = SIGMA_RHO_ABS + SIGMA_RHO_REL * np.abs(obs_rho_val)
    sigma_age = SIGMA_AGE_ABS_S + SIGMA_AGE_REL * np.maximum(obs_age_s, 0.0)

    if USE_UNIFORM_OBS_SIGMA:
        sigma_rho = np.full_like(sigma_rho, float(np.median(sigma_rho)))
        sigma_age = np.full_like(sigma_age, float(np.median(sigma_age)))

    sigma_rho *= OBS_SIGMA_SCALE
    sigma_age *= OBS_SIGMA_SCALE

    # ApRES uncertainties: multi-site std (depth-dependent), already computed by detrending.
    # sigma_apres_myr comes directly from detrend_apres_multisite() with floor applied.
    sigma_apres_myr *= OBS_SIGMA_SCALE
    sigma_apres_mps = sigma_apres_myr / float(YEAR_S)

    print(f"  median sigma_rho = {np.median(sigma_rho):.2f} kg/m³")
    print(f"  median sigma_age = {np.median(sigma_age)/float(YEAR_S):.2f} yr")
    print(f"  median sigma_apres = {np.median(sigma_apres_myr)*100:.3f} cm/yr "
          f"(range {sigma_apres_myr.min()*100:.2f}–{sigma_apres_myr.max()*100:.2f} cm/yr)")

    # -------------------------------------------------------------------------
    # Observation meshes + Functions
    # -------------------------------------------------------------------------
    rho_xy = safe_vertex_coords(H0, obs_rho_depth)
    rho_vm = fd.VertexOnlyMesh(mesh, rho_xy, missing_points_behaviour="error")
    Qrho   = fd.FunctionSpace(rho_vm, "DG", 0)
    dx_rho = fd.dx(domain=rho_vm)
    N_rho  = float(Qrho.dim())

    rho_obs_fn = fd.Function(Qrho, name="rho_obs");  rho_obs_fn.dat.data[:] = obs_rho_val
    rho_sig_fn = fd.Function(Qrho, name="rho_sig");  rho_sig_fn.dat.data[:] = sigma_rho
    rho_model  = fd.Function(Qrho, name="rho_model")

    age_xy = safe_vertex_coords(H0, obs_age_depth)
    age_vm = fd.VertexOnlyMesh(mesh, age_xy, missing_points_behaviour="error")
    Qage   = fd.FunctionSpace(age_vm, "DG", 0)
    dx_age = fd.dx(domain=age_vm)
    N_age  = float(Qage.dim())

    age_obs_fn = fd.Function(Qage, name="age_obs");  age_obs_fn.dat.data[:] = obs_age_s
    age_sig_fn = fd.Function(Qage, name="age_sig");  age_sig_fn.dat.data[:] = sigma_age
    age_model  = fd.Function(Qage, name="age_model")

    # -------------------------------------------------------------------------
    # Kernel observation operator (adjoint-friendly alternative to VertexOnlyMesh)
    # -------------------------------------------------------------------------
    # Precompute normalised Gaussian kernel Functions phi_i on the main mesh.
    # phi_i approximates delta(x - x_i) with ∫ phi_i dx = 1, so that
    #   fd.assemble(field * phi_i * dx_domain) ≈ field(x_i).
    # This uses only standard Firedrake assemble operations (no VertexOnlyMesh),
    # which are adjoint-consistent and give Taylor remainder order ~2.
    #
    # IMPORTANT: obs/sigma stored as plain Python floats (not fd.Constant) so
    # that pyadjoint AdjFloat scalar accumulation works correctly; mixing
    # AdjFloat with UFL Constant breaks the ReducedFunctional.
    rho_obs_vals: list = []
    rho_sig_vals: list = []
    age_obs_vals: list = []
    age_sig_vals: list = []
    rho_kernels:  list = []
    age_kernels:  list = []

    # ApRES kernel observation operator lists
    apres_obs_vals: list = []   # detrended dR/dt in m/s
    apres_sig_vals: list = []   # sigma in m/s
    apres_kernels:  list = []   # Gaussian kernels at observation depths
    apres_base_kernel = None    # kernel at base depth for model anomaly subtraction
    N_apres = float(N_apres_raw)

    if OBS_MISFIT_METHOD == "kernel":
        with stop_annotating():
            _x, = fd.SpatialCoordinate(mesh)
            # Mesh x-coordinates of observation points (x = H0 - depth)
            _x_rho = [float(xy[0]) for xy in rho_xy]
            _x_age = [float(xy[0]) for xy in age_xy]

            rho_obs_vals = [float(v) for v in obs_rho_val]
            rho_sig_vals = [float(v) for v in sigma_rho]
            age_obs_vals = [float(v) for v in obs_age_s]
            age_sig_vals = [float(v) for v in sigma_age]

            def _make_kernels(x_points, sigma_m, tag):
                kernels = []
                for j, xp in enumerate(x_points):
                    wgt  = fd.exp(-0.5 * ((_x - xp) / float(sigma_m)) ** 2)
                    denom = float(fd.assemble(wgt * dx_domain)) + 1.0e-30
                    phi  = fd.Function(V, name=f"phi_{tag}_{j:04d}")
                    phi.interpolate(wgt / denom)
                    kernels.append(phi)
                return kernels

            rho_kernels = _make_kernels(_x_rho, OBS_KERNEL_SIGMA_M, "rho")
            age_kernels = _make_kernels(_x_age, OBS_KERNEL_SIGMA_M, "age")

            # ApRES kernels at observation depths (same Gaussian kernel pattern)
            _apres_xy = safe_vertex_coords(H0, apres_depth)
            _x_apres  = [float(xy[0]) for xy in _apres_xy]
            apres_obs_vals = [float(v) for v in apres_detrended_mps]
            apres_sig_vals = [float(v) for v in sigma_apres_mps]
            apres_kernels  = _make_kernels(_x_apres, OBS_KERNEL_SIGMA_M, "apres")

            # Base kernel at reference depth for model anomaly subtraction.
            # Model anomaly = dR/dt(z) - dR/dt(z_base), matching detrended obs → 0 at base.
            _base_xy = safe_vertex_coords(H0, np.array([APRES_BASE_DEPTH]))
            _x_base  = [float(xy[0]) for xy in _base_xy]
            apres_base_kernel = _make_kernels(_x_base, OBS_KERNEL_SIGMA_M, "apres_base")[0]

            print(f"  Kernel obs operator: {len(rho_kernels)} rho kernels, "
                  f"{len(age_kernels)} age kernels, "
                  f"{len(apres_kernels)} apres kernels + 1 base kernel "
                  f"(sigma={OBS_KERNEL_SIGMA_M} m)")

    # Pre-allocate scratch Functions used in the objective.
    # Allocated off-tape so the allocation is not a tape block;
    # only the interpolate calls inside forward_mapping() are tracked.
    with stop_annotating():
        # dR/dt field for ApRES forward operator
        _dRdt_fn = fd.Function(V, name="dRdt_apres")
        # Prior scratch function
        _prior_fn = fd.Function(V, name="prior_scratch")

    # -------------------------------------------------------------------------
    # Density IC from observed SP19 profile
    # -------------------------------------------------------------------------
    # Using the observed density profile as the IC (rather than a generic
    # exponential) is essential when observations extend below ~30 m.  The
    # exponential rho0+(rho_i-rho0)*(1-exp(-z/15)) gives ~900 kg/m³ at 50 m,
    # but SP19 shows only ~730 kg/m³ there.  That 170 kg/m³ bias cannot be
    # corrected by kc optimisation in a 275-year run, so it would corrupt the
    # gradient and prevent convergence on the deep density observations.
    #
    # The IC is created BEFORE continue_annotation() so it is never on the tape
    # and has zero gradient.  Inside forward_mapping(), rho.assign(rho_ic_fn)
    # resets density to the observed profile on every tape replay.
    _ic_depth_raw, _ic_rho_raw = load_sp19_density(SP19_DENS_CSV, depth_max=2.0 * H0)
    # Prepend the surface point (depth=0, rho=INITIAL["rho_surf"])
    _ic_depth = np.concatenate([[0.0], _ic_depth_raw])
    _ic_rho   = np.concatenate([[float(INITIAL["rho_surf"])], _ic_rho_raw])
    # Mesh coordinates (physical, 0 = bottom, H0 = surface)
    _mesh_x_flat = (mesh.coordinates.dat.data_ro[:, 0]
                    if mesh.coordinates.dat.data_ro.ndim == 2
                    else mesh.coordinates.dat.data_ro)
    _mesh_depth  = H0 - _mesh_x_flat
    # Linear interpolation; extrapolate conservatively beyond last data point
    # (data ends at ~127.5 m, mesh bottom is at H0=130 m — only 2.5 m gap).
    _rho_at_nodes = np.interp(_mesh_depth, _ic_depth, _ic_rho,
                              left=float(INITIAL["rho_surf"]),
                              right=float(_ic_rho[-1]))
    _rho_at_nodes = np.clip(_rho_at_nodes, 200.0, float(P0.rho_i) - 0.5)

    # -------------------------------------------------------------------------
    # State Functions (allocated once; reset inside forward_mapping each call)
    # The spinup is run ON TAPE inside forward_mapping() using the controlled
    # parameters, so the adjoint propagates through the full compaction history.
    # -------------------------------------------------------------------------
    H     = fd.Function(V, name="H")
    rho   = fd.Function(V, name="rho")
    w     = fd.Function(V, name="w")
    sigma = fd.Function(V, name="sigma")
    r2    = fd.Function(V, name="r2")
    age   = fd.Function(V, name="age")

    rho_ic_fn = fd.Function(V, name="rho_ic")
    rho_ic_fn.dat.data[:] = _rho_at_nodes

    # -------------------------------------------------------------------------
    # Grain-radius-squared IC: smooth steady-state linear profile
    # -------------------------------------------------------------------------
    _T_spin_mean = float(np.mean(spin_T))
    _dr2dt_spin  = float(INITIAL["kg"]) * np.exp(
        -float(P0.Eg) / (float(P0.R) * _T_spin_mean)
    )   # m² s⁻¹
    _w_surf_mean = float(np.mean(spin_A)) * float(P0.rho_i) / float(INITIAL["rho_surf"]) / float(YEAR_S)
    _dr2dz_spin  = _dr2dt_spin / abs(_w_surf_mean)          # m² m⁻¹
    _r2_ic_nodes = float(INITIAL["r2_surf"]) + _dr2dz_spin * _mesh_depth
    _r2_ic_nodes = np.clip(_r2_ic_nodes, float(INITIAL["r2_surf"]), 2.5e-7)
    r2_ic_fn = fd.Function(V, name="r2_ic")
    r2_ic_fn.dat.data[:] = _r2_ic_nodes

    print(f"\nSpinup ({SPINUP_YEARS} yr) will run ON TAPE inside forward_mapping().")
    print(f"  Density IC : interpolated from SP19 obs profile "
          f"({_ic_depth[1]:.2f}–{_ic_depth[-1]:.2f} m, {len(_ic_depth_raw)} points)")
    print(f"  r2 IC      : steady-state linear  {float(INITIAL['r2_surf']):.1e} m² (surface) → "
          f"{_r2_ic_nodes.max():.2e} m² (base),  dr2/dz = {_dr2dz_spin:.2e} m²/m")
    print(f"  Age IC     : zero (age=0 everywhere at spinup start).")
    _A_spin_mean = float(np.mean(spin_A))
    _age_ss_est = float(np.trapz(_rho_at_nodes, _mesh_depth)) / (_A_spin_mean * float(P0.rho_i))
    print(f"  Age SS est : mass_col/(A_mean·rho_i) = {_age_ss_est:.0f} yr"
          f"  (mean Buizert A = {_A_spin_mean:.4f} m/yr;"
          f"  spinup {SPINUP_YEARS} yr {'≥' if SPINUP_YEARS >= _age_ss_est else '<'} this,"
          f" age profile fully on tape)")

    # -------------------------------------------------------------------------
    # Controls (log-space)
    # -------------------------------------------------------------------------
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()

    def _add_log_ctrl(name, init_val):
        m  = make_real(R, math.log(float(init_val)), f"log_{name}")
        lb = make_real(R, math.log(float(BOUNDS[name][0])), f"log_{name}_lb")
        ub = make_real(R, math.log(float(BOUNDS[name][1])), f"log_{name}_ub")
        return m, lb, ub

    log_kc0,    lb_kc0,    ub_kc0    = _add_log_ctrl("kc0",     INITIAL["kc0"])
    log_kc1,    lb_kc1,    ub_kc1    = _add_log_ctrl("kc1",     INITIAL["kc1"])
    log_kg,     lb_kg,     ub_kg     = _add_log_ctrl("kg",      INITIAL["kg"])
    log_rho_s,  lb_rho_s,  ub_rho_s  = _add_log_ctrl("rho_surf",INITIAL["rho_surf"])
    log_r2_s,   lb_r2_s,   ub_r2_s   = _add_log_ctrl("r2_surf", INITIAL["r2_surf"])

    controls = [log_kc0, log_kc1, log_kg, log_rho_s, log_r2_s]
    names    = ["kc0", "kc1", "kg", "rho_surf", "r2_surf"]
    lbs      = [lb_kc0, lb_kc1, lb_kg, lb_rho_s, lb_r2_s]
    ubs      = [ub_kc0, ub_kc1, ub_kg, ub_rho_s, ub_r2_s]
    bounds   = list(zip(lbs, ubs))

    kc0_expr      = fd.exp(log_kc0)
    kc1_expr      = fd.exp(log_kc1)
    kg_expr       = fd.exp(log_kg)
    rho_surf_expr = fd.exp(log_rho_s)
    r2_surf_expr  = fd.exp(log_r2_s)

    # -------------------------------------------------------------------------
    # Build model + solver with controlled kg
    # -------------------------------------------------------------------------
    params = FirnParameters(kg=kg_expr)
    model  = FirnModel(params)
    solver = FirnColumnSolver(model)

    # Forcing Functions (updated each timestep inside forward_mapping; not controls).
    # accum is in m ice eq / yr — same convention as the synthetic test.
    Ts    = make_real(R, run_T[0], "Ts")
    accum = make_real(R, run_A[0], "accum")   # m ice eq / yr
    Hs    = make_real(R, float(params.c_i) * (run_T[0] - float(params.T_ref)), "Hs")
    dt    = make_real(R, dt_s, "dt")

    # rho_surf_for_solver: a Real-space Function that shadows rho_surf_expr.
    rho_surf_fs = make_real(R, INITIAL["rho_surf"], "rho_surf_fs")
    rho_surf_fs.interpolate(rho_surf_expr)

    # BCs (rho_surf and r2_surf are controls; accum in m ice eq/yr)
    bcs, bc_vals = make_bcs(V, params, accum, rho_surf_expr, Hs)
    rho_surf_bc  = bc_vals["rho_surf_bc"]
    w_surf_bc    = bc_vals["w_surf_bc"]
    bc_sigma     = fd.DirichletBC(V, make_real(R, 0.0, "sig0"), SURFACE_ID)
    r2_surf_bc   = fd.Function(V, name="r2_surf_bc")
    r2_surf_bc.interpolate(r2_surf_expr)
    bc_r2        = fd.DirichletBC(V, r2_surf_bc, SURFACE_ID)
    bc_age       = fd.DirichletBC(V, make_real(R, 0.0, "age0"), SURFACE_ID)
    rhoCoef      = fd.Function(V, name="rhoCoef")

    # Domain length (for normalising prior terms)
    with stop_annotating():
        domain_len = float(fd.assemble(fd.Constant(1.0) * dx_domain)) + 1e-30

    # Objective component handles (captured on first tape build)
    J_PARTS = {"rho": None, "age": None, "apres": None, "prior": None}

    # -------------------------------------------------------------------------
    # Forward mapping  (taped, time-varying ERA5 forcing)
    # -------------------------------------------------------------------------
    def forward_mapping():
        """Pure mapping: controls → objective.

        Runs the full forward model on tape:
          1. Fixed initial condition (ON TAPE, no control dependency).
          2. SPINUP_YEARS of constant-forcing spin-up (ON TAPE, controlled params).
          3. ERA5_START_YEAR → ERA5_END_YEAR time-varying run (ON TAPE).
          Zero stop_annotating() calls inside — all forcing assignments are on the
          tape so rf(controls) replay correctly restores Ts/Hs/accum at every step.

        By putting the spinup on tape the adjoint propagates through the full
        compaction history, giving non-zero ∂J/∂kc at intermediate and deep depths.
        """
        # --- Refresh control-dependent BCs (must be inside the mapping) ---
        rho_surf_fs.interpolate(rho_surf_expr)
        rho_surf_bc.interpolate(rho_surf_expr)
        r2_surf_bc.interpolate(r2_surf_expr)

        # --- Fixed initial condition (ON TAPE, but IC doesn't depend on controls) ---
        rho.assign(rho_ic_fn)
        H.assign(float(params.c_i) * (float(spin_T[0]) - float(params.T_ref)))
        sigma.assign(0.0)
        r2.assign(r2_ic_fn)
        w.assign(0.0)
        age.assign(0.0)   # age=0 everywhere; spinup ≥ max_obs_age puts full profile on tape

        # --- Spinup (ON TAPE, time-varying Buizert 2021 reconstruction) ---
        n_spin = int(round(SPINUP_YEARS / SPINUP_DT_YEARS))
        for _k in range(n_spin):
            T_sk = float(spin_T[_k])
            A_sk = float(spin_A[_k])
            Ts.assign(T_sk)
            Hs.assign(float(params.c_i) * (T_sk - float(params.T_ref)))
            accum.assign(A_sk)
            # w_surf annotated: captures ∂J/∂rho_surf
            w_surf_bc.interpolate(w_surf_value(params, accum, rho_surf_expr))

            s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
            rhoCoef.interpolate((1.0 - s) * kc0_expr + s * kc1_expr)

            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                stress=sigma, grain_radius2=r2,
                stress_boundary_condition=bc_sigma,
                grain_radius2_boundary_condition=bc_r2,
                rhoCoef=rhoCoef,
                age=age, age_boundary_condition=bc_age,
            )

        # --- Time-stepping loop (ERA5_START_YEAR → ERA5_END_YEAR) ---
        for k in range(n_steps):
            T_k = float(run_T[k])
            A_k = float(run_A[k])   # m ice eq / yr
            Ts.assign(T_k)
            Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)

            # w_surf_bc IS annotated: captures d(J)/d(rho_surf) through the BC.
            w_surf_bc.interpolate(w_surf_value(params, accum, rho_surf_expr))

            # Density-dependent kc blending
            s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
            rhoCoef.interpolate((1.0 - s) * kc0_expr + s * kc1_expr)

            # Advance state by one year (in-place update)
            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                stress=sigma, grain_radius2=r2,
                stress_boundary_condition=bc_sigma,
                grain_radius2_boundary_condition=bc_r2,
                rhoCoef=rhoCoef,
                age=age, age_boundary_condition=bc_age,
            )

        # --- ApRES dR/dt forward operator ---
        # Compute model-predicted apparent-range change rate at the final state.
        #
        # dR/dt(x) = [n(rho)/n_ice] * w  +  CRIM_FACTOR * [rho*w - rho_s*w_s]
        #            ---- Term 1 ----        ----------- Term 2 -----------
        #
        # Term 1: Physical displacement scaled by local refractive index ratio.
        # Term 2: Density-change path-length correction (Case et al. 2020).
        #         Derived from continuity (∂ρ/∂t = -∂(ρw)/∂x):
        #           ∫_0^z ∂ρ/∂t dz' = ρ(x)·w(x) - ρ_surf·w_surf
        #         This avoids computing a cumulative integral on the tape.
        #
        # Surface mass flux: ρ_surf·w_surf = -accum·ρ_i/YEAR_S
        # (independent of rho_surf — it cancels in ρ_s * (-A·ρ_i/ρ_s/YEAR_S))
        _rho_w_surf = -accum * float(params.rho_i) / float(YEAR_S)

        # n(rho)/n_ice ratio
        _n_ratio = (1.0 + (N_ICE - 1.0) * rho / RHO_ICE_CRIM) / N_ICE

        # Combined dR/dt expression
        _dRdt_expr = _n_ratio * w + CRIM_FACTOR * (rho * w - _rho_w_surf)

        # Interpolate into Function BEFORE assembling (adjoint-safe pattern:
        # InterpolateBlock for nonlinear UFL, then linear AssembleBlock).
        _dRdt_fn.interpolate(_dRdt_expr)

        # Compute ApRES misfit (detrended comparison)
        # Model anomaly = dR/dt(z) - dR/dt(z_base), so both model and
        # detrended observations approach 0 at the firn-ice transition.
        _dRdt_base = fd.assemble(_dRdt_fn * apres_base_kernel * dx_domain)

        J_apres = 0.0
        for _phi_a, _obs_v, _sig_v in zip(apres_kernels, apres_obs_vals, apres_sig_vals):
            _pred_v  = fd.assemble(_dRdt_fn * _phi_a * dx_domain) - _dRdt_base
            _r_apres = (_pred_v - _obs_v) / _sig_v
            J_apres += 0.5 * _r_apres * _r_apres

        # --- Final-time misfits ---
        # "kernel": adjoint-friendly Gaussian-integral approximation of point obs.
        # "vertex": VertexOnlyMesh point sampling (often causes 1st-order Taylor
        #           remainder / broken gradient — kept only for debugging).
        if OBS_MISFIT_METHOD == "kernel":
            J_rho = 0.0
            for phi, obs_c, sig_c in zip(rho_kernels, rho_obs_vals, rho_sig_vals):
                pred   = fd.assemble(rho * phi * dx_domain)
                r      = (pred - obs_c) / sig_c
                J_rho += 0.5 * r * r

            J_age = 0.0
            for phi, obs_c, sig_c in zip(age_kernels, age_obs_vals, age_sig_vals):
                pred   = fd.assemble(age * phi * dx_domain)
                r      = (pred - obs_c) / sig_c
                J_age += 0.5 * r * r
        else:
            rho_model.interpolate(rho)
            mis_rho = (rho_model - rho_obs_fn) / rho_sig_fn
            J_rho   = 0.5 * fd.assemble(mis_rho ** 2 * dx_rho)

            age_model.interpolate(age)
            mis_age = (age_model - age_obs_fn) / age_sig_fn
            J_age   = 0.5 * fd.assemble(mis_age ** 2 * dx_age)

        J = J_rho + J_age + J_apres

        # --- Weak log-space priors ---
        J_prior = 0.0
        if USE_PRIOR:
            def _prior_term(log_ctrl, truth_val, sigma_log):
                _prior_fn.interpolate(
                    (log_ctrl - math.log(float(truth_val))) / sigma_log
                )
                return 0.5 * fd.assemble(_prior_fn * _prior_fn * dx_domain) / domain_len

            J_prior = (
                _prior_term(log_kc0,   TRUTH["kc0"],     PRIOR_SIGMA_LOG)
                + _prior_term(log_kc1, TRUTH["kc1"],     PRIOR_SIGMA_LOG)
                + _prior_term(log_kg,  TRUTH["kg"],      PRIOR_SIGMA_LOG)
                + _prior_term(log_rho_s, TRUTH["rho_surf"], RHO_SURF_PRIOR_SIGMA_LOG)
                + _prior_term(log_r2_s,  TRUTH["r2_surf"],  R2_SURF_PRIOR_SIGMA_LOG)
            )
            J = J + J_prior

        if J_PARTS["rho"] is None:
            J_PARTS["rho"]   = J_rho
            J_PARTS["age"]   = J_age
            J_PARTS["apres"] = J_apres
            J_PARTS["prior"] = J_prior

        return J

    # Build the tape
    J  = forward_mapping()
    rf = ReducedFunctional(J, [Control(c) for c in controls])
    pause_annotation()

    print(f"\nInitial objective J(m0) = {float(J):.6e}")
    print(f"  J_rho    = {float(J_PARTS['rho']):.6e}")
    print(f"  J_age    = {float(J_PARTS['age']):.6e}")
    print(f"  J_apres  = {float(J_PARTS['apres']):.6e}  (N={N_apres})")
    print(f"  J_prior  = {float(J_PARTS['prior']):.6e}")

    # -------------------------------------------------------------------------
    # Pre-optimisation diagnostic: data vs. initial model
    # -------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        depth_nodes_diag = H0 - mesh.coordinates.dat.data_ro[:, 0] \
            if mesh.coordinates.dat.data_ro.ndim == 2 \
            else H0 - mesh.coordinates.dat.data_ro
        idx_sort_diag = np.argsort(depth_nodes_diag)
        d_plot_diag  = depth_nodes_diag[idx_sort_diag]

        rho_init_plot = rho.dat.data_ro[idx_sort_diag].copy()
        age_init_plot = age.dat.data_ro[idx_sort_diag].copy() / float(YEAR_S)

        fig0, axes0 = plt.subplots(1, 3, figsize=(15, 6))

        ax = axes0[0]
        ax.plot(rho_init_plot, d_plot_diag, "b-", lw=1.5, label="Model (initial)")
        ax.errorbar(obs_rho_val, obs_rho_depth, xerr=sigma_rho,
                    fmt="r.", ms=3, elinewidth=0.5, label="SP19 obs")
        ax.set_xlabel("Density (kg/m³)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_xlim(200, 950)
        ax.set_title("Density — initial vs obs")
        ax.legend(fontsize=8)

        ax = axes0[1]
        ax.plot(age_init_plot, d_plot_diag, "b-", lw=1.5, label="Model (initial)")
        ax.errorbar(obs_age_s / float(YEAR_S), obs_age_depth,
                    xerr=sigma_age / float(YEAR_S),
                    fmt="r.", ms=3, elinewidth=0.5, label="SP19 obs")
        ax.set_xlabel("Age (yr)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title("Age — initial vs obs")
        ax.legend(fontsize=8)

        # ApRES dR/dt anomaly: initial model vs detrended data
        ax = axes0[2]
        dRdt_raw_init = _dRdt_fn.dat.data_ro[idx_sort_diag] * float(YEAR_S) * 100.0
        # Subtract model value at base depth to get anomaly (like the cost function)
        _base_idx = np.argmin(np.abs(d_plot_diag - APRES_BASE_DEPTH))
        dRdt_init_anom = dRdt_raw_init - dRdt_raw_init[_base_idx]
        ax.plot(dRdt_init_anom, d_plot_diag, "b-", lw=1.5, label="Model anomaly (initial)")
        ax.errorbar(apres_detrended_myr * 100, apres_depth,
                    xerr=sigma_apres_myr * 100,
                    fmt="g.", ms=4, elinewidth=0.5, label=f"ApRES detrended {APRES_SITE}")
        ax.set_xlabel("dR/dt anomaly (cm/yr)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title("ApRES dR/dt anomaly — initial vs obs")
        ax.legend(fontsize=8)
        ax.axvline(0, color="grey", lw=0.5)

        fig0.suptitle(
            f"South Pole — initial model vs data\n"
            f"(spinup {SPINUP_YEARS} yr Buizert 2021 + ERA5 {ERA5_START_YEAR}–{ERA5_END_YEAR};  "
            f"T_spin {spin_T.min()-273.15:.1f}–{spin_T.max()-273.15:.1f} °C,  "
            f"A_spin mean {np.mean(spin_A):.4f} m/yr)"
        )
        fig0.tight_layout()
        diag_path = OUT_DIR / f"{OUT_PREFIX}_initial_vs_data.png"
        fig0.savefig(diag_path, dpi=150)
        print(f"\nInitial-vs-data plot saved to {diag_path}")
        plt.close(fig0)
    except Exception as _e:
        print(f"(Initial diagnostic plot skipped: {_e})")

    # -------------------------------------------------------------------------
    # Pre-optimisation sensitivity / gradient check
    # -------------------------------------------------------------------------
    # Verify objective responds to a control perturbation
    with stop_annotating():
        m0  = get_control_values(controls)
        m1  = m0.copy()
        lb0 = _as_scalar(bounds[0][0])
        ub0 = _as_scalar(bounds[0][1])
        m1[0] = min(max(m1[0] + 0.5, lb0), ub0)
        set_control_values(controls, m1)
    with stop_annotating():
        J1 = float(rf(controls))
    with stop_annotating():
        set_control_values(controls, m0)
    print(f"  Sensitivity check: ΔJ from log_kc0+0.5 = {J1 - float(J):+.3e}")

    # Taylor test (gradient check)
    taylor_test_multicontrol(rf, controls, bounds=bounds)

    # -------------------------------------------------------------------------
    # Optimisation
    # -------------------------------------------------------------------------
    minimisation_parameters["Status Test"]["Iteration Limit"]    = MAX_ITER
    minimisation_parameters["Status Test"]["Gradient Tolerance"] = GRAD_TOL
    minimisation_parameters["General"]["Secant"]["Type"]          = "Limited-Memory BFGS"
    minimisation_parameters["General"]["Secant"]["Maximum Storage"] = 25
    minimisation_parameters["General"]["Initial Trust Region Radius"] = 1.0e-2

    min_problem = MinimizationProblem(rf, bounds=bounds)

    try:
        optimiser = LinMoreOptimiser(
            min_problem, minimisation_parameters,
            checkpoint_dir=str(OUT_DIR / f"{OUT_PREFIX}_optim_checkpoint"),
        )
    except TypeError:
        optimiser = LinMoreOptimiser(min_problem, minimisation_parameters,
                                    auto_checkpoint=False)

    J_hist: list[float] = []
    _n_eval = [0]   # mutable container so the nested callback can increment it

    def _eval_cb(val, *args, **kwargs):
        _n_eval[0] += 1
        J_hist.append(float(val))
        if _n_eval[0] % 5 == 0 or _n_eval[0] == 1:
            print(f"  [iter {_n_eval[0]:03d}] J = {float(val):.6e}")

    rf.eval_cb_post = _eval_cb

    print(f"\nStarting L-BFGS optimisation (max {MAX_ITER} iters) ...")
    optimiser.run()

    # Commit ROL-updated checkpoints back to the Function objects
    for c in controls:
        commit_checkpoint(c)

    # -------------------------------------------------------------------------
    # Extract and report results
    # -------------------------------------------------------------------------
    m_map_phys = {}
    m_map_log  = {}
    for nm, c in zip(names, controls):
        log_val  = float(control_scalar_value(c))
        phys_val = math.exp(log_val)
        m_map_phys[nm] = phys_val
        m_map_log[nm]  = log_val

    print("\nMAP parameter estimates:")
    for nm in names:
        print(f"  {nm:12s} = {m_map_phys[nm]:.6e}  (initial: {INITIAL[nm]:.6e})")

    # Evaluate final objective (inside stop_annotating to avoid extending the tape)
    with stop_annotating():
        set_control_values(controls, [m_map_log[nm] for nm in names])
    with stop_annotating():
        J_map = float(rf(controls))
    print(f"\nFinal J(MAP) = {J_map:.6e}")

    # -------------------------------------------------------------------------
    # Save map_v2.json
    # -------------------------------------------------------------------------
    map_out = {
        "script"         : str(Path(__file__).name),
        "era5_window"    : [int(ERA5_START_YEAR), int(ERA5_END_YEAR)],
        "n_steps"        : int(n_steps),
        "obs_types"      : ["rho", "age", "apres"],
        "N_rho"          : int(N_rho),
        "N_age"          : int(N_age),
        "N_apres"        : int(N_apres),
        "APRES_SITE"     : APRES_SITE,
        "note"           : "no manual weights; σ determines relative contribution",
        "J_map"          : float(J_map),
        "J_hist"         : J_hist,
        "m_map_phys"     : m_map_phys,
        "m_map_log"      : m_map_log,
        "INITIAL"        : INITIAL,
        "TRUTH"          : TRUTH,
        "BOUNDS"         : {k: list(v) for k, v in BOUNDS.items()},
    }
    map_path = OUT_DIR / "map_v2.json"
    with open(map_path, "w") as fh:
        json.dump(map_out, fh, indent=2)
    print(f"\nMAP results saved to {map_path}")

    # -------------------------------------------------------------------------
    # Profile plots
    # -------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        depth_nodes = H0 - mesh.coordinates.dat.data_ro[:, 0] \
            if mesh.coordinates.dat.data_ro.ndim == 2 \
            else H0 - mesh.coordinates.dat.data_ro
        idx_sort = np.argsort(depth_nodes)
        d_plot   = depth_nodes[idx_sort]

        rho_plot  = rho.dat.data_ro[idx_sort]
        age_plot  = age.dat.data_ro[idx_sort] / float(YEAR_S)

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        ax = axes[0]
        ax.plot(rho_plot, d_plot, "b-", lw=1.5, label="Model (MAP)")
        ax.errorbar(obs_rho_val, obs_rho_depth, xerr=sigma_rho,
                    fmt="r.", ms=3, elinewidth=0.5, label="SP19 obs")
        ax.set_xlabel("Density (kg/m³)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title("Density")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.plot(age_plot, d_plot, "b-", lw=1.5, label="Model (MAP)")
        ax.errorbar(obs_age_s / float(YEAR_S), obs_age_depth,
                    xerr=sigma_age / float(YEAR_S),
                    fmt="r.", ms=3, elinewidth=0.5, label="SP19 obs")
        ax.set_xlabel("Age (yr)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title("Age (firn layers)")
        ax.legend(fontsize=8)

        # ApRES dR/dt anomaly: MAP model vs detrended data
        ax = axes[2]
        dRdt_raw_plot = _dRdt_fn.dat.data_ro[idx_sort] * float(YEAR_S) * 100.0
        _base_idx_map = np.argmin(np.abs(d_plot - APRES_BASE_DEPTH))
        dRdt_anom_plot = dRdt_raw_plot - dRdt_raw_plot[_base_idx_map]
        ax.plot(dRdt_anom_plot, d_plot, "b-", lw=1.5, label="Model anomaly (MAP)")
        ax.errorbar(apres_detrended_myr * 100, apres_depth,
                    xerr=sigma_apres_myr * 100,
                    fmt="g.", ms=4, elinewidth=0.5, label=f"ApRES detrended {APRES_SITE}")
        ax.set_xlabel("dR/dt anomaly (cm/yr)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_title("ApRES dR/dt anomaly (detrended)")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"South Pole MAP inversion v2  ({ERA5_START_YEAR}–{ERA5_END_YEAR})\n"
            + "  ".join(f"{k}={m_map_phys[k]:.2e}" for k in ["kc0", "kc1", "kg"])
        )
        fig.tight_layout()
        plot_path = OUT_DIR / f"{OUT_PREFIX}_profiles.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Profiles saved to {plot_path}")
        plt.close(fig)

        if len(J_hist) > 1:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.semilogy(range(1, len(J_hist) + 1), J_hist, "k-o", ms=3)
            ax2.set_xlabel("RF evaluation")
            ax2.set_ylabel("J (log scale)")
            ax2.set_title("Objective history")
            fig2.tight_layout()
            hist_path = OUT_DIR / f"{OUT_PREFIX}_objective_history.png"
            fig2.savefig(hist_path, dpi=150)
            print(f"Objective history saved to {hist_path}")
            plt.close(fig2)

    except Exception as _e:
        print(f"(Plotting skipped: {_e})")
