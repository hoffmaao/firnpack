"""southpole_inversion_v1.py

MAP inversion for South Pole firn densification parameters using real observations
and time-varying ERA5 forcing.

Forward model : 1D firn column (FirnColumnSolver full-density mode)
Spinup        : 200 years at Buizert 2021 pre-industrial temperature (-51.28°C / 221.87 K),
                RUN ON TAPE with the controlled parameters so the adjoint propagates
                through the full compaction history needed to constrain shallow firn density.
Taped run     : ERA5 annual temperature (bias-corrected -4.71 K) + accumulation, 1940 → 2015
Temperature   : ERA5 has a ~5°C warm bias at South Pole (underestimated winter inversions).
                Corrected ERA5 and Buizert spinup are continuous at the 1940 boundary.

Key design choice — on-tape spinup:
  Without this, the deep density misfit (20–100m) dominates J but has near-zero gradient
  because density at those depths is set by centuries of compaction history that a 75-year
  taped run cannot change.  Putting the spinup on tape gives ∂J/∂kc meaningful signal.

Controls (log-space, same as synthetic twin-experiment script):
    kc0      – densification rate coefficient, low-density regime
    kc1      – densification rate coefficient, high-density regime
    kg       – grain-growth rate coefficient
    rho_surf – surface density boundary condition (kg/m³)
    r2_surf  – surface grain-radius² boundary condition (m²)

Observations (snapshots at ERA5_END_YEAR):
    rho(z)   – SP19 ice-core depth-density profile (sp19_density.csv)
    age(z)   – SP19 ice-core depth-age profile, firn layers (sp19_depth_age.csv)
    ε̇(z)    – USP50 borehole compaction strain rates (Stevens et al. 2023, USAP-DC 601680);
               column-average log-strain-rate for 12 boreholes spanning 4–106 m depth.

Output:
    <OUT_DIR>/map.json  – MAP parameter estimates, compatible with
                          firn_uq_separate_v6_bounds_sampling_plotfix.py

Run:
    cd test/southpole/scripts
    OMP_NUM_THREADS=1 python southpole_inversion_v1.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
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

from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.solvers.firn_solver import FirnColumnSolver
from firnpack.constants import year as YEAR_S

# =============================================================================
# Paths
# =============================================================================
_HERE = Path(__file__).parent
_SOUTHPOLE = _HERE.parent
PROCESSED_DIR = _SOUTHPOLE / "processed"
ERA5_CSV       = PROCESSED_DIR / "era5_monthly_point.csv"
SP19_DENS_CSV  = PROCESSED_DIR / "sp19_density.csv"
SP19_AGE_CSV   = PROCESSED_DIR / "sp19_depth_age.csv"

# USP50 string-potentiometer compaction data (Stevens et al. 2023, USAP-DC 601680)
_USP50_DIR          = _SOUTHPOLE / "data/usap_dc/601680/usapdc_601680"
USP50_LENGTHS_CSV   = _USP50_DIR / "USP50_borehole_lengths.csv"
USP50_SPECS_CSV     = _USP50_DIR / "USP50_instrument_hole_specs.csv"

BUIZERT_TEMP_CSV  = PROCESSED_DIR / "buizert2021_spice_temp.csv"
BUIZERT_ACCUM_CSV = PROCESSED_DIR / "buizert2021_spice_accum.csv"

OUT_DIR = _SOUTHPOLE / "results"
OUT_PREFIX = "southpole_map"

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
#
# The key physics: the firn model imposes a FREE condition at the bottom boundary
# (velocity determined by the interior, not prescribed).  In steady state, mass
# conservation gives  rho(z)·w(z) = -A·rho_i  throughout, so the age at any
# coordinate x (0 = bottom, H0 = surface) satisfies:
#
#   age(x) = YEAR_S / (A · rho_i) · ∫_x^H0 rho(z) dz
#
# We initialise the age field from this formula using the observed SP19 density
# profile and the pre-industrial accumulation A_spin.  This gives the correct
# steady-state ages at every depth without needing an arbitrarily long on-tape
# spinup, so we can include age observations throughout the full 130-m column
# (ages up to ~1130 yr).  The filter below is purely a depth filter.
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

# USP50 compaction strain-rate observation uncertainty.
# 20 % relative accounts for spatial variability (USP50 is ~50 km upstream of pole).
SIGMA_STRAIN_REL = 0.20
STRAIN_WEIGHT    = 0.0   # cost-function weight for strain-rate term
# USP50 measures time-integrated column shortening (d(ln L)/dt over the
# observation period, 2017–present) — a time-averaged, depth-integrated
# quantity.  The model provides an *instantaneous* strain rate at ERA5 end
# (2015).  These are fundamentally different observables (unlike radar-based
# point strain rates used in synthetic tests), so the strain term is excluded
# from the objective.  The USP50 data is still loaded and plotted for reference.

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
    # The Taylor test's bound-clipping formula pushes r2_surf toward lb at max(eps);
    # with lb=1e-11 and INITIAL=1e-9, r2_surf reaches ~1.6e-11, causing density
    # overflow (drhodt ∝ 1/r2) that propagates as NaN into the enthalpy solve.
    # Lower bound = INITIAL (1e-9) prevents this; if the optimal r2_surf is smaller,
    # the solution will sit at this bound and the gradient will indicate it.
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

    The full depth range is usable because the spinup runs ON TAPE for 1200 yr
    starting from age=0 everywhere.  After 1275 yr total (spinup + ERA5), the
    age profile at every depth carries a non-zero adjoint w.r.t. the controls
    (see the note at SPINUP_YEARS in the settings block).

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


def load_usp50_strain_rates(
    lengths_csv: Path,
    specs_csv:   Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_top_m, z_bot_m, L0_m, strain_rate_per_s) for all USP50 boreholes.

    Strain rate is log-strain rate dε/dt = d(ln L)/dt [s⁻¹], negative for compaction.
    Missing values are coded -9999 in the CSV; hole 10 had unreliable 2017 data (masked).
    """
    specs   = pd.read_csv(specs_csv)
    lengths = pd.read_csv(lengths_csv, index_col=0, parse_dates=True)

    n_inst = len(specs)
    z_top  = specs["init_borehole_top"].values.astype(float)
    z_bot  = specs["init_borehole_bottom"].values.astype(float)
    L0     = specs["init_borehole_length"].values.astype(float)
    strain_rate = np.full(n_inst, np.nan)

    for ii in range(n_inst):
        col  = f"hole_length{ii + 1}"
        L_ts = lengths[col].copy()
        # Mask sentinel values (missing data coded as -9999)
        L_ts = L_ts.where(L_ts > 0.0)
        valid = L_ts.dropna()
        if len(valid) < 2:
            continue
        L_first  = float(valid.iloc[0])
        L_last   = float(valid.iloc[-1])
        dt_s     = (valid.index[-1] - valid.index[0]).total_seconds()
        if dt_s < 86400.0:  # less than one day — skip
            continue
        strain_rate[ii] = np.log(L_last / L_first) / dt_s  # 1/s, negative = compaction

    return z_top, z_bot, L0, strain_rate


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

    Junction note: at era5_start_year the Buizert accumulation (~0.098 m/yr at
    1940 CE) is higher than the ERA5 record (~0.073 m/yr), reflecting a real
    discrepancy between the ice-core reconstruction and reanalysis.  The jump is
    abrupt but physically represents model uncertainty in the 20th-century
    accumulation rate.  Temperature transitions smoothly (Buizert ~-51.2°C at
    1940 CE matches bias-corrected ERA5 by construction).
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
    # blocks to the tape during replay.  Without this, some pyadjoint versions
    # re-enable annotation inside rf.__call__, growing the tape on every evaluation
    # and corrupting the checkpointed intermediate values used by rf.derivative().
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

    print("Loading USP50 borehole strain rates ...")
    _usp50_ztop, _usp50_zbot, _usp50_L0, _usp50_sr = load_usp50_strain_rates(
        USP50_LENGTHS_CSV, USP50_SPECS_CSV)
    # Keep only boreholes within the model column with valid (negative = compaction) data
    _usp50_ok = (
        (_usp50_zbot <= H0)
        & np.isfinite(_usp50_sr)
        & (_usp50_sr < 0.0)
    )
    usp50_z_top  = _usp50_ztop[_usp50_ok]
    usp50_z_bot  = _usp50_zbot[_usp50_ok]
    usp50_L0     = _usp50_L0[_usp50_ok]
    usp50_sr_obs = _usp50_sr[_usp50_ok]        # 1/s, negative = compaction
    N_usp50      = len(usp50_sr_obs)
    usp50_sr_sig = np.abs(usp50_sr_obs) * SIGMA_STRAIN_REL
    print(f"  N_usp50 = {N_usp50}, z_bot range {usp50_z_bot.min():.1f}–{usp50_z_bot.max():.1f} m")
    print(f"  SR range {usp50_sr_obs.min()*float(YEAR_S)*100:.4f}–"
          f"{usp50_sr_obs.max()*float(YEAR_S)*100:.4f} %/yr")

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

    print(f"  median sigma_rho = {np.median(sigma_rho):.2f} kg/m³")
    print(f"  median sigma_age = {np.median(sigma_age)/float(YEAR_S):.2f} yr")

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
            print(f"  Kernel obs operator: {len(rho_kernels)} rho kernels, "
                  f"{len(age_kernels)} age kernels (sigma={OBS_KERNEL_SIGMA_M} m)")

    # USP50 box kernels — uniform average over each borehole column.
    # phi_box_i(x) = 1/L0_i  if  x ∈ [H0 - z_bot_i, H0 - z_top_i]  else 0
    # Normalised so that ∫ phi_box_i dx = 1 over the borehole column.
    # Built with stop_annotating() so the kernels are not on the tape.
    usp50_box_kernels: list = []
    with stop_annotating():
        _sx, = fd.SpatialCoordinate(mesh)
        for _ii in range(N_usp50):
            _x_bot_i = float(H0 - usp50_z_bot[_ii])   # deeper end (lower mesh coord)
            _x_top_i = float(H0 - usp50_z_top[_ii])   # shallower end (higher mesh coord)
            _L0_i    = float(usp50_L0[_ii])
            _box_expr = fd.conditional(
                fd.ge(_sx, _x_bot_i),
                fd.conditional(fd.le(_sx, _x_top_i),
                               fd.Constant(1.0 / _L0_i),
                               fd.Constant(0.0)),
                fd.Constant(0.0),
            )
            _phi_box = fd.Function(V, name=f"phi_usp50_{_ii:02d}")
            _phi_box.interpolate(_box_expr)
            usp50_box_kernels.append(_phi_box)
        print(f"  USP50 box kernels: {len(usp50_box_kernels)} boreholes built")

    # Pre-allocate the strain-rate Function used in J_strain.
    # Allocated off-tape so that the allocation itself is not a tape block;
    # only the interpolate calls inside forward_mapping() are tracked.
    with stop_annotating():
        _sr_fn = fd.Function(V, name="strain_rate_usp50")
        # Pre-allocate scratch Function for the prior term.
        # _prior_term() interpolates the (nonlinear) quadratic into this Function
        # before assembling, so pyadjoint sees an InterpolateBlock (UFL symbolic diff)
        # rather than an AssembleBlock whose adjoint can give the wrong sign for
        # nonlinear expressions.  Allocated off-tape with stop_annotating().
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
    # r2.assign(scalar) creates a step-function discontinuity at the surface:
    # the BC node is held at r2_surf (e.g. 1e-9 m²) while interior nodes are
    # set to the scalar (e.g. 2.5e-7 m²), a 250× jump.  The purely-advective
    # CG1 r2 PDE has no diffusion stabilisation → Gibbs-phenomenon oscillations
    # near this sharp interface → negative r2 at some nodes → r_eff < 0 →
    # drhodt sign-flip → ρ goes negative → K < 0 → enthalpy diverges → ρ = NaN.
    #
    # Fix: initialise with the analytic steady-state profile.
    # Steady state of  ∂r2/∂t + w·∂r2/∂x = dr2dt  is  r2(d) = r2_surf + (dr2dt/|w|)·d
    # where d = depth from surface.  This matches the BC at d=0 with no jump.
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
    # The solver's surface_density argument must be assignable (not a bare UFL expr).
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
    J_PARTS = {"rho": None, "age": None, "strain": None, "prior": None}

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
        # MUST be on tape so that rf(controls) tape-replay correctly resets the
        # state on every evaluation.  If this were inside stop_annotating(), the
        # replay would skip it and the spinup would start from leftover state
        # (non-deterministic J, wrong gradient, Taylor order ~1 instead of ~2).
        # The adjoint through these assignments is zero (rho_ic_fn is not a
        # control), so they add no spurious gradient.
        #
        # Density: use the observed SP19 profile (not a generic exponential).
        # This is essential for deep observations (>30 m) where the exponential
        # IC is ~170 kg/m³ too dense, biasing the gradient irrecoverably.
        rho.assign(rho_ic_fn)
        H.assign(float(params.c_i) * (float(spin_T[0]) - float(params.T_ref)))
        sigma.assign(0.0)
        r2.assign(r2_ic_fn)  # smooth steady-state linear profile (see r2_ic_fn construction above)
        # A uniform r2.assign(scalar) creates a sharp step at the surface (BC vs IC),
        # producing Gibbs oscillations in the purely-advective CG1 r2 PDE → negative
        # r2 → r_eff<0 → wrong-sign drhodt → ρ<0 → K<0 → enthalpy divergence → NaN.
        w.assign(0.0)
        age.assign(0.0)   # age=0 everywhere; spinup ≥ max_obs_age puts full profile on tape

        # --- Spinup (ON TAPE, time-varying Buizert 2021 reconstruction) ---
        # T_sk and A_sk are Python floats (not controls), so the adjoint through
        # these AssignBlocks is zero — no spurious gradient.  Assigning inside
        # the loop is required so that rf(controls) tape-replay restores the
        # correct forcing at each spinup step (same principle as the ERA5 loop).
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
            # Update data-driven forcing ON TAPE.
            # CRITICAL: these assigns MUST be on the tape (not in stop_annotating)
            # so that rf(controls) replay correctly sets Ts/Hs/accum at each
            # step k.  If they were inside stop_annotating(), replay would skip
            # them and the forcing would stay at stale ERA5 end-values throughout
            # the entire replay, producing wrong J and a wrong gradient direction.
            # Since T_k and A_k are Python floats (not controls), the adjoint
            # through these AssignBlocks is zero — no spurious gradient.
            T_k = float(run_T[k])
            A_k = float(run_A[k])   # m ice eq / yr
            Ts.assign(T_k)
            Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)

            # w_surf_bc IS annotated: captures d(J)/d(rho_surf) through the BC.
            # accum (m ice eq/yr) is data; rho_surf_expr is the control.
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

        # --- USP50 column-average strain-rate misfits ---
        # Compute the Arthern densification rate and vertical strain rate at the
        # final state (ERA5_END_YEAR) using the rhoCoef from the last ERA5 step.
        #
        # Key adjoint design: interpolate the nonlinear strain-rate expression
        # into a CG1 Function (_sr_fn) BEFORE assembling.  This splits the
        # computation into:
        #   (a) InterpolateBlock — nonlinear, differentiated by UFL symbolics
        #   (b) AssembleBlock    — linear in _sr_fn, adjoint is just phi_box
        # Assembling a complex nonlinear UFL expression directly (without
        # interpolating first) can give wrong adjoint signs with pyadjoint
        # when the expression contains ratios or max_value calls.
        drhodt_end = model.densification_rate_full_density(
            rho, sigma, r2, rhoCoef, T=Ts)
        _sr_fn.interpolate(
            model.vertical_strain_rate_from_continuity(rho, drhodt_end))

        J_strain = 0.0
        for _phi_b, _obs_sr, _sig_sr in zip(usp50_box_kernels, usp50_sr_obs, usp50_sr_sig):
            _pred_sr  = fd.assemble(_sr_fn * _phi_b * dx_domain)
            _r_sr     = (_pred_sr - float(_obs_sr)) / float(_sig_sr)
            J_strain += 0.5 * _r_sr * _r_sr
        J_strain /= float(N_usp50)

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
            J_rho /= N_rho

            J_age = 0.0
            for phi, obs_c, sig_c in zip(age_kernels, age_obs_vals, age_sig_vals):
                pred   = fd.assemble(age * phi * dx_domain)
                r      = (pred - obs_c) / sig_c
                J_age += 0.5 * r * r
            J_age /= N_age
        else:
            rho_model.interpolate(rho)
            mis_rho = (rho_model - rho_obs_fn) / rho_sig_fn
            J_rho   = 0.5 * fd.assemble(mis_rho ** 2 * dx_rho) / N_rho

            age_model.interpolate(age)
            mis_age = (age_model - age_obs_fn) / age_sig_fn
            J_age   = 0.5 * fd.assemble(mis_age ** 2 * dx_age) / N_age

        J = J_rho + J_age + STRAIN_WEIGHT * J_strain

        # --- Weak log-space priors ---
        J_prior = 0.0
        if USE_PRIOR:
            def _prior_term(log_ctrl, truth_val, sigma_log):
                # Interpolate the nonlinear (quadratic) expression into a Function
                # BEFORE assembling.  Assembling a nonlinear UFL expression directly
                # gives the wrong AssembleBlock adjoint sign (pyadjoint limitation).
                # InterpolateBlock differentiates via UFL symbolics → correct adjoint.
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
            J_PARTS["rho"]    = J_rho
            J_PARTS["age"]    = J_age
            J_PARTS["strain"] = J_strain
            J_PARTS["prior"]  = J_prior

        return J

    # Build the tape
    J  = forward_mapping()
    rf = ReducedFunctional(J, [Control(c) for c in controls])
    pause_annotation()

    print(f"\nInitial objective J(m0) = {float(J):.6e}")
    print(f"  J_rho    = {float(J_PARTS['rho']):.6e}")
    print(f"  J_age    = {float(J_PARTS['age']):.6e}")
    print(f"  J_strain = {float(J_PARTS['strain']):.6e}  (weight={STRAIN_WEIGHT})")
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

        fig0, axes0 = plt.subplots(1, 2, figsize=(10, 6))

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

        fig0.suptitle(
            f"South Pole — initial model vs SP19 data\n"
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
    # Save map.json (compatible with firn_uq_separate_v6_bounds_sampling_plotfix.py)
    # -------------------------------------------------------------------------
    map_out = {
        "script"         : str(Path(__file__).name),
        "era5_window"    : [int(ERA5_START_YEAR), int(ERA5_END_YEAR)],
        "n_steps"        : int(n_steps),
        "obs_types"      : ["rho", "age"] + (["strain"] if STRAIN_WEIGHT > 0 else []),
        "N_rho"          : int(N_rho),
        "N_age"          : int(N_age),
        "N_usp50"        : int(N_usp50),
        "STRAIN_WEIGHT"  : float(STRAIN_WEIGHT),
        "SIGMA_STRAIN_REL": float(SIGMA_STRAIN_REL),
        "J_map"          : float(J_map),
        "J_hist"         : J_hist,
        "m_map_phys"     : m_map_phys,
        "m_map_log"      : m_map_log,
        "INITIAL"        : INITIAL,
        "TRUTH"          : TRUTH,
        "BOUNDS"         : {k: list(v) for k, v in BOUNDS.items()},
    }
    map_path = OUT_DIR / "map.json"
    with open(map_path, "w") as fh:
        json.dump(map_out, fh, indent=2)
    print(f"\nMAP results saved to {map_path}")

    # -------------------------------------------------------------------------
    # Optional: profile plots
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

        rho_plot = rho.dat.data_ro[idx_sort]
        age_plot = age.dat.data_ro[idx_sort] / float(YEAR_S)
        # Strain rate profile from the last forward solve (in %/yr for readability)
        sr_plot_pyr = _sr_fn.dat.data_ro[idx_sort] * float(YEAR_S) * 100.0

        fig, axes = plt.subplots(1, 3, figsize=(14, 6))

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

        ax = axes[2]
        # Background: full model strain-rate profile (context only)
        ax.plot(sr_plot_pyr, d_plot, "b-", lw=0.8, alpha=0.4, label="Model profile")
        # Box-kernel averaged model strain rates — the correct comparison with USP50.
        # Each value is the depth-average of ε̇(z) over the borehole column
        # [z_top, z_bot], matching how USP50 measures column-integrated shortening.
        # NOTE: model is at ERA5 end (2015); USP50 observations are 2017–2018.
        # Strain is NOT in the objective (STRAIN_WEIGHT=0); shown here for reference.
        _model_box_sr_pyr = np.array([
            float(fd.assemble(_sr_fn * _phi_b * dx_domain)) * float(YEAR_S) * 100.0
            for _phi_b in usp50_box_kernels
        ])
        _usp50_sr_pyr     = usp50_sr_obs * float(YEAR_S) * 100.0
        _usp50_sr_sig_pyr = usp50_sr_sig * float(YEAR_S) * 100.0
        _usp50_z_mid      = 0.5 * (usp50_z_top + usp50_z_bot)
        _usp50_z_half     = 0.5 * (usp50_z_bot - usp50_z_top)
        ax.errorbar(_model_box_sr_pyr, _usp50_z_mid,
                    yerr=_usp50_z_half,
                    fmt="bs", ms=4, elinewidth=0.8, capsize=3,
                    label="Model box-avg")
        ax.errorbar(_usp50_sr_pyr, _usp50_z_mid,
                    xerr=_usp50_sr_sig_pyr, yerr=_usp50_z_half,
                    fmt="ro", ms=4, elinewidth=0.8, capsize=3,
                    label="USP50 obs (2017–18)")
        ax.set_xlabel("Strain rate (%/yr)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title("Compaction strain rate (diagnostic)")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"South Pole MAP inversion  ({ERA5_START_YEAR}–{ERA5_END_YEAR})\n"
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
