"""southpole_inversion_v10_sp19.py

MAP inversion for South Pole using Arthern Mode A (basic densification).

Changes from v9:
  - Mode A (Arthern/Ligtenberg) instead of Mode B (Kingslake full-density)
    No stress/grain radius prognostics — simpler physics, um-fdm style
  - 5 controls: Ec, kc0, kc1, rho_surf, Q_geo (enough DOF for rho/age/T)
  - Uses all three data types (rho + age + T) by default

Run:
    cd test/southpole/scripts
    OMP_NUM_THREADS=1 python southpole_inversion_v10_sp19.py
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
# (Buizert spinup no longer used — IC from FirnMICE checkpoint)

OUT_DIR = _SOUTHPOLE / "results"

# =============================================================================
# Settings
# =============================================================================

# Mesh / geometry
H0 = 130.0        # column height (m)
NZ = 100           # number of elements
STRETCH_P = 2.5    # power-law stretching exponent
SURFACE_ID = 2     # Firedrake boundary marker for x = H0

# Spinup from Buizert 2021 SPICEcore reconstruction (off-tape)
BUIZERT_TEMP_CSV  = PROCESSED_DIR / "buizert2021_spice_temp.csv"
BUIZERT_ACCUM_CSV = PROCESSED_DIR / "buizert2021_spice_accum.csv"
SPINUP_START_CE = 740         # start year for spinup (CE)
SPINUP_DT_YEARS = 10.0        # coarse dt for spinup

# Time (on tape)
ERA5_START_YEAR = 1940
CORE_YEAR       = 2015    # SP19 ice core year (density + age observations)
ERA5_END_YEAR   = CORE_YEAR
DT_YEARS        = 0.5     # sub-annual timestep for ERA5 period

# ERA5 bias corrections
# NOTE: raw ERA5 T (-45.9°C) matches SPICEcore borehole (-45.5°C) well.
# The previous -4.71 K correction was too aggressive.  Set to 0.
ERA5_T_OFFSET_K    = 0.0      # K (no temperature bias correction)
ERA5_A_OFFSET_M_YR = +0.0163  # m ice eq/yr

# Borehole temperature observations (SPICEcore USP50)
BOREHOLE_T_CSV = PROCESSED_DIR / "spicecore_borehole_T.csv"
SIGMA_T_ABS     = 0.5        # K, measurement uncertainty

# Basal thermal BC
BASAL_HEAT_FLUX_W_M2 = 0.05  # W/m², typical Antarctic geothermal heat flux

# Observation depth limits
RHO_OBS_DEPTH_MAX = H0
AGE_OBS_YEAR_MIN  = 0
AGE_OBS_STRIDE    = 5         # subsample age obs to ~match density count

# Data type selection via environment variable
# "rho", "age", "T", or combinations like "rho+age", "rho+age+T"
OBS_TYPES = os.environ.get("FIRN_OBS", "rho+age+T").split("+")
USE_RHO_OBS = "rho" in OBS_TYPES
USE_AGE_OBS = "age" in OBS_TYPES
USE_BOREHOLE_T = "T" in OBS_TYPES

# Observation uncertainties
SIGMA_RHO_ABS   = 15.0      # kg/m3
SIGMA_RHO_REL   = 0.03
SIGMA_AGE_ABS_S = 1.0 * float(YEAR_S)  # 1 year
SIGMA_AGE_REL   = 0.02
USE_UNIFORM_OBS_SIGMA = True
OBS_SIGMA_SCALE = 1.0

# Observation operator: point evaluation via VertexOnlyMesh (icepack-style)

_obs_tag = "+".join(OBS_TYPES)
OUT_PREFIX = f"southpole_map_v10_{_obs_tag}"

# Smooth density-coefficient switch
P0 = FirnParameters()
RHO_M = float(getattr(P0, "rho_m", 550.0))
RHO_SMOOTH = 20.0

# Controls -- starting point and bounds
# div_h: horizontal divergence (s^-1) from MEaSUREs velocity, 50 km average
#   Negative = convergent (thickening), positive = divergent (thinning)
#   Can be positive or negative -> use LINEAR parameterization (not log)
#   div_h offsets the vertical velocity: w_base = w_surf - H*div_h (approx)
DIV_H_INIT     = -1.6e-12   # s^-1 (50 km average from MEaSUREs)
DIV_H_SIGMA    =  9.7e-11   # s^-1 (spatial std from 50 km patch)

TRUTH = dict(
    kc0      = float(getattr(P0, "kc0",    9.2e-9)),
    kc1      = float(getattr(P0, "kc1",    3.7e-9)),
    kg       = float(getattr(P0, "kg",     1.3e-7)),
    Ec       = float(getattr(P0, "Ec", 60e3)),
    rho_surf = 350.0,
    r2_surf  = 2.5e-7,
    Q_geo    = 0.05,              # W/m², basal geothermal heat flux
    div_h    = DIV_H_INIT,
    A_scale  = 1.0,               # multiplicative accumulation factor
    T_offset = 0.0,               # K, temperature additive offset
)

INITIAL = dict(
    kc0      = float(getattr(P0, "kc0",    9.2e-9)),
    kc1      = float(getattr(P0, "kc1",    3.7e-9)),
    kg       = float(getattr(P0, "kg",     1.3e-7)),
    Ec       = float(getattr(P0, "Ec", 60e3)),
    rho_surf = 350.0,
    r2_surf  = 2.5e-7,
    Q_geo    = 0.05,              # W/m²
    div_h    = DIV_H_INIT,
    A_scale  = 1.0,
    T_offset = 0.0,
)

BOUNDS = dict(
    kc0      = (1.0e-11, 1.0e-3),
    kc1      = (1.0e-11, 1.0e-3),
    kg       = (1.0e-13, 1.0e-3),
    Ec       = (20e3,    120e3),      # J/mol, activation energy for creep
    rho_surf = (100.0,   700.0),      # kg/m³
    r2_surf  = (1.0e-9, 1.0e-6),
    Q_geo    = (1.0e-4,  1.0),        # W/m², geothermal heat flux
    div_h    = (-10.0, 10.0),         # normalized units (div_h_norm), ~10 sigma
    A_scale  = (0.5,    2.0),         # 50%-200% of ERA5 accumulation
    T_offset = (-10.0,  10.0),        # K, additive temperature offset
)

# Priors
USE_PRIOR = True
PRIOR_SIGMA_LOG = 1.5
EC_PRIOR_SIGMA_LOG       = 0.5        # tighter: exp(0.5)≈1.65x variation
RHO_SURF_PRIOR_SIGMA_LOG = 0.25
R2_SURF_PRIOR_SIGMA_LOG  = 0.8
QGEO_PRIOR_SIGMA_LOG     = 1.0        # exp(1)≈2.7x variation
A_SCALE_PRIOR_SIGMA_LOG  = 0.3        # exp(0.3)≈1.35x variation
T_OFFSET_PRIOR_SIGMA     = 3.0        # K, Gaussian prior in linear space
# div_h prior is Gaussian in linear space (not log), sigma from velocity data
DIV_H_PRIOR_SIGMA        = DIV_H_SIGMA  # 9.7e-11 s^-1

# Optimiser
MAX_ITER = int(os.environ.get("FIRN_MAX_ITER", "50"))
GRAD_TOL = 1.0e-10
SEED = 0

# All prognostic fields (rho, r2, age) use DG1 with upwind inflow BCs.
# Surface values enter through the DG inflow flux at the physical velocity
# scale — no artificial penalty weighting needed.

# Hessian / UQ
HESSIAN_EPS = 1.0e-2       # FD step for Hessian computation (log-space)
N_ENSEMBLE  = int(os.environ.get("FIRN_N_ENSEMBLE", "50"))


# =============================================================================
# Data loading
# =============================================================================

def load_era5_annual(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    t_rows = df[df["t2m_K"].notna()][["year", "t2m_K"]].copy()
    a_rows = df[df["net_accum_m_iceeq_month"].notna()][["year", "net_accum_m_iceeq_month"]].copy()
    T_by_yr = t_rows.groupby("year")["t2m_K"].mean()
    A_by_yr = a_rows.groupby("year")["net_accum_m_iceeq_month"].sum()
    years = np.array(sorted(T_by_yr.index.intersection(A_by_yr.index)), dtype=int)
    return years, T_by_yr.loc[years].values.astype(float), A_by_yr.loc[years].values.astype(float)


def load_sp19_density(csv_path: Path, depth_max: float = 100.0):
    df = pd.read_csv(csv_path)
    df = df[df["depth_m"] <= depth_max].copy()
    depth = df["depth_m"].values.astype(float)
    rho = df["rho_kgm3"].values.astype(float) * 1000.0  # g/cm3 -> kg/m3
    idx = np.argsort(depth)
    return depth[idx], rho[idx]


def load_sp19_age_firn(csv_path: Path, core_year: int = CORE_YEAR,
                        year_min: int = AGE_OBS_YEAR_MIN,
                        depth_max: float = H0,
                        stride: int = 1):
    df = pd.read_csv(csv_path)
    mask = (df["year_CE"] >= year_min) & (df["year_CE"] <= core_year) & (df["depth_m"] <= depth_max)
    df = df[mask].copy()
    depth = df["depth_m"].values.astype(float)
    age_yr = float(core_year) - df["year_CE"].values.astype(float)
    age_s = age_yr * float(YEAR_S)
    idx = np.argsort(depth)
    depth, age_s = depth[idx], age_s[idx]
    if stride > 1:
        depth, age_s = depth[::stride], age_s[::stride]
    return depth, age_s


def load_firnmice_profiles(h5_path: Path, fm_H0: float, target_H0: float,
                           stretch_p: float = 3.0):
    """Load FirnMICE state from checkpoint via h5py and return depth-sorted profiles.

    Uses h5py to read raw data (avoids PETSc HDF5 close issues with mixed
    Firedrake + h5py data in the same file).

    Returns dict of {field_name: (depth_m, values)} for the upper `target_H0` meters.
    """
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        g = f["firnmice"]
        # Time-series metadata gives us the last index
        time_yr = np.asarray(g["time_years"], dtype=float)
        last_idx = len(time_yr) - 1
        depth_nodes = np.asarray(g["depth_nodes_m"], dtype=float)

    # Load functions via Firedrake (need the mesh for DOF ordering).
    # The CheckpointFile close can fail when the h5 also has h5py data,
    # but the data is loaded successfully before close — catch the error.
    try:
        with fd.CheckpointFile(str(h5_path), "r") as chk:
            fm_mesh = chk.load_mesh()
            fields = {}
            for name in ["enthalpy", "density", "velocity", "stress", "grain_radius2", "age"]:
                try:
                    f = chk.load_function(fm_mesh, name, idx=last_idx)
                    fields[name] = np.asarray(f.dat.data_ro, dtype=float).copy()
                except Exception as e:
                    print(f"  [warn] Could not load '{name}': {e}")
    except Exception:
        pass  # PETSc HDF5 close error — data already extracted

    # Extract mesh coordinates -> depth
    fm_x = fm_mesh.coordinates.dat.data_ro
    if fm_x.ndim == 2:
        fm_x = fm_x[:, 0]
    fm_depth = fm_H0 - np.asarray(fm_x, dtype=float)
    fm_sidx = np.argsort(fm_depth)
    fm_depth_sorted = fm_depth[fm_sidx]

    profiles = {}
    for name, vals in fields.items():
        profiles[name] = (fm_depth_sorted, vals[fm_sidx])

    # Trim to target_H0
    for name in list(profiles.keys()):
        d, v = profiles[name]
        mask = d <= target_H0 + 1.0
        profiles[name] = (d[mask], v[mask])

    return profiles, last_idx


# =============================================================================
# Firedrake / mesh helpers
# =============================================================================

def make_real(R: fd.FunctionSpace, value: float, name: str) -> fd.Function:
    f = fd.Function(R, name=name)
    f.dat.data[:] = float(value)
    return f


BASE_ID = 1  # Firedrake boundary marker for x = 0 (base of column)


def w_surf_value(params, accum_m_iceeq_yr, rho_s):
    """Surface velocity from accumulation and surface density."""
    return -accum_m_iceeq_yr * params.rho_i / rho_s / float(YEAR_S)


def make_bcs(V, params, accum_fn, rho_s_fixed, Hs_fn, surface_id=SURFACE_ID):
    """Build Dirichlet BCs for H and w at the surface.

    w_surf uses a FIXED rho_s to avoid pyadjoint issues with control-dependent
    Dirichlet BCs. rho surface BC is handled via inflow_values (DG upwind flux).
    """
    bc_H = fd.DirichletBC(V, Hs_fn, surface_id)
    w_surf_bc = fd.Function(V, name="w_surf_bc")
    w_surf_bc.interpolate(w_surf_value(params, accum_fn, fd.Constant(float(rho_s_fixed))))
    bc_w = fd.DirichletBC(V, w_surf_bc, surface_id)
    return [bc_H, None, bc_w], {"w_surf_bc": w_surf_bc}


def build_stretched_mesh(H0: float, nz: int, p: float):
    mesh = fd.IntervalMesh(nz, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** p)
    coord_fs = mesh.coordinates.function_space()
    mesh.coordinates.assign(fd.Function(coord_fs).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    return mesh, V


def safe_vertex_coords(H0: float, depths_m: np.ndarray) -> np.ndarray:
    d = np.clip(np.asarray(depths_m, dtype=float), 0.0, H0)
    x = H0 - d
    eps = 1.0e-8 * max(H0, 1.0)
    x = np.where(x <= 0.0, eps, x)
    x = np.where(x >= H0, H0 - eps, x)
    return np.vstack([x]).T


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("South Pole MAP inversion v9  (obs={_obs_tag}, on-tape spinup)")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\nLoading ERA5 forcing ...")
    era5_years, era5_T, era5_A = load_era5_annual(ERA5_CSV)

    phase1_mask = (era5_years >= ERA5_START_YEAR) & (era5_years < CORE_YEAR)
    phase1_years = era5_years[phase1_mask]
    phase1_T = era5_T[phase1_mask] + ERA5_T_OFFSET_K
    phase1_A = era5_A[phase1_mask] + ERA5_A_OFFSET_M_YR
    n_phase1 = len(phase1_years)

    # Sub-annual: repeat each year's forcing for DT_YEARS sub-steps
    n_substeps = int(round(1.0 / DT_YEARS))
    era5_T_sub = np.repeat(phase1_T, n_substeps)
    era5_A_sub = np.repeat(phase1_A, n_substeps)
    n_era5_total = len(era5_T_sub)

    dt_s = float(DT_YEARS * YEAR_S)
    print(f"  ERA5: {phase1_years[0]}-{phase1_years[-1]} ({n_phase1} yr, "
          f"{n_era5_total} steps @ dt={DT_YEARS}yr) -> SP19 obs at {CORE_YEAR}")

    # Mean SP forcing for equilibration
    _sp_T_mean = float(np.mean(phase1_T))
    _sp_A_mean = float(np.mean(phase1_A))

    if USE_RHO_OBS:
        print("Loading SP19 density observations ...")
        obs_rho_depth, obs_rho_val = load_sp19_density(SP19_DENS_CSV, RHO_OBS_DEPTH_MAX)
        print(f"  N_rho = {len(obs_rho_depth)}, depth {obs_rho_depth[0]:.2f}-{obs_rho_depth[-1]:.2f} m")
    else:
        print("Density observations DISABLED.")
        obs_rho_depth = np.array([])
        obs_rho_val = np.array([])

    if USE_AGE_OBS:
        print("Loading SP19 age observations ...")
        obs_age_depth, obs_age_s = load_sp19_age_firn(SP19_AGE_CSV, stride=AGE_OBS_STRIDE)
        print(f"  N_age = {len(obs_age_depth)}, depth {obs_age_depth[0]:.2f}-{obs_age_depth[-1]:.2f} m")
    else:
        print("Age observations DISABLED.")
        obs_age_depth = np.array([])
        obs_age_s = np.array([])

    # Borehole temperature observations
    obs_T_depth = np.array([])
    obs_T_K = np.array([])
    sigma_T = np.array([])
    if USE_BOREHOLE_T:
        print("Loading SPICEcore borehole temperature ...")
        _bh = pd.read_csv(BOREHOLE_T_CSV)
        _bh = _bh[_bh["depth_m"] <= H0].copy()
        obs_T_depth = _bh["depth_m"].values.astype(float)
        obs_T_K = (_bh["T_C"].values.astype(float) + 273.15)  # °C -> K
        # Convert T to enthalpy for comparison with model H field
        obs_T_H = float(P0.c_i) * (obs_T_K - float(P0.T_ref))  # J/kg
        sigma_T = np.full_like(obs_T_K, float(SIGMA_T_ABS))
        # Sigma in enthalpy units: σ_H = c_i * σ_T
        sigma_T_H = float(P0.c_i) * sigma_T
        print(f"  N_T = {len(obs_T_depth)}, depth {obs_T_depth[0]:.0f}-{obs_T_depth[-1]:.0f} m")
        print(f"  T range: {obs_T_K.min()-273.15:.1f} to {obs_T_K.max()-273.15:.1f} °C")

    # -------------------------------------------------------------------------
    # Mesh + spaces
    # -------------------------------------------------------------------------
    mesh, V = build_stretched_mesh(H0, NZ, STRETCH_P)
    R_space = fd.FunctionSpace(mesh, "R", 0)
    dx_domain = fd.dx(domain=mesh)

    # -------------------------------------------------------------------------
    # Observation uncertainties
    # -------------------------------------------------------------------------
    if len(obs_rho_val) > 0:
        sigma_rho = SIGMA_RHO_ABS + SIGMA_RHO_REL * np.abs(obs_rho_val)
    else:
        sigma_rho = np.array([])
    if len(obs_age_s) > 0:
        sigma_age = SIGMA_AGE_ABS_S + SIGMA_AGE_REL * np.maximum(obs_age_s, 0.0)
    else:
        sigma_age = np.array([])
    if USE_UNIFORM_OBS_SIGMA:
        if len(sigma_rho) > 0:
            sigma_rho = np.full_like(sigma_rho, float(np.median(sigma_rho)))
        if len(sigma_age) > 0:
            sigma_age = np.full_like(sigma_age, float(np.median(sigma_age)))
    if len(sigma_rho) > 0:
        sigma_rho *= OBS_SIGMA_SCALE
    if len(sigma_age) > 0:
        sigma_age *= OBS_SIGMA_SCALE

    # -------------------------------------------------------------------------
    # Observation operators via VertexOnlyMesh (icepack-style point eval)
    # -------------------------------------------------------------------------
    # Collect all observation points into a single VOM per field type.
    # Each VOM holds the observation locations; we interpolate model fields
    # onto it, then compute chi-squared misfit directly.

    with stop_annotating():
        # Convert observation depths to mesh x-coordinates
        # x = H0 - depth, clipped to stay inside the domain
        eps_x = 1.0e-8 * H0

        def _depth_to_x(depths):
            x = H0 - np.clip(depths, 0.0, H0)
            x = np.where(x <= 0, eps_x, x)
            x = np.where(x >= H0, H0 - eps_x, x)
            return x

        # --- Density VOM ---
        vom_rho = None
        rho_pred_fn = None
        N_rho = len(obs_rho_depth)
        if N_rho > 0:
            rho_x = _depth_to_x(obs_rho_depth)
            vom_rho = fd.VertexOnlyMesh(mesh, rho_x.reshape(-1, 1))
            P0_rho = fd.FunctionSpace(vom_rho, "DG", 0)
            rho_obs_fn = fd.Function(P0_rho, name="rho_obs")
            rho_obs_fn.dat.data[:] = obs_rho_val
            rho_sig_fn = fd.Function(P0_rho, name="rho_sig")
            rho_sig_fn.dat.data[:] = sigma_rho
            rho_pred_fn = fd.Function(P0_rho, name="rho_pred")

        # --- Age VOM ---
        vom_age = None
        age_pred_fn = None
        N_age = len(obs_age_depth)
        if N_age > 0:
            age_x = _depth_to_x(obs_age_depth)
            vom_age = fd.VertexOnlyMesh(mesh, age_x.reshape(-1, 1))
            P0_age = fd.FunctionSpace(vom_age, "DG", 0)
            age_obs_fn = fd.Function(P0_age, name="age_obs")
            age_obs_fn.dat.data[:] = obs_age_s
            age_sig_fn = fd.Function(P0_age, name="age_sig")
            age_sig_fn.dat.data[:] = sigma_age
            age_pred_fn = fd.Function(P0_age, name="age_pred")

        # --- Borehole T VOM ---
        vom_T = None
        T_pred_fn = None
        N_T = len(obs_T_depth)
        if USE_BOREHOLE_T and N_T > 0:
            T_x = _depth_to_x(obs_T_depth)
            vom_T = fd.VertexOnlyMesh(mesh, T_x.reshape(-1, 1))
            P0_T = fd.FunctionSpace(vom_T, "DG", 0)
            T_obs_fn = fd.Function(P0_T, name="T_obs")
            T_obs_fn.dat.data[:] = obs_T_H    # enthalpy units
            T_sig_fn = fd.Function(P0_T, name="T_sig")
            T_sig_fn.dat.data[:] = sigma_T_H  # enthalpy units
            T_pred_fn = fd.Function(P0_T, name="T_pred")

        print(f"  Point obs: {len(obs_rho_depth)} rho, "
              f"{len(obs_age_depth)} age, {len(obs_T_depth)} T")

    with stop_annotating():
        _prior_fn = fd.Function(V, name="prior_scratch")
        domain_len = float(fd.assemble(fd.Constant(1.0) * dx_domain)) + 1e-30

    # -------------------------------------------------------------------------
    # State Functions
    # -------------------------------------------------------------------------
    _mesh_x_flat = (mesh.coordinates.dat.data_ro[:, 0]
                    if mesh.coordinates.dat.data_ro.ndim == 2
                    else mesh.coordinates.dat.data_ro)
    _mesh_depth = H0 - _mesh_x_flat

    # Mode A with DG1 density and age (upwind advection, penalty/inflow BCs)
    V_dg = fd.FunctionSpace(mesh, "DG", 1)

    H     = fd.Function(V, name="H")
    rho   = fd.Function(V_dg, name="rho")
    w     = fd.Function(V, name="w")
    sigma = fd.Function(V, name="sigma")
    r2    = fd.Function(V, name="r2")
    age   = fd.Function(V_dg, name="age")

    # Simple analytical IC (will be overwritten by on-tape spinup)
    xi = fd.SpatialCoordinate(mesh)[0]
    _depth_expr = H0 - xi

    # -------------------------------------------------------------------------
    # Controls
    # -------------------------------------------------------------------------
    # Log-space controls (positive-definite parameters)
    log_kc0   = make_real(R_space, math.log(float(INITIAL["kc0"])),      "log_kc0")
    log_kc1   = make_real(R_space, math.log(float(INITIAL["kc1"])),      "log_kc1")
    log_kg    = make_real(R_space, math.log(float(INITIAL["kg"])),       "log_kg")
    log_Ec    = make_real(R_space, math.log(float(INITIAL["Ec"])),       "log_Ec")
    log_rho_s = make_real(R_space, math.log(float(INITIAL["rho_surf"])), "log_rho_s")
    log_r2_s  = make_real(R_space, math.log(float(INITIAL["r2_surf"])),  "log_r2_s")
    log_Qgeo  = make_real(R_space, math.log(float(INITIAL["Q_geo"])),    "log_Qgeo")
    log_Ascale = make_real(R_space, math.log(float(INITIAL["A_scale"])), "log_Ascale")

    # Normalized div_h control: div_h = DIV_H_INIT + DIV_H_SIGMA * div_h_norm
    # so that div_h_norm has O(1) range, matching the log-space controls.
    div_h_norm = make_real(R_space, 0.0, "div_h_norm")  # 0 = initial value
    div_h_ctrl = DIV_H_INIT + DIV_H_SIGMA * div_h_norm  # UFL expression

    # T_offset: linear control, additive temperature bias in K
    T_offset_ctrl = make_real(R_space, float(INITIAL["T_offset"]), "T_offset")

    # 5 controls for Arthern Mode A: rate coefficients, temperature sensitivity,
    # surface density, and basal heat flux (for T profile)
    controls = [log_kc0, log_kc1, log_Ec, log_rho_s, log_Qgeo]
    names    = ["kc0", "kc1", "Ec", "rho_surf", "Q_geo"]
    LOG_CONTROLS = {"kc0", "kc1", "Ec", "rho_surf", "Q_geo"}
    n_ctrl = len(controls)

    # Bounds in control space (log for log-params, linear for div_h)
    lb_vec = np.array([
        math.log(BOUNDS[nm][0]) if nm in LOG_CONTROLS else BOUNDS[nm][0]
        for nm in names
    ])
    ub_vec = np.array([
        math.log(BOUNDS[nm][1]) if nm in LOG_CONTROLS else BOUNDS[nm][1]
        for nm in names
    ])

    lbs_fns, ubs_fns = [], []
    for i, nm in enumerate(names):
        lb_fn = make_real(R_space, float(lb_vec[i]), f"{nm}_lb")
        ub_fn = make_real(R_space, float(ub_vec[i]), f"{nm}_ub")
        lbs_fns.append(lb_fn)
        ubs_fns.append(ub_fn)
    bounds_pairs = list(zip(lbs_fns, ubs_fns))

    # Prior sigmas in control space
    prior_sigmas = np.array([
        PRIOR_SIGMA_LOG,           # kc0
        PRIOR_SIGMA_LOG,           # kc1
        EC_PRIOR_SIGMA_LOG,        # Ec
        RHO_SURF_PRIOR_SIGMA_LOG,  # rho_surf
        QGEO_PRIOR_SIGMA_LOG,      # Q_geo
    ])

    # -------------------------------------------------------------------------
    # Build model + solver
    # -------------------------------------------------------------------------
    kc0_expr      = fd.exp(log_kc0)
    kg_expr       = fd.exp(log_kg)
    Ec_expr       = fd.exp(log_Ec)
    rho_surf_expr = fd.exp(log_rho_s)

    # Fixed (non-control) expressions — kg is not a control in Mode A
    kc1_expr    = fd.exp(log_kc1)
    Q_geo_expr  = fd.exp(log_Qgeo)
    kg_fixed    = fd.Constant(float(INITIAL["kg"]))

    # kc0, kc1, Ec, kg enter the Arthern rate directly via FirnParameters
    params = FirnParameters(kc0=kc0_expr, kc1=kc1_expr, kg=kg_fixed, Ec=Ec_expr,
                            basal_heat_flux_W_m2=Q_geo_expr)
    model  = FirnModel(params)
    solver = FirnColumnSolver(model, surface_id=SURFACE_ID)

    Ts    = make_real(R_space, phase1_T[0], "Ts")
    accum = make_real(R_space, phase1_A[0], "accum")
    Hs    = make_real(R_space, float(params.c_i) * (phase1_T[0] - float(params.T_ref)), "Hs")
    dt    = make_real(R_space, dt_s, "dt")

    # CG1 function for rho_surf BC value (R-to-CG1 interpolate works for adjoint)
    rho_surf_fs = fd.Function(V, name="rho_surf_fs")
    rho_surf_fs.interpolate(rho_surf_expr)

    RHO_S_FIXED = float(INITIAL["rho_surf"])
    bcs, bc_vals = make_bcs(V, params, accum, RHO_S_FIXED, Hs)
    w_surf_bc = bc_vals["w_surf_bc"]
    bc_age = None  # DG uses inflow value for age (default = 0)

    # DG inflow values for density and age (upwind flux BC)
    inflow_values = {
        "rho": rho_surf_expr,
    }

    # -------------------------------------------------------------------------
    # Load Buizert 2021 forcing for on-tape spinup
    # -------------------------------------------------------------------------
    print("\nLoading Buizert 2021 SPICEcore reconstruction ...")
    _buiz_T_df = pd.read_csv(BUIZERT_TEMP_CSV).sort_values("year_CE")
    _buiz_A_df = pd.read_csv(BUIZERT_ACCUM_CSV).sort_values("year_CE")
    _spin_years_1yr = np.arange(float(SPINUP_START_CE), float(ERA5_START_YEAR), 1.0)
    _spin_T_1yr = np.interp(_spin_years_1yr, _buiz_T_df["year_CE"].values,
                            _buiz_T_df["temp"].values + 273.15)
    _spin_A_1yr = np.interp(_spin_years_1yr, _buiz_A_df["year_CE"].values,
                            _buiz_A_df["accum"].values)
    _blk = int(round(SPINUP_DT_YEARS))
    _n_blk = len(_spin_T_1yr) // _blk
    spin_T = np.array([_spin_T_1yr[i*_blk:(i+1)*_blk].mean() for i in range(_n_blk)])
    spin_A = np.array([_spin_A_1yr[i*_blk:(i+1)*_blk].mean() for i in range(_n_blk)])
    n_spin = len(spin_T)
    _spinup_years = n_spin * SPINUP_DT_YEARS
    spinup_dt_s = float(SPINUP_DT_YEARS * YEAR_S)
    print(f"  Spinup: {SPINUP_START_CE}-{ERA5_START_YEAR} CE ({n_spin} steps, dt={SPINUP_DT_YEARS:.0f} yr)")
    print(f"  T range: {spin_T.min()-273.15:.1f} to {spin_T.max()-273.15:.1f} °C")
    print(f"  A range: {spin_A.min():.4f} to {spin_A.max():.4f} m/yr")
    print(f"  Total tape: {n_spin} spinup + {n_era5_total} ERA5 = {n_spin + n_era5_total} steps")

    # -------------------------------------------------------------------------
    # Forward mapping (ALL on tape: spinup + ERA5)
    # -------------------------------------------------------------------------
    def forward_mapping():
        """Controls -> objective. Spinup + ERA5 all on tape."""
        rho_surf_fs.interpolate(rho_surf_expr)

        # --- Simple analytical IC ---
        _T0 = float(spin_T[0])
        _A0 = float(spin_A[0])
        H.assign(float(params.c_i) * (_T0 - float(params.T_ref)))
        rho.interpolate(fd.Constant(float(INITIAL["rho_surf"])) +
                        (fd.Constant(float(params.rho_i)) - fd.Constant(float(INITIAL["rho_surf"]))) *
                        (1.0 - fd.exp(-_depth_expr / 20.0)))
        w.assign(-_A0 * float(params.rho_i) / float(INITIAL["rho_surf"]) / float(YEAR_S))
        sigma.assign(0.0)
        r2.interpolate(fd.Constant(float(INITIAL["r2_surf"])))
        age.assign(0.0)

        # --- Phase 1: Buizert spinup (on tape, dt=10yr) ---
        dt.assign(spinup_dt_s)
        for k in range(n_spin):
            T_k = float(spin_T[k])
            A_k = float(spin_A[k])
            Ts.assign(T_k)
            Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)
            accum.assign(A_k)
            w_surf_bc.interpolate(w_surf_value(params, accum, fd.Constant(RHO_S_FIXED)))

            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                inflow_values=inflow_values,
                age=age, age_boundary_condition=bc_age,
            )

        # --- Phase 2: ERA5 1940 -> 2015 (on tape, dt=0.5yr) ---
        dt.assign(dt_s)
        for k in range(n_era5_total):
            T_k = float(era5_T_sub[k])
            A_k = float(era5_A_sub[k])
            Ts.assign(T_k)
            Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)
            accum.assign(A_k)
            w_surf_bc.interpolate(w_surf_value(params, accum, fd.Constant(RHO_S_FIXED)))

            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                inflow_values=inflow_values,
                age=age, age_boundary_condition=bc_age,
            )

        # === Misfit at CORE_YEAR (icepack-style: 1/N normalization) ===
        # J_data = (1/N) * ½ Σ ((y - y_obs)/σ)²  →  O(1) when fit is good
        J = 0.0

        # Density misfit
        if vom_rho is not None:
            rho_pred_fn.interpolate(rho)
            r_rho = (rho_pred_fn - rho_obs_fn) / rho_sig_fn
            J = J + 0.5 / float(N_rho) * fd.assemble(
                r_rho * r_rho * fd.dx(domain=vom_rho))

        # Age misfit
        if vom_age is not None:
            age_pred_fn.interpolate(age)
            r_age = (age_pred_fn - age_obs_fn) / age_sig_fn
            J = J + 0.5 / float(N_age) * fd.assemble(
                r_age * r_age * fd.dx(domain=vom_age))

        # Borehole temperature misfit
        if vom_T is not None:
            T_pred_fn.interpolate(H)
            r_T = (T_pred_fn - T_obs_fn) / T_sig_fn
            J = J + 0.5 / float(N_T) * fd.assemble(
                r_T * r_T * fd.dx(domain=vom_T))

        # Priors (normalized by n_ctrl so J_prior ~ O(1) per control)
        if USE_PRIOR:
            _n_ctrl_inv = 1.0 / float(n_ctrl)
            # Log-space priors for positive-definite controls
            for log_ctrl, truth_val, sig_log in [
                (log_kc0,    TRUTH["kc0"],      PRIOR_SIGMA_LOG),
                (log_kc1,    TRUTH["kc1"],      PRIOR_SIGMA_LOG),
                (log_Ec,     TRUTH["Ec"],        EC_PRIOR_SIGMA_LOG),
                (log_rho_s,  TRUTH["rho_surf"],  RHO_SURF_PRIOR_SIGMA_LOG),
                (log_Qgeo,   TRUTH["Q_geo"],     QGEO_PRIOR_SIGMA_LOG),
            ]:
                _prior_fn.interpolate(
                    (log_ctrl - math.log(float(truth_val))) / sig_log
                )
                J = J + _n_ctrl_inv * 0.5 * fd.assemble(_prior_fn * _prior_fn * dx_domain) / domain_len

        return J

    # -------------------------------------------------------------------------
    # Helper: set control values from numpy vector
    # -------------------------------------------------------------------------
    def set_controls(x_vec):
        for c, val in zip(controls, x_vec):
            c.assign(float(val))

    def get_controls():
        return np.array([float(c.dat.data_ro[0]) for c in controls])

    def extract_gradient(dJ_list):
        """Convert pyadjoint Cofunction derivatives to scalar gradients.

        For R-space controls, rf.derivative() returns L2 dual Cofunctions.
        The scalar gradient dJ/dc = cofunction_dof * domain_length.
        """
        return np.array([float(g.dat.data_ro[0]) * domain_len for g in dJ_list])

    # -------------------------------------------------------------------------
    # Build tape + ReducedFunctional
    # -------------------------------------------------------------------------
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()

    x0 = get_controls()
    J_init = forward_mapping()
    rf = ReducedFunctional(J_init, [Control(c) for c in controls])
    pause_annotation()

    print(f"\nInitial objective J(m0) = {float(J_init):.6e}")

    # -------------------------------------------------------------------------
    # Taylor test (verify adjoint)
    # -------------------------------------------------------------------------
    print("\nTaylor remainder test ...")
    with stop_annotating():
        set_controls(x0)
    with stop_annotating():
        J0_val = float(rf(controls))
        g_list = rf.derivative()
    g0_vec = extract_gradient(g_list)

    rng = np.random.default_rng(SEED)
    d = rng.standard_normal(len(x0))
    d = d / (np.linalg.norm(d) + 1e-30) * 0.01
    gdotd = g0_vec @ d

    print(f"  J(m)    = {J0_val:.6e}")
    print(f"  <dJ, d> = {gdotd:.6e}")
    print("  eps        |J(m+eps d)-J(m)|     |remainder|          order")
    epsilons = [1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3]
    prev_r, prev_eps = None, None
    for eps_val in epsilons:
        m1 = np.clip(x0 + eps_val * d, lb_vec, ub_vec)
        try:
            with stop_annotating():
                set_controls(m1)
                J1_val = float(rf(controls))
        except Exception as e:
            print(f"  {eps_val:9.3e}  FAILED ({type(e).__name__})")
            continue
        r1 = abs(J1_val - J0_val - eps_val * gdotd)
        order = ""
        if prev_r is not None and r1 > 0 and prev_r > 0:
            order = f"{math.log(prev_r / r1) / math.log(prev_eps / eps_val):.2f}"
        print(f"  {eps_val:9.3e}  {abs(J1_val - J0_val):20.6e}  {r1:20.6e}  {order}")
        prev_r, prev_eps = r1, eps_val

    # Reset controls and state
    with stop_annotating():
        set_controls(x0)
        rf(controls)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        depth_nodes = (H0 - mesh.coordinates.dat.data_ro[:, 0]
                       if mesh.coordinates.dat.data_ro.ndim == 2
                       else H0 - mesh.coordinates.dat.data_ro)
        idx_sort = np.argsort(depth_nodes)
        d_plot = depth_nodes[idx_sort]

        rho_init = rho.dat.data_ro[idx_sort].copy()
        age_init = age.dat.data_ro[idx_sort].copy() / float(YEAR_S)

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        ax = axes[0]
        ax.plot(rho_init, d_plot, "b-", lw=1.5, label="Model (initial)")
        ax.errorbar(obs_rho_val, obs_rho_depth, xerr=sigma_rho,
                    fmt="r.", ms=3, elinewidth=0.5, label="SP19")
        ax.set_xlabel("Density (kg/m3)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title(f"Density at {CORE_YEAR}")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.plot(age_init, d_plot, "b-", lw=1.5, label="Model (initial)")
        ax.errorbar(obs_age_s / float(YEAR_S), obs_age_depth,
                    xerr=sigma_age / float(YEAR_S),
                    fmt="r.", ms=3, elinewidth=0.5, label="SP19")
        ax.set_xlabel("Age (yr)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title(f"Age at {CORE_YEAR}")
        ax.legend(fontsize=8)

        fig.suptitle("South Pole v5_sp19 - initial model vs data")
        fig.tight_layout()
        diag_path = OUT_DIR / f"{OUT_PREFIX}_initial_vs_data.png"
        fig.savefig(diag_path, dpi=150)
        print(f"\nInitial-vs-data plot saved to {diag_path}")
        plt.close(fig)
    except Exception as _e:
        print(f"(Initial diagnostic plot skipped: {_e})")

    # -------------------------------------------------------------------------
    # Lin-More Trust-Region Optimisation
    # -------------------------------------------------------------------------
    J_hist: list[float] = []
    _n_eval = [0]

    def _eval_cb(val, *args, **kwargs):
        _n_eval[0] += 1
        J_hist.append(float(val))
        if _n_eval[0] % 5 == 0 or _n_eval[0] == 1:
            print(f"  [iter {_n_eval[0]:03d}] J = {float(val):.6e}")
    rf.eval_cb_post = _eval_cb

    minimisation_parameters["Status Test"]["Iteration Limit"] = MAX_ITER
    minimisation_parameters["Status Test"]["Gradient Tolerance"] = GRAD_TOL
    minimisation_parameters["General"]["Secant"]["Type"] = "Limited-Memory BFGS"
    minimisation_parameters["General"]["Secant"]["Maximum Storage"] = 25
    minimisation_parameters["General"]["Initial Trust Region Radius"] = 1.0

    min_problem = MinimizationProblem(rf, bounds=bounds_pairs)
    try:
        optimiser = LinMoreOptimiser(
            min_problem, minimisation_parameters,
            checkpoint_dir=str(OUT_DIR / f"{OUT_PREFIX}_optim_checkpoint"),
        )
    except TypeError:
        optimiser = LinMoreOptimiser(min_problem, minimisation_parameters,
                                    auto_checkpoint=False)

    print(f"\nStarting Lin-More trust-region optimisation (max {MAX_ITER} iters) ...")
    try:
        optimiser.run()
    except fd.exceptions.ConvergenceError as e:
        print(f"\n  Optimizer stopped early (ConvergenceError after {_n_eval[0]} evals)")
        print(f"  Continuing with best result so far ...")

    # Commit control checkpoints
    for c in controls:
        if hasattr(c, "block_variable") and getattr(c.block_variable, "checkpoint", None) is not None:
            c.assign(c.block_variable.checkpoint)

    x_map = get_controls()

    # -------------------------------------------------------------------------
    # Extract MAP results
    # -------------------------------------------------------------------------
    m_map_phys = {}
    m_map_raw = {}
    for i, nm in enumerate(names):
        m_map_raw[nm] = float(x_map[i])
        if nm in LOG_CONTROLS:
            m_map_phys[nm] = math.exp(float(x_map[i]))
        elif nm == "div_h":
            m_map_phys[nm] = DIV_H_INIT + DIV_H_SIGMA * float(x_map[i])
        else:
            m_map_phys[nm] = float(x_map[i])

    print("\nMAP parameter estimates:")
    for nm in names:
        print(f"  {nm:12s} = {m_map_phys[nm]:.6e}  (initial: {INITIAL[nm]:.6e})")

    # Re-run forward at MAP to get profiles
    with stop_annotating():
        set_controls(x_map)
        J_map = float(rf(controls))
    print(f"\nFinal J(MAP) = {J_map:.6e}")

    # -------------------------------------------------------------------------
    # Misfit-only Hessian via FD of misfit gradient (Isaac et al. 2015)
    #
    # Following UQ_forward framework:
    #   1. Compute H_mis = d²J_mis/dm² (misfit only, no prior)
    #   2. Prior precision A = diag(1/σ²_prior) (analytical)
    #   3. Eigendecomposition of A⁻¹ H_mis → eigenvalues λᵢ, eigenvectors wᵢ
    #   4. Posterior variance for QoI Q:
    #      σ²_Q = ∇Q^T Γ_prior ∇Q - Σᵢ (λᵢ/(λᵢ+1)) (∇Q · wᵢ)²
    # -------------------------------------------------------------------------

    # Build a misfit-only ReducedFunctional (no priors in the objective)
    print("\nBuilding misfit-only tape for Hessian computation ...")
    tape_mis = get_working_tape()
    tape_mis.clear_tape()
    continue_annotation()

    set_controls(x_map)

    def forward_mapping_misfit_only():
        """Forward mapping returning misfit only (no prior terms).
        Same spinup+ERA5 as forward_mapping but without priors."""
        rho_surf_fs.interpolate(rho_surf_expr)

        # --- Simple analytical IC ---
        _T0 = float(spin_T[0])
        _A0 = float(spin_A[0])
        H.assign(float(params.c_i) * (_T0 - float(params.T_ref)))
        rho.interpolate(fd.Constant(float(INITIAL["rho_surf"])) +
                        (fd.Constant(float(params.rho_i)) - fd.Constant(float(INITIAL["rho_surf"]))) *
                        (1.0 - fd.exp(-_depth_expr / 20.0)))
        w.assign(-_A0 * float(params.rho_i) / float(INITIAL["rho_surf"]) / float(YEAR_S))
        sigma.assign(0.0)
        r2.interpolate(fd.Constant(float(INITIAL["r2_surf"])))
        age.assign(0.0)

        # --- Phase 1: Buizert spinup ---
        dt.assign(spinup_dt_s)
        for k in range(n_spin):
            T_k = float(spin_T[k])
            A_k = float(spin_A[k])
            Ts.assign(T_k)
            Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)
            accum.assign(A_k)
            w_surf_bc.interpolate(w_surf_value(params, accum, fd.Constant(RHO_S_FIXED)))
            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                inflow_values=inflow_values,
                age=age, age_boundary_condition=bc_age,
            )

        # --- Phase 2: ERA5 ---
        dt.assign(dt_s)
        for k in range(n_era5_total):
            T_k = float(era5_T_sub[k])
            A_k = float(era5_A_sub[k])
            Ts.assign(T_k)
            Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)
            accum.assign(A_k)
            w_surf_bc.interpolate(w_surf_value(params, accum, fd.Constant(RHO_S_FIXED)))
            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                inflow_values=inflow_values,
                age=age, age_boundary_condition=bc_age,
            )

        # Misfit only (no prior) — icepack-style 1/N normalization
        J_mis = 0.0
        if vom_rho is not None:
            rho_pred_fn.interpolate(rho)
            r_rho = (rho_pred_fn - rho_obs_fn) / rho_sig_fn
            J_mis = J_mis + 0.5 / float(N_rho) * fd.assemble(
                r_rho * r_rho * fd.dx(domain=vom_rho))
        if vom_age is not None:
            age_pred_fn.interpolate(age)
            r_age = (age_pred_fn - age_obs_fn) / age_sig_fn
            J_mis = J_mis + 0.5 / float(N_age) * fd.assemble(
                r_age * r_age * fd.dx(domain=vom_age))
        if vom_T is not None:
            T_pred_fn.interpolate(H)
            r_T = (T_pred_fn - T_obs_fn) / T_sig_fn
            J_mis = J_mis + 0.5 / float(N_T) * fd.assemble(
                r_T * r_T * fd.dx(domain=vom_T))
        return J_mis

    J_mis_init = forward_mapping_misfit_only()
    rf_mis = ReducedFunctional(J_mis_init, [Control(c) for c in controls])
    pause_annotation()

    print(f"  J_mis(MAP) = {float(J_mis_init):.6e}")

    # -------------------------------------------------------------------------
    # FD Hessian of misfit only
    # -------------------------------------------------------------------------
    n_ctrl = len(controls)
    print(f"\nComputing {n_ctrl}x{n_ctrl} misfit Hessian (FD eps = {HESSIAN_EPS}) ...")

    # Baseline misfit gradient at MAP
    hessian_ok = True
    try:
        with stop_annotating():
            set_controls(x_map)
            rf_mis(controls)
            dJ_base = rf_mis.derivative()
            g_mis_base = extract_gradient(dJ_base)
        print(f"  Baseline misfit gradient: |g_mis| = {np.linalg.norm(g_mis_base):.2e}")
    except Exception as e:
        print(f"  Baseline misfit gradient FAILED: {e}")
        hessian_ok = False

    H_mis = np.zeros((n_ctrl, n_ctrl))

    if hessian_ok:
        for i in range(n_ctrl):
            x_pert = x_map.copy()
            eps_i = HESSIAN_EPS
            if x_map[i] + eps_i > ub_vec[i] - 1e-10:
                eps_i = -HESSIAN_EPS
            x_pert[i] += eps_i
            x_pert = np.clip(x_pert, lb_vec, ub_vec)
            actual_eps = x_pert[i] - x_map[i]
            if abs(actual_eps) < 1e-14:
                print(f"  H_mis column {i} ({names[i]}): SKIPPED (at bound)")
                continue
            try:
                with stop_annotating():
                    set_controls(x_pert)
                    rf_mis(controls)
                    dJ_pert = rf_mis.derivative()
                    g_pert = extract_gradient(dJ_pert)
                H_mis[:, i] = (g_pert - g_mis_base) / actual_eps
                print(f"  H_mis column {i} ({names[i]}): done")
            except Exception as e:
                print(f"  H_mis column {i} ({names[i]}): FAILED ({type(e).__name__})")

    # Symmetrize
    H_mis = 0.5 * (H_mis + H_mis.T)

    # Clamp any small negative eigenvalues to zero (numerical noise)
    eigvals_mis, eigvecs_mis = np.linalg.eigh(H_mis)
    eigvals_mis = np.maximum(eigvals_mis, 0.0)
    H_mis = eigvecs_mis @ np.diag(eigvals_mis) @ eigvecs_mis.T
    print(f"  H_mis eigenvalues (clamped): {eigvals_mis}")

    # -------------------------------------------------------------------------
    # Prior precision and covariance (analytical for scalar controls)
    # -------------------------------------------------------------------------
    # A_prior = diag(1/σ²_prior)  (prior precision in log-space)
    # Γ_prior = diag(σ²_prior)    (prior covariance in log-space)
    A_prior = np.diag(1.0 / prior_sigmas**2)
    Gamma_prior = np.diag(prior_sigmas**2)

    # -------------------------------------------------------------------------
    # Eigendecomposition of A_prior^{-1} @ H_mis  (Isaac et al. 2015)
    # -------------------------------------------------------------------------
    # Solves:  A^{-1} H_mis w_i = λ_i w_i
    # λ_i measures how much the data constrains direction w_i beyond the prior.
    # λ_i >> 1: data strongly constrains this direction
    # λ_i << 1: prior dominates (no data information)
    AinvH = Gamma_prior @ H_mis   # A^{-1} H = Γ_prior H_mis
    eigvals_gen, eigvecs_gen = np.linalg.eig(AinvH)

    # Sort by descending eigenvalue (most constrained first)
    idx_sort_eig = np.argsort(-eigvals_gen.real)
    eigvals_gen = eigvals_gen.real[idx_sort_eig]
    eigvecs_gen = eigvecs_gen.real[:, idx_sort_eig]

    print("\n  Generalized eigenvalues (A⁻¹ H_mis):")
    for i in range(n_ctrl):
        D_i = eigvals_gen[i] / (eigvals_gen[i] + 1.0) if eigvals_gen[i] > 0 else 0.0
        print(f"    λ_{i} = {eigvals_gen[i]:12.4e}  "
              f"D_{i} = λ/(λ+1) = {D_i:.4f}  "
              f"({'data constrains' if D_i > 0.5 else 'prior dominates'})")

    # -------------------------------------------------------------------------
    # Posterior covariance (exact for scalar controls)
    # -------------------------------------------------------------------------
    # H_post = H_mis + A_prior  →  Cov_post = H_post^{-1}
    H_posterior = H_mis + A_prior
    eigvals_post = np.linalg.eigvalsh(H_posterior)
    print(f"\n  H_posterior eigenvalues: {eigvals_post}")

    if np.all(eigvals_post > 0):
        Cov_posterior = np.linalg.inv(H_posterior)
        sigma_post_log = np.sqrt(np.diag(Cov_posterior))
    else:
        print("  WARNING: H_posterior not positive definite — using prior covariance")
        Cov_posterior = Gamma_prior.copy()
        sigma_post_log = prior_sigmas.copy()

    # Correlation matrix
    D_inv = np.diag(1.0 / (sigma_post_log + 1e-30))
    Corr_posterior = D_inv @ Cov_posterior @ D_inv

    print("\n  Posterior parameter uncertainty (log-space):")
    print(f"  {'Control':12s}  {'MAP':>12s}  {'σ_prior':>10s}  {'σ_post':>10s}  {'reduction':>10s}")
    for i, nm in enumerate(names):
        red = 1.0 - sigma_post_log[i] / prior_sigmas[i]
        print(f"  {nm:12s}  {m_map_phys[nm]:12.4e}  {prior_sigmas[i]:10.4f}  "
              f"{sigma_post_log[i]:10.4f}  {red:9.1%}")

    print("\n  Posterior correlation matrix:")
    print(f"  {'':12s}  " + "  ".join(f"{nm:>10s}" for nm in names))
    for i, nm_i in enumerate(names):
        row = "  ".join(f"{Corr_posterior[i, j]:10.3f}" for j in range(n_ctrl))
        print(f"  {nm_i:12s}  {row}")

    # -------------------------------------------------------------------------
    # Posterior profile uncertainty via sensitivity propagation
    #
    # For a QoI Q_j (e.g., rho or age at observation point j):
    #   σ²_Q = ∇Q^T Γ_prior ∇Q - Σᵢ (λᵢ/(λᵢ+1)) (∇Q · wᵢ)²
    #
    # ∇Q is the 5-vector of sensitivities dQ/dm at the MAP.
    # We compute this via FD of the forward model for each obs kernel.
    # -------------------------------------------------------------------------
    print(f"\nComputing profile uncertainty via sensitivity propagation ...")

    # Get mesh info
    depth_nodes = (H0 - mesh.coordinates.dat.data_ro[:, 0]
                   if mesh.coordinates.dat.data_ro.ndim == 2
                   else H0 - mesh.coordinates.dat.data_ro)
    idx_sort = np.argsort(depth_nodes)
    d_plot = depth_nodes[idx_sort]

    # MAP profiles (from the last forward run)
    rho_map_profile = rho.dat.data_ro[idx_sort].copy()
    age_map_profile = age.dat.data_ro[idx_sort].copy()

    # Compute sensitivities via FD of a full unannotated forward run.
    # We run the full forward model (not tape replay) to get fresh state.

    def _run_forward_and_observe(x_vec):
        """Run full forward model unannotated (spinup+ERA5) and return VOM observations."""
        set_controls(x_vec)
        rho_surf_fs.interpolate(rho_surf_expr)

        # Analytical IC
        _T0 = float(spin_T[0])
        _A0 = float(spin_A[0])
        H.assign(float(params.c_i) * (_T0 - float(params.T_ref)))
        rho.interpolate(fd.Constant(float(INITIAL["rho_surf"])) +
                        (fd.Constant(float(params.rho_i)) - fd.Constant(float(INITIAL["rho_surf"]))) *
                        (1.0 - fd.exp(-_depth_expr / 20.0)))
        w.assign(-_A0 * float(params.rho_i) / float(INITIAL["rho_surf"]) / float(YEAR_S))
        sigma.assign(0.0)
        r2.interpolate(fd.Constant(float(INITIAL["r2_surf"])))
        age.assign(0.0)

        # Spinup
        dt.assign(spinup_dt_s)
        for k in range(n_spin):
            T_k, A_k = float(spin_T[k]), float(spin_A[k])
            Ts.assign(T_k); Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)
            w_surf_bc.interpolate(w_surf_value(params, accum, fd.Constant(RHO_S_FIXED)))
            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                inflow_values=inflow_values,
                age=age, age_boundary_condition=bc_age,
            )

        # ERA5
        dt.assign(dt_s)
        for k in range(n_era5_total):
            T_k, A_k = float(era5_T_sub[k]), float(era5_A_sub[k])
            Ts.assign(T_k); Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)
            w_surf_bc.interpolate(w_surf_value(params, accum, fd.Constant(RHO_S_FIXED)))
            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                inflow_values=inflow_values,
                age=age, age_boundary_condition=bc_age,
            )
        # Point evaluation via VOM interpolation
        rho_pred_fn.interpolate(rho)
        rho_obs = rho_pred_fn.dat.data_ro.copy()
        age_obs = np.array([])
        if vom_age is not None:
            age_pred_fn.interpolate(age)
            age_obs = age_pred_fn.dat.data_ro.copy()
        return rho_obs, age_obs

    # Baseline at MAP
    rho_pred_base, age_pred_base = _run_forward_and_observe(x_map)
    _age_info = ""
    if len(age_pred_base) > 0:
        _age_info = f"  age range [{age_pred_base.min()/float(YEAR_S):.0f}, {age_pred_base.max()/float(YEAR_S):.0f}] yr"
    print(f"  Baseline: rho range [{rho_pred_base.min():.1f}, {rho_pred_base.max():.1f}]{_age_info}")

    n_rho_obs = len(obs_rho_depth)
    n_age_obs = len(obs_age_depth)
    grad_rho = np.zeros((n_rho_obs, n_ctrl))
    grad_age = np.zeros((n_age_obs, n_ctrl))

    for i in range(n_ctrl):
        x_pert = x_map.copy()
        eps_i = HESSIAN_EPS
        if x_map[i] + eps_i > ub_vec[i] - 1e-10:
            eps_i = -HESSIAN_EPS
        x_pert[i] += eps_i
        x_pert = np.clip(x_pert, lb_vec, ub_vec)
        actual_eps = x_pert[i] - x_map[i]
        if abs(actual_eps) < 1e-14:
            continue
        try:
            rho_pred_pert, age_pred_pert = _run_forward_and_observe(x_pert)
            grad_rho[:, i] = (rho_pred_pert - rho_pred_base) / actual_eps
            grad_age[:, i] = (age_pred_pert - age_pred_base) / actual_eps
            print(f"  Sensitivity column {i} ({names[i]}): "
                  f"|∂rho/∂m|={np.linalg.norm(grad_rho[:, i]):.2e}  "
                  f"|∂age/∂m|={np.linalg.norm(grad_age[:, i]):.2e}")
        except Exception as e:
            print(f"  Sensitivity column {i} ({names[i]}): FAILED ({type(e).__name__})")

    # Reset to MAP
    _run_forward_and_observe(x_map)

    # Compute posterior variance for each observation point
    # σ²_Q = ∇Q^T Γ_prior ∇Q - Σᵢ (λᵢ/(λᵢ+1)) (∇Q · wᵢ)²
    D_reduction = np.array([lam / (lam + 1.0) if lam > 0 else 0.0 for lam in eigvals_gen])

    def compute_posterior_sigma(grad_Q):
        """Compute prior and posterior sigma for a QoI with gradient grad_Q (n_obs, n_ctrl)."""
        n_obs = grad_Q.shape[0]
        sigma_prior_Q = np.zeros(n_obs)
        sigma_post_Q = np.zeros(n_obs)
        for j in range(n_obs):
            gj = grad_Q[j, :]  # (n_ctrl,)
            # Prior variance: ∇Q^T Γ_prior ∇Q
            var_prior = gj @ Gamma_prior @ gj
            # Data reduction: Σᵢ D_i (∇Q · wᵢ)²
            var_reduction = 0.0
            for k in range(n_ctrl):
                proj = gj @ eigvecs_gen[:, k]
                var_reduction += D_reduction[k] * proj**2
            var_post = max(var_prior - var_reduction, 0.0)
            sigma_prior_Q[j] = math.sqrt(max(var_prior, 0.0))
            sigma_post_Q[j] = math.sqrt(var_post)
        return sigma_prior_Q, sigma_post_Q

    sigma_prior_rho, sigma_post_rho = compute_posterior_sigma(grad_rho)
    sigma_prior_age, sigma_post_age = compute_posterior_sigma(grad_age)

    print(f"\n  Density: mean σ_prior = {sigma_prior_rho.mean():.2f} kg/m³, "
          f"mean σ_post = {sigma_post_rho.mean():.2f} kg/m³")
    if n_age_obs > 0:
        print(f"  Age:     mean σ_prior = {sigma_prior_age.mean()/float(YEAR_S):.2f} yr, "
              f"mean σ_post = {sigma_post_age.mean()/float(YEAR_S):.2f} yr")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    map_out = {
        "script":     str(Path(__file__).name),
        "era5_window": [int(ERA5_START_YEAR), int(ERA5_END_YEAR)],
        "core_year":  int(CORE_YEAR),
        "n_era5":     int(n_phase1),
        "obs_types":  ["rho", "age"],
        "J_map":      float(J_map),
        "J_hist":     J_hist,
        "m_map_phys": m_map_phys,
        "m_map_raw":  m_map_raw,
        "INITIAL":    INITIAL,
        "TRUTH":      TRUTH,
        "BOUNDS":     {k: list(v) for k, v in BOUNDS.items()},
        "H_mis":          H_mis.tolist(),
        "A_prior":        A_prior.tolist(),
        "H_posterior":    H_posterior.tolist(),
        "cov_posterior":  Cov_posterior.tolist(),
        "corr_posterior": Corr_posterior.tolist(),
        "sigma_post_log": sigma_post_log.tolist(),
        "sigma_prior_log": prior_sigmas.tolist(),
        "eigvals_gen":    eigvals_gen.tolist(),
        "eigvecs_gen":    eigvecs_gen.tolist(),
    }
    map_path = OUT_DIR / f"{OUT_PREFIX}.json"
    with open(map_path, "w") as fh:
        json.dump(map_out, fh, indent=2)
    print(f"\nMAP results saved to {map_path}")

    # Save profile uncertainties
    uq_path = OUT_DIR / f"{OUT_PREFIX}_uq.npz"
    np.savez(uq_path,
             depth_rho_obs=obs_rho_depth,
             depth_age_obs=obs_age_depth,
             depth_profile=d_plot,
             rho_map=rho_map_profile,
             age_map=age_map_profile,
             sigma_prior_rho=sigma_prior_rho,
             sigma_post_rho=sigma_post_rho,
             sigma_prior_age=sigma_prior_age,
             sigma_post_age=sigma_post_age,
             grad_rho=grad_rho,
             grad_age=grad_age,
             eigvals_gen=eigvals_gen,
             eigvecs_gen=eigvecs_gen,
             x_map=x_map,
             names=names)
    print(f"UQ profiles saved to {uq_path}")

    # -------------------------------------------------------------------------
    # Profile plots with uncertainty bands
    # -------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        ax = axes[0]
        # Prior and posterior CI at observation points
        ax.fill_betweenx(obs_rho_depth,
                         rho_pred_base - 2*sigma_prior_rho,
                         rho_pred_base + 2*sigma_prior_rho,
                         alpha=0.1, color="gray", label="Prior 95% CI")
        ax.fill_betweenx(obs_rho_depth,
                         rho_pred_base - 2*sigma_post_rho,
                         rho_pred_base + 2*sigma_post_rho,
                         alpha=0.3, color="blue", label="Posterior 95% CI")
        ax.plot(rho_map_profile, d_plot, "b-", lw=1.5, label=f"MAP at {CORE_YEAR}")
        ax.errorbar(obs_rho_val, obs_rho_depth, xerr=sigma_rho,
                    fmt="r.", ms=3, elinewidth=0.5, label=f"SP19 ({CORE_YEAR})")
        ax.set_xlabel("Density (kg/m3)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title(f"Density at {CORE_YEAR}")
        ax.legend(fontsize=7)

        ax = axes[1]
        if len(obs_age_depth) > 0:
            ax.fill_betweenx(obs_age_depth,
                             (age_pred_base - 2*sigma_prior_age) / float(YEAR_S),
                             (age_pred_base + 2*sigma_prior_age) / float(YEAR_S),
                             alpha=0.1, color="gray", label="Prior 95% CI")
            ax.fill_betweenx(obs_age_depth,
                             (age_pred_base - 2*sigma_post_age) / float(YEAR_S),
                             (age_pred_base + 2*sigma_post_age) / float(YEAR_S),
                             alpha=0.3, color="blue", label="Posterior 95% CI")
            ax.errorbar(obs_age_s / float(YEAR_S), obs_age_depth,
                        xerr=sigma_age / float(YEAR_S),
                        fmt="r.", ms=3, elinewidth=0.5, label=f"SP19 ({CORE_YEAR})")
        ax.plot(age_map_profile / float(YEAR_S), d_plot, "b-", lw=1.5, label=f"MAP at {CORE_YEAR}")
        ax.set_xlabel("Age (yr)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title(f"Age at {CORE_YEAR}")
        ax.legend(fontsize=7)

        fig.suptitle(
            f"South Pole MAP inversion v9\n"
            f"rho/age at {CORE_YEAR} | "
            + "  ".join(f"{k}={m_map_phys[k]:.2e}" for k in ["kc0", "kc1", "kg"])
        )
        fig.tight_layout()
        plot_path = OUT_DIR / f"{OUT_PREFIX}_profiles.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Profiles saved to {plot_path}")
        plt.close(fig)

        # Objective history
        if len(J_hist) > 1:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.semilogy(range(1, len(J_hist) + 1), J_hist, "k-o", ms=3)
            ax2.set_xlabel("Function evaluation")
            ax2.set_ylabel("J (log scale)")
            ax2.set_title("Objective history")
            fig2.tight_layout()
            hist_path = OUT_DIR / f"{OUT_PREFIX}_objective_history.png"
            fig2.savefig(hist_path, dpi=150)
            print(f"Objective history saved to {hist_path}")
            plt.close(fig2)

        # Eigenspectrum
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.bar(range(n_ctrl), eigvals_gen, color="steelblue")
        ax3.axhline(1.0, color="red", ls="--", lw=1, label="λ = 1 (prior = data)")
        ax3.set_xticks(range(n_ctrl))
        ax3.set_xticklabels(names, rotation=30, ha="right")
        ax3.set_ylabel("Eigenvalue λ")
        ax3.set_title("Generalized eigenvalues (A⁻¹ H_mis)")
        ax3.legend()
        fig3.tight_layout()
        eig_path = OUT_DIR / f"{OUT_PREFIX}_eigenspectrum.png"
        fig3.savefig(eig_path, dpi=150)
        print(f"Eigenspectrum saved to {eig_path}")
        plt.close(fig3)

    except Exception as _e:
        print(f"(Plotting skipped: {_e})")

    print("\nDone.")
