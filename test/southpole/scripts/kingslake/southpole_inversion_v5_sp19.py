"""southpole_inversion_v5_sp19.py

MAP inversion for South Pole firn densification parameters using SP19
ice-core density and depth-age observations only (co-located data, no ApRES).

Uses pyadjoint (firedrake.adjoint) for automatic differentiation and
gadopt's Lin-More trust-region optimiser.

Post-optimisation: computes the 5x5 Hessian at MAP via finite differences
of the gradient, then derives the posterior covariance, generates posterior
samples, and runs a forward ensemble.

Forward model : 1D firn column (FirnColumnSolver full-density mode)
Spinup        : 1200 years at Buizert 2021 reconstruction, RUN ON TAPE
Taped run     : ERA5 1940 -> 2015 (CORE_YEAR)
  Evaluate rho/age misfit against SP19 ice core at CORE_YEAR.

Controls (log-space):
    kc0      -- densification rate coefficient, low-density regime
    kc1      -- densification rate coefficient, high-density regime
    kg       -- grain-growth rate coefficient
    rho_surf -- surface density boundary condition (kg/m3)
    r2_surf  -- surface grain-radius-squared boundary condition (m2)

Observations:
    rho(z)   -- SP19 ice-core density
    age(z)   -- SP19 ice-core depth-age

Run:
    cd test/southpole/scripts
    OMP_NUM_THREADS=1 python southpole_inversion_v5_sp19.py
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
BUIZERT_TEMP_CSV  = PROCESSED_DIR / "buizert2021_spice_temp.csv"
BUIZERT_ACCUM_CSV = PROCESSED_DIR / "buizert2021_spice_accum.csv"

OUT_DIR = _SOUTHPOLE / "results"

# =============================================================================
# Settings
# =============================================================================

# Mesh / geometry
H0 = 130.0        # column height (m)
NZ = 100           # number of elements
STRETCH_P = 2.5    # power-law stretching exponent
SURFACE_ID = 2     # Firedrake boundary marker for x = H0

# Time
SPINUP_YEARS    = 1200
SPINUP_DT_YEARS = 10.0
ERA5_START_YEAR = 1940
CORE_YEAR       = 2015    # SP19 ice core year (density + age observations)
ERA5_END_YEAR   = CORE_YEAR
DT_YEARS        = 1.0

# ERA5 bias corrections
ERA5_T_OFFSET_K    = -4.71   # K
ERA5_A_OFFSET_M_YR = +0.0163 # m ice eq/yr
SPINUP_T_K         = 221.87  # K (-51.28 C, Buizert 2021)

# Observation depth limits
RHO_OBS_DEPTH_MAX = H0
AGE_OBS_YEAR_MIN  = 0
AGE_OBS_STRIDE    = 5         # subsample age obs to ~match density count
USE_AGE_OBS       = True      # include age observations in the misfit

# Observation uncertainties
SIGMA_RHO_ABS   = 15.0      # kg/m3
SIGMA_RHO_REL   = 0.03
SIGMA_AGE_ABS_S = 1.0 * float(YEAR_S)  # 1 year
SIGMA_AGE_REL   = 0.02
USE_UNIFORM_OBS_SIGMA = True
OBS_SIGMA_SCALE = 1.0

# Observation operator
OBS_MISFIT_METHOD = "kernel"
OBS_KERNEL_SIGMA_M = 0.5

OUT_PREFIX = f"southpole_map_v5_sp19"

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
    Ec       = float(getattr(P0, "Ec",     60e3)),
    rho_surf = 350.0,
    r2_surf  = 2.5e-7,
    div_h    = DIV_H_INIT,
)

INITIAL = dict(
    kc0      = float(getattr(P0, "kc0",    9.2e-9)),
    kc1      = float(getattr(P0, "kc1",    3.7e-9)),
    kg       = float(getattr(P0, "kg",     1.3e-7)),
    Ec       = 58e3,                       # J/mol, closer to SP19 fit than default 60e3
    rho_surf = 350.0,
    r2_surf  = 2.5e-7,
    div_h    = DIV_H_INIT,
)

BOUNDS = dict(
    kc0      = (1.0e-11, 1.0e-3),
    kc1      = (1.0e-11, 1.0e-3),
    kg       = (1.0e-13, 1.0e-3),
    Ec       = (20e3,    120e3),      # J/mol, activation energy for creep
    rho_surf = (1.0,     917.0),
    r2_surf  = (1.0e-9, 1.0e-6),
    div_h    = (-10.0, 10.0),  # normalized units (div_h_norm), ~10 sigma
)

# Priors
USE_PRIOR = True
PRIOR_SIGMA_LOG = 1.5
EC_PRIOR_SIGMA_LOG       = 0.5        # tighter: exp(0.5)≈1.65x variation
RHO_SURF_PRIOR_SIGMA_LOG = 0.25
R2_SURF_PRIOR_SIGMA_LOG  = 0.8
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


def load_buizert_spinup(temp_csv, accum_csv, spinup_start_year_ce, era5_start_year=ERA5_START_YEAR):
    df_T = pd.read_csv(temp_csv).sort_values("year_CE")
    df_A = pd.read_csv(accum_csv).sort_values("year_CE")
    years_annual = np.arange(float(spinup_start_year_ce), float(era5_start_year), 1.0)
    T_annual = np.interp(years_annual, df_T["year_CE"].values, df_T["temp"].values + 273.15)
    A_annual = np.interp(years_annual, df_A["year_CE"].values, df_A["accum"].values)
    return years_annual, T_annual, A_annual


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

    w_surf is computed from accumulation and a FIXED rho_s (not control-dependent)
    to avoid pyadjoint issues with control-dependent Dirichlet BCs.
    rho and r2 surface BCs are handled by penalty terms.
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
    print("South Pole MAP inversion v5_sp19  (pyadjoint + Lin-More + Hessian UQ)")
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

    dt_s = float(DT_YEARS * YEAR_S)
    print(f"  ERA5: {phase1_years[0]}-{phase1_years[-1]} ({n_phase1} steps) -> SP19 obs at {CORE_YEAR}")

    print("Loading Buizert 2021 spinup reconstruction ...")
    _spinup_start_ce = ERA5_START_YEAR - SPINUP_YEARS
    _spin_years_1yr, _spin_T_1yr, _spin_A_1yr = load_buizert_spinup(
        BUIZERT_TEMP_CSV, BUIZERT_ACCUM_CSV,
        spinup_start_year_ce=_spinup_start_ce,
    )
    # Resample to SPINUP_DT_YEARS resolution (block-average)
    _blk = int(round(SPINUP_DT_YEARS))
    _n_blk = len(_spin_T_1yr) // _blk
    spin_T = np.array([_spin_T_1yr[i*_blk:(i+1)*_blk].mean() for i in range(_n_blk)])
    spin_A = np.array([_spin_A_1yr[i*_blk:(i+1)*_blk].mean() for i in range(_n_blk)])
    spin_years = np.array([_spin_years_1yr[i*_blk] for i in range(_n_blk)])
    print(f"  Spinup: {spin_years[0]:.0f}-{spin_years[-1]:.0f} CE ({len(spin_years)} steps, dt={SPINUP_DT_YEARS:.0f} yr)")

    print("Loading SP19 density observations ...")
    obs_rho_depth, obs_rho_val = load_sp19_density(SP19_DENS_CSV, RHO_OBS_DEPTH_MAX)
    print(f"  N_rho = {len(obs_rho_depth)}, depth {obs_rho_depth[0]:.2f}-{obs_rho_depth[-1]:.2f} m")

    if USE_AGE_OBS:
        print("Loading SP19 age observations ...")
        obs_age_depth, obs_age_s = load_sp19_age_firn(SP19_AGE_CSV, stride=AGE_OBS_STRIDE)
        print(f"  N_age = {len(obs_age_depth)}, depth {obs_age_depth[0]:.2f}-{obs_age_depth[-1]:.2f} m")
    else:
        print("Age observations DISABLED.")
        obs_age_depth = np.array([])
        obs_age_s = np.array([])

    # -------------------------------------------------------------------------
    # Mesh + spaces
    # -------------------------------------------------------------------------
    mesh, V = build_stretched_mesh(H0, NZ, STRETCH_P)
    R_space = fd.FunctionSpace(mesh, "R", 0)
    dx_domain = fd.dx(domain=mesh)

    # -------------------------------------------------------------------------
    # Observation uncertainties
    # -------------------------------------------------------------------------
    sigma_rho = SIGMA_RHO_ABS + SIGMA_RHO_REL * np.abs(obs_rho_val)
    if len(obs_age_s) > 0:
        sigma_age = SIGMA_AGE_ABS_S + SIGMA_AGE_REL * np.maximum(obs_age_s, 0.0)
    else:
        sigma_age = np.array([])
    if USE_UNIFORM_OBS_SIGMA:
        sigma_rho = np.full_like(sigma_rho, float(np.median(sigma_rho)))
        if len(sigma_age) > 0:
            sigma_age = np.full_like(sigma_age, float(np.median(sigma_age)))
    sigma_rho *= OBS_SIGMA_SCALE
    if len(sigma_age) > 0:
        sigma_age *= OBS_SIGMA_SCALE

    # -------------------------------------------------------------------------
    # Kernel observation operators
    # -------------------------------------------------------------------------
    rho_obs_vals: list = []
    rho_sig_vals: list = []
    age_obs_vals: list = []
    age_sig_vals: list = []
    rho_kernels: list = []
    age_kernels: list = []

    with stop_annotating():
        _x, = fd.SpatialCoordinate(mesh)
        rho_xy = safe_vertex_coords(H0, obs_rho_depth)
        _x_rho = [float(xy[0]) for xy in rho_xy]

        rho_obs_vals = [float(v) for v in obs_rho_val]
        rho_sig_vals = [float(v) for v in sigma_rho]
        age_obs_vals = [float(v) for v in obs_age_s]
        age_sig_vals = [float(v) for v in sigma_age]

        def _make_kernels(x_points, sigma_m, tag):
            kernels = []
            for j, xp in enumerate(x_points):
                wgt = fd.exp(-0.5 * ((_x - xp) / float(sigma_m)) ** 2)
                denom = float(fd.assemble(wgt * dx_domain)) + 1.0e-30
                phi = fd.Function(V, name=f"phi_{tag}_{j:04d}")
                phi.interpolate(wgt / denom)
                kernels.append(phi)
            return kernels

        rho_kernels = _make_kernels(_x_rho, OBS_KERNEL_SIGMA_M, "rho")
        if len(obs_age_depth) > 0:
            age_xy = safe_vertex_coords(H0, obs_age_depth)
            _x_age = [float(xy[0]) for xy in age_xy]
            age_kernels = _make_kernels(_x_age, OBS_KERNEL_SIGMA_M, "age")

        print(f"  Kernel obs operator: {len(rho_kernels)} rho, "
              f"{len(age_kernels)} age "
              f"(sigma={OBS_KERNEL_SIGMA_M} m)")

    with stop_annotating():
        _prior_fn = fd.Function(V, name="prior_scratch")
        domain_len = float(fd.assemble(fd.Constant(1.0) * dx_domain)) + 1e-30

    # -------------------------------------------------------------------------
    # Density IC from SP19
    # -------------------------------------------------------------------------
    _ic_depth_raw, _ic_rho_raw = load_sp19_density(SP19_DENS_CSV, depth_max=2.0 * H0)
    _ic_depth = np.concatenate([[0.0], _ic_depth_raw])
    _ic_rho = np.concatenate([[float(INITIAL["rho_surf"])], _ic_rho_raw])
    _mesh_x_flat = (mesh.coordinates.dat.data_ro[:, 0]
                    if mesh.coordinates.dat.data_ro.ndim == 2
                    else mesh.coordinates.dat.data_ro)
    _mesh_depth = H0 - _mesh_x_flat
    _rho_at_nodes = np.interp(_mesh_depth, _ic_depth, _ic_rho,
                              left=float(INITIAL["rho_surf"]),
                              right=float(_ic_rho[-1]))
    _rho_at_nodes = np.clip(_rho_at_nodes, 200.0, float(P0.rho_i) - 0.5)

    # -------------------------------------------------------------------------
    # State Functions
    # -------------------------------------------------------------------------
    # DG1 space for r2 and age (upwind advection, no Gibbs oscillations)
    V_dg = fd.FunctionSpace(mesh, "DG", 1)

    H     = fd.Function(V, name="H")
    rho   = fd.Function(V_dg, name="rho")
    w     = fd.Function(V, name="w")
    sigma = fd.Function(V, name="sigma")
    r2    = fd.Function(V_dg, name="r2")
    age   = fd.Function(V_dg, name="age")

    # Build CG1 IC then interpolate into DG1
    _rho_ic_cg = fd.Function(V, name="rho_ic_cg")
    _rho_ic_cg.dat.data[:] = _rho_at_nodes
    rho_ic_fn = fd.Function(V_dg, name="rho_ic")
    rho_ic_fn.interpolate(_rho_ic_cg)

    # Grain-radius-squared IC (interpolate CG profile into DG)
    _T_spin_mean = float(np.mean(spin_T))
    _dr2dt_spin = float(INITIAL["kg"]) * np.exp(
        -float(P0.Eg) / (float(P0.R) * _T_spin_mean))
    _w_surf_mean = float(np.mean(spin_A)) * float(P0.rho_i) / float(INITIAL["rho_surf"]) / float(YEAR_S)
    _dr2dz_spin = _dr2dt_spin / abs(_w_surf_mean)
    _r2_ic_nodes_cg = float(INITIAL["r2_surf"]) + _dr2dz_spin * _mesh_depth
    _r2_ic_nodes_cg = np.clip(_r2_ic_nodes_cg, float(INITIAL["r2_surf"]), 2.5e-7)
    _r2_ic_cg = fd.Function(V, name="r2_ic_cg")
    _r2_ic_cg.dat.data[:] = _r2_ic_nodes_cg
    r2_ic_fn = fd.Function(V_dg, name="r2_ic")
    r2_ic_fn.interpolate(_r2_ic_cg)

    print(f"\nSpinup ({SPINUP_YEARS} yr) will run ON TAPE.")
    print(f"  Density IC: SP19 ({len(_ic_depth_raw)} points)")
    print(f"  r2 IC: steady-state linear {float(INITIAL['r2_surf']):.1e} -> {_r2_ic_nodes_cg.max():.2e} m2")

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

    # Normalized div_h control: div_h = DIV_H_INIT + DIV_H_SIGMA * div_h_norm
    # so that div_h_norm has O(1) range, matching the log-space controls.
    div_h_norm = make_real(R_space, 0.0, "div_h_norm")  # 0 = initial value
    div_h_ctrl = DIV_H_INIT + DIV_H_SIGMA * div_h_norm  # UFL expression

    # All controls in a single list; names track which are log vs linear
    controls = [log_kc0, log_kc1, log_kg, log_Ec, log_rho_s, log_r2_s, div_h_norm]
    names    = ["kc0", "kc1", "kg", "Ec", "rho_surf", "r2_surf", "div_h"]
    LOG_CONTROLS = {"kc0", "kc1", "kg", "Ec", "rho_surf", "r2_surf"}  # log-parameterized

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
    # Log-params: sigma is in log-space (dimensionless)
    # div_h: sigma is in s^-1 (linear)
    prior_sigmas = np.array([
        PRIOR_SIGMA_LOG,           # kc0
        PRIOR_SIGMA_LOG,           # kc1
        PRIOR_SIGMA_LOG,           # kg
        EC_PRIOR_SIGMA_LOG,        # Ec
        RHO_SURF_PRIOR_SIGMA_LOG,  # rho_surf
        R2_SURF_PRIOR_SIGMA_LOG,   # r2_surf
        DIV_H_PRIOR_SIGMA,         # div_h (linear, s^-1)
    ])

    # -------------------------------------------------------------------------
    # Build model + solver
    # -------------------------------------------------------------------------
    kc0_expr      = fd.exp(log_kc0)
    kc1_expr      = fd.exp(log_kc1)
    kg_expr       = fd.exp(log_kg)
    Ec_expr       = fd.exp(log_Ec)
    rho_surf_expr = fd.exp(log_rho_s)
    r2_surf_expr  = fd.exp(log_r2_s)

    params = FirnParameters(kg=kg_expr, Ec=Ec_expr)
    model  = FirnModel(params)
    # div_h_ctrl enters the velocity solve via horizontal_divergence
    solver = FirnColumnSolver(model, surface_id=SURFACE_ID,
                               horizontal_divergence=div_h_ctrl)

    Ts    = make_real(R_space, phase1_T[0], "Ts")
    accum = make_real(R_space, phase1_A[0], "accum")
    Hs    = make_real(R_space, float(params.c_i) * (phase1_T[0] - float(params.T_ref)), "Hs")
    dt    = make_real(R_space, dt_s, "dt")

    rho_surf_fs = make_real(R_space, INITIAL["rho_surf"], "rho_surf_fs")
    rho_surf_fs.interpolate(rho_surf_expr)

    RHO_S_FIXED = float(INITIAL["rho_surf"])
    bcs, bc_vals = make_bcs(V, params, accum, RHO_S_FIXED, Hs)
    w_surf_bc = bc_vals["w_surf_bc"]
    bc_sigma = fd.DirichletBC(V, make_real(R_space, 0.0, "sig0"), SURFACE_ID)
    bc_age  = fd.DirichletBC(V, make_real(R_space, 0.0, "age0"), SURFACE_ID)
    rhoCoef = fd.Function(V, name="rhoCoef")

    # DG inflow values: surface BC imposed through upwind flux at the
    # physical velocity scale — no artificial penalty weighting.
    inflow_values = {
        "rho": rho_surf_expr,
        "r2":  r2_surf_expr,
    }

    # -------------------------------------------------------------------------
    # Forward mapping (taped)
    # -------------------------------------------------------------------------
    def forward_mapping():
        """Controls -> objective. Must be called with annotation enabled."""
        rho_surf_fs.interpolate(rho_surf_expr)


        rho.assign(rho_ic_fn)
        H.assign(float(params.c_i) * (float(spin_T[0]) - float(params.T_ref)))
        sigma.assign(0.0)
        r2.assign(r2_ic_fn)
        w.assign(0.0)
        age.assign(0.0)

        # --- Spinup (ON TAPE, Buizert 2021, coarse dt) ---
        dt.assign(float(SPINUP_DT_YEARS * YEAR_S))
        n_spin = int(round(SPINUP_YEARS / SPINUP_DT_YEARS))
        for _k in range(n_spin):
            T_sk = float(spin_T[_k])
            A_sk = float(spin_A[_k])
            Ts.assign(T_sk)
            Hs.assign(float(params.c_i) * (T_sk - float(params.T_ref)))
            accum.assign(A_sk)
            w_surf_bc.interpolate(w_surf_value(params, accum, fd.Constant(RHO_S_FIXED)))

            s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
            rhoCoef.interpolate((1.0 - s) * kc0_expr + s * kc1_expr)

            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                stress=sigma, grain_radius2=r2,
                stress_boundary_condition=bc_sigma,
                grain_radius2_boundary_condition=None,
                inflow_values=inflow_values,
                rhoCoef=rhoCoef,
                age=age, age_boundary_condition=bc_age,
            )

        # --- ERA5 1940 -> CORE_YEAR (2015), fine dt ---
        dt.assign(float(DT_YEARS * YEAR_S))
        for k in range(n_phase1):
            T_k = float(phase1_T[k])
            A_k = float(phase1_A[k])
            Ts.assign(T_k)
            Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)
            w_surf_bc.interpolate(w_surf_value(params, accum, fd.Constant(RHO_S_FIXED)))

            s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
            rhoCoef.interpolate((1.0 - s) * kc0_expr + s * kc1_expr)

            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                stress=sigma, grain_radius2=r2,
                stress_boundary_condition=bc_sigma,
                grain_radius2_boundary_condition=None,
                inflow_values=inflow_values,
                rhoCoef=rhoCoef,
                age=age, age_boundary_condition=bc_age,
            )

        # === Misfit at CORE_YEAR ===
        J = 0.0

        # Density misfit (normalized by domain length, consistent with FE
        # ice-sheet inversions that normalize by area)
        for phi, obs_c, sig_c in zip(rho_kernels, rho_obs_vals, rho_sig_vals):
            pred = fd.assemble(rho * phi * dx_domain)
            r = (pred - obs_c) / sig_c
            J = J + 0.5 * r * r / domain_len

        # Age misfit
        for phi, obs_c, sig_c in zip(age_kernels, age_obs_vals, age_sig_vals):
            pred = fd.assemble(age * phi * dx_domain)
            r = (pred - obs_c) / sig_c
            J = J + 0.5 * r * r / domain_len

        # Priors
        if USE_PRIOR:
            # Log-space priors for positive-definite controls
            for log_ctrl, truth_val, sig_log in [
                (log_kc0,   TRUTH["kc0"],      PRIOR_SIGMA_LOG),
                (log_kc1,   TRUTH["kc1"],      PRIOR_SIGMA_LOG),
                (log_kg,    TRUTH["kg"],        PRIOR_SIGMA_LOG),
                (log_Ec,    TRUTH["Ec"],        EC_PRIOR_SIGMA_LOG),
                (log_rho_s, TRUTH["rho_surf"],  RHO_SURF_PRIOR_SIGMA_LOG),
                (log_r2_s,  TRUTH["r2_surf"],   R2_SURF_PRIOR_SIGMA_LOG),
            ]:
                _prior_fn.interpolate(
                    (log_ctrl - math.log(float(truth_val))) / sig_log
                )
                J = J + 0.5 * fd.assemble(_prior_fn * _prior_fn * dx_domain) / domain_len

            # Linear-space prior for div_h (can be negative)
            _prior_fn.interpolate(
                (div_h_ctrl - fd.Constant(float(TRUTH["div_h"]))) / fd.Constant(DIV_H_PRIOR_SIGMA)
            )
            J = J + 0.5 * fd.assemble(_prior_fn * _prior_fn * dx_domain) / domain_len

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
        """Forward mapping returning misfit only (no prior terms)."""
        rho_surf_fs.interpolate(rho_surf_expr)

        rho.assign(rho_ic_fn)
        H.assign(float(params.c_i) * (float(spin_T[0]) - float(params.T_ref)))
        sigma.assign(0.0)
        r2.assign(r2_ic_fn)
        w.assign(0.0)
        age.assign(0.0)

        dt.assign(float(SPINUP_DT_YEARS * YEAR_S))
        n_spin = int(round(SPINUP_YEARS / SPINUP_DT_YEARS))
        for _k in range(n_spin):
            T_sk = float(spin_T[_k])
            A_sk = float(spin_A[_k])
            Ts.assign(T_sk)
            Hs.assign(float(params.c_i) * (T_sk - float(params.T_ref)))
            accum.assign(A_sk)
            w_surf_bc.interpolate(w_surf_value(params, accum, fd.Constant(RHO_S_FIXED)))

            s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
            rhoCoef.interpolate((1.0 - s) * kc0_expr + s * kc1_expr)
            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                stress=sigma, grain_radius2=r2,
                stress_boundary_condition=bc_sigma,
                grain_radius2_boundary_condition=None,
                inflow_values=inflow_values,
                rhoCoef=rhoCoef,
                age=age, age_boundary_condition=bc_age,
            )

        dt.assign(float(DT_YEARS * YEAR_S))
        for k in range(n_phase1):
            T_k = float(phase1_T[k])
            A_k = float(phase1_A[k])
            Ts.assign(T_k)
            Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)
            w_surf_bc.interpolate(w_surf_value(params, accum, fd.Constant(RHO_S_FIXED)))

            s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
            rhoCoef.interpolate((1.0 - s) * kc0_expr + s * kc1_expr)
            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                stress=sigma, grain_radius2=r2,
                stress_boundary_condition=bc_sigma,
                grain_radius2_boundary_condition=None,
                inflow_values=inflow_values,
                rhoCoef=rhoCoef,
                age=age, age_boundary_condition=bc_age,
            )

        # Misfit only (no prior), normalized by domain length
        J_mis = 0.0
        for phi, obs_c, sig_c in zip(rho_kernels, rho_obs_vals, rho_sig_vals):
            pred = fd.assemble(rho * phi * dx_domain)
            r = (pred - obs_c) / sig_c
            J_mis = J_mis + 0.5 * r * r / domain_len
        for phi, obs_c, sig_c in zip(age_kernels, age_obs_vals, age_sig_vals):
            pred = fd.assemble(age * phi * dx_domain)
            r = (pred - obs_c) / sig_c
            J_mis = J_mis + 0.5 * r * r / domain_len
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
        """Run full forward model unannotated and return kernel observations."""
        set_controls(x_vec)
        rho_surf_fs.interpolate(rho_surf_expr)

        rho.assign(rho_ic_fn)
        H.assign(float(params.c_i) * (float(spin_T[0]) - float(params.T_ref)))
        sigma.assign(0.0)
        r2.assign(r2_ic_fn)
        w.assign(0.0)
        age.assign(0.0)

        dt.assign(float(SPINUP_DT_YEARS * YEAR_S))
        n_spin = int(round(SPINUP_YEARS / SPINUP_DT_YEARS))
        for _k in range(n_spin):
            T_sk = float(spin_T[_k])
            A_sk = float(spin_A[_k])
            Ts.assign(T_sk)
            Hs.assign(float(params.c_i) * (T_sk - float(params.T_ref)))
            accum.assign(A_sk)
            w_surf_bc.interpolate(w_surf_value(params, accum, fd.Constant(RHO_S_FIXED)))

            s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
            rhoCoef.interpolate((1.0 - s) * kc0_expr + s * kc1_expr)
            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                stress=sigma, grain_radius2=r2,
                stress_boundary_condition=bc_sigma,
                grain_radius2_boundary_condition=None,
                inflow_values=inflow_values,
                rhoCoef=rhoCoef,
                age=age, age_boundary_condition=bc_age,
            )
        dt.assign(float(DT_YEARS * YEAR_S))
        for k in range(n_phase1):
            T_k = float(phase1_T[k])
            A_k = float(phase1_A[k])
            Ts.assign(T_k)
            Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)
            w_surf_bc.interpolate(w_surf_value(params, accum, fd.Constant(RHO_S_FIXED)))

            s = 0.5 * (1.0 + fd.tanh((rho - RHO_M) / RHO_SMOOTH))
            rhoCoef.interpolate((1.0 - s) * kc0_expr + s * kc1_expr)
            solver.prognostic_solve(
                enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=accum, surface_density=rho_surf_fs,
                boundary_conditions=bcs, surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                stress=sigma, grain_radius2=r2,
                stress_boundary_condition=bc_sigma,
                grain_radius2_boundary_condition=None,
                inflow_values=inflow_values,
                rhoCoef=rhoCoef,
                age=age, age_boundary_condition=bc_age,
            )
        rho_obs = np.array([float(fd.assemble(rho * phi * dx_domain)) for phi in rho_kernels])
        age_obs = np.array([float(fd.assemble(age * phi * dx_domain)) for phi in age_kernels])
        return rho_obs, age_obs

    # Baseline at MAP
    rho_pred_base, age_pred_base = _run_forward_and_observe(x_map)
    _age_info = ""
    if len(age_pred_base) > 0:
        _age_info = f"  age range [{age_pred_base.min()/float(YEAR_S):.0f}, {age_pred_base.max()/float(YEAR_S):.0f}] yr"
    print(f"  Baseline: rho range [{rho_pred_base.min():.1f}, {rho_pred_base.max():.1f}]{_age_info}")

    n_rho_obs = len(rho_kernels)
    n_age_obs = len(age_kernels)
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
    if len(age_kernels) > 0:
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
            f"South Pole MAP inversion v5_sp19\n"
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
