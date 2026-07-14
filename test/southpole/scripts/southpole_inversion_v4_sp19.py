"""southpole_inversion_v4_sp19.py

MAP inversion for South Pole firn densification parameters using SP19
ice-core density and depth-age observations only (co-located data, no ApRES).

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
    OMP_NUM_THREADS=1 python southpole_inversion_v4_sp19.py
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
SPINUP_DT_YEARS = 1.0
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

OUT_PREFIX = f"southpole_map_v4_sp19"

# Smooth density-coefficient switch
P0 = FirnParameters()
RHO_M = float(getattr(P0, "rho_m", 550.0))
RHO_SMOOTH = 20.0

# Controls -- starting point and bounds
TRUTH = dict(
    kc0      = float(getattr(P0, "kc0",    9.2e-9)),
    kc1      = float(getattr(P0, "kc1",    3.7e-9)),
    kg       = float(getattr(P0, "kg",     1.3e-7)),
    rho_surf = 350.0,
    r2_surf  = 5.0e-9,
)

INITIAL = dict(
    kc0      = float(getattr(P0, "kc0",    9.2e-9)),
    kc1      = float(getattr(P0, "kc1",    3.7e-9)),
    kg       = float(getattr(P0, "kg",     1.3e-7)),
    rho_surf = 350.0,
    r2_surf  = 5.0e-9,
)

BOUNDS = dict(
    kc0      = (1.0e-11, 1.0e-5),
    kc1      = (1.0e-11, 1.0e-5),
    kg       = (1.0e-13, 1.0e-5),
    rho_surf = (100.0,   900.0),
    r2_surf  = (1.0e-9, 1.0e-7),
)

# Priors
USE_PRIOR = True
PRIOR_SIGMA_LOG = 1.5
RHO_SURF_PRIOR_SIGMA_LOG = 0.25
R2_SURF_PRIOR_SIGMA_LOG  = 0.8

# Optimiser
# "adjoint" = Lin-More trust region (requires working adjoint gradient)
# "nelder-mead" = gradient-free Nelder-Mead (scipy.optimize, robust but slower)
OPTIM_METHOD = os.environ.get("FIRN_OPTIM", "adjoint").lower()
MAX_ITER = int(os.environ.get("FIRN_MAX_ITER", "25"))
GRAD_TOL = 1.0e-10
SEED = 0


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
                        depth_max: float = H0):
    df = pd.read_csv(csv_path)
    mask = (df["year_CE"] >= year_min) & (df["year_CE"] <= core_year) & (df["depth_m"] <= depth_max)
    df = df[mask].copy()
    depth = df["depth_m"].values.astype(float)
    age_yr = float(core_year) - df["year_CE"].values.astype(float)
    age_s = age_yr * float(YEAR_S)
    idx = np.argsort(depth)
    return depth[idx], age_s[idx]


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


def control_scalar_value(f) -> float:
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
        if hasattr(c, "block_variable") and getattr(c.block_variable, "checkpoint", None) is not None:
            c.block_variable.checkpoint.assign(float(v))


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


def w_surf_value_fixed(params, accum_m_iceeq_yr, rho_s_fixed):
    """Surface velocity using a FIXED rho_s (not a control-dependent expression).

    This avoids the pyadjoint issue where interpolating a control-dependent
    expression into a Dirichlet BC produces order-1 Taylor remainders.
    """
    return -accum_m_iceeq_yr * params.rho_i / fd.Constant(float(rho_s_fixed)) / float(YEAR_S)


# Penalty strength for soft BCs on density and r2 at the surface.
# Large enough to enforce the BC tightly, small enough to avoid ill-conditioning.
PENALTY_STRENGTH = 1.0e6


def make_bcs(V, params, accum_fn, rho_s_fixed, Hs_fn, surface_id=SURFACE_ID):
    """Build Dirichlet BCs for H and w (NOT control-dependent).

    rho and r2 surface BCs are handled by penalty terms in the weak form
    (see penalty_bcs argument to prognostic_solve), which allows pyadjoint
    to differentiate through them correctly.
    """
    bc_H = fd.DirichletBC(V, Hs_fn, surface_id)
    # w uses a FIXED rho_s for the BC (not the control)
    w_surf_bc = fd.Function(V, name="w_surf_bc")
    w_surf_bc.interpolate(w_surf_value_fixed(params, accum_fn, rho_s_fixed))
    bc_w = fd.DirichletBC(V, w_surf_bc, surface_id)
    # rho BC slot is None — handled by penalty
    return [bc_H, None, bc_w], {"w_surf_bc": w_surf_bc}


# =============================================================================
# Taylor remainder test
# =============================================================================

def taylor_test_multicontrol(rf, controls, bounds=None, seed=1, epsilons=None):
    if epsilons is None:
        epsilons = [1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3]

    with stop_annotating():
        m0 = np.array([control_scalar_value(c) for c in controls], dtype=float)
        set_control_values(controls, m0)

    with stop_annotating():
        J0 = float(rf(controls))
        g_list = rf.derivative()

    with stop_annotating():
        rng = np.random.default_rng(seed)
        d = rng.standard_normal(len(m0))

        if bounds is not None:
            lb_ub = [(_as_scalar(lb), _as_scalar(ub)) for lb, ub in bounds]
            span = np.array([max(u - l, 1e-14) for l, u in lb_ub])
            d = d / (np.linalg.norm(d) + 1e-30)
            d = d * 0.05 * span
            for i, (lbv, ubv) in enumerate(lb_ub):
                d[i] = min(d[i], 0.9 * (ubv - m0[i]) / (max(epsilons) + 1e-30))
                d[i] = max(d[i], 0.9 * (lbv - m0[i]) / (max(epsilons) + 1e-30))
        else:
            d = d / (np.linalg.norm(d) + 1e-30) * 1e-2 * (np.abs(m0) + 1.0)

        dx_ctrl = fd.dx(domain=controls[0].function_space().mesh())
        gdotd = 0.0
        for gf, dv in zip(g_list, d):
            tmp = fd.Function(gf.function_space())
            tmp.assign(float(dv))
            gdotd += float(fd.assemble(gf * tmp * dx_ctrl))

    print("\nTaylor remainder test (multi-control):")
    print(f"  J(m)      = {J0:.6e}")
    print(f"  <dJ, d>   = {gdotd:.6e}")
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

    print("=" * 65)
    print("South Pole MAP inversion v4_sp19  (density + age only)")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\nLoading ERA5 forcing ...")
    era5_years, era5_T, era5_A = load_era5_annual(ERA5_CSV)

    # ERA5_START_YEAR -> CORE_YEAR (density/age snapshot at CORE_YEAR)
    phase1_mask = (era5_years >= ERA5_START_YEAR) & (era5_years < CORE_YEAR)
    phase1_years = era5_years[phase1_mask]
    phase1_T = era5_T[phase1_mask] + ERA5_T_OFFSET_K
    phase1_A = era5_A[phase1_mask] + ERA5_A_OFFSET_M_YR
    n_phase1 = len(phase1_years)

    dt_s = float(DT_YEARS * YEAR_S)
    print(f"  ERA5: {phase1_years[0]}-{phase1_years[-1]} ({n_phase1} steps) -> SP19 obs at {CORE_YEAR}")

    print("Loading Buizert 2021 spinup reconstruction ...")
    _spinup_start_ce = ERA5_START_YEAR - SPINUP_YEARS
    spin_years, spin_T, spin_A = load_buizert_spinup(
        BUIZERT_TEMP_CSV, BUIZERT_ACCUM_CSV,
        spinup_start_year_ce=_spinup_start_ce,
    )
    print(f"  Spinup: {spin_years[0]:.0f}-{spin_years[-1]:.0f} CE ({len(spin_years)} steps)")

    print("Loading SP19 density observations ...")
    obs_rho_depth, obs_rho_val = load_sp19_density(SP19_DENS_CSV, RHO_OBS_DEPTH_MAX)
    print(f"  N_rho = {len(obs_rho_depth)}, depth {obs_rho_depth[0]:.2f}-{obs_rho_depth[-1]:.2f} m")

    print("Loading SP19 age observations ...")
    obs_age_depth, obs_age_s = load_sp19_age_firn(SP19_AGE_CSV)
    print(f"  N_age = {len(obs_age_depth)}, depth {obs_age_depth[0]:.2f}-{obs_age_depth[-1]:.2f} m")

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
    sigma_age = SIGMA_AGE_ABS_S + SIGMA_AGE_REL * np.maximum(obs_age_s, 0.0)
    if USE_UNIFORM_OBS_SIGMA:
        sigma_rho = np.full_like(sigma_rho, float(np.median(sigma_rho)))
        sigma_age = np.full_like(sigma_age, float(np.median(sigma_age)))
    sigma_rho *= OBS_SIGMA_SCALE
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

    assert OBS_MISFIT_METHOD == "kernel", "v4_sp19 only supports kernel observation operator"

    with stop_annotating():
        _x, = fd.SpatialCoordinate(mesh)
        rho_xy = safe_vertex_coords(H0, obs_rho_depth)
        age_xy = safe_vertex_coords(H0, obs_age_depth)
        _x_rho = [float(xy[0]) for xy in rho_xy]
        _x_age = [float(xy[0]) for xy in age_xy]

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
        age_kernels = _make_kernels(_x_age, OBS_KERNEL_SIGMA_M, "age")

        print(f"  Kernel obs operator: {len(rho_kernels)} rho, "
              f"{len(age_kernels)} age "
              f"(sigma={OBS_KERNEL_SIGMA_M} m)")

    # Pre-allocate scratch Functions
    with stop_annotating():
        _prior_fn = fd.Function(V, name="prior_scratch")

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
    H     = fd.Function(V, name="H")
    rho   = fd.Function(V, name="rho")
    w     = fd.Function(V, name="w")
    sigma = fd.Function(V, name="sigma")
    r2    = fd.Function(V, name="r2")
    age   = fd.Function(V, name="age")

    rho_ic_fn = fd.Function(V, name="rho_ic")
    rho_ic_fn.dat.data[:] = _rho_at_nodes

    # Grain-radius-squared IC
    _T_spin_mean = float(np.mean(spin_T))
    _dr2dt_spin = float(INITIAL["kg"]) * np.exp(
        -float(P0.Eg) / (float(P0.R) * _T_spin_mean))
    _w_surf_mean = float(np.mean(spin_A)) * float(P0.rho_i) / float(INITIAL["rho_surf"]) / float(YEAR_S)
    _dr2dz_spin = _dr2dt_spin / abs(_w_surf_mean)
    _r2_ic_nodes = float(INITIAL["r2_surf"]) + _dr2dz_spin * _mesh_depth
    _r2_ic_nodes = np.clip(_r2_ic_nodes, float(INITIAL["r2_surf"]), 2.5e-7)
    r2_ic_fn = fd.Function(V, name="r2_ic")
    r2_ic_fn.dat.data[:] = _r2_ic_nodes

    print(f"\nSpinup ({SPINUP_YEARS} yr) will run ON TAPE.")
    print(f"  Density IC: SP19 ({len(_ic_depth_raw)} points)")
    print(f"  r2 IC: steady-state linear {float(INITIAL['r2_surf']):.1e} -> {_r2_ic_nodes.max():.2e} m2")

    # -------------------------------------------------------------------------
    # Controls
    # -------------------------------------------------------------------------
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()

    def _add_log_ctrl(name, init_val):
        m  = make_real(R_space, math.log(float(init_val)), f"log_{name}")
        lb = make_real(R_space, math.log(float(BOUNDS[name][0])), f"log_{name}_lb")
        ub = make_real(R_space, math.log(float(BOUNDS[name][1])), f"log_{name}_ub")
        return m, lb, ub

    log_kc0,   lb_kc0,   ub_kc0   = _add_log_ctrl("kc0",     INITIAL["kc0"])
    log_kc1,   lb_kc1,   ub_kc1   = _add_log_ctrl("kc1",     INITIAL["kc1"])
    log_kg,    lb_kg,    ub_kg    = _add_log_ctrl("kg",      INITIAL["kg"])
    log_rho_s, lb_rho_s, ub_rho_s = _add_log_ctrl("rho_surf", INITIAL["rho_surf"])
    log_r2_s,  lb_r2_s,  ub_r2_s  = _add_log_ctrl("r2_surf", INITIAL["r2_surf"])

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
    # Build model + solver
    # -------------------------------------------------------------------------
    params = FirnParameters(kg=kg_expr)
    model  = FirnModel(params)
    solver = FirnColumnSolver(model, surface_id=SURFACE_ID)

    Ts    = make_real(R_space, phase1_T[0], "Ts")
    accum = make_real(R_space, phase1_A[0], "accum")
    Hs    = make_real(R_space, float(params.c_i) * (phase1_T[0] - float(params.T_ref)), "Hs")
    dt    = make_real(R_space, dt_s, "dt")

    rho_surf_fs = make_real(R_space, INITIAL["rho_surf"], "rho_surf_fs")
    rho_surf_fs.interpolate(rho_surf_expr)

    # BCs: H and w use Dirichlet (fixed rho_s in w_surf, NOT control-dependent).
    # rho and r2 use penalty BCs (control-dependent, adjoint-safe).
    RHO_S_FIXED = float(INITIAL["rho_surf"])  # fixed reference for w_surf BC
    bcs, bc_vals = make_bcs(V, params, accum, RHO_S_FIXED, Hs)
    w_surf_bc   = bc_vals["w_surf_bc"]
    bc_sigma = fd.DirichletBC(V, make_real(R_space, 0.0, "sig0"), SURFACE_ID)
    bc_age  = fd.DirichletBC(V, make_real(R_space, 0.0, "age0"), SURFACE_ID)
    rhoCoef = fd.Function(V, name="rhoCoef")

    # Penalty BCs: control-dependent surface values for rho and r2
    # These enter the weak form (not as Dirichlet BCs), so pyadjoint can
    # differentiate through them correctly (order-2 Taylor remainder).
    penalty_bcs = {
        "rho": (rho_surf_expr, PENALTY_STRENGTH),
        "r2":  (r2_surf_expr,  PENALTY_STRENGTH),
    }

    with stop_annotating():
        domain_len = float(fd.assemble(fd.Constant(1.0) * dx_domain)) + 1e-30

    J_PARTS = {"rho": None, "age": None, "prior": None}

    # -------------------------------------------------------------------------
    # Forward mapping (taped)
    # -------------------------------------------------------------------------
    def forward_mapping():
        """Controls -> objective.

        Runs the full forward model on tape in two stages:
          1. Spinup (Buizert 2021, 1200 yr)
          2. ERA5: 1940 -> CORE_YEAR (2015) -> snapshot density/age for SP19
        """
        # Refresh control-dependent fields (NOT BCs — BCs use penalty)
        rho_surf_fs.interpolate(rho_surf_expr)

        # Fixed IC
        rho.assign(rho_ic_fn)
        H.assign(float(params.c_i) * (float(spin_T[0]) - float(params.T_ref)))
        sigma.assign(0.0)
        r2.assign(r2_ic_fn)
        w.assign(0.0)
        age.assign(0.0)

        # --- Spinup (ON TAPE, Buizert 2021) ---
        n_spin = int(round(SPINUP_YEARS / SPINUP_DT_YEARS))
        for _k in range(n_spin):
            T_sk = float(spin_T[_k])
            A_sk = float(spin_A[_k])
            Ts.assign(T_sk)
            Hs.assign(float(params.c_i) * (T_sk - float(params.T_ref)))
            accum.assign(A_sk)
            w_surf_bc.interpolate(w_surf_value_fixed(params, accum, RHO_S_FIXED))

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
                penalty_bcs=penalty_bcs,
                rhoCoef=rhoCoef,
                age=age, age_boundary_condition=bc_age,
            )

        # --- ERA5 1940 -> CORE_YEAR (2015) ---
        for k in range(n_phase1):
            T_k = float(phase1_T[k])
            A_k = float(phase1_A[k])
            Ts.assign(T_k)
            Hs.assign(float(params.c_i) * (T_k - float(params.T_ref)))
            accum.assign(A_k)
            w_surf_bc.interpolate(w_surf_value_fixed(params, accum, RHO_S_FIXED))

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
                penalty_bcs=penalty_bcs,
                rhoCoef=rhoCoef,
                age=age, age_boundary_condition=bc_age,
            )

        # === SNAPSHOT at CORE_YEAR (2015): density + age misfits ===
        J_rho = 0.0
        for phi, obs_c, sig_c in zip(rho_kernels, rho_obs_vals, rho_sig_vals):
            pred = fd.assemble(rho * phi * dx_domain)
            r = (pred - obs_c) / sig_c
            J_rho += 0.5 * r * r

        J_age = 0.0
        for phi, obs_c, sig_c in zip(age_kernels, age_obs_vals, age_sig_vals):
            pred = fd.assemble(age * phi * dx_domain)
            r = (pred - obs_c) / sig_c
            J_age += 0.5 * r * r

        J = J_rho + J_age

        # --- Priors ---
        J_prior = 0.0
        if USE_PRIOR:
            def _prior_log(log_ctrl, truth_val, sigma_log):
                _prior_fn.interpolate(
                    (log_ctrl - math.log(float(truth_val))) / sigma_log
                )
                return 0.5 * fd.assemble(_prior_fn * _prior_fn * dx_domain) / domain_len

            J_prior = (
                _prior_log(log_kc0,   TRUTH["kc0"],      PRIOR_SIGMA_LOG)
                + _prior_log(log_kc1, TRUTH["kc1"],      PRIOR_SIGMA_LOG)
                + _prior_log(log_kg,  TRUTH["kg"],       PRIOR_SIGMA_LOG)
                + _prior_log(log_rho_s, TRUTH["rho_surf"], RHO_SURF_PRIOR_SIGMA_LOG)
                + _prior_log(log_r2_s,  TRUTH["r2_surf"],  R2_SURF_PRIOR_SIGMA_LOG)
            )

            J = J + J_prior

        if J_PARTS["rho"] is None:
            J_PARTS["rho"]   = J_rho
            J_PARTS["age"]   = J_age
            J_PARTS["prior"] = J_prior

        return J

    # Build the tape
    J = forward_mapping()
    rf = ReducedFunctional(J, [Control(c) for c in controls])
    pause_annotation()

    print(f"\nInitial objective J(m0) = {float(J):.6e}")
    print(f"  J_rho   = {float(J_PARTS['rho']):.6e}")
    print(f"  J_age   = {float(J_PARTS['age']):.6e}")
    print(f"  J_prior = {float(J_PARTS['prior']):.6e}")

    # -------------------------------------------------------------------------
    # Pre-optimisation diagnostic plot
    # -------------------------------------------------------------------------
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

        fig.suptitle(
            f"South Pole v4_sp19 - initial model vs data\n"
            f"rho/age at {CORE_YEAR} (SP19)"
        )
        fig.tight_layout()
        diag_path = OUT_DIR / f"{OUT_PREFIX}_initial_vs_data.png"
        fig.savefig(diag_path, dpi=150)
        print(f"\nInitial-vs-data plot saved to {diag_path}")
        plt.close(fig)
    except Exception as _e:
        print(f"(Initial diagnostic plot skipped: {_e})")

    # -------------------------------------------------------------------------
    # Optimisation
    # -------------------------------------------------------------------------
    J_hist: list[float] = []

    if OPTIM_METHOD == "nelder-mead":
        # ----- Gradient-free Nelder-Mead via scipy -----
        # All controls are log-space.
        from scipy.optimize import minimize as scipy_minimize

        m0_vec = get_control_values(controls)
        lb_vec = np.array([_as_scalar(lb) for lb, _ in bounds])
        ub_vec = np.array([_as_scalar(ub) for _, ub in bounds])
        scipy_bounds = list(zip(lb_vec, ub_vec))

        _n_eval = [0]

        _J_penalty = 1.0e12  # returned when the forward solve fails

        def _scipy_objective(x):
            _n_eval[0] += 1
            try:
                with stop_annotating():
                    set_control_values(controls, x)
                with stop_annotating():
                    Jval = float(rf(controls))
                if not np.isfinite(Jval):
                    Jval = _J_penalty
            except Exception as _e:
                print(f"  [eval {_n_eval[0]:03d}] FAILED: {_e}")
                Jval = _J_penalty

            J_hist.append(Jval)

            # Log progress
            phys = {}
            for i, nm in enumerate(names):
                phys[nm] = math.exp(x[i])
            if _n_eval[0] % 5 == 0 or _n_eval[0] == 1:
                print(f"  [eval {_n_eval[0]:03d}] J = {Jval:.4e}  "
                      f"kc0={phys['kc0']:.2e}  kc1={phys['kc1']:.2e}  "
                      f"kg={phys['kg']:.2e}  rho_s={phys['rho_surf']:.0f}")
            return Jval

        print(f"\nStarting Nelder-Mead optimisation (max {MAX_ITER} evals) ...")
        result = scipy_minimize(
            _scipy_objective,
            m0_vec,
            method="Nelder-Mead",
            bounds=scipy_bounds,
            options={
                "maxfev": MAX_ITER,
                "xatol": 1e-6,
                "fatol": 1.0,
                "adaptive": True,
                "disp": True,
            },
        )
        print(f"  Nelder-Mead terminated: {result.message}")
        print(f"  Final J = {result.fun:.6e}  ({result.nfev} function evaluations)")

        with stop_annotating():
            set_control_values(controls, result.x)

    else:
        # ----- Adjoint-based Lin-More trust region -----
        with stop_annotating():
            m0 = get_control_values(controls)
            m1 = m0.copy()
            m1[0] = min(max(m1[0] + 0.5, _as_scalar(bounds[0][0])), _as_scalar(bounds[0][1]))
            set_control_values(controls, m1)
        with stop_annotating():
            J1 = float(rf(controls))
        with stop_annotating():
            set_control_values(controls, m0)
        print(f"  Sensitivity check: dJ from log_kc0+0.5 = {J1 - float(J):+.3e}")

        taylor_test_multicontrol(rf, controls, bounds=bounds)

        minimisation_parameters["Status Test"]["Iteration Limit"] = MAX_ITER
        minimisation_parameters["Status Test"]["Gradient Tolerance"] = GRAD_TOL
        minimisation_parameters["General"]["Secant"]["Type"] = "Limited-Memory BFGS"
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

        _n_eval = [0]
        def _eval_cb(val, *args, **kwargs):
            _n_eval[0] += 1
            J_hist.append(float(val))
            if _n_eval[0] % 5 == 0 or _n_eval[0] == 1:
                print(f"  [iter {_n_eval[0]:03d}] J = {float(val):.6e}")
        rf.eval_cb_post = _eval_cb

        print(f"\nStarting L-BFGS optimisation (max {MAX_ITER} iters) ...")
        optimiser.run()

        for c in controls:
            commit_checkpoint(c)

    # -------------------------------------------------------------------------
    # Extract results
    # -------------------------------------------------------------------------
    m_map_phys = {}
    m_map_raw = {}
    for nm, c in zip(names, controls):
        raw_val = float(control_scalar_value(c))
        m_map_raw[nm] = raw_val
        m_map_phys[nm] = math.exp(raw_val)

    print("\nMAP parameter estimates:")
    for nm in names:
        print(f"  {nm:12s} = {m_map_phys[nm]:.6e}  (initial: {INITIAL[nm]:.6e})")

    with stop_annotating():
        set_control_values(controls, [m_map_raw[nm] for nm in names])
    with stop_annotating():
        J_map = float(rf(controls))
    print(f"\nFinal J(MAP) = {J_map:.6e}")

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
    }
    map_path = OUT_DIR / f"{OUT_PREFIX}.json"
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

        depth_nodes = (H0 - mesh.coordinates.dat.data_ro[:, 0]
                       if mesh.coordinates.dat.data_ro.ndim == 2
                       else H0 - mesh.coordinates.dat.data_ro)
        idx_sort = np.argsort(depth_nodes)
        d_plot = depth_nodes[idx_sort]

        rho_plot = rho.dat.data_ro[idx_sort]
        age_plot = age.dat.data_ro[idx_sort] / float(YEAR_S)

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        ax = axes[0]
        ax.plot(rho_plot, d_plot, "b-", lw=1.5, label=f"Model at {CORE_YEAR}")
        ax.errorbar(obs_rho_val, obs_rho_depth, xerr=sigma_rho,
                    fmt="r.", ms=3, elinewidth=0.5, label=f"SP19 ({CORE_YEAR})")
        ax.set_xlabel("Density (kg/m3)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title(f"Density at {CORE_YEAR}")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.plot(age_plot, d_plot, "b-", lw=1.5, label=f"Model at {CORE_YEAR}")
        ax.errorbar(obs_age_s / float(YEAR_S), obs_age_depth,
                    xerr=sigma_age / float(YEAR_S),
                    fmt="r.", ms=3, elinewidth=0.5, label=f"SP19 ({CORE_YEAR})")
        ax.set_xlabel("Age (yr)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title(f"Age at {CORE_YEAR}")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"South Pole MAP inversion v4_sp19\n"
            f"rho/age at {CORE_YEAR} | "
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
