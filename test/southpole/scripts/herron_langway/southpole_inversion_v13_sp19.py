"""southpole_inversion_v13_sp19.py

MAP inversion with Herron & Langway densification on a static oversized mesh
with free-surface tracking.

Key features:
  - Static mesh [0, H_MAX] — no mesh motion, adjoint-safe
  - Surface height h(t) tracked as fd.Constant, evolved from mass balance
  - Masked densification: drhodt * chi(x, h) — zero above surface
  - Base velocity BC (w_base from accumulation)
  - CG density with Dirichlet BCs (matching um-fdm)
  - 5 controls: hl_k0, hl_k1, hl_Ea1, hl_Ea2, Q_base
  - All on tape (spinup + ERA5)
  - Icepack-style 1/N normalization

Run:
    cd test/southpole/scripts/herron_langway
    OMP_NUM_THREADS=1 python southpole_inversion_v13_sp19.py
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
from firn.physics.densification import herron_langway
from firn.constants import year as YEAR_S

# =============================================================================
# Paths
# =============================================================================
_HERE = Path(__file__).parent
_SOUTHPOLE = _HERE.parent.parent
PROCESSED_DIR = _SOUTHPOLE / "processed"
ERA5_CSV       = PROCESSED_DIR / "era5_monthly_point.csv"
SP19_DENS_CSV  = PROCESSED_DIR / "sp19_density.csv"
SP19_AGE_CSV   = PROCESSED_DIR / "sp19_depth_age.csv"
BUIZERT_TEMP_CSV  = PROCESSED_DIR / "buizert2021_spice_temp.csv"
BUIZERT_ACCUM_CSV = PROCESSED_DIR / "buizert2021_spice_accum.csv"
BOREHOLE_T_CSV = PROCESSED_DIR / "spicecore_borehole_T.csv"
OUT_DIR = _SOUTHPOLE / "results"

# =============================================================================
# Settings
# =============================================================================
H0 = 130.0           # nominal column height
H_MAX = 250.0        # oversized mesh height (room for surface to rise)
NZ = 120             # elements
STRETCH_P = 2.0      # less aggressive stretching for wider domain
SURFACE_ID = 2       # x = H_MAX boundary
BASE_ID = 1          # x = 0 boundary
RHO_SURF = 350.0     # fixed surface density
MASK_WIDTH = 1.0     # surface mask smoothing (m)

SPINUP_START_CE = 740
ERA5_START_YEAR = 1940
CORE_YEAR = 2015
SPINUP_DT_YEARS = 10.0
ERA5_DT_YEARS = 0.5

# Observation uncertainties
SIGMA_RHO_ABS = 15.0
SIGMA_RHO_REL = 0.03
SIGMA_AGE_ABS_S = 1.0 * float(YEAR_S)
SIGMA_AGE_REL = 0.02
SIGMA_T_ABS = 0.5
AGE_OBS_STRIDE = 5

# Data selection
OBS_TYPES = os.environ.get("FIRN_OBS", "rho+age+T").split("+")
USE_RHO_OBS = "rho" in OBS_TYPES
USE_AGE_OBS = "age" in OBS_TYPES
USE_T_OBS = "T" in OBS_TYPES

OUT_PREFIX = "southpole_map_v13"
P0 = FirnParameters()

# Controls
INITIAL = dict(
    hl_k0  = float(P0.hl_k0_prefactor),
    hl_k1  = float(P0.hl_k1_prefactor),
    hl_Ea1 = float(P0.hl_Ea_stage1),
    hl_Ea2 = float(P0.hl_Ea_stage2),
    Q_base = 0.05,
)

BOUNDS = dict(
    hl_k0  = (0.01,  10000.0),
    hl_k1  = (0.01,  1.0e6),
    hl_Ea1 = (100.0, 1.0e6),
    hl_Ea2 = (100.0, 1.0e6),
    Q_base = (-5.0,  5.0),
)

USE_PRIOR = True
HL_K_PRIOR_SIGMA_LOG  = 1.0
HL_EA_PRIOR_SIGMA_LOG = 0.3
QBASE_PRIOR_SIGMA     = 0.5
MAX_ITER = int(os.environ.get("FIRN_MAX_ITER", "30"))
HESSIAN_EPS = 1e-2

# =============================================================================
# Data loading
# =============================================================================
def load_era5_annual(csv_path):
    df = pd.read_csv(csv_path)
    t_rows = df[df["t2m_K"].notna()][["year", "t2m_K"]].copy()
    a_rows = df[df["net_accum_m_iceeq_month"].notna()][["year", "net_accum_m_iceeq_month"]].copy()
    T_by_yr = t_rows.groupby("year")["t2m_K"].mean()
    A_by_yr = a_rows.groupby("year")["net_accum_m_iceeq_month"].sum()
    years = np.array(sorted(T_by_yr.index.intersection(A_by_yr.index)), dtype=int)
    return years, T_by_yr.loc[years].values, A_by_yr.loc[years].values

def make_real(R, val, name):
    f = fd.Function(R, name=name); f.assign(float(val)); return f

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"South Pole v13 — H&L free surface (obs={'+'.join(OBS_TYPES)})")
    print("=" * 65)

    # ----- Load data -----
    era5_years, era5_T, era5_A = load_era5_annual(ERA5_CSV)
    mask = (era5_years >= ERA5_START_YEAR) & (era5_years < CORE_YEAR)
    era5_T_phase = era5_T[mask]
    era5_A_phase = era5_A[mask] + 0.0163
    n_sub = int(round(1.0 / ERA5_DT_YEARS))
    era5_T_sub = np.repeat(era5_T_phase, n_sub)
    era5_A_sub = np.repeat(era5_A_phase, n_sub)
    n_era5 = len(era5_T_sub)
    dt_era5_s = float(ERA5_DT_YEARS * YEAR_S)

    buiz_T_df = pd.read_csv(BUIZERT_TEMP_CSV).sort_values("year_CE")
    buiz_A_df = pd.read_csv(BUIZERT_ACCUM_CSV).sort_values("year_CE")
    spin_yr = np.arange(float(SPINUP_START_CE), float(ERA5_START_YEAR), 1.0)
    spin_T_1yr = np.interp(spin_yr, buiz_T_df["year_CE"].values, buiz_T_df["temp"].values + 273.15)
    spin_A_1yr = np.interp(spin_yr, buiz_A_df["year_CE"].values, buiz_A_df["accum"].values)
    blk = int(SPINUP_DT_YEARS)
    n_blk = len(spin_T_1yr) // blk
    spin_T = np.array([spin_T_1yr[i*blk:(i+1)*blk].mean() for i in range(n_blk)])
    spin_A = np.array([spin_A_1yr[i*blk:(i+1)*blk].mean() for i in range(n_blk)])
    n_spin = len(spin_T)
    dt_spin_s = float(SPINUP_DT_YEARS * YEAR_S)

    print(f"  Spinup: {n_spin} steps × {SPINUP_DT_YEARS}yr, ERA5: {n_era5} steps × {ERA5_DT_YEARS}yr")
    print(f"  Total tape: {n_spin + n_era5} steps")

    # Load observations
    obs_rho_depth = obs_rho_val = sigma_rho = np.array([])
    obs_age_depth = obs_age_s = sigma_age = np.array([])
    obs_T_depth = obs_T_H = sigma_T_H = np.array([])

    if USE_RHO_OBS:
        df = pd.read_csv(SP19_DENS_CSV); df = df[df["depth_m"] <= H0]
        obs_rho_depth = df["depth_m"].values; obs_rho_val = df["rho_kgm3"].values * 1000
        sigma_rho = SIGMA_RHO_ABS + SIGMA_RHO_REL * np.abs(obs_rho_val)
        sigma_rho = np.full_like(sigma_rho, np.median(sigma_rho))
    if USE_AGE_OBS:
        df = pd.read_csv(SP19_AGE_CSV)
        df = df[(df["year_CE"] <= CORE_YEAR) & (df["depth_m"] <= H0)]
        obs_age_depth = df["depth_m"].values[::AGE_OBS_STRIDE]
        obs_age_s = (CORE_YEAR - df["year_CE"].values[::AGE_OBS_STRIDE]) * float(YEAR_S)
        sigma_age = SIGMA_AGE_ABS_S + SIGMA_AGE_REL * np.maximum(obs_age_s, 0)
        sigma_age = np.full_like(sigma_age, np.median(sigma_age))
    if USE_T_OBS:
        df = pd.read_csv(BOREHOLE_T_CSV); df = df[df["depth_m"] <= H0]
        obs_T_depth = df["depth_m"].values
        obs_T_K = df["T_C"].values + 273.15
        obs_T_H = float(P0.c_i) * (obs_T_K - float(P0.T_ref))
        sigma_T_H = float(P0.c_i) * np.full_like(obs_T_K, SIGMA_T_ABS)

    N_rho, N_age, N_T = len(obs_rho_depth), len(obs_age_depth), len(obs_T_depth)
    print(f"  Obs: {N_rho} rho, {N_age} age, {N_T} T")

    # ----- Mesh -----
    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi_ref = fd.SpatialCoordinate(mesh)[0]
    x_phys = H_MAX * (1.0 - (1.0 - xi_ref) ** STRETCH_P)
    mesh.coordinates.assign(
        fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([x_phys]))
    )
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)
    dx = fd.dx(domain=mesh)
    xi = fd.SpatialCoordinate(mesh)[0]

    with stop_annotating():
        _prior_fn = fd.Function(V, name="prior_scratch")
        domain_len = float(fd.assemble(fd.Constant(1.0) * dx)) + 1e-30

    # ----- State fields -----
    H_field = fd.Function(V, name="H")
    rho = fd.Function(V, name="rho")
    w = fd.Function(V, name="w")
    age = fd.Function(V, name="age")

    # ----- Controls -----
    log_k0  = make_real(R, math.log(INITIAL["hl_k0"]),  "log_k0")
    log_k1  = make_real(R, math.log(INITIAL["hl_k1"]),  "log_k1")
    log_Ea1 = make_real(R, math.log(INITIAL["hl_Ea1"]), "log_Ea1")
    log_Ea2 = make_real(R, math.log(INITIAL["hl_Ea2"]), "log_Ea2")
    Q_base_ctrl = make_real(R, INITIAL["Q_base"], "Q_base")

    controls = [log_k0, log_k1, log_Ea1, log_Ea2, Q_base_ctrl]
    names = ["hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "Q_base"]
    LOG_CONTROLS = {"hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2"}
    n_ctrl = len(controls)

    lb_vec = np.array([math.log(BOUNDS[nm][0]) if nm in LOG_CONTROLS else BOUNDS[nm][0] for nm in names])
    ub_vec = np.array([math.log(BOUNDS[nm][1]) if nm in LOG_CONTROLS else BOUNDS[nm][1] for nm in names])
    lbs_fns = [make_real(R, float(lb_vec[i]), f"{names[i]}_lb") for i in range(n_ctrl)]
    ubs_fns = [make_real(R, float(ub_vec[i]), f"{names[i]}_ub") for i in range(n_ctrl)]
    bounds_pairs = list(zip(lbs_fns, ubs_fns))

    prior_sigmas = np.array([
        HL_K_PRIOR_SIGMA_LOG,   # k0
        HL_K_PRIOR_SIGMA_LOG,   # k1
        HL_EA_PRIOR_SIGMA_LOG,  # Ea1
        HL_EA_PRIOR_SIGMA_LOG,  # Ea2
        QBASE_PRIOR_SIGMA,      # Q_base (linear)
    ])

    # ----- Model + Solver -----
    params = FirnParameters(
        hl_k0_prefactor=fd.exp(log_k0),
        hl_k1_prefactor=fd.exp(log_k1),
        hl_Ea_stage1=fd.exp(log_Ea1),
        hl_Ea_stage2=fd.exp(log_Ea2),
        basal_heat_flux_W_m2=Q_base_ctrl,
    )
    model = FirnModel(params, densification_rate_fn=herron_langway)
    solver = FirnColumnSolver(model, surface_id=SURFACE_ID)

    # Boundary functions
    Ts = make_real(R, float(spin_T[0]), "Ts")
    Hs = make_real(R, float(params.c_i) * (spin_T[0] - float(params.T_ref)), "Hs")
    accum = make_real(R, float(spin_A[0]), "accum")
    rho_s_fn = make_real(R, RHO_SURF, "rho_s")
    dt_fn = make_real(R, dt_spin_s, "dt")

    # BCs: enthalpy Dirichlet at top, density Dirichlet at top, velocity at BASE
    bc_H = fd.DirichletBC(V, Hs, SURFACE_ID)
    bc_rho = fd.DirichletBC(V, rho_s_fn, SURFACE_ID)
    w_base_fn = make_real(R, -float(spin_A[0]) / float(YEAR_S), "w_base")
    bc_w = fd.DirichletBC(V, w_base_fn, BASE_ID)
    bc_age = fd.DirichletBC(V, make_real(R, 0.0, "a0"), SURFACE_ID)

    # Surface height and mask
    h_surf = fd.Constant(H0)
    mask_w = fd.Constant(MASK_WIDTH)
    chi = fd.Function(V, name="chi")

    # ----- Forward mapping -----
    def forward_mapping():
        """Full spinup + ERA5 on tape with free surface tracking."""

        # Analytical IC below h=H0, rho_surf above
        _T0 = float(spin_T[0])
        H_field.assign(float(params.c_i) * (_T0 - float(params.T_ref)))
        rho.interpolate(fd.conditional(
            fd.lt(xi, h_surf),
            fd.Constant(RHO_SURF) + (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) *
            (1.0 - fd.exp(-(h_surf - xi) / 20.0)),
            fd.Constant(RHO_SURF),
        ))
        w.assign(-float(spin_A[0]) * float(params.rho_i) / RHO_SURF / float(YEAR_S))
        age.assign(0.0)
        h_surf.assign(H0)

        # Spinup
        dt_fn.assign(dt_spin_s)
        for k in range(n_spin):
            Tk, Ak = float(spin_T[k]), float(spin_A[k])
            Ts.assign(Tk)
            Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
            accum.assign(Ak)
            w_base_fn.assign(-Ak / float(YEAR_S))

            # Update mask
            chi.interpolate(model.surface_mask(xi, h_surf, mask_w))

            solver.prognostic_solve(
                enthalpy=H_field, density=rho, firn_velocity=w, dt=dt_fn,
                accumulation=accum, surface_density=rho_s_fn,
                boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age, age_boundary_condition=bc_age,
            )

            # Update surface height from mass balance
            T_field = model.temperature_from_enthalpy(H_field)
            bdot = fd.Constant(float(Ak) * float(params.rho_i) / float(params.spy))
            if model.densification_rate_fn is not None:
                drhodt_expr = model.densification_rate_fn(rho, T_field, params=params, bdot=bdot)
            else:
                drhodt_expr = model.densification_rate_arthern(rho, T_field, bdot)
            drhodt_masked = drhodt_expr * chi
            dhdt = model.surface_tendency(drhodt_masked, accum, RHO_SURF, dx)
            h_surf.assign(float(h_surf) + float(dt_fn) * dhdt)

        # ERA5
        dt_fn.assign(dt_era5_s)
        for k in range(n_era5):
            Tk, Ak = float(era5_T_sub[k]), float(era5_A_sub[k])
            Ts.assign(Tk)
            Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
            accum.assign(Ak)
            w_base_fn.assign(-Ak / float(YEAR_S))

            chi.interpolate(model.surface_mask(xi, h_surf, mask_w))

            solver.prognostic_solve(
                enthalpy=H_field, density=rho, firn_velocity=w, dt=dt_fn,
                accumulation=accum, surface_density=rho_s_fn,
                boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age, age_boundary_condition=bc_age,
            )

            T_field = model.temperature_from_enthalpy(H_field)
            bdot = fd.Constant(float(Ak) * float(params.rho_i) / float(params.spy))
            if model.densification_rate_fn is not None:
                drhodt_expr = model.densification_rate_fn(rho, T_field, params=params, bdot=bdot)
            else:
                drhodt_expr = model.densification_rate_arthern(rho, T_field, bdot)
            drhodt_masked = drhodt_expr * chi
            dhdt = model.surface_tendency(drhodt_masked, accum, RHO_SURF, dx)
            h_surf.assign(float(h_surf) + float(dt_fn) * dhdt)

        # ----- Misfit at CORE_YEAR -----
        # Build VOMs at final surface height
        h_final = float(h_surf)
        eps_x = 1e-8 * H_MAX

        def _depth_to_x(depths):
            x = h_final - np.clip(depths, 0, h_final)
            x = np.where(x <= 0, eps_x, x)
            x = np.where(x >= H_MAX, H_MAX - eps_x, x)
            return x

        J = 0.0

        if N_rho > 0:
            rho_x = _depth_to_x(obs_rho_depth)
            vom_rho = fd.VertexOnlyMesh(mesh, rho_x.reshape(-1, 1))
            P0_rho = fd.FunctionSpace(vom_rho, "DG", 0)
            rho_obs_fn = fd.Function(P0_rho); rho_obs_fn.dat.data[:] = obs_rho_val
            rho_sig_fn = fd.Function(P0_rho); rho_sig_fn.dat.data[:] = sigma_rho
            rho_pred = fd.Function(P0_rho); rho_pred.interpolate(rho)
            r = (rho_pred - rho_obs_fn) / rho_sig_fn
            J = J + 0.5 / float(N_rho) * fd.assemble(r * r * fd.dx(domain=vom_rho))

        if N_age > 0:
            age_x = _depth_to_x(obs_age_depth)
            vom_age = fd.VertexOnlyMesh(mesh, age_x.reshape(-1, 1))
            P0_age = fd.FunctionSpace(vom_age, "DG", 0)
            age_obs_fn = fd.Function(P0_age); age_obs_fn.dat.data[:] = obs_age_s
            age_sig_fn = fd.Function(P0_age); age_sig_fn.dat.data[:] = sigma_age
            age_pred = fd.Function(P0_age); age_pred.interpolate(age)
            r = (age_pred - age_obs_fn) / age_sig_fn
            J = J + 0.5 / float(N_age) * fd.assemble(r * r * fd.dx(domain=vom_age))

        if N_T > 0:
            T_x = _depth_to_x(obs_T_depth)
            vom_T = fd.VertexOnlyMesh(mesh, T_x.reshape(-1, 1))
            P0_T = fd.FunctionSpace(vom_T, "DG", 0)
            T_obs_fn = fd.Function(P0_T); T_obs_fn.dat.data[:] = obs_T_H
            T_sig_fn = fd.Function(P0_T); T_sig_fn.dat.data[:] = sigma_T_H
            T_pred = fd.Function(P0_T); T_pred.interpolate(H_field)
            r = (T_pred - T_obs_fn) / T_sig_fn
            J = J + 0.5 / float(N_T) * fd.assemble(r * r * fd.dx(domain=vom_T))

        # Priors
        if USE_PRIOR:
            _n_inv = 1.0 / float(n_ctrl)
            for log_ctrl, truth_val, sig in [
                (log_k0,  INITIAL["hl_k0"],  HL_K_PRIOR_SIGMA_LOG),
                (log_k1,  INITIAL["hl_k1"],  HL_K_PRIOR_SIGMA_LOG),
                (log_Ea1, INITIAL["hl_Ea1"], HL_EA_PRIOR_SIGMA_LOG),
                (log_Ea2, INITIAL["hl_Ea2"], HL_EA_PRIOR_SIGMA_LOG),
            ]:
                _prior_fn.interpolate((log_ctrl - math.log(float(truth_val))) / sig)
                J = J + _n_inv * 0.5 * fd.assemble(_prior_fn * _prior_fn * dx) / domain_len
            # Q_base linear prior
            _prior_fn.interpolate(
                (Q_base_ctrl - fd.Constant(float(INITIAL["Q_base"]))) / fd.Constant(QBASE_PRIOR_SIGMA)
            )
            J = J + _n_inv * 0.5 * fd.assemble(_prior_fn * _prior_fn * dx) / domain_len

        return J

    # ----- Build tape -----
    tape = get_working_tape(); tape.clear_tape()
    continue_annotation()

    J_init = forward_mapping()
    rf = ReducedFunctional(J_init, [Control(c) for c in controls])
    pause_annotation()

    print(f"\nInitial J = {float(J_init):.6e}")
    print(f"Surface at CORE_YEAR: h = {float(h_surf):.1f}m")

    # ----- Taylor test -----
    x0 = np.array([float(c.dat.data_ro[0]) for c in controls])
    print("\nTaylor remainder test ...")
    with stop_annotating():
        for c, v in zip(controls, x0): c.assign(float(v))
        J0 = float(rf(controls))
        g_list = rf.derivative()
    g0 = np.array([float(g.dat.data_ro[0]) * domain_len for g in g_list])
    rng = np.random.default_rng(0)
    d = rng.standard_normal(n_ctrl) * 0.01
    gdotd = g0 @ d
    print(f"  J(m) = {J0:.6e}, <dJ,d> = {gdotd:.6e}")
    print(f"  eps        |J(m+eps d)-J(m)|   |remainder|        order")
    prev_r = None
    for eps in [1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3]:
        m1 = np.clip(x0 + eps * d, lb_vec, ub_vec)
        try:
            with stop_annotating():
                for c, v in zip(controls, m1): c.assign(float(v))
                J1 = float(rf(controls))
            r = abs(J1 - J0 - eps * gdotd)
            order = ""
            if prev_r is not None and r > 0 and prev_r > 0:
                order = f"{math.log(prev_r/r)/math.log(2):.2f}"
            print(f"  {eps:.3e}  {abs(J1-J0):.6e}  {r:.6e}  {order}")
            prev_r = r
        except Exception as e:
            print(f"  {eps:.3e}  FAILED ({type(e).__name__})")

    # ----- Optimize -----
    with stop_annotating():
        for c, v in zip(controls, x0): c.assign(float(v))
        rf(controls)

    J_hist = []
    _n_eval = [0]
    def _eval_cb(val, *a, **kw):
        _n_eval[0] += 1
        J_hist.append(float(val))
        if _n_eval[0] % 5 == 0 or _n_eval[0] == 1:
            print(f"  [iter {_n_eval[0]:03d}] J = {float(val):.6e}")
    rf.eval_cb_post = _eval_cb

    minimisation_parameters["Status Test"]["Iteration Limit"] = MAX_ITER
    minimisation_parameters["Status Test"]["Gradient Tolerance"] = 1e-10
    min_problem = MinimizationProblem(rf, bounds=bounds_pairs)
    try:
        optimiser = LinMoreOptimiser(min_problem, minimisation_parameters,
                                      checkpoint_dir=str(OUT_DIR / f"{OUT_PREFIX}_optim_checkpoint"))
    except TypeError:
        optimiser = LinMoreOptimiser(min_problem, minimisation_parameters, auto_checkpoint=False)

    print(f"\nStarting optimization (max {MAX_ITER} iters) ...")
    try:
        optimiser.run()
    except fd.exceptions.ConvergenceError as e:
        print(f"\n  Stopped early ({_n_eval[0]} evals): {e}")

    # ----- MAP results -----
    x_map = np.array([float(c.dat.data_ro[0]) for c in controls])
    m_map = {}
    for i, nm in enumerate(names):
        m_map[nm] = math.exp(x_map[i]) if nm in LOG_CONTROLS else x_map[i]

    print("\nMAP parameters:")
    for nm in names:
        print(f"  {nm:12s} = {m_map[nm]:.6e}  (initial: {INITIAL[nm]:.6e})")

    with stop_annotating():
        for c, v in zip(controls, x_map): c.assign(float(v))
        J_map = float(rf(controls))
    print(f"\nFinal J(MAP) = {J_map:.6e}")
    print(f"Surface h = {float(h_surf):.1f}m")

    # ----- Save -----
    map_out = {
        "script": Path(__file__).name,
        "J_map": J_map, "J_hist": J_hist,
        "m_map": m_map, "INITIAL": INITIAL, "names": names,
        "h_final": float(h_surf),
    }
    with open(OUT_DIR / f"{OUT_PREFIX}.json", "w") as f:
        json.dump(map_out, f, indent=2)
    print(f"Saved {OUT_DIR / f'{OUT_PREFIX}.json'}")
    print("\nDone.")
