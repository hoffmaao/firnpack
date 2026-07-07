"""southpole_inversion_v19_stokes.py

Stokes compressible densification inversion (n=1).
Follows v18 pattern: CG density, fixed mesh, on-tape spinup, LinMore.

Adds temperature data + Q_geo control to v18's density-only setup.

9 controls (all log-space): A_glen, Q_glen, c1_a, c2_a, c1_b, c2_b, Q_base, k_ice, w_scale
  - Stokes rate: A_glen (rate factor, calibrated 5e-15), Q_glen (activation energy)
  - Stokes compressibility: c1_a, c2_a, c1_b, c2_b
  - Enthalpy: Q_base (basal heat flux), k_ice (thermal conductivity)
  - Velocity: w_scale (basal velocity scale factor)
Data: density + age (relaxed σ) + temperature.
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import firedrake as fd
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, pause_annotation,
    stop_annotating, get_working_tape,
)
from gadopt.inverse import MinimizationProblem, LinMoreOptimiser, minimisation_parameters
from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.physics.densification import stokes_compressible
from firn.constants import year as YEAR_S

_HERE = Path(__file__).parent
_SP = _HERE.parent.parent
PROC = _SP / "processed"
OUT = _SP / "results"

H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF = 350.0
SPINUP_DT = 10.0
ERA5_DT = 0.5
P0 = FirnParameters()

# All controls in log space: optimizer sees log(x), model sees exp(log_x)
INITIAL = dict(
    A_glen=5.0e-15,    # Glen rate factor [Pa^-1 s^-1] (calibrated for 130m SP column)
    Q_glen=60.0e3,     # activation energy [J/mol]
    c1_a=13.22240,     # stage-1 a-intercept
    c2_a=15.78652,     # stage-1 a-slope
    c1_b=15.09371,     # stage-1 b-intercept
    c2_b=20.46489,     # stage-1 b-slope
    Q_base=0.0,        # basal heat flux [W/m²] (can be negative = cooling)
    k_ice=2.1,         # thermal conductivity of pure ice [W/m/K]
    w_scale=1.0,       # basal velocity scale factor (dimensionless)
)
BOUNDS = dict(
    A_glen=(1.0e-17, 1.0e-12),
    Q_glen=(10.0e3, 500.0e3),
    c1_a=(5.0, 30.0),
    c2_a=(5.0, 40.0),
    c1_b=(1.0, 30.0),
    c2_b=(5.0, 50.0),
    Q_base=(-0.50, 0.50),
    k_ice=(0.5, 5.0),
    w_scale=(0.2, 5.0),
)
# Most controls are log-space; Q_base is linear (can be negative)
LOG = set(INITIAL.keys()) - {"Q_base"}
PRIOR_SIG = dict(A_glen=1.0, Q_glen=0.5, c1_a=0.1, c2_a=0.1,
                 c1_b=0.1, c2_b=0.1, Q_base=0.10,  # linear, W/m²
                 k_ice=0.3, w_scale=0.3)
MAX_ITER = int(os.environ.get("FIRN_MAX_ITER", "50"))


def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("v19: Stokes n=1, rho+age+T, c1+c2+Q_base+k_ice, LinMore")
    print("=" * 60)

    # ---- Load forcing ----
    era5 = pd.read_csv(PROC / "era5_monthly_point.csv")
    tR = era5[era5["t2m_K"].notna()].groupby("year")["t2m_K"].mean()
    aR = era5[era5["net_accum_m_iceeq_month"].notna()].groupby("year")["net_accum_m_iceeq_month"].sum()
    yrs = sorted(tR.index.intersection(aR.index))
    m = (np.array(yrs) >= 1940) & (np.array(yrs) < 2015)
    eT = np.repeat(tR.loc[yrs].values[m], 2)
    eA = np.repeat((aR.loc[yrs].values[m] + 0.0163), 2)
    n_era5 = len(eT); dt_era5 = ERA5_DT * YEAR_S

    bT = pd.read_csv(PROC / "buizert2021_spice_temp.csv").sort_values("year_CE")
    bA = pd.read_csv(PROC / "buizert2021_spice_accum.csv").sort_values("year_CE")
    # Buizert temps are ~6°C colder than borehole observations at this site.
    # Adjust to match borehole deep T (~-45.5°C vs Buizert mean ~-51.3°C).
    BUIZERT_T_OFFSET = 5.8  # °C warmer
    yr1 = np.arange(740, 1940, 1.0)
    sT1 = np.interp(yr1, bT["year_CE"].values, bT["temp"].values + BUIZERT_T_OFFSET + 273.15)
    print(f"  Buizert T offset: +{BUIZERT_T_OFFSET}°C (mean: {sT1.mean()-273.15:.1f}°C)")
    sA1 = np.interp(yr1, bA["year_CE"].values, bA["accum"].values)
    blk = int(SPINUP_DT)
    sT = np.array([sT1[i*blk:(i+1)*blk].mean() for i in range(len(sT1)//blk)])
    sA = np.array([sA1[i*blk:(i+1)*blk].mean() for i in range(len(sA1)//blk)])
    n_spin = len(sT); dt_spin = SPINUP_DT * YEAR_S
    print(f"  Tape: {n_spin} spinup + {n_era5} ERA5 = {n_spin+n_era5} steps")

    # ---- Load observations: density + age + temperature ----
    df = pd.read_csv(PROC / "sp19_density.csv"); df = df[df["depth_m"] <= H0]
    obs_rho_d, obs_rho = df["depth_m"].values, df["rho_kgm3"].values * 1000
    sig_rho = np.full_like(obs_rho, np.median(15.0 + 0.03 * np.abs(obs_rho)))

    df = pd.read_csv(PROC / "sp19_depth_age.csv")
    df = df[(df["year_CE"] <= 2015) & (df["depth_m"] <= H0)]
    obs_age_d = df["depth_m"].values[::5]
    obs_age_yr = 2015.0 - df["year_CE"].values[::5]
    sig_age_yr = 10.0 + 0.05 * obs_age_yr  # 5% relative, min 10 yr

    df = pd.read_csv(PROC / "spicecore_borehole_T.csv"); df = df[df["depth_m"] <= H0]
    obs_T_d = df["depth_m"].values
    obs_T_C = df["T_C"].values
    sig_T_C = np.full(len(obs_T_d), 0.5)  # 0.5°C

    N_rho, N_age, N_T = len(obs_rho_d), len(obs_age_d), len(obs_T_d)
    print(f"  Obs: {N_rho} rho, {N_age} age, {N_T} T")

    # ---- Mesh ----
    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
    mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)
    dx = fd.dx(domain=mesh)

    # ---- VOMs ----
    eps_x = 1e-8 * H0
    def d2x(depths):
        x = H0 - np.clip(depths, 0, H0)
        return np.where(x <= 0, eps_x, np.where(x >= H0, H0 - eps_x, x))

    with stop_annotating():
        _prior = fd.Function(V, name="prior")
        dlen = float(fd.assemble(fd.Constant(1.0) * dx)) + 1e-30

        vom_rho = fd.VertexOnlyMesh(mesh, d2x(obs_rho_d).reshape(-1, 1))
        Pr = fd.FunctionSpace(vom_rho, "DG", 0)
        rho_obs_f = fd.Function(Pr); rho_obs_f.dat.data[:] = obs_rho
        rho_sig_f = fd.Function(Pr); rho_sig_f.dat.data[:] = sig_rho
        rho_pred_f = fd.Function(Pr)

        vom_age = fd.VertexOnlyMesh(mesh, d2x(obs_age_d).reshape(-1, 1))
        Pa = fd.FunctionSpace(vom_age, "DG", 0)
        age_obs_f = fd.Function(Pa); age_obs_f.dat.data[:] = obs_age_yr
        age_sig_f = fd.Function(Pa); age_sig_f.dat.data[:] = sig_age_yr
        age_pred_f = fd.Function(Pa)

        vom_T = fd.VertexOnlyMesh(mesh, d2x(obs_T_d).reshape(-1, 1))
        Pt = fd.FunctionSpace(vom_T, "DG", 0)
        T_obs_f = fd.Function(Pt); T_obs_f.dat.data[:] = obs_T_C
        T_sig_f = fd.Function(Pt); T_sig_f.dat.data[:] = sig_T_C
        T_pred_f = fd.Function(Pt)

    # ---- State ----
    H_f = fd.Function(V, name="H"); rho_f = fd.Function(V, name="rho")
    w_f = fd.Function(V, name="w"); age_f = fd.Function(V, name="age")
    depth = H0 - xi

    # ---- Controls (log-space for positive params, linear for Q_base) ----
    raw_ctrls = {}
    for nm in INITIAL:
        if nm in LOG:
            raw_ctrls[nm] = mk(R, math.log(INITIAL[nm]), f"l_{nm}")
        else:
            raw_ctrls[nm] = mk(R, INITIAL[nm], f"{nm}")
    ctrls = [raw_ctrls[nm] for nm in INITIAL]
    names = list(INITIAL.keys())
    n_ctrl = len(ctrls)

    lb = np.array([math.log(BOUNDS[n][0]) if n in LOG else BOUNDS[n][0] for n in names])
    ub = np.array([math.log(BOUNDS[n][1]) if n in LOG else BOUNDS[n][1] for n in names])
    lb_fns = [mk(R, float(lb[i]), f"{names[i]}_lb") for i in range(n_ctrl)]
    ub_fns = [mk(R, float(ub[i]), f"{names[i]}_ub") for i in range(n_ctrl)]
    bounds = list(zip(lb_fns, ub_fns))

    # Physical-space expressions
    p_A = fd.exp(raw_ctrls["A_glen"])
    p_Q = fd.exp(raw_ctrls["Q_glen"])
    p_c1a = fd.exp(raw_ctrls["c1_a"])
    p_c2a = fd.exp(raw_ctrls["c2_a"])
    p_c1b = fd.exp(raw_ctrls["c1_b"])
    p_c2b = fd.exp(raw_ctrls["c2_b"])
    p_Qbase = raw_ctrls["Q_base"]  # linear, can be negative
    p_kice = fd.exp(raw_ctrls["k_ice"])
    p_wscale = fd.exp(raw_ctrls["w_scale"])

    # ---- Model ----
    params = FirnParameters(
        stokes_n_glen=1.0,
        stokes_A_glen=p_A,
        stokes_E_lin=1.0,   # folded into A_glen
        stokes_Q_glen=p_Q,
        stokes_c1_a=p_c1a,
        stokes_c2_a=p_c2a,
        stokes_c1_b=p_c1b,
        stokes_c2_b=p_c2b,
        basal_heat_flux_W_m2=p_Qbase,
        k_ice=p_kice,
    )
    model = FirnModel(params, densification_rate_fn=stokes_compressible)
    solver = FirnColumnSolver(model, surface_id=SID)

    Ts = mk(R, float(sT[0]), "Ts")
    Hs = mk(R, float(params.c_i) * (sT[0] - float(params.T_ref)), "Hs")
    ac = mk(R, float(sA[0]), "ac")
    rs = mk(R, RHO_SURF, "rs")
    # Basal velocity: w_base = -bdot * w_scale (control-dependent)
    # Use CG1 to avoid R-to-R interpolation adjoint bug
    wb_cg1 = fd.Function(V, name="wb")
    wb_cg1.assign(-sA[0] / YEAR_S)
    dt = mk(R, dt_spin, "dt")

    BASE_ID = 1  # left boundary of IntervalMesh
    bc_H = fd.DirichletBC(V, Hs, SID)
    bc_rho = fd.DirichletBC(V, rs, SID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)
    # Mass-flux BC: constrain velocity at base, surface velocity emerges
    bc_w_base = fd.DirichletBC(V, wb_cg1, BASE_ID)

    # ---- Forward model ----
    def forward():
        T0, A0 = float(sT[0]), float(sA[0])
        H_f.assign(float(params.c_i) * (T0 - float(params.T_ref)))
        rho_f.interpolate(fd.Constant(RHO_SURF) +
            (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) *
            (1.0 - fd.exp(-depth / 20.0)))
        w_f.assign(-A0 / YEAR_S)
        age_f.assign(0.0)

        # Spinup
        dt.assign(dt_spin)
        for k in range(n_spin):
            Tk, Ak = float(sT[k]), float(sA[k])
            Ts.assign(Tk); Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
            ac.assign(Ak)
            # Basal velocity: control-dependent via w_scale
            wb_cg1.interpolate(fd.Constant(-Ak / YEAR_S) * p_wscale)
            solver.prognostic_solve(
                enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt,
                accumulation=ac, surface_density=rs,
                boundary_conditions=[bc_H, bc_rho, None],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age,
                mass_flux_bc=bc_w_base,
            )

        # ERA5
        dt.assign(dt_era5)
        for k in range(n_era5):
            Tk, Ak = float(eT[k]), float(eA[k])
            Ts.assign(Tk); Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
            ac.assign(Ak)
            wb_cg1.interpolate(fd.Constant(-Ak / YEAR_S) * p_wscale)
            solver.prognostic_solve(
                enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt,
                accumulation=ac, surface_density=rs,
                boundary_conditions=[bc_H, bc_rho, None],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age,
                mass_flux_bc=bc_w_base,
            )

        # Misfit: density + age + temperature (icepack 1/(2N) normalization)
        J = 0.0

        # Density
        rho_pred_f.interpolate(rho_f)
        r = (rho_pred_f - rho_obs_f) / rho_sig_f
        J = J + 0.5 / float(N_rho) * fd.assemble(r * r * fd.dx(domain=vom_rho))

        # Age (seconds → years via CG1 intermediate)
        _age_yr = fd.Function(V, name="age_yr")
        _age_yr.interpolate(age_f / fd.Constant(YEAR_S))
        age_pred_f.interpolate(_age_yr)
        r = (age_pred_f - age_obs_f) / age_sig_f
        J = J + 0.5 / float(N_age) * fd.assemble(r * r * fd.dx(domain=vom_age))

        # Temperature (enthalpy → °C via CG1 intermediate)
        _T_C = fd.Function(V, name="T_C")
        _T_C.interpolate(H_f / fd.Constant(float(P0.c_i)))
        T_pred_f.interpolate(_T_C)
        r = (T_pred_f - T_obs_f) / T_sig_f
        J = J + 0.5 / float(N_T) * fd.assemble(r * r * fd.dx(domain=vom_T))

        # Priors (log-space for LOG params, linear for Q_base)
        _n_inv = 1.0 / n_ctrl
        for c, nm in zip(ctrls, names):
            if nm in LOG:
                _prior.interpolate((c - math.log(float(INITIAL[nm]))) / PRIOR_SIG[nm])
            else:
                _prior.interpolate((c - fd.Constant(float(INITIAL[nm]))) / fd.Constant(PRIOR_SIG[nm]))
            J = J + _n_inv * 0.5 * fd.assemble(_prior * _prior * dx) / dlen
        return J

    # ---- Build tape ----
    tape = get_working_tape(); tape.clear_tape()
    continue_annotation()
    J_init = forward()
    rf = ReducedFunctional(J_init, [Control(c) for c in ctrls])
    pause_annotation()
    print(f"\nJ_init = {float(J_init):.4f}")

    # ---- Taylor test ----
    x0 = np.array([float(c.dat.data_ro[0]) for c in ctrls])
    print("\nTaylor test ...")
    with stop_annotating():
        for c, v in zip(ctrls, x0): c.assign(float(v))
        J0 = float(rf(ctrls))
        g_list = rf.derivative()
    g = np.array([float(gi.dat.data_ro[0]) * dlen for gi in g_list])
    rng = np.random.default_rng(0)
    d = rng.standard_normal(n_ctrl) * 0.01
    gd = g @ d
    print(f"  J={J0:.4f}, <dJ,d>={gd:.6e}")
    prev = None
    for eps in [0.1, 0.05, 0.025, 0.0125, 0.00625]:
        m1 = np.clip(x0 + eps * d, lb, ub)
        try:
            with stop_annotating():
                for c, v in zip(ctrls, m1): c.assign(float(v))
                J1 = float(rf(ctrls))
            r = abs(J1 - J0 - eps * gd)
            o = f"{math.log(prev / r) / math.log(2):.2f}" if prev and r > 0 else ""
            print(f"  {eps:.4f}  {abs(J1 - J0):.6e}  {r:.6e}  {o}")
            prev = r
        except Exception as e:
            print(f"  {eps:.4f}  FAILED: {e}")

    # ---- Optimize ----
    with stop_annotating():
        for c, v in zip(ctrls, x0): c.assign(float(v))
        rf(ctrls)

    J_hist = []
    _ne = [0]
    def _cb(val, *a, **kw):
        _ne[0] += 1; J_hist.append(float(val))
        if _ne[0] % 5 == 0 or _ne[0] == 1:
            print(f"  [iter {_ne[0]:03d}] J = {float(val):.6e}")
    rf.eval_cb_post = _cb

    minimisation_parameters["Status Test"]["Iteration Limit"] = MAX_ITER
    minimisation_parameters["Status Test"]["Gradient Tolerance"] = 1e-10
    mp = MinimizationProblem(rf, bounds=bounds)
    opt = LinMoreOptimiser(mp, minimisation_parameters, auto_checkpoint=False)
    print(f"\nOptimizing (max {MAX_ITER} iters) ...")
    try:
        opt.run()
    except fd.exceptions.ConvergenceError as e:
        print(f"\n  Stopped ({_ne[0]} evals)")

    # Commit ROL's best iterate
    for c in ctrls:
        if hasattr(c, "block_variable") and getattr(c.block_variable, "checkpoint", None) is not None:
            c.assign(c.block_variable.checkpoint)

    x_map = np.array([float(c.dat.data_ro[0]) for c in ctrls])
    m_map = {n: (math.exp(x_map[i]) if n in LOG else x_map[i]) for i, n in enumerate(names)}
    print("\nMAP:")
    for n in names:
        print(f"  {n:12s} = {m_map[n]:.4e}  (init: {INITIAL[n]:.4e})")
    with stop_annotating():
        for c, v in zip(ctrls, x_map): c.assign(float(v))
        Jm = float(rf(ctrls))
    print(f"\nJ_MAP = {Jm:.4f}")

    with open(OUT / "southpole_map_v19_stokes.json", "w") as f:
        json.dump({"J_map": Jm, "J_hist": J_hist, "m_map": m_map, "INITIAL": INITIAL}, f, indent=2)
    print("Done.")
