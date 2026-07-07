"""southpole_inversion_v15_stokes.py

Stokes compressible densification inversion (n=1).
CG density, fixed mesh, on-tape spinup. Smooth physics → clean adjoint.

Controls (all log-space):
  - E_lin:  enhancement factor (overall rate magnitude)
  - Q_glen: activation energy (temperature sensitivity)
  - c2_a:   stage-1 compressibility slope for a(rho_hat)
  - c2_b:   stage-1 compressibility slope for b(rho_hat)

Data: density + age + temperature (all three).
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

# Stokes n=1 defaults from Rathmann (firndens-1d)
INITIAL = dict(
    E_lin=5.0e9,       # enhancement factor for n=1
    Q_glen=60.0e3,     # activation energy [J/mol]
    c2_a=15.78652,     # stage-1 a-slope
    c2_b=20.46489,     # stage-1 b-slope
    Q_base=0.05,       # basal heat flux [W/m²]
)
BOUNDS = dict(
    E_lin=(1.0e6, 1.0e13),
    Q_glen=(10.0e3, 200.0e3),
    c2_a=(5.0, 40.0),
    c2_b=(5.0, 40.0),
    Q_base=(-5.0, 5.0),
)
LOG = {"E_lin", "Q_glen"}
PRIOR_SIG = dict(E_lin=1.0, Q_glen=0.5, c2_a=0.3, c2_b=0.3, Q_base=0.5)
MAX_ITER = int(os.environ.get("FIRN_MAX_ITER", "30"))


def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("v15: Stokes n=1, CG, fixed mesh, on-tape spinup")
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
    yr1 = np.arange(740, 1940, 1.0)
    sT1 = np.interp(yr1, bT["year_CE"].values, bT["temp"].values + 273.15)
    sA1 = np.interp(yr1, bA["year_CE"].values, bA["accum"].values)
    blk = int(SPINUP_DT)
    sT = np.array([sT1[i*blk:(i+1)*blk].mean() for i in range(len(sT1)//blk)])
    sA = np.array([sA1[i*blk:(i+1)*blk].mean() for i in range(len(sA1)//blk)])
    n_spin = len(sT); dt_spin = SPINUP_DT * YEAR_S
    print(f"  Tape: {n_spin} spinup + {n_era5} ERA5 = {n_spin+n_era5} steps")

    # ---- Load observations ----
    df = pd.read_csv(PROC / "sp19_density.csv"); df = df[df["depth_m"] <= H0]
    obs_rho_d, obs_rho = df["depth_m"].values, df["rho_kgm3"].values * 1000
    sig_rho = np.full_like(obs_rho, np.median(15.0 + 0.03 * np.abs(obs_rho)))

    df = pd.read_csv(PROC / "sp19_depth_age.csv")
    df = df[(df["year_CE"] <= 2015) & (df["depth_m"] <= H0)]
    obs_age_d = df["depth_m"].values[::5]
    # Age in YEARS (solver returns seconds; convert at comparison time)
    obs_age_yr = 2015.0 - df["year_CE"].values[::5]
    sig_age_yr = 1.0 + 0.005 * obs_age_yr  # 1 yr at surface → 7 yr at 1200 yr

    # Temperature in CELSIUS (solver returns enthalpy; convert at comparison time)
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

    # ---- Controls ----
    log_E = mk(R, math.log(INITIAL["E_lin"]), "lE")
    log_Q = mk(R, math.log(INITIAL["Q_glen"]), "lQ")
    c2a = mk(R, INITIAL["c2_a"], "c2a")
    c2b = mk(R, INITIAL["c2_b"], "c2b")
    Q_base = mk(R, INITIAL["Q_base"], "Qb")

    ctrls = [log_E, log_Q, c2a, c2b, Q_base]
    names = ["E_lin", "Q_glen", "c2_a", "c2_b", "Q_base"]
    n_ctrl = len(ctrls)

    lb = np.array([math.log(BOUNDS[n][0]) if n in LOG else BOUNDS[n][0] for n in names])
    ub = np.array([math.log(BOUNDS[n][1]) if n in LOG else BOUNDS[n][1] for n in names])
    lb_fns = [mk(R, float(lb[i]), f"{names[i]}_lb") for i in range(n_ctrl)]
    ub_fns = [mk(R, float(ub[i]), f"{names[i]}_ub") for i in range(n_ctrl)]
    bounds = list(zip(lb_fns, ub_fns))

    # ---- Model ----
    params = FirnParameters(
        stokes_n_glen=1.0,
        stokes_E_lin=fd.exp(log_E),
        stokes_Q_glen=fd.exp(log_Q),
        stokes_c2_a=c2a,
        stokes_c2_b=c2b,
        basal_heat_flux_W_m2=Q_base,
    )
    model = FirnModel(params, densification_rate_fn=stokes_compressible)
    solver = FirnColumnSolver(model, surface_id=SID)

    Ts = mk(R, float(sT[0]), "Ts")
    Hs = mk(R, float(params.c_i) * (sT[0] - float(params.T_ref)), "Hs")
    ac = mk(R, float(sA[0]), "ac")
    rs = mk(R, RHO_SURF, "rs")
    ws = mk(R, -sA[0] * float(params.rho_i) / RHO_SURF / YEAR_S, "ws")
    dt = mk(R, dt_spin, "dt")

    bc_H = fd.DirichletBC(V, Hs, SID)
    bc_rho = fd.DirichletBC(V, rs, SID)
    bc_w = fd.DirichletBC(V, ws, SID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)

    # ---- Forward model ----
    def forward():
        T0, A0 = float(sT[0]), float(sA[0])
        H_f.assign(float(params.c_i) * (T0 - float(params.T_ref)))
        rho_f.interpolate(fd.Constant(RHO_SURF) +
            (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) *
            (1.0 - fd.exp(-depth / 20.0)))
        w_f.assign(-A0 * float(params.rho_i) / RHO_SURF / YEAR_S)
        age_f.assign(0.0)

        # Spinup
        dt.assign(dt_spin)
        for k in range(n_spin):
            Tk, Ak = float(sT[k]), float(sA[k])
            Ts.assign(Tk); Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
            ac.assign(Ak); ws.assign(-Ak * float(params.rho_i) / RHO_SURF / YEAR_S)
            solver.prognostic_solve(
                enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt,
                accumulation=ac, surface_density=rs,
                boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age,
            )

        # ERA5
        dt.assign(dt_era5)
        for k in range(n_era5):
            Tk, Ak = float(eT[k]), float(eA[k])
            Ts.assign(Tk); Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
            ac.assign(Ak); ws.assign(-Ak * float(params.rho_i) / RHO_SURF / YEAR_S)
            solver.prognostic_solve(
                enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt,
                accumulation=ac, surface_density=rs,
                boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age,
            )

        # Misfit (icepack 1/N normalization)
        J = 0.0
        rho_pred_f.interpolate(rho_f)
        r = (rho_pred_f - rho_obs_f) / rho_sig_f
        J = J + 0.5 / float(N_rho) * fd.assemble(r * r * fd.dx(domain=vom_rho))

        # Age: convert seconds → years via CG1 intermediate, then VOM
        _age_yr = fd.Function(V, name="age_yr")
        _age_yr.interpolate(age_f / fd.Constant(YEAR_S))
        age_pred_f.interpolate(_age_yr)
        r = (age_pred_f - age_obs_f) / age_sig_f
        J = J + 0.5 / float(N_age) * fd.assemble(r * r * fd.dx(domain=vom_age))

        # Temperature: convert enthalpy → °C via CG1 intermediate, then VOM
        _T_C = fd.Function(V, name="T_C")
        _T_C.interpolate(H_f / fd.Constant(float(P0.c_i)))
        T_pred_f.interpolate(_T_C)
        r = (T_pred_f - T_obs_f) / T_sig_f
        J = J + 0.5 / float(N_T) * fd.assemble(r * r * fd.dx(domain=vom_T))

        # Priors
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

    with open(OUT / "southpole_map_v15_stokes.json", "w") as f:
        json.dump({"J_map": Jm, "J_hist": J_hist, "m_map": m_map, "INITIAL": INITIAL}, f, indent=2)
    print("Done.")
