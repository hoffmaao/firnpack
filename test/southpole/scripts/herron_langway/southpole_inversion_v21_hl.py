"""southpole_inversion_v21_hl.py

Tape-rebuild H&L inversion: clears and rebuilds the pyadjoint tape for
every J/gradient evaluation, fixing the tape replay bug that caused
time-varying forcing to be silently dropped in v14-v20.

Uses scipy L-BFGS-B (no ReducedFunctional replay needed).

5 controls: hl_k0, hl_k1, hl_Ea1, hl_Ea2 (log-space), Q_base (linear)
Forcing: Buizert 740-1940 (+5.8 C offset) + ERA5 1940-2015
Data: density + age + temperature
"""
from __future__ import annotations
import functools, json, math, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize

import firedrake as fd
from firedrake.adjoint import (
    Control, continue_annotation, pause_annotation,
    stop_annotating, get_working_tape,
)
from pyadjoint import compute_gradient
from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.solvers.firn_solver import FirnColumnSolver
from firnpack.physics.densification import herron_langway as _herron_langway
herron_langway = functools.partial(_herron_langway, smooth=True)
from firnpack.constants import year as YEAR_S

_HERE = Path(__file__).parent
_SP = _HERE.parent.parent
PROC = _SP / "processed"
OUT = _SP / "results"

H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF = 350.0
SPINUP_DT = 10.0   # years per spinup block
ERA5_DT = 0.5      # years per ERA5 step
BUIZERT_T_OFFSET = 5.8  # cloud-to-surface correction (C)

P0 = FirnParameters()

INITIAL = dict(hl_k0=11.0, hl_k1=575.0, hl_Ea1=10160.0, hl_Ea2=21400.0,
               Q_base=0.05)
BOUNDS = dict(
    hl_k0=(0.5, 500.0), hl_k1=(10.0, 50000.0),
    hl_Ea1=(3000.0, 40000.0), hl_Ea2=(5000.0, 80000.0),
    Q_base=(-0.2, 0.2),
)
LOG = {"hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2"}
PRIOR_SIG = dict(hl_k0=1.0, hl_k1=1.0, hl_Ea1=0.3, hl_Ea2=0.3, Q_base=0.05)
MAX_ITER = int(os.environ.get("FIRN_MAX_ITER", "50"))

# Data weights (set to 0 to disable a data type)
W_RHO = 1.0
W_AGE = float(os.environ.get("FIRN_W_AGE", "0.0"))   # age off by default
W_T   = 1.0

NAMES = ["hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "Q_base"]
N_CTRL = len(NAMES)

def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("v21: H&L, tape-rebuild, Buizert+ERA5, scipy L-BFGS-B")
    print("=" * 60)

    # ---- Load forcing ----
    era5 = pd.read_csv(PROC / "era5_monthly_point.csv")
    tR = era5[era5["t2m_K"].notna()].groupby("year")["t2m_K"].mean()
    aR = era5[era5["net_accum_m_iceeq_month"].notna()].groupby("year")["net_accum_m_iceeq_month"].sum()
    yrs = sorted(tR.index.intersection(aR.index))
    m = (np.array(yrs) >= 1940) & (np.array(yrs) < 2015)
    eT = np.repeat(tR.loc[yrs].values[m], 2)            # ERA5 K (already surface T)
    eA = np.repeat((aR.loc[yrs].values[m] + 0.0163), 2) # m ice-eq/yr

    bT = pd.read_csv(PROC / "buizert2021_spice_temp.csv").sort_values("year_CE")
    bA = pd.read_csv(PROC / "buizert2021_spice_accum.csv").sort_values("year_CE")
    yr1 = np.arange(740, 1940, 1.0)
    sT1 = np.interp(yr1, bT["year_CE"].values,
                     bT["temp"].values + BUIZERT_T_OFFSET + 273.15)
    sA1 = np.interp(yr1, bA["year_CE"].values, bA["accum"].values)
    blk = int(SPINUP_DT)
    sT = np.array([sT1[i*blk:(i+1)*blk].mean() for i in range(len(sT1)//blk)])
    sA = np.array([sA1[i*blk:(i+1)*blk].mean() for i in range(len(sA1)//blk)])
    n_spin = len(sT); dt_spin = SPINUP_DT * YEAR_S
    n_era5 = len(eT); dt_era5 = ERA5_DT * YEAR_S
    n_total = n_spin + n_era5

    all_T = np.concatenate([sT, eT])
    all_A = np.concatenate([sA, eA])
    all_dt = np.concatenate([np.full(n_spin, dt_spin), np.full(n_era5, dt_era5)])

    print(f"  Buizert T: mean {sT.mean()-273.15:.1f} C (with +{BUIZERT_T_OFFSET} C offset)")
    print(f"  ERA5 T: mean {eT.mean()-273.15:.1f} C")
    print(f"  Tape: {n_spin} spinup + {n_era5} ERA5 = {n_total} steps")

    # ---- Load observations ----
    df = pd.read_csv(PROC / "sp19_density.csv"); df = df[df["depth_m"] <= H0]
    obs_rho_d, obs_rho = df["depth_m"].values, df["rho_kgm3"].values * 1000
    sig_rho = np.full_like(obs_rho, np.median(15.0 + 0.03 * np.abs(obs_rho)))

    df = pd.read_csv(PROC / "sp19_depth_age.csv")
    df = df[(df["year_CE"] <= 2015) & (df["depth_m"] <= H0)]
    obs_age_d = df["depth_m"].values[::5]
    obs_age_s = (2015.0 - df["year_CE"].values[::5]) * YEAR_S
    obs_age_yr = obs_age_s / YEAR_S
    sig_age = (1.0 + 0.005 * obs_age_yr) * YEAR_S

    df = pd.read_csv(PROC / "spicecore_borehole_T.csv"); df = df[df["depth_m"] <= H0]
    obs_T_d = df["depth_m"].values
    obs_T_H = float(P0.c_i) * (df["T_C"].values + 273.15 - float(P0.T_ref))
    sig_T_H = float(P0.c_i) * np.full(len(obs_T_d), 2.0)  # 2 C structural error

    N_rho, N_age, N_T = len(obs_rho_d), len(obs_age_d), len(obs_T_d)
    print(f"  Obs: {N_rho} rho, {N_age} age, {N_T} T (sig_T=2 C)")
    print(f"  Weights: W_rho={W_RHO}, W_age={W_AGE}, W_T={W_T}")

    # ---- Mesh ----
    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
    mesh.coordinates.assign(
        fd.Function(mesh.coordinates.function_space()).interpolate(
            fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)
    dx = fd.dx(domain=mesh)

    # ---- VOMs (off-tape, geometry only) ----
    eps_x = 1e-8 * H0
    def d2x(depths):
        x = H0 - np.clip(depths, 0, H0)
        return np.where(x <= 0, eps_x, np.where(x >= H0, H0-eps_x, x))

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
        age_obs_f = fd.Function(Pa); age_obs_f.dat.data[:] = obs_age_s
        age_sig_f = fd.Function(Pa); age_sig_f.dat.data[:] = sig_age
        age_pred_f = fd.Function(Pa)

        vom_T = fd.VertexOnlyMesh(mesh, d2x(obs_T_d).reshape(-1, 1))
        Pt = fd.FunctionSpace(vom_T, "DG", 0)
        T_obs_f = fd.Function(Pt); T_obs_f.dat.data[:] = obs_T_H
        T_sig_f = fd.Function(Pt); T_sig_f.dat.data[:] = sig_T_H
        T_pred_f = fd.Function(Pt)

    # ---- State ----
    H_f = fd.Function(V, name="H"); rho_f = fd.Function(V, name="rho")
    w_f = fd.Function(V, name="w"); age_f = fd.Function(V, name="age")
    depth = H0 - xi

    # ---- Controls (persistent Function objects, values updated each eval) ----
    log_k0 = mk(R, math.log(INITIAL["hl_k0"]), "lk0")
    log_k1 = mk(R, math.log(INITIAL["hl_k1"]), "lk1")
    log_Ea1 = mk(R, math.log(INITIAL["hl_Ea1"]), "lEa1")
    log_Ea2 = mk(R, math.log(INITIAL["hl_Ea2"]), "lEa2")
    Q_base = mk(R, INITIAL["Q_base"], "Qb")

    ctrl_fns = [log_k0, log_k1, log_Ea1, log_Ea2, Q_base]

    x0 = np.array([float(c.dat.data_ro[0]) for c in ctrl_fns])
    lb = np.array([math.log(BOUNDS[n][0]) if n in LOG else BOUNDS[n][0] for n in NAMES])
    ub = np.array([math.log(BOUNDS[n][1]) if n in LOG else BOUNDS[n][1] for n in NAMES])

    # Variable scaling: x_scaled = (x - x0) / scale, so 1 scaled unit = 1 prior sigma
    scale = np.array([PRIOR_SIG[n] for n in NAMES])
    lb_s = (lb - x0) / scale
    ub_s = (ub - x0) / scale

    # ---- Model (UFL expressions reference control Functions) ----
    params = FirnParameters(
        hl_k0_prefactor=fd.exp(log_k0), hl_k1_prefactor=fd.exp(log_k1),
        hl_Ea_stage1=fd.exp(log_Ea1), hl_Ea_stage2=fd.exp(log_Ea2),
        basal_heat_flux_W_m2=Q_base,
    )
    model = FirnModel(params, densification_rate_fn=herron_langway)
    solver = FirnColumnSolver(model, surface_id=SID)

    # ---- Forcing / BC Functions ----
    Ts = mk(R, float(all_T[0]), "Ts")
    Hs = mk(R, float(params.c_i)*(all_T[0]-float(params.T_ref)), "Hs")
    ac = mk(R, float(all_A[0]), "ac")
    rs = mk(R, RHO_SURF, "rs")
    ws = mk(R, -all_A[0]*float(params.rho_i)/RHO_SURF/YEAR_S, "ws")
    dt_r = mk(R, dt_spin, "dt")

    bc_H = fd.DirichletBC(V, Hs, SID)
    bc_rho = fd.DirichletBC(V, rs, SID)
    bc_w = fd.DirichletBC(V, ws, SID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)

    c_i_f = float(params.c_i)
    T_ref_f = float(params.T_ref)
    rho_i_f = float(params.rho_i)

    # ---- Forward model ----
    def forward():
        T0, A0 = float(all_T[0]), float(all_A[0])
        H_f.assign(c_i_f * (T0 - T_ref_f))
        rho_f.interpolate(fd.Constant(RHO_SURF) +
            (fd.Constant(rho_i_f) - fd.Constant(RHO_SURF)) *
            (1.0 - fd.exp(-depth / 20.0)))
        w_f.assign(-A0 * rho_i_f / RHO_SURF / YEAR_S)
        age_f.assign(0.0)

        for k in range(n_total):
            Tk, Ak = float(all_T[k]), float(all_A[k])
            Ts.assign(Tk)
            Hs.assign(c_i_f * (Tk - T_ref_f))
            ac.assign(Ak)
            ws.assign(-Ak * rho_i_f / RHO_SURF / YEAR_S)
            dt_r.assign(float(all_dt[k]))

            solver.prognostic_solve(
                enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_r,
                accumulation=ac, surface_density=rs,
                boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age,
            )

        # ---- Misfit ----
        J = 0.0
        rho_pred_f.interpolate(rho_f)
        r = (rho_pred_f - rho_obs_f) / rho_sig_f
        J_rho = 0.5 / float(N_rho) * fd.assemble(r * r * fd.dx(domain=vom_rho))
        J = J + W_RHO * J_rho

        age_pred_f.interpolate(age_f)
        r = (age_pred_f - age_obs_f) / age_sig_f
        J_age = 0.5 / float(N_age) * fd.assemble(r * r * fd.dx(domain=vom_age))
        J = J + W_AGE * J_age

        T_pred_f.interpolate(H_f)
        r = (T_pred_f - T_obs_f) / T_sig_f
        J_T = 0.5 / float(N_T) * fd.assemble(r * r * fd.dx(domain=vom_T))
        J = J + W_T * J_T

        # ---- Prior ----
        _n_inv = 1.0 / N_CTRL
        J_prior = 0.0
        for c, nm in zip(ctrl_fns, NAMES):
            if nm in LOG:
                _prior.interpolate((c - math.log(float(INITIAL[nm]))) / PRIOR_SIG[nm])
            else:
                _prior.interpolate(
                    (c - fd.Constant(float(INITIAL[nm]))) / fd.Constant(PRIOR_SIG[nm]))
            J_prior = J_prior + _n_inv * 0.5 * fd.assemble(_prior * _prior * dx) / dlen
        J = J + J_prior

        # Diagnostics (off-tape extraction)
        with stop_annotating():
            rho_arr = rho_f.dat.data_ro
            age_arr = age_f.dat.data_ro / YEAR_S
            forward._diag = {
                "J_rho": float(J_rho), "J_age": float(J_age),
                "J_T": float(J_T), "J_prior": float(J_prior),
                "rho_range": (float(rho_arr.min()), float(rho_arr.max())),
                "age_range": (float(age_arr.min()), float(age_arr.max())),
            }
        return J

    # ---- Tape-rebuild evaluation ----
    tape = get_working_tape()
    n_eval = [0]
    J_hist = []

    def eval_J_and_grad(xs):
        """Evaluate J and gradient in SCALED space."""
        x = x0 + xs * scale
        x = np.clip(x, lb, ub)
        t0 = time.perf_counter()
        tape.clear_tape()
        continue_annotation()

        for c, v in zip(ctrl_fns, x):
            c.assign(float(v))

        try:
            J = forward()
            controls = [Control(c) for c in ctrl_fns]
            dJ = compute_gradient(J, controls)
            J_val = float(J)
            grad_raw = np.array([float(g.dat.data_ro[0]) * dlen for g in dJ])
            grad_s = grad_raw * scale  # chain rule: dJ/dx_s = dJ/dx * dx/dx_s = dJ/dx * scale
        except Exception as e:
            print(f"  *** Forward/adjoint failed: {e}")
            J_val = 1e8
            grad_s = xs * 100.0  # push back toward x0 (xs=0)

        pause_annotation()

        n_eval[0] += 1
        J_hist.append(J_val)
        dt = time.perf_counter() - t0
        m_phys = {n: (math.exp(x[i]) if n in LOG else x[i])
                  for i, n in enumerate(NAMES)}
        dd = forward._diag
        print(f"  [{n_eval[0]:03d}] J={J_val:.4f}  (rho={dd['J_rho']:.1f} age={dd['J_age']:.1f} "
              f"T={dd['J_T']:.1f})  |g_s|={np.linalg.norm(grad_s):.2e}  ({dt:.0f}s)")
        print(f"         k0={m_phys['hl_k0']:.2f} k1={m_phys['hl_k1']:.1f} "
              f"Ea1={m_phys['hl_Ea1']:.0f} Ea2={m_phys['hl_Ea2']:.0f} "
              f"Q={m_phys['Q_base']:.4f}  "
              f"rho=[{dd['rho_range'][0]:.0f},{dd['rho_range'][1]:.0f}]")
        return J_val, grad_s

    # ---- Verify: J is reproducible ----
    print("\nVerification (two forward runs, same controls):")
    tape.clear_tape()
    continue_annotation()
    for c, v in zip(ctrl_fns, x0): c.assign(float(v))
    J1 = float(forward())
    pause_annotation()

    tape.clear_tape()
    continue_annotation()
    for c, v in zip(ctrl_fns, x0): c.assign(float(v))
    J2 = float(forward())
    pause_annotation()
    print(f"  J_run1 = {J1:.6f}")
    print(f"  J_run2 = {J2:.6f}")
    print(f"  Match: {abs(J1-J2) < 1e-10}")
    d = forward._diag
    print(f"  Components: J_rho={d['J_rho']:.1f}  J_age={d['J_age']:.1f}  "
          f"J_T={d['J_T']:.1f}  J_prior={d['J_prior']:.4f}")
    print(f"  rho: [{d['rho_range'][0]:.0f}, {d['rho_range'][1]:.0f}] kg/m3")
    print(f"  age: [{d['age_range'][0]:.0f}, {d['age_range'][1]:.0f}] yr")

    # ---- Taylor test ----
    print("\nTaylor test ...")
    tape.clear_tape()
    continue_annotation()
    for c, v in zip(ctrl_fns, x0): c.assign(float(v))
    J0_form = forward()
    controls_tt = [Control(c) for c in ctrl_fns]
    dJ0 = compute_gradient(J0_form, controls_tt)
    pause_annotation()

    J0 = float(J0_form)
    g = np.array([float(gi.dat.data_ro[0]) * dlen for gi in dJ0])
    print(f"  J0 = {J0:.6f},  |g| = {np.linalg.norm(g):.4e}")
    print(f"  Per-component: " + "  ".join(f"dJ/d{n}={g[i]:.3e}" for i,n in enumerate(NAMES)))
    rng = np.random.default_rng(42)
    dd = rng.standard_normal(N_CTRL) * 0.01
    gd = g @ dd
    print(f"  <g,d> = {gd:.6e}")
    prev = None
    for eps in [0.1, 0.05, 0.025, 0.0125, 0.00625]:
        x_pert = np.clip(x0 + eps * dd, lb, ub)
        tape.clear_tape()
        continue_annotation()
        for c, v in zip(ctrl_fns, x_pert): c.assign(float(v))
        J1 = float(forward())
        pause_annotation()
        r = abs(J1 - J0 - eps * gd)
        o = f"{math.log(prev/r)/math.log(2):.2f}" if prev and r > 0 else ""
        print(f"  eps={eps:.4f}  |dJ|={abs(J1-J0):.6e}  rem={r:.6e}  {o}")
        prev = r

    # ---- Optimize (in scaled space, auto-scaled by gradient) ----
    # Re-scale so |g_s| ~ 1 at x0: scale_i = 1/|dJ/dx_i|
    # dJ/dxs = dJ/dx * scale, so scale = 1/|g| gives |dJ/dxs| ~ 1
    g0_raw = g  # gradient at x0 from Taylor test
    scale = 1.0 / np.maximum(np.abs(g0_raw), 1e-6)
    lb_s = (lb - x0) / scale
    ub_s = (ub - x0) / scale

    print(f"\nOptimizing (L-BFGS-B auto-scaled, max {MAX_ITER} iters) ...")
    print(f"  Auto-scale: {dict(zip(NAMES, scale))}")
    n_eval[0] = 0
    J_hist.clear()
    xs0 = np.zeros(N_CTRL)
    result = sp_minimize(
        eval_J_and_grad, xs0, jac=True, method="L-BFGS-B",
        bounds=list(zip(lb_s, ub_s)),
        options={"maxiter": MAX_ITER, "ftol": 1e-12, "gtol": 1e-10,
                 "maxfun": MAX_ITER * 3, "maxls": 30, "disp": False},
    )

    x_map = x0 + result.x * scale
    m_map = {n: (math.exp(x_map[i]) if n in LOG else x_map[i])
             for i, n in enumerate(NAMES)}
    print(f"\nResult: {result.message}")
    print(f"  {result.nit} iterations, {result.nfev} function evaluations")
    print(f"\nMAP:")
    for n in NAMES:
        print(f"  {n:12s} = {m_map[n]:.4e}  (init: {INITIAL[n]:.4e})")
    print(f"\nJ_MAP = {result.fun:.4f}")

    with open(OUT / "southpole_map_v21_hl.json", "w") as f:
        json.dump({"J_map": result.fun, "J_hist": J_hist,
                   "m_map": m_map, "INITIAL": INITIAL,
                   "BUIZERT_T_OFFSET": BUIZERT_T_OFFSET,
                   "scipy_message": result.message,
                   "scipy_nit": result.nit}, f, indent=2)
    print(f"\nSaved {OUT / 'southpole_map_v21_hl.json'}")
    print("Done.")
