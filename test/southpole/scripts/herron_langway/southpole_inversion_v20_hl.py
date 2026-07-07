"""southpole_inversion_v20_hl.py

H&L inversion with corrected Buizert temperature (+5.8 C offset)
and time-varying forcing (Buizert 740-1940 + ERA5 1940-2015).

To avoid the tape replay bug (pyadjoint skips non-control-dependent
R-space assigns), all per-step forcing values are pre-created as
Firedrake Constants before the tape. A small T_offset control makes
every forcing interpolation control-dependent, ensuring correct replay.

6 controls:
    hl_k0, hl_k1, hl_Ea1, hl_Ea2 (log-space) — H&L densification
    Q_base (linear) — basal heat flux [W/m^2]
    T_offset (linear) — additional temperature bias [K], initial=0

Data: density + age + temperature
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
from firn.physics.densification import herron_langway
from firn.constants import year as YEAR_S

_HERE = Path(__file__).parent
_SP = _HERE.parent.parent
PROC = _SP / "processed"
OUT = _SP / "results"

H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF = 350.0
SPINUP_DT = 10.0   # years per spinup block
ERA5_DT = 0.5      # years per ERA5 step

# Cloud-to-surface temperature correction for Buizert reconstruction
BUIZERT_T_OFFSET = 5.8  # degrees C

P0 = FirnParameters()

INITIAL = dict(hl_k0=11.0, hl_k1=575.0, hl_Ea1=10160.0, hl_Ea2=21400.0,
               Q_base=0.0, T_offset=0.0)
BOUNDS = dict(
    hl_k0=(0.01, 10000.0), hl_k1=(0.01, 1e6),
    hl_Ea1=(100.0, 1e6), hl_Ea2=(100.0, 1e6),
    Q_base=(-0.5, 0.5), T_offset=(-5.0, 5.0),
)
LOG = {"hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2"}
PRIOR_SIG = dict(hl_k0=1.0, hl_k1=1.0, hl_Ea1=0.3, hl_Ea2=0.3,
                 Q_base=0.10, T_offset=1.0)
MAX_ITER = int(os.environ.get("FIRN_MAX_ITER", "50"))

def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("v20: H&L, time-varying forcing, Buizert+ERA5, rho+age+T")
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

    print(f"  Buizert T: mean {sT.mean()-273.15:.1f} C (with +{BUIZERT_T_OFFSET} C offset)")
    print(f"  ERA5 T: mean {eT.mean()-273.15:.1f} C")
    print(f"  Tape: {n_spin} spinup + {n_era5} ERA5 = {n_total} steps")

    # Pre-create forcing Constants (before tape, never reassigned)
    all_T = np.concatenate([sT, eT])
    all_A = np.concatenate([sA, eA])

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
    sig_T_H = float(P0.c_i) * np.full(len(obs_T_d), 2.0)  # 2 C — structural model error

    N_rho, N_age, N_T = len(obs_rho_d), len(obs_age_d), len(obs_T_d)
    print(f"  Obs: {N_rho} rho, {N_age} age, {N_T} T (sig_T=2 C)")

    # ---- Mesh ----
    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
    mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)
    dx = fd.dx(domain=mesh)

    # Pre-create per-step forcing as R-space Functions (never reassigned, survive replay)
    with stop_annotating():
        Tk_consts = [mk(R, float(all_T[k]), f"Tk{k}") for k in range(n_total)]
        Ak_consts = [mk(R, float(all_A[k]), f"Ak{k}") for k in range(n_total)]

    # ---- VOMs ----
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

    # ---- Controls ----
    log_k0 = mk(R, math.log(INITIAL["hl_k0"]), "lk0")
    log_k1 = mk(R, math.log(INITIAL["hl_k1"]), "lk1")
    log_Ea1 = mk(R, math.log(INITIAL["hl_Ea1"]), "lEa1")
    log_Ea2 = mk(R, math.log(INITIAL["hl_Ea2"]), "lEa2")
    Q_base = mk(R, INITIAL["Q_base"], "Qb")
    T_offset = mk(R, INITIAL["T_offset"], "Toff")

    ctrls = [log_k0, log_k1, log_Ea1, log_Ea2, Q_base, T_offset]
    names = ["hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "Q_base", "T_offset"]
    n_ctrl = len(ctrls)

    lb = np.array([math.log(BOUNDS[n][0]) if n in LOG else BOUNDS[n][0] for n in names])
    ub = np.array([math.log(BOUNDS[n][1]) if n in LOG else BOUNDS[n][1] for n in names])
    lb_fns = [mk(R, float(lb[i]), f"{names[i]}_lb") for i in range(n_ctrl)]
    ub_fns = [mk(R, float(ub[i]), f"{names[i]}_ub") for i in range(n_ctrl)]
    bounds = list(zip(lb_fns, ub_fns))

    # ---- Model ----
    params = FirnParameters(
        hl_k0_prefactor=fd.exp(log_k0), hl_k1_prefactor=fd.exp(log_k1),
        hl_Ea_stage1=fd.exp(log_Ea1), hl_Ea_stage2=fd.exp(log_Ea2),
        basal_heat_flux_W_m2=Q_base,
    )
    model = FirnModel(params, densification_rate_fn=herron_langway)
    solver = FirnColumnSolver(model, surface_id=SID)

    # CG1 intermediates for control-dependent forcing (avoids tape replay bug)
    Ts_eff = fd.Function(V, name="Ts_eff")
    Hs_eff = fd.Function(V, name="Hs_eff")
    ws_eff = fd.Function(V, name="ws_eff")
    rs = mk(R, RHO_SURF, "rs")
    dt_spin_r = mk(R, dt_spin, "dt_spin")
    dt_era5_r = mk(R, dt_era5, "dt_era5")
    c_i = fd.Constant(float(params.c_i))
    T_ref = fd.Constant(float(params.T_ref))
    rho_i_over_rho_s_yr = fd.Constant(float(params.rho_i) / RHO_SURF / YEAR_S)

    bc_H = fd.DirichletBC(V, Hs_eff, SID)
    bc_rho = fd.DirichletBC(V, rs, SID)
    bc_w = fd.DirichletBC(V, ws_eff, SID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)

    # ---- Forward ----
    def forward():
        T0, A0 = float(all_T[0]), float(all_A[0])
        H_f.assign(float(params.c_i) * (T0 - float(params.T_ref)))
        rho_f.interpolate(fd.Constant(RHO_SURF) +
            (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) *
            (1.0 - fd.exp(-depth / 20.0)))
        w_f.assign(-A0 * float(params.rho_i) / RHO_SURF / YEAR_S)
        age_f.assign(0.0)

        for k in range(n_total):
            # Control-dependent forcing via CG1 interpolation.
            # T_offset makes these blocks control-dependent → replayed correctly.
            # Tk_consts[k] is a pre-created Constant (never reassigned).
            Ts_eff.interpolate(Tk_consts[k] + T_offset)
            Hs_eff.interpolate(c_i * (Ts_eff - T_ref))
            ws_eff.interpolate(-Ak_consts[k] * rho_i_over_rho_s_yr)

            dt_k = dt_spin_r if k < n_spin else dt_era5_r

            solver.prognostic_solve(
                enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_k,
                accumulation=Ak_consts[k],
                surface_density=rs,
                boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts_eff, enthalpy_bc_constant=None,
                age=age_f, age_boundary_condition=bc_age,
            )

        # ---- Misfit ----
        J = 0.0

        rho_pred_f.interpolate(rho_f)
        r = (rho_pred_f - rho_obs_f) / rho_sig_f
        J = J + 0.5/float(N_rho) * fd.assemble(r*r*fd.dx(domain=vom_rho))

        age_pred_f.interpolate(age_f)
        r = (age_pred_f - age_obs_f) / age_sig_f
        J = J + 0.5/float(N_age) * fd.assemble(r*r*fd.dx(domain=vom_age))

        T_pred_f.interpolate(H_f)
        r = (T_pred_f - T_obs_f) / T_sig_f
        J = J + 0.5/float(N_T) * fd.assemble(r*r*fd.dx(domain=vom_T))

        # Priors
        _n_inv = 1.0 / n_ctrl
        for c, nm in zip(ctrls, names):
            if nm in LOG:
                _prior.interpolate((c - math.log(float(INITIAL[nm]))) / PRIOR_SIG[nm])
            else:
                _prior.interpolate(
                    (c - fd.Constant(float(INITIAL[nm]))) / fd.Constant(PRIOR_SIG[nm])
                )
            J = J + _n_inv * 0.5 * fd.assemble(_prior*_prior*dx) / dlen
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
    print(f"  J_init={float(J_init):.4f}, J_replay={J0:.4f} (should match)")
    rng = np.random.default_rng(0)
    d = rng.standard_normal(n_ctrl) * 0.01
    gd = g @ d
    print(f"  <dJ,d>={gd:.6e}")
    prev = None
    for eps in [0.1, 0.05, 0.025, 0.0125, 0.00625]:
        m1 = np.clip(x0 + eps*d, lb, ub)
        try:
            with stop_annotating():
                for c, v in zip(ctrls, m1): c.assign(float(v))
                J1 = float(rf(ctrls))
            r = abs(J1 - J0 - eps*gd)
            o = f"{math.log(prev/r)/math.log(2):.2f}" if prev and r > 0 else ""
            print(f"  {eps:.4f}  {abs(J1-J0):.6e}  {r:.6e}  {o}")
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

    with open(OUT / "southpole_map_v20_hl.json", "w") as f:
        json.dump({"J_map": Jm, "J_hist": J_hist, "m_map": m_map, "INITIAL": INITIAL,
                   "BUIZERT_T_OFFSET": BUIZERT_T_OFFSET}, f, indent=2)
    print("Done.")
