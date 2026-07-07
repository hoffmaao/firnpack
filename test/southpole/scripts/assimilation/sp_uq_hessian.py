"""sp_uq_hessian.py — FD Hessian of the joint posterior at the MAP (Laplace UQ).

Same forward/objective machinery as sp_joint_assimilate.py (verbatim: data,
kernels, misfits, priors, stepper), evaluated at the MAP from results/sp_joint.json.
Central finite differences OF THE ADJOINT GRADIENT give the 12x12 Hessian of J
(the negative log posterior in internal coords: log-space for the 5 LOGS params,
linear for Q_base and the 6 T-knots).

Note the error model implied by the joint J: each dataset's chi^2 is divided by
its N (sigma inflated by sqrt(N_d)) and priors by N_CTRL — the Hessian is the
curvature of THAT posterior, i.e. consistent with the MAP but conservative.

Output: results/sp_uq_hessian.json  {H, H_sym, x_star, names, J0, eig, ...}
Env: FIRN_FD_H (default 1e-2), FIRN_SPIN_YEARS (default 2500).
Run: PYTHONPATH=src OMP_NUM_THREADS=1 /home/andrew/venv-firedrake-2026/bin/python sp_uq_hessian.py
"""
from __future__ import annotations
import functools, json, math, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import firedrake as fd
from firedrake.adjoint import (Control, continue_annotation, pause_annotation,
                               stop_annotating, get_working_tape)
from pyadjoint import compute_gradient
from firn.models.firn import FirnParameters, FirnModel
from firn.physics.densification import herron_langway as _hl
from firn.constants import year as YEAR_S
herron_langway = functools.partial(_hl, smooth=True)
sys.stdout.reconfigure(line_buffering=True)

PROC = Path(__file__).parent.parent.parent / "processed"
OUT = Path(__file__).parent.parent.parent / "results"
TAG = "sp_uq_hessian"
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, BDOT, DT_YEARS = 350.0, 0.085, 5.0
SPIN_YEARS = float(os.environ.get("FIRN_SPIN_YEARS", "2500"))
FD_H = float(os.environ.get("FIRN_FD_H", "1e-2"))
K_ICE_BASE, BETA, N_ICE = 2.1, 5.0e2, math.sqrt(3.18)

KNOT_YEARS = np.array([1000., 1300., 1550., 1750., 1900., 2015.])
N_KNOT = len(KNOT_YEARS)
INIT = dict(hl_k0=10.79, hl_k1=570.7, hl_Ea1=10432.0, hl_Ea2=21875.0, k_factor=0.5, Q_base=0.0)
T_KNOT_INIT = np.array([-45.50, -45.45, -45.35, -45.20, -45.10, -45.00])
LOGS = {"hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "k_factor"}
PRIOR_SIG = dict(hl_k0=1.0, hl_k1=1.0, hl_Ea1=0.5, hl_Ea2=0.5, k_factor=1.0, Q_base=0.05)
T_PRIOR_SIG = 0.6
SCAL_NAMES = ["hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "k_factor", "Q_base"]
NAMES = SCAL_NAMES + [f"Tk{i}" for i in range(N_KNOT)]
N_CTRL = len(NAMES)
W_RHO, W_AGE, W_V, W_T = 1.0, 1.0, 1.0, 1.0

# ---- MAP point (internal coords) ----
MAP = json.load(open(OUT / "sp_joint.json"))
mm = MAP["m_map"]
x_star = np.array([math.log(mm[n]) if n in LOGS else mm[n] for n in SCAL_NAMES]
                  + list(MAP["T_knots"]))
print("=" * 64)
print(f"FD Hessian at MAP (h={FD_H:g}, {2*N_CTRL+1} gradient evals)")
print(f"  J_MAP(saved) = {MAP['J']:.6f}")
print("=" * 64)
P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)
def mk(v, n): f = fd.Function(fd.FunctionSpace(mesh, "R", 0), name=n); f.assign(float(v)); return f

# ---- data (verbatim from sp_joint_assimilate.py) ----
def _sub(df, n): return df.iloc[::max(1, len(df)//n)]
dens = _sub(pd.read_csv(PROC/"sp19_density.csv").query("depth_m<=@H0"), 45)
obs_rho_d, obs_rho = dens.depth_m.values, dens.rho_kgm3.values*1000.0
sig_rho = 15.0 + 0.03*obs_rho
age = pd.read_csv(PROC/"sp19_depth_age.csv"); age["age_yr"] = 2015.0-age.year_CE
age = _sub(age.query("depth_m<=@H0 and age_yr>=0"), 40)
obs_age_d, obs_age_s = age.depth_m.values, age.age_yr.values*YEAR_S
sig_age_s = (10.0+0.03*age.age_yr.values)*YEAR_S
bT = _sub(pd.read_csv(PROC/"spicecore_borehole_T.csv").query("depth_m<=@H0"), 40)
obs_T_d, obs_T_H = bT.depth_m.values, c_i*(bT.T_C.values+273.15-T_ref)
sig_T_H = c_i*np.full(len(obs_T_d), 0.2)
ap = pd.read_csv(PROC/"apres_vertical_velocity_processed.csv")
ap = ap[(ap.range_m<=H0) & ap.v_smooth_m_yr.notna() & (ap.coherence>0.5)]
_vb = np.arange(8.0, H0, 8.0); _vm = 0.5*(_vb[:-1]+_vb[1:]); ap["_b"] = np.digitize(ap.range_m.values, _vb)
_vr = [(_vm[i], ap.v_smooth_m_yr[ap._b==i+1].median(), ap.v_smooth_m_yr[ap._b==i+1].std(),
        int((ap._b==i+1).sum())) for i in range(len(_vm))]
_vr = [r for r in _vr if r[3]>=3]
obs_v_d = np.array([r[0] for r in _vr]); obs_v = np.array([r[1] for r in _vr])
sig_v = np.maximum(np.array([r[2]/max(r[3],1)**0.5 for r in _vr]), 0.01)
N_rho, N_age, N_T, N_v = len(obs_rho_d), len(obs_age_d), len(obs_T_d), len(obs_v_d)
print(f"  obs: {N_rho} rho, {N_age} age, {N_T} T, {N_v} velocity")

# ---- mesh ----
mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0*(1.0-(1.0-fd.SpatialCoordinate(mesh)[0])**STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); R = fd.FunctionSpace(mesh, "R", 0)
dx = fd.dx(domain=mesh); ds = fd.ds(domain=mesh)
depth = H0-fd.SpatialCoordinate(mesh)[0]; psi = fd.TestFunction(V); xc = fd.SpatialCoordinate(mesh)[0]

# ---- kernels ----
def make_kernels(depths):
    ks = []
    with stop_annotating():
        for d in depths:
            s = float(np.clip(0.04*d+0.5, 0.5, 3.0))
            phi = fd.Function(V).interpolate(fd.exp(-0.5*((xc-(H0-d))/s)**2))
            phi.dat.data[:] /= float(fd.assemble(phi*dx))+1e-30
            ks.append(phi)
    return ks
with stop_annotating(): dlen = float(fd.assemble(fd.Constant(1.0)*dx))+1e-30
ker_rho, ker_age, ker_T, ker_v = (make_kernels(obs_rho_d), make_kernels(obs_age_d),
                                  make_kernels(obs_T_d), make_kernels(obs_v_d))
_prior = fd.Function(V)
def misfit(field, kernels, obs, sig, w, Nrm):
    J = 0.0; sq = 0.0
    for phi, o, s in zip(kernels, obs, sig):
        pred = fd.assemble(field*phi*dx); r = (pred-float(o))/float(s); J = J + r*r
        with stop_annotating(): sq += float(r)**2
    misfit._last = (sq/len(kernels))**0.5
    return w*0.5/Nrm*J

# ---- step timeline / bracket ----
n_steps = int(SPIN_YEARS/DT_YEARS)
step_years = 2015.0 - (n_steps-1-np.arange(n_steps))*DT_YEARS
bracket = []
for yr in step_years:
    if yr <= KNOT_YEARS[0]: bracket.append((0, None))
    elif yr >= KNOT_YEARS[-1]: bracket.append((N_KNOT-1, None))
    else:
        j = int(np.searchsorted(KNOT_YEARS, yr)-1)
        f = float((yr-KNOT_YEARS[j])/(KNOT_YEARS[j+1]-KNOT_YEARS[j])); bracket.append((j, f))

# ---- controls at MAP ----
log_k0 = mk(x_star[0], "lk0"); log_k1 = mk(x_star[1], "lk1")
log_Ea1 = mk(x_star[2], "lE1"); log_Ea2 = mk(x_star[3], "lE2")
log_kf = mk(x_star[4], "lkf"); Q_base = mk(x_star[5], "Qb")
Tknots = [mk(float(x_star[6+i]), f"Tk{i}") for i in range(N_KNOT)]
ctrl_fns = [log_k0, log_k1, log_Ea1, log_Ea2, log_kf, Q_base] + Tknots
PRIOR_CENTER = {n: (math.log(INIT[n]) if n in LOGS else INIT[n]) for n in SCAL_NAMES}

# ---- model + stepper (verbatim) ----
params = FirnParameters(
    hl_k0_prefactor=fd.exp(log_k0), hl_k1_prefactor=fd.exp(log_k1),
    hl_Ea_stage1=fd.exp(log_Ea1), hl_Ea_stage2=fd.exp(log_Ea2),
    k_ice=K_ICE_BASE*fd.exp(log_kf), basal_heat_flux_W_m2=Q_base)
model = FirnModel(params, densification_rate_fn=herron_langway)
ws0 = -BDOT*rho_i/RHO_SURF/YEAR_S
H_f=fd.Function(V,name="H"); rho_f=fd.Function(V,name="rho"); w_f=fd.Function(V,name="w"); age_f=fd.Function(V,name="age")
H_o=fd.Function(V); rho_o=fd.Function(V); w_o=fd.Function(V); age_o=fd.Function(V); Ts_eff=fd.Function(V)
ac=mk(BDOT,"ac"); rs=mk(RHO_SURF,"rs"); ws=mk(ws0,"ws"); dt_r=mk(DT_YEARS*YEAR_S,"dt")
bc_rho=fd.DirichletBC(V,rs,SID); bc_w=fd.DirichletBC(V,ws,SID); bc_age=fd.DirichletBC(V,mk(0.0,"a0"),SID)
bdot_mass=ac*rho_i/P0.spy
def Ts_expr_for(k):
    j, f = bracket[k]
    if f is None: return Tknots[j] + fd.Constant(273.15)
    return fd.Constant(1.0-f)*Tknots[j] + fd.Constant(f)*Tknots[j+1] + fd.Constant(273.15)
sp_lin={"ksp_type":"preonly","pc_type":"lu"}
Ht=fd.TrialFunction(V); rt=fd.TrialFunction(V); wt=fd.TrialFunction(V); at=fd.TrialFunction(V)
F_H=model.enthalpy_form(Ht,H_o,rho_f,w_f,psi,dt_r)+fd.Constant(BETA)*(Ht-c_i*(Ts_eff-fd.Constant(T_ref)))*psi*ds(SID)
solv_H=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(F_H),fd.rhs(F_H),H_f,constant_jacobian=False),solver_parameters=sp_lin)
T_expr=model.temperature_from_enthalpy(H_f)
F_rho,_=model.density_form(rt,rho_o,T_expr,w_f,None,bdot_mass,psi,dt_r)
solv_rho=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(F_rho),fd.rhs(F_rho),rho_f,bcs=[bc_rho],constant_jacobian=False),solver_parameters=sp_lin)
drhodt_expr=herron_langway(rho_f,T_expr,params=params,bdot=bdot_mass)
delta_w=model.velocity_delta(wt,w_o,rho_f,drhodt_expr,psi,regularization=1e-3)
solv_w=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(delta_w),fd.rhs(delta_w),w_f,bcs=[bc_w],constant_jacobian=False),solver_parameters=sp_lin)
F_age=model.age_form(at,age_o,w_f,w_o,psi,dt_r)
solv_age=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(F_age),fd.rhs(F_age),age_f,bcs=[bc_age],constant_jacobian=False),solver_parameters=sp_lin)

def forward():
    T0=float(T_KNOT_INIT.mean())+273.15
    H_f.assign(c_i*(T0-T_ref)); H_o.assign(H_f)
    rho_f.interpolate(RHO_SURF+(820.0-RHO_SURF)*(1.0-fd.exp(-depth/25.0))); rho_o.assign(rho_f)
    w_f.assign(ws0); w_o.assign(w_f); age_f.assign(0.0); age_o.assign(0.0)
    for k in range(n_steps):
        Ts_eff.interpolate(Ts_expr_for(k))
        solv_H.solve(); solv_rho.solve(); solv_w.solve(); solv_age.solve()
        H_o.assign(H_f); rho_o.assign(rho_f); w_o.assign(w_f); age_o.assign(age_f)
    J_rho=misfit(rho_f,ker_rho,obs_rho,sig_rho,W_RHO,N_rho)
    J_age=misfit(age_f,ker_age,obs_age_s,sig_age_s,W_AGE,N_age)
    J_T=misfit(H_f,ker_T,obs_T_H,sig_T_H,W_T,N_T)
    Jv=0.0
    for phi,o,s in zip(ker_v,obs_v,sig_v):
        pred=(fd.assemble(w_f*phi*dx)-ws0)/N_ICE*YEAR_S; r=(pred-float(o))/float(s); Jv=Jv+r*r
    J_v=W_V*0.5/N_v*Jv
    J=J_rho+J_age+J_T+J_v
    Jp=0.0
    for c,nm in zip(ctrl_fns[:6], SCAL_NAMES):
        _prior.interpolate((c-fd.Constant(PRIOR_CENTER[nm]))/fd.Constant(PRIOR_SIG[nm]))
        Jp=Jp+0.5/N_CTRL*fd.assemble(_prior*_prior*dx)/dlen
    for i,tk in enumerate(Tknots):
        _prior.interpolate((tk-fd.Constant(float(T_KNOT_INIT[i])))/fd.Constant(T_PRIOR_SIG))
        Jp=Jp+0.5/N_CTRL*fd.assemble(_prior*_prior*dx)/dlen
    return J+Jp

tape=get_working_tape()
def grad_at(x):
    t0=time.perf_counter(); tape.clear_tape(); continue_annotation()
    for c,v in zip(ctrl_fns,x): c.assign(float(v))
    J=forward(); dJ=compute_gradient(J,[Control(c) for c in ctrl_fns])
    pause_annotation()
    g=np.array([float(gi.dat.data_ro[0])*dlen for gi in dJ])
    return float(J), g, time.perf_counter()-t0

J0, g0, dt0 = grad_at(x_star)
print(f"\nJ(x*) = {J0:.6f} (saved MAP {MAP['J']:.6f}, diff {abs(J0-MAP['J']):.2e}) [{dt0:.0f}s]")
print(f"|g(x*)| = {np.linalg.norm(g0):.3e} (should be small at MAP)")

H = np.zeros((N_CTRL, N_CTRL))
for j in range(N_CTRL):
    xp = x_star.copy(); xp[j] += FD_H
    xm = x_star.copy(); xm[j] -= FD_H
    _, gp, tp = grad_at(xp)
    _, gm, tm = grad_at(xm)
    H[:, j] = (gp - gm) / (2*FD_H)
    print(f"  [{j+1:2d}/{N_CTRL}] {NAMES[j]:9s} d|g|={np.linalg.norm(gp-gm):.3e} "
          f"H_jj={H[j,j]:+.3e} ({tp+tm:.0f}s)")

H_sym = 0.5*(H+H.T)
asym = np.linalg.norm(H-H.T)/max(np.linalg.norm(H_sym), 1e-30)
eig = np.linalg.eigvalsh(H_sym)
print(f"\nAsymmetry |H-H^T|/|H_sym| = {asym:.2e}")
print("Eigenvalues:", " ".join(f"{e:.3e}" for e in eig))
json.dump(dict(names=NAMES, x_star=x_star.tolist(), fd_h=FD_H, J0=J0,
               J_map_saved=MAP["J"], g0=g0.tolist(), H=H.tolist(),
               H_sym=H_sym.tolist(), eig=eig.tolist(), asym=asym,
               logs=sorted(LOGS)),
          open(OUT/f"{TAG}.json", "w"), indent=1)
print(f"Saved {OUT/(TAG+'.json')}")
print("Done.")
