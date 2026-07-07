"""sp_joint_assimilate_ezz.py — joint SP inversion + ice-dynamic vertical strain.

Identical to sp_joint_assimilate.py (12 controls: H&L densification {k0,k1,Ea1,Ea2}
+ conductivity k_factor + 6 surface-T knots + Q_base) PLUS a 13th control:

  ezz_yr — dynamic (ice-flow) vertical strain rate [1/yr], negative = thinning.
           Enters the velocity closure via the model's horizontal_divergence
           term (div_h = -ezz / YEAR_S):  dw/dz = -(1/rho) drho/dt - div_h.
           Density/age/enthalpy forms are UNCHANGED (densification is the
           material rate; dynamic strain thins WITHOUT densifying — that
           density-free signature is what separates it from compaction).

Motivation (2026-07-01 structural audit): deep (95-128 m) ApRES strain exceeds
compaction-only model strain by ~5e-5/yr — see HANDOFF.md.

Warm start: densif/kf/Q/T-knots at the sp_joint.json MAP, ezz at 0. Priors are
UNCHANGED from sp_joint (centers at the original INIT), ezz prior N(0, 1e-4/yr).
With ezz=0 the start state replays the old MAP forward exactly (J differs only
by the 0.5/N_CTRL prior renormalization, 12->13).

Env: FIRN_VERIFY_ONLY=1, FIRN_FD_CHECK=1, FIRN_FD_NAMES=ezz_yr (restrict FD),
FIRN_MAX_ITER, FIRN_SPIN_YEARS.
"""
from __future__ import annotations
import functools, json, math, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize as sp_minimize
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
TAG = "sp_joint_ezz"
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, BDOT, DT_YEARS = 350.0, 0.085, 5.0
SPIN_YEARS = float(os.environ.get("FIRN_SPIN_YEARS", "2500"))
MAX_ITER = int(os.environ.get("FIRN_MAX_ITER", "60"))
VERIFY_ONLY = os.environ.get("FIRN_VERIFY_ONLY", "0") == "1"
FD_CHECK = os.environ.get("FIRN_FD_CHECK", "0") == "1"
FD_NAMES = [s for s in os.environ.get("FIRN_FD_NAMES", "").split(",") if s]
K_ICE_BASE, BETA, N_ICE = 2.1, 5.0e2, math.sqrt(3.18)

# ---- controls: priors as in sp_joint (INIT = prior centers) + ezz ----
KNOT_YEARS = np.array([1000., 1300., 1550., 1750., 1900., 2015.])
N_KNOT = len(KNOT_YEARS)
INIT = dict(hl_k0=10.79, hl_k1=570.7, hl_Ea1=10432.0, hl_Ea2=21875.0,
            k_factor=0.5, Q_base=0.0, ezz_yr=0.0)
T_KNOT_INIT = np.array([-45.50, -45.45, -45.35, -45.20, -45.10, -45.00])
LOGS = {"hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "k_factor"}
PRIOR_SIG = dict(hl_k0=1.0, hl_k1=1.0, hl_Ea1=0.5, hl_Ea2=0.5, k_factor=1.0,
                 Q_base=0.05, ezz_yr=1.0e-4)
T_PRIOR_SIG = 0.6
SCAL_NAMES = ["hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "k_factor", "Q_base", "ezz_yr"]
N_SCAL = len(SCAL_NAMES)
NAMES = SCAL_NAMES + [f"Tk{i}" for i in range(N_KNOT)]
N_CTRL = len(NAMES)
W_RHO, W_AGE, W_V, W_T = 1.0, 1.0, 1.0, 1.0

# warm start at the 12-control MAP (fall back to INIT if missing)
try:
    _prev = json.load(open(OUT / "sp_joint.json"))
    START = dict(_prev["m_map"]); START["ezz_yr"] = 0.0
    T_KNOT_START = np.array(_prev["T_knots"])
    print(f"warm start from sp_joint.json (J={_prev['J']:.4f})")
except FileNotFoundError:
    START = dict(INIT); T_KNOT_START = T_KNOT_INIT.copy()

print("="*64)
print(f"JOINT+ezz inversion: densif + conductivity + ezz + {N_KNOT} T-knots + Q ({N_CTRL} ctrls)")
print(f"  spinup={SPIN_YEARS:.0f}yr dt={DT_YEARS} = {int(SPIN_YEARS/DT_YEARS)} steps")
print("="*64)
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

# ---- controls (warm start) ----
log_k0 = mk(math.log(START["hl_k0"]), "lk0"); log_k1 = mk(math.log(START["hl_k1"]), "lk1")
log_Ea1 = mk(math.log(START["hl_Ea1"]), "lE1"); log_Ea2 = mk(math.log(START["hl_Ea2"]), "lE2")
log_kf = mk(math.log(START["k_factor"]), "lkf"); Q_base = mk(START["Q_base"], "Qb")
ezz_yr = mk(START["ezz_yr"], "ezz")
Tknots = [mk(float(T_KNOT_START[i]), f"Tk{i}") for i in range(N_KNOT)]
ctrl_fns = [log_k0, log_k1, log_Ea1, log_Ea2, log_kf, Q_base, ezz_yr] + Tknots
x0 = np.array([float(c.dat.data_ro[0]) for c in ctrl_fns])
def _b(nm, lo, hi): return (math.log(lo), math.log(hi)) if nm in LOGS else (lo, hi)
BND = dict(hl_k0=_b("hl_k0",0.5,500), hl_k1=_b("hl_k1",10,50000), hl_Ea1=_b("hl_Ea1",3000,40000),
           hl_Ea2=_b("hl_Ea2",5000,80000), k_factor=_b("k_factor",0.05,3.0), Q_base=(-0.2,0.2),
           ezz_yr=(-3.0e-4, 2.0e-4))
lb = np.array([BND[n][0] for n in SCAL_NAMES]+[-48.0]*N_KNOT)
ub = np.array([BND[n][1] for n in SCAL_NAMES]+[-43.0]*N_KNOT)
PRIOR_CENTER = {n: (math.log(INIT[n]) if n in LOGS else INIT[n]) for n in SCAL_NAMES}

# ---- model + stepper (velocity gains the dynamic-strain term) ----
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
# div_h = -(ezz)/YEAR_S : dynamic thinning (ezz<0) => positive horizontal divergence
div_h_expr = fd.Constant(-1.0/YEAR_S)*ezz_yr
delta_w=model.velocity_delta(wt,w_o,rho_f,drhodt_expr,psi,regularization=1e-3,
                             horizontal_divergence=div_h_expr)
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
    J_rho=misfit(rho_f,ker_rho,obs_rho,sig_rho,W_RHO,N_rho); rms_rho=misfit._last
    J_age=misfit(age_f,ker_age,obs_age_s,sig_age_s,W_AGE,N_age); rms_age=misfit._last
    J_T=misfit(H_f,ker_T,obs_T_H,sig_T_H,W_T,N_T); rms_T=misfit._last
    Jv=0.0; sqv=0.0
    for phi,o,s in zip(ker_v,obs_v,sig_v):
        pred=(fd.assemble(w_f*phi*dx)-ws0)/N_ICE*YEAR_S; r=(pred-float(o))/float(s); Jv=Jv+r*r
        with stop_annotating(): sqv+=float(r)**2
    J_v=W_V*0.5/N_v*Jv; rms_v=(sqv/N_v)**0.5
    J=J_rho+J_age+J_T+J_v
    Jp=0.0
    for c,nm in zip(ctrl_fns[:N_SCAL], SCAL_NAMES):
        _prior.interpolate((c-fd.Constant(PRIOR_CENTER[nm]))/fd.Constant(PRIOR_SIG[nm]))
        Jp=Jp+0.5/N_CTRL*fd.assemble(_prior*_prior*dx)/dlen
    for i,tk in enumerate(Tknots):
        _prior.interpolate((tk-fd.Constant(float(T_KNOT_INIT[i])))/fd.Constant(T_PRIOR_SIG))
        Jp=Jp+0.5/N_CTRL*fd.assemble(_prior*_prior*dx)/dlen
    J=J+Jp
    with stop_annotating():
        forward._diag=dict(J_rho=float(J_rho),J_age=float(J_age),J_T=float(J_T),J_v=float(J_v),Jp=float(Jp),
            rms_rho=rms_rho,rms_age=rms_age,rms_T=rms_T,rms_v=rms_v,
            rho_max=float(rho_f.dat.data_ro.max()),age_b=float(age_f.dat.data_ro.max())/YEAR_S,
            Tmin=float(H_f.dat.data_ro.min())/c_i+T_ref-273.15,Tmax=float(H_f.dat.data_ro.max())/c_i+T_ref-273.15)
    return J

tape=get_working_tape(); n_eval=[0]; J_hist=[]; scale=np.ones(N_CTRL)
def eval_J_and_grad(xs):
    x=np.clip(x0+xs*scale, lb, ub); t0=time.perf_counter(); tape.clear_tape(); continue_annotation()
    for c,v in zip(ctrl_fns,x): c.assign(float(v))
    try:
        J=forward(); dJ=compute_gradient(J,[Control(c) for c in ctrl_fns])
        Jv=float(J); g=np.array([float(gi.dat.data_ro[0])*dlen for gi in dJ])*scale
    except Exception as e:
        print(f"  *** {e}"); Jv=1e8; g=xs*100.0
    pause_annotation(); n_eval[0]+=1; J_hist.append(Jv); d=forward._diag
    print(f"  [{n_eval[0]:03d}] J={Jv:.4f} (r={d['rms_rho']:.2f} a={d['rms_age']:.2f} T={d['rms_T']:.2f} v={d['rms_v']:.2f}) "
          f"ezz={x[6]:+.2e} kf={math.exp(x[4]):.3f} rho_mx={d['rho_max']:.0f} |g|={np.linalg.norm(g):.1e} ({time.perf_counter()-t0:.0f}s)")
    return Jv, g

print("\nReproducibility (ezz=0 should replay the sp_joint MAP forward):")
def _run():
    tape.clear_tape(); continue_annotation()
    for c,v in zip(ctrl_fns,x0): c.assign(float(v))
    J=float(forward()); pause_annotation(); return J
J1=_run(); J2=_run(); d=forward._diag
print(f"  J1={J1:.8f} J2={J2:.8f} match={abs(J1-J2)<1e-10}")
print(f"  rms: rho={d['rms_rho']:.2f} age={d['rms_age']:.2f} T={d['rms_T']:.2f} v={d['rms_v']:.2f}")
print(f"  rho_max={d['rho_max']:.0f} age_b={d['age_b']:.0f} T=[{d['Tmin']:.2f},{d['Tmax']:.2f}]")

print("\nGradient:")
tape.clear_tape(); continue_annotation()
for c,v in zip(ctrl_fns,x0): c.assign(float(v))
J0f=forward(); dJ0=compute_gradient(J0f,[Control(c) for c in ctrl_fns]); pause_annotation()
J0=float(J0f); g=np.array([float(gi.dat.data_ro[0])*dlen for gi in dJ0])
print("  " + " ".join(f"{n}={g[i]:.1e}" for i,n in enumerate(NAMES)))
if FD_CHECK:
    def Jat(xv):
        tape.clear_tape(); continue_annotation()
        for c,v in zip(ctrl_fns,xv): c.assign(float(v))
        Jv=float(forward()); pause_annotation(); return Jv
    print("  FD check:")
    for i,nm in enumerate(NAMES):
        if FD_NAMES and nm not in FD_NAMES: continue
        h = 1e-3 if nm != "ezz_yr" else 2e-6   # ezz lives at 1e-5..1e-4 scale
        ep=x0.copy(); ep[i]+=h; em=x0.copy(); em[i]-=h; fdg=(Jat(ep)-Jat(em))/(2*h)
        print(f"    {nm:9s} adj={g[i]:+.3e} FD={fdg:+.3e} ratio={g[i]/fdg if abs(fdg)>1e-12 else float('nan'):+.4f}")
if VERIFY_ONLY:
    print("\nVERIFY_ONLY — stop."); sys.exit(0)

# Prior-sigma scaling. The parent script scaled by 1/|g|, which is fine from a
# cold start but explodes at a warm start (the 12 old controls sit at ~zero
# gradient -> scales 1e3-1e4 -> first L-BFGS-B step slams everything onto the
# box bounds and the solves diverge). Prior sigmas make the implicit initial
# metric ~ the prior.
scale=np.array([PRIOR_SIG[n] for n in SCAL_NAMES]+[T_PRIOR_SIG]*N_KNOT)
lb_s=(lb-x0)/scale; ub_s=(ub-x0)/scale
print(f"\nOptimize (L-BFGS-B, max {MAX_ITER}):")
n_eval[0]=0; J_hist.clear()
res=sp_minimize(eval_J_and_grad, np.zeros(N_CTRL), jac=True, method="L-BFGS-B",
                bounds=list(zip(lb_s,ub_s)), options={"maxiter":MAX_ITER,"ftol":1e-8,"gtol":1e-7,"maxls":30})
x_map=x0+res.x*scale
m={n:(math.exp(x_map[i]) if n in LOGS else x_map[i]) for i,n in enumerate(NAMES)}
print(f"\n{res.message}  J={res.fun:.4f}")
for n in SCAL_NAMES: print(f"  {n:10s} = {m[n]:.4e}")
for i,yr in enumerate(KNOT_YEARS): print(f"  T({yr:.0f}) = {x_map[N_SCAL+i]:.3f}C")
json.dump(dict(J=res.fun, m_map=m, T_knots=x_map[N_SCAL:].tolist(), knot_years=KNOT_YEARS.tolist(),
               J_hist=J_hist, INIT=INIT, START=START), open(OUT/f"{TAG}.json","w"), indent=2)
print(f"Saved {OUT/(TAG+'.json')}")
print("Done.")
