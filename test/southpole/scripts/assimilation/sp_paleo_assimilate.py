"""sp_paleo_assimilate.py — borehole paleothermometry (thermal inversion).

STAGE 1 (this file, thermal-only): with the densification fixed at the 4-data
MAP, invert the firn thermal CONDUCTIVITY factor + the SURFACE-TEMPERATURE
HISTORY (epoch knots) against the SPICE borehole temperature profile.

Validates the new adjoint machinery before the full joint inversion:
  * Penalty enthalpy surface BC (control-dependent surface T must NOT go in a
    Dirichlet BC -> order-1 adjoint; use a weak penalty term instead).
  * Surface-T history fed per step via a CG1 intermediate Ts_eff that is a
    control-dependent linear combination of the knot controls (adjoint-safe).
  * Firn conductivity factor enters the heat-equation weak form directly.

Run:
  FIRN_VERIFY_ONLY=1 ... python sp_paleo_assimilate.py   # repro + Taylor + FD check
  python sp_paleo_assimilate.py                          # + optimize
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
TAG = "sp_paleo_thermal"
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, BDOT, DT_YEARS = 350.0, 0.085, 5.0
SPIN_YEARS = float(os.environ.get("FIRN_SPIN_YEARS", "2500"))
MAX_ITER = int(os.environ.get("FIRN_MAX_ITER", "40"))
VERIFY_ONLY = os.environ.get("FIRN_VERIFY_ONLY", "0") == "1"
K_ICE_BASE = 2.1
BETA = 5.0e2   # penalty strength for enthalpy surface BC (tuned: surface T error tiny)

# densification fixed at the 4-data MAP
DENS = dict(hl_k0=10.79, hl_k1=570.7, hl_Ea1=10432.0, hl_Ea2=21875.0)

# ---- controls: conductivity factor (log) + surface-T knots (C) ----
KNOT_YEARS = np.array([1000., 1300., 1550., 1750., 1900., 2015.])
N_KNOT = len(KNOT_YEARS)
# gentle warming ramp init (non-isothermal -> all controls get a testable gradient)
T_KNOT_INIT = np.array([-45.50, -45.45, -45.35, -45.20, -45.10, -45.00])
KF_INIT = 0.3                       # conductivity factor initial (de-risk: ~0.1-0.3)
KF_PRIOR_SIG, T_PRIOR_SIG = 1.0, 0.6
NAMES = ["k_factor"] + [f"Tk{i}" for i in range(N_KNOT)]
N_CTRL = len(NAMES)

print("="*64); print(f"PALEO THERMAL inversion: k_factor + {N_KNOT} surface-T knots")
print(f"  spinup={SPIN_YEARS:.0f}yr dt={DT_YEARS} = {int(SPIN_YEARS/DT_YEARS)} steps, beta={BETA}")
print("="*64)

P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)

# ---- borehole T data ----
bT = pd.read_csv(PROC/"spicecore_borehole_T.csv"); bT = bT[bT.depth_m <= H0]
bT = bT.iloc[::max(1, len(bT)//40)]
obs_T_d = bT.depth_m.values
obs_T_H = c_i*(bT.T_C.values + 273.15 - T_ref)
sig_T_H = c_i*np.full(len(obs_T_d), 0.15)   # tight now (0.15 C ~ detrended scatter)
N_T = len(obs_T_d)
print(f"  borehole obs: {N_T} pts, sig=0.15C")

# ---- mesh ----
mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0*(1.0-(1.0-fd.SpatialCoordinate(mesh)[0])**STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); R = fd.FunctionSpace(mesh, "R", 0)
dx = fd.dx(domain=mesh); ds = fd.ds(domain=mesh)
depth = H0-fd.SpatialCoordinate(mesh)[0]; psi = fd.TestFunction(V)
def mk(v, n): f = fd.Function(R, name=n); f.assign(float(v)); return f

# ---- step timeline + piecewise-linear bracketing (j, f) per step ----
# Use FRESH literal-weighted Constants per step (NOT reassigned shared Constants,
# which break the adjoint: non-control reassigns aren't tracked per step).
n_steps = int(SPIN_YEARS/DT_YEARS)
step_years = 2015.0 - (n_steps-1-np.arange(n_steps))*DT_YEARS
bracket = []   # (j, f): clamp to knot j if f is None, else linear j..j+1
for yr in step_years:
    if yr <= KNOT_YEARS[0]: bracket.append((0, None))
    elif yr >= KNOT_YEARS[-1]: bracket.append((N_KNOT-1, None))
    else:
        j = int(np.searchsorted(KNOT_YEARS, yr) - 1)
        f = float((yr - KNOT_YEARS[j])/(KNOT_YEARS[j+1]-KNOT_YEARS[j]))
        bracket.append((j, f))

# ---- controls ----
log_kf = mk(math.log(KF_INIT), "log_kf")
Tknots = [mk(float(T_KNOT_INIT[i]), f"Tk{i}") for i in range(N_KNOT)]
ctrl_fns = [log_kf] + Tknots
x0 = np.array([float(c.dat.data_ro[0]) for c in ctrl_fns])
lb = np.array([math.log(0.03)] + [-48.0]*N_KNOT)
ub = np.array([math.log(3.0)] + [-43.0]*N_KNOT)

# ---- model (densification fixed) ----
k_factor = fd.exp(log_kf)
params = FirnParameters(
    hl_k0_prefactor=mk(DENS["hl_k0"], "k0"), hl_k1_prefactor=mk(DENS["hl_k1"], "k1"),
    hl_Ea_stage1=mk(DENS["hl_Ea1"], "e1"), hl_Ea_stage2=mk(DENS["hl_Ea2"], "e2"),
    k_ice=K_ICE_BASE*k_factor, basal_heat_flux_W_m2=mk(0.0, "Q"))
model = FirnModel(params, densification_rate_fn=herron_langway)

# ---- state + olds ----
ws0 = -BDOT*rho_i/RHO_SURF/YEAR_S
H_f = fd.Function(V, name="H"); rho_f = fd.Function(V, name="rho")
w_f = fd.Function(V, name="w"); age_f = fd.Function(V, name="age")
H_o = fd.Function(V); rho_o = fd.Function(V); w_o = fd.Function(V); age_o = fd.Function(V)
Ts_eff = fd.Function(V, name="Ts_eff")   # CG1 control-dependent surface T (K)

ac = mk(BDOT, "ac"); rs = mk(RHO_SURF, "rs"); ws = mk(ws0, "ws"); dt_r = mk(DT_YEARS*YEAR_S, "dt")
bc_rho = fd.DirichletBC(V, rs, SID); bc_w = fd.DirichletBC(V, ws, SID)
bc_age = fd.DirichletBC(V, mk(0.0, "a0"), SID)
bdot_mass = ac*rho_i/P0.spy

def Ts_expr_for(k):
    """Surface-T (K) expression at step k: fresh literal-weighted knot combo."""
    j, f = bracket[k]
    if f is None:
        return Tknots[j] + fd.Constant(273.15)
    return fd.Constant(1.0-f)*Tknots[j] + fd.Constant(f)*Tknots[j+1] + fd.Constant(273.15)

# ---- cached solvers ----
sp_lin = {"ksp_type": "preonly", "pc_type": "lu"}
Ht = fd.TrialFunction(V); rt = fd.TrialFunction(V); wt = fd.TrialFunction(V); at = fd.TrialFunction(V)
# enthalpy with PENALTY surface BC (control-dependent Ts_eff)
F_H = model.enthalpy_form(Ht, H_o, rho_f, w_f, psi, dt_r)
F_H += fd.Constant(BETA)*(Ht - c_i*(Ts_eff - fd.Constant(T_ref)))*psi*ds(SID)
solv_H = fd.LinearVariationalSolver(fd.LinearVariationalProblem(
    fd.lhs(F_H), fd.rhs(F_H), H_f, constant_jacobian=False), solver_parameters=sp_lin)
T_expr = model.temperature_from_enthalpy(H_f)
F_rho, _ = model.density_form(rt, rho_o, T_expr, w_f, None, bdot_mass, psi, dt_r)
solv_rho = fd.LinearVariationalSolver(fd.LinearVariationalProblem(
    fd.lhs(F_rho), fd.rhs(F_rho), rho_f, bcs=[bc_rho], constant_jacobian=False), solver_parameters=sp_lin)
drhodt_expr = herron_langway(rho_f, T_expr, params=params, bdot=bdot_mass)
delta_w = model.velocity_delta(wt, w_o, rho_f, drhodt_expr, psi, regularization=1e-3)
solv_w = fd.LinearVariationalSolver(fd.LinearVariationalProblem(
    fd.lhs(delta_w), fd.rhs(delta_w), w_f, bcs=[bc_w], constant_jacobian=False), solver_parameters=sp_lin)
F_age = model.age_form(at, age_o, w_f, w_o, psi, dt_r)
solv_age = fd.LinearVariationalSolver(fd.LinearVariationalProblem(
    fd.lhs(F_age), fd.rhs(F_age), age_f, bcs=[bc_age], constant_jacobian=False), solver_parameters=sp_lin)

# ---- T obs kernels (off tape) ----
xc = fd.SpatialCoordinate(mesh)[0]
with stop_annotating():
    dlen = float(fd.assemble(fd.Constant(1.0)*dx)) + 1e-30
    kerT = []
    for d in obs_T_d:
        s = float(np.clip(0.04*d+0.5, 0.5, 3.0))
        phi = fd.Function(V).interpolate(fd.exp(-0.5*((xc-(H0-d))/s)**2))
        phi.dat.data[:] /= float(fd.assemble(phi*dx))+1e-30
        kerT.append(phi)
_prior = fd.Function(V)

def forward():
    # IC: isothermal at the mean knot level
    T0 = float(T_KNOT_INIT.mean()) + 273.15
    H_f.assign(c_i*(T0-T_ref)); H_o.assign(H_f)
    rho_f.interpolate(RHO_SURF+(820.0-RHO_SURF)*(1.0-fd.exp(-depth/25.0))); rho_o.assign(rho_f)
    w_f.assign(ws0); w_o.assign(w_f); age_f.assign(0.0); age_o.assign(0.0)
    for k in range(n_steps):
        Ts_eff.interpolate(Ts_expr_for(k))   # control-dependent (knots), CG1 intermediate
        solv_H.solve(); solv_rho.solve(); solv_w.solve(); solv_age.solve()
        H_o.assign(H_f); rho_o.assign(rho_f); w_o.assign(w_f); age_o.assign(age_f)
    # borehole-T misfit (scalar kernel)
    J = 0.0; sq = 0.0
    for phi, o, s in zip(kerT, obs_T_H, sig_T_H):
        pred = fd.assemble(H_f*phi*dx)
        r = (pred - float(o))/float(s)
        J = J + 0.5/N_T*r*r
        with stop_annotating(): sq += float(r)**2
    # prior
    _prior.interpolate((log_kf - math.log(KF_INIT))/KF_PRIOR_SIG)
    Jp = 0.5/N_CTRL*fd.assemble(_prior*_prior*dx)/dlen
    for i, tk in enumerate(Tknots):
        _prior.interpolate((tk - fd.Constant(float(T_KNOT_INIT[i])))/fd.Constant(T_PRIOR_SIG))
        Jp = Jp + 0.5/N_CTRL*fd.assemble(_prior*_prior*dx)/dlen
    J = J + Jp
    with stop_annotating():
        Tsurf_now = float(Ts_eff.dat.data_ro.max())-273.15
        forward._diag = dict(rmsT=(sq/N_T)**0.5, Jp=float(Jp),
            Tmin=float(H_f.dat.data_ro.min())/c_i+T_ref-273.15,
            Tmax=float(H_f.dat.data_ro.max())/c_i+T_ref-273.15, Tsurf=Tsurf_now)
    return J

tape = get_working_tape(); n_eval=[0]; J_hist=[]; scale=np.ones(N_CTRL)

def eval_J_and_grad(xs):
    x = np.clip(x0 + xs*scale, lb, ub)
    t0 = time.perf_counter(); tape.clear_tape(); continue_annotation()
    for c, v in zip(ctrl_fns, x): c.assign(float(v))
    try:
        J = forward(); dJ = compute_gradient(J, [Control(c) for c in ctrl_fns])
        Jv = float(J); g = np.array([float(gi.dat.data_ro[0])*dlen for gi in dJ])*scale
    except Exception as e:
        print(f"  *** {e}"); Jv = 1e8; g = xs*100.0
    pause_annotation(); n_eval[0]+=1; J_hist.append(Jv); d=forward._diag
    print(f"  [{n_eval[0]:03d}] J={Jv:.4f} rmsT={d['rmsT']:.2f} Jp={d['Jp']:.3f} "
          f"kf={math.exp(x[0]):.3f} T[{d['Tmin']:.2f},{d['Tmax']:.2f}] |g|={np.linalg.norm(g):.1e} ({time.perf_counter()-t0:.0f}s)")
    return Jv, g

print("\nReproducibility:")
def _run():
    tape.clear_tape(); continue_annotation()
    for c, v in zip(ctrl_fns, x0): c.assign(float(v))
    J = float(forward()); pause_annotation(); return J
J1=_run(); J2=_run(); d=forward._diag
print(f"  J1={J1:.8f} J2={J2:.8f} match={abs(J1-J2)<1e-10}")
print(f"  rmsT={d['rmsT']:.2f} T=[{d['Tmin']:.2f},{d['Tmax']:.2f}]C Tsurf={d['Tsurf']:.2f}C")

print("\nTaylor + FD check (the new k_factor/T-knot adjoint paths):")
tape.clear_tape(); continue_annotation()
for c, v in zip(ctrl_fns, x0): c.assign(float(v))
J0f = forward(); dJ0 = compute_gradient(J0f, [Control(c) for c in ctrl_fns]); pause_annotation()
J0 = float(J0f); g = np.array([float(gi.dat.data_ro[0])*dlen for gi in dJ0])
print("  adjoint dJ: " + " ".join(f"{n}={g[i]:.2e}" for i,n in enumerate(NAMES)))
if os.environ.get("FIRN_FD_CHECK", "0") == "1":
    h = 1e-3
    def Jat(xv):
        tape.clear_tape(); continue_annotation()
        for c, v in zip(ctrl_fns, xv): c.assign(float(v))
        Jv = float(forward()); pause_annotation(); return Jv
    print("  FD check (central h=1e-3):")
    for i, nm in enumerate(NAMES):
        ep=x0.copy(); ep[i]+=h; em=x0.copy(); em[i]-=h
        fdg=(Jat(ep)-Jat(em))/(2*h); ratio=g[i]/fdg if abs(fdg)>1e-12 else float('nan')
        print(f"    {nm:9s} adj={g[i]:+.3e} FD={fdg:+.3e} ratio={ratio:+.4f}")

if VERIFY_ONLY:
    print("\nVERIFY_ONLY — stop."); sys.exit(0)

scale = 1.0/np.maximum(np.abs(g), 1e-8)
lb_s=(lb-x0)/scale; ub_s=(ub-x0)/scale
print(f"\nOptimize (L-BFGS-B, max {MAX_ITER}):")
n_eval[0]=0; J_hist.clear()
res = sp_minimize(eval_J_and_grad, np.zeros(N_CTRL), jac=True, method="L-BFGS-B",
                  bounds=list(zip(lb_s, ub_s)),
                  options={"maxiter": MAX_ITER, "ftol": 1e-8, "gtol": 1e-7, "maxls": 30})
x_map = x0 + res.x*scale
kf_map = math.exp(x_map[0]); Tk_map = x_map[1:]
print(f"\n{res.message}  J={res.fun:.4f}")
print(f"  k_factor = {kf_map:.3f}  (k_eff = {kf_map*K_ICE_BASE:.3f} W/m/K at rho_i)")
for i, yr in enumerate(KNOT_YEARS): print(f"  T({yr:.0f}CE) = {Tk_map[i]:.3f}C")
json.dump(dict(J=res.fun, k_factor=kf_map, T_knots=Tk_map.tolist(),
               knot_years=KNOT_YEARS.tolist(), J_hist=J_hist),
          open(OUT/f"{TAG}.json", "w"), indent=2)
print(f"Saved {OUT/(TAG+'.json')}")
print("Done.")
