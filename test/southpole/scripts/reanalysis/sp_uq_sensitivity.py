"""sp_uq_sensitivity.py — sensitivities of the reanalysis headline to the 12 controls.

Forward-only. For each internal control coordinate (log-space for the 5 LOGS
params, linear for Q_base and the 6 T-knots) run the CTRL/FULL scenario pair at
x* +/- h and central-difference the attribution diagnostics:

    q = [ h'(2015),  mean dh'/dt 1957-2015,  dFAC(2015) ]

Both scenarios are re-run per perturbed x (CTRL uses Tk0 from the same x), so
the derivative is of the *attribution*, not of a fixed-baseline height.
Scalar params live in R-space Functions reassigned between runs — forms stay
symbolically identical, no recompilation across the 25 parameter sets.

Output: results/sp_uq_sensitivity.json {S (3x12), base, names, fd_h}
Run: PYTHONPATH=src OMP_NUM_THREADS=1 /home/andrew/venv-firedrake-2026/bin/python sp_uq_sensitivity.py
"""
from __future__ import annotations
import functools, json, math, os, sys, time
from pathlib import Path
import numpy as np
import firedrake as fd
from firn.models.firn import FirnParameters, FirnModel
from firn.physics.densification import herron_langway as _hl
from firn.constants import year as YEAR_S
herron_langway = functools.partial(_hl, smooth=True)
sys.stdout.reconfigure(line_buffering=True)

OUT = Path(__file__).parent.parent.parent / "results"
TAG = "sp_uq_sensitivity"
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, BDOT, DT_YEARS = 350.0, 0.085, 5.0
SPIN_YEARS = float(os.environ.get("FIRN_SPIN_YEARS", "2500"))
FD_H = float(os.environ.get("FIRN_FD_H", "1e-2"))
K_ICE_BASE, BETA = 2.1, 5.0e2
LOGS = {"hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "k_factor"}
SCAL_NAMES = ["hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "k_factor", "Q_base"]

MAP = json.load(open(OUT / "sp_joint.json"))
mm = MAP["m_map"]
KNOT_YEARS = np.array(MAP["knot_years"])
N_KNOT = len(KNOT_YEARS)
NAMES = SCAL_NAMES + [f"Tk{i}" for i in range(N_KNOT)]
N_CTRL = len(NAMES)
x_star = np.array([math.log(mm[n]) if n in LOGS else mm[n] for n in SCAL_NAMES]
                  + list(MAP["T_knots"]))
print("=" * 64)
print(f"Reanalysis sensitivities dq/dx at MAP (h={FD_H:g}, {2*N_CTRL+1} scenario pairs)")
print("=" * 64)

P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)

# ---- mesh / spaces ----
mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0 * (1.0 - (1.0 - fd.SpatialCoordinate(mesh)[0]) ** STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); R = fd.FunctionSpace(mesh, "R", 0)
dx = fd.dx(domain=mesh); ds = fd.ds(domain=mesh)
xc = fd.SpatialCoordinate(mesh)[0]; depth = H0 - xc; psi = fd.TestFunction(V)
def mk(v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f
xs = fd.Function(V).interpolate(xc).dat.data_ro.copy()
i_bot = int(np.argmin(xs))

# ---- scalar controls as R-space Functions (reassigned per parameter set) ----
log_k0 = mk(x_star[0], "lk0"); log_k1 = mk(x_star[1], "lk1")
log_Ea1 = mk(x_star[2], "lE1"); log_Ea2 = mk(x_star[3], "lE2")
log_kf = mk(x_star[4], "lkf"); Q_base = mk(x_star[5], "Qb")
scal_fns = [log_k0, log_k1, log_Ea1, log_Ea2, log_kf, Q_base]
params = FirnParameters(
    hl_k0_prefactor=fd.exp(log_k0), hl_k1_prefactor=fd.exp(log_k1),
    hl_Ea_stage1=fd.exp(log_Ea1), hl_Ea_stage2=fd.exp(log_Ea2),
    k_ice=K_ICE_BASE*fd.exp(log_kf), basal_heat_flux_W_m2=Q_base)
model = FirnModel(params, densification_rate_fn=herron_langway)

# ---- stepper (as sp_reanalysis.py) ----
ws0 = -BDOT*rho_i/RHO_SURF/YEAR_S
H_f=fd.Function(V,name="H"); rho_f=fd.Function(V,name="rho"); w_f=fd.Function(V,name="w"); age_f=fd.Function(V,name="age")
H_o=fd.Function(V); rho_o=fd.Function(V); w_o=fd.Function(V); age_o=fd.Function(V); Ts_eff=fd.Function(V)
ac=mk(BDOT,"ac"); rs=mk(RHO_SURF,"rs"); ws=mk(ws0,"ws"); dt_r=mk(DT_YEARS*YEAR_S,"dt")
bc_rho=fd.DirichletBC(V,rs,SID); bc_w=fd.DirichletBC(V,ws,SID); bc_age=fd.DirichletBC(V,mk(0.0,"a0"),SID)
bdot_mass=ac*rho_i/P0.spy
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

n_steps = int(SPIN_YEARS/DT_YEARS)
step_years = 2015.0 - (n_steps-1-np.arange(n_steps))*DT_YEARS
i1957 = int(np.argmin(np.abs(step_years - 1957.0)))

IC_T0C = float(np.mean(x_star[6:]))  # IC memory cancels in the CTRL/FULL pair difference

def run_series(ts_C):
    """One scenario at the currently-assigned scalar params; returns wbot series + final FAC."""
    H_f.assign(c_i*(IC_T0C+273.15-T_ref)); H_o.assign(H_f)
    rho_f.interpolate(RHO_SURF+(820.0-RHO_SURF)*(1.0-fd.exp(-depth/25.0))); rho_o.assign(rho_f)
    w_f.assign(ws0); w_o.assign(w_f); age_f.assign(0.0); age_o.assign(0.0)
    wbot = np.zeros(n_steps)
    for k in range(n_steps):
        Ts_eff.assign(float(ts_C[k])+273.15)
        solv_H.solve(); solv_rho.solve(); solv_w.solve(); solv_age.solve()
        H_o.assign(H_f); rho_o.assign(rho_f); w_o.assign(w_f); age_o.assign(age_f)
        wbot[k] = float(w_f.dat.data_ro[i_bot])*YEAR_S
    fac = float(fd.assemble((1.0-rho_f/rho_i)*dx))
    return wbot, fac

def attribution(x):
    """q(x) = [h'(2015) m, mean dh'/dt 1957-2015 m/yr, dFAC(2015) m]."""
    for f, v in zip(scal_fns, x[:6]): f.assign(float(v))
    tk = x[6:]
    ts_full = np.interp(step_years, KNOT_YEARS, tk)
    ts_ctrl = np.full(n_steps, tk[0])
    wb_c, fac_c = run_series(ts_ctrl)
    wb_f, fac_f = run_series(ts_full)
    dhdt = wb_c - wb_f
    return np.array([np.sum(dhdt)*DT_YEARS, np.mean(dhdt[i1957:]), fac_f - fac_c])

t0 = time.perf_counter()
q0 = attribution(x_star)
print(f"base: h'(2015)={q0[0]*100:+.2f} cm, rate(1957-2015)={q0[1]*1000:+.3f} mm/yr, "
      f"dFAC={q0[2]*100:+.2f} cm  [{time.perf_counter()-t0:.0f}s/pair]")

S = np.zeros((3, N_CTRL))
for j in range(N_CTRL):
    xp = x_star.copy(); xp[j] += FD_H
    xm = x_star.copy(); xm[j] -= FD_H
    qp = attribution(xp); qm = attribution(xm)
    S[:, j] = (qp - qm) / (2*FD_H)
    print(f"  [{j+1:2d}/{N_CTRL}] {NAMES[j]:9s} dh'/dx={S[0,j]*100:+.3f} cm/unit "
          f"drate/dx={S[1,j]*1000:+.4f} mm/yr/unit")

json.dump(dict(names=NAMES, x_star=x_star.tolist(), fd_h=FD_H,
               q_names=["hprime_2015_m", "rate_1957_2015_m_yr", "dfac_2015_m"],
               q_base=q0.tolist(), S=S.tolist(), logs=sorted(LOGS)),
          open(OUT/f"{TAG}.json", "w"), indent=1)
print(f"Saved {OUT/(TAG+'.json')}")
print("Done.")
