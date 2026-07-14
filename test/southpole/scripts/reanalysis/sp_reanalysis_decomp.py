"""sp_reanalysis_decomp.py — attribution decomposition under the b(t) MAP.

Four forward scenarios of the calibrated model (default MAP: sp_joint_bdot.json):

  CTRL  : T = T(1000CE),  b = b(1000CE)      (no climate change)
  TONLY : T = T(t) MAP,   b = b(1000CE)      (temperature response only)
  BONLY : T = T(1000CE),  b = b(t) MAP       (accumulation response only)
  FULL  : T = T(t),       b = b(t)           (the reanalysis)

Surface-height anomaly per scenario X (surface-following 130 m column,
steady deep dynamics; constant ezz cancels):  dh'/dt = |w_bot|_X - |w_bot|_CTRL.
Decomposition of each h': ice-equivalent mass part m' = ∫(b_X - b_CTRL)dt and
air part ΔFAC; h' ≈ m' + ΔFAC (residual = basal-export anomaly, reported).
Additivity check: h'_FULL vs h'_TONLY + h'_BONLY.

Outputs: results/sp_reanalysis_decomp.{json,png}
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv python> sp_reanalysis_decomp.py
"""
from __future__ import annotations
import functools, json, os, sys, time
from pathlib import Path
import numpy as np
import firedrake as fd
from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.physics.densification import herron_langway as _hl
from firnpack.constants import year as YEAR_S
herron_langway = functools.partial(_hl, smooth=True)
sys.stdout.reconfigure(line_buffering=True)

OUT = Path(__file__).parent.parent.parent / "results"
TAG = "sp_reanalysis_decomp"
MAP_FILE = os.environ.get("FIRN_MAP", "sp_joint_bdot.json")
H0, STRETCH_P, SID = 130.0, 2.5, 2
NZ = int(os.environ.get("FIRN_NZ", "100"))
RHO_SURF = 350.0
DT_YEARS = float(os.environ.get("FIRN_DT_YEARS", "5.0"))
SPIN_YEARS = float(os.environ.get("FIRN_SPIN_YEARS", "2500"))
K_ICE_BASE, BETA = 2.1, 5.0e2

MAP = json.load(open(OUT / MAP_FILE))
m = MAP["m_map"]
EZZ_YR = float(m.get("ezz_yr", 0.0))
T_KY, T_K = np.asarray(MAP["knot_years"]), np.asarray(MAP["T_knots"])
B_KY, B_K = np.asarray(MAP["b_knot_years"]), np.asarray(MAP["b_knots"])
print("=" * 64)
print(f"SP reanalysis DECOMPOSITION — {MAP_FILE}")
print(f"  T(1000)={T_K[0]:.2f}C  b(1000)={B_K[0]:.4f} m/yr  ezz={EZZ_YR:.2e}/yr")
print("=" * 64)

_pk = dict(
    hl_k0_prefactor=m["hl_k0"], hl_k1_prefactor=m["hl_k1"],
    hl_Ea_stage1=m["hl_Ea1"], hl_Ea_stage2=m["hl_Ea2"],
    basal_heat_flux_W_m2=m["Q_base"],
    hl_stage2_shape=float(m.get("s2_shape", 1.0)))
if "k_snow_scale" in m:   # r5-style MAP: Calonne-2019 conductivity
    _pk.update(conductivity_law="calonne2019",
               k_snow_scale=float(m["k_snow_scale"]),
               k_firn_scale=float(m["k_firn_scale"]))
else:                     # legacy quadratic law
    _pk.update(k_ice=K_ICE_BASE * m["k_factor"])
params = FirnParameters(**_pk)
c_i, T_ref, rho_i = float(params.c_i), float(params.T_ref), float(params.rho_i)
model = FirnModel(params, densification_rate_fn=herron_langway)

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

H_f=fd.Function(V); rho_f=fd.Function(V); w_f=fd.Function(V); age_f=fd.Function(V)
H_o=fd.Function(V); rho_o=fd.Function(V); w_o=fd.Function(V); age_o=fd.Function(V); Ts_eff=fd.Function(V)
ac=mk(B_K[0],"ac"); rs=mk(RHO_SURF,"rs"); ws=mk(-B_K[0]*rho_i/RHO_SURF/YEAR_S,"ws"); dt_r=mk(DT_YEARS*YEAR_S,"dt")
bc_rho=fd.DirichletBC(V,rs,SID); bc_w=fd.DirichletBC(V,ws,SID); bc_age=fd.DirichletBC(V,mk(0.0,"a0"),SID)
bdot_mass=ac*rho_i/params.spy
sp_lin={"ksp_type":"preonly","pc_type":"lu"}
Ht=fd.TrialFunction(V); rt=fd.TrialFunction(V); wt=fd.TrialFunction(V); at=fd.TrialFunction(V)
F_H=model.enthalpy_form(Ht,H_o,rho_f,w_f,psi,dt_r)+fd.Constant(BETA)*(Ht-c_i*(Ts_eff-fd.Constant(T_ref)))*psi*ds(SID)
solv_H=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(F_H),fd.rhs(F_H),H_f,constant_jacobian=False),solver_parameters=sp_lin)
T_expr=model.temperature_from_enthalpy(H_f)
F_rho,_=model.density_form(rt,rho_o,T_expr,w_f,None,bdot_mass,psi,dt_r)
solv_rho=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(F_rho),fd.rhs(F_rho),rho_f,bcs=[bc_rho],constant_jacobian=False),solver_parameters=sp_lin)
drhodt_expr=herron_langway(rho_f,T_expr,params=params,bdot=bdot_mass)
delta_w=model.velocity_delta(wt,w_o,rho_f,drhodt_expr,psi,regularization=1e-3,
                             horizontal_divergence=fd.Constant(-EZZ_YR/YEAR_S))
solv_w=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(delta_w),fd.rhs(delta_w),w_f,bcs=[bc_w],constant_jacobian=False),solver_parameters=sp_lin)
F_age=model.age_form(at,age_o,w_f,w_o,psi,dt_r)
solv_age=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(F_age),fd.rhs(F_age),age_f,bcs=[bc_age],constant_jacobian=False),solver_parameters=sp_lin)

n_steps = int(SPIN_YEARS / DT_YEARS)
step_years = 2015.0 - (n_steps - 1 - np.arange(n_steps)) * DT_YEARS
Ts_full = np.interp(step_years, T_KY, T_K)
Ts_ctrl = np.full(n_steps, T_K[0])
b_full = np.exp(np.interp(step_years, B_KY, np.log(B_K)))
b_ctrl = np.full(n_steps, B_K[0])

def run(name, ts_C, b_series):
    t0 = time.perf_counter()
    T0 = float(T_K.mean()) + 273.15
    H_f.assign(c_i*(T0-T_ref)); H_o.assign(H_f)
    rho_f.interpolate(RHO_SURF+(820.0-RHO_SURF)*(1.0-fd.exp(-depth/25.0))); rho_o.assign(rho_f)
    w_f.assign(float(ws.dat.data_ro[0])); w_o.assign(w_f); age_f.assign(0.0); age_o.assign(0.0)
    fac = np.zeros(n_steps); wbot = np.zeros(n_steps); tmean = np.zeros(n_steps)
    for k in range(n_steps):
        Ts_eff.assign(float(ts_C[k]) + 273.15)
        ac.assign(float(b_series[k]))
        ws.assign(-float(b_series[k]) * rho_i / RHO_SURF / YEAR_S)
        solv_H.solve(); solv_rho.solve(); solv_w.solve(); solv_age.solve()
        H_o.assign(H_f); rho_o.assign(rho_f); w_o.assign(w_f); age_o.assign(age_f)
        fac[k] = float(fd.assemble((1.0 - rho_f / rho_i) * dx))
        wbot[k] = float(w_f.dat.data_ro[i_bot]) * YEAR_S
        tmean[k] = float(fd.assemble(H_f * dx)) / (H0 * c_i) + T_ref - 273.15
    print(f"  [{name}] {time.perf_counter()-t0:.0f}s  FAC(2015)={fac[-1]:.3f}  |w_bot|={-wbot[-1]:.4f}")
    return dict(fac=fac, wbot=wbot, tmean=tmean, b=np.asarray(b_series, dtype=float))

print("scenarios:")
S = dict(
    CTRL=run("CTRL", Ts_ctrl, b_ctrl),
    TONLY=run("TONLY", Ts_full, b_ctrl),
    BONLY=run("BONLY", Ts_ctrl, b_full),
    FULL=run("FULL", Ts_full, b_full),
)

def anomalies(X):
    dhdt = S["CTRL"]["wbot"] - X["wbot"]                 # m/yr
    hp = np.cumsum(dhdt) * DT_YEARS                      # m
    mp = np.cumsum(X["b"] - S["CTRL"]["b"]) * DT_YEARS   # m ice eq (mass part)
    ap = X["fac"] - S["CTRL"]["fac"]                     # m air (FAC part)
    return hp, mp, ap
HP, MP, AP = {}, {}, {}
for k in ["TONLY", "BONLY", "FULL"]:
    HP[k], MP[k], AP[k] = anomalies(S[k])

i1957 = int(np.argmin(np.abs(step_years - 1957.0)))
def rate(h): return float(np.mean(np.gradient(h, DT_YEARS)[i1957:]) * 1000)  # mm/yr
addit = HP["FULL"] - (HP["TONLY"] + HP["BONLY"])
closure = HP["FULL"] - (MP["FULL"] + AP["FULL"])
print("\n===== DECOMPOSITION (2015, cm; rate 1957-2015, mm/yr) =====")
for k in ["TONLY", "BONLY", "FULL"]:
    print(f"  {k:6s} h'={HP[k][-1]*100:+7.2f} cm  (mass {MP[k][-1]*100:+7.2f}, air {AP[k][-1]*100:+6.2f})"
          f"   rate={rate(HP[k]):+6.2f} mm/yr")
print(f"  additivity |FULL-(T+B)|(2015) = {abs(addit[-1])*100:.2f} cm")
print(f"  closure    |h'-(mass+air)|(2015) = {abs(closure[-1])*100:.2f} cm (basal-export anomaly)")

res = dict(map_file=MAP_FILE, years=step_years.tolist(),
           Ts_full=Ts_full.tolist(), b_full=b_full.tolist(),
           T_ctrl=float(T_K[0]), b_ctrl=float(B_K[0]),
           headline={k: dict(h_2015_cm=float(HP[k][-1]*100), mass_2015_cm=float(MP[k][-1]*100),
                             air_2015_cm=float(AP[k][-1]*100), rate_1957_2015_mm_yr=rate(HP[k]))
                     for k in ["TONLY", "BONLY", "FULL"]},
           hprime={k: HP[k].tolist() for k in HP}, mass={k: MP[k].tolist() for k in MP},
           air={k: AP[k].tolist() for k in AP})
json.dump(res, open(OUT / f"{TAG}.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
show = step_years >= 1000.0; yr = step_years[show]
fig, AX = plt.subplots(2, 2, figsize=(13, 9))
ax = AX[0,0]
ax.plot(yr, Ts_full[show], "C3-", lw=1.5); ax.axhline(T_K[0], color="k", ls="--", lw=1)
ax.set_title("(a) Surface-T forcing (MAP)"); ax.set_ylabel("°C"); ax.grid(alpha=0.3)
ax = AX[0,1]
ax.plot(yr, b_full[show], "C2-", lw=1.5, label="b(t) MAP")
if "b_prior_centers" in MAP:
    ax.plot(MAP["b_knot_years"], MAP["b_prior_centers"], "k.", ms=6, label="prior (Buizert/ERA5)")
ax.axhline(B_K[0], color="k", ls="--", lw=1)
ax.set_title("(b) Accumulation forcing"); ax.set_ylabel("m ice yr$^{-1}$")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = AX[1,0]
for k, c in [("TONLY","C3"), ("BONLY","C2"), ("FULL","C0")]:
    ax.plot(yr, HP[k][show]*100, c+"-", lw=1.8, label=k)
ax.plot(yr, (HP["TONLY"]+HP["BONLY"])[show]*100, "k:", lw=1.2, label="T+B (additivity)")
ax.axhline(0, color="k", lw=0.8)
ax.set_title("(c) Firn surface-height anomaly by driver"); ax.set_ylabel("cm")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = AX[1,1]
ax.plot(yr, HP["FULL"][show]*100, "C0-", lw=2, label="FULL h'")
ax.plot(yr, MP["FULL"][show]*100, "C2--", lw=1.5, label="ice-equivalent mass part")
ax.plot(yr, AP["FULL"][show]*100, "C1--", lw=1.5, label="air (ΔFAC) part")
ax.axhline(0, color="k", lw=0.8)
ax.set_title("(d) FULL anomaly decomposition"); ax.set_ylabel("cm")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
for a in AX.flat: a.set_xlabel("year CE")
fig.suptitle(f"South Pole firn reanalysis decomposition — {Path(MAP_FILE).stem}", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / f"{TAG}.png", dpi=130)
print(f"Saved {OUT/f'{TAG}.json'} and .png")
print("Done.")
