"""sp_layer_fit_diagnostic.py — does the MAP forward REPRODUCE the observed
annual-layer thickness where its b(t) spikes (deposition ~1950, ~13 m)?

The r7 b(t) spikes to 0.123 at 1950 while raw layers say 0.093. Under steady
mass conservation d(age)/dz = rho/(b*rho_i), so the inversion observable and
the raw apparent accumulation are the SAME relation — they should agree. This
runs the MAP forward and compares, in DEPTH space:
  * model d(age)/dz(z) vs observed d(age)/dz(z)   (the assimilated observable)
  * model implied b = rho_model * (dz/d age) / rho_i  vs the control b(t)
    (internal consistency: does the control b match the model's OWN layers?)
  * where deposition-1950 lands in each.

If the model layer thickness MATCHES the observed thin layers at 13 m while
b(1950)=0.123, then b is trading against density/velocity (spike is a
compensation, not real accum). If the model MISSES (too-thick layers there),
the spike is a failed fit. Either way the spike is not a clean accumulation
signal.

Output: results/sp_layer_fit_diagnostic.png
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> sp_layer_fit_diagnostic.py
Env: FIRN_MAP (sp_joint_r7.json), FIRN_T_SHIFT (-5.8 for r5b+ MAPs)
"""
from __future__ import annotations
import functools, json, math, os
from pathlib import Path
import numpy as np, pandas as pd
import firedrake as fd
from firedrake.adjoint import stop_annotating
from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.physics.densification import herron_langway as _hl
from firnpack.constants import year as YEAR_S
herron_langway = functools.partial(_hl, smooth=True)

PROC = Path(__file__).parent.parent.parent / "processed"
OUT = Path(__file__).parent.parent.parent / "results"
MAP_FILE = os.environ.get("FIRN_MAP", "sp_joint_r7.json")
T_SHIFT = float(os.environ.get("FIRN_T_SHIFT", "-5.8"))
JM = json.load(open(OUT / MAP_FILE)); m = JM["m_map"]
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, DT_YEARS, SPIN_YEARS = 350.0, 5.0, 2500.0
K_ICE_BASE, BETA = 2.1, 5.0e2
P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)
n_steps = int(SPIN_YEARS / DT_YEARS)
step_years = 2015.0 - (n_steps - 1 - np.arange(n_steps)) * DT_YEARS

T_series = np.interp(step_years, np.asarray(JM["knot_years"]), np.asarray(JM["T_knots"]))
b_series = np.exp(np.interp(step_years, np.asarray(JM["b_knot_years"]),
                            np.log(np.asarray(JM["b_knots"]))))
_pk = dict(hl_k0_prefactor=m["hl_k0"], hl_k1_prefactor=m["hl_k1"],
           hl_Ea_stage1=m["hl_Ea1"], hl_Ea_stage2=m["hl_Ea2"],
           basal_heat_flux_W_m2=m["Q_base"], hl_stage2_shape=float(m.get("s2_shape", 1.0)))
if "k_snow_scale" in m:
    _pk.update(conductivity_law="calonne2019", k_snow_scale=float(m["k_snow_scale"]),
               k_firn_scale=float(m["k_firn_scale"]))
else:
    _pk.update(k_ice=K_ICE_BASE * m["k_factor"])
params = FirnParameters(**_pk)
model = FirnModel(params, densification_rate_fn=herron_langway)

mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0 * (1.0 - (1.0 - fd.SpatialCoordinate(mesh)[0]) ** STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); Rs = fd.FunctionSpace(mesh, "R", 0)
dx = fd.dx(domain=mesh); ds = fd.ds(domain=mesh)
depth = H0 - fd.SpatialCoordinate(mesh)[0]; psi = fd.TestFunction(V)
def mk(v, n): f = fd.Function(Rs, name=n); f.assign(float(v)); return f
ws0 = -b_series[0] * rho_i / RHO_SURF / YEAR_S
H_f=fd.Function(V); rho_f=fd.Function(V); w_f=fd.Function(V); age_f=fd.Function(V)
H_o=fd.Function(V); rho_o=fd.Function(V); w_o=fd.Function(V); age_o=fd.Function(V); Ts=fd.Function(V)
ac=mk(b_series[0],"ac"); rs=mk(RHO_SURF,"rs"); ws=mk(ws0,"ws"); dt_r=mk(DT_YEARS*YEAR_S,"dt")
bcr=fd.DirichletBC(V,rs,SID); bcw=fd.DirichletBC(V,ws,SID); bca=fd.DirichletBC(V,mk(0.0,"a0"),SID)
bm=ac*rho_i/P0.spy
Ht=fd.TrialFunction(V); rt=fd.TrialFunction(V); wt=fd.TrialFunction(V); at=fd.TrialFunction(V)
sp={"ksp_type":"preonly","pc_type":"lu"}
FH=model.enthalpy_form(Ht,H_o,rho_f,w_f,psi,dt_r)+fd.Constant(BETA)*(Ht-c_i*(Ts-fd.Constant(T_ref)))*psi*ds(SID)
sH=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(FH),fd.rhs(FH),H_f,constant_jacobian=False),solver_parameters=sp)
Te=model.temperature_from_enthalpy(H_f)
Fr,_=model.density_form(rt,rho_o,Te,w_f,None,bm,psi,dt_r)
sR=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(Fr),fd.rhs(Fr),rho_f,bcs=[bcr],constant_jacobian=False),solver_parameters=sp)
dr=herron_langway(rho_f,Te,params=params,bdot=bm)
dw=model.velocity_delta(wt,w_o,rho_f,dr,psi,regularization=1e-3,
                        horizontal_divergence=fd.Constant(-float(m.get("ezz_yr",0.0))/YEAR_S))
sW=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(dw),fd.rhs(dw),w_f,bcs=[bcw],constant_jacobian=False),solver_parameters=sp)
Fa=model.age_form(at,age_o,w_f,w_o,psi,dt_r)
sA=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(Fa),fd.rhs(Fa),age_f,bcs=[bca],constant_jacobian=False),solver_parameters=sp)
with stop_annotating():
    H_f.assign(c_i*(float(T_series.mean())+273.15-T_ref)); H_o.assign(H_f)
    rho_f.interpolate(RHO_SURF+(820.0-RHO_SURF)*(1.0-fd.exp(-depth/25.0))); rho_o.assign(rho_f)
    w_f.assign(ws0); w_o.assign(w_f); age_f.assign(0.0); age_o.assign(0.0)
    for k in range(n_steps):
        Ts.assign(float(T_series[k])+273.15); ac.assign(float(b_series[k]))
        ws.assign(-float(b_series[k])*rho_i/RHO_SURF/YEAR_S)
        sH.solve(); sR.solve(); sW.solve(); sA.solve()
        H_o.assign(H_f); rho_o.assign(rho_f); w_o.assign(w_f); age_o.assign(age_f)
xs = mesh.coordinates.dat.data_ro.reshape(-1); dm = H0 - xs; o = np.argsort(dm)
md = dm[o]; mrho = rho_f.dat.data_ro[o]; mage = age_f.dat.data_ro[o] / YEAR_S
# model d(age)/dz and implied b
mdadz = np.gradient(mage, md)                       # yr/m
mb_implied = (1.0 / mdadz) * mrho / rho_i           # m ice/yr, = model layer thickness * rho/rho_i
mdep = 2015.0 - mage

# observed layer data
a = pd.read_csv(PROC/"sp19_depth_age.csv"); a=a[(a.depth_m<=H0)&(a.year_CE<=2015)].sort_values("depth_m")
oz, oyr = a.depth_m.values, a.year_CE.values
oage = 2015.0 - oyr
odadz = np.gradient(oage, oz)
dobs = pd.read_csv(PROC/"sp19_density.csv")
orho = np.interp(oz, dobs.depth_m.values, dobs.rho_kgm3.values*1000.0)
ob_app = (1.0/odadz) * orho / rho_i
# control b(t) on deposition-year
cby, cbk = np.asarray(JM["b_knot_years"]), np.asarray(JM["b_knots"])

# depth where model & obs hit deposition 1950
def depth_at_year(dep, dpt, y):
    oo=np.argsort(dep); return float(np.interp(y, dep[oo], dpt[oo]))
z50_m = depth_at_year(mdep, md, 1950); z50_o = depth_at_year(2015.0-oage, oz, 1950)
print(f"deposition 1950 at depth: model {z50_m:.2f} m, observed {z50_o:.2f} m  (registration Δ={z50_m-z50_o:+.2f} m)")
def val_at(z_arr, v_arr, z): return float(np.interp(z, z_arr, v_arr))
print(f"at ~13 m: model dage/dz={val_at(md,mdadz,13):.2f}, obs={val_at(oz,odadz,13):.2f} yr/m")
print(f"          model implied b={val_at(md,mb_implied,13):.4f}, obs b_app={val_at(oz,ob_app,13):.4f}")
print(f"control b(1950)={np.exp(np.interp(1950,cby,np.log(cbk))):.4f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C_M,C_O,C_C="#2563EB","#111827","#EA580C"
fig,AX=plt.subplots(1,3,figsize=(15,5.5))
ax=AX[0]
ax.plot(odadz, oz, "-", color=C_O, lw=1, alpha=0.5, label="observed (SP19)")
ax.plot(mdadz, md, "-", color=C_M, lw=2, label="model (r7 MAP)")
ax.axhline(z50_o, color=C_C, ls=":", lw=1); ax.text(ax.get_xlim()[1]*0.6, z50_o-1, "dep. 1950", color=C_C, fontsize=8)
ax.set_ylim(30,0); ax.set_xlabel("d(age)/dz (yr/m)"); ax.set_ylabel("depth (m)")
ax.set_title("(a) assimilated observable: layer gradient"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax=AX[1]
ax.plot(ob_app, oz, "-", color=C_O, lw=1, alpha=0.5, label="obs b_app = ρλ/ρ_i")
ax.plot(mb_implied, md, "-", color=C_M, lw=2, label="model layer→b = ρλ/ρ_i")
ax.axhline(z50_o, color=C_C, ls=":", lw=1)
ax.set_ylim(30,0); ax.set_xlabel("implied accumulation (m ice/yr)"); ax.set_ylabel("depth (m)")
ax.set_xlim(0.05,0.16); ax.set_title("(b) layer-implied b (model vs obs, depth space)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax=AX[2]
ax.plot(2015.0-oage, ob_app, "-", color=C_O, lw=1, alpha=0.5, label="obs b_app")
ax.plot(mdep, mb_implied, "-", color=C_M, lw=1.6, label="model layer→b")
ax.plot(cby, cbk, "o-", color=C_C, ms=4, lw=2, label="CONTROL b(t)")
ax.axvline(1950, color=C_C, ls=":", lw=1)
ax.set_xlim(1850,2015); ax.set_ylim(0.05,0.16)
ax.set_xlabel("deposition year"); ax.set_ylabel("accumulation (m ice/yr)")
ax.set_title("(c) control b(t) vs layer-implied (model & obs)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle(f"Layer-fit diagnostic — {MAP_FILE}: does the control b(t) match the model's own layers?", fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"sp_layer_fit_diagnostic.png", dpi=140)
print(f"Saved {OUT/'sp_layer_fit_diagnostic.png'}")
