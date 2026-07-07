"""plot_joint_result.py — 5-panel plot of the full joint inversion result.

Reads sp_joint.json, runs the forward at the joint MAP (densification +
conductivity + surface-T history) and at the literature/initial baseline, and
plots density, age, velocity, temperature vs data + the recovered surface-T
history.
"""
from __future__ import annotations
import functools, json
from pathlib import Path
import numpy as np, pandas as pd
import firedrake as fd
from firedrake.adjoint import stop_annotating
from firn.models.firn import FirnParameters, FirnModel
from firn.physics.densification import herron_langway as _hl
from firn.constants import year as YEAR_S
herron_langway = functools.partial(_hl, smooth=True)

PROC = Path(__file__).parent.parent.parent / "processed"
OUT = Path(__file__).parent.parent.parent / "results"
J = json.load(open(OUT/"sp_joint.json")); m = J["m_map"]
Tk = np.array(J["T_knots"]); KY = np.array(J["knot_years"])
print(f"MAP: { {k: round(v,3) for k,v in m.items()} }")
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, BDOT, DT_YEARS, SPIN_YEARS = 350.0, 0.085, 5.0, 2500.0
K_ICE_BASE, BETA, N_ICE = 2.1, 5.0e2, np.sqrt(3.18)
P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)
n_steps = int(SPIN_YEARS/DT_YEARS)
step_years = 2015.0 - (n_steps-1-np.arange(n_steps))*DT_YEARS

dens = pd.read_csv(PROC/"sp19_density.csv").query("depth_m<=@H0")
age = pd.read_csv(PROC/"sp19_depth_age.csv"); age["age_yr"]=2015.0-age.year_CE; age=age.query("depth_m<=@H0 and age_yr>=0")
bT = pd.read_csv(PROC/"spicecore_borehole_T.csv").query("depth_m<=@H0")
ap = pd.read_csv(PROC/"apres_vertical_velocity_processed.csv")
ap = ap[(ap.range_m<=H0)&ap.v_smooth_m_yr.notna()&(ap.coherence>0.5)]
_vb=np.arange(8.0,H0,8.0); _vm=0.5*(_vb[:-1]+_vb[1:]); ap["_b"]=np.digitize(ap.range_m.values,_vb)
_vr=[(_vm[i],ap.v_smooth_m_yr[ap._b==i+1].median(),int((ap._b==i+1).sum())) for i in range(len(_vm))]
_vr=[r for r in _vr if r[2]>=3]; obs_v_d=np.array([r[0] for r in _vr]); obs_v=np.array([r[1] for r in _vr])

mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0*(1.0-(1.0-fd.SpatialCoordinate(mesh)[0])**STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); Rs = fd.FunctionSpace(mesh, "R", 0)
dx=fd.dx(domain=mesh); ds=fd.ds(domain=mesh); depth=H0-fd.SpatialCoordinate(mesh)[0]; psi=fd.TestFunction(V)
def mk(v,n): f=fd.Function(Rs,name=n); f.assign(float(v)); return f

def run(dens_p, kf, Q, T_series):
    params=FirnParameters(hl_k0_prefactor=mk(dens_p["hl_k0"],"k0"),hl_k1_prefactor=mk(dens_p["hl_k1"],"k1"),
        hl_Ea_stage1=mk(dens_p["hl_Ea1"],"e1"),hl_Ea_stage2=mk(dens_p["hl_Ea2"],"e2"),
        k_ice=K_ICE_BASE*kf, basal_heat_flux_W_m2=mk(Q,"Q"))
    model=FirnModel(params,densification_rate_fn=herron_langway)
    ws0=-BDOT*rho_i/RHO_SURF/YEAR_S
    H_f=fd.Function(V);rho_f=fd.Function(V);w_f=fd.Function(V);age_f=fd.Function(V)
    H_o=fd.Function(V);rho_o=fd.Function(V);w_o=fd.Function(V);age_o=fd.Function(V);Ts=fd.Function(V)
    H_f.assign(c_i*(float(T_series.mean())+273.15-T_ref));H_o.assign(H_f)
    rho_f.interpolate(RHO_SURF+(820.0-RHO_SURF)*(1.0-fd.exp(-depth/25.0)));rho_o.assign(rho_f)
    w_f.assign(ws0);w_o.assign(w_f);age_f.assign(0.0);age_o.assign(0.0)
    ac=mk(BDOT,"ac");rs=mk(RHO_SURF,"rs");ws=mk(ws0,"ws");dt_r=mk(DT_YEARS*YEAR_S,"dt")
    bcr=fd.DirichletBC(V,rs,SID);bcw=fd.DirichletBC(V,ws,SID);bca=fd.DirichletBC(V,mk(0.0,"a0"),SID)
    bm=ac*rho_i/P0.spy; Ht=fd.TrialFunction(V);rt=fd.TrialFunction(V);wt=fd.TrialFunction(V);at=fd.TrialFunction(V)
    sp={"ksp_type":"preonly","pc_type":"lu"}
    FH=model.enthalpy_form(Ht,H_o,rho_f,w_f,psi,dt_r)+fd.Constant(BETA)*(Ht-c_i*(Ts-fd.Constant(T_ref)))*psi*ds(SID)
    sH=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(FH),fd.rhs(FH),H_f,constant_jacobian=False),solver_parameters=sp)
    Te=model.temperature_from_enthalpy(H_f)
    Fr,_=model.density_form(rt,rho_o,Te,w_f,None,bm,psi,dt_r)
    sR=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(Fr),fd.rhs(Fr),rho_f,bcs=[bcr],constant_jacobian=False),solver_parameters=sp)
    dr=herron_langway(rho_f,Te,params=params,bdot=bm); dw=model.velocity_delta(wt,w_o,rho_f,dr,psi,regularization=1e-3)
    sW=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(dw),fd.rhs(dw),w_f,bcs=[bcw],constant_jacobian=False),solver_parameters=sp)
    Fa=model.age_form(at,age_o,w_f,w_o,psi,dt_r)
    sA=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(Fa),fd.rhs(Fa),age_f,bcs=[bca],constant_jacobian=False),solver_parameters=sp)
    with stop_annotating():
        for k in range(n_steps):
            Ts.interpolate(fd.Constant(T_series[k]+273.15)); sH.solve();sR.solve();sW.solve();sA.solve()
            H_o.assign(H_f);rho_o.assign(rho_f);w_o.assign(w_f);age_o.assign(age_f)
    xs=mesh.coordinates.dat.data_ro.reshape(-1);d=H0-xs;o=np.argsort(d)
    Tp=fd.Function(V).interpolate(H_f/c_i+T_ref)
    return dict(d=d[o],rho=rho_f.dat.data_ro[o],age=age_f.dat.data_ro[o]/YEAR_S,
                w=w_f.dat.data_ro[o]*YEAR_S,T=Tp.dat.data_ro[o]-273.15)

LIT = dict(hl_k0=11.0,hl_k1=575.0,hl_Ea1=10160.0,hl_Ea2=21400.0)
print("running baseline (literature, const T, default k)...")
pb = run(LIT, 1.0, 0.0, np.full(n_steps, -45.4))
print("running joint MAP...")
pm = run({k:m[k] for k in ["hl_k0","hl_k1","hl_Ea1","hl_Ea2"]}, m["k_factor"], m["Q_base"],
         np.interp(step_years, KY, Tk))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 5, figsize=(22, 7))
ax[0].plot(dens.rho_kgm3*1000, dens.depth_m, "k.", ms=3); ax[0].plot(pb["rho"],pb["d"],"C1--",lw=1.3,label="literature"); ax[0].plot(pm["rho"],pm["d"],"C0-",lw=2,label="joint MAP")
ax[0].set_title("Density"); ax[0].set_xlabel("kg/m³"); ax[0].set_ylabel("depth (m)"); ax[0].legend(fontsize=8)
ax[1].plot(age.age_yr, age.depth_m, "k.", ms=2); ax[1].plot(pb["age"],pb["d"],"C1--",lw=1.3); ax[1].plot(pm["age"],pm["d"],"C0-",lw=2); ax[1].set_title("Age"); ax[1].set_xlabel("yr")
ws0y=-BDOT*rho_i/RHO_SURF
ax[2].plot(obs_v, obs_v_d, "k.", ms=5); ax[2].plot((pb["w"]-ws0y)/N_ICE,pb["d"],"C1--",lw=1.3); ax[2].plot((pm["w"]-ws0y)/N_ICE,pm["d"],"C0-",lw=2); ax[2].set_title("Velocity (ApRES)"); ax[2].set_xlabel("m/yr")
ax[3].plot(bT.T_C, bT.depth_m, "k.", ms=3); ax[3].plot(pb["T"],pb["d"],"C1--",lw=1.3); ax[3].plot(pm["T"],pm["d"],"C0-",lw=2); ax[3].set_title("Temperature"); ax[3].set_xlabel("°C")
for a in ax[:4]: a.invert_yaxis(); a.grid(alpha=0.3)
ax[4].plot(Tk, KY, "C0-o", lw=2); ax[4].set_title("Recovered surface-T history"); ax[4].set_xlabel("°C"); ax[4].set_ylabel("year CE"); ax[4].grid(alpha=0.3)
fig.suptitle(f"South Pole JOINT inversion: density+age+velocity+temperature  |  k_ice={m['k_factor']*K_ICE_BASE:.2f}")
fig.tight_layout(); fig.savefig(OUT/"sp_joint_result.png", dpi=110)
print(f"Saved {OUT/'sp_joint_result.png'}")
