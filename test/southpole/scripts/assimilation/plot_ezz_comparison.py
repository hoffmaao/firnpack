"""plot_ezz_comparison.py — before/after figure for the dynamic-strain fix.

Runs the forward model at the sp_joint.json MAP (12 controls, no dynamic strain)
and the sp_joint_ezz.json MAP (13 controls, ezz), overlays both on the four
datasets, plus an apparent-strain panel (the discriminating diagnostic) and the
assimilation RMS summary. Output: results/sp_ezz_comparison.png
"""
from __future__ import annotations
import functools, json
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
J_OLD = json.load(open(OUT/"sp_joint.json"))
J_NEW = json.load(open(OUT/"sp_joint_ezz.json"))
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, BDOT, DT_YEARS, SPIN_YEARS = 350.0, 0.085, 5.0, 2500.0
K_ICE_BASE, BETA, N_ICE = 2.1, 5.0e2, np.sqrt(3.18)
P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)
n_steps = int(SPIN_YEARS/DT_YEARS)
step_years = 2015.0 - (n_steps-1-np.arange(n_steps))*DT_YEARS

# ---- data ----
dens = pd.read_csv(PROC/"sp19_density.csv").query("depth_m<=@H0")
age = pd.read_csv(PROC/"sp19_depth_age.csv"); age["age_yr"]=2015.0-age.year_CE
age = age.query("depth_m<=@H0 and age_yr>=0")
bT = pd.read_csv(PROC/"spicecore_borehole_T.csv").query("depth_m<=@H0")
ap = pd.read_csv(PROC/"apres_vertical_velocity_processed.csv")
ap = ap[(ap.range_m<=H0)&ap.v_smooth_m_yr.notna()&(ap.coherence>0.5)]
_vb=np.arange(8.0,H0,8.0); _vm=0.5*(_vb[:-1]+_vb[1:]); ap["_b"]=np.digitize(ap.range_m.values,_vb)
vbin = []
for i in range(len(_vm)):
    s = ap[ap._b==i+1]
    if len(s) >= 3:
        vbin.append((_vm[i], s.v_smooth_m_yr.median(),
                     max(s.v_smooth_m_yr.std()/np.sqrt(len(s)), 0.01),
                     np.abs(s.strain_rate_yr).median(),
                     np.abs(s.strain_rate_yr).quantile(0.25),
                     np.abs(s.strain_rate_yr).quantile(0.75)))
vbin = np.array(vbin)  # depth, v, sig_v, |strain| med, q25, q75

# ---- mesh + forward (pattern from plot_joint_result.py) ----
mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0*(1.0-(1.0-fd.SpatialCoordinate(mesh)[0])**STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); Rs = fd.FunctionSpace(mesh, "R", 0)
dx=fd.dx(domain=mesh); ds=fd.ds(domain=mesh); depth=H0-fd.SpatialCoordinate(mesh)[0]; psi=fd.TestFunction(V)
def mk(v,n): f=fd.Function(Rs,name=n); f.assign(float(v)); return f

def run(mm, T_knots, knot_years):
    ezz = float(mm.get("ezz_yr", 0.0))
    T_series = np.interp(step_years, np.asarray(knot_years), np.asarray(T_knots))
    params=FirnParameters(hl_k0_prefactor=mk(mm["hl_k0"],"k0"),hl_k1_prefactor=mk(mm["hl_k1"],"k1"),
        hl_Ea_stage1=mk(mm["hl_Ea1"],"e1"),hl_Ea_stage2=mk(mm["hl_Ea2"],"e2"),
        k_ice=K_ICE_BASE*mm["k_factor"], basal_heat_flux_W_m2=mk(mm["Q_base"],"Q"))
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
    dr=herron_langway(rho_f,Te,params=params,bdot=bm)
    dw=model.velocity_delta(wt,w_o,rho_f,dr,psi,regularization=1e-3,
                            horizontal_divergence=fd.Constant(-ezz/YEAR_S))
    sW=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(dw),fd.rhs(dw),w_f,bcs=[bcw],constant_jacobian=False),solver_parameters=sp)
    Fa=model.age_form(at,age_o,w_f,w_o,psi,dt_r)
    sA=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(Fa),fd.rhs(Fa),age_f,bcs=[bca],constant_jacobian=False),solver_parameters=sp)
    with stop_annotating():
        for k in range(n_steps):
            Ts.assign(float(T_series[k])+273.15); sH.solve();sR.solve();sW.solve();sA.solve()
            H_o.assign(H_f);rho_o.assign(rho_f);w_o.assign(w_f);age_o.assign(age_f)
    xs=mesh.coordinates.dat.data_ro.reshape(-1);d=H0-xs;o=np.argsort(d)
    dd=d[o]; wv=w_f.dat.data_ro[o]*YEAR_S
    v_app=(wv-(-BDOT*rho_i/RHO_SURF))/N_ICE
    return dict(d=dd,rho=rho_f.dat.data_ro[o],age=age_f.dat.data_ro[o]/YEAR_S,
                v=v_app, strain=np.abs(np.gradient(v_app, dd)),
                T=(H_f.dat.data_ro[o]/c_i+T_ref)-273.15)

print("running OLD MAP (no dynamic strain)...")
pold = run(J_OLD["m_map"], J_OLD["T_knots"], J_OLD["knot_years"])
print("running NEW MAP (ezz=%.2e/yr)..." % J_NEW["m_map"]["ezz_yr"])
pnew = run(J_NEW["m_map"], J_NEW["T_knots"], J_NEW["knot_years"])

# per-point age residuals (sigma = 10 + 3% of age)
sig_age = 10.0 + 0.03*age.age_yr.values
res_old = (np.interp(age.depth_m.values, pold["d"], pold["age"]) - age.age_yr.values)/sig_age
res_new = (np.interp(age.depth_m.values, pnew["d"], pnew["age"]) - age.age_yr.values)/sig_age

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CO, CN = "C1", "C0"
fig, AX = plt.subplots(2, 3, figsize=(15, 10))
ax = AX[0,0]
ax.plot(dens.rho_kgm3*1000, dens.depth_m, "k.", ms=3, alpha=0.6, label="SP19 data")
ax.plot(pold["rho"], pold["d"], CO+"--", lw=1.6, label="no strain (J=0.81)")
ax.plot(pnew["rho"], pnew["d"], CN+"-", lw=2, label="with $\\dot\\epsilon_{zz}$ (J=0.58)")
ax.set_title("(a) Density"); ax.set_xlabel("kg m$^{-3}$"); ax.set_ylabel("depth (m)")
ax.legend(fontsize=8, loc="lower left")

ax = AX[0,1]
ax.plot(age.age_yr, age.depth_m, "k.", ms=2, alpha=0.6)
ax.plot(pold["age"], pold["d"], CO+"--", lw=1.6)
ax.plot(pnew["age"], pnew["d"], CN+"-", lw=2)
ax.set_title("(b) Age"); ax.set_xlabel("yr")

ax = AX[0,2]
ax.errorbar(vbin[:,1], vbin[:,0], xerr=vbin[:,2], fmt="k.", ms=6, capsize=2, label="ApRES 8-m bins")
ax.plot(pold["v"], pold["d"], CO+"--", lw=1.6)
ax.plot(pnew["v"], pnew["d"], CN+"-", lw=2)
ax.set_title("(c) Apparent vertical velocity"); ax.set_xlabel("m yr$^{-1}$")

ax = AX[1,0]
m40 = vbin[:,0] >= 40.0
ax.errorbar(vbin[m40,3], vbin[m40,0],
            xerr=[vbin[m40,3]-vbin[m40,4], vbin[m40,5]-vbin[m40,3]],
            fmt="k.", ms=6, capsize=2, label="ApRES |strain| (med, IQR)")
mm40o = pold["d"] >= 40.0; mm40n = pnew["d"] >= 40.0
ax.plot(pold["strain"][mm40o], pold["d"][mm40o], CO+"--", lw=1.6)
ax.plot(pnew["strain"][mm40n], pnew["d"][mm40n], CN+"-", lw=2)
ax.set_xscale("log")
ax.set_title("(d) Apparent strain $|dv/dz|$ (>40 m)"); ax.set_xlabel("yr$^{-1}$")
ax.set_ylabel("depth (m)"); ax.legend(fontsize=8, loc="lower right")

ax = AX[1,1]
ax.plot(res_old, age.depth_m, CO+".", ms=3, alpha=0.6, label=f"no strain (rms {np.sqrt(np.mean(res_old**2)):.2f}σ)")
ax.plot(res_new, age.depth_m, CN+".", ms=3, alpha=0.6, label=f"with strain (rms {np.sqrt(np.mean(res_new**2)):.2f}σ)")
ax.axvline(0, color="k", lw=0.8)
ax.set_title("(e) Age residuals (model − obs)/σ"); ax.set_xlabel("σ")
ax.legend(fontsize=8, loc="lower left")

ax = AX[1,2]
# assimilation-reported kernel RMS (sigma units) from the two runs' logs
labels = ["density", "age", "temperature", "velocity"]
rms_old = [0.66, 0.71, 0.45, 0.63]; rms_new = [0.55, 0.31, 0.45, 0.63]
y = np.arange(len(labels))
ax.barh(y-0.18, rms_old, height=0.36, color=CO, label="no strain")
ax.barh(y+0.18, rms_new, height=0.36, color=CN, label="with $\\dot\\epsilon_{zz}$")
ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlabel("assimilation RMS (σ units)"); ax.set_title("(f) Fit summary")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

for a in [AX[0,0], AX[0,1], AX[0,2], AX[1,0], AX[1,1]]:
    a.invert_yaxis(); a.grid(alpha=0.3)
ez = J_NEW["m_map"]["ezz_yr"]
fig.suptitle("South Pole joint inversion with vs without ice-dynamic vertical strain "
             f"($\\dot\\epsilon_{{zz}}$ = {ez:.2e} yr$^{{-1}}$)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT/"sp_ezz_comparison.png", dpi=120)
print(f"Saved {OUT/'sp_ezz_comparison.png'}")
