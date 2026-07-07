"""plot_paleo_result.py — plot the paleothermometry result.

Reads sp_paleo_thermal.json (k_factor + surface-T knots), runs the forward at the
MAP (variable conductivity + recovered surface-T history) and at a baseline
(default conductivity, constant T), and plots the borehole-T fit + the recovered
surface-temperature history.
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
R = json.load(open(OUT/"sp_paleo_thermal.json"))
kf_map = R["k_factor"]; Tk = np.array(R["T_knots"]); KY = np.array(R["knot_years"])
print(f"MAP: k_factor={kf_map:.3f}, T_knots={np.round(Tk,3)}")

H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, BDOT, DT_YEARS, SPIN_YEARS = 350.0, 0.085, 5.0, 2500.0
K_ICE_BASE = 2.1; BETA = 5.0e2
DENS = dict(hl_k0=10.79, hl_k1=570.7, hl_Ea1=10432.0, hl_Ea2=21875.0)
P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)
n_steps = int(SPIN_YEARS/DT_YEARS)
step_years = 2015.0 - (n_steps-1-np.arange(n_steps))*DT_YEARS
T_hist = np.interp(step_years, KY, Tk)   # MAP surface-T history on the step grid

bT = pd.read_csv(PROC/"spicecore_borehole_T.csv"); bT = bT[bT.depth_m <= H0]

mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0*(1.0-(1.0-fd.SpatialCoordinate(mesh)[0])**STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); R_ = fd.FunctionSpace(mesh, "R", 0)
dx = fd.dx(domain=mesh); ds = fd.ds(domain=mesh)
depth = H0-fd.SpatialCoordinate(mesh)[0]; psi = fd.TestFunction(V)
def mk(v, n): f = fd.Function(R_, name=n); f.assign(float(v)); return f

def run(k_factor, T_series):
    params = FirnParameters(
        hl_k0_prefactor=mk(DENS["hl_k0"], "k0"), hl_k1_prefactor=mk(DENS["hl_k1"], "k1"),
        hl_Ea_stage1=mk(DENS["hl_Ea1"], "e1"), hl_Ea_stage2=mk(DENS["hl_Ea2"], "e2"),
        k_ice=K_ICE_BASE*k_factor, basal_heat_flux_W_m2=mk(0.0, "Q"))
    model = FirnModel(params, densification_rate_fn=herron_langway)
    ws0 = -BDOT*rho_i/RHO_SURF/YEAR_S
    H_f=fd.Function(V); rho_f=fd.Function(V); w_f=fd.Function(V); age_f=fd.Function(V)
    H_o=fd.Function(V); rho_o=fd.Function(V); w_o=fd.Function(V); age_o=fd.Function(V)
    Ts_eff=fd.Function(V)
    T0=float(T_series.mean())+273.15
    H_f.assign(c_i*(T0-T_ref)); H_o.assign(H_f)
    rho_f.interpolate(RHO_SURF+(820.0-RHO_SURF)*(1.0-fd.exp(-depth/25.0))); rho_o.assign(rho_f)
    w_f.assign(ws0); w_o.assign(w_f); age_f.assign(0.0); age_o.assign(0.0)
    ac=mk(BDOT,"ac"); rs=mk(RHO_SURF,"rs"); ws=mk(ws0,"ws"); dt_r=mk(DT_YEARS*YEAR_S,"dt")
    bc_rho=fd.DirichletBC(V,rs,SID); bc_w=fd.DirichletBC(V,ws,SID); bc_age=fd.DirichletBC(V,mk(0.0,"a0"),SID)
    bdot_mass=ac*rho_i/P0.spy
    Ht=fd.TrialFunction(V); rt=fd.TrialFunction(V); wt=fd.TrialFunction(V); at=fd.TrialFunction(V)
    sp={"ksp_type":"preonly","pc_type":"lu"}
    F_H=model.enthalpy_form(Ht,H_o,rho_f,w_f,psi,dt_r)+fd.Constant(BETA)*(Ht-c_i*(Ts_eff-fd.Constant(T_ref)))*psi*ds(SID)
    sH=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(F_H),fd.rhs(F_H),H_f,constant_jacobian=False),solver_parameters=sp)
    Te=model.temperature_from_enthalpy(H_f)
    F_rho,_=model.density_form(rt,rho_o,Te,w_f,None,bdot_mass,psi,dt_r)
    sR=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(F_rho),fd.rhs(F_rho),rho_f,bcs=[bc_rho],constant_jacobian=False),solver_parameters=sp)
    dr=herron_langway(rho_f,Te,params=params,bdot=bdot_mass)
    dw=model.velocity_delta(wt,w_o,rho_f,dr,psi,regularization=1e-3)
    sW=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(dw),fd.rhs(dw),w_f,bcs=[bc_w],constant_jacobian=False),solver_parameters=sp)
    F_age=model.age_form(at,age_o,w_f,w_o,psi,dt_r)
    sA=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(F_age),fd.rhs(F_age),age_f,bcs=[bc_age],constant_jacobian=False),solver_parameters=sp)
    with stop_annotating():
        for k in range(n_steps):
            Ts_eff.interpolate(fd.Constant(T_series[k]+273.15))
            sH.solve(); sR.solve(); sW.solve(); sA.solve()
            H_o.assign(H_f); rho_o.assign(rho_f); w_o.assign(w_f); age_o.assign(age_f)
    xs=mesh.coordinates.dat.data_ro.reshape(-1); d=H0-xs; o=np.argsort(d)
    Tp=fd.Function(V).interpolate(H_f/c_i+T_ref)
    return d[o], Tp.dat.data_ro[o]-273.15

print("running baseline (k=1, const -45.4)...")
db, Tb = run(1.0, np.full(n_steps, -45.4))
print("running MAP (variable k + recovered history)...")
dm, Tm = run(kf_map, T_hist)
def rms(d, T): return float(np.sqrt(np.mean((np.interp(bT.depth_m, d, T)-bT.T_C)**2)))
print(f"\nborehole-T RMS: baseline {rms(db,Tb):.3f}C -> MAP {rms(dm,Tm):.3f}C")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 7))
ax[0].plot(bT.T_C, bT.depth_m, "k.", ms=5, label="SPICE borehole")
ax[0].plot(Tb, db, "C1--", lw=1.5, label="baseline (k=2.1, const T)")
ax[0].plot(Tm, dm, "C0-", lw=2, label=f"MAP (k_ice={kf_map*K_ICE_BASE:.2f}, paleo-T)")
ax[0].invert_yaxis(); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
ax[0].set_xlabel("T (°C)"); ax[0].set_ylabel("depth (m)"); ax[0].set_title("Borehole temperature fit")
ax[1].plot(Tk, KY, "C0-o", lw=2)
ax[1].set_xlabel("surface T (°C)"); ax[1].set_ylabel("year CE")
ax[1].set_title("Recovered surface-T history"); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"sp_paleo_result.png", dpi=120)
print(f"Saved {OUT/'sp_paleo_result.png'}")
