"""finalize_sp_assim.py — plot initial vs MAP firn profiles against all SP data.

Runs the forward model at the literature-initial and the assimilated-MAP H&L
parameters and overlays SP19 density, SP19 age, SPICE borehole T, and ApRES
velocity.  The ApRES velocity is an INDEPENDENT cross-check here: the MAP came
from assimilating density+age+temperature only, so agreement with ApRES
validates the recovered densification physics.
"""
from __future__ import annotations
import functools, math
from pathlib import Path
import numpy as np, pandas as pd
import firedrake as fd
from firedrake.adjoint import stop_annotating
from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.solvers.firn_solver import FirnColumnSolver
from firnpack.physics.densification import herron_langway as _hl
from firnpack.constants import year as YEAR_S
herron_langway = functools.partial(_hl, smooth=True)

PROC = Path(__file__).parent.parent.parent / "processed"
OUT = Path(__file__).parent.parent.parent / "results"
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, T_SURF_C, BDOT, DT_YEARS, SPIN_YEARS = 350.0, -45.5, 0.085, 5.0, 2000.0
N_ICE = math.sqrt(3.18)

import json, sys
INITIAL = dict(hl_k0=11.0, hl_k1=575.0, hl_Ea1=10160.0, hl_Ea2=21400.0, Q_base=0.0)
# MAP: from JSON arg if given, else the rho+age+T result.
MAP = dict(hl_k0=10.38, hl_k1=499.9, hl_Ea1=10279.0, hl_Ea2=21673.0, Q_base=0.0048)
if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
    MAP = json.load(open(sys.argv[1]))["m_map"]
    print(f"MAP from {sys.argv[1]}: {MAP}")

P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)

# data
dens = pd.read_csv(PROC/"sp19_density.csv"); dens = dens[dens.depth_m <= H0]
age = pd.read_csv(PROC/"sp19_depth_age.csv"); age["age_yr"] = 2015.0-age.year_CE
age = age[(age.depth_m <= H0) & (age.age_yr >= 0)]
bT = pd.read_csv(PROC/"spicecore_borehole_T.csv"); bT = bT[bT.depth_m <= H0]
ap = pd.read_csv(PROC/"apres_vertical_velocity_processed.csv")
ap = ap[(ap.range_m <= H0) & ap.v_smooth_m_yr.notna() & (ap.coherence > 0.5)]
_vb = np.arange(8.0, H0, 8.0); _vm = 0.5*(_vb[:-1]+_vb[1:])
ap["_b"] = np.digitize(ap.range_m.values, _vb)
vrows = [(_vm[i], ap.v_smooth_m_yr[ap._b == i+1].median(), int((ap._b == i+1).sum()))
         for i in range(len(_vm))]
vrows = [r for r in vrows if r[2] >= 3]
obs_v_d = np.array([r[0] for r in vrows]); obs_v = np.array([r[1] for r in vrows])

# mesh / model
mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0*(1.0-(1.0-fd.SpatialCoordinate(mesh)[0])**STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); R = fd.FunctionSpace(mesh, "R", 0)
depth = H0-fd.SpatialCoordinate(mesh)[0]
def mk(v, n): f = fd.Function(R, name=n); f.assign(float(v)); return f

def run(P):
    log = {}
    params = FirnParameters(
        hl_k0_prefactor=mk(P["hl_k0"], "k0"), hl_k1_prefactor=mk(P["hl_k1"], "k1"),
        hl_Ea_stage1=mk(P["hl_Ea1"], "e1"), hl_Ea_stage2=mk(P["hl_Ea2"], "e2"),
        basal_heat_flux_W_m2=mk(P["Q_base"], "Q"))
    model = FirnModel(params, densification_rate_fn=herron_langway)
    solver = FirnColumnSolver(model, surface_id=SID)
    T_surf_K = T_SURF_C+273.15; ws0 = -BDOT*rho_i/RHO_SURF/YEAR_S
    H_f = fd.Function(V); rho_f = fd.Function(V); w_f = fd.Function(V); age_f = fd.Function(V)
    H_f.assign(c_i*(T_surf_K-T_ref))
    rho_f.interpolate(RHO_SURF+(820.0-RHO_SURF)*(1.0-fd.exp(-depth/25.0)))
    w_f.assign(ws0); age_f.assign(0.0)
    Ts = mk(T_surf_K, "Ts"); Hs = mk(c_i*(T_surf_K-T_ref), "Hs")
    ac = mk(BDOT, "ac"); rs = mk(RHO_SURF, "rs"); ws = mk(ws0, "ws"); dt_r = mk(DT_YEARS*YEAR_S, "dt")
    bcs = [fd.DirichletBC(V, Hs, SID), fd.DirichletBC(V, rs, SID), fd.DirichletBC(V, ws, SID)]
    bc_age = fd.DirichletBC(V, mk(0.0, "a0"), SID)
    with stop_annotating():
        for _ in range(int(SPIN_YEARS/DT_YEARS)):
            solver.prognostic_solve(enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_r,
                accumulation=ac, surface_density=rs, boundary_conditions=bcs,
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age)
    xs = mesh.coordinates.dat.data_ro.reshape(-1); d = H0-xs; o = np.argsort(d)
    Tp = fd.Function(V).interpolate(H_f/c_i+T_ref)
    return dict(d=d[o], rho=rho_f.dat.data_ro[o], age=age_f.dat.data_ro[o]/YEAR_S,
                w=w_f.dat.data_ro[o]*YEAR_S, T=Tp.dat.data_ro[o]-273.15)

print("running initial..."); pi = run(INITIAL)
print("running MAP...");     pm = run(MAP)

def rms(pred_d, pred, obs_d, obs):
    p = np.interp(obs_d, pred_d, pred); return float(np.sqrt(np.mean((p-obs)**2)))
print("\n           RMS misfit   initial -> MAP")
print(f"  density   {rms(pi['d'],pi['rho'],dens.depth_m,dens.rho_kgm3*1000):6.1f} -> "
      f"{rms(pm['d'],pm['rho'],dens.depth_m,dens.rho_kgm3*1000):6.1f} kg/m3")
print(f"  age       {rms(pi['d'],pi['age'],age.depth_m,age.age_yr):6.1f} -> "
      f"{rms(pm['d'],pm['age'],age.depth_m,age.age_yr):6.1f} yr")
ws0_yr = -BDOT*rho_i/RHO_SURF
vi = np.interp(obs_v_d, pi['d'], (pi['w']-ws0_yr)/N_ICE)
vm = np.interp(obs_v_d, pm['d'], (pm['w']-ws0_yr)/N_ICE)
print(f"  velocity  {np.sqrt(np.mean((vi-obs_v)**2)):.4f} -> "
      f"{np.sqrt(np.mean((vm-obs_v)**2)):.4f} m/yr (ApRES, NOT assimilated)")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(18, 7), sharey=True)
ax[0].plot(dens.rho_kgm3*1000, dens.depth_m, "k.", ms=3, label="SP19")
ax[0].plot(pi["rho"], pi["d"], "C1--", lw=1.5, label="literature")
ax[0].plot(pm["rho"], pm["d"], "C0-", lw=2, label="MAP")
ax[0].set_xlabel("density (kg/m³)"); ax[0].set_ylabel("depth (m)"); ax[0].set_title("Density")
ax[1].plot(age.age_yr, age.depth_m, "k.", ms=2, label="SP19")
ax[1].plot(pi["age"], pi["d"], "C1--", lw=1.5); ax[1].plot(pm["age"], pm["d"], "C0-", lw=2)
ax[1].set_xlabel("age (yr)"); ax[1].set_title("Age")
ax[2].plot(obs_v, obs_v_d, "k.", ms=6, label="ApRES")
ax[2].plot((pi["w"]-ws0_yr)/N_ICE, pi["d"], "C1--", lw=1.5)
ax[2].plot((pm["w"]-ws0_yr)/N_ICE, pm["d"], "C0-", lw=2)
ax[2].set_xlabel("(w-w_s)/n_ice (m/yr)"); ax[2].set_title("Velocity (ApRES)")
ax[3].plot(bT.T_C, bT.depth_m, "k.", ms=3, label="SPICE")
ax[3].plot(pi["T"], pi["d"], "C1--", lw=1.5); ax[3].plot(pm["T"], pm["d"], "C0-", lw=2)
ax[3].set_xlabel("T (°C)"); ax[3].set_title("Temperature")
for a in ax: a.invert_yaxis(); a.grid(alpha=0.3); a.legend(fontsize=8)
fig.suptitle("South Pole firn assimilation — literature vs MAP (joint density+age+temperature+velocity)")
fig.tight_layout(); fig.savefig(OUT/"sp_assim_result.png", dpi=120)
print(f"\nSaved {OUT/'sp_assim_result.png'}")
