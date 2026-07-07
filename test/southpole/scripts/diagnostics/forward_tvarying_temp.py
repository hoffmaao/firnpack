"""forward_tvarying_temp.py

Test whether TIME-VARYING surface temperature forcing reproduces the observed
borehole temperature structure (warm shallow, cold deep) that constant forcing
cannot.  Drives the Cummings (H,rho,w,age) column with the Buizert (740-1940,
+5.8 cloud->surface offset) + ERA5 (1940-2015) surface-temperature history,
leveled so the long-term mean matches the borehole deep T.

Both the thermal equation AND the densification see the time-varying T (physically
correct).  Plots T, density, age vs data, comparing constant vs time-varying.
"""
from __future__ import annotations
import functools, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import firedrake as fd
from firedrake.adjoint import stop_annotating
from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.physics.densification import herron_langway as _hl
from firn.constants import year as YEAR_S
herron_langway = functools.partial(_hl, smooth=True)

PROC = Path(__file__).parent.parent.parent / "processed"
OUT = Path(__file__).parent.parent.parent / "results"
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, BDOT, DT_YEARS, SPIN_YEARS = 350.0, 0.085, 5.0, 2000.0
BUIZERT_OFFSET = 5.8
T_LEVEL_C = -45.45   # long-term mean to match borehole deep T

P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)
n_steps = int(SPIN_YEARS / DT_YEARS)
end_year = 2015.0
years = end_year - (n_steps - 1 - np.arange(n_steps)) * DT_YEARS   # step -> calendar year

# ---- build surface-T history on the step timeline ----
bT = pd.read_csv(PROC/"buizert2021_spice_temp.csv").sort_values("year_CE")
e = pd.read_csv(PROC/"era5_monthly_point.csv")
eT = (e[e.t2m_K.notna()].groupby("year").t2m_K.mean() - 273.15)
def surfT(yr):
    if yr >= 1940 and yr <= eT.index.max():
        return float(np.interp(yr, eT.index.values, eT.values))
    return float(np.interp(yr, bT.year_CE.values, bT.temp.values)) + BUIZERT_OFFSET
T_raw = np.array([surfT(y) for y in years])
# level so the mean over the thermally-relevant recent window matches T_LEVEL_C
T_hist = T_raw - T_raw.mean() + T_LEVEL_C
print(f"surface-T history: {years[0]:.0f}-{years[-1]:.0f} CE, "
      f"range [{T_hist.min():.2f},{T_hist.max():.2f}]C, "
      f"recent(>=1980) mean {T_hist[years>=1980].mean():.2f}C, "
      f"old(<1500) mean {T_hist[years<1500].mean():.2f}C")

# ---- data ----
dens = pd.read_csv(PROC/"sp19_density.csv"); dens = dens[dens.depth_m <= H0]
age = pd.read_csv(PROC/"sp19_depth_age.csv"); age["age_yr"] = 2015.0-age.year_CE
age = age[(age.depth_m <= H0) & (age.age_yr >= 0)]
bTd = pd.read_csv(PROC/"spicecore_borehole_T.csv"); bTd = bTd[bTd.depth_m <= H0]

# ---- mesh / model ----
mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0*(1.0-(1.0-fd.SpatialCoordinate(mesh)[0])**STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); R = fd.FunctionSpace(mesh, "R", 0)
depth = H0-fd.SpatialCoordinate(mesh)[0]
def mk(v, n): f = fd.Function(R, name=n); f.assign(float(v)); return f

def run(T_series):
    """T_series: array of surface T (C) per step, or None for constant T_LEVEL_C."""
    params = FirnParameters(basal_heat_flux_W_m2=mk(0.0, "Q"))
    model = FirnModel(params, densification_rate_fn=herron_langway)
    solver = FirnColumnSolver(model, surface_id=SID)
    ws0 = -BDOT*rho_i/RHO_SURF/YEAR_S
    H_f = fd.Function(V); rho_f = fd.Function(V); w_f = fd.Function(V); age_f = fd.Function(V)
    T0 = T_LEVEL_C+273.15
    H_f.assign(c_i*(T0-T_ref))
    rho_f.interpolate(RHO_SURF+(820.0-RHO_SURF)*(1.0-fd.exp(-depth/25.0)))
    w_f.assign(ws0); age_f.assign(0.0)
    Ts = mk(T0, "Ts"); Hs = mk(c_i*(T0-T_ref), "Hs"); ac = mk(BDOT, "ac")
    rs = mk(RHO_SURF, "rs"); ws = mk(ws0, "ws"); dt_r = mk(DT_YEARS*YEAR_S, "dt")
    bcs = [fd.DirichletBC(V, Hs, SID), fd.DirichletBC(V, rs, SID), fd.DirichletBC(V, ws, SID)]
    bc_age = fd.DirichletBC(V, mk(0.0, "a0"), SID)
    with stop_annotating():
        for k in range(n_steps):
            Tk = (T_LEVEL_C if T_series is None else T_series[k]) + 273.15
            Ts.assign(Tk); Hs.assign(c_i*(Tk-T_ref))
            solver.prognostic_solve(enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_r,
                accumulation=ac, surface_density=rs, boundary_conditions=bcs,
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age)
    xs = mesh.coordinates.dat.data_ro.reshape(-1); d = H0-xs; o = np.argsort(d)
    Tp = fd.Function(V).interpolate(H_f/c_i+T_ref)
    return dict(d=d[o], rho=rho_f.dat.data_ro[o], age=age_f.dat.data_ro[o]/YEAR_S,
                T=Tp.dat.data_ro[o]-273.15)

print("running constant-T..."); pc = run(None)
print("running time-varying-T..."); pv = run(T_hist)

def rms(p, pred, od, ov): return float(np.sqrt(np.mean((np.interp(od, p['d'], pred)-ov)**2)))
print(f"\nT RMS:   constant {rms(pc,pc['T'],bTd.depth_m,bTd.T_C):.3f}C  ->  "
      f"time-varying {rms(pv,pv['T'],bTd.depth_m,bTd.T_C):.3f}C")
print(f"rho RMS: constant {rms(pc,pc['rho'],dens.depth_m,dens.rho_kgm3*1000):.1f}  ->  "
      f"time-varying {rms(pv,pv['rho'],dens.depth_m,dens.rho_kgm3*1000):.1f} kg/m3")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
ax[0].plot(bTd.T_C, bTd.depth_m, "k.", ms=4, label="SPICE borehole")
ax[0].plot(pc["T"], pc["d"], "C1--", lw=1.5, label="constant T")
ax[0].plot(pv["T"], pv["d"], "C0-", lw=2, label="time-varying T")
ax[0].set_xlabel("T (°C)"); ax[0].set_ylabel("depth (m)"); ax[0].set_title("Temperature"); ax[0].legend()
ax[1].plot(dens.rho_kgm3*1000, dens.depth_m, "k.", ms=3, label="SP19")
ax[1].plot(pc["rho"], pc["d"], "C1--", lw=1.5); ax[1].plot(pv["rho"], pv["d"], "C0-", lw=2)
ax[1].set_xlabel("density (kg/m³)"); ax[1].set_title("Density")
ax[2].plot(age.age_yr, age.depth_m, "k.", ms=2, label="SP19")
ax[2].plot(pc["age"], pc["d"], "C1--", lw=1.5); ax[2].plot(pv["age"], pv["d"], "C0-", lw=2)
ax[2].set_xlabel("age (yr)"); ax[2].set_title("Age")
for a in ax: a.invert_yaxis(); a.grid(alpha=0.3)
fig.suptitle("Constant vs time-varying surface-T forcing (literature H&L)")
fig.tight_layout(); fig.savefig(OUT/"forward_tvarying_temp.png", dpi=120)
print(f"Saved {OUT/'forward_tvarying_temp.png'}")
