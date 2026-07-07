"""forward_paleo_test.py

De-risk paleothermometry: can the coupled advection-diffusion thermal model map
a PAST surface warm period to a sub-surface temperature bump like the SPICE
borehole (warmest ~58 m, coldest ~120 m)?  Prescribe a synthetic surface-T
history (baseline + a Gaussian warm period at year Y0 + a recent component) and
compare the resulting depth profile to the borehole.  No inversion — just tests
whether the structure is reproducible, which decides if the inversion is worth
building.
"""
from __future__ import annotations
import functools
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
RHO_SURF, BDOT, DT_YEARS, SPIN_YEARS = 350.0, 0.085, 5.0, 2500.0
P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)
n_steps = int(SPIN_YEARS/DT_YEARS)
years = 2015.0 - (n_steps-1-np.arange(n_steps))*DT_YEARS

# synthetic surface-T history: baseline + past warm period + recent warming
def Thist(T_base, warm_amp, warm_yr, warm_w, recent_amp):
    g = warm_amp*np.exp(-((years-warm_yr)/warm_w)**2)
    rec = recent_amp*np.clip((years-1850.0)/165.0, 0, 1)
    return T_base + g + rec

bTd = pd.read_csv(PROC/"spicecore_borehole_T.csv"); bTd = bTd[bTd.depth_m <= H0]

mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0*(1.0-(1.0-fd.SpatialCoordinate(mesh)[0])**STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); R = fd.FunctionSpace(mesh, "R", 0)
depth = H0-fd.SpatialCoordinate(mesh)[0]
def mk(v, n): f = fd.Function(R, name=n); f.assign(float(v)); return f

def run(T_series, k_ice=2.1):
    model = FirnModel(FirnParameters(basal_heat_flux_W_m2=mk(0.0, "Q"), k_ice=k_ice),
                      densification_rate_fn=herron_langway)
    solver = FirnColumnSolver(model, surface_id=SID)
    ws0 = -BDOT*rho_i/RHO_SURF/YEAR_S
    H_f = fd.Function(V); rho_f = fd.Function(V); w_f = fd.Function(V); age_f = fd.Function(V)
    T0 = T_series[0]+273.15
    H_f.assign(c_i*(T0-T_ref))
    rho_f.interpolate(RHO_SURF+(820.0-RHO_SURF)*(1.0-fd.exp(-depth/25.0)))
    w_f.assign(ws0); age_f.assign(0.0)
    Ts = mk(T0, "Ts"); Hs = mk(c_i*(T0-T_ref), "Hs"); ac = mk(BDOT, "ac")
    rs = mk(RHO_SURF, "rs"); ws = mk(ws0, "ws"); dt_r = mk(DT_YEARS*YEAR_S, "dt")
    bcs = [fd.DirichletBC(V, Hs, SID), fd.DirichletBC(V, rs, SID), fd.DirichletBC(V, ws, SID)]
    bc_age = fd.DirichletBC(V, mk(0.0, "a0"), SID)
    with stop_annotating():
        for k in range(n_steps):
            Tk = T_series[k]+273.15
            Ts.assign(Tk); Hs.assign(c_i*(Tk-T_ref))
            solver.prognostic_solve(enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_r,
                accumulation=ac, surface_density=rs, boundary_conditions=bcs,
                surface_temperature=Ts, enthalpy_bc_constant=Hs,
                age=age_f, age_boundary_condition=bc_age)
    xs = mesh.coordinates.dat.data_ro.reshape(-1); d = H0-xs; o = np.argsort(d)
    Tp = fd.Function(V).interpolate(H_f/c_i+T_ref)
    return d[o], Tp.dat.data_ro[o]-273.15

# Fixed warm@1600 history; vary firn thermal conductivity to see if lower
# conductivity (less damping) preserves the paleoclimate structure.
Th = Thist(-45.30, 0.9, 1600, 130, 0.25)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 7))
ax[0].plot(bTd.T_C, bTd.depth_m, "k.", ms=5, label="SPICE borehole")
for k_ice, lbl in [(2.1, "k_ice=2.1 (default)"), (2.1/3, "k_ice/3"), (2.1/8, "k_ice/8")]:
    print(f"running {lbl} ...")
    d, T = run(Th, k_ice=k_ice)
    ax[0].plot(T, d, "-", lw=1.8, label=lbl)
ax[1].plot(Th, years, "C3", lw=1.5)
ax[0].invert_yaxis(); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
ax[0].set_xlabel("T (°C)"); ax[0].set_ylabel("depth (m)")
ax[0].set_title("Effect of firn conductivity (warm@1600 history)")
ax[1].set_xlabel("surface T (°C)"); ax[1].set_ylabel("year CE"); ax[1].set_title("Prescribed surface-T history")
ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"forward_paleo_conductivity.png", dpi=120)
print(f"Saved {OUT/'forward_paleo_conductivity.png'}")
