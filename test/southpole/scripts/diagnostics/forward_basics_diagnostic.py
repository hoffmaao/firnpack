"""forward_basics_diagnostic.py

START FROM BASICS — Evan Cummings (H, rho, w, age) firn column.

Run the forward model with CONSTANT literature forcing to STEADY STATE and
compare rho(z), age(z), w(z), T(z) against South Pole data (SP19 density,
SP19 depth-age, SPICE borehole T, ApRES vertical velocity).

No inversion, no pyadjoint tape, no time-varying forcing.  This isolates the
forward model's behaviour so we can see *what is actually happening* before
adding any assimilation machinery.

Key questions this answers:
  1. With literature params + correct T (~-45.5 C) + enough spinup, does the
     model reach the observed deep density (~844 kg/m3) and age (~1197 yr)?
  2. Is the deep density limited by physics, by spinup duration, or by the
     advection scheme (unstabilised CG vs DG upwind)?

Usage:
  python forward_basics_diagnostic.py [law] [space] [spin_years]
    law   = arthern | hl     (default hl)
    space = cg | dg          (default cg)  -- density/age function space
"""
from __future__ import annotations
import functools, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

import firedrake as fd
from firedrake.adjoint import stop_annotating  # only to be safe; not annotating
from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.solvers.firn_solver import FirnColumnSolver
from firnpack.physics.densification import (
    herron_langway as _hl, arthern_ligtenberg as _al)
from firnpack.constants import year as YEAR_S

_HERE = Path(__file__).parent
PROC = _HERE.parent.parent / "processed"
OUT = _HERE.parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- configuration ----------------
LAW = sys.argv[1] if len(sys.argv) > 1 else "hl"
SPACE = sys.argv[2] if len(sys.argv) > 2 else "cg"
SPIN_YEARS = float(sys.argv[3]) if len(sys.argv) > 3 else 4000.0

H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF = 350.0
T_SURF_C = -45.5            # near-isothermal borehole value
BDOT = 0.085               # m ice-eq / yr (SP literature ~0.08, ERA5 ~0.091)
DT_YEARS = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

print("=" * 64)
print(f"FORWARD BASICS DIAGNOSTIC: law={LAW}, space={SPACE}, "
      f"spin={SPIN_YEARS:.0f} yr")
print(f"  T_surf={T_SURF_C} C, bdot={BDOT} m ice/yr, rho_surf={RHO_SURF}")
print("=" * 64)

# ---------------- load data ----------------
dens = pd.read_csv(PROC / "sp19_density.csv")
dens = dens[dens.depth_m <= H0]
obs_rho_d = dens.depth_m.values
obs_rho = dens.rho_kgm3.values * 1000.0

age = pd.read_csv(PROC / "sp19_depth_age.csv")
age["age_yr"] = 2015.0 - age.year_CE
age = age[(age.depth_m <= H0) & (age.age_yr >= 0)]
obs_age_d = age.depth_m.values
obs_age = age.age_yr.values

bT = pd.read_csv(PROC / "spicecore_borehole_T.csv")
bT = bT[bT.depth_m <= H0]
obs_T_d = bT.depth_m.values
obs_T = bT.T_C.values

# ApRES: median smoothed vertical velocity per range bin, firn only
ap = pd.read_csv(PROC / "apres_vertical_velocity_processed.csv")
ap = ap[(ap.range_m <= H0) & ap.v_smooth_m_yr.notna()]
ap_bins = np.linspace(0, H0, 27)
ap_mid = 0.5 * (ap_bins[:-1] + ap_bins[1:])
ap["bin"] = np.digitize(ap.range_m.values, ap_bins)
ap_v = np.array([ap.v_smooth_m_yr[ap.bin == i + 1].median()
                 for i in range(len(ap_mid))])

# ---------------- model ----------------
P0 = FirnParameters()
c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)
print(f"  constants: c_i={c_i:.1f}, T_ref={T_ref:.2f} K, rho_i={rho_i:.1f}")

if LAW == "hl":
    dens_fn = functools.partial(_hl, smooth=True)
elif LAW == "arthern":
    dens_fn = functools.partial(_al, smooth=True)
else:
    raise SystemExit(f"unknown law {LAW}")

mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xi = fd.SpatialCoordinate(mesh)[0]
x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
mesh.coordinates.assign(
    fd.Function(mesh.coordinates.function_space()).interpolate(
        fd.as_vector([x_phys])))
V = fd.FunctionSpace(mesh, "CG", 1)
depth = H0 - fd.SpatialCoordinate(mesh)[0]

is_dg = (SPACE == "dg")
Vrho = fd.FunctionSpace(mesh, "DG", 0) if is_dg else V
Vage = fd.FunctionSpace(mesh, "DG", 0) if is_dg else V
R = fd.FunctionSpace(mesh, "R", 0)

T_surf_K = T_SURF_C + 273.15
H_f = fd.Function(V, name="H")
rho_f = fd.Function(Vrho, name="rho")
w_f = fd.Function(V, name="w")
age_f = fd.Function(Vage, name="age")

# initial conditions
H_f.assign(c_i * (T_surf_K - T_ref))
rho_f.interpolate(RHO_SURF + (rho_i - RHO_SURF) * (1.0 - fd.exp(-depth / 25.0)))
ws0 = -BDOT * rho_i / RHO_SURF / YEAR_S
w_f.assign(ws0)
age_f.assign(0.0)

params = FirnParameters()
model = FirnModel(params, densification_rate_fn=dens_fn)
solver = FirnColumnSolver(model, surface_id=SID)

def mk(v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f

Ts = mk(T_surf_K, "Ts")
Hs = mk(c_i * (T_surf_K - T_ref), "Hs")
ac = mk(BDOT, "ac")
rs = mk(RHO_SURF, "rs")
ws = mk(ws0, "ws")
dt_r = mk(DT_YEARS * YEAR_S, "dt")

bc_H = fd.DirichletBC(V, Hs, SID)
bc_age = fd.DirichletBC(Vage, mk(0.0, "a0"), SID) if not is_dg else None

# BC plumbing differs for CG vs DG density/age
if is_dg:
    # DG uses inflow values (upwind flux) instead of Dirichlet
    common = dict(
        enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_r,
        accumulation=ac, surface_density=rs,
        boundary_conditions=[bc_H, None, fd.DirichletBC(V, ws, SID)],
        surface_temperature=Ts, enthalpy_bc_constant=Hs,
        age=age_f, age_boundary_condition=None,
        inflow_values={"rho": rs},
    )
else:
    bc_rho = fd.DirichletBC(Vrho, rs, SID)
    bc_w = fd.DirichletBC(V, ws, SID)
    common = dict(
        enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_r,
        accumulation=ac, surface_density=rs,
        boundary_conditions=[bc_H, bc_rho, bc_w],
        surface_temperature=Ts, enthalpy_bc_constant=Hs,
        age=age_f, age_boundary_condition=bc_age,
    )

# ---------------- helpers ----------------
def profile(field):
    """Return (depth_sorted, value_sorted) for a CG1 or DG0 field."""
    Vf = field.function_space()
    if Vf.ufl_element().family() == "Discontinuous Lagrange":
        xc = fd.Function(fd.FunctionSpace(mesh, "DG", 0)).interpolate(
            fd.SpatialCoordinate(mesh)[0])
        xs = xc.dat.data_ro
    else:
        xs = mesh.coordinates.dat.data_ro.reshape(-1)
    d = H0 - xs
    v = field.dat.data_ro
    o = np.argsort(d)
    return d[o], v[o]

def at_depths(field, depths):
    d, v = profile(field)
    return np.interp(depths, d, v)

# ---------------- spinup loop ----------------
nsteps = int(SPIN_YEARS / DT_YEARS)
record = []
t0 = time.perf_counter()
with stop_annotating():
    for k in range(nsteps):
        solver.prognostic_solve(**common)
        if (k + 1) % 50 == 0 or k == nsteps - 1:
            yr = (k + 1) * DT_YEARS
            r120 = float(at_depths(rho_f, [120.0])[0])
            a120 = float(at_depths(age_f, [120.0])[0]) / YEAR_S
            rmax = float(rho_f.dat.data_ro.max())
            record.append((yr, r120, a120, rmax))
        if (k + 1) % 500 == 0:
            yr = (k + 1) * DT_YEARS
            print(f"  t={yr:6.0f} yr  rho(120m)={record[-1][1]:6.1f}  "
                  f"age(120m)={record[-1][2]:6.0f}  rho_max={record[-1][3]:6.1f}  "
                  f"({time.perf_counter()-t0:5.1f}s)")
elapsed = time.perf_counter() - t0
print(f"  spinup done: {nsteps} steps in {elapsed:.1f}s "
      f"({1000*elapsed/nsteps:.1f} ms/step)")

# ---------------- compare to data ----------------
T_f = fd.Function(V, name="T").interpolate(H_f / c_i + T_ref)
md, mrho = profile(rho_f)
mda, mage = profile(age_f)
mdw, mw = profile(w_f)
mdt, mT = profile(T_f)

print("\n--- MODEL vs DATA ---")
for z in [50, 100, 120, 127]:
    mr = float(at_depths(rho_f, [z])[0])
    dr = float(np.interp(z, obs_rho_d, obs_rho))
    ma = float(at_depths(age_f, [z])[0]) / YEAR_S
    da = float(np.interp(z, obs_age_d, obs_age))
    print(f"  z={z:3d}m: rho model={mr:6.1f} data={dr:6.1f} (d={mr-dr:+6.1f})  "
          f"| age model={ma:6.0f} data={da:6.0f} (d={ma-da:+6.0f})")
print(f"  rho_max model={mrho.max():.1f}  data_max={obs_rho.max():.1f}")
print(f"  age@base model={mage.max()/YEAR_S:.0f}  data@130m=1197")
w_surf = mw[np.argmin(mdw)] * YEAR_S   # m/yr
w_base = mw[np.argmax(mdw)] * YEAR_S   # m/yr
print(f"  w_surf model={w_surf:.4f} m/yr  expected={-BDOT*rho_i/RHO_SURF:.4f} m/yr")
print(f"  w_base model={w_base:.4f} m/yr  expected={-BDOT*rho_i/obs_rho.max():.4f} m/yr")

# converged?
r120_last = record[-1][1]
r120_prev = record[max(0, len(record) - 11)][1]
print(f"  convergence: rho(120m) changed {r120_last - r120_prev:+.2f} kg/m3 "
      f"over last ~500 yr")

# ---------------- plot ----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(18, 7), sharey=True)
    ax[0].plot(obs_rho, obs_rho_d, "k.", ms=3, label="SP19 data")
    ax[0].plot(mrho, md, "-", lw=2, label=f"model ({LAW},{SPACE})")
    ax[0].set_xlabel("density (kg/m³)"); ax[0].set_ylabel("depth (m)")
    ax[0].axvline(rho_i, ls=":", c="gray"); ax[0].legend(); ax[0].set_title("Density")

    ax[1].plot(obs_age, obs_age_d, "k.", ms=3, label="SP19 age")
    ax[1].plot(mage / YEAR_S, mda, "-", lw=2, label="model")
    ax[1].set_xlabel("age (yr)"); ax[1].legend(); ax[1].set_title("Age")

    mw_yr = mw * YEAR_S
    w_surf_yr = mw_yr[np.argmin(mdw)]
    ax[2].plot(mw_yr, mdw, "-", lw=2, label="model w")
    ax[2].plot(mw_yr - w_surf_yr, mdw, "--", lw=1.5, label="model w - w_surf")
    ax[2].plot(ap_v, ap_mid, "rs", ms=4, label="ApRES v_smooth")
    ax[2].set_xlabel("vertical velocity (m/yr)"); ax[2].legend()
    ax[2].set_title("Velocity")

    ax[3].plot(obs_T, obs_T_d, "k.", ms=3, label="SPICE borehole")
    ax[3].plot(mT - 273.15, mdt, "-", lw=2, label="model T")
    ax[3].set_xlabel("temperature (°C)"); ax[3].legend(); ax[3].set_title("Temperature")

    for a in ax:
        a.invert_yaxis(); a.grid(alpha=0.3)
    fig.suptitle(f"Forward basics: {LAW} densification, {SPACE} advection, "
                 f"{SPIN_YEARS:.0f} yr spinup, T={T_SURF_C}°C, bdot={BDOT}")
    fig.tight_layout()
    fn = OUT / f"forward_basics_{LAW}_{SPACE}.png"
    fig.savefig(fn, dpi=110)
    print(f"\nSaved {fn}")
except Exception as e:
    print(f"plot failed: {e}")

print("Done.")
