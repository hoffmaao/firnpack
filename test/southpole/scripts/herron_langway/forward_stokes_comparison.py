"""Compare all pluggable densification laws on a South Pole forward run.

Runs H&L, Arthern, and Stokes-compressible (n=1, n=3) on the same forcing.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import firedrake as fd
from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.physics.densification import (
    herron_langway,
    stokes_compressible,
)
from firn.constants import year as YEAR_S

_HERE = Path(__file__).parent
_SP = _HERE.parent.parent
PROC = _SP / "processed"
OUT = _SP / "results"

H0, NZ, STRETCH_P = 130.0, 100, 2.5
SID, BID = 2, 1
RHO_SURF = 350.0
DT_YEARS = 10.0


def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f


def run_forward(label, params, rate_fn, spin_T, spin_A):
    """Run a forward model with given densification law."""
    t0 = time.time()
    print(f"\n--- {label} ---")

    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
    mesh.coordinates.assign(
        fd.Function(mesh.coordinates.function_space()).interpolate(
            fd.as_vector([x_phys])
        )
    )
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)

    model = FirnModel(params, densification_rate_fn=rate_fn)
    solver = FirnColumnSolver(model, surface_id=SID)

    H = fd.Function(V, name="H")
    rho = fd.Function(V, name="rho")
    w = fd.Function(V, name="w")
    age = fd.Function(V, name="age")

    T0, A0 = float(spin_T[0]), float(spin_A[0])
    depth = H0 - xi
    H.assign(float(params.c_i) * (T0 - float(params.T_ref)))
    rho.interpolate(
        fd.Constant(RHO_SURF)
        + (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF))
        * (1.0 - fd.exp(-depth / 20.0))
    )
    w.assign(-A0 * float(params.rho_i) / RHO_SURF / YEAR_S)
    age.assign(0.0)

    dt_fn = mk(R, DT_YEARS * YEAR_S, "dt")
    Ts = mk(R, T0, "Ts")
    Hs = mk(R, float(params.c_i) * (T0 - float(params.T_ref)), "Hs")
    ac = mk(R, A0, "ac")
    rs = mk(R, RHO_SURF, "rs")
    ws = mk(R, -A0 * float(params.rho_i) / RHO_SURF / YEAR_S, "ws")

    bc_H = fd.DirichletBC(V, Hs, SID)
    bc_rho = fd.DirichletBC(V, rs, SID)
    bc_w = fd.DirichletBC(V, ws, SID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)

    for k in range(len(spin_T)):
        Tk, Ak = float(spin_T[k]), float(spin_A[k])
        Ts.assign(Tk)
        Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
        ac.assign(Ak)
        ws.assign(-Ak * float(params.rho_i) / RHO_SURF / YEAR_S)
        solver.prognostic_solve(
            enthalpy=H, density=rho, firn_velocity=w, dt=dt_fn,
            accumulation=ac, surface_density=rs,
            boundary_conditions=[bc_H, bc_rho, bc_w],
            surface_temperature=Ts, enthalpy_bc_constant=Hs,
            age=age, age_boundary_condition=bc_age,
        )

    elapsed = time.time() - t0
    coords = mesh.coordinates.dat.data_ro.flatten()
    idx = np.argsort(coords)
    d = H0 - coords[idx]
    rho_arr = rho.dat.data_ro[idx]
    age_arr = age.dat.data_ro[idx] / YEAR_S
    T_arr = H.dat.data_ro[idx] / float(params.c_i) + float(params.T_ref) - 273.15

    print(f"  Time: {elapsed:.0f}s")
    print(f"  rho: [{rho_arr.min():.0f}, {rho_arr.max():.0f}] kg/m³")
    print(f"  age: [{age_arr.min():.0f}, {age_arr.max():.0f}] yr")
    return {"label": label, "depth": d, "rho": rho_arr, "age_yr": age_arr, "T_C": T_arr}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    # Forcing: Buizert 740–1940 CE
    bT = pd.read_csv(PROC / "buizert2021_spice_temp.csv").sort_values("year_CE")
    bA = pd.read_csv(PROC / "buizert2021_spice_accum.csv").sort_values("year_CE")
    yr1 = np.arange(740, 1940, 1.0)
    sT1 = np.interp(yr1, bT["year_CE"].values, bT["temp"].values + 273.15)
    sA1 = np.interp(yr1, bA["year_CE"].values, bA["accum"].values)
    blk = int(DT_YEARS)
    sT = np.array([sT1[i * blk:(i + 1) * blk].mean() for i in range(len(sT1) // blk)])
    sA = np.array([sA1[i * blk:(i + 1) * blk].mean() for i in range(len(sA1) // blk)])
    print(f"Forcing: 740-1940 CE, {len(sT)} steps × dt={DT_YEARS}yr")

    # Observations
    df_rho = pd.read_csv(PROC / "sp19_density.csv")
    df_rho = df_rho[df_rho["depth_m"] <= H0]

    # --- Run each law ---
    runs = []

    # 1) Herron & Langway (default)
    runs.append(run_forward(
        "H&L (default)",
        FirnParameters(),
        herron_langway,
        sT, sA,
    ))

    # 2) Stokes n=1 (linear)
    params_s1 = FirnParameters(stokes_n_glen=1.0)
    runs.append(run_forward(
        "Stokes n=1",
        params_s1,
        stokes_compressible,
        sT, sA,
    ))

    # 3) Stokes n=3 (standard Glen) — needs smaller dt; skip if unstable
    params_s3 = FirnParameters(stokes_n_glen=3.0, stokes_E_lin=1.0)
    try:
        runs.append(run_forward(
            "Stokes n=3",
            params_s3,
            stokes_compressible,
            sT, sA,
        ))
    except Exception as e:
        print(f"  Stokes n=3 failed: {e}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    colors = ["C0", "C1", "C2"]
    styles = ["-", "--", ":"]

    ax = axes[0]
    ax.plot(df_rho["rho_kgm3"].values * 1000, df_rho["depth_m"].values,
            "k.", ms=2, alpha=0.4, label="SP19")
    for r, c, s in zip(runs, colors, styles):
        ax.plot(r["rho"], r["depth"], color=c, ls=s, lw=2, label=r["label"])
    ax.set_xlabel("Density (kg/m³)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title("Density")
    ax.legend(fontsize=8)
    ax.set_ylim(H0 + 5, -5)

    ax = axes[1]
    for r, c, s in zip(runs, colors, styles):
        ax.plot(r["age_yr"], r["depth"], color=c, ls=s, lw=2, label=r["label"])
    ax.set_xlabel("Age (yr)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title("Age")
    ax.legend(fontsize=8)
    ax.set_ylim(H0 + 5, -5)

    ax = axes[2]
    for r, c, s in zip(runs, colors, styles):
        ax.plot(r["T_C"], r["depth"], color=c, ls=s, lw=2, label=r["label"])
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title("Temperature")
    ax.legend(fontsize=8)
    ax.set_ylim(H0 + 5, -5)

    fig.suptitle("Densification Law Comparison: South Pole", fontsize=13)
    fig.tight_layout()
    out = OUT / "forward_stokes_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
