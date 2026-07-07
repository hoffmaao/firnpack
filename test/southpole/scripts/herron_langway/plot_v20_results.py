"""Plot v20 inversion results: run forward at MAP params, compare to data."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import firedrake as fd
from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.physics.densification import herron_langway
from firn.constants import year as YEAR_S

_HERE = Path(__file__).parent
_SP = _HERE.parent.parent
PROC = _SP / "processed"
OUT = _SP / "results"

H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF = 350.0
SPINUP_DT = 10.0
ERA5_DT = 0.5
BUIZERT_T_OFFSET = 5.8

def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f

def run_forward(label, m, plot_color, plot_style):
    """Run forward model with given parameters, return profiles."""
    print(f"\n--- {label} ---")
    for k, v in m.items():
        print(f"  {k} = {v:.4e}")
    params = FirnParameters(
        hl_k0_prefactor=m["hl_k0"], hl_k1_prefactor=m["hl_k1"],
        hl_Ea_stage1=m["hl_Ea1"], hl_Ea_stage2=m["hl_Ea2"],
        basal_heat_flux_W_m2=m.get("Q_base", 0.0),
    )
    model = FirnModel(params, densification_rate_fn=herron_langway)
    solver = FirnColumnSolver(model, surface_id=SID)

    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
    mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)

    H_f = fd.Function(V); rho_f = fd.Function(V)
    w_f = fd.Function(V); age_f = fd.Function(V)
    depth = H0 - xi

    # Constant forcing (mean Buizert with offset)
    bT = pd.read_csv(PROC / "buizert2021_spice_temp.csv").sort_values("year_CE")
    bA = pd.read_csv(PROC / "buizert2021_spice_accum.csv").sort_values("year_CE")
    T_mean_K = bT["temp"].values.mean() + BUIZERT_T_OFFSET + 273.15
    A_mean = bA["accum"].values.mean()

    Ts = mk(R, T_mean_K, "Ts")
    Hs = mk(R, float(params.c_i)*(T_mean_K-float(params.T_ref)), "Hs")
    ac = mk(R, A_mean, "ac")
    rs = mk(R, RHO_SURF, "rs")
    ws = mk(R, -A_mean*float(params.rho_i)/RHO_SURF/YEAR_S, "ws")
    dt = mk(R, 5.0 * YEAR_S, "dt")

    bc_H = fd.DirichletBC(V, Hs, SID)
    bc_rho = fd.DirichletBC(V, rs, SID)
    bc_w = fd.DirichletBC(V, ws, SID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)

    H_f.assign(float(params.c_i) * (T_mean_K - float(params.T_ref)))
    rho_f.interpolate(fd.Constant(RHO_SURF) +
        (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) *
        (1.0 - fd.exp(-depth / 20.0)))
    w_f.assign(-A_mean * float(params.rho_i) / RHO_SURF / YEAR_S)
    age_f.assign(0.0)

    n_steps = 300  # 1500 yr at dt=5 yr
    for k in range(n_steps):
        solver.prognostic_solve(
            enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt,
            accumulation=ac, surface_density=rs,
            boundary_conditions=[bc_H, bc_rho, bc_w],
            surface_temperature=Ts, enthalpy_bc_constant=Hs,
            age=age_f, age_boundary_condition=bc_age,
        )

    # Extract profiles
    coords = mesh.coordinates.dat.data_ro.flatten()
    idx = np.argsort(coords)
    d = H0 - coords[idx]
    rho_arr = rho_f.dat.data_ro[idx]
    age_arr = age_f.dat.data_ro[idx] / float(YEAR_S)
    T_arr = H_f.dat.data_ro[idx] / float(params.c_i) + float(params.T_ref) - 273.15

    print(f"  rho: [{rho_arr.min():.0f}, {rho_arr.max():.0f}] kg/m3")
    print(f"  age: [{age_arr.min():.0f}, {age_arr.max():.0f}] yr")
    print(f"  T:   [{T_arr.min():.2f}, {T_arr.max():.2f}] C")

    return {"depth": d, "rho": rho_arr, "age_yr": age_arr, "T_C": T_arr,
            "color": plot_color, "style": plot_style, "label": label}


if __name__ == "__main__":
    with open(OUT / "southpole_map_v20_hl.json") as f:
        res = json.load(f)

    runs = [
        run_forward("Initial", res["INITIAL"], "C0", "--"),
        run_forward("MAP", res["m_map"], "C1", "-"),
    ]

    # Load observations
    df_rho = pd.read_csv(PROC / "sp19_density.csv")
    df_rho = df_rho[df_rho["depth_m"] <= H0]

    df_age = pd.read_csv(PROC / "sp19_depth_age.csv")
    df_age = df_age[(df_age["year_CE"] <= 2015) & (df_age["depth_m"] <= H0)]

    df_T = pd.read_csv(PROC / "spicecore_borehole_T.csv")
    df_T = df_T[df_T["depth_m"] <= H0]

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 7))

    ax = axes[0]
    ax.plot(df_rho["rho_kgm3"].values * 1000, df_rho["depth_m"].values,
            "k.", ms=2, alpha=0.4, label="SP19")
    for r in runs:
        ax.plot(r["rho"], r["depth"], color=r["color"], ls=r["style"],
                lw=2, label=r["label"])
    ax.set_xlabel("Density (kg/m3)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_title("Density"); ax.legend(fontsize=9)
    ax.set_ylim(H0 + 5, -5); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(2015.0 - df_age["year_CE"].values, df_age["depth_m"].values,
            "k.", ms=2, alpha=0.4, label="SP19")
    for r in runs:
        ax.plot(r["age_yr"], r["depth"], color=r["color"], ls=r["style"],
                lw=2, label=r["label"])
    ax.set_xlabel("Age (yr)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_title("Age"); ax.legend(fontsize=9)
    ax.set_ylim(H0 + 5, -5); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(df_T["T_C"].values, df_T["depth_m"].values,
            "k.", ms=3, alpha=0.6, label="Borehole")
    for r in runs:
        ax.plot(r["T_C"], r["depth"], color=r["color"], ls=r["style"],
                lw=2, label=r["label"])
    ax.set_xlabel("Temperature (C)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_title("Temperature"); ax.legend(fontsize=9)
    ax.set_ylim(H0 + 5, -5); ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.semilogy(res["J_hist"], "C3-o", ms=3)
    ax.set_xlabel("Evaluation"); ax.set_ylabel("J")
    ax.set_title("Objective History"); ax.grid(True, alpha=0.3)

    fig.suptitle(f"v20 H&L Inversion (Buizert +{BUIZERT_T_OFFSET} C): Initial vs MAP vs Data",
                 fontsize=13)
    fig.tight_layout()
    out = OUT / "v20_results_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
