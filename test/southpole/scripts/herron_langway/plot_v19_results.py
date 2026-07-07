"""Plot v19 Stokes density+T inversion results against SP19 data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import firedrake as fd
from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.physics.densification import stokes_compressible, herron_langway
from firn.constants import year as YEAR_S

_HERE = Path(__file__).parent
_SP = _HERE.parent.parent
PROC = _SP / "processed"
OUT = _SP / "results"
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF = 350.0
SPINUP_DT = 10.0
ERA5_DT = 0.5

def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f

def load_forcing():
    era5 = pd.read_csv(PROC / "era5_monthly_point.csv")
    tR = era5[era5["t2m_K"].notna()].groupby("year")["t2m_K"].mean()
    aR = era5[era5["net_accum_m_iceeq_month"].notna()].groupby("year")["net_accum_m_iceeq_month"].sum()
    yrs = sorted(tR.index.intersection(aR.index))
    mm = (np.array(yrs) >= 1940) & (np.array(yrs) < 2015)
    eT = np.repeat(tR.loc[yrs].values[mm], 2)
    eA = np.repeat((aR.loc[yrs].values[mm] + 0.0163), 2)
    bT = pd.read_csv(PROC / "buizert2021_spice_temp.csv").sort_values("year_CE")
    bA = pd.read_csv(PROC / "buizert2021_spice_accum.csv").sort_values("year_CE")
    yr1 = np.arange(740, 1940, 1.0)
    BUIZERT_T_OFFSET = 5.8  # match borehole deep T
    sT1 = np.interp(yr1, bT["year_CE"].values, bT["temp"].values + BUIZERT_T_OFFSET + 273.15)
    sA1 = np.interp(yr1, bA["year_CE"].values, bA["accum"].values)
    blk = int(SPINUP_DT)
    sT = np.array([sT1[i*blk:(i+1)*blk].mean() for i in range(len(sT1)//blk)])
    sA = np.array([sA1[i*blk:(i+1)*blk].mean() for i in range(len(sA1)//blk)])
    return sT, sA, eT, eA

def run_forward(label, params, rate_fn, sT, sA, eT, eA, use_mass_flux_bc=False):
    print(f"\n--- {label} {'(mass-flux BC)' if use_mass_flux_bc else ''} ---")
    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
    mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1); R = fd.FunctionSpace(mesh, "R", 0)
    model = FirnModel(params, densification_rate_fn=rate_fn)
    solver = FirnColumnSolver(model, surface_id=SID)
    H = fd.Function(V); rho = fd.Function(V); w = fd.Function(V); age = fd.Function(V)
    depth = H0 - xi
    T0, A0 = float(sT[0]), float(sA[0])
    H.assign(float(params.c_i) * (T0 - float(params.T_ref)))
    rho.interpolate(fd.Constant(RHO_SURF) + (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) * (1.0 - fd.exp(-depth / 20.0)))
    w.assign(-A0 / YEAR_S); age.assign(0.0)
    dt = mk(R, SPINUP_DT * YEAR_S, "dt")
    Ts = mk(R, T0, "Ts"); Hs = mk(R, float(params.c_i) * (T0 - float(params.T_ref)), "Hs")
    ac = mk(R, A0, "ac"); rs = mk(R, RHO_SURF, "rs")
    bc_H = fd.DirichletBC(V, Hs, SID); bc_rho = fd.DirichletBC(V, rs, SID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)
    if use_mass_flux_bc:
        wb = mk(R, -A0 / YEAR_S, "wb")
        bc_w_base = fd.DirichletBC(V, wb, 1)  # base boundary
        def step(Tk, Ak):
            Ts.assign(Tk); Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
            ac.assign(Ak); wb.assign(-Ak / YEAR_S)
            solver.prognostic_solve(enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=ac, surface_density=rs, boundary_conditions=[bc_H, bc_rho, None],
                surface_temperature=Ts, enthalpy_bc_constant=Hs, age=age,
                age_boundary_condition=bc_age, mass_flux_bc=bc_w_base)
    else:
        ws = mk(R, -A0 * float(params.rho_i) / RHO_SURF / YEAR_S, "ws")
        bc_w = fd.DirichletBC(V, ws, SID)
        def step(Tk, Ak):
            Ts.assign(Tk); Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
            ac.assign(Ak); ws.assign(-Ak * float(params.rho_i) / RHO_SURF / YEAR_S)
            solver.prognostic_solve(enthalpy=H, density=rho, firn_velocity=w, dt=dt,
                accumulation=ac, surface_density=rs, boundary_conditions=[bc_H, bc_rho, bc_w],
                surface_temperature=Ts, enthalpy_bc_constant=Hs, age=age,
                age_boundary_condition=bc_age)
    dt.assign(SPINUP_DT * YEAR_S)
    for k in range(len(sT)): step(float(sT[k]), float(sA[k]))
    dt.assign(ERA5_DT * YEAR_S)
    for k in range(len(eT)): step(float(eT[k]), float(eA[k]))
    coords = mesh.coordinates.dat.data_ro.flatten(); idx = np.argsort(coords)
    d = H0 - coords[idx]; rho_arr = rho.dat.data_ro[idx]
    age_arr = age.dat.data_ro[idx] / YEAR_S
    T_arr = H.dat.data_ro[idx] / float(params.c_i) + float(params.T_ref) - 273.15
    print(f"  rho: [{rho_arr.min():.0f}, {rho_arr.max():.0f}], age: [{age_arr.min():.0f}, {age_arr.max():.0f}], T: [{T_arr.min():.2f}, {T_arr.max():.2f}]")
    return {"depth": d, "rho": rho_arr, "age_yr": age_arr, "T_C": T_arr, "label": label}

if __name__ == "__main__":
    with open(OUT / "southpole_map_v19_stokes.json") as f:
        v19 = json.load(f)
    sT, sA, eT, eA = load_forcing()
    runs = []
    runs.append(run_forward("H&L (default)", FirnParameters(), herron_langway, sT, sA, eT, eA))
    runs.append(run_forward("Stokes (default)", FirnParameters(stokes_n_glen=1.0), stokes_compressible, sT, sA, eT, eA))
    m = v19["m_map"]
    params_map = FirnParameters(
        stokes_n_glen=1.0,
        stokes_A_glen=m.get("A_glen", 6.8e-26),
        stokes_E_lin=m.get("E_lin", 1.0),
        stokes_Q_glen=m["Q_glen"],
        stokes_c1_a=m["c1_a"], stokes_c2_a=m["c2_a"],
        stokes_c1_b=m["c1_b"], stokes_c2_b=m["c2_b"],
        stokes_rhohat_thres=m.get("rhohat_thres", 0.81),
        basal_heat_flux_W_m2=m.get("Q_base", 0.05),
        k_ice=m.get("k_ice", 2.1),
    )
    runs.append(run_forward("Stokes v19 MAP", params_map, stokes_compressible, sT, sA, eT, eA,
                            use_mass_flux_bc=True))

    df_rho = pd.read_csv(PROC / "sp19_density.csv"); df_rho = df_rho[df_rho["depth_m"] <= H0]
    df_age = pd.read_csv(PROC / "sp19_depth_age.csv")
    df_age = df_age[(df_age["year_CE"] <= 2015) & (df_age["depth_m"] <= H0)]
    df_T = pd.read_csv(PROC / "spicecore_borehole_T.csv"); df_T = df_T[df_T["depth_m"] <= H0]

    colors = ["C0", "C2", "C1"]; styles = ["--", ":", "-"]; lws = [1.5, 1.5, 2.5]
    fig, axes = plt.subplots(1, 4, figsize=(22, 7))

    ax = axes[0]
    ax.plot(df_rho["rho_kgm3"].values * 1000, df_rho["depth_m"].values, "k.", ms=2, alpha=0.4, label="SP19")
    for r, c, s, lw in zip(runs, colors, styles, lws):
        ax.plot(r["rho"], r["depth"], color=c, ls=s, lw=lw, label=r["label"])
    ax.set_xlabel("Density (kg/m³)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_ylim(H0+5, -5)
    ax.set_title("Density (in objective)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(2015.0 - df_age["year_CE"].values, df_age["depth_m"].values, "k.", ms=2, alpha=0.4, label="SP19")
    for r, c, s, lw in zip(runs, colors, styles, lws):
        ax.plot(r["age_yr"], r["depth"], color=c, ls=s, lw=lw, label=r["label"])
    ax.set_xlabel("Age (yr)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_ylim(H0+5, -5)
    ax.set_title("Age (in objective, relaxed σ)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax = axes[2]
    ax.plot(df_T["T_C"].values, df_T["depth_m"].values, "k.", ms=3, alpha=0.6, label="Borehole")
    for r, c, s, lw in zip(runs, colors, styles, lws):
        ax.plot(r["T_C"], r["depth"], color=c, ls=s, lw=lw, label=r["label"])
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_ylim(H0+5, -5)
    ax.set_title("Temperature (in objective)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax = axes[3]
    ax.plot(v19["J_hist"], "C1-o", ms=3)
    ax.set_xlabel("Iteration"); ax.set_ylabel("J")
    ax.set_title("Objective (rho + T)"); ax.grid(True, alpha=0.2)

    fig.suptitle(f"v19 Stokes Inversion (density + temperature): MAP vs Data\n"
                 f"Taylor test: 2.00 | J: {v19['J_hist'][0]:.2f} → {v19['J_map']:.2f} | "
                 f"8 controls | Q_base={m['Q_base']:.4f} W/m², k_ice={m['k_ice']:.3f} W/m/K",
                 fontsize=13)
    fig.tight_layout()
    out = OUT / "v19_stokes_results.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
