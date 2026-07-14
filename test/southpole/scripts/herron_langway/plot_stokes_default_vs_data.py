"""Plot default Stokes vs H&L vs SP19 data — diagnostic comparison."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import firedrake as fd
from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.solvers.firn_solver import FirnColumnSolver
from firnpack.physics.densification import stokes_compressible, herron_langway
from firnpack.constants import year as YEAR_S

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
    sT1 = np.interp(yr1, bT["year_CE"].values, bT["temp"].values + 273.15)
    sA1 = np.interp(yr1, bA["year_CE"].values, bA["accum"].values)
    blk = int(SPINUP_DT)
    sT = np.array([sT1[i * blk:(i + 1) * blk].mean() for i in range(len(sT1) // blk)])
    sA = np.array([sA1[i * blk:(i + 1) * blk].mean() for i in range(len(sA1) // blk)])
    return sT, sA, eT, eA


def run_forward(label, params, rate_fn, sT, sA, eT, eA):
    print(f"\n--- {label} ---")
    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
    mesh.coordinates.assign(
        fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)

    model = FirnModel(params, densification_rate_fn=rate_fn)
    solver = FirnColumnSolver(model, surface_id=SID)

    H = fd.Function(V); rho = fd.Function(V); w = fd.Function(V); age = fd.Function(V)
    depth = H0 - xi
    T0, A0 = float(sT[0]), float(sA[0])
    H.assign(float(params.c_i) * (T0 - float(params.T_ref)))
    rho.interpolate(fd.Constant(RHO_SURF) +
        (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF)) *
        (1.0 - fd.exp(-depth / 20.0)))
    w.assign(-A0 * float(params.rho_i) / RHO_SURF / YEAR_S)
    age.assign(0.0)

    dt = mk(R, SPINUP_DT * YEAR_S, "dt")
    Ts = mk(R, T0, "Ts"); Hs = mk(R, float(params.c_i) * (T0 - float(params.T_ref)), "Hs")
    ac = mk(R, A0, "ac"); rs = mk(R, RHO_SURF, "rs")
    ws = mk(R, -A0 * float(params.rho_i) / RHO_SURF / YEAR_S, "ws")
    bc_H = fd.DirichletBC(V, Hs, SID)
    bc_rho = fd.DirichletBC(V, rs, SID)
    bc_w = fd.DirichletBC(V, ws, SID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)

    def step(Tk, Ak):
        Ts.assign(Tk); Hs.assign(float(params.c_i) * (Tk - float(params.T_ref)))
        ac.assign(Ak); ws.assign(-Ak * float(params.rho_i) / RHO_SURF / YEAR_S)
        solver.prognostic_solve(
            enthalpy=H, density=rho, firn_velocity=w, dt=dt,
            accumulation=ac, surface_density=rs,
            boundary_conditions=[bc_H, bc_rho, bc_w],
            surface_temperature=Ts, enthalpy_bc_constant=Hs,
            age=age, age_boundary_condition=bc_age)

    # Spinup
    dt.assign(SPINUP_DT * YEAR_S)
    for k in range(len(sT)):
        step(float(sT[k]), float(sA[k]))
    # ERA5
    dt.assign(ERA5_DT * YEAR_S)
    for k in range(len(eT)):
        step(float(eT[k]), float(eA[k]))

    coords = mesh.coordinates.dat.data_ro.flatten()
    idx = np.argsort(coords)
    d = H0 - coords[idx]
    rho_arr = rho.dat.data_ro[idx]
    age_arr = age.dat.data_ro[idx] / YEAR_S
    T_arr = H.dat.data_ro[idx] / float(params.c_i) + float(params.T_ref) - 273.15

    # Compute overburden stress profile
    rho_sorted = rho_arr  # top to bottom (d increases)
    g = 9.81
    sigma = np.zeros_like(d)
    for i in range(1, len(d)):
        dz = d[i] - d[i - 1]
        sigma[i] = sigma[i - 1] + 0.5 * (rho_sorted[i - 1] + rho_sorted[i]) * g * dz

    # Compute densification rate at each point (numerical)
    # drho/dt ≈ rho * 2A * (sigma/C)^n, but let's just compute it from the model
    import math
    _d = dict(
        n_glen=1.0, A_glen=6.8e-26, Q_glen=60.0e3, A_ref_T=248.15,
        c1_a=13.22240, c2_a=15.78652, c1_b=15.09371, c2_b=20.46489,
        E_lin=5.0e9,
    )
    # Override with params if stokes
    for attr in ["n_glen", "A_glen", "Q_glen", "A_ref_T", "c1_a", "c2_a", "c1_b", "c2_b", "E_lin"]:
        val = getattr(params, f"stokes_{attr}", None)
        if val is not None:
            _d[attr] = float(val)

    R_gas = 8.3144621
    rho_i = float(params.rho_i)
    n_glen = _d["n_glen"]
    A_T = _d["A_glen"] * np.exp(-_d["Q_glen"] / R_gas * (1.0 / (T_arr + 273.15) - 1.0 / _d["A_ref_T"]))
    if abs(n_glen - 1.0) < 0.01:
        A_eff = A_T * _d["E_lin"]
    else:
        A_eff = A_T

    rho_hat = rho_arr / rho_i
    # Stage 1 compressibility
    a_comp = np.exp(_d["c1_a"] - _d["c2_a"] * rho_hat)
    b_comp = np.exp(_d["c1_b"] - _d["c2_b"] * rho_hat)
    C_comp = 4.0 / (3.0 * a_comp) + 1.0 / b_comp
    eps_v = 2.0 * A_eff * (sigma / C_comp) ** n_glen
    drhodt = rho_arr * eps_v  # kg/m³/s
    drhodt_yr = drhodt * YEAR_S  # kg/m³/yr

    print(f"  rho: [{rho_arr.min():.0f}, {rho_arr.max():.0f}] kg/m³")
    print(f"  age: [{age_arr.min():.0f}, {age_arr.max():.0f}] yr")
    print(f"  T:   [{T_arr.min():.2f}, {T_arr.max():.2f}] °C")
    print(f"  sigma: [{sigma.min():.0f}, {sigma.max():.1f}] Pa")
    print(f"  drho/dt: [{drhodt_yr.min():.2f}, {drhodt_yr.max():.2f}] kg/m³/yr")
    return {"depth": d, "rho": rho_arr, "age_yr": age_arr, "T_C": T_arr,
            "sigma": sigma, "rho_hat": rho_hat, "a_comp": a_comp, "b_comp": b_comp,
            "C_comp": C_comp, "drhodt_yr": drhodt_yr, "A_eff": A_eff,
            "label": label}


if __name__ == "__main__":
    sT, sA, eT, eA = load_forcing()

    runs = []
    # H&L default
    runs.append(run_forward("H&L (default)", FirnParameters(),
                            herron_langway, sT, sA, eT, eA))
    # Stokes default
    runs.append(run_forward("Stokes n=1 (default)", FirnParameters(stokes_n_glen=1.0),
                            stokes_compressible, sT, sA, eT, eA))

    # Load observations
    df_rho = pd.read_csv(PROC / "sp19_density.csv")
    df_rho = df_rho[df_rho["depth_m"] <= H0]
    df_age = pd.read_csv(PROC / "sp19_depth_age.csv")
    df_age = df_age[(df_age["year_CE"] <= 2015) & (df_age["depth_m"] <= H0)]

    # ---- Plot ----
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))

    # Top row: profiles vs data
    colors = ["C0", "C1"]
    styles = ["--", "-"]
    lws = [2.0, 2.5]

    # Density
    ax = axes[0, 0]
    ax.plot(df_rho["rho_kgm3"].values * 1000, df_rho["depth_m"].values,
            "k.", ms=2, alpha=0.4, label="SP19")
    for r, c, s, lw in zip(runs, colors, styles, lws):
        ax.plot(r["rho"], r["depth"], color=c, ls=s, lw=lw, label=r["label"])
    ax.set_xlabel("Density (kg/m³)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_ylim(H0 + 5, -5)
    ax.set_title("Density Profile"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Age
    ax = axes[0, 1]
    ax.plot(2015.0 - df_age["year_CE"].values, df_age["depth_m"].values,
            "k.", ms=2, alpha=0.4, label="SP19")
    for r, c, s, lw in zip(runs, colors, styles, lws):
        ax.plot(r["age_yr"], r["depth"], color=c, ls=s, lw=lw, label=r["label"])
    ax.set_xlabel("Age (yr)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_ylim(H0 + 5, -5)
    ax.set_title("Age Profile"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Density residual
    ax = axes[0, 2]
    obs_d = df_rho["depth_m"].values
    obs_rho = df_rho["rho_kgm3"].values * 1000
    for r, c, s, lw in zip(runs, colors, styles, lws):
        pred = np.interp(obs_d, r["depth"], r["rho"])
        ax.plot(pred - obs_rho, obs_d, color=c, ls=s, lw=lw, label=r["label"], alpha=0.7)
    ax.axvline(0, color="k", ls=":", lw=0.5)
    ax.set_xlabel("ρ_pred − ρ_obs (kg/m³)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_ylim(H0 + 5, -5)
    ax.set_title("Density Residual"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Age residual
    ax = axes[0, 3]
    obs_age_d = df_age["depth_m"].values
    obs_age_yr = 2015.0 - df_age["year_CE"].values
    for r, c, s, lw in zip(runs, colors, styles, lws):
        pred = np.interp(obs_age_d, r["depth"], r["age_yr"])
        ax.plot(pred - obs_age_yr, obs_age_d, color=c, ls=s, lw=lw, label=r["label"], alpha=0.7)
    ax.axvline(0, color="k", ls=":", lw=0.5)
    ax.set_xlabel("age_pred − age_obs (yr)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_ylim(H0 + 5, -5)
    ax.set_title("Age Residual"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Bottom row: Stokes diagnostics
    r_stokes = runs[1]

    # Overburden stress
    ax = axes[1, 0]
    ax.plot(r_stokes["sigma"] / 1e3, r_stokes["depth"], "C1-", lw=2)
    ax.set_xlabel("Overburden stress σ (kPa)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_ylim(H0 + 5, -5)
    ax.set_title("Overburden Stress"); ax.grid(True, alpha=0.2)

    # Compressibility functions a, b, C
    ax = axes[1, 1]
    ax.semilogy(r_stokes["depth"], r_stokes["a_comp"], "C1-", lw=2, label="a(ρ̂)")
    ax.semilogy(r_stokes["depth"], r_stokes["b_comp"], "C2-", lw=2, label="b(ρ̂)")
    ax.semilogy(r_stokes["depth"], r_stokes["C_comp"], "C3--", lw=2, label="C = 4/(3a)+1/b")
    ax.set_xlabel("Depth (m)"); ax.set_ylabel("Compressibility")
    ax.set_title("Compressibility a, b, C vs Depth"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Densification rate
    ax = axes[1, 2]
    ax.plot(r_stokes["drhodt_yr"], r_stokes["depth"], "C1-", lw=2, label="Stokes")
    # Also compute H&L rate for comparison
    r_hl = runs[0]
    # H&L: approximate rate from profile difference
    ax.set_xlabel("dρ/dt (kg/m³/yr)"); ax.set_ylabel("Depth (m)")
    ax.invert_yaxis(); ax.set_ylim(H0 + 5, -5)
    ax.set_title("Densification Rate (Stokes)"); ax.grid(True, alpha=0.2)

    # Effective A vs depth
    ax = axes[1, 3]
    ax.semilogy(r_stokes["depth"], r_stokes["A_eff"], "C1-", lw=2)
    ax.set_xlabel("Depth (m)"); ax.set_ylabel("A_eff (Pa⁻¹ s⁻¹)")
    ax.set_title("Effective Rate Factor A(T)·E_lin"); ax.grid(True, alpha=0.2)

    fig.suptitle("Default Stokes n=1 vs H&L vs SP19 Data — Diagnostic", fontsize=14)
    fig.tight_layout()
    out = OUT / "stokes_default_vs_data_diagnostic.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
