"""Run FirnMICE experiments with Stokes compressible at varying Glen exponents.

Compares n=1, 1.8, 3, 4 alongside H&L as a reference.

Usage:
    OMP_NUM_THREADS=1 python test/firnmice/firnmice_stokes_n_sweep.py
"""
from __future__ import annotations
import os, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import firedrake as fd
from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.physics.densification import herron_langway, stokes_compressible, calibrate_A_glen
from firn.constants import year as YEAR_S

OUT = Path(os.environ.get("FIRNMICE_OUT",
           str(Path(__file__).parent / "results")))

@dataclass(frozen=True)
class Experiment:
    name: str
    T0_C: float
    a0: float
    dT: float = 0.0
    da: float = 0.0
    t_step: float = 100.0

EXPERIMENTS = [
    Experiment("ex1", T0_C=-50.0, a0=0.10, dT=+5.0),
    Experiment("ex2", T0_C=-40.0, a0=0.10, dT=+5.0),
    Experiment("ex3", T0_C=-30.0, a0=0.10, dT=+5.0),
    Experiment("ex4", T0_C=-30.0, a0=0.02, da=+0.05),
    Experiment("ex5", T0_C=-30.0, a0=0.15, da=+0.05),
    Experiment("ex6", T0_C=-30.0, a0=0.25, da=+0.05),
]

H0 = 200.0
NZ = 120
STRETCH_P = 3.0
DT_YR = 1.0
SPINUP_YR = int(os.environ.get("FIRNMICE_SPINUP", "500"))
RUN_YR = int(os.environ.get("FIRNMICE_RUN", "500"))
RHO_SURF = 360.0
SURFACE_ID = 2

# Sweep: H&L reference + Stokes at different n with calibrated A_glen
N_VALUES = [1.0, 1.8, 3.0]
LAWS = {"H&L": (herron_langway, FirnParameters())}
for n in N_VALUES:
    if abs(n - 1.0) < 0.01:
        # n=1: use default A_glen with E_lin enhancement
        p = FirnParameters(stokes_n_glen=n)
    else:
        # n>1: calibrate A_glen to match n=1 reference rate at 200 kPa
        A_cal = calibrate_A_glen(n)
        p = FirnParameters(stokes_n_glen=n, stokes_A_glen=A_cal, stokes_E_lin=1.0)
        print(f"  n={n}: calibrated A_glen={A_cal:.4e}")
    LAWS[f"Stokes n={n:g}"] = (stokes_compressible, p)


def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f


def compute_dip(rho, params, mesh):
    dx = fd.dx(domain=mesh)
    phi = 1.0 - rho / params.rho_i
    return float(fd.assemble(fd.max_value(phi, 0.0) * dx))


def compute_bco(mesh, rho, age, rho_bco=815.0):
    coords = mesh.coordinates.dat.data_ro.flatten()
    idx = np.argsort(coords)[::-1]
    d = coords.max() - coords[idx]
    r = rho.dat.data_ro[idx]
    a = age.dat.data_ro[idx] / YEAR_S
    for i in range(len(r) - 1):
        if r[i] < rho_bco <= r[i + 1]:
            frac = (rho_bco - r[i]) / (r[i + 1] - r[i])
            return d[i] + frac * (d[i + 1] - d[i]), a[i] + frac * (a[i + 1] - a[i])
    return np.nan, np.nan


def run_experiment(exp, law_name, rate_fn, params):
    mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
    mesh.coordinates.assign(
        fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([x_phys]))
    )
    V = fd.FunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)

    model = FirnModel(params, densification_rate_fn=rate_fn)
    solver = FirnColumnSolver(model, surface_id=SURFACE_ID)

    H = fd.Function(V); rho = fd.Function(V)
    w = fd.Function(V); age = fd.Function(V)
    depth = H0 - xi

    T0_K = exp.T0_C + 273.15
    A0 = exp.a0
    H.assign(float(params.c_i) * (T0_K - float(params.T_ref)))
    rho.interpolate(
        fd.Constant(RHO_SURF)
        + (fd.Constant(float(params.rho_i)) - fd.Constant(RHO_SURF))
        * (1.0 - fd.exp(-depth / 20.0))
    )
    w.assign(-A0 * float(params.rho_i) / RHO_SURF / YEAR_S)
    age.assign(0.0)

    dt = mk(R, DT_YR * YEAR_S, "dt")
    Ts = mk(R, T0_K, "Ts")
    Hs = mk(R, float(params.c_i) * (T0_K - float(params.T_ref)), "Hs")
    ac = mk(R, A0, "ac")
    rs = mk(R, RHO_SURF, "rs")
    ws = mk(R, -A0 * float(params.rho_i) / RHO_SURF / YEAR_S, "ws")

    bc_H = fd.DirichletBC(V, Hs, SURFACE_ID)
    bc_rho = fd.DirichletBC(V, rs, SURFACE_ID)
    bc_w = fd.DirichletBC(V, ws, SURFACE_ID)
    bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SURFACE_ID)

    def step(T_K, A_m):
        Ts.assign(T_K)
        Hs.assign(float(params.c_i) * (T_K - float(params.T_ref)))
        ac.assign(A_m)
        ws.assign(-A_m * float(params.rho_i) / RHO_SURF / YEAR_S)
        solver.prognostic_solve(
            enthalpy=H, density=rho, firn_velocity=w, dt=dt,
            accumulation=ac, surface_density=rs,
            boundary_conditions=[bc_H, bc_rho, bc_w],
            surface_temperature=Ts, enthalpy_bc_constant=Hs,
            age=age, age_boundary_condition=bc_age,
        )

    for _ in range(SPINUP_YR):
        step(T0_K, A0)

    times, dips, bco_depths, bco_ages = [], [], [], []
    for n in range(RUN_YR):
        t = float(n)
        T_K = T0_K + (exp.dT if t >= exp.t_step else 0.0)
        A_m = exp.a0 + (exp.da if t >= exp.t_step else 0.0)
        step(T_K, A_m)
        if n % 10 == 0 or n == RUN_YR - 1:
            times.append(t + 1)
            dips.append(compute_dip(rho, params, mesh))
            bd, ba = compute_bco(mesh, rho, age)
            bco_depths.append(bd)
            bco_ages.append(ba)

    coords = mesh.coordinates.dat.data_ro.flatten()
    idx = np.argsort(coords)
    return {
        "times": np.array(times), "dip": np.array(dips),
        "bco_depth": np.array(bco_depths), "bco_age": np.array(bco_ages),
        "depth": H0 - coords[idx], "rho": rho.dat.data_ro[idx],
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"FirnMICE Stokes n-sweep: spinup={SPINUP_YR}yr, run={RUN_YR}yr")
    print(f"Laws: {list(LAWS.keys())}")

    all_results = {}
    for exp in EXPERIMENTS:
        all_results[exp.name] = {}
        for law_name, (rate_fn, params) in LAWS.items():
            t0 = time.time()
            print(f"\n  {exp.name} / {law_name} ...", end="", flush=True)
            try:
                res = run_experiment(exp, law_name, rate_fn, params)
                all_results[exp.name][law_name] = res
                print(f" done ({time.time()-t0:.0f}s, rho=[{res['rho'].min():.0f},{res['rho'].max():.0f}])")
            except Exception as e:
                print(f" FAILED: {e}")

    # --- Plotting ---
    # Color scheme: H&L is black, Stokes uses a colormap by n
    import matplotlib.cm as cm
    n_colors = cm.viridis(np.linspace(0.15, 0.85, len(N_VALUES)))
    plot_styles = {"H&L": {"color": "k", "ls": "-", "lw": 2.5}}
    for i, n in enumerate(N_VALUES):
        plot_styles[f"Stokes n={n:g}"] = {
            "color": n_colors[i], "ls": "--", "lw": 1.8
        }
    law_names = list(LAWS.keys())

    temp_exps = EXPERIMENTS[:3]
    accum_exps = EXPERIMENTS[3:]

    def _plot_group(exps, group_label, suffix):
        ne = len(exps)

        # Density profiles
        fig, axes = plt.subplots(1, ne, figsize=(5 * ne, 7), sharey=True)
        if ne == 1: axes = [axes]
        for i, exp in enumerate(exps):
            ax = axes[i]
            for ln in law_names:
                if ln in all_results[exp.name]:
                    r = all_results[exp.name][ln]
                    ax.plot(r["rho"], r["depth"], **plot_styles[ln], label=ln)
            desc = f"ΔT={exp.dT:+.0f}K" if exp.dT else f"Δa={exp.da:+.2f}"
            ax.set_title(f"{exp.name}: T₀={exp.T0_C}°C, a₀={exp.a0}\n({desc})", fontsize=10)
            ax.set_xlabel("Density (kg/m³)")
            if i == 0: ax.set_ylabel("Depth (m)")
            ax.invert_yaxis(); ax.set_ylim(150, -5)
            ax.legend(fontsize=8, loc="lower left"); ax.grid(True, alpha=0.2)
        fig.suptitle(f"FirnMICE {group_label}: Density Profiles — Stokes n sweep", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f"stokes_n_{suffix}_density.png", dpi=150); plt.close(fig)

        # ΔDIP
        fig, axes = plt.subplots(1, ne, figsize=(5 * ne, 4), sharey=True)
        if ne == 1: axes = [axes]
        for i, exp in enumerate(exps):
            ax = axes[i]
            for ln in law_names:
                if ln in all_results[exp.name]:
                    r = all_results[exp.name][ln]
                    dip0 = r["dip"][0] if len(r["dip"]) > 0 else 0
                    ax.plot(r["times"], r["dip"] - dip0, **plot_styles[ln], label=ln)
            ax.axvline(exp.t_step, color="gray", ls=":", lw=1, alpha=0.6)
            ax.set_title(f"{exp.name}"); ax.set_xlabel("Time (yr)")
            if i == 0: ax.set_ylabel("ΔDIP (m)")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        fig.suptitle(f"FirnMICE {group_label}: ΔDIP — Stokes n sweep", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f"stokes_n_{suffix}_dip.png", dpi=150); plt.close(fig)

        # BCO depth
        fig, axes = plt.subplots(1, ne, figsize=(5 * ne, 4), sharey=True)
        if ne == 1: axes = [axes]
        for i, exp in enumerate(exps):
            ax = axes[i]
            for ln in law_names:
                if ln in all_results[exp.name]:
                    r = all_results[exp.name][ln]
                    ax.plot(r["times"], r["bco_depth"], **plot_styles[ln], label=ln)
            ax.axvline(exp.t_step, color="gray", ls=":", lw=1, alpha=0.6)
            ax.set_title(f"{exp.name}"); ax.set_xlabel("Time (yr)")
            if i == 0: ax.set_ylabel("BCO depth (m)")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        fig.suptitle(f"FirnMICE {group_label}: BCO Depth — Stokes n sweep", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f"stokes_n_{suffix}_bco_depth.png", dpi=150); plt.close(fig)

        # BCO age
        fig, axes = plt.subplots(1, ne, figsize=(5 * ne, 4), sharey=True)
        if ne == 1: axes = [axes]
        for i, exp in enumerate(exps):
            ax = axes[i]
            for ln in law_names:
                if ln in all_results[exp.name]:
                    r = all_results[exp.name][ln]
                    ax.plot(r["times"], r["bco_age"], **plot_styles[ln], label=ln)
            ax.axvline(exp.t_step, color="gray", ls=":", lw=1, alpha=0.6)
            ax.set_title(f"{exp.name}"); ax.set_xlabel("Time (yr)")
            if i == 0: ax.set_ylabel("BCO age (yr)")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        fig.suptitle(f"FirnMICE {group_label}: BCO Age — Stokes n sweep", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f"stokes_n_{suffix}_bco_age.png", dpi=150); plt.close(fig)

    _plot_group(temp_exps, "Temperature Step (+5K)", "temp")
    _plot_group(accum_exps, "Accumulation Step (+0.05 m/yr)", "accum")

    print(f"\nFigures saved to {OUT}/stokes_n_*.png")
    print("Done.")
