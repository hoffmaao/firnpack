"""test_firnmice_checkpoint.py

FirnMICE (Firn Model Intercomparison Experiment) step-change experiments
implemented as a pytest-style integration test.

This test is intentionally written in the same "driver" spirit as
`forward_step_change_accumulation_basal_strain_plus_strain_case_timeplots_500yr.py`,
but adapted to the FirnMICE protocol:

  - 6 experiments (ex1..ex6)
  - step change at t = 100 yr
  - run for 2000 yr after the step experiment begins
  - initial condition produced by a spinup at the experiment's baseline forcing

Outputs
-------
For each experiment we write:
  1) An HDF5 Firedrake checkpoint file (mesh + functions over time)
  2) Diagnostics (time series) stored inside the same .h5 using h5py
  3) A small set of figures similar in spirit to Lundin et al. (FirnMICE)

Controls
--------
The full FirnMICE suite is expensive for CI. By default, this test runs a
*shortened* version unless you opt in.

Environment variables:
  FIRNMICE_RUN_FULL=1          Run full 2000-yr experiments (default: 0)
  FIRNMICE_SPINUP_YEARS=5000   Spinup duration (default: 1000 short / 5000 full)
  FIRNMICE_DT_YEARS=1          Timestep in years (default: 1)
  FIRNMICE_SAVE_EVERY_YEARS=1  Save interval for checkpoints (default: 10 short / 1 full)
  FIRNMICE_PLOTS=1             Write figures (default: 1)
  FIRNMICE_OUTPUT_DIR=...      Output directory (default: tmp_path)
  FIRNMICE_SEASONAL=0          Add optional seasonal T cycle (default: 0)
  FIRNMICE_H0=1000             Column height (m) (default: 1000)
  FIRNMICE_NZ=...              Number of vertical cells (default: 120 short / 220 full)
  FIRNMICE_STRETCH_P=3         Mesh stretching exponent (default: 3)

Notes
-----
* We save state variables using Firedrake's `CheckpointFile`.
* Time-series diagnostics are appended to the same HDF5 file using h5py.
  Firedrake itself uses h5py under the hood, so this remains a single-file
  artifact per experiment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest


# ----------------------------------------------------------------------------
# Imports: prefer package layout, fall back to local modules if needed
# ----------------------------------------------------------------------------

try:
    import firedrake as fd
except Exception:  # pragma: no cover
    fd = None


try:
    from firnpack.models.firn import FirnModel, FirnParameters
    from firnpack.solvers.firn_solver import FirnColumnSolver
    from firnpack.constants import year
except ModuleNotFoundError:  # pragma: no cover
    # Running from a flat directory with local files.
    from firnpack import FirnModel, FirnParameters  # type: ignore
    from firn_solver import FirnColumnSolver  # type: ignore
    from constants import year  # type: ignore


YEAR_S = float(year)


# ----------------------------------------------------------------------------
# FirnMICE experiment definition
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class FirnMICEExperiment:
    """One FirnMICE step-change experiment."""

    name: str
    T0_C: float
    a0_mieq_yr: float
    dT_C: float = 0.0
    da_mieq_yr: float = 0.0
    t_step_yr: float = 100.0
    t_end_yr: float = 2000.0

    def mean_T_C(self, t_yr: float) -> float:
        return self.T0_C + (self.dT_C if t_yr >= self.t_step_yr else 0.0)

    def accum_mieq_yr(self, t_yr: float) -> float:
        return self.a0_mieq_yr + (self.da_mieq_yr if t_yr >= self.t_step_yr else 0.0)


FIRNMICE_EXPERIMENTS: List[FirnMICEExperiment] = [
    # ex1..ex3: +5 K step at t=100 yr, accumulation fixed 0.1 m i.e./yr
    FirnMICEExperiment("ex1", T0_C=-50.0, a0_mieq_yr=0.10, dT_C=+5.0),
    FirnMICEExperiment("ex2", T0_C=-40.0, a0_mieq_yr=0.10, dT_C=+5.0),
    FirnMICEExperiment("ex3", T0_C=-30.0, a0_mieq_yr=0.10, dT_C=+5.0),
    # ex4..ex6: +0.05 m i.e./yr step at t=100 yr, temperature fixed -30 C
    FirnMICEExperiment("ex4", T0_C=-30.0, a0_mieq_yr=0.02, da_mieq_yr=+0.05),
    FirnMICEExperiment("ex5", T0_C=-30.0, a0_mieq_yr=0.15, da_mieq_yr=+0.05),
    FirnMICEExperiment("ex6", T0_C=-30.0, a0_mieq_yr=0.25, da_mieq_yr=+0.05),
]


# ----------------------------------------------------------------------------
# Helpers (env, mesh, BCs, diagnostics, plotting)
# ----------------------------------------------------------------------------


def _float_env(name: str, default: float) -> float:
    v = os.getenv(name, "")
    return float(v) if v.strip() else float(default)


def _int_env(name: str, default: int) -> int:
    v = os.getenv(name, "")
    return int(v) if v.strip() else int(default)


def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name, "")
    if not v.strip():
        return bool(default)
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def make_real(R: "fd.FunctionSpace", value: float, name: str) -> "fd.Function":
    f = fd.Function(R, name=name)
    f.dat.data[:] = float(value)
    return f


def build_stretched_depth_mesh(H0: float, nz: int, stretch_p: float) -> Tuple["fd.Mesh", "fd.FunctionSpace"]:
    """1D IntervalMesh with a stretched vertical coordinate.

    Coordinate convention:
      x = 0      bottom
      x = H0     surface
    """
    mesh = fd.IntervalMesh(nz, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    x_phys = H0 * (1.0 - (1.0 - xi) ** stretch_p)
    coord_fs = mesh.coordinates.function_space()
    mesh.coordinates.assign(fd.Function(coord_fs).interpolate(fd.as_vector([x_phys])))
    V = fd.FunctionSpace(mesh, "CG", 1)
    return mesh, V


def depth_from_mesh(mesh: "fd.Mesh", H0: float) -> np.ndarray:
    """Return depth array (m), with depth=0 at the surface."""
    x = mesh.coordinates.dat.data_ro
    if x.ndim == 2:
        x = x[:, 0]
    return (H0 - np.asarray(x)).copy()


def seasonal_T_offset_K(t_yr: float, amplitude_K: float = 10.0) -> float:
    """Optional FirnMICE seasonal temperature cycle."""
    return float(amplitude_K) * (np.cos(2.0 * np.pi * t_yr) + 0.3 * np.cos(4.0 * np.pi * t_yr))


def w_s_bc(params: FirnParameters, accum_mieq_yr: "fd.Function", rho_surf: "fd.Function") -> "fd.Expr":
    """Surface kinematic BC for w (negative downward)."""
    return -accum_mieq_yr * params.rho_i / rho_surf / YEAR_S


@dataclass
class SurfaceBCs:
    bc_H: "fd.DirichletBC"
    bc_rho: "fd.DirichletBC"
    bc_w: "fd.DirichletBC"
    w_surf_bc: "fd.Function"


def make_surface_bcs(
    V: "fd.FunctionSpace",
    params: FirnParameters,
    *,
    accum: "fd.Function",
    rho_surf: "fd.Function",
    Hs_bc: "fd.Function",
    surface_id: int = 2,
) -> SurfaceBCs:
    """Create surface Dirichlet BCs and a mutable w_surf_bc function."""
    bc_H = fd.DirichletBC(V, Hs_bc, surface_id)

    rho_s_bc = fd.Function(V, name="rho_surf_bc")
    rho_s_bc.interpolate(rho_surf)
    bc_rho = fd.DirichletBC(V, rho_s_bc, surface_id)

    w_surf_bc = fd.Function(V, name="w_surf_bc")
    w_surf_bc.interpolate(w_s_bc(params, accum, rho_surf))
    bc_w = fd.DirichletBC(V, w_surf_bc, surface_id)

    return SurfaceBCs(bc_H=bc_H, bc_rho=bc_rho, bc_w=bc_w, w_surf_bc=w_surf_bc)


def update_surface_velocity_bc(bcs: SurfaceBCs, params: FirnParameters, accum: "fd.Function", rho_surf: "fd.Function") -> None:
    bcs.w_surf_bc.interpolate(w_s_bc(params, accum, rho_surf))


def set_surface_temperature(params: FirnParameters, Ts: "fd.Function", Hs: "fd.Function", Ts_K: float) -> None:
    Ts.assign(float(Ts_K))
    Hs.assign(float(params.c_i) * (float(Ts_K) - float(params.T_ref)))


@dataclass
class FirnState:
    H: "fd.Function"
    rho: "fd.Function"
    w: "fd.Function"
    sigma: "fd.Function"
    r2: "fd.Function"
    age: "fd.Function"


def compute_DIP_m(model: FirnModel, rho: "fd.Function") -> float:
    """Depth-integrated porosity (DIP) in meters."""
    dx = fd.dx(domain=rho.function_space().mesh())
    return float(fd.assemble(model.porosity(rho) * dx))


def _first_crossing_depth(depth_m: np.ndarray, rho_kg_m3: np.ndarray, rho_target: float) -> float:
    idx = np.where(rho_kg_m3 >= float(rho_target))[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(depth_m[0])
    z0, z1 = float(depth_m[i - 1]), float(depth_m[i])
    r0, r1 = float(rho_kg_m3[i - 1]), float(rho_kg_m3[i])
    if r1 == r0:
        return float(z1)
    return z0 + (float(rho_target) - r0) * (z1 - z0) / (r1 - r0)


def compute_BCO_depth_age(
    *,
    depth_m_sorted: np.ndarray,
    rho_sorted: np.ndarray,
    age_sorted_s: np.ndarray,
    rho_bco: float = 815.0,
) -> Tuple[float, float]:
    """Return BCO depth (m) and age (yr) at rho=815 kg/m^3.

    If density reversals occur, choose the *shallowest* crossing.
    """
    z_bco = _first_crossing_depth(depth_m_sorted, rho_sorted, rho_bco)
    if not np.isfinite(z_bco):
        return float("nan"), float("nan")
    age_bco_s = float(np.interp(z_bco, depth_m_sorted, age_sorted_s))
    return float(z_bco), float(age_bco_s / YEAR_S)


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_timeseries_to_h5(h5_path: Path, group: str, arrays: Dict[str, np.ndarray], attrs: Dict[str, Any]) -> None:
    """Append numpy arrays + attrs into an existing Firedrake checkpoint .h5."""
    import h5py

    with h5py.File(h5_path, "a") as f:
        g = f.require_group(group)
        for k, v in arrays.items():
            if k in g:
                del g[k]
            g.create_dataset(k, data=np.asarray(v))
        for k, v in attrs.items():
            g.attrs[k] = v


def plot_timeseries(
    *,
    time_yr: np.ndarray,
    y: np.ndarray,
    ylabel: str,
    out_png: Path,
    title: str,
    step_year: float = 100.0,
    xlim: Tuple[float, float] | None = (0.0, 1000.0),
) -> None:
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    t = np.asarray(time_yr, dtype=float)
    yy = np.asarray(y, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.2))
    ax.plot(t, yy, linewidth=1.6)
    ax.axvline(float(step_year), linestyle="--", linewidth=1.0, color="k", alpha=0.5)
    ax.set_xlabel("time (yr)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _safe_mkdir(out_png.parent)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_density_profiles(
    *,
    depth_m: np.ndarray,
    rho0: np.ndarray,
    rho_end: np.ndarray,
    out_png: Path,
    title: str,
) -> None:
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    z = np.asarray(depth_m, dtype=float)
    r0 = np.asarray(rho0, dtype=float)
    r1 = np.asarray(rho_end, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 6.0))
    ax.plot(r0, z, linestyle="--", linewidth=1.6, label="t=0")
    ax.plot(r1, z, linestyle="-", linewidth=1.8, label="t=end")
    ax.invert_yaxis()
    ax.set_xlabel("density (kg/m$^3$)")
    ax.set_ylabel("depth (m)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _safe_mkdir(out_png.parent)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Core runner
# ----------------------------------------------------------------------------


def run_firnmice_experiment(
    *,
    exp: FirnMICEExperiment,
    out_dir: Path,
    H0: float,
    NZ: int,
    STRETCH_P: float,
    dt_years: float,
    spinup_years: float,
    save_every_years: float,
    make_plots: bool,
    use_seasonal: bool,
    rho_surf_kg_m3: float = 360.0,
    rho_bco_kg_m3: float = 815.0,
) -> Dict[str, Any]:
    """Run one FirnMICE experiment and write artifacts."""

    assert fd is not None

    _safe_mkdir(out_dir)
    h5_path = out_dir / f"firnmice_{exp.name}.h5"
    fig_dir = out_dir / "figures" / exp.name

    # --- Mesh + spaces ---
    mesh, V = build_stretched_depth_mesh(H0, NZ, STRETCH_P)
    R = fd.FunctionSpace(mesh, "R", 0)

    # --- Model + solver ---
    params = FirnParameters()
    model = FirnModel(params)
    solver = FirnColumnSolver(model, horizontal_divergence=0.0)

    # --- Forcing scalars (Real space) ---
    Ts = make_real(R, exp.mean_T_C(0.0) + 273.15, "Ts")
    accum = make_real(R, exp.accum_mieq_yr(0.0), "accum")
    rho_surf = make_real(R, float(rho_surf_kg_m3), "rho_surf")
    dt = make_real(R, float(dt_years) * YEAR_S, "dt")
    Hs = make_real(R, float(params.c_i) * (float(Ts.dat.data_ro[0]) - float(params.T_ref)), "Hs")

    # --- State variables (CG1) ---
    H = fd.Function(V, name="enthalpy")
    rho = fd.Function(V, name="density")
    w = fd.Function(V, name="firn_velocity")
    sigma = fd.Function(V, name="stress")
    r2 = fd.Function(V, name="grain_radius2")
    age = fd.Function(V, name="age")

    # --- Initial conditions ---
    set_surface_temperature(params, Ts, Hs, float(Ts.dat.data_ro[0]))
    H.assign(float(Hs.dat.data_ro[0]))

    # Reasonable initial density profile (relaxes during spinup)
    x = fd.SpatialCoordinate(mesh)[0]
    depth = float(H0) - x
    rho_efold = 20.0
    rho.interpolate(float(rho_surf_kg_m3) + (float(params.rho_i) - float(rho_surf_kg_m3)) * (1.0 - fd.exp(-depth / rho_efold)))

    # Initial velocity guess
    w.assign(-float(accum.dat.data_ro[0]) * float(params.rho_i) / float(rho_surf_kg_m3) / YEAR_S)

    sigma.assign(0.0)
    r2.assign(float(getattr(params, "r2_surf", 2.5e-7)))
    age.assign(0.0)

    state = FirnState(H=H, rho=rho, w=w, sigma=sigma, r2=r2, age=age)

    # --- BCs ---
    SURFACE_ID = 2
    surface_bcs = make_surface_bcs(V, params, accum=accum, rho_surf=rho_surf, Hs_bc=Hs, surface_id=SURFACE_ID)
    bc_sigma = fd.DirichletBC(V, make_real(R, 0.0, "sigma_surf"), SURFACE_ID)
    bc_r2 = fd.DirichletBC(V, make_real(R, float(getattr(params, "r2_surf", 2.5e-7)), "r2_surf"), SURFACE_ID)
    bc_age = fd.DirichletBC(V, make_real(R, 0.0, "age_surf"), SURFACE_ID)

    bcs_list = [surface_bcs.bc_H, surface_bcs.bc_rho, surface_bcs.bc_w]

    # --- Full-density kc smoothing field (matches your existing driver style) ---
    rhoCoef = fd.Function(V, name="rhoCoef")
    kc0_val = float(getattr(params, "kc0", 9.2e-9))
    kc1_val = float(getattr(params, "kc1", 3.7e-9))
    rho_m = float(getattr(params, "rho_m", 550.0))
    rho_smooth = 20.0

    def update_rhoCoef() -> None:
        s = 0.5 * (1.0 + fd.tanh((state.rho - rho_m) / rho_smooth))
        rhoCoef.interpolate((1.0 - s) * kc0_val + s * kc1_val)

    # --- Precompute depth ordering for BCO diagnostics ---
    depth_nodes = depth_from_mesh(mesh, H0)
    sort_idx = np.argsort(depth_nodes)
    depth_sorted = depth_nodes[sort_idx]

    # ------------------------------------------------------------------
    # Spinup (not written to checkpoint by default)
    # ------------------------------------------------------------------
    n_spin = int(round(float(spinup_years) / float(dt_years)))
    for _ in range(n_spin):
        update_rhoCoef()
        solver.prognostic_solve(
            enthalpy=state.H,
            density=state.rho,
            firn_velocity=state.w,
            dt=dt,
            accumulation=accum,
            surface_density=rho_surf,
            boundary_conditions=bcs_list,
            surface_temperature=Ts,
            enthalpy_bc_constant=Hs,
            stress=state.sigma,
            grain_radius2=state.r2,
            stress_boundary_condition=bc_sigma,
            grain_radius2_boundary_condition=bc_r2,
            rhoCoef=rhoCoef,
            age=state.age,
            age_boundary_condition=bc_age,
        )

    # ------------------------------------------------------------------
    # Main FirnMICE transient (t = 0 .. t_end)
    # ------------------------------------------------------------------
    n_steps = int(round(float(exp.t_end_yr) / float(dt_years)))
    save_every_steps = max(1, int(round(float(save_every_years) / float(dt_years))))

    time_yr: List[float] = []
    Ts_K_series: List[float] = []
    accum_series: List[float] = []
    dip_series_m: List[float] = []
    bco_depth_m: List[float] = []
    bco_age_yr: List[float] = []

    # For quick profile figures (avoid re-loading from checkpoint)
    rho0_profile: np.ndarray | None = None
    rho_end_profile: np.ndarray | None = None

    with fd.CheckpointFile(str(h5_path), "w") as chk:
        chk.save_mesh(mesh)

        def save_snapshot(idx: int, t_yr: float) -> None:
            nonlocal rho0_profile, rho_end_profile
            # Firedrake fields
            chk.save_function(state.H, name="enthalpy", idx=idx)
            chk.save_function(state.rho, name="density", idx=idx)
            chk.save_function(state.w, name="velocity", idx=idx)
            chk.save_function(state.sigma, name="stress", idx=idx)
            chk.save_function(state.r2, name="grain_radius2", idx=idx)
            chk.save_function(state.age, name="age", idx=idx)

            # Diagnostics (numpy)
            time_yr.append(float(t_yr))
            Ts_K_series.append(float(Ts.dat.data_ro[0]))
            accum_series.append(float(accum.dat.data_ro[0]))

            dip_series_m.append(compute_DIP_m(model, state.rho))

            rho_sorted = np.asarray(state.rho.dat.data_ro)[sort_idx]
            age_sorted_s = np.asarray(state.age.dat.data_ro)[sort_idx]

            # Cache first and last density profiles for plotting.
            if idx == 0:
                rho0_profile = rho_sorted.copy()
            rho_end_profile = rho_sorted.copy()
            zb, ab = compute_BCO_depth_age(
                depth_m_sorted=depth_sorted,
                rho_sorted=rho_sorted,
                age_sorted_s=age_sorted_s,
                rho_bco=float(rho_bco_kg_m3),
            )
            bco_depth_m.append(zb)
            bco_age_yr.append(ab)

        # initial snapshot at t=0
        save_idx = 0
        save_snapshot(save_idx, 0.0)

        for n in range(1, n_steps + 1):
            t = float(n) * float(dt_years)

            # Forcing
            T_mean_C = exp.mean_T_C(t)
            Ts_K = T_mean_C + 273.15
            if use_seasonal:
                Ts_K += seasonal_T_offset_K(t)
            a_mieq_yr = exp.accum_mieq_yr(t)

            accum.assign(float(a_mieq_yr))
            set_surface_temperature(params, Ts, Hs, Ts_K)
            update_surface_velocity_bc(surface_bcs, params, accum, rho_surf)

            # Step
            update_rhoCoef()
            solver.prognostic_solve(
                enthalpy=state.H,
                density=state.rho,
                firn_velocity=state.w,
                dt=dt,
                accumulation=accum,
                surface_density=rho_surf,
                boundary_conditions=bcs_list,
                surface_temperature=Ts,
                enthalpy_bc_constant=Hs,
                stress=state.sigma,
                grain_radius2=state.r2,
                stress_boundary_condition=bc_sigma,
                grain_radius2_boundary_condition=bc_r2,
                rhoCoef=rhoCoef,
                age=state.age,
                age_boundary_condition=bc_age,
            )

            if (n % save_every_steps) == 0 or n == n_steps:
                save_idx += 1
                save_snapshot(save_idx, t)

    # ------------------------------------------------------------------
    # Append diagnostics + metadata to the same H5 file
    # ------------------------------------------------------------------
    time_yr_arr = np.asarray(time_yr)
    Ts_K_arr = np.asarray(Ts_K_series)
    accum_arr = np.asarray(accum_series)
    dip_arr = np.asarray(dip_series_m)
    bco_z_arr = np.asarray(bco_depth_m)
    bco_age_arr = np.asarray(bco_age_yr)

    diag_arrays = dict(
        time_years=time_yr_arr,
        Ts_K=Ts_K_arr,
        accum_mieq_yr=accum_arr,
        DIP_m=dip_arr,
        BCO_depth_m=bco_z_arr,
        BCO_age_yr=bco_age_arr,
        depth_nodes_m=depth_sorted,
    )
    diag_attrs = dict(
        experiment=exp.name,
        T0_C=exp.T0_C,
        a0_mieq_yr=exp.a0_mieq_yr,
        dT_C=exp.dT_C,
        da_mieq_yr=exp.da_mieq_yr,
        t_step_yr=exp.t_step_yr,
        t_end_yr=exp.t_end_yr,
        spinup_years=float(spinup_years),
        dt_years=float(dt_years),
        save_every_years=float(save_every_years),
        H0=float(H0),
        NZ=int(NZ),
        STRETCH_P=float(STRETCH_P),
        rho_surf_kg_m3=float(rho_surf_kg_m3),
        rho_bco_kg_m3=float(rho_bco_kg_m3),
        seasonal_cycle=int(bool(use_seasonal)),
    )
    write_timeseries_to_h5(h5_path, group="firnmice", arrays=diag_arrays, attrs=diag_attrs)

    # ------------------------------------------------------------------
    # Figures (optional)
    # ------------------------------------------------------------------
    if make_plots:
        plot_timeseries(
            time_yr=time_yr_arr,
            y=dip_arr,
            ylabel="DIP (m)",
            out_png=fig_dir / f"{exp.name}_DIP.png",
            title=f"{exp.name}: depth-integrated porosity (DIP)",
            step_year=exp.t_step_yr,
            xlim=(0.0, 1000.0),
        )
        plot_timeseries(
            time_yr=time_yr_arr,
            y=dip_arr - dip_arr[0],
            ylabel="ΔDIP (m)",
            out_png=fig_dir / f"{exp.name}_delta_DIP.png",
            title=f"{exp.name}: change in DIP relative to t=0",
            step_year=exp.t_step_yr,
            xlim=(0.0, 1000.0),
        )
        plot_timeseries(
            time_yr=time_yr_arr,
            y=bco_age_arr,
            ylabel="BCO age (yr)",
            out_png=fig_dir / f"{exp.name}_BCO_age.png",
            title=f"{exp.name}: bubble close-off (ρ={rho_bco_kg_m3:g} kg/m³) age",
            step_year=exp.t_step_yr,
            xlim=(0.0, 1000.0),
        )
        plot_timeseries(
            time_yr=time_yr_arr,
            y=bco_z_arr,
            ylabel="BCO depth (m)",
            out_png=fig_dir / f"{exp.name}_BCO_depth.png",
            title=f"{exp.name}: bubble close-off (ρ={rho_bco_kg_m3:g} kg/m³) depth",
            step_year=exp.t_step_yr,
            xlim=(0.0, 1000.0),
        )

        # Density profile (t=0 vs t=end)
        if (rho0_profile is not None) and (rho_end_profile is not None):
            plot_density_profiles(
                depth_m=depth_sorted,
                rho0=rho0_profile,
                rho_end=rho_end_profile,
                out_png=fig_dir / f"{exp.name}_density_profiles.png",
                title=f"{exp.name}: density profiles (t=0 vs t=end)",
            )

    return dict(
        h5_path=str(h5_path),
        time_years=time_yr_arr,
        DIP_m=dip_arr,
        BCO_depth_m=bco_z_arr,
        BCO_age_yr=bco_age_arr,
    )


# ----------------------------------------------------------------------------
# Pytest entry point
# ----------------------------------------------------------------------------


@pytest.mark.slow
def test_firnmice_suite_checkpoint(tmp_path: Path) -> None:
    """Run FirnMICE experiments (optionally shortened) and write HDF5 checkpoints."""

    if fd is None:
        pytest.skip("firedrake not available")

    run_full = _bool_env("FIRNMICE_RUN_FULL", False)
    make_plots = _bool_env("FIRNMICE_PLOTS", True)
    use_seasonal = _bool_env("FIRNMICE_SEASONAL", False)

    # Geometry: FirnMICE uses ~1000 m to represent the ice-sheet thermal mass.
    H0 = _float_env("FIRNMICE_H0", 1000.0)
    NZ = _int_env("FIRNMICE_NZ", 320 if run_full else 220)
    STRETCH_P = _float_env("FIRNMICE_STRETCH_P", 3.0)

    dt_years = _float_env("FIRNMICE_DT_YEARS", 1.0)
    spinup_years_default = 5000.0 if run_full else 1000.0
    spinup_years = _float_env("FIRNMICE_SPINUP_YEARS", spinup_years_default)
    save_every_years_default = 1.0 if run_full else 10.0
    save_every_years = _float_env("FIRNMICE_SAVE_EVERY_YEARS", save_every_years_default)

    out_dir = Path(os.getenv("FIRNMICE_OUTPUT_DIR", str(tmp_path / "firnmice")))
    _safe_mkdir(out_dir)

    # Short mode: truncate main run length (keeps step at 100 yrs but only run to 200 yrs).
    t_end_short = 200.0

    for exp0 in FIRNMICE_EXPERIMENTS:
        exp = exp0
        if not run_full:
            exp = FirnMICEExperiment(
                name=exp0.name,
                T0_C=exp0.T0_C,
                a0_mieq_yr=exp0.a0_mieq_yr,
                dT_C=exp0.dT_C,
                da_mieq_yr=exp0.da_mieq_yr,
                t_step_yr=exp0.t_step_yr,
                t_end_yr=t_end_short,
            )

        res = run_firnmice_experiment(
            exp=exp,
            out_dir=out_dir,
            H0=H0,
            NZ=NZ,
            STRETCH_P=STRETCH_P,
            dt_years=dt_years,
            spinup_years=spinup_years,
            save_every_years=save_every_years,
            make_plots=make_plots,
            use_seasonal=use_seasonal,
            rho_surf_kg_m3=360.0,
        )

        # -----------------------------------------------------------------
        # Minimal, robust assertions (avoid brittle numeric targets)
        # -----------------------------------------------------------------
        time = res["time_years"]
        DIP = res["DIP_m"]
        bco_z = res["BCO_depth_m"]
        bco_age = res["BCO_age_yr"]

        assert np.isfinite(time).all()
        assert np.isfinite(DIP).all()

        # We expect to reach BCO in a 1000 m domain for these forcings.
        assert np.isfinite(bco_z).any(), f"BCO depth never found in {exp.name}"
        assert np.isfinite(bco_age).any(), f"BCO age never found in {exp.name}"

        # Sign sanity checks using the final saved value.
        # Warming reduces DIP; accumulation increase increases DIP.
        if exp.dT_C != 0.0:
            assert DIP[-1] <= DIP[0] + 1e-8
        if exp.da_mieq_yr != 0.0:
            assert DIP[-1] >= DIP[0] - 1e-8


if __name__ == "__main__":  # pragma: no cover
    # Allow running as a script without pytest.
    from tempfile import TemporaryDirectory

    if fd is None:
        raise SystemExit("firedrake not available")

    run_full = _bool_env("FIRNMICE_RUN_FULL", True)
    make_plots = _bool_env("FIRNMICE_PLOTS", True)
    use_seasonal = _bool_env("FIRNMICE_SEASONAL", False)

    H0 = _float_env("FIRNMICE_H0", 1000.0)
    NZ = _int_env("FIRNMICE_NZ", 320 if run_full else 220)
    STRETCH_P = _float_env("FIRNMICE_STRETCH_P", 3.0)

    dt_years = _float_env("FIRNMICE_DT_YEARS", 1.0)
    spinup_years = _float_env("FIRNMICE_SPINUP_YEARS", 5000.0 if run_full else 1000.0)
    save_every_years = _float_env("FIRNMICE_SAVE_EVERY_YEARS", 1.0 if run_full else 10.0)

    out_dir_env = os.getenv("FIRNMICE_OUTPUT_DIR", "").strip()
    if out_dir_env:
        out_dir = Path(out_dir_env)
        _safe_mkdir(out_dir)
        temp_ctx = None
    else:
        temp_ctx = TemporaryDirectory()
        out_dir = Path(temp_ctx.name) / "firnmice"
        _safe_mkdir(out_dir)

    try:
        for exp in FIRNMICE_EXPERIMENTS:
            print(f"[run] {exp.name}: T0={exp.T0_C:+.1f}°C, a0={exp.a0_mieq_yr:g} m i.e./yr")
            run_firnmice_experiment(
                exp=exp,
                out_dir=out_dir,
                H0=H0,
                NZ=NZ,
                STRETCH_P=STRETCH_P,
                dt_years=dt_years,
                spinup_years=spinup_years,
                save_every_years=save_every_years,
                make_plots=make_plots,
                use_seasonal=use_seasonal,
                rho_surf_kg_m3=360.0,
            )
        print(f"\n[done] Outputs in: {out_dir}")
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()
