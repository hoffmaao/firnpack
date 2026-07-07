#!/usr/bin/env python3
"""plot_firnmice_5panel_summary.py

Create a *single* 5-panel summary figure across FirnMICE experiments (ex1..ex6)
using the HDF5 outputs written by ``test_firnmice_checkpoint.py``.

Panels (left-to-right)
----------------------
  1) Final density profiles (t = end) for experiments I–VI
  2) DIP time series
  3) ΔDIP time series (relative to t=0)
  4) BCO depth time series
  5) BCO age time series

The script reads:
  - time-series diagnostics from the ``/firnmice`` group (via h5py)
  - final density profiles from Firedrake checkpoints (via firedrake)

Example
-------
  python plot_firnmice_5panel_summary.py \
      --input outputs/firnmice \
      --output outputs/firnmice/figures/firnmice_5panel_summary.png

Notes
-----
* This script assumes each experiment file is named ``firnmice_exN.h5``.
* Firedrake is required to load the checkpointed density field.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}


@dataclass(frozen=True)
class ExpData:
    name: str
    roman: str
    color: str
    time_yr: np.ndarray
    dip_m: np.ndarray
    bco_depth_m: np.ndarray
    bco_age_yr: np.ndarray
    depth_m: np.ndarray
    rho_kg_m3: np.ndarray


def _read_firnmice_group(h5_path: Path) -> Dict[str, np.ndarray]:
    import h5py

    with h5py.File(h5_path, "r") as f:
        if "firnmice" not in f:
            raise KeyError(f"Group '/firnmice' not found in {h5_path}")
        g = f["firnmice"]
        out = {
            "time_years": np.asarray(g["time_years"], dtype=float),
            "DIP_m": np.asarray(g["DIP_m"], dtype=float),
            "BCO_depth_m": np.asarray(g["BCO_depth_m"], dtype=float),
            "BCO_age_yr": np.asarray(g["BCO_age_yr"], dtype=float),
        }
        # optional but nice to have
        if "depth_nodes_m" in g:
            out["depth_nodes_m"] = np.asarray(g["depth_nodes_m"], dtype=float)
        else:
            out["depth_nodes_m"] = None
        # attrs
        out["H0"] = float(g.attrs.get("H0", np.nan))
    return out


def _load_final_density_profile(h5_path: Path, *, idx_last: int, H0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Load final density profile from Firedrake CheckpointFile.

    Returns depth (m, 0 at surface, positive downward) and density (kg/m^3)
    sorted by depth.
    """
    try:
        import firedrake as fd
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Firedrake is required to load density profiles from checkpoint files. "
            "Install/activate Firedrake and retry."
        ) from e

    with fd.CheckpointFile(str(h5_path), "r") as chk:
        # Firedrake versions differ slightly; try the common patterns.
        try:
            mesh = chk.load_mesh()
        except TypeError:  # pragma: no cover
            mesh = chk.load_mesh("mesh")

        rho = None
        try:
            rho = chk.load_function(mesh, "density", idx=idx_last)
        except Exception:
            # Fallback: construct Function and load into it
            V = fd.FunctionSpace(mesh, "CG", 1)
            rho = fd.Function(V, name="density")
            chk.load_function(rho, name="density", idx=idx_last)

        coords = mesh.coordinates.dat.data_ro
        if coords.ndim == 2:
            x = np.asarray(coords[:, 0], dtype=float)
        else:
            x = np.asarray(coords, dtype=float)
        depth = float(H0) - x
        rho_arr = np.asarray(rho.dat.data_ro, dtype=float)
        sidx = np.argsort(depth)
        return depth[sidx], rho_arr[sidx]


def _discover_experiment_files(root: Path) -> List[Tuple[int, Path]]:
    files: List[Tuple[int, Path]] = []
    for k in range(1, 7):
        p = root / f"firnmice_ex{k}.h5"
        if p.exists():
            files.append((k, p))
    if not files:
        raise FileNotFoundError(f"No firnmice_ex*.h5 files found under: {root}")
    return files


def make_figure(exps: List[ExpData], out_path: Path, *, step_year: float = 100.0, dpi: int = 300) -> None:
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass

    import matplotlib.pyplot as plt

    # Big, readable defaults
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.linewidth": 1.0,
        }
    )

    # 5 square panels.
    panel = 4.0  # inches per square axis
    fig_w = 5 * panel
    fig_h = panel + 1.2  # extra space for legend
    fig, axes = plt.subplots(1, 5, figsize=(fig_w, fig_h), sharex=False, sharey=False)

    ax_rho, ax_dip, ax_ddip, ax_bcoz, ax_bcoa = axes
    for ax in axes:
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass

    # --- Panel 1: density profiles ---
    handles = []
    labels = []
    for e in exps:
        (ln,) = ax_rho.plot(e.rho_kg_m3, e.depth_m, lw=2.0, color=e.color)
        handles.append(ln)
        labels.append(f"Exp {e.roman}")

    ax_rho.invert_yaxis()
    ax_rho.set_title("Density")
    ax_rho.set_xlabel(r"$\rho$ (kg m$^{-3}$)")
    ax_rho.set_ylabel("depth (m)")
    ax_rho.grid(True, alpha=0.25)

    # --- Shared time axis extents ---
    tmax = float(max(np.nanmax(e.time_yr) for e in exps))
    tmin = float(min(np.nanmin(e.time_yr) for e in exps))
    xlim = (tmin, tmax)

    def _plot_timeseries(ax, ygetter, title: str, ylabel: str) -> None:
        for e in exps:
            ax.plot(e.time_yr, ygetter(e), lw=2.0, color=e.color)
        ax.axvline(step_year, ls="--", lw=1.2, color="k", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("time (yr)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.25)

    # --- Panels 2–5 ---
    _plot_timeseries(ax_dip, lambda e: e.dip_m, "DIP", "DIP (m)")
    _plot_timeseries(ax_ddip, lambda e: e.dip_m - e.dip_m[0], "ΔDIP", "ΔDIP (m)")
    _plot_timeseries(ax_bcoz, lambda e: e.bco_depth_m, "BCO depth", "depth (m)")
    _plot_timeseries(ax_bcoa, lambda e: e.bco_age_yr, "BCO age", "age (yr)")

    # BCO depth convention: depth increases downward, so invert to match density panel
    ax_bcoz.invert_yaxis()

    # Unified legend beneath all panels
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=2.5,
    )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a 5-panel FirnMICE summary figure from HDF5 outputs")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/firnmice"),
        help="Directory containing firnmice_ex1.h5 ... firnmice_ex6.h5 (default: outputs/firnmice)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <input>/figures/firnmice_5panel_summary.png)",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--step-year", type=float, default=100.0)
    args = ap.parse_args()

    root = args.input
    out_png = args.output or (root / "figures" / "firnmice_5panel_summary.png")

    # Matplotlib default cycle gives us stable, distinct colours.
    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(colors) < 6:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    exps: List[ExpData] = []
    for k, h5 in _discover_experiment_files(root):
        g = _read_firnmice_group(h5)
        time = g["time_years"]
        dip = g["DIP_m"]
        bcoz = g["BCO_depth_m"]
        bcoa = g["BCO_age_yr"]
        H0 = g["H0"]
        if not np.isfinite(H0):
            raise ValueError(f"Missing H0 attribute in /firnmice group of {h5}")
        idx_last = int(len(time) - 1)
        depth, rho = _load_final_density_profile(h5, idx_last=idx_last, H0=H0)
        exps.append(
            ExpData(
                name=f"ex{k}",
                roman=ROMAN[k],
                color=colors[k - 1],
                time_yr=time,
                dip_m=dip,
                bco_depth_m=bcoz,
                bco_age_yr=bcoa,
                depth_m=depth,
                rho_kg_m3=rho,
            )
        )

    make_figure(exps, out_png, step_year=float(args.step_year), dpi=int(args.dpi))
    print(f"[done] Wrote: {out_png}")


if __name__ == "__main__":
    main()
