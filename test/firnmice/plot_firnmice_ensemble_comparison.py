#!/usr/bin/env python3
"""plot_firnmice_ensemble_comparison.py

Overlay published FirnMICE participant model outputs with our firn model runs.

Reads published data from zip archives in ZIP_DIR, and (optionally) our model
h5 output from H5_DIR.  Edit the configuration block below to change paths.

Two output figures
------------------
  1. firnmice_dip_comparison.png    – DIP and ΔDIP time series (2 rows × 6 cols)
  2. firnmice_bco_comparison.png    – BCO depth and BCO age time series (2 rows × 6 cols)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------

EXP_LABELS = {
    1: "Exp I\n$T_0=-50°C$, $\\Delta T=+5°C$\n$A=0.10$ m/yr",
    2: "Exp II\n$T_0=-40°C$, $\\Delta T=+5°C$\n$A=0.10$ m/yr",
    3: "Exp III\n$T_0=-30°C$, $\\Delta T=+5°C$\n$A=0.10$ m/yr",
    4: "Exp IV\n$T=-30°C$, $A_0=0.02$ m/yr\n$\\Delta A=+0.05$ m/yr",
    5: "Exp V\n$T=-30°C$, $A_0=0.15$ m/yr\n$\\Delta A=+0.05$ m/yr",
    6: "Exp VI\n$T=-30°C$, $A_0=0.25$ m/yr\n$\\Delta A=+0.05$ m/yr",
}

T_STEP_YR = 100.0  # forcing step applied at this time

# The 7 known participant models
KNOWN_MODELS = [
    "Aorsi",
    "Arthern",
    "Christo_Barnola",
    "Christo_HL_dynamic",
    "Cummings",
    "Ligtenberg",
    "Simonsen",
]

# Model display names (short)
MODEL_DISPLAY = {
    "Aorsi": "Aorsi",
    "Arthern": "Arthern",
    "Christo_Barnola": "Christo (B)",
    "Christo_HL_dynamic": "Christo (HL)",
    "Cummings": "Cummings",
    "Ligtenberg": "Ligtenberg",
    "Simonsen": "Simonsen",
}

# Assign a distinct color to each published model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (after backend set)
import matplotlib.cm as cm

_BASE_COLORS = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
if len(_BASE_COLORS) < len(KNOWN_MODELS):
    _BASE_COLORS = [f"C{i}" for i in range(len(KNOWN_MODELS))]
MODEL_COLORS = {m: _BASE_COLORS[i] for i, m in enumerate(KNOWN_MODELS)}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _find_zip_files(zip_dir: Path) -> List[Path]:
    """Return sorted list of FirnMICE zip files found in zip_dir."""
    zips = sorted(zip_dir.glob("FirnMICE-*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"No FirnMICE-*.zip files found in {zip_dir}"
        )
    return zips


def _load_txt_from_zip(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as f:
        return np.loadtxt(io.TextIOWrapper(f))


def load_published_porosity(
    zip_dir: Path,
) -> Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]:
    """Load all published DIP time series from zip archives.

    Returns
    -------
    dict mapping (model_name, exp_num) → (time_yr, dip_m)
    """
    result: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
    for zpath in _find_zip_files(zip_dir):
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if (
                    name.endswith("Porosity.txt")
                    and "._" not in name
                    and "deriv" not in name
                    and "full" not in name
                ):
                    parts = name.split("/")
                    if len(parts) < 4:
                        continue
                    model = parts[1]
                    exp_str = parts[2]  # e.g. "Experiment1"
                    if not exp_str.startswith("Experiment"):
                        continue
                    try:
                        exp_num = int(exp_str.replace("Experiment", ""))
                    except ValueError:
                        continue
                    data = _load_txt_from_zip(zf, name)
                    # row 0 = time axis (yr), row 1 = DIP (m)
                    time_yr = data[0, :]
                    dip_m = data[1, :]
                    result[(model, exp_num)] = (time_yr, dip_m)
    return result


def load_published_bco(
    zip_dir: Path,
) -> Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load all published BCO depth/age time series from zip archives.

    Returns
    -------
    dict mapping (model_name, exp_num) → (time_yr, bco_depth_m, bco_age_yr)
    """
    result: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for zpath in _find_zip_files(zip_dir):
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if (
                    "Rho815_trun.txt" in name
                    and "._" not in name
                ):
                    parts = name.split("/")
                    if len(parts) < 4:
                        continue
                    model = parts[1]
                    exp_str = parts[2]
                    if not exp_str.startswith("Experiment"):
                        continue
                    try:
                        exp_num = int(exp_str.replace("Experiment", ""))
                    except ValueError:
                        continue
                    data = _load_txt_from_zip(zf, name)
                    time_yr = data[0, :]
                    bco_depth_m = data[1, :]
                    bco_age_yr = data[2, :]
                    result[(model, exp_num)] = (time_yr, bco_depth_m, bco_age_yr)
    return result


def load_our_model(
    h5_dir: Path, exp_num: int
) -> Optional[Dict[str, np.ndarray]]:
    """Load our model h5 output for one experiment.

    Returns None if file does not exist.
    """
    h5_path = h5_dir / f"firnmice_ex{exp_num}.h5"
    if not h5_path.exists():
        return None
    try:
        import h5py

        with h5py.File(h5_path, "r") as f:
            g = f["firnmice"]
            return {
                "time_yr": np.asarray(g["time_years"], dtype=float),
                "dip_m": np.asarray(g["DIP_m"], dtype=float),
                "bco_depth_m": np.asarray(g["BCO_depth_m"], dtype=float),
                "bco_age_yr": np.asarray(g["BCO_age_yr"], dtype=float),
            }
    except Exception as e:
        print(f"  [warn] Could not load {h5_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Truncation helper
# ---------------------------------------------------------------------------


def _trunc(
    time_yr: np.ndarray, values: np.ndarray, t_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    mask = time_yr <= t_max
    return time_yr[mask], values[mask]


def _delta(values: np.ndarray) -> np.ndarray:
    """Return anomaly relative to first time step."""
    return values - values[0]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _style_ax(ax: "plt.Axes", title: str, ylabel: str, xlim: Tuple) -> None:
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xlabel("time (yr)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.axvline(T_STEP_YR, ls="--", lw=0.8, color="k", alpha=0.4)
    ax.set_xlim(*xlim)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)


def make_dip_figure(
    porosity_data: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]],
    h5_dir: Optional[Path],
    out_path: Path,
    *,
    t_max: float,
    dpi: int = 200,
) -> None:
    """2-row × 6-col figure: DIP and ΔDIP time series."""
    fig, axes = plt.subplots(2, 6, figsize=(18, 6), sharex=False, sharey=False)

    for col, exp_num in enumerate(range(1, 7)):
        ax_dip = axes[0, col]
        ax_ddip = axes[1, col]
        xlim = (0.0, t_max)

        # --- Published models ---
        models_plotted: List[Tuple[str, object]] = []
        for model in KNOWN_MODELS:
            key = (model, exp_num)
            if key not in porosity_data:
                continue
            t, dip = porosity_data[key]
            t, dip = _trunc(t, dip, t_max)
            color = MODEL_COLORS[model]
            (ln,) = ax_dip.plot(t, dip, lw=1.0, color=color, alpha=0.75)
            ax_ddip.plot(t, _delta(dip), lw=1.0, color=color, alpha=0.75)
            models_plotted.append((MODEL_DISPLAY[model], ln))

        # --- Our model ---
        if h5_dir is not None:
            ours = load_our_model(h5_dir, exp_num)
            if ours is not None:
                t_o, dip_o = _trunc(ours["time_yr"], ours["dip_m"], t_max)
                (ln_ours,) = ax_dip.plot(
                    t_o, dip_o, lw=2.5, color="k", zorder=10, label="Firngrain"
                )
                ax_ddip.plot(
                    t_o, _delta(dip_o), lw=2.5, color="k", zorder=10
                )
                models_plotted.append(("Firngrain", ln_ours))

        _style_ax(ax_dip, EXP_LABELS[exp_num], "DIP (m)", xlim)
        _style_ax(ax_ddip, "", "ΔDIP (m)", xlim)

        if col == 0:
            axes[0, col].set_ylabel("DIP (m)", fontsize=8)
            axes[1, col].set_ylabel("ΔDIP (m)", fontsize=8)
        else:
            axes[0, col].set_ylabel("")
            axes[1, col].set_ylabel("")

    # Row labels
    axes[0, 0].annotate(
        "DIP", xy=(-0.25, 0.5), xycoords="axes fraction",
        fontsize=10, fontweight="bold", va="center", ha="right", rotation=90,
    )
    axes[1, 0].annotate(
        "ΔDIP", xy=(-0.25, 0.5), xycoords="axes fraction",
        fontsize=10, fontweight="bold", va="center", ha="right", rotation=90,
    )

    # Legend (use first experiment's handles)
    handles_all: List = []
    labels_all: List[str] = []
    for col in range(6):
        ax = axes[0, col]
        h, l = ax.get_legend_handles_labels()
        for lbl, ln in zip(l, h):
            if lbl not in labels_all:
                labels_all.append(lbl)
                handles_all.append(ln)

    # Build legend from MODEL_COLORS + our model
    legend_handles = []
    legend_labels = []
    for model in KNOWN_MODELS:
        # Check if model appears in any experiment
        appears = any(
            (model, e) in porosity_data for e in range(1, 7)
        )
        if appears:
            lh = plt.Line2D([0], [0], color=MODEL_COLORS[model], lw=1.5)
            legend_handles.append(lh)
            legend_labels.append(MODEL_DISPLAY[model])
    # Add our model if any h5 files found
    if h5_dir is not None:
        any_ours = any(
            (h5_dir / f"firnmice_ex{e}.h5").exists() for e in range(1, 7)
        )
        if any_ours:
            legend_handles.append(plt.Line2D([0], [0], color="k", lw=2.5))
            legend_labels.append("Firngrain")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=min(8, len(legend_labels)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=8,
        handlelength=2.0,
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_path}")


def make_bco_figure(
    bco_data: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    h5_dir: Optional[Path],
    out_path: Path,
    *,
    t_max: float,
    dpi: int = 200,
) -> None:
    """2-row × 6-col figure: BCO depth and BCO age time series."""
    fig, axes = plt.subplots(2, 6, figsize=(18, 6), sharex=False, sharey=False)

    for col, exp_num in enumerate(range(1, 7)):
        ax_depth = axes[0, col]
        ax_age = axes[1, col]
        xlim = (0.0, t_max)

        # Collect models that have BCO data for this experiment
        has_data = False
        for model in KNOWN_MODELS:
            key = (model, exp_num)
            if key not in bco_data:
                continue
            t, bco_z, bco_a = bco_data[key]
            t_tr, bco_z = _trunc(t, bco_z, t_max)
            _,    bco_a  = _trunc(t, bco_a, t_max)
            t = t_tr
            color = MODEL_COLORS[model]
            ax_depth.plot(t, bco_z, lw=1.0, color=color, alpha=0.75,
                          label=MODEL_DISPLAY[model])
            ax_age.plot(t, bco_a, lw=1.0, color=color, alpha=0.75)
            has_data = True

        # Our model
        if h5_dir is not None:
            ours = load_our_model(h5_dir, exp_num)
            if ours is not None:
                t_o = ours["time_yr"]
                bco_z_o = ours["bco_depth_m"]
                bco_a_o = ours["bco_age_yr"]
                _, bco_z_o = _trunc(t_o, bco_z_o, t_max)
                t_o, bco_a_o = _trunc(t_o, bco_a_o, t_max)
                ax_depth.plot(t_o, bco_z_o, lw=2.5, color="k", zorder=10,
                              label="Firngrain")
                ax_age.plot(t_o, bco_a_o, lw=2.5, color="k", zorder=10)
                has_data = True

        if not has_data:
            ax_depth.text(
                0.5, 0.5, "no data", ha="center", va="center",
                transform=ax_depth.transAxes, fontsize=8, color="gray"
            )
            ax_age.text(
                0.5, 0.5, "no data", ha="center", va="center",
                transform=ax_age.transAxes, fontsize=8, color="gray"
            )

        _style_ax(ax_depth, EXP_LABELS[exp_num], "BCO depth (m)", xlim)
        _style_ax(ax_age, "", "BCO age (yr)", xlim)
        ax_depth.invert_yaxis()

    axes[0, 0].annotate(
        "BCO depth", xy=(-0.30, 0.5), xycoords="axes fraction",
        fontsize=10, fontweight="bold", va="center", ha="right", rotation=90,
    )
    axes[1, 0].annotate(
        "BCO age", xy=(-0.30, 0.5), xycoords="axes fraction",
        fontsize=10, fontweight="bold", va="center", ha="right", rotation=90,
    )

    # Build legend from available models
    legend_handles = []
    legend_labels = []
    for model in KNOWN_MODELS:
        appears = any((model, e) in bco_data for e in range(1, 7))
        if appears:
            lh = plt.Line2D([0], [0], color=MODEL_COLORS[model], lw=1.5)
            legend_handles.append(lh)
            legend_labels.append(MODEL_DISPLAY[model])
    if h5_dir is not None:
        any_ours = any(
            (h5_dir / f"firnmice_ex{e}.h5").exists() for e in range(1, 7)
        )
        if any_ours:
            legend_handles.append(plt.Line2D([0], [0], color="k", lw=2.5))
            legend_labels.append("Firngrain")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=min(8, len(legend_labels)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=8,
        handlelength=2.0,
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Configuration — edit these paths as needed
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent

ZIP_DIR = _HERE / "southpole/data"          # directory with FirnMICE-*.zip files
H5_DIR: Optional[Path] = _HERE / "southpole/results/firnmice"  # firngrain h5 outputs
OUT_DIR = _HERE / "southpole/results/firnmice/figures"         # output directory for PNG files
T_MAX = 500.0                               # truncate time axis to this many years
DPI = 200

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading published data from: {ZIP_DIR}")
    porosity_data = load_published_porosity(ZIP_DIR)
    bco_data = load_published_bco(ZIP_DIR)
    print(f"  Found DIP data for {len(porosity_data)} (model, exp) pairs")
    print(f"  Found BCO data for {len(bco_data)} (model, exp) pairs")

    if H5_DIR is not None:
        print(f"Looking for our model output in: {H5_DIR}")

    print(f"Generating figures (t_max={T_MAX} yr) ...")

    make_dip_figure(porosity_data, H5_DIR, OUT_DIR / "firnmice_dip_comparison.png",
                    t_max=T_MAX, dpi=DPI)
    make_bco_figure(bco_data, H5_DIR, OUT_DIR / "firnmice_bco_comparison.png",
                    t_max=T_MAX, dpi=DPI)

    print("[done]")
