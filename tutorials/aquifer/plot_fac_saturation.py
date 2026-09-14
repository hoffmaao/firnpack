"""Firn air content through time, and saturation as a depth-time field.

Two views of the same column:

``fac``          Firn air content - the depth-integrated porosity, in metres of
                 air. This is the pore volume available to hold meltwater, and
                 an aquifer filling shows up as the part of it that is below
                 the water table being consumed. Plotted as the total and as
                 the part still above the water table, because the total also
                 falls through ordinary densification and on its own cannot
                 distinguish compaction from flooding.

``saturation``   S(z, t) = theta / porosity. The aquifer is the region at
                 S -> 1: its top edge is the water table and its thickness is
                 the aquifer thickness, both directly comparable with the
                 radar and core observations (water table 10-20 m, base near
                 28 m; Montgomery et al. 2017).

Saturation is used rather than water content because theta alone confounds two
things - a cell can hold little water because it is dry or because it has
little pore space left. Dividing by porosity separates them, which matters in
a column whose porosity is halving with depth.

Run: python tutorials/aquifer/plot_fac_saturation.py [run_name ...]
     (default: the experiments in config.ERA5_EXPERIMENTS that have output)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from config import ERA5_EXPERIMENTS, FIGURES, OUTPUT

INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#e3e3df"
# Sequential single hue: saturation is a magnitude.
WATER = LinearSegmentedColormap.from_list(
    "water", ["#f7fafd", "#cddff2", "#82b2e5", "#2a78d6", "#123a6b"])
# Observed aquifer geometry (Koenig 2014; Montgomery 2017)
OBS_TABLE, OBS_BASE = (10.0, 20.0), 27.7


def _aquifer_zone(depth, sat, threshold=0.5):
    """Top and bottom of the saturated zone standing on the base, or nans.

    The column is confined, so the aquifer is by construction the saturated
    run that reaches the bottom of the domain. A perched melt lens with dry
    firn beneath it is not an aquifer and reports no water table, and a lens
    above a real aquifer is ignored.

    Deliberate duplicate of ``firnpack.aquifer.water_table_depth``, which
    applies the same rule; this copy also returns the bottom, so the thickness
    printed beside the water table describes the same zone. They cannot share
    code: this script is a pure reader of ``output/`` and runs without
    Firedrake, while importing ``firnpack.aquifer`` pulls in
    ``firnpack.models.firn``, which imports Firedrake unconditionally. Change
    one and change the other.
    """
    wet = np.asarray(sat) >= threshold
    if not wet.size or not wet[-1]:
        return float("nan"), float("nan")
    edges = np.flatnonzero(np.diff(np.concatenate([[0], wet.view(np.int8), [0]])))
    starts, ends = edges[::2], edges[1::2]
    return float(depth[starts[-1]]), float(depth[ends[-1] - 1])


def _tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)


def load(names):
    """The ERA5 experiments by default, in the order config lists them.

    Any run name can be asked for explicitly, but the default is the
    experiment table rather than every JSON in output/, so the synthetic
    contrast cases plot.py owns do not land in this figure.
    """
    names = list(names) if names else list(ERA5_EXPERIMENTS)
    runs = {}
    for name in names:
        path = OUTPUT / f"aquifer_{name}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        if "S_profiles" in r and len(r["S_profiles"]):
            runs[name] = r
    if not runs:
        raise SystemExit("no runs with S_profiles; re-run the driver")
    return runs


def figure(runs):
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(5.6 * n, 8.2), squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1.7]})

    for k, (name, r) in enumerate(runs.items()):
        t = np.asarray(r["time_years"])
        d = np.asarray(r["depth_m"])
        S = np.asarray(r["S_profiles"]).T          # depth x time

        # --- firn air content ---
        ax = axes[0][k]
        fac = np.asarray(r["fac_m"])
        ax.plot(t, fac, color="#1a1a19", lw=1.8, label="total")
        if "fac_above_wt_m" in r and len(r["fac_above_wt_m"]):
            ax.plot(t, np.asarray(r["fac_above_wt_m"]), color="#2a78d6",
                    lw=1.8, label="above the water table")
        ax.set_ylabel("firn air content (m)", fontsize=9, color=MUTED)
        ax.set_title(name.replace("_", " "), fontsize=10.5, color=INK)
        ax.grid(color=GRID, lw=0.6, alpha=0.9)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
        _tidy(ax)

        # --- saturation, depth vs time ---
        ax = axes[1][k]
        im = ax.pcolormesh(t, d, S, cmap=WATER, vmin=0.0, vmax=1.0,
                           shading="auto", rasterized=True)
        cs = ax.contour(t, d, S, levels=[0.5], colors="#d94801", linewidths=1.4)
        ax.axhspan(OBS_TABLE[0], OBS_TABLE[1], color="#d94801", alpha=0.16,
                   lw=0, zorder=4)
        ax.axhline(OBS_BASE, color="#d94801", lw=1.0, ls=":", zorder=5)
        ax.set_xlabel("time (years)", fontsize=9, color=MUTED)
        if k == 0:
            ax.set_ylabel("depth (m)", fontsize=9, color=MUTED)
        ax.invert_yaxis()
        _tidy(ax)
        del cs

    # Beside the saturation row only, which is all it describes. The bar's
    # length is capped at aspect * fraction * row width, so with one panel the
    # default aspect leaves it short and centred up into the FAC axis.
    cb = fig.colorbar(im, ax=list(axes[1]), fraction=0.022, pad=0.015,
                      aspect=40)
    cb.set_label("saturation  S = $\\theta$ / porosity", fontsize=9, color=MUTED)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("Pore space and how much of it holds water\n"
                 "orange band: observed water table 10-20 m (Montgomery "
                 "et al. 2017); dotted: observed aquifer base 27.7 m; "
                 "orange line: modelled S = 0.5",
                 fontsize=11.5, color=INK, y=0.99)
    out = FIGURES / "aquifer_fac_saturation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def table(runs):
    print(f"{'run':<26} {'FAC0':>7} {'FACend':>7} {'drawn':>7} "
          f"{'S max':>6} {'table':>7} {'thick':>7} {'refroze':>8} {'drained':>8}")
    for name, r in runs.items():
        d = np.asarray(r["depth_m"])
        S = np.asarray(r["S_profiles"])
        fac = np.asarray(r["fac_m"])
        last = S[-1]
        tbl, base = _aquifer_zone(d, last)
        thick = (base - tbl) if np.isfinite(tbl) else 0.0
        melt = max(float(r["melt_cum_kg_m2"][-1]), 1e-9)
        refr = 100.0 * float(r["refreeze_cum_kg_m2"][-1]) / melt
        drn = 100.0 * float(r.get("drained_kg_m2", 0.0)) / melt
        print(f"{name:<26} {fac[0]:7.2f} {fac[-1]:7.2f} {fac[0]-fac[-1]:7.2f} "
              f"{last.max():6.2f} {tbl:7.1f} {thick:7.1f} {refr:7.0f}% {drn:7.0f}%")
    print(f"\nobserved: water table {OBS_TABLE[0]:.0f}-{OBS_TABLE[1]:.0f} m, "
          f"base {OBS_BASE:.1f} m, thickness ~11.5 m")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = load(sys.argv[1:])
    table(runs)
    print(f"Saved {figure(runs)}")
