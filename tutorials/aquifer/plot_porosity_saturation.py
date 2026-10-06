"""Porosity and saturation as depth-time fields.

Two views of the same column, side by side per run:

``porosity``    phi(z, t) = 1 - rho/rho_i, the pore space itself. It is the
                container: densification closes it from the top down, and an
                aquifer can only exist where some is left. Bubble close-off,
                drawn as a contour, is the depth below which the pore space
                stops being connected, and is the floor a water table should
                rest on.

``saturation``  S(z, t) = theta/phi, how much of that container holds water.
                The aquifer is the region at S -> 1: its top edge is the water
                table and its thickness the aquifer thickness, both directly
                comparable with the radar and core observations.

Plotting them together answers a question neither answers alone: whether a
water table sits where it does because that is where the water reached, or
because that is where the pore space ran out.

Run: python tutorials/aquifer/plot_porosity_saturation.py [run_name ...]
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
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

sys.path.insert(0, str(Path(__file__).parent))
from config import ERA5_EXPERIMENTS, FIGURES, OUTPUT

INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#e3e3df"
RHO_I, CLOSEOFF = 917.0, 830.0
# Two sequential single-hue ramps: both fields are magnitudes, and keeping
# them in different hues stops the two rows being read as one field.
PORE = LinearSegmentedColormap.from_list(
    "pore", ["#fdf6ec", "#f6dcb0", "#e8b06a", "#c87f33", "#8a4f12"])
WATER = LinearSegmentedColormap.from_list(
    "water", ["#f7fafd", "#cddff2", "#82b2e5", "#2a78d6", "#123a6b"])
# Observed geometry: water table from Montgomery et al. 2017, aquifer base
# 27.7 m from the same survey.
OBS_TABLE, OBS_BASE = (10.0, 20.0), 27.7


def _closeoff(depth, rho, threshold=CLOSEOFF):
    """Shallowest depth below which density stays at or above the threshold.

    Not the shallowest crossing. A column that grows ice lenses - which this
    one does, every spring - has isolated nodes above 830 kg/m^3 metres above
    any sealed horizon, and the first-crossing rule reports one of those: it
    put close-off at 8.1 m in the strongest-melt experiment, where the sealed
    horizon is nearer 45 m. Mirrors firnpack.firnmice._closeoff_depth, which
    cannot be imported here because it pulls in Firedrake and these plot
    scripts are specified to run without it.
    """
    below = np.asarray(rho) >= threshold
    if not below.any() or not below[-1]:
        return float("nan")
    k = len(below)
    while k > 0 and below[k - 1]:
        k -= 1
    return float(depth[k])


def _tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)


def load(names):
    names = list(names) if names else list(ERA5_EXPERIMENTS)
    runs = {}
    for name in names:
        path = OUTPUT / f"aquifer_{name}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        if r.get("S_profiles") and r.get("rho_profiles"):
            runs[name] = r
    if not runs:
        raise SystemExit("no runs with profiles; run run_era5.py first")
    return runs


def figure(runs):
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 7.6), squeeze=False,
                             sharex="col", sharey=True)

    for k, (name, r) in enumerate(runs.items()):
        t = np.asarray(r["time_years"])
        d = np.asarray(r["depth_m"])
        rho = np.asarray(r["rho_profiles"]).T           # depth x time
        S = np.asarray(r["S_profiles"]).T
        phi = 1.0 - rho / RHO_I

        ax = axes[0][k]
        im_p = ax.pcolormesh(t, d, phi, cmap=PORE, vmin=0.0, vmax=0.6,
                             shading="auto", rasterized=True)
        # The rho = 830 contour. It traces the sealed horizon AND every ice
        # lens above it, which is why it breaks into diagonal streaks: each
        # is a lens formed in one melt season and advected down. The sealed
        # horizon proper is the table's close-off column.
        ax.contour(t, d, rho, levels=[CLOSEOFF], colors="#1a1a19",
                   linewidths=1.0, linestyles="--", alpha=0.8)
        ax.set_title(name.replace("_", " "), fontsize=10.5, color=INK)
        if k == 0:
            ax.set_ylabel("depth (m)", fontsize=9, color=MUTED)
        # set_ylim, not invert_yaxis: the axes are shared, so inverting once
        # per row inverts twice and puts the surface at the bottom
        ax.set_ylim(d.max(), 0.0)
        _tidy(ax)

        ax = axes[1][k]
        im_s = ax.pcolormesh(t, d, S, cmap=WATER, norm=PowerNorm(0.35, vmin=0.0, vmax=1.0),
                           # Power norm, not linear. An unsaturated percolation
                           # zone sits at S ~ 0.02-0.08 - a few percent of pore
                           # space by volume, which is what is observed - and a
                           # linear 0-1 scale renders all of it as blank white,
                           # hiding the very transport the figure is about.
                             shading="auto", rasterized=True)
        ax.contour(t, d, S, levels=[0.5], colors="#d94801", linewidths=1.3)
        ax.axhspan(OBS_TABLE[0], OBS_TABLE[1], color="#d94801", alpha=0.16,
                   lw=0, zorder=4)
        ax.axhline(OBS_BASE, color="#d94801", lw=1.0, ls=":", zorder=5)
        ax.set_xlabel("time (years)", fontsize=9, color=MUTED)
        if k == 0:
            ax.set_ylabel("depth (m)", fontsize=9, color=MUTED)
        ax.set_ylim(d.max(), 0.0)
        _tidy(ax)

    for im, axrow, label in ((im_p, axes[0], "porosity  $\\phi = 1 - \\rho/\\rho_i$"),
                             (im_s, axes[1], "saturation  S = $\\theta/\\phi$")):
        cb = fig.colorbar(im, ax=list(axrow), fraction=0.020, pad=0.012)
        cb.set_label(label, fontsize=9, color=MUTED)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_visible(False)

    fig.suptitle("The pore space, and how much of it holds water\n"
                 "dashed: $\\rho$ = 830 kg m$^{-3}$, the sealed horizon and the "
                 "ice lenses above it;  "
                 "orange band: observed water table 10-20 m;  "
                 "dotted: observed aquifer base 27.7 m",
                 fontsize=11.5, color=INK, y=0.99)
    out = FIGURES / "aquifer_porosity_saturation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def table(runs):
    print(f"{'run':<20} {'phi(10m)':>9} {'phi(30m)':>9} {'close-off':>10} "
          f"{'table':>7} {'thick':>7} {'S max':>6}")
    for name, r in runs.items():
        d = np.asarray(r["depth_m"])
        rho = np.asarray(r["rho_profiles"])[-1]
        S = np.asarray(r["S_profiles"])[-1]
        phi = 1.0 - rho / RHO_I
        co = _closeoff(d, rho)
        wet = np.flatnonzero(S >= 0.5)
        if wet.size:
            edges = np.flatnonzero(np.diff(np.concatenate(
                [[0], (S >= 0.5).view(np.int8), [0]])))
            starts, ends = edges[::2], edges[1::2]
            tbl, base = d[starts[-1]], d[ends[-1] - 1]   # deepest run
        else:
            tbl = base = float("nan")
        j10, j30 = int(np.argmin(abs(d - 10))), int(np.argmin(abs(d - 30)))
        print(f"{name:<20} {phi[j10]:9.3f} {phi[j30]:9.3f} "
              f"{co:10.1f} "
              f"{tbl:7.1f} {base - tbl if np.isfinite(tbl) else 0.0:7.1f} "
              f"{S.max():6.2f}")
    print(f"\nobserved: water table {OBS_TABLE[0]:.0f}-{OBS_TABLE[1]:.0f} m, "
          f"base {OBS_BASE:.1f} m, thickness ~11.5 m")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = load(sys.argv[1:])
    table(runs)
    print(f"Saved {figure(runs)}")
