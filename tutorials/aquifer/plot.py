"""tutorials/aquifer/plot.py - is the firn aquifer perennial?

A pure reader: no Firedrake, no solve. Everything comes from the JSON run.py
wrote, so figures are cheap to rebuild and are never committed.

Three figures, each answering a different part of "perennial":

``aquifer_depth_time``   theta(z, t) per case. The definitive one: a perennial
                        aquifer is a wet band that survives the winter *at
                        depth*, which a column-total series cannot show - total
                        storage looks the same whether water persisted or
                        drained away and returned next summer.
``aquifer_persistence``  column storage through time with the winter minima
                        marked, plus the annual winter minimum. Perennial means
                        the minima stay above zero, so they are the quantity to
                        plot, not the peaks.
``aquifer_fate``         where the melt went: refrozen vs retained.

Run: python tutorials/aquifer/plot.py
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
from config import FIGURES, OUTPUT

ORDER = ["se_greenland", "low_accum", "cold", "low_melt"]
NICE = {"se_greenland": "SE Greenland", "low_accum": "Low accumulation",
        "cold": "Cold", "low_melt": "Low melt"}

# Categorical slots 1-4, assigned in fixed order and validated for CVD
# separation against a white surface. Aqua and yellow fall below 3:1 contrast,
# so every series also carries a direct label and the table below is printed.
CAT = {"se_greenland": "#2a78d6", "low_accum": "#eb6834",
       "cold": "#1baf7a", "low_melt": "#eda100"}

# Sequential single hue, light -> dark, for water content (a magnitude).
WATER = LinearSegmentedColormap.from_list(
    "water", ["#f4f7fb", "#c5d9f1", "#7fb0e3", "#2a78d6", "#12386b"])

INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#e3e3df"


def load():
    runs = {}
    for name in ORDER:
        path = OUTPUT / f"aquifer_{name}.json"
        if path.exists():
            runs[name] = json.loads(path.read_text())
    if not runs:
        raise SystemExit(f"no runs in {OUTPUT}; run tutorials/aquifer/run.py first")
    return runs


def _tidy(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _winter_minima(t, y):
    """Per-water-year minimum, taken over the cold half of each year.

    The melt season is centred on t = 0.5, so a winter straddles the year
    boundary; taking the minimum over a calendar year would find the same
    trough twice and miss the one that matters.
    """
    yrs, mins = [], []
    for y0 in range(int(np.floor(t.max()))):
        sel = (t >= y0 + 0.75) & (t < y0 + 1.25)
        if sel.sum():
            yrs.append(y0 + 1)
            mins.append(y[sel].min())
    return np.asarray(yrs), np.asarray(mins)


# ----------------------------------------------------------------------
def depth_time_figure(runs):
    """theta(z, t): does the wet layer survive the winters?

    Skips any case whose JSON predates the depth-time snapshots, so a partially
    regenerated output directory still plots what it has.
    """
    runs = {k: v for k, v in runs.items() if "theta_profiles" in v}
    if not runs:
        return None
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.0), sharey=True,
                             squeeze=False)
    axes = axes[0]

    vmax = max(np.asarray(r["theta_profiles"]).max() for r in runs.values())
    for ax, (name, r) in zip(axes, runs.items()):
        t = np.asarray(r["time_years"])
        d = np.asarray(r["depth_m"])
        th = np.asarray(r["theta_profiles"]).T          # depth x time
        # Square-root norm. The saturated body reaches theta ~ 0.28 while the
        # shallow seasonal water that actually shows survival-through-winter is
        # 0.01-0.05; on a linear ramp the aquifer sets vmax and flattens the
        # very signal the figure is about. Still one hue, still monotonic in
        # magnitude - only the spacing of the steps changes.
        im = ax.pcolormesh(t, d, th, cmap=WATER,
                           norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax),
                           shading="auto", rasterized=True)
        # Shade the winters. Perenniality is a claim about what is still wet
        # inside these bands, so mark them rather than the year boundaries.
        for y0 in range(int(np.floor(t.max())) + 1):
            ax.axvspan(y0 + 0.75, min(y0 + 1.25, t.max()),
                       color="#1a1a19", alpha=0.055, lw=0, zorder=3)
        ax.set_title(f"{NICE[name]}"
                     f"{'  - perennial' if r['persists'] else '  - refreezes away'}",
                     fontsize=10, color=INK)
        ax.set_xlabel("time (years)", fontsize=9, color=MUTED)
        _tidy(ax)
        ax.grid(False)
    axes[0].set_ylabel("depth (m)", fontsize=9, color=MUTED)
    axes[0].invert_yaxis()

    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label("liquid water content  $\\theta$  (m$^3$ m$^{-3}$, sqrt scale)",
                 fontsize=9, color=MUTED)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("A perennial aquifer is a wet band that survives the winter",
                 fontsize=12, color=INK, x=0.5, y=1.00)
    fig.text(0.5, 0.945, "shaded bands are winters", ha="center",
             fontsize=9, color=MUTED)
    out = FIGURES / "aquifer_depth_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def persistence_figure(runs):
    """Storage through time with the winter minima called out."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.4),
                                  gridspec_kw={"height_ratios": [2, 1]})

    for name, r in runs.items():
        t = np.asarray(r["time_years"])
        liq = np.asarray(r["storage_kg_m2"]) - r["storage0_kg_m2"]
        c = CAT[name]
        ax.plot(t, liq, color=c, lw=2.0, solid_capstyle="round")
        yrs, mins = _winter_minima(t, liq)
        ax.plot(yrs, mins, "o", ms=8, mfc="white", mec=c, mew=2.0, zorder=5)
        # direct label: identity never rests on colour alone
        ax.annotate(NICE[name], xy=(t[-1], liq[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    fontsize=9, color=INK)

    ax.axhline(0.0, color=MUTED, lw=1.0, ls="--")
    ax.set_ylabel("liquid water above the retention trace (kg m$^{-2}$)",
                  fontsize=9, color=MUTED)
    ax.set_title("Open circles are winter minima - perennial means they stay above zero",
                 fontsize=11, color=INK, loc="left")
    _tidy(ax)
    ax.set_xlim(0, None)
    ax.margins(x=0.16)

    width = 0.8 / max(len(runs), 1)
    for i, (name, r) in enumerate(runs.items()):
        t = np.asarray(r["time_years"])
        liq = np.asarray(r["storage_kg_m2"]) - r["storage0_kg_m2"]
        yrs, mins = _winter_minima(t, liq)
        ax2.bar(yrs + (i - (len(runs) - 1) / 2) * width, mins, width * 0.92,
                color=CAT[name], label=NICE[name], linewidth=0)
    ax2.axhline(0.0, color=MUTED, lw=1.0)
    ax2.set_xlabel("winter (year)", fontsize=9, color=MUTED)
    ax2.set_ylabel("winter minimum (kg m$^{-2}$)", fontsize=9, color=MUTED)
    ax2.set_yscale("symlog", linthresh=10)
    ax2.legend(frameon=False, fontsize=9, ncol=len(runs), loc="upper left")
    _tidy(ax2)

    fig.tight_layout()
    out = FIGURES / "aquifer_persistence.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


def fate_figure(runs):
    """Where the melt went. One axis: both quantities are kg/m^2."""
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    names = list(runs)
    y = np.arange(len(names))
    refroze = np.array([runs[n]["refreeze_cum_kg_m2"][-1] for n in names])
    melt = np.array([runs[n]["melt_cum_kg_m2"][-1] for n in names])
    retained = np.clip(melt - refroze, 0, None)

    ax.barh(y, refroze, 0.55, color="#c5d9f1", label="refrozen", linewidth=0)
    ax.barh(y, retained, 0.55, left=refroze + 0.004 * melt.max(),
            color="#2a78d6", label="retained as liquid", linewidth=0)
    for i, n in enumerate(names):
        pct = 100.0 * refroze[i] / max(melt[i], 1e-9)
        ax.annotate(f"{pct:.0f}% refrozen", xy=(melt[i], y[i]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK)
    ax.set_yticks(y, [NICE[n] for n in names], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("cumulative meltwater over the run (kg m$^{-2}$)",
                  fontsize=9, color=MUTED)
    ax.set_title("Fate of the melt", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _tidy(ax)
    ax.margins(x=0.20)
    fig.tight_layout()
    out = FIGURES / "aquifer_fate.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


def table(runs):
    """The table view the contrast WARN on two palette slots obliges."""
    print(f"{'case':<18} {'melt':>8} {'refrozen':>9} {'%':>6} {'liquid':>8} "
          f"{'winter min':>11} {'perennial':>10}")
    for name, r in runs.items():
        t = np.asarray(r["time_years"])
        liq = np.asarray(r["storage_kg_m2"]) - r["storage0_kg_m2"]
        _, mins = _winter_minima(t, liq)
        melt = r["melt_cum_kg_m2"][-1]
        refroze = r["refreeze_cum_kg_m2"][-1]
        print(f"{NICE[name]:<18} {melt:8.0f} {refroze:9.0f} "
              f"{100*refroze/max(melt,1e-9):6.1f} {liq[-1]:8.0f} "
              f"{(mins[-3:].min() if mins.size else float('nan')):11.1f} "
              f"{'YES' if r['persists'] else 'no':>10}")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = load()
    table(runs)
    for out in (depth_time_figure(runs), persistence_figure(runs),
                fate_figure(runs)):
        print(f"Saved {out}" if out else
              "(depth-time figure skipped: no run carries theta profiles yet)")
