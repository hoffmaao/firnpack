"""Plotting for firn-column diagnostics.

The case studies all draw the same few things: a depth profile of observations
with 1-sigma bars against the model's prediction, and a standardized-residual
panel. Before this module each case study retyped the palette and the
depth-axis convention by hand -- the same four hex codes appeared in seven
figure scripts, which is how they drift apart.

Depth is DOWN everywhere here: `depth_axis` inverts y once so no caller has to
remember. Nothing in this module imports Firedrake; it takes arrays, so figures
can be rebuilt from a results JSON without a solve.

Palette note: the categorical hues below are validated, not chosen by eye --
worst adjacent CVD separation is dE 35.2 (protan) against a target of 12, all
five sit inside the L 0.43-0.77 band, and all clear 3:1 contrast on a light
surface. Do not substitute hues without re-validating; the published figures
use these.
"""
from __future__ import annotations

import colorsys
import json
import os
from pathlib import Path

import numpy as np

# ---- ink -------------------------------------------------------------------
# Text and axes wear ink, never a series colour; a coloured mark beside a label
# is what carries identity.
OBS = "#111827"        # observations: near-black, the measured thing
MODEL = "#2563EB"      # model prediction
RESID = "#EA580C"      # residuals
GRID = "#9CA3AF"       # recessive: grid and rules only

# ---- categorical: one fixed hue per observable, assigned in order ----------
# Never cycle this. A new observable takes the next unused hue; a per-site
# family (velocity) ramps off its base hue instead of stealing a new one, so
# "same kind, different column" reads as a family rather than five unrelated
# quantities.
BLOCK_COLORS = {
    "rho": "#2563EB",
    "age": "#EA580C",
    "dage": "#059669",
    "T": "#7C3AED",
    "v": "#B45309",
    "seas": "#0891B2",
    "comp": "#DB2777",
}

BLOCK_UNITS = {
    "rho": "density (kg m$^{-3}$)",
    "age": "age (yr)",
    "dage": "d(age)/dz (yr m$^{-1}$)",
    "T": "T (°C)",
    "v": "w (m yr$^{-1}$)",
    "seas": "ln amplitude ratio",
    "comp": "compaction rate (m yr$^{-1}$)",
}


# Markers carry family membership as a SECOND channel alongside the lightness
# ramp. A five-step ramp inside one hue has adjacent steps ~dE 11 apart under
# protanopia -- inside the 8-12 floor band, which is only legal with secondary
# encoding. Shape is that encoding, and unlike colour it survives greyscale
# print and every CVD type.
FAMILY_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def newest_map(out_dir, need=("m_map",), env=None, pattern="*.json"):
    """Newest-by-mtime results JSON under `out_dir` carrying every key in `need`.

    A fresh inversion becomes the plotted one regardless of how its tag sorts.
    The key filter is what makes this safe: `out_dir` also collects artifacts
    that are not MAPs (``<tag>_hessian.json``, ``*_marginals.json``,
    reanalysis dumps), and a name-only match on those hands the caller a dict
    with no ``m_map``. Returns None when nothing qualifies, so the caller can
    emit its own "run run.py first" message.
    """
    if env and os.environ.get(env):
        return Path(os.environ[env])
    best = None
    for p in Path(out_dir).glob(pattern):
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(d, dict) and all(d.get(k) is not None for k in need):
            if best is None or p.stat().st_mtime > best.stat().st_mtime:
                best = p
    return best


def base_label(label: str) -> str:
    """'v_x11n6' -> 'v'. Per-site velocity blocks share one base observable."""
    return label.split("_", 1)[0]


def block_marker(index: int = 0, n: int = 1) -> str:
    """Marker for a family member; 'o' when the block stands alone."""
    return "o" if n <= 1 else FAMILY_MARKERS[index % len(FAMILY_MARKERS)]


def block_color(label: str, index: int = 0, n: int = 1) -> str:
    """Colour for an observation block.

    A per-site block (``v_x11n6``) gets a lightness step off its base hue, so a
    five-site figure shows distinguishable shades that still read as one
    observable. `index`/`n` place it within its family. Lightness is the right
    axis for this: it survives every CVD type, so the family stays separable
    where five unrelated hues would not.

    The spread is deliberately narrow. `v` (#B45309) and `age` (#EA580C) are
    nearly the same hue (26° vs 21°), so a wide lightness ramp off `v` walks
    into `age`'s identity -- at ±0.18 the light end came out #F47E25, which is
    `age` for practical purposes. Keep it tight, or move the family to a hue of
    its own and re-validate.
    """
    base = BLOCK_COLORS.get(base_label(label), BLOCK_COLORS["v"])
    if n <= 1 or "_" not in label:
        return base
    r, g, b = (int(base[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    lo, hi = max(0.32, l - 0.10), min(0.62, l + 0.10)
    l_i = lo + (hi - lo) * (index / max(n - 1, 1))
    r, g, b = colorsys.hls_to_rgb(h, l_i, s)
    return "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))


def depth_axis(ax, hmax: float = 132.0, xlabel: str | None = None):
    """The one convention every firn profile shares: depth increases downward."""
    ax.invert_yaxis()
    ax.set_ylim(hmax, 0)
    ax.set_ylabel("depth (m)")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3, color=GRID)
    return ax


def profile(ax, depth, obs, sig=None, pred=None, *, title=None, xlabel=None,
            model_label="model", obs_label="observations ±1σ", hmax=132.0):
    """Observations with 1-sigma bars against a model profile.

    Markers stay small deliberately: these panels carry 40-124 points and a
    dashboard-sized marker would occlude the structure being judged.
    """
    depth, obs = np.asarray(depth), np.asarray(obs)
    if sig is not None:
        ax.errorbar(obs, depth, xerr=np.asarray(sig), fmt="o", ms=3.5,
                    color=OBS, elinewidth=0.7, capsize=0, alpha=0.75,
                    label=obs_label)
    else:
        ax.plot(obs, depth, "o", ms=3.5, color=OBS, alpha=0.75, label=obs_label)
    if pred is not None:
        pred = np.asarray(pred)
        o = np.argsort(depth)
        ax.plot(pred[o], depth[o], "-", color=MODEL, lw=2, label=model_label)
    depth_axis(ax, hmax=hmax, xlabel=xlabel)
    if title:
        ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=8)
    return ax


def residual_panel(ax, blocks, *, hmax=132.0, xlim=3.2, title=None):
    """Standardized residuals for every block on one depth axis.

    `blocks` maps label -> (depth, residual_in_sigma). The shaded band is ±1σ:
    a well-specified error model puts about two thirds of the points inside it,
    which is what this panel is for.

    The legend is parked in a reserved strip to the right of the data rather
    than left to `loc="best"`, which puts it over the middle of the scatter --
    i.e. on top of the very residuals the panel exists to show.
    """
    fam = {}
    for lab in blocks:
        fam.setdefault(base_label(lab), []).append(lab)
    for base, labs in fam.items():
        for i, lab in enumerate(sorted(labs)):
            d, r = blocks[lab]
            ax.plot(np.asarray(r), np.asarray(d), block_marker(i, len(labs)),
                    ms=3.5, color=block_color(lab, i, len(labs)), alpha=0.75,
                    mew=0, label=lab)
    ax.axvline(0, color=OBS, lw=0.8)
    ax.axvspan(-1, 1, color=GRID, alpha=0.25, lw=0)
    # reserve clear space on the right for the legend; data still spans ±xlim
    ax.set_xlim(-xlim, xlim * 1.62)
    ax.axvline(xlim, color=GRID, lw=0.6, ls=":")
    depth_axis(ax, hmax=hmax, xlabel="(model − obs) / σ")
    ax.set_title(title or "standardized residuals", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92,
              borderpad=0.4, handletextpad=0.4)
    return ax


def save(fig, path, dpi: int = 140):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    return path
