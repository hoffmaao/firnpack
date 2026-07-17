"""South Pole figures, rebuilt from output/.

One of the tutorial's two scripts: run.py computes and writes output/, this
draws figures/ from them. It reads JSON and nothing else -- no Firedrake, no
solve -- so it runs in seconds on any machine and cannot silently disagree with
the run it is drawing.

That is only possible because run.py stores the per-block obs/sig/pred arrays
alongside the parameters. The figure scripts this replaces each re-ran the
engine forward to regenerate predictions, which meant "plot" quietly meant
"solve", and every one of them had to reproduce run.py's configuration exactly
to do it.

Each figure is skipped, with a note, when its inputs are absent -- a fresh clone
has no output/ at all, and long runs land one at a time.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 python tutorials/southpole/plot.py
"""
from __future__ import annotations

import json
import math
import string
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from firnpack import plot as fp

HERE = Path(__file__).parent
RESULTS = HERE / "output"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

HMAX = 132.0  # the H_col=130 m cut, plus a margin

# Physical constants for display units. The engine stores enthalpy and age in
# SI; the figures are read by humans.
from firnpack.models.firn import FirnParameters  # noqa: E402
from firnpack.constants import year as YEAR_S  # noqa: E402

_p = FirnParameters()
C_I, T_REF = float(_p.c_i), float(_p.T_ref)

TITLES = {
    "rho": "density (SP19)",
    "age": "depth–age (SP19)",
    "dage": "layer gradient d(age)/dz",
    "T": "borehole temperature",
    "v": "vertical velocity (ApRES)",
}


def _to_display(block):
    """Convert one stored block to display units.

    The engine assimilates enthalpy and age in SI, so T and age blocks carry
    J/kg and seconds; everything downstream of here is what a reader sees.
    """
    d = np.asarray(block["depths"], float)
    o = np.asarray(block["obs"], float)
    s = np.asarray(block["sig"], float)
    p = np.asarray(block["pred"], float)
    lab = block["label"]

    if lab == "age":
        o, s, p = o / YEAR_S, s / YEAR_S, p / YEAR_S
    elif lab == "T":
        o = o / C_I + T_REF - 273.15
        p = p / C_I + T_REF - 273.15
        s = s / C_I
    return d, o, s, p, fp.BLOCK_UNITS.get(fp.base_label(lab), lab)


def _title(lab):
    t = TITLES[fp.base_label(lab)]
    return f"{t} — {lab[2:]}" if lab.startswith("v_") else t


def _results_with_blocks():
    """Every results JSON that carries the arrays a misfit figure needs.

    A MAP written before the engine stored blocks has parameters only. Notably
    the frozen sp_joint_r8.json is one of those, and it cannot be backfilled:
    it predates the current velocity physics, so a forward at its parameters
    replays at J=146.9 against its stored J=81.3. It is a warm-start input, not
    a plottable result.
    """
    out = {}
    for p in sorted(RESULTS.glob("*.json")):
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(d, dict) and d.get("obs") and d.get("profiles"):
            out[p.stem] = d
    return out


def figure_misfits(results):
    """Observations vs the model, per observable, plus standardized residuals.

    The paper's fit-quality figure. The panel grid follows the blocks actually
    present, so restoring the age block or adding a velocity site never leaves
    a hole or a mislabelled panel.
    """
    if not results:
        print("  sp_misfits: SKIP — no output/*.json carries obs blocks.\n"
              "    Run run.py first. (sp_joint_r8.json is a warm-start input:\n"
              "    it predates the current velocity physics and is not replayable.)")
        return

    tag, r = sorted(results.items())[-1]
    blocks = {b["label"]: b for b in r["obs"]}
    order = [l for l in ("rho", "age", "dage", "T") if l in blocks] + \
            sorted(l for l in blocks if l.startswith("v"))

    npan = len(order) + 1
    nc = min(3, npan)
    nr = math.ceil(npan / nc)
    fig, AX = plt.subplots(nr, nc, figsize=(5 * nc, 4.5 * nr), squeeze=False)
    axes = list(AX.flat)

    for i, (ax, lab) in enumerate(zip(axes, order)):
        d, o, s, p, unit = _to_display(blocks[lab])
        rms = r["diag"].get("rms_" + lab, float("nan"))
        fp.profile(ax, d, o, sig=s, pred=p, hmax=HMAX, xlabel=unit,
                   model_label=f"model ({tag})",
                   title=f"({string.ascii_lowercase[i]}) {_title(lab)} — rms {rms:.2f}σ")

    resid = {}
    for lab in order:
        d, o, s, p, _ = _to_display(blocks[lab])
        resid[lab] = (d, (p - o) / s)
    for j in range(npan, len(axes)):
        axes[j].set_visible(False)
    fp.residual_panel(
        axes[len(order)], resid, hmax=HMAX,
        title=f"({string.ascii_lowercase[len(order)]}) standardized residuals, all observables",
    )

    fig.suptitle(f"South Pole joint assimilation — observations vs the {tag} MAP "
                 f"(J = {r['J']:.1f})", fontsize=13)
    print(f"  sp_misfits: {fp.save(fig, FIGS / 'sp_misfits.png')}")
    plt.close(fig)


def figure_resolution():
    """Do the estimates move when the discretisation is refined?

    Needs only m_map, so it works for MAPs written before the engine stored
    blocks -- including r8.
    """
    R_GAS, T_SITE = 8.314, 222.35  # -50.8 C
    cases = [("r8 (NZ=100, dt=5)", "sp_joint_r8", fp.OBS, "-"),
             ("NZ=200, dt=5", "sp_engine_nz200", fp.MODEL, "--"),
             ("NZ=100, dt=2.5", "sp_engine_dt25", fp.RESID, ":")]

    def rates(m, b=0.09):
        return (m["hl_k0"] * math.exp(-m["hl_Ea1"] / (R_GAS * T_SITE)) * b,
                m["hl_k1"] * math.exp(-m["hl_Ea2"] / (R_GAS * T_SITE)) * math.sqrt(b))

    loaded = []
    for lab, stem, c, ls in cases:
        p = RESULTS / f"{stem}.json"
        if p.exists():
            loaded.append((lab, json.load(open(p)), c, ls))
    if len(loaded) < 2:
        print("  resolution_sensitivity: SKIP — need r8 plus at least one "
              "refined case (sp_engine_nz200 / sp_engine_dt25).")
        return

    print(f"    {'case':22s} {'c0 (stage-1)':>14s} {'c1 (stage-2)':>14s} {'s2':>7s}")
    for lab, r, _c, _ls in loaded:
        m = r["m_map"]
        c0, c1 = rates(m)
        print(f"    {lab:22s} {c0:14.4e} {c1:14.4e} {m['s2_shape']:7.3f}")

    fig, AX = plt.subplots(1, 2, figsize=(11, 4.2), squeeze=False)
    ax = AX[0][0]
    for lab, r, c, ls in loaded:
        if "T_knots" in r and "knot_years" in r:
            ax.plot(r["knot_years"], r["T_knots"], ls, color=c, lw=1.8, label=lab)
    ax.set_xlabel("year")
    ax.set_ylabel("T anomaly (K)")
    ax.set_title("(a) temperature history", fontsize=10.5)
    ax.grid(alpha=0.3, color=fp.GRID)
    ax.legend(fontsize=8)

    ax = AX[0][1]
    for lab, r, c, ls in loaded:
        if "b_knots" in r and "b_knot_years" in r:
            ax.plot(r["b_knot_years"], r["b_knots"], ls, color=c, lw=1.8, label=lab)
    ax.set_xlabel("year")
    ax.set_ylabel("accumulation (m i.e. yr$^{-1}$)")
    ax.set_title("(b) accumulation history", fontsize=10.5)
    ax.grid(alpha=0.3, color=fp.GRID)
    ax.legend(fontsize=8)

    fig.suptitle("South Pole — discretisation sensitivity of the recovered histories",
                 fontsize=12)
    print(f"  resolution_sensitivity: {fp.save(fig, FIGS / 'resolution_sensitivity.png')}")
    plt.close(fig)


def figure_profiles(results):
    """The modelled column itself: density, age, temperature and velocity."""
    if not results:
        print("  sp_profiles: SKIP — no output/*.json carries model profiles.")
        return

    tag, r = sorted(results.items())[-1]
    pr = r["profiles"]
    d = np.asarray(pr["depth"], float)

    panels = [("rho", "density (kg m$^{-3}$)", fp.BLOCK_COLORS["rho"]),
              ("age_yr", "age (yr)", fp.BLOCK_COLORS["age"]),
              ("T_C", "T (°C)", fp.BLOCK_COLORS["T"]),
              ("w_m_yr", "w (m yr$^{-1}$)", fp.BLOCK_COLORS["v"])]

    fig, AX = plt.subplots(1, 4, figsize=(16, 4.4), squeeze=False)
    for i, (key, xlabel, color) in enumerate(panels):
        ax = AX[0][i]
        ax.plot(np.asarray(pr[key], float), d, "-", color=color, lw=2)
        fp.depth_axis(ax, hmax=HMAX, xlabel=xlabel)
        ax.set_title(f"({string.ascii_lowercase[i]}) {xlabel}", fontsize=10.5)

    fig.suptitle(f"South Pole — modelled column at the {tag} MAP", fontsize=12)
    print(f"  sp_profiles: {fp.save(fig, FIGS / 'sp_profiles.png')}")
    plt.close(fig)


def main():
    results = _results_with_blocks()
    if results:
        print(f"results with blocks: {', '.join(sorted(results))}")
    print("figures:")
    figure_misfits(results)
    figure_profiles(results)
    figure_resolution()


if __name__ == "__main__":
    main()
