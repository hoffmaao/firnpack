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

    Ordered oldest-first by file mtime, so "the newest MAP" is the last entry --
    a fresh inversion becomes the plotted one regardless of how its tag sorts.
    """
    found = []
    for p in RESULTS.glob("*.json"):
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(d, dict) and d.get("obs") and d.get("profiles"):
            found.append((p.stat().st_mtime, p.stem, d))
    return {stem: d for _, stem, d in sorted(found)}


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

    tag, r = list(results.items())[-1]
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
                   title=f"({string.ascii_lowercase[i]}) {_title(lab)} — "
                         f"rms {rms:.2f}σ (N={len(d)})")
        # NB the coherent 55-63 m borehole-T warm band (~9 m wide, ~100 mK) is a
        # suspected logging artifact — sub-diffusion-kernel and too large for
        # conductivity layering. It is carried in the kernel-consistent sigma,
        # not chased by the model. No longer annotated in-figure: the paper
        # caption discusses it.

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
    """The modelled column with its observations and structure.

    Model profiles from the stored MAP, observations overlaid from the stored
    blocks (density and borehole T — the observables that live in profile
    space; d(age)/dz and differenced velocity are gradient/differenced
    operators and belong to the misfit figure). Close-off and the derived
    column numbers are annotated so the figure answers "what column did the
    inversion settle on" at a glance.
    """
    if not results:
        print("  sp_profiles: SKIP — no output/*.json carries model profiles.")
        return

    tag, r = list(results.items())[-1]
    pr = r["profiles"]
    d = np.asarray(pr["depth"], float)
    rho = np.asarray(pr["rho"], float)
    age = np.asarray(pr["age_yr"], float)
    Tc = np.asarray(pr["T_C"], float)
    w = np.asarray(pr["w_m_yr"], float)
    blocks = {b["label"]: b for b in r.get("obs", [])}
    zco = float(np.interp(830.0, rho, d)) if rho.max() > 830 else float("nan")
    fac = float(np.trapz(np.clip(1.0 - rho / 917.0, 0, None), d))

    panels = [("rho", "density (kg m$^{-3}$)", fp.BLOCK_COLORS["rho"]),
              ("age_yr", "age (yr)", fp.BLOCK_COLORS["age"]),
              ("T_C", "T (°C)", fp.BLOCK_COLORS["T"]),
              ("w_m_yr", "w (m yr$^{-1}$)", fp.BLOCK_COLORS["v"])]
    prof = {"rho": rho, "age_yr": age, "T_C": Tc, "w_m_yr": w}

    fig, AX = plt.subplots(1, 4, figsize=(16, 4.6), squeeze=False)
    for i, (key, xlabel, color) in enumerate(panels):
        ax = AX[0][i]
        # observations first (underneath): density and borehole T
        if key == "rho" and "rho" in blocks:
            b = blocks["rho"]
            ax.errorbar(b["obs"], b["depths"], xerr=b["sig"], fmt="o",
                        color=fp.OBS, ms=2.5, alpha=0.55, lw=0.8,
                        label="observed (SP19)")
        if key == "T_C" and "T" in blocks:
            bd, bo, bs, _, _ = _to_display(blocks["T"])
            ax.errorbar(bo, bd, xerr=bs, fmt="o", color=fp.OBS, ms=3,
                        alpha=0.6, lw=0.8, label="observed (borehole)")
        ax.plot(prof[key], d, "-", color=color, lw=2, label="model")
        if np.isfinite(zco):
            ax.axhline(zco, color=fp.GRID, ls=":", lw=1.2)
        fp.depth_axis(ax, hmax=HMAX, xlabel=xlabel)
        ax.set_title(f"({string.ascii_lowercase[i]}) {xlabel}", fontsize=10.5)
        if key in ("rho", "T_C"):
            ax.legend(fontsize=7.5, loc="lower left")

    AX[0][0].annotate(f"close-off (830): {zco:.1f} m\nFAC: {fac:.1f} m",
                      xy=(0.04, 0.16), xycoords="axes fraction", fontsize=8,
                      bbox=dict(boxstyle="round", fc="w", alpha=0.85))
    AX[0][1].annotate(f"age @ close-off: {np.interp(zco, d, age):.0f} yr\n"
                      f"age @ 100 m: {np.interp(100.0, d, age):.0f} yr",
                      xy=(0.04, 0.16), xycoords="axes fraction", fontsize=8,
                      bbox=dict(boxstyle="round", fc="w", alpha=0.85))
    AX[0][2].annotate(f"T base: {Tc[np.argmax(d)]:.2f} °C", xy=(0.04, 0.30),
                      xycoords="axes fraction", fontsize=8,
                      bbox=dict(boxstyle="round", fc="w", alpha=0.85))
    AX[0][3].annotate(f"w surf: {w[np.argmin(d)]:.3f} m/yr\n"
                      f"w base: {w[np.argmax(d)]:.3f} m/yr",
                      xy=(0.04, 0.16), xycoords="axes fraction", fontsize=8,
                      bbox=dict(boxstyle="round", fc="w", alpha=0.85))

    fig.suptitle(f"South Pole — modelled column at the {tag} MAP "
                 f"(dotted line: pore close-off)", fontsize=12)
    print(f"  sp_profiles: {fp.save(fig, FIGS / 'sp_profiles.png')}")
    plt.close(fig)


def figure_T_history(results):
    """The reconstructed surface-temperature history, across recent MAPs.

    The paper-facing deliverable of the T recalibration: how the recovered
    T(t) moved as the borehole block went from the hand-set flat 0.2 K sigma
    to the data-derived one. Prior band from the newest MAP that stores it
    (T_prior_centers/T_prior_sigma -- older results predate those keys).
    """
    withT = {t: r for t, r in results.items()
             if "T_knots" in r and "knot_years" in r}
    if not withT:
        print("  sp_T_history: SKIP — no results carry T_knots.")
        return

    fig, ax = plt.subplots(figsize=(9, 4.6))
    newest = list(withT)[-1]
    pr = withT[newest]
    if "T_prior_centers" in pr:
        ky = np.asarray(pr["knot_years"], float)
        pc = np.asarray(pr["T_prior_centers"], float)
        # scalar for a uniform prior, per-knot array once T_knots.sigma varies
        ps = np.broadcast_to(np.asarray(pr["T_prior_sigma"], float), ky.shape)
        lab = (f"prior ±{ps[0]:g} K" if np.allclose(ps, ps[0])
               else f"prior ±{ps.min():g}–{ps.max():g} K")
        ax.fill_between(ky, pc - ps, pc + ps, color=fp.GRID, alpha=0.35,
                        label=lab)
        ax.plot(ky, pc, "-", color=fp.GRID, lw=1)
    styles = ["-", "--", ":", "-."]
    colors = [fp.OBS, fp.MODEL, fp.RESID]
    for i, (tag, r) in enumerate(withT.items()):
        ax.plot(r["knot_years"], r["T_knots"],
                styles[i % len(styles)], color=colors[i % len(colors)],
                lw=2.2 if tag == newest else 1.4,
                marker="o" if tag == newest else None, ms=4,
                label=f"{tag} (J = {r.get('J_map', r.get('J', float('nan'))):.1f})")
    # Laplace marginal bars on the newest MAP, when its Hessian read-out exists.
    # NB these are per-knot MARGINALS: mid-record knots are individually
    # prior-bound (the data constrain kernel-smoothed combinations), so wide
    # bars there do NOT mean the smooth history is unconstrained.
    mp = RESULTS / f"{newest}_hessian_marginals.json"
    if mp.exists():
        mg = json.load(open(mp))
        tsig = {nm: s for nm, s in zip(mg["names"], mg["sigma_post"])
                if nm.startswith("Tk")}
        ky = np.asarray(withT[newest]["knot_years"], float)
        tv = np.asarray(withT[newest]["T_knots"], float)
        es = np.array([tsig.get(f"Tk{i}", np.nan) for i in range(len(ky))])
        ax.errorbar(ky, tv, yerr=es, fmt="none", ecolor=fp.OBS, alpha=0.55,
                    capsize=2, lw=1.1, label="Laplace marginal ±1σ")
    ax.set_xlabel("year CE")
    ax.set_ylabel("surface temperature (°C)")
    ax.set_title("South Pole — recovered surface-temperature history "
                 "(datum: USP50-corrected, ≈ −51 °C firn)", fontsize=11)
    ax.grid(alpha=0.3, color=fp.GRID)
    ax.legend(fontsize=8)
    print(f"  sp_T_history: {fp.save(fig, FIGS / 'sp_T_history.png')}")
    plt.close(fig)


def figure_reanalysis():
    """Firn-driven surface-height change from the scenario differencing.

    Pure reader of output/sp_reanalysis.{json,npz} written by
    diagnostics/reanalysis.py.
    """
    jp, zp = RESULTS / "sp_reanalysis.json", RESULTS / "sp_reanalysis.npz"
    if not (jp.exists() and zp.exists()):
        print("  sp_reanalysis: SKIP — run diagnostics/reanalysis.py first.")
        return
    r = json.load(open(jp))
    npz = np.load(zp)
    yr = np.asarray(r["years"], float)
    show = yr >= float(r["knot_years"][0])
    y = yr[show]
    hl = r["headline"]

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.34, wspace=0.26)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(y, np.asarray(r["Ts_full"])[show], "-", color=fp.OBS, lw=1.6,
            label="recovered T(t)")
    ax.plot(r["knot_years"], r["T_knots"], "o", color=fp.OBS, ms=4)
    ax.axhline(r["T_knots"][0], color="k", ls="--", lw=1, label="CTRL (T at 1000 CE)")
    ax.set_ylabel("surface T (°C)")
    ax.set_title("(a) temperature forcing", fontsize=10.5)
    ax.grid(alpha=0.3, color=fp.GRID); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(y, np.asarray(r["b_full"])[show], "-", color=fp.MODEL, lw=1.6,
            label="recovered b(t)")
    ax.axhline(r["b_ctrl"][0], color="k", ls="--", lw=1, label="CTRL (b at 1000 CE)")
    ax.set_ylabel("accumulation (m i.e. yr$^{-1}$)")
    ax.set_title("(b) accumulation forcing", fontsize=10.5)
    ax.grid(alpha=0.3, color=fp.GRID); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(y, np.asarray(r["fac_full"])[show], "-", color=fp.OBS, lw=1.6, label="FULL")
    ax.plot(y, np.asarray(r["fac_tonly"])[show], "-", color=fp.MODEL, lw=1.2, label="T only")
    ax.plot(y, np.asarray(r["fac_ctrl"])[show], "k--", lw=1, label="CTRL")
    ax.set_ylabel("firn air content 0–130 m (m)")
    ax.set_title("(c) firn air content", fontsize=10.5)
    ax.grid(alpha=0.3, color=fp.GRID); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(y, np.asarray(r["hprime_m"])[show] * 100, "-", color=fp.OBS, lw=2,
            label="h' total (from $|w_{bot}|$)")
    ax.plot(y, np.asarray(r["hprime_T_m"])[show] * 100, "-", color=fp.MODEL, lw=1.5,
            label="T-driven")
    ax.plot(y, np.asarray(r["hprime_b_m"])[show] * 100, ":", color=fp.MODEL, lw=1.5,
            label="b-driven (incl. ice-eq mass)")
    ax.plot(y, np.asarray(r["dfac_T_m"])[show] * 100, "--", color=fp.RESID, lw=1.2,
            label=r"$\Delta$FAC (T only) cross-check")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("surface-height anomaly (cm)")
    ax.set_title("(d) firn-driven height anomaly vs 1000 CE climate", fontsize=10.5)
    ax.grid(alpha=0.3, color=fp.GRID); ax.legend(fontsize=8, loc="upper right")
    hT = np.asarray(r["hprime_T_m"])[yr >= 1957.0]
    rate_T = np.mean(np.diff(hT)) / r["dt_years"] * 1000.0  # mm/yr
    ax.annotate(f"2015: total {hl['hprime_2015_cm']:+.0f} cm "
                f"(T {hl['hprime_T_2015_cm']:+.0f}, b {hl['hprime_b_2015_cm']:+.0f})\n"
                f"T-driven dh'/dt 1957–2015: {rate_T:+.2f} mm/yr",
                xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8,
                bbox=dict(boxstyle="round", fc="w", alpha=0.85))

    ax = fig.add_subplot(gs[2, :])
    sy = np.asarray(npz["snap_years"], float)
    ssel = sy >= float(r["knot_years"][0])
    drho = (np.asarray(npz["rho_full"]) - np.asarray(npz["rho_ctrl"]))[ssel]
    vmax = max(float(np.abs(drho).max()), 1e-6)
    pc = ax.pcolormesh(sy[ssel], np.asarray(npz["depth"], float), drho.T,
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
    ax.invert_yaxis()
    ax.set_xlabel("year CE"); ax.set_ylabel("depth (m)")
    ax.set_title(r"(e) density anomaly $\rho_{FULL}-\rho_{CTRL}$ (kg m$^{-3}$)",
                 fontsize=10.5)
    fig.colorbar(pc, ax=ax, pad=0.01)

    fig.suptitle(f"South Pole firn reanalysis — {r['map_file']} MAP, "
                 f"scenario differencing through the engine", fontsize=12.5)
    print(f"  sp_reanalysis: {fp.save(fig, FIGS / 'sp_reanalysis.png')}")
    plt.close(fig)


def main():
    results = _results_with_blocks()
    if results:
        print(f"results with blocks: {', '.join(results)}")
    print("figures:")
    figure_misfits(results)
    figure_profiles(results)
    figure_T_history(results)
    figure_reanalysis()
    figure_resolution()


if __name__ == "__main__":
    main()
