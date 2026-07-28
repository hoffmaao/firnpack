"""Summit figures, rebuilt from output/.

One of the tutorial's two scripts: run.py inverts for Summit's own densification
law and writes output/, this draws figures/ from them. It reads JSON only --
no Firedrake, no solve.

Two figures:
  law_comparison  -- the transferability result: the laws independently
                     recovered at South Pole and Summit, overlaid at a common
                     (T, b). Needs only m_map, so it works from any MAP.
  summit_misfits  -- observations vs the model at the Summit MAP, from the
                     obs/sig/pred blocks run.py now stores. Skipped when no
                     result carries them.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 python tutorials/summit/plot.py
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
from firnpack.constants import year as YEAR_S
from firnpack.models.firn import FirnParameters

HERE = Path(__file__).parent
RESULTS = HERE / "output"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

_newest_map = fp.newest_map


SP_MAP = _newest_map(HERE.parent / "southpole" / "output", env="FIRN_SP_MAP") \
    or HERE.parent / "southpole" / "output" / "sp_joint_r8.json"
SUMMIT_MAP = _newest_map(RESULTS, env="FIRN_SUMMIT_MAP") \
    or RESULTS / "summit_invert.json"

R_GAS, RHO_I, RHO_M = 8.314, 917.0, 550.0

_p = FirnParameters()
C_I, T_REF = float(_p.c_i), float(_p.T_ref)

TITLES = {"rho": "density (FirnCover 2017)", "rho_g": "density (GRIP 1991)",
          "age": "depth–age (GISP2)", "dage": "layer gradient d(age)/dz (1989)",
          "comp": "compaction rate (FirnCover coils)", "T": "firn temperature"}
HMAX = 92.0  # GRIP density reaches ~90 m; compaction coils to 22 m


def _rate_curve(m, T_C, b):
    """dρ/dt(ρ) for a recovered Herron–Langway law at a reference (T, b)."""
    T = T_C + 273.15
    rho = np.linspace(360, 900, 300)
    phi = np.maximum((RHO_I - rho) / (RHO_I - RHO_M), 1e-4)
    switch = 0.5 * (1 + np.tanh((rho - RHO_M) / 20.0))
    c0 = m["hl_k0"] * math.exp(-m["hl_Ea1"] / (R_GAS * T)) * b
    c1 = m["hl_k1"] * math.exp(-m["hl_Ea2"] / (R_GAS * T)) * math.sqrt(b) * phi ** (m.get("s2_shape", 1) - 1)
    return rho, ((1 - switch) * c0 + switch * c1) * (RHO_I - rho)


def figure_law_comparison():
    """Do the laws recovered at two contrasting ice sheets agree?"""
    if not (SP_MAP.exists() and SUMMIT_MAP.exists()):
        missing = [str(p) for p in (SP_MAP, SUMMIT_MAP) if not p.exists()]
        print(f"  law_comparison: SKIP — missing {', '.join(missing)}.")
        return

    sp = json.load(open(SP_MAP))["m_map"]
    su = json.load(open(SUMMIT_MAP))
    ms = su["m_map"]

    fig, AX = plt.subplots(1, 2, figsize=(12, 5.4))

    ax = AX[0]
    for m, col, lab in [(sp, fp.MODEL, "South Pole (−51°C, 0.09 m/yr)"),
                        (ms, fp.RESID, "Summit (−29°C, 0.25 m/yr)")]:
        rho, c = _rate_curve(m, -40.0, 0.15)
        ax.plot(rho, c, "-", color=col, lw=2.2, label=f"{lab} recovered law")
    ax.set_yscale("log")
    ax.set_xlabel("density ρ (kg m$^{-3}$)")
    ax.set_ylabel("dρ/dt (kg m$^{-3}$ yr$^{-1}$)")
    ax.axvline(RHO_M, color=fp.GRID, ls=":", lw=1)
    ax.set_title("(a) recovered densification law — two ice sheets, one physics\n"
                 "(evaluated at a common −40°C, 0.15 m/yr)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3, which="both", color=fp.GRID)

    # Rate ratios computed from the JSONs, not hardcoded, so the annotation
    # tracks a rerun rather than freezing a past result's numbers.
    ratios = {k: ms[k] / sp[k] for k in ("hl_k0", "hl_k1", "s2_shape") if sp.get(k)}
    ax.text(0.03, 0.06,
            "stage-1 rate ratio Summit/SP = {hl_k0:.2f}\n"
            "stage-2 rate ratio = {hl_k1:.2f}\n"
            "s2 shape ratio = {s2_shape:.2f}".format(**ratios),
            transform=ax.transAxes, fontsize=8.5, color=fp.OBS,
            bbox=dict(boxstyle="round", fc="w", alpha=0.85))

    ax = AX[1]
    by, bk = np.array(su["b_knot_years"]), np.array(su["b_knots"])
    bc = np.array(su["b_prior_centers"])
    yy = np.linspace(by.min(), by.max(), 300)
    ax.plot(yy, np.exp(np.interp(yy, by, np.log(bk))), "-", color=fp.RESID, lw=2,
            label="recovered b(t)")
    ax.plot(by, bk, "o", color=fp.RESID, ms=4)
    ax.axhline(bc[0], color=fp.GRID, ls="--", lw=1, label=f"Osman prior ({bc[0]:.3f})")
    ax.set_xlabel("year CE")
    ax.set_ylabel("accumulation (m ice eq / yr)")
    ax.set_title("(b) Summit accumulation history (recovered)", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, color=fp.GRID)
    ax.set_ylim(0.12, 0.30)

    fig.suptitle("Firn densification law transferability — South Pole vs Summit "
                 "(independent inversions, same literature prior)", fontsize=12)
    print(f"  law_comparison: {fp.save(fig, FIGS / 'law_comparison.png')}")
    print("    param      SP     Summit   ratio")
    for k in ("hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "s2_shape"):
        if k in sp and k in ms:
            print(f"    {k:9s} {sp[k]:9.3f} {ms[k]:9.3f}  {ms[k]/sp[k]:.3f}")
    plt.close(fig)


def _to_display(block):
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


def figure_misfits():
    """Observations vs the model at the Summit MAP, from stored blocks."""
    if not SUMMIT_MAP.exists():
        print("  summit_misfits: SKIP — run run.py first (no summit_invert.json).")
        return
    r = json.load(open(SUMMIT_MAP))
    if not (r.get("obs") and r.get("diag")):
        print("  summit_misfits: SKIP — summit_invert.json predates block storage; re-run run.py.")
        return

    blocks = {b["label"]: b for b in r["obs"]}
    order = [l for l in ("rho", "rho_g", "age", "dage", "comp", "T") if l in blocks]

    npan = len(order) + 1
    nc = min(3, npan); nr = math.ceil(npan / nc)
    fig, AX = plt.subplots(nr, nc, figsize=(5 * nc, 4.5 * nr), squeeze=False)
    axes = list(AX.flat)
    for i, (ax, lab) in enumerate(zip(axes, order)):
        d, o, s, p, unit = _to_display(blocks[lab])
        rms = r["diag"].get("rms_" + lab, float("nan"))
        fp.profile(ax, d, o, sig=s, pred=p, hmax=HMAX, xlabel=unit,
                   title=f"({string.ascii_lowercase[i]}) {TITLES.get(lab, lab)} — "
                         f"rms {rms:.2f}σ (N={len(d)})")
    for j in range(npan, len(axes)):
        axes[j].set_visible(False)

    resid = {}
    for lab in order:
        d, o, s, p, _ = _to_display(blocks[lab])
        resid[lab] = (d, (p - o) / s)
    fp.residual_panel(axes[len(order)], resid, hmax=HMAX,
                      title=f"({string.ascii_lowercase[len(order)]}) standardized residuals")

    fig.suptitle(f"Summit assimilation — observations vs the MAP (J = {r['J']:.1f})",
                 fontsize=13)
    print(f"  summit_misfits: {fp.save(fig, FIGS / 'summit_misfits.png')}")
    plt.close(fig)


def figure_reanalysis():
    """Summit firn structure + accumulation-driven height change.

    Pure reader of output/summit_reanalysis.{json,npz} written by
    diagnostics/reanalysis.py.
    """
    jp, zp = RESULTS / "summit_reanalysis.json", RESULTS / "summit_reanalysis.npz"
    if not (jp.exists() and zp.exists()):
        print("  summit_reanalysis: SKIP — run diagnostics/reanalysis.py first.")
        return
    r = json.load(open(jp))
    npz = np.load(zp)
    yr = np.asarray(r["years"], float)
    show = yr >= float(r["b_knot_years"][0])
    y = yr[show]
    hl = r["headline"]

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.34, wspace=0.26)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(y, np.asarray(r["b_full"])[show], "-", color=fp.MODEL, lw=1.6,
            label="recovered b(t)")
    ax.axhline(r["b_ctrl"][0], color="k", ls="--", lw=1, label="CTRL (b at 1000 CE)")
    if r.get("b_prior_centers"):
        ax.axhline(r["b_prior_centers"][0], color=fp.GRID, ls=":", lw=1.2,
                   label=f"Osman prior ({r['b_prior_centers'][0]:.3f})")
    ax.set_ylabel("accumulation (m i.e. yr$^{-1}$)")
    ax.set_title("(a) accumulation forcing", fontsize=10.5)
    ax.grid(alpha=0.3, color=fp.GRID); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(y, np.asarray(r["fac_full"])[show], "-", color=fp.OBS, lw=1.6, label="FULL")
    ax.plot(y, np.asarray(r["fac_ctrl"])[show], "k--", lw=1, label="CTRL")
    ax.set_ylabel("firn air content 0–150 m (m)")
    ax.set_title("(b) firn air content", fontsize=10.5)
    ax.grid(alpha=0.3, color=fp.GRID); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    sy = np.asarray(r["snap_years"], float)
    ssel = sy >= float(r["b_knot_years"][0])
    ax.plot(sy[ssel], np.asarray(r["zco_full_m"])[ssel], "-", color=fp.OBS, lw=1.6,
            label="FULL")
    ax.plot(sy[ssel], np.asarray(r["zco_ctrl_m"])[ssel], "k--", lw=1, label="CTRL")
    ax.invert_yaxis()
    ax.set_ylabel("close-off depth ρ=830 (m)")
    ax.set_title("(c) close-off depth (firn structure)", fontsize=10.5)
    ax.grid(alpha=0.3, color=fp.GRID); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(y, np.asarray(r["hprime_m"])[show] * 100, "-", color=fp.OBS, lw=2,
            label="h' total (from $|w_{bot}|$)")
    ax.plot(y, np.asarray(r["mass_m"])[show] * 100, ":", color=fp.MODEL, lw=1.5,
            label="ice-eq mass part")
    ax.plot(y, np.asarray(r["dfac_m"])[show] * 100, "--", color=fp.RESID, lw=1.5,
            label=r"$\Delta$FAC (firn-process/air part)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("surface-height anomaly (cm)")
    ax.set_title("(d) accumulation-driven height anomaly vs 1000 CE", fontsize=10.5)
    ax.grid(alpha=0.3, color=fp.GRID); ax.legend(fontsize=8, loc="best")
    ax.annotate(f"2015: total {hl['hprime_2015_cm']:+.0f} cm "
                f"(mass {hl['mass_2015_cm']:+.0f}, air {hl['dfac_2015_cm']:+.0f})\n"
                f"dh'/dt 1957–2015: {hl['dhdt_1957_2015_mm_yr']:+.2f} mm/yr",
                xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8,
                bbox=dict(boxstyle="round", fc="w", alpha=0.85))

    ax = fig.add_subplot(gs[2, :])
    drho = (np.asarray(npz["rho_full"]) - np.asarray(npz["rho_ctrl"]))[ssel]
    vmax = max(float(np.abs(drho).max()), 1e-6)
    pc = ax.pcolormesh(sy[ssel], np.asarray(npz["depth"], float), drho.T,
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
    ax.invert_yaxis()
    ax.set_xlabel("year CE"); ax.set_ylabel("depth (m)")
    ax.set_title(r"(e) density anomaly $\rho_{FULL}-\rho_{CTRL}$ (kg m$^{-3}$)",
                 fontsize=10.5)
    fig.colorbar(pc, ax=ax, pad=0.01)

    fig.suptitle(f"Summit firn reanalysis — {r['map_file']} MAP, accumulation-driven "
                 f"(T forcing constant)", fontsize=12.5)
    print(f"  summit_reanalysis: {fp.save(fig, FIGS / 'summit_reanalysis.png')}")
    plt.close(fig)


def main():
    print(f"maps: SP={SP_MAP.name}, Summit={SUMMIT_MAP.name}")
    print("figures:")
    figure_law_comparison()
    figure_misfits()
    figure_reanalysis()


if __name__ == "__main__":
    main()
