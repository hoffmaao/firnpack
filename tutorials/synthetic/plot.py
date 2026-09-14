"""Synthetic OSSE figure, rebuilt from output/.

One of the tutorial's two scripts: run.py recovers a known truth from noisy
synthetic data and writes output/, this draws the recovery figure. It reads
JSON only -- no Firedrake, no solve.

The figure is the method-credibility anchor: densification parameters,
surface-T history and accumulation history, truth vs recovered. It needs only
the recovered m_map and knots, so it does not depend on the engine's stored
obs/pred blocks.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 python tutorials/synthetic/plot.py
"""
from __future__ import annotations
import json, math, string
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np

from firnpack import plot as fp
from firnpack.models.firn import FirnParameters

_p = FirnParameters(); C_I, T_REF = float(_p.c_i), float(_p.T_ref)
HMAX = 132.0

HERE = Path(__file__).parent
R = HERE / "output"
FIGS = HERE / "figures"; FIGS.mkdir(exist_ok=True)

# newest OSSE MAP by mtime (synthetic_osse_trend supersedes the flat-truth
# synthetic_osse; env FIRN_OSSE_MAP overrides). The m_map filter keeps the
# tag-matched non-MAP artifacts (_hessian, _marginals) out of the running.
MAP_PATH = fp.newest_map(R, need=("m_map",), env="FIRN_OSSE_MAP",
                         pattern="synthetic_osse*.json") \
    or R / "synthetic_osse_tight.json"
TRUTH_PATH = R / "synthetic_truth.json"
if not (MAP_PATH.exists() and TRUTH_PATH.exists()):
    missing = [str(p) for p in (MAP_PATH, TRUTH_PATH) if not p.exists()]
    raise SystemExit(f"nothing to plot - run run.py first (missing {', '.join(missing)})")

mp = json.load(open(MAP_PATH))
tr = json.load(open(TRUTH_PATH))
m = mp["m_map"]; TRUTH = tr["truth"]
C_T, C_R = fp.OBS, fp.MODEL  # truth in ink, recovered in model-blue

# more-square 2x2 layout: params as a wide top banner, the two history panels
# side by side below (was a flat 1x3 row, ~3:1)
fig = plt.figure(figsize=(12, 9.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.28, wspace=0.22)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])

# (a) scalar params: recovered/truth ratio (log axis for the rate params)
ax = axA
keys = [k for k in TRUTH if k in m]
xr = np.arange(len(keys)); ratio = [m[k]/TRUTH[k] for k in keys]
ax.axhline(1.0, color=C_T, lw=1)
ax.bar(xr, np.clip(ratio, 0.7, 1.3), color=C_R, alpha=0.8, width=0.6)
for i, k in enumerate(keys):
    r = ratio[i]
    # keep the value label inside the axis; flag bars clipped at the band edge
    off = " ↓" if r < 0.7 else (" ↑" if r > 1.3 else "")
    yt = min(max(r + 0.02, 0.735), 1.24)
    ax.text(i, yt, f"{r:.2f}{off}", ha="center", va="bottom", fontsize=8,
            color=(C_R if not off else "#B45309"))
ax.set_xticks(xr); ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("recovered / truth"); ax.set_ylim(0.7, 1.3)
ax.set_title("(a) densification + conductivity params (±30% band; ↑↓ = off-scale)")
ax.grid(alpha=0.3, axis="y")

# (b) T-history - RECONSTRUCTION shown only over the age of the snowpack.
# The firn column spans a finite age (surface -> base); it can only reconstruct
# surface T over that span. Epochs OLDER than the firn base are not a firn
# reconstruction - they are the model's INITIALIZATION (they set the deep
# thermal state), and their surface-T information lives in the deep-ice /
# borehole temperature, not the firn. So we plot the recovered curve only over
# the snowpack age and shade the older region as the initialization domain.
ax = axB
ty = np.array(tr["T_years"]); tv = np.array(tr["T_truth"])
ry = np.array(mp["knot_years"]); rv = np.array(mp["T_knots"])
base_yr = None
prof = mp.get("profiles")
if prof and prof.get("age_yr"):
    base_yr = 2015.0 - float(np.max(prof["age_yr"]))     # snowpack base epoch
ax.plot(ty, tv, "-", color=C_T, lw=2, ms=4,
        marker="o" if len(ty) <= 12 else None, label="truth")
if base_yr is not None:
    keep = ry >= base_yr
    ax.plot(ry[keep], rv[keep], "s--", color=C_R, lw=1.8, ms=5,
            label="reconstructed (firn)")
    # older = initialization / deep-ice domain (not firn-reconstructed)
    ax.axvspan(ry.min() - 20, base_yr, color=fp.GRID, alpha=0.18, lw=0)
    ax.axvline(base_yr, color=fp.GRID, ls=":", lw=1.2)
    # white bbox: this sits in the lower-left corner, where the truth curve runs
    # through it and left the grey text unreadable.
    ax.annotate(f"initialization /\ndeep-ice domain\n(older than the\nsnowpack, ~{base_yr:.0f} CE)",
                xy=(0.02, 0.05), xycoords="axes fraction", fontsize=7.5, color="#6B7280",
                bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="none", alpha=0.85))
else:
    ax.plot(ry, rv, "s--", color=C_R, lw=1.8, ms=5, label="reconstructed")
if "dT_exp" in tr:
    ax.annotate(f"truth trend: +{tr['dT_exp']:.1f} K exp(t/{tr['tau']:.0f} yr)",
                xy=(0.03, 0.92), xycoords="axes fraction", fontsize=8)
ax.set_xlabel("year CE"); ax.set_ylabel("surface T (°C)")
ax.set_title("(b) surface-T reconstruction (over the snowpack age)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (c) accumulation
ax = axC
by = np.array(tr["b_years"]); bv = np.array(tr["b_truth"])
rby = np.array(mp["b_knot_years"]); rbv = np.array(mp["b_knots"])
ax.plot(by, bv, "-", color=C_T, lw=2, ms=4,
        marker="o" if len(by) <= 12 else None, label="truth")
ax.plot(rby, rbv, "s--", color=C_R, lw=1.8, ms=5, label="recovered")
if "db_exp" in tr:
    ax.annotate(f"truth trend: +{100*tr['db_exp']:.0f}% exp(t/{tr['tau']:.0f} yr)",
                xy=(0.03, 0.92), xycoords="axes fraction", fontsize=8)
ax.set_xlabel("year CE"); ax.set_ylabel("accumulation (m ice/yr)")
ax.set_title("(c) accumulation history"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.suptitle(f"Synthetic OSSE ({MAP_PATH.stem}) - known truth vs recovery: recalibrated sigmas, "
             "SP knot layouts, exponential recent trends in the truth", fontsize=12, y=0.99)
fig.savefig(FIGS/"osse_recovery.png", dpi=140, bbox_inches="tight")
print("recovery (recovered/truth):")
for k in keys: print(f"  {k:12s} {m[k]/TRUTH[k]:.3f}")
print(f"Saved {FIGS/'osse_recovery.png'}")


# ===== misfit figure (the OSSE analog of sp_misfits) =========================
# The recovered model against the synthetic observations it was inverted from.
# Because generation noise == assimilation sigma by construction, a well-behaved
# OSSE sits near 1 sigma everywhere: this figure is the OSSE's fit-quality check,
# NOT a data-vs-model surprise like the real sites.
_M_TITLES = {"rho": "density", "dage": "layer gradient d(age)/dz",
             "T": "borehole T (deep, 7-m)", "Tsh": "firn T (shallow RTD)",
             "comp": "compaction rate (coils)", "v": "vertical velocity (ApRES)"}


def _osse_display(block):
    """One stored block to display units (enthalpy blocks -> deg C)."""
    d = np.asarray(block["depths"], float)
    o = np.asarray(block["obs"], float)
    s = np.asarray(block["sig"], float)
    pr = np.asarray(block["pred"], float)
    lab = block["label"]
    if lab in ("T", "Tsh"):
        o = o / C_I + T_REF - 273.15
        pr = pr / C_I + T_REF - 273.15
        s = s / C_I
        unit = "T (°C)"
    else:
        unit = fp.BLOCK_UNITS.get(lab, lab)
    return d, o, s, pr, unit


if mp.get("obs") and mp.get("diag"):
    blocks = {b["label"]: b for b in mp["obs"]}
    order = [l for l in ("rho", "dage", "T", "Tsh", "comp", "v") if l in blocks]
    npan = len(order) + 1
    nc = min(3, npan); nr = math.ceil(npan / nc)
    # tall panels (7.2 per row): stretch the depth axis so the model-vs-obs
    # deviation is easy to read at every depth
    figM, AXM = plt.subplots(nr, nc, figsize=(5 * nc, 7.2 * nr), squeeze=False)
    axesM = list(AXM.flat)
    for i, (ax, lab) in enumerate(zip(axesM, order)):
        d, o, s, pr, unit = _osse_display(blocks[lab])
        rms = mp["diag"].get("rms_" + lab, float("nan"))
        fp.profile(ax, d, o, sig=s, pred=pr, hmax=HMAX, xlabel=unit,
                   model_label="recovered model", obs_label="synthetic obs ±1σ",
                   title=f"({string.ascii_lowercase[i]}) {_M_TITLES[lab]} - "
                         f"rms {rms:.2f}σ (N={len(d)})")
    resid = {}
    for lab in order:
        d, o, s, pr, _ = _osse_display(blocks[lab])
        # "Tsh" -> "T_sh" so the residual panel groups both T blocks in one family
        resid["T_sh" if lab == "Tsh" else lab] = (d, (pr - o) / s)
    for j in range(npan, len(axesM)):
        axesM[j].set_visible(False)
    fp.residual_panel(axesM[len(order)], resid, hmax=HMAX,
                      title=f"({string.ascii_lowercase[len(order)]}) standardized "
                            "residuals, all observables")
    figM.suptitle(f"Synthetic OSSE ({MAP_PATH.stem}) - recovered model vs synthetic "
                  f"observations (J = {mp['J']:.1f}); generation noise = assimilation σ",
                  fontsize=13, y=0.995)
    # explicit layout (not fp.save): reserve top strip for the suptitle so the
    # tall top-row panel titles do not collide with it
    figM.tight_layout(rect=[0, 0, 1, 0.985])
    figM.savefig(FIGS / "osse_misfits.png", dpi=140)
    print(f"  osse_misfits: {FIGS / 'osse_misfits.png'}")
    plt.close(figM)
else:
    print("  osse_misfits: SKIP - MAP carries no obs/diag blocks.")


# ===== reanalysis figure: RECOVERED vs TRUTH height-change attribution ========
# The OSSE's decisive test - does the inversion recover the right height-change
# DRIVERS, given known truth? Recovered = solid, truth = dashed throughout.
_RA = R / "synthetic_reanalysis.json"
if _RA.exists():
    ra = json.load(open(_RA))
    rec, tru = ra["recovered"], ra["truth"]
    yr = np.asarray(rec["years"], float)
    show = yr >= 1000.0
    y = yr[show]

    figR = plt.figure(figsize=(15, 9))
    gs = figR.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    ax = figR.add_subplot(gs[0, 0])
    ax.plot(y, np.asarray(rec["Ts_full"])[show], "-", color=C_R, lw=1.8, label="recovered")
    ax.plot(y, np.asarray(tru["Ts_full"])[show], "--", color=C_T, lw=1.8, label="truth")
    ax.set_ylabel("surface T (°C)"); ax.set_title("(a) T forcing", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = figR.add_subplot(gs[0, 1])
    ax.plot(y, np.asarray(rec["b_full"])[show], "-", color=C_R, lw=1.8, label="recovered")
    ax.plot(y, np.asarray(tru["b_full"])[show], "--", color=C_T, lw=1.8, label="truth")
    ax.set_ylabel("accumulation (m ice/yr)"); ax.set_title("(b) b forcing", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = figR.add_subplot(gs[0, 2])
    ax.plot(y, np.asarray(rec["dfac_m"])[show] * 100, "-", color=C_R, lw=2, label="recovered")
    ax.plot(y, np.asarray(tru["dfac_m"])[show] * 100, "--", color=C_T, lw=2, label="truth")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("ΔFAC (cm)")
    ax.set_title("(c) firn AIR response (ΔFAC)", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # absolute values and absolute error only. A ratio here divides by a truth
    # ΔFAC that can sit near zero, which printed "~25138% recovered" when the
    # baseline made truth -1 cm.
    ax.annotate(f"2015: rec {rec['dfac_m'][-1]*100:+.0f}, truth {tru['dfac_m'][-1]*100:+.0f} cm"
                f"  (err {(rec['dfac_m'][-1]-tru['dfac_m'][-1])*100:+.0f} cm)",
                xy=(0.40, 0.28), xycoords="axes fraction", fontsize=8,
                bbox=dict(boxstyle="round", fc="w", alpha=0.85))

    ax = figR.add_subplot(gs[1, 0])
    ax.plot(y, np.asarray(rec["hprime_T_m"])[show] * 100, "-", color=C_R, lw=2, label="recovered")
    ax.plot(y, np.asarray(tru["hprime_T_m"])[show] * 100, "--", color=C_T, lw=2, label="truth")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("T-driven h' (cm)"); ax.set_title("(d) T-driven height change", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = figR.add_subplot(gs[1, 1])
    ax.plot(y, np.asarray(rec["hprime_m"])[show] * 100, "-", color=C_R, lw=2, label="recovered")
    ax.plot(y, np.asarray(tru["hprime_m"])[show] * 100, "--", color=C_T, lw=2, label="truth")
    ax.plot(y, np.asarray(rec["mass_m"])[show] * 100, ":", color=C_R, lw=1.4, label="recovered mass")
    ax.plot(y, np.asarray(tru["mass_m"])[show] * 100, ":", color=C_T, lw=1.4, label="truth mass")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("total h' (cm)")
    ax.set_title("(e) TOTAL height change h' and mass term", fontsize=10.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    ax = figR.add_subplot(gs[1, 2])
    comps = ["air\n(ΔFAC)", "T-driven", "mass", "total"]
    rvals = [rec["dfac_m"][-1], rec["hprime_T_m"][-1], rec["mass_m"][-1], rec["hprime_m"][-1]]
    tvals = [tru["dfac_m"][-1], tru["hprime_T_m"][-1], tru["mass_m"][-1], tru["hprime_m"][-1]]
    xb = np.arange(len(comps)); w = 0.38
    ax.bar(xb - w/2, np.array(rvals) * 100, w, color=C_R, alpha=0.85, label="recovered")
    ax.bar(xb + w/2, np.array(tvals) * 100, w, color=C_T, alpha=0.85, label="truth")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xb); ax.set_xticklabels(comps, fontsize=8)
    ax.set_ylabel("h'(2015) (cm)")
    ax.set_title("(f) 2015 attribution: recovered vs truth", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Title states what THIS run measured. It used to assert a fixed narrative
    # ("air recovered, mass not"), which silently inverted when the underlying
    # MAP changed - the plotted numbers then contradicted the title.
    _by = ra.get("base_year")
    _blab = "oldest knot" if _by is None else f"year {_by:.0f}"
    _ff_r = float(np.asarray(rec["fac_full"])[-1]); _ff_t = float(np.asarray(tru["fac_full"])[-1])
    figR.suptitle(
        f"Synthetic OSSE reanalysis - recovered vs truth attribution   (CTRL baseline: {_blab})\n"
        f"FAC_full recovered to {_ff_r-_ff_t:+.3f} m of truth and is baseline-independent; "
        f"every CTRL-differenced term below moves with that baseline choice",
        fontsize=12.5)
    print(f"  osse_reanalysis: {fp.save(figR, FIGS / 'osse_reanalysis.png')}")
    plt.close(figR)
else:
    print("  osse_reanalysis: SKIP - run diagnostics/reanalysis.py first.")
