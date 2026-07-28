"""knot_response_probe.py — can T-knots fit the deep borehole-T residual AT ALL?

Measures each T-knot's RESPONSE FUNCTION at the borehole obs depths by direct
perturbation (+dT on one knot, untaped engine forward, difference the T preds),
then least-squares fits the observed T residual with the span of those
responses. Because the LSQ ignores every other block and all priors, the
result is an UPPER BOUND on what a re-inversion with that layout could remove
from the T misfit — a few minutes of forwards instead of a 6-hour L-BFGS-B to
decide whether a denser layout is worth running.

Physics being tested: a surface pulse old enough to reach 60-100 m has a
diffusion kernel ±80-100 m wide, so surface forcing can only imprint BROAD
structure at depth. Sub-kernel-wavelength residual oscillation is not fittable
by any T(t) and must be charged to representativeness (or non-surface physics:
conductivity layering, borehole artifacts).

Env: FIRN_MAP (default sp_tightT.json), FIRN_KNOTS (the layout to probe;
     default knots_mid_T.json), FIRN_DKNOT (perturbation K, default 0.5).
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/diagnostics/knot_response_probe.py
"""
from __future__ import annotations
import copy, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("FIRN_KNOTS", "knots_mid_T.json")
from config import build_cfg
from firnpack.inverse import assimilate

OUT = HERE / "output"
MAP = json.load(open(OUT / os.environ.get("FIRN_MAP", "sp_tightT.json")))
DK = float(os.environ.get("FIRN_DKNOT", "0.5"))

cfg = build_cfg().cfg
years = np.asarray(cfg.T_knots.years, float)
# the warm interpolant the engine would use on this layout
wy = np.asarray(MAP["knot_years"], float); wv = np.asarray(MAP["T_knots"], float)
base_T = np.interp(years, wy, wv)

def fwd(Tk):
    w = copy.deepcopy(MAP)
    w["knot_years"] = years.tolist(); w["T_knots"] = np.asarray(Tk, float).tolist()
    r = assimilate(cfg, mode="forward", warm=w, annotate=False, verbose=False)
    tb = [b for b in r["obs"] if b["label"] == "T"][0]
    return np.array(tb["pred"]), np.array(tb["obs"]), np.array(tb["sig"]), np.array(tb["depths"])

t0 = time.perf_counter()
p0, o, s, zobs = fwd(base_T)
resid = (o - p0) / s                       # standardized, what LSQ must explain
print(f"base forward: T rms {np.sqrt(np.mean(resid**2)):.3f} sigma "
      f"({time.perf_counter()-t0:.0f}s); probing {len(years)} knots at +{DK} K")

R = np.zeros((len(zobs), len(years)))      # standardized response of knot j
for j, y in enumerate(years):
    Tk = base_T.copy(); Tk[j] += DK
    pj, *_ = fwd(Tk)
    R[:, j] = (pj - p0) / s / DK
    c_i = 2009.0
    prof = (pj - p0) / c_i * 1000.0        # mK per DK
    zmax = zobs[int(np.argmax(np.abs(prof)))]
    half = np.abs(prof) >= 0.5 * np.abs(prof).max()
    print(f"  knot {y:6.0f}: response peak {prof[int(np.argmax(np.abs(prof)))]/DK:+7.1f} mK/K "
          f"@ {zmax:5.0f} m, FWHM {zobs[half].max()-zobs[half].min():5.0f} m")

np.savez(OUT / "knot_response_probe.npz", years=years, R=R, resid=resid,
         zobs=zobs, sig=s)

# Ridge-regularized bound: the mid-record responses are near-collinear (they
# all peak at the column base with ~90 m FWHM), so unregularized LSQ "explains"
# the residual with +-1e8 K amplitude combinations the 0.6 K knot prior forbids.
# Penalize amplitudes at the prior sigma — the same weight the inversion uses —
# so the bound is what a re-inversion could ACTUALLY reach on the T block.
PRIOR_SIG = float(cfg.T_knots.sigma)

def ridge_rms(cols, lam=1.0 / PRIOR_SIG):
    A = R[:, cols]
    n = A.shape[1]
    Aa = np.vstack([A, lam * np.eye(n)])
    bb = np.concatenate([resid, np.zeros(n)])
    coef, *_ = np.linalg.lstsq(Aa, bb, rcond=None)
    return float(np.sqrt(np.mean((resid - A @ coef) ** 2))), coef

rms0 = float(np.sqrt(np.mean(resid ** 2)))
# which knots count as "the added ones": FIRN_NEW_KNOTS is a comma list of
# years; the baseline span is the probed layout minus these
NEW = set(float(v) for v in
          os.environ.get("FIRN_NEW_KNOTS", "1650,1800,1875,1915").split(","))
cur = [j for j, y in enumerate(years) if y not in NEW]
allc = list(range(len(years)))
rms_cur, coef_cur = ridge_rms(cur)
rms_all, coef_all = ridge_rms(allc)
print(f"\nT rms now                        : {rms0:.3f} sigma")
print(f"reachable, {len(cur):2d}-knot baseline    : {rms_cur:.3f} sigma  "
      f"(ridge at prior sigma {PRIOR_SIG} K; max |amp| {np.abs(coef_cur).max():.2f} K)")
print(f"reachable, +{len(NEW)} knots {sorted(int(v) for v in NEW)}: {rms_all:.3f} sigma  "
      f"(max |amp| {np.abs(coef_all).max():.2f} K)")
print(f"marginal gain of the added knots : {rms_cur - rms_all:.3f} sigma")
print("added-knot ridge amplitudes (K): " +
      " ".join(f"{years[j]:.0f}:{coef_all[allc.index(j)]:+.2f}"
               for j in range(len(years)) if years[j] in NEW))
