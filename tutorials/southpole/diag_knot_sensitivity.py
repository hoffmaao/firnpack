"""diag_knot_sensitivity.py — WHERE in time can the data see T(t) and b(t)?

Knot COUNT is closed (temporal probe, both arms: denser layouts inject
null-space noise). Knot PLACEMENT is not: the current layout encodes an
untested guess (b-knots 100 yr apart pre-1800 but 25-30 yr apart post-1900;
T-knots 300 yr down to 7 yr). Andrew: "not all accumulation in the sequence is
equal -- can we test this sensitivity?"

METHOD (no Hessian, no external source): perturb ONE knot by +1 PRIOR sigma,
run forward, and measure how far the predictions move in units of OBSERVATION
sigma:

    dchi2_k = sum_i ( [d_i(m + sigma_k e_k) - d_i(m)] / sigma_i )^2

This is the Fisher information in direction k scaled by the prior width. For a
Gaussian the implied posterior/prior width is

    sigma_post/sigma_prior = 1/sqrt(1 + dchi2_k),   f = 1 - that

dchi2 >> 1 -> the data can see a full prior excursion of that knot.
dchi2 << 1 -> invisible: the inversion can only hand back the prior there.

*** LIMITATION: this is the DIAGONAL. It ignores TRADES between knots, so f is
an UPPER BOUND on information. A knot can look informed here and still be
degenerate with its neighbour. Low dchi2 is conclusive (invisible); high dchi2
is necessary but not sufficient. A full Hessian is the way to settle trades. ***

Run at the CORRECTED sigma_dage, else dage's 4.84x over-weighting inflates
every sensitivity and the ranking just measures the bug (see task #9).

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/diag_knot_sensitivity.py
"""
from __future__ import annotations
import json, os, copy
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
R = HERE / "results"
MAP_PATH = os.environ.get("FIRN_WARM_JSON", str(R / "sp_r10_final.json"))
# structure-function value (diag_dage_sigma_origin.py); r8 used 2.2 -- immaterial
SIG_DAGE = os.environ.get("FIRN_SIG_DAGE_SCALE", "2.26")

os.environ.update(FIRN_VEL_SITE="x17s2+x11n0+x11n2+x11n6+x11s2",
                  FIRN_VEL_SRC="zeising", FIRN_SEAS="1", FIRN_HCOL="300",
                  FIRN_SIG_DAGE_SCALE=SIG_DAGE)
ns = {"__file__": str(HERE / "run.py"), "__name__": "cfgbuild"}
exec(open(HERE / "run.py").read().split("warm = json.load")[0], ns)
cfg = ns["cfg"]
from firnpack.inverse import assimilate

warm0 = json.load(open(MAP_PATH))
SIG_T = float(cfg.T_knots.sigma)          # C
SIG_B = float(cfg.b_knots.sigma)          # log-space fraction
print(f"MAP {Path(MAP_PATH).name}; sigma_dage x{SIG_DAGE}; "
      f"prior sigma: T {SIG_T} C, b x/{np.exp(SIG_B):.3f} (log {SIG_B})")


def predict(w):
    r = assimilate(cfg, mode="forward", warm=w, verbose=False)
    return ({b["label"]: np.array(b["pred"]) for b in r["obs"]},
            {b["label"]: np.array(b["sig"]) for b in r["obs"]}, float(r["J"]))


base, SIG, J0 = predict(warm0)
print(f"baseline J = {J0:.2f}\n")

rows = []
for kind, key, yrs_key, s_prior, log in (
        ("T", "T_knots", "knot_years", SIG_T, False),
        ("b", "b_knots", "b_knot_years", SIG_B, True)):
    years = warm0[yrs_key]
    for k, yr in enumerate(years):
        w = copy.deepcopy(warm0)
        if log:
            w[key][k] = float(w[key][k]) * float(np.exp(s_prior))
        else:
            w[key][k] = float(w[key][k]) + s_prior
        pk, _, _ = predict(w)
        per_block, tot = {}, 0.0
        for lab in base:
            d = (pk[lab] - base[lab]) / SIG[lab]
            c = float(np.sum(d ** 2))
            per_block[lab] = c
            tot += c
        ratio = 1.0 / np.sqrt(1.0 + tot)
        rows.append(dict(kind=kind, k=k, year=float(yr), dchi2=tot,
                         post_over_prior=float(ratio), f=float(1 - ratio),
                         by_block=per_block))
        top = max(per_block, key=per_block.get) if per_block else "-"
        print(f"  {kind}[{k:2d}] {yr:6.0f}: dchi2 = {tot:10.3f}  "
              f"sig_post/sig_prior = {ratio:5.3f}  f = {1-ratio:5.3f}   "
              f"(most from {top})")

print(f"\n===== WHERE CAN THE DATA SEE THE HISTORY? =====")
for kind, lbl in (("T", "temperature"), ("b", "accumulation")):
    rs = [r for r in rows if r["kind"] == kind]
    print(f"\n{lbl} knots (prior sigma "
          f"{'%.2f C' % SIG_T if kind=='T' else 'x/%.3f' % np.exp(SIG_B)}):")
    print(f"{'year':>7} {'dchi2':>10} {'f':>7}  {'verdict':<16} {'dominant obs':>14}")
    for r in rs:
        v = ("INVISIBLE" if r["dchi2"] < 1 else
             "weak" if r["dchi2"] < 10 else
             "informed" if r["dchi2"] < 100 else "strong")
        top = max(r["by_block"], key=r["by_block"].get)
        print(f"{r['year']:7.0f} {r['dchi2']:10.3f} {r['f']:7.3f}  {v:<16} {top:>14}")
    inv = [r["year"] for r in rs if r["dchi2"] < 1]
    if inv:
        print(f"  -> INVISIBLE knots (data cannot see a full prior excursion): "
              f"{', '.join('%.0f' % y for y in inv)}")
        print(f"     These return the prior no matter what. Placement there is wasted.")

json.dump(dict(map=Path(MAP_PATH).name, sig_dage_scale=float(SIG_DAGE),
               sigma_T=SIG_T, sigma_b_log=SIG_B, J0=J0, knots=rows),
          open(R / "knot_sensitivity.json", "w"), indent=1)
print(f"\nSaved {R/'knot_sensitivity.json'}")
print("REMINDER: diagonal only -- ignores trades between knots, so f is an")
print("UPPER bound. Low dchi2 is conclusive; high dchi2 is necessary, not")
print("sufficient. Settle trades with the full Hessian.")
