"""score_vel_strata.py — rank ApRES sites by consistency with the STRATA-ONLY model.

The 40-km region around South Pole carries real flow variation, so pooling all
ApRES sites and calling their spread "noise" conflates genuine between-site
differences with measurement error (the pipeline-A pooled block did exactly
that: its sigma was ~100% between-site spread, ~70x the instrument precision).
The right question is instead: which site is the SP strata's own column?

This scores every Zeising site against a MAP inverted WITHOUT any velocity on
the tape (default sp_r10_novel) — so the model's w(z) is implied purely by
density + depth-age + d(age)/dz + borehole T. A site consistent with the
observed strata is predicted by that model for free; an inconsistent one is a
different column.

This is the CHEAP first cut (one forward eval, no optimization). It ranks
candidates; score_vel_screen.py's dJ_other leave-one-in screen (which needs one
inversion per site) is the expensive confirmation.

Env:
  FIRN_WARM_JSON   MAP to score against   (default results/sp_r10_novel.json)
  FIRN_HCOL_SCAN   comma list of column depths to try (default "130,300") —
                   the MAP's provenance does not record FIRN_HCOL, so scan and
                   report which reproduces the MAP's logged non-velocity rms.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/score_vel_strata.py
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
R = HERE / "results"

# every site present in the Zeising raw-burst product
import pandas as pd
ZSITES = sorted(pd.read_csv(HERE / "data" / "apres_zeising_processed.csv").site.unique())
MAP_PATH = os.environ.get("FIRN_WARM_JSON", str(R / "sp_r10_novel.json"))
# H_col=300 CONFIRMED for the r10 MAPs (2026-07-14: it reproduced r10_novel's
# logged rms exactly, 130 did not). The old H_col scan + logged-rms probe are
# retired — those logged values were under the SUPERSEDED error model (age
# block present, sigma_dage on the 4% floor), so they no longer compare.
SCAN = [float(x) for x in os.environ.get("FIRN_HCOL_SCAN", "300").split(",")]
LOGGED = {}

from firnpack.inverse import assimilate

warm = json.load(open(MAP_PATH))
print(f"scoring against {Path(MAP_PATH).name}  (J_logged={warm['J']:.2f}, "
      f"{warm['message']})")
print(f"zeising sites: {' '.join(ZSITES)}\n")


def build(hcol):
    """run.py's config, with EVERY zeising site as its own diagnostic block."""
    env = dict(FIRN_VEL_SITE="+".join(ZSITES), FIRN_VEL_SRC="zeising",
               FIRN_SEAS="1", FIRN_HCOL=str(hcol))
    os.environ.update(env)
    ns = {"__file__": str(HERE / "run.py"), "__name__": "cfgbuild"}
    exec(open(HERE / "run.py").read().split("warm = json.load")[0], ns)
    return ns["cfg"]


results = {}
for hcol in SCAN:
    print(f"===== H_col = {hcol:.0f} m =====")
    r = assimilate(build(hcol), mode="forward", warm=warm, verbose=False)
    rms = {k[4:]: v for k, v in r["diag"].items() if k.startswith("rms_")}
    if LOGGED:   # retired: kept only if a probe target is supplied
        dev = {k: rms[k] - LOGGED[k] for k in LOGGED if k in rms}
        worst = max(abs(v) for v in dev.values()) if dev else float("nan")
        for k in LOGGED:
            if k in rms:
                print(f"    {k:>5}: {rms[k]:5.2f}  (logged {LOGGED[k]:.2f}, "
                      f"delta {dev[k]:+.2f})")
        print(f"  -> max deviation {worst:.2f}")
    else:
        worst = 0.0
        print("  non-velocity rms: " + " ".join(f"{k}={v:.2f}" for k, v in rms.items()
                                                if not k.startswith("v")))
    results[hcol] = (r, rms, worst)
    print()

# use the H_col that reproduces the MAP
best_h = min(results, key=lambda h: results[h][2])
r, rms, worst = results[best_h]
print(f"===== RANKING at H_col = {best_h:.0f} m "
      f"(max non-velocity deviation {worst:.2f}) =====")
if worst >= 0.05:
    print("  WARNING: no scanned H_col reproduces the MAP's logged rms — the")
    print("  MAP's provenance is unrecorded, so this ranking is provisional.")

rows = []
for b in r["obs"]:
    if not b["label"].startswith("v_"):
        continue
    site = b["label"][2:]
    o, p, s = (np.array(b[k]) for k in ("obs", "pred", "sig"))
    res = (p - o) / s
    chi2 = float(np.sum(res**2))
    rows.append(dict(site=site, n=len(o), rms=float(np.sqrt(np.mean(res**2))),
                     chi2=chi2, chi2_red=chi2 / len(o),
                     bias=float(np.mean(res)),
                     rms_mm=float(np.sqrt(np.mean((p - o)**2)) * 1000)))

print(f"{'site':>8} {'N':>4} {'rms(sig)':>9} {'chi2/N':>8} {'bias(sig)':>10} "
      f"{'rms(mm/yr)':>11}")
for d in sorted(rows, key=lambda d: d["chi2_red"]):
    print(f"{d['site']:>8} {d['n']:4d} {d['rms']:9.2f} {d['chi2_red']:8.2f} "
          f"{d['bias']:+10.2f} {d['rms_mm']:11.2f}")

# ---------------------------------------------------------------------------
# Is the ranking explained by DISTANCE from the core?
#
# If the between-site spread is real flow variation across the ~40 km region
# (rather than measurement noise), a site's consistency with the SP19 strata
# should degrade with its distance from the core. That is a falsifiable
# prediction, and it separates the two ways a site can be inconsistent:
# genuinely-a-different-column (scales with distance) vs bad data (does not).
# ---------------------------------------------------------------------------
DEG_KM = 111.195   # meridional degree; sites and core are both within ~19 km
                   # of the pole, so pole-distance proxies core-distance (the
                   # SP19 core sits ~1.1 km from the pole -- an offset at this
                   # range, not a confound).
BAD = 100.0        # chi2/N above this = data-quality failure, not geography

loc = pd.read_csv(HERE / "data" / "apres_site_locations.csv")
loc["dist_km"] = (90.0 - loc.latitude.abs()) * DEG_KM
chi = {d["site"]: d["chi2_red"] for d in rows}
loc["chi2_red"] = loc.site.map(chi)

clean = loc[loc.chi2_red.notna() & (loc.chi2_red < BAD)].sort_values("dist_km")
bad = loc[loc.chi2_red >= BAD]

print("\n===== distance from the core vs strata-consistency =====")
print(f"{'site':>8} {'dist(km)':>9} {'chi2/N':>8}")
for _, d in clean.iterrows():
    print(f"{d.site:>8} {d.dist_km:9.2f} {d.chi2_red:8.2f}")

dist, c2 = clean.dist_km.values, clean.chi2_red.values
sp_r = float(np.corrcoef(np.argsort(np.argsort(dist)),
                         np.argsort(np.argsort(c2)))[0, 1])
pe_r = float(np.corrcoef(dist, c2)[0, 1])
slope, icept = (float(x) for x in np.polyfit(dist, c2, 1))
print(f"\nSpearman rank corr = {sp_r:+.3f}   Pearson = {pe_r:+.3f}   (N={len(clean)})")
print(f"linear fit: chi2/N = {slope:.4f}*dist_km + {icept:+.2f}")
print(f"  -> extrapolated to the core: chi2/N = {icept:+.2f}")
print(f"  CAVEAT: N={len(clean)} sites and the nearest is {dist.min():.1f} km out,")
print("  so the intercept is an extrapolation, not a measurement.")

for _, d in bad.iterrows():
    print(f"\n{d.site}: chi2/N {d.chi2_red:.0f} at {d.dist_km:.2f} km, but a CLEAN "
          f"site sits at {dist.min():.2f} km\n  -> data-quality failure, not a distance effect")

out = R / "vel_strata_ranking.json"
json.dump(dict(map_scored=Path(MAP_PATH).name, H_col=best_h,
               hcol_probe_max_dev=worst, sites=rows,
               distance=dict(spearman=sp_r, pearson=pe_r, slope=slope,
                             intercept=icept, n=len(clean),
                             nearest_km=float(dist.min()))), open(out, "w"), indent=1)
print(f"\nSaved {out}")
print("\nInterpretation: the model here NEVER saw velocity. A site with")
print("chi2/N ~ 1 is the column the SP strata already imply; a large chi2/N")
print("is a different column, and pooling it in as 'noise' is what inflated")
print("the old sigma to ~70x the instrument precision.")
