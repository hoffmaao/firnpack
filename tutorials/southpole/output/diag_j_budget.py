"""diag_j_budget.py — WHICH observations actually drive the inversion?

Panel-by-panel rms hides the weighting: a block's pull on the controls is its
chi^2 SHARE of J, which is 0.5 * N_b * rms_b^2 -- so a block with many points
at a slightly-too-small sigma can silently dominate the objective.

Reports the budget at the given MAP, and re-does it under a per-block sigma
rescale (the Desroziers factor = each block's own rms), i.e. what the budget
would be if every block's error model were self-consistent.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/diag_j_budget.py
"""
from __future__ import annotations
import json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # the tutorial dir, not output/
import sys
sys.path.insert(0, str(HERE))  # sibling config module
from config import build_cfg
R = HERE / "results"
MAP_PATH = os.environ.get("FIRN_WARM_JSON", str(R / "sp_r10_final.json"))

# Score the MAP under the error model it was PRODUCED with: r10 ran with ezz
# free, so pinning it here would charge r10's firn-diagnosed ezz a large prior
# penalty and swamp the very shares this table exists to show.
os.environ.update(FIRN_VEL_SITE="x17s2+x11n0+x11n2+x11n6+x11s2",
                  FIRN_VEL_SRC="zeising", FIRN_SEAS="1", FIRN_HCOL="300",
                  FIRN_EZZ_SITE="none")
_b = build_cfg()

from firnpack.inverse import assimilate
from firnpack.inverse.diagnostics import j_budget
warm = json.load(open(MAP_PATH))
r = assimilate(_b.cfg, mode="forward", warm=warm, verbose=False)

budget = j_budget(r)
rows = budget["blocks"]
J_obs = budget["J_obs"]
J_tot = budget["J"]
print(f"\nMAP {Path(MAP_PATH).name}: J = {J_tot:.2f}   "
      f"(obs blocks {J_obs:.2f}, priors+rest {J_tot-J_obs:.2f})")

print(f"\n===== AS RUN (the sigmas r10 actually used) =====")
print(f"{'block':>10} {'N':>5} {'rms':>6} {'chi2':>9} {'% of J':>8}")
for d in sorted(rows, key=lambda d: -d["chi2"]):
    print(f"{d['label']:>10} {d['n']:5d} {d['rms']:6.2f} {d['chi2']:9.2f} "
          f"{100*d['chi2']/J_tot:7.1f}%")

# ---- self-consistent error model: rescale each block's sigma by its own rms -
print(f"\n===== IF each block's sigma were self-consistent "
      f"(sigma *= its own rms) =====")
print(f"{'block':>10} {'N':>5} {'sig x':>6} {'chi2':>9} {'% of J':>8}")
J2_obs = sum(0.5*d["n"] for d in rows)      # every block -> rms 1 by construction
J2 = J2_obs + (J_tot - J_obs)
for d in sorted(rows, key=lambda d: -0.5*d["n"]):
    print(f"{d['label']:>10} {d['n']:5d} {d['rms']:6.2f} {0.5*d['n']:9.2f} "
          f"{100*0.5*d['n']/J2:7.1f}%")
print(f"\n  J would be {J2:.1f} instead of {J_tot:.1f}")

print("\n===== reading =====")
top = max(rows, key=lambda d: d["chi2"])
print(f"  As run, '{top['label']}' carries {100*top['chi2']/J_tot:.0f}% of the "
      f"objective on {top['n']} points at rms {top['rms']:.2f}.")
print("  A block whose residual is WHITE and UNFITTABLE (see")
print("  diag_dage_residual.py) is noise: weight spent there is weight the")
print("  other observables do not get. This is a WEIGHTING statement, separate")
print("  from how the misfit should be DISPLAYED in the paper's figure.")

json.dump(dict(map=Path(MAP_PATH).name, J=J_tot, blocks=rows),
          open(R/"j_budget.json", "w"), indent=1)
print(f"\nSaved {R/'j_budget.json'}")
