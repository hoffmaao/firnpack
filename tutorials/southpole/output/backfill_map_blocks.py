"""Backfill obs/pred/sig blocks into a MAP JSON written before the engine saved them.

run.py now ends every optimisation with a forward at the MAP and stores the
per-block obs/sig/pred arrays and the model profiles in its results JSON, so
plot.py can draw figures without touching Firedrake. MAPs produced before that
-- notably the frozen reference sp_joint_r8.json -- carry only the parameters,
which is why the old figure scripts each had to re-solve to draw anything.

This runs one forward at the stored MAP and merges the missing arrays in.
Existing keys are never modified: r8 carries hand-written provenance
(H_col, INIT, START, weighting, ...) that is not reproducible from a solve, so
the merge only adds. m_map is re-read afterwards and compared, and the script
refuses to write if the parameters moved -- a backfill that changed the MAP
would mean the config no longer matches the result it claims to describe.

Run (from the repo root):
    PYTHONPATH=src OMP_NUM_THREADS=1 python \
        tutorials/southpole/output/backfill_map_blocks.py

    FIRN_MAP_JSON=results/sp_engine_dense.json FIRN_KNOTS=knots_dense.json ... \
        to backfill a different MAP; FIRN_KNOTS must match that MAP's layout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # the tutorial dir, not output/
import sys
sys.path.insert(0, str(HERE))  # sibling config module
from config import build_cfg

MAP_PATH = Path(os.environ.get("FIRN_MAP_JSON", HERE / "results/sp_joint_r8.json"))
if not MAP_PATH.is_absolute():
    MAP_PATH = HERE / MAP_PATH

existing = json.load(open(MAP_PATH))
if "obs" in existing and "profiles" in existing:
    print(f"{MAP_PATH.name}: already has obs+profiles; nothing to do.")
    raise SystemExit(0)

# run.py's config, built by exec'ing it up to the point it loads a warm start.
_b = build_cfg()

from firnpack.inverse import assimilate  # noqa: E402  (after the cfg exec)

print(f"forward at {MAP_PATH.name} ...")
r = assimilate(_b.cfg, mode="forward", warm=existing, verbose=False)
print(f"  J = {r['J']:.4f}   (stored J = {existing.get('J', float('nan')):.4f})")
print("  rms: " + " ".join(f"{k[4:]}={v:.2f}" for k, v in r["diag"].items()
                           if k.startswith("rms_")))

# A forward at the MAP must reproduce the MAP's own J. Checking the parameters
# is NOT sufficient and will not catch the failure that matters: the warm start
# pins m_map, so the parameters always match, while the *error model* around
# them may not. r8 was fitted under the pre-2026-07 sigmas and replays at
# J=146.86 under today's defaults against its stored J=81.3061 -- same
# parameters, different sigmas. Writing those arrays in would leave a file
# whose J says one error model and whose obs/sig say another.
stored_J = existing.get("J")
if stored_J is None:
    raise SystemExit("refusing to write: MAP has no stored J to check against")
if abs(r["J"] - stored_J) > 1e-3 * max(1.0, abs(stored_J)):
    raise SystemExit(
        f"refusing to write: forward J={r['J']:.4f} does not reproduce stored "
        f"J={stored_J:.4f}.\nThe parameters are pinned by the warm start, so this is "
        f"the error model or knot layout disagreeing, not the MAP.\nFor the frozen r8, "
        f"replay it under the error model it was fitted with:\n"
        f"  FIRN_SIG_DAGE_LEGACY=1 FIRN_SIG_DAGE_SCALE=2.2 FIRN_AGE_BLOCK=1 FIRN_BASAL=Q"
    )

stored = existing.get("m_map", {})
drift = {k: (stored[k], r["params"][k]) for k in stored
         if k in r["params"] and abs(r["params"][k] - stored[k]) > 1e-9 * max(1.0, abs(stored[k]))}
if drift:
    for k, (a, b) in drift.items():
        print(f"  !! {k}: stored {a:.6e} vs forward {b:.6e}")
    raise SystemExit("refusing to write: the forward does not reproduce the stored MAP")

merged = dict(existing)  # never overwrite: r8's provenance keys are hand-written
for key in ("obs", "profiles", "diag"):
    merged[key] = r[key]
merged["J_forward"] = r["J"]

json.dump(merged, open(MAP_PATH, "w"), indent=2)
added = ", ".join(k for k in ("obs", "profiles", "diag") if k not in existing)
print(f"\nAdded [{added}] to {MAP_PATH.name} "
      f"({MAP_PATH.stat().st_size / 1024:.0f} KB). Existing keys untouched.")
