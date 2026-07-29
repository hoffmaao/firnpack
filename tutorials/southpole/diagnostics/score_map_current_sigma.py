"""score_map_current_sigma.py — re-score an archived MAP under the CURRENT error model.

Writes a forward-mode result JSON (same schema as an optimize result, so
plot.py can draw misfit figures from it) for a stored MAP evaluated against
the error model the config builds TODAY. The point: a MAP's own JSON carries
the sigmas it was inverted under, so after a recalibration the misfit figure
keeps showing the old bars until a new inversion lands — this closes that gap
without waiting hours for L-BFGS-B.

The tag is suffixed "_rescored" so it can never be mistaken for (or clobber)
an inverted MAP.

Env: FIRN_MAP (default sp_fresh.json); usual FIRN_* knobs.
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/diagnostics/score_map_current_sigma.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from config import build_cfg
from firnpack.inverse import assimilate

OUT = HERE / "output"
MAP_NAME = os.environ.get("FIRN_MAP", "sp_fresh.json")
MAP = json.load(open(OUT / MAP_NAME))
tag = Path(MAP_NAME).stem + "_rescored"

cfg = build_cfg().cfg
r = assimilate(cfg, mode="forward", warm=MAP, annotate=False, verbose=False)

out = dict(name=cfg.name, J=r["J"], message=f"forward re-score of {MAP_NAME} "
           "under the current error model (not an inversion)",
           m_map=MAP["m_map"], n_ctrl=r["n_ctrl"])
for k in ("T_knots", "knot_years", "T_prior_centers", "T_prior_sigma",
          "b_knots", "b_knot_years", "b_prior_centers"):
    if k in MAP: out[k] = MAP[k]
out.update(J_map=r["J"], diag=r["diag"], profiles=r["profiles"],
           params=r["params"], obs=r["obs"], n_steps=r["n_steps"])
json.dump(out, open(OUT / f"{tag}.json", "w"), indent=2)
print(f"J = {r['J']:.4f} under the current error model "
      f"(stored J_map was {MAP.get('J_map', float('nan')):.4f} under its own)")
print("  rms: " + " ".join(f"{k[4:]}={v:.3f}" for k, v in r["diag"].items()
                           if k.startswith("rms_")))
print(f"wrote {OUT / (tag + '.json')} — run plot.py to redraw figures")
