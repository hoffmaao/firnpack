"""tutorials/summit/run.py — Summit (Greenland) inversion via the shared engine.

Recovers Summit's OWN densification law + accumulation history from density
(composite core) + layer-gradient d(age)/dz (GISP2), under a prescribed
near-isothermal temperature forcing (FirnCover deep-mean -28.8 C; Summit's
12 m firn-T can't support a deep thermal inversion, so T is an input, not a
target). Same neutral literature-H&L prior as South Pole, so the two recovered
laws are directly comparable — the paper's transferability result.

No velocity/ApRES and no deep borehole T at Summit -> those observables and the
ezz / conductivity / Q_base controls are simply dropped (fixed).

The config (including the data-derived error model and its legacy guards) lives
in the sibling config.py; diagnostics/ scripts build the same config from
there. The archived summit_invert.json reproduces exactly (J=81.9829) under
FIRN_SIG_RHO_LEGACY=1 FIRN_SIG_DAGE_LEGACY=1 FIRN_AGE_BLOCK=1.

Modes (env FIRN_MODE): "verify" (replay + FD), "validate" (forward J at the
warm start), "optimize" (default). FIRN_WARM_JSON warm-starts from a MAP json.
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/summit/run.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from firnpack.inverse import assimilate

sys.path.insert(0, str(Path(__file__).parent))  # for the sibling config module
from config import build_cfg

_b = build_cfg()
cfg, MODE, T_FORCE = _b.cfg, _b.MODE, _b.T_FORCE

print("Summit inversion: " + ", ".join(f"{ob.n} {ob.label}" for ob in cfg.obs)
      + f"; T forcing {T_FORCE} C; {len(cfg.b_knots.years)} b-knots (accum)")
_wj = os.environ.get("FIRN_WARM_JSON")
warm = json.load(open(_wj)) if _wj else None
if MODE == "verify":
    assimilate(cfg, mode="verify", warm=warm, fd_names=["hl_Ea2","s2_shape","b1600"])
elif MODE == "validate":
    r = assimilate(cfg, mode="forward", warm=warm)
    print(f"\nENGINE forward J = {r['J']:.4f}")
    print("  rms: " + " ".join(f"{k[4:]}={v:.3f}" for k, v in r["diag"].items()
                               if k.startswith("rms_")))
else:
    assimilate(cfg, mode="optimize", warm=warm)
