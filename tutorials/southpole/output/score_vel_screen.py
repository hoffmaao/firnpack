"""score_vel_screen.py — rank ApRES sites by the misfit of the OTHER data.

For each screen result (sp_vel_<site>.json), evaluate the engine forward at
that MAP (with the matching velocity config) and score

    J_other = sum over {rho, age, dage, T} of 0.5 * N_b * rms_b^2

i.e. the per-point chi^2 of everything EXCEPT velocity. Rank by
dJ_other = J_other(site) - J_other(none): a velocity profile consistent with
the rest of the system costs ~nothing; an inconsistent one drags the physics
and degrades the other fits. Also reports where each site pulls ezz.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/score_vel_screen.py
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; R = HERE/"results"
from firnpack.inverse import assimilate

SITES = ["none","x11n0","x11n10","x11n2","x11n6","x11s2","x17n2","x17s2","x5n2","x5s2"]
NB = {}  # per-block N, filled from the first forward
rows = []
for s in SITES:
    p = R/f"sp_vel_{s}.json"
    if not p.exists():
        print(f"(missing {p.name})"); continue
    warm = json.load(open(p))
    os.environ["FIRN_VEL_SITE"] = s
    ns = {"__file__": str(HERE/"run.py"), "__name__": "cfgbuild"}
    src = open(HERE/"run.py").read().split("warm = json.load")[0]
    exec(src, ns)
    r = assimilate(ns["cfg"], mode="forward", warm=warm, verbose=False)
    Jo = 0.0
    for b in r["obs"]:
        if b["label"].startswith("v"): continue
        res = (np.array(b["pred"])-np.array(b["obs"]))/np.array(b["sig"])
        Jo += 0.5*float(np.sum(res**2))
    ez = warm["m_map"].get("ezz_yr", float("nan"))
    vrms = r["diag"].get("rms_v", float("nan"))
    rows.append((s, Jo, ez, vrms, warm["J"]))
    print(f"  scored {s}: J_other={Jo:.2f} ezz={ez:+.2e} v_rms={vrms if vrms==vrms else float('nan'):.2f}")

base = dict((s, J) for s, J, *_ in rows)["none"]
print("\n===== RANKING (dJ_other vs no-velocity control) =====")
print(f"{'site':8s} {'J_other':>9s} {'dJ_other':>9s} {'ezz (/yr)':>11s} {'v rms':>6s}")
for s, Jo, ez, vrms, Jt in sorted(rows, key=lambda t: t[1]):
    mark = "  <- control" if s == "none" else ""
    print(f"{s:8s} {Jo:9.2f} {Jo-base:+9.2f} {ez:+11.2e} {vrms:6.2f}{mark}")
json.dump([dict(site=s, J_other=Jo, dJ_other=Jo-base, ezz=ez, v_rms=vrms)
           for s, Jo, ez, vrms, _ in rows],
          open(R/"vel_screen_ranking.json", "w"), indent=1)
print(f"Saved {R/'vel_screen_ranking.json'}")
