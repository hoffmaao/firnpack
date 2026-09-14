"""tutorials/aquifer/run.py - do firn aquifers persist? A 1D confined column.

Runs the four contrast experiments in ``firnpack.aquifer`` and writes one JSON
per site into ``output/``. ``plot.py`` rebuilds the figures from those without
needing Firedrake.

The question is Kuipers Munneke et al. (2014)'s: an aquifer persists when
accumulation buries summer meltwater below the winter cold wave faster than
that cold wave can refreeze it. Each contrast case changes exactly one control
away from the SE Greenland base case - accumulation, cold content, or melt -
so the answer can be attributed.

Scope is 1D and confined: no lateral transport; water leaves by refreezing or
by riding out of the base with the compacting firn. Lateral drainage sets the
equilibrium water-table depth; it does not decide survival, which is what
these runs measure.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/aquifer/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import COLUMN, OUTPUT, PERSIST_FLOOR_KG_M2, PERSIST_WINDOW_YEARS

from firnpack.aquifer import AQUIFER_SITES, run_aquifer_column, persists

OUTPUT.mkdir(parents=True, exist_ok=True)

only = sys.argv[1:] or None
print(f"Aquifer experiments (1D, confined): {COLUMN['run_years']:.0f} yr after "
      f"{COLUMN['spinup_years']:.0f} yr spinup, dt = {COLUMN['dt_days']:.0f} d")

failed = []
for site in AQUIFER_SITES:
    if only and site.name not in only:
        continue
    print(f"\n=== {site.name}: {site.label}")
    # One case failing must not cost the others: these are independent
    # experiments and a contrast is only interpretable against the cases that
    # did run. Report the failure and carry on.
    try:
        res = run_aquifer_column(site=site, **COLUMN)
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        failed.append(site.name)
        continue
    res["persists"] = persists(res, last_years=PERSIST_WINDOW_YEARS,
                               floor_kg_m2=PERSIST_FLOOR_KG_M2)

    t = np.asarray(res["time_years"])
    s = np.asarray(res["storage_kg_m2"])
    sel = t >= t[-1] - PERSIST_WINDOW_YEARS
    print(f"    melt {res['melt_cum_kg_m2'][-1]:8.0f}   "
          f"refroze {res['refreeze_cum_kg_m2'][-1]:8.0f}   "
          f"stored {s[-1]:8.0f} kg/m2   "
          f"budget residual {res['budget_residual_kg_m2']:+.1f}")
    base = res["storage0_kg_m2"]
    print(f"    last {PERSIST_WINDOW_YEARS:.0f} yr liquid water above the "
          f"retention trace ({base:.0f} kg/m2): "
          f"winter minimum {(s[sel] - base).min():8.1f}, "
          f"summer maximum {(s[sel] - base).max():8.1f} kg/m2")
    print(f"    PERENNIAL WATER: {'YES' if res['persists'] else 'NO'}")

    payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in res.items()}
    path = OUTPUT / f"aquifer_{site.name}.json"
    path.write_text(json.dumps(payload))
    print(f"    -> {path}")

if failed:
    print(f"\n{len(failed)} case(s) did not complete: {', '.join(failed)}")
