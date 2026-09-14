"""tutorials/aquifer/run_era5.py - the SE Greenland aquifer under ERA5 forcing.

Runs the experiments in ``config.ERA5_EXPERIMENTS``: the real aquifer belt,
forced with 3-hourly ERA5 through the firnpack surface energy balance, one
block of years cycled until the column stops adjusting. Writes one JSON per
experiment into ``output/``; ``plot_fac_saturation.py`` and ``plot.py`` read
those without Firedrake.

Why equilibrium runs rather than a transient through the record: a column
started from a dry spinup carries a deep cold reservoir that decades of melt
have to overcome, so a 14-year transient measures the initial condition. The
firn at an aquifer site has been taking melt for longer than that.
``ReanalysisSite`` wraps time modulo the record, so a long run cycles the
same climate and the column equilibrates to it.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/aquifer/run_era5.py [name ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import ERA5_COLUMN, ERA5_EXPERIMENTS, ERA5_FORCING, OUTPUT

from firnpack.aquifer import ReanalysisSite, run_aquifer_column
from firnpack.models.firn_richards import FirnRichardsParameters
from firnpack.surface_energy import SurfaceEnergyParameters

OUTPUT.mkdir(parents=True, exist_ok=True)

PERMEABILITY = {
    "calonne": lambda: FirnRichardsParameters(perm_scale=1.0),
    "deep0.1": lambda: FirnRichardsParameters(perm_scale=1.0, perm_scale_deep=0.1),
}


def run_one(name: str) -> dict:
    y0, y1, floor, perm = ERA5_EXPERIMENTS[name]
    site = ReanalysisSite(
        ERA5_FORCING, y0, y1, albedo="model",
        seb_params=SurfaceEnergyParameters(albedo_firn=floor), name=name,
        label=f"ERA5 {y0}-{y1}, albedo floor {floor:.2f}, {perm}")
    t0 = time.time()
    r = run_aquifer_column(site=site, richards_params=PERMEABILITY[perm](),
                           verbose=False, **ERA5_COLUMN)
    r["experiment"] = dict(name=name, years=[y0, y1], albedo_floor=floor,
                           permeability=perm, seconds=time.time() - t0)
    return r


def summarise(r: dict) -> str:
    t = np.asarray(r["time_years"])
    yrs = max(t[-1] - t[0], 1e-9)
    melt = r["melt_cum_kg_m2"][-1]
    refr = r["refreeze_cum_kg_m2"][-1]
    drain = r.get("drained_kg_m2", 0.0)
    d = np.asarray(r["depth_m"])
    rho = np.asarray(r["rho_profiles"])[-1]
    S = np.asarray(r["S_profiles"])
    co = d[rho >= 830.0]
    from firnpack.aquifer import wet_layer
    wt_top, _ = wet_layer(d, S[-1], 0.5)
    ablation = r.get("ablation_steps", 0)
    return (f"melt {melt / yrs:5.0f}  refroze {refr / yrs:5.0f} ({100 * refr / melt:.0f}%)  "
            f"drained {drain / yrs:4.0f}  net recharge {(melt - refr - drain) / yrs:5.0f} kg/m2/yr | "
            f"close-off {co.min() if co.size else float('nan'):5.1f} m  "
            f"water table {wt_top:5.1f} m  "
            f"S max {S.max():.2f} | budget {r['budget_residual_kg_m2']:+.2f}"
            + (f" | ABLATION STEPS {ablation}" if ablation else ""))


if __name__ == "__main__":
    names = sys.argv[1:] or list(ERA5_EXPERIMENTS)
    unknown = [n for n in names if n not in ERA5_EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiment(s) {unknown}; "
                         f"choose from {list(ERA5_EXPERIMENTS)}")
    failed = []
    for name in names:
        y0, y1, floor, perm = ERA5_EXPERIMENTS[name]
        print(f"\n=== {name}: ERA5 {y0}-{y1}, albedo floor {floor}, {perm}, "
              f"{ERA5_COLUMN['run_years']:.0f} yr", flush=True)
        # Independent experiments: one failing must not cost the others.
        try:
            r = run_one(name)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            state = getattr(exc, "state", None)
            if state:
                print(f"    at t = {state['t_yr']:.3f} yr, melt {state['melt_m_yr']:.2f} m/yr, "
                      f"head range [{min(state['head_dg']):.0f}, {max(state['head_dg']):.0f}] m")
                # the column as the failed step saw it, for a restart or a look
                path = OUTPUT / f"failed_{name}.json"
                path.write_text(json.dumps(state))
                print(f"    state -> {path}")
            failed.append(name)
            continue
        print("    " + summarise(r))
        payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in r.items()}
        path = OUTPUT / f"aquifer_{name}.json"
        path.write_text(json.dumps(payload))
        print(f"    -> {path}  ({r['experiment']['seconds']:.0f} s)")
    if failed:
        print(f"\n{len(failed)} experiment(s) did not complete: {', '.join(failed)}")
