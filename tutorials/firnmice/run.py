"""tutorials/firnmice/run.py - the FirnMICE step-change intercomparison suite.

Forward-model verification: the six synthetic step-change experiments of
Lundin and others (2017), run with this model so its response can be placed
against the other firn models that contributed to the intercomparison.

  ex1-ex3   +5 K surface-temperature step at t = 100 yr, accumulation fixed
            at 0.10 m i.e./yr, at base temperatures -50, -40 and -30 C.
  ex4-ex6   +0.05 m i.e./yr accumulation step at t = 100 yr, temperature
            fixed at -30 C, at base accumulations 0.02, 0.15 and 0.25 m i.e./yr.

The two step directions isolate the two controls on firn structure: warming
speeds densification, so depth-integrated porosity (DIP) falls and the bubble
close-off (BCO) horizon rises; extra accumulation buries the column faster, so
DIP rises and BCO deepens.

The column is 1000 m so the domain carries the thermal mass of the ice sheet
beneath the firn and the basal boundary cannot contaminate the near-surface
gradients. The mesh is stretched toward the surface where the density gradient
is sharp.

Each experiment writes output/firnmice_<name>.h5 - a Firedrake CheckpointFile of
the state plus a `/firnmice` group of time-series diagnostics. plot.py is a pure
reader of those files and draws the summary figure.

Env flags (all optional):
  FIRNMICE_FULL=1              full 2000-yr experiments (default: 200-yr short run)
  FIRNMICE_SPINUP_YEARS        spinup length (default 5000 full / 1000 short)
  FIRNMICE_DT_YEARS            timestep (default 1.0)
  FIRNMICE_NZ                  elements (default 320 full / 220 short)
  FIRNMICE_SEASONAL=1          add a seasonal cycle to the surface temperature

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/firnmice/run.py
"""
from __future__ import annotations

import os
from pathlib import Path

from firnpack.firnmice import (
    FIRNMICE_EXPERIMENTS,
    FirnMICEExperiment,
    run_firnmice_experiment,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes")


def _num(name: str, default: float) -> float:
    v = os.getenv(name, "")
    return float(v) if v.strip() else float(default)


FULL = _flag("FIRNMICE_FULL")
# short mode keeps the step at 100 yr but stops at 200 yr, so the suite stays
# runnable interactively; the full protocol integrates 2000 yr past the step.
T_END_SHORT = 200.0

H0 = _num("FIRNMICE_H0", 1000.0)
NZ = int(_num("FIRNMICE_NZ", 320 if FULL else 220))
STRETCH_P = _num("FIRNMICE_STRETCH_P", 3.0)
DT_YEARS = _num("FIRNMICE_DT_YEARS", 1.0)
SPINUP_YEARS = _num("FIRNMICE_SPINUP_YEARS", 5000.0 if FULL else 1000.0)
SAVE_EVERY = _num("FIRNMICE_SAVE_EVERY_YEARS", 1.0 if FULL else 10.0)

print(f"FirnMICE suite - {'FULL' if FULL else 'SHORT'} mode")
print(f"  column {H0:.0f} m, NZ={NZ}, dt={DT_YEARS:.2f} yr, spinup={SPINUP_YEARS:.0f} yr")
print(f"  output -> {OUT}")

# FIRNMICE_ONLY=ex4 (or ex1,ex4) re-runs a subset in place, e.g. after a change
# to a diagnostic, without repeating the whole ~2.4 h suite.
_only = [s.strip() for s in os.getenv("FIRNMICE_ONLY", "").split(",") if s.strip()]
_todo = [e for e in FIRNMICE_EXPERIMENTS if not _only or e.name in _only]
if _only:
    print(f"  running subset: {', '.join(e.name for e in _todo)}")

for exp0 in _todo:
    exp = exp0 if FULL else FirnMICEExperiment(
        name=exp0.name, T0_C=exp0.T0_C, a0_mieq_yr=exp0.a0_mieq_yr,
        dT_C=exp0.dT_C, da_mieq_yr=exp0.da_mieq_yr,
        t_step_yr=exp0.t_step_yr, t_end_yr=T_END_SHORT,
    )
    res = run_firnmice_experiment(
        exp=exp, out_dir=OUT, H0=H0, NZ=NZ, STRETCH_P=STRETCH_P,
        dt_years=DT_YEARS, spinup_years=SPINUP_YEARS,
        save_every_years=SAVE_EVERY,
        # figures are plot.py's job - the run writes data only
        make_plots=False, use_seasonal=_flag("FIRNMICE_SEASONAL"),
        rho_surf_kg_m3=360.0,
    )
    dip = res["DIP_m"]
    print(f"  {exp.name}: DIP {dip[0]:.2f} -> {dip[-1]:.2f} m "
          f"({len(res['time_years'])} samples)")

print("\nDone. Rebuild the figure with: "
      "PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/firnmice/plot.py")
