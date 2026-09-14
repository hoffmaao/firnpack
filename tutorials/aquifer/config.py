"""tutorials/aquifer/config.py - shared settings for the aquifer experiments.

Kept separate from run.py so the diagnostics and the plotter agree with the run
without importing Firedrake.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).parent
OUTPUT = HERE / "output"
FIGURES = HERE / "figures"


def _f(name: str, default: float) -> float:
    v = os.getenv(name, "")
    return float(v) if v.strip() else default


# Column and integration settings. The column is deep enough that the pore
# space is nowhere near full over the run - with no lateral outlet, a shallow
# column would fill and the experiment would be measuring the domain rather
# than the climate.
COLUMN = dict(
    H0=_f("AQ_H0", 60.0),
    NZ=int(_f("AQ_NZ", 140)),
    # near-uniform: heavy surface stretching makes the top cell thinner than a
    # single step's melt and the Richards solve cannot take it (see
    # firnpack.aquifer.run_aquifer_column)
    stretch_p=_f("AQ_STRETCH", 1.0),
    spinup_years=_f("AQ_SPINUP", 50.0),
    run_years=_f("AQ_YEARS", 12.0),
    dt_days=_f("AQ_DT_DAYS", 3.0),
    save_every_days=_f("AQ_SAVE_DAYS", 15.0),
)

# Storage below this counts as "no liquid water left" for the persistence test.
PERSIST_FLOOR_KG_M2 = _f("AQ_FLOOR", 1.0)
PERSIST_WINDOW_YEARS = _f("AQ_WINDOW", 3.0)


# ---------------------------------------------------------------------------
# ERA5-forced experiments at the real site (run_era5.py)
# ---------------------------------------------------------------------------
# Forcing: data/era5_hourly_aoi_seb.csv.gz, 3-hourly ERA5 over the ice-only
# cells of the 1200-1800 m band of the SE Greenland aquifer belt, run through
# firnpack.surface_energy for melt. Each experiment cycles one block of years
# until the column has equilibrated to that climate.
ERA5_FORCING = HERE / "data" / "era5_hourly_aoi_seb.csv.gz"

ERA5_COLUMN = dict(
    H0=60.0, NZ=140, stretch_p=1.0,
    spinup_years=20.0, run_years=_f("AQ_ERA5_YEARS", 80.0),
    dt_days=3.0, save_every_days=30.0,
)

# name -> (first year, last year, albedo floor, permeability)
#
# albedo floor   the aged-firn albedo the ageing model decays to. It is the
#                largest free knob in the energy balance and the melt is
#                nearly linear in it, so it is swept rather than fixed:
#                0.76 -> 0.46, 0.74 -> 0.54, 0.72 -> 0.62, 0.70 -> 0.71
#                m w.e./yr. Mapping and its provenance:
#                ReanalysisSite.ALBEDO_FIRN_SE_GREENLAND in firnpack.aquifer.
# permeability   "calonne"  Calonne et al. (2012) from density and grain size
#                "deep0.1"  the same, reduced 10x only in firn denser than
#                           ~600 kg/m3, where the Helheim aquifer conductivity
#                           was measured (Miller et al. 2017) and Calonne runs
#                           high. Whether recharge is sensitive to this
#                           is an open question: the earlier "insensitive"
#                           result was obtained with a gravity flux that
#                           took the receiving cell's conductivity, which
#                           would be insensitive to the donor's by
#                           construction. Withdrawn pending the re-run.
ERA5_EXPERIMENTS = {
    "recent_a76":     (2006, 2019, 0.76, "calonne"),
    "recent_a74":     (2006, 2019, 0.74, "calonne"),
    "recent_a72":     (2006, 2019, 0.72, "calonne"),
    "recent_a70":     (2006, 2019, 0.70, "calonne"),
    "recent_a72_deep": (2006, 2019, 0.72, "deep0.1"),
    "midcentury_a72": (1940, 1959, 0.72, "calonne"),
}
