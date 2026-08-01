"""Climate history data for firn column boundary conditions.

A ClimateHistory holds time-varying environmental records that drive
the firn model: surface temperature, accumulation rate, and optionally
other fields.  It can be built from pandas DataFrames, numpy arrays,
or loaded from CSV files.

These are not simple "forcing" - they represent the full complexity of
environmental conditions at a site, potentially combining reanalysis
products, paleoclimate reconstructions, and station observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class ClimateHistory:
    """Time-varying climate record for driving a firn column model.

    Parameters
    ----------
    years : array-like
        Calendar years (CE) for each record. Must be monotonically increasing.
    temperature : array-like
        Surface temperature in Kelvin at each year.
    accumulation : array-like
        Surface accumulation rate in m ice-equivalent per year.
    source : str, optional
        Provenance label (e.g. "ERA5", "Buizert2021", "station").
    temperature_sigma : array-like or float, optional
        Uncertainty in temperature (K). Scalar or per-year.
    accumulation_sigma : array-like or float, optional
        Uncertainty in accumulation (m ice eq / yr). Scalar or per-year.
    """
    years: np.ndarray
    temperature: np.ndarray   # K
    accumulation: np.ndarray  # m ice eq / yr
    source: str = ""
    temperature_sigma: float | np.ndarray = 0.0
    accumulation_sigma: float | np.ndarray = 0.0

    def __post_init__(self):
        self.years = np.asarray(self.years, dtype=float)
        self.temperature = np.asarray(self.temperature, dtype=float)
        self.accumulation = np.asarray(self.accumulation, dtype=float)
        if len(self.years) != len(self.temperature):
            raise ValueError("years and temperature must have the same length")
        if len(self.years) != len(self.accumulation):
            raise ValueError("years and accumulation must have the same length")

    def __len__(self):
        return len(self.years)

    @property
    def dt_years(self) -> float:
        """Average timestep in years."""
        return float(np.mean(np.diff(self.years)))

    @property
    def span(self) -> tuple[float, float]:
        """(start_year, end_year) in CE."""
        return (float(self.years[0]), float(self.years[-1]))

    def resample(self, dt_years: float) -> "ClimateHistory":
        """Block-average to a coarser timestep."""
        blk = int(round(dt_years / self.dt_years))
        if blk <= 1:
            return self
        n_blk = len(self.years) // blk
        years_new = np.array([self.years[i*blk:(i+1)*blk].mean() for i in range(n_blk)])
        T_new = np.array([self.temperature[i*blk:(i+1)*blk].mean() for i in range(n_blk)])
        A_new = np.array([self.accumulation[i*blk:(i+1)*blk].mean() for i in range(n_blk)])
        return ClimateHistory(
            years=years_new, temperature=T_new, accumulation=A_new,
            source=self.source + f" (resampled dt={dt_years}yr)",
        )

    def sub_annual(self, dt_years: float) -> "ClimateHistory":
        """Repeat each year's data for sub-annual timesteps."""
        n_sub = int(round(1.0 / dt_years))
        return ClimateHistory(
            years=np.repeat(self.years, n_sub),
            temperature=np.repeat(self.temperature, n_sub),
            accumulation=np.repeat(self.accumulation, n_sub),
            source=self.source + f" (sub-annual dt={dt_years}yr)",
        )

    def concatenate(self, other: "ClimateHistory") -> "ClimateHistory":
        """Join two records end-to-end (self first, other second)."""
        return ClimateHistory(
            years=np.concatenate([self.years, other.years]),
            temperature=np.concatenate([self.temperature, other.temperature]),
            accumulation=np.concatenate([self.accumulation, other.accumulation]),
            source=f"{self.source} + {other.source}",
        )

    def __repr__(self):
        return (f"ClimateHistory({self.span[0]:.0f}-{self.span[1]:.0f} CE, "
                f"n={len(self)}, dt≈{self.dt_years:.1f}yr, source='{self.source}')")
