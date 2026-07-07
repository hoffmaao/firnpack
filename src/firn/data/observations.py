"""Observation data for firn model calibration and inversion.

Sparse point observations from ice cores (density, age) and boreholes
(temperature) used as targets in MAP/Bayesian inversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PointData:
    """Sparse point observations at discrete depths.

    Parameters
    ----------
    depth : array-like
        Observation depths below the surface (m). Must be positive.
    values : array-like
        Observed values at each depth.
    sigma : array-like or float
        Measurement uncertainty (1-σ). Scalar or per-observation.
    name : str, optional
        Human-readable label (e.g. "SP19 density").
    units : str, optional
        Physical units (e.g. "kg/m3", "yr", "K").
    """
    depth: np.ndarray
    values: np.ndarray
    sigma: np.ndarray | float
    name: str = ""
    units: str = ""

    def __post_init__(self):
        self.depth = np.asarray(self.depth, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if np.isscalar(self.sigma):
            self.sigma = np.full_like(self.values, float(self.sigma))
        else:
            self.sigma = np.asarray(self.sigma, dtype=float)

    def __len__(self):
        return len(self.depth)

    def subsample(self, stride: int) -> "PointData":
        """Return every `stride`-th observation."""
        return PointData(
            depth=self.depth[::stride],
            values=self.values[::stride],
            sigma=self.sigma[::stride],
            name=self.name,
            units=self.units,
        )

    def clip_depth(self, max_depth: float) -> "PointData":
        """Return observations shallower than max_depth."""
        mask = self.depth <= max_depth
        return PointData(
            depth=self.depth[mask],
            values=self.values[mask],
            sigma=self.sigma[mask],
            name=self.name,
            units=self.units,
        )

    def __repr__(self):
        return (f"PointData('{self.name}', n={len(self)}, "
                f"depth={self.depth.min():.1f}-{self.depth.max():.1f}m)")


@dataclass
class Observations:
    """Collection of observation datasets for a firn site.

    Each field is an optional PointData. The inversion framework
    uses whichever fields are provided.
    """
    density: Optional[PointData] = None
    age: Optional[PointData] = None
    temperature: Optional[PointData] = None

    @property
    def fields(self) -> dict[str, PointData]:
        """Return a dict of non-None observation fields."""
        out = {}
        if self.density is not None:
            out["density"] = self.density
        if self.age is not None:
            out["age"] = self.age
        if self.temperature is not None:
            out["temperature"] = self.temperature
        return out

    def __repr__(self):
        parts = []
        for name, data in self.fields.items():
            parts.append(f"{name}: {len(data)} pts")
        return f"Observations({', '.join(parts)})"
