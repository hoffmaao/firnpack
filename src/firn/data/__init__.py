"""Data abstractions for firn modeling.

Climate histories, boundary conditions, and observation data used to
drive and calibrate firn column models.  These are NOT simple "forcing" —
they carry metadata, uncertainties, and may come from heterogeneous sources
(reanalyses, reconstructions, station measurements, ice cores, boreholes).
"""

from firn.data.climate import ClimateHistory
from firn.data.observations import Observations, PointData

__all__ = [
    "ClimateHistory",
    "Observations",
    "PointData",
]
