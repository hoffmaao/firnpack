"""McMurdo ApRES processing helpers.

This package is a Python translation / adaptation of the MATLAB processing
workflow you provided (Pr_pRES_preprocess.m, Pr_pRES_strain.m and the
supporting functions in Codes.zip).

The public entry point for end users is usually:

    python scripts/mcmurdo_apres_meltrates.py

The functions here are intentionally kept fairly "MATLAB-like" in naming
and structure so it is easier to cross-reference the original workflow.
"""

from .config import ProcessingConfig
from .preprocess import preprocess_file
from .strain import align_coarse, align_fine, fit_ice, strain_melt_between_profiles

__all__ = [
    "ProcessingConfig",
    "preprocess_file",
    "align_coarse",
    "align_fine",
    "fit_ice",
    "strain_melt_between_profiles",
]
