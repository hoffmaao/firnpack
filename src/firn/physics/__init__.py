"""Pluggable physical process rate functions for the firn model.

Each rate function follows the icepack-style "action functional" pattern:
it takes a dict-like of fields/parameters and returns a UFL expression for
the rate. Users can supply their own to customize model behavior.
"""

from firn.physics.densification import (
    arthern_ligtenberg,
    herron_langway,
    kingslake,
    stokes_compressible,
)

__all__ = [
    "arthern_ligtenberg",
    "herron_langway",
    "kingslake",
    "stokes_compressible",
]
