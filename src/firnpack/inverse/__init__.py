"""firnpack.inverse — shared adjoint assimilation engine.

One config-driven engine behind all three paper case studies (synthetic OSSE,
South Pole, Summit). Refactored faithfully from the proven South Pole driver
(sp_joint_assimilate_r5.py); the South Pole config reproduces its frozen MAP.

Public API:
    from firnpack.inverse import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl, assimilate
"""
from .config import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl
from .engine import assimilate

__all__ = ["SiteConfig", "ObsBlock", "ScalarCtrl", "KnotCtrl", "assimilate"]
