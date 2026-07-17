"""firnpack.inverse — adjoint data assimilation for firn columns.

The assimilation is modelled on icepack's statistics interface: a
``StatisticsProblem`` bundles the forward simulation, the model-data misfit
(loss), the prior (regularization) and the controls, and a
``MaximumProbabilityEstimator`` finds the MAP. The estimator's backend is scipy
L-BFGS-B on the Real-space controls (not icepack's ROL), which reproduces the
frozen South Pole MAP.

    from firnpack.inverse import StatisticsProblem, MaximumProbabilityEstimator

``SiteConfig`` + ``assimilate`` remain the declarative front end that builds and
solves the problem for the three case studies; they are being folded into the
StatisticsProblem interface across the tutorial refactor.
"""
from .statistics import StatisticsProblem, MaximumProbabilityEstimator
from .config import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl
from .engine import assimilate

__all__ = ["StatisticsProblem", "MaximumProbabilityEstimator",
           "SiteConfig", "ObsBlock", "ScalarCtrl", "KnotCtrl", "assimilate"]
