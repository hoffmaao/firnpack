"""firnpack.inverse.config — declarative configuration for a site assimilation.

A SiteConfig is a pure-data description of one inverse problem: the column,
the forcing knot layouts, which scalar controls are active, and a list of
observation blocks. The engine (engine.py) consumes it. Site scripts build it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ObsBlock:
    """One observation type, mapped to a model prediction operator by `kind`.

    kind:
      "rho"      -> kernel-average of density field         (obs kg/m^3)
      "age"      -> kernel-average of age field             (obs SECONDS)
      "enthalpy" -> kernel-average of enthalpy field        (obs J/kg; convert T)
      "dagedz"   -> kernel-average of -d(age)/dx / YEAR_S   (obs yr/m)
      "velocity" -> (kernel-avg w - w_surface)/n_ice*YEAR_S (obs m/yr, ApRES)
    """
    kind: str
    depths: np.ndarray
    obs: np.ndarray
    sig: np.ndarray
    weight: float = 1.0
    wmax: float = 3.0          # max Gaussian-kernel half-width (m)
    label: str = ""
    # kind "dRdt_diff" (offset-immune ApRES): obs are v(z_i) - v(ref_depth),
    # model pred = nfac_i * (<w phi_ref> - <w phi_i>) * year, with
    # nfac_i = mean refractive index of the pair / n_ice (data-side, from
    # the observed density profile). Immune to per-site antenna offsets.
    ref_depth: float | None = None
    nfac: np.ndarray | None = None
    # Correct firn-ApRES kinematics (Case & Kingslake 2022: W = c/n(z) dT/dtau,
    # reflectors = material surfaces): reported range rate = n(z) w(z) / n_ice
    # with the LOCAL index, so the differenced pred = nfac_ref*w_ref - nfac_j*w_j
    # with nfac = n(z_j)/n_ice, nfac_ref = n(z_ref)/n_ice. If nfac_ref is None
    # the old (superseded, path-mean) form nfac_j*(w_ref - w_j) is reproduced.
    nfac_ref: float | None = None
    # kind "seas_lnamp" (seasonal-amplitude damping, e.g. USP50 RTD string):
    # obs are ln A(z_i)/A(ref_depth); model pred = -int_ref^z_i sqrt(w rho c/2k) dz
    # (WKB damping integral; operator error quantified offline, folded into sig)
    # with k from the ON-TAPE conductivity law -> constrains k_snow/k_firn
    # INSIDE the inversion. aux_z/aux_val = DATA-side density profile of the
    # instrumented hole (the wave is damped in THAT column); model column if None.
    aux_z: np.ndarray | None = None
    aux_val: np.ndarray | None = None

    def __post_init__(self):
        self.depths = np.asarray(self.depths, float)
        self.obs = np.asarray(self.obs, float)
        self.sig = np.asarray(self.sig, float)
        if not self.label:
            self.label = self.kind

    @property
    def n(self):
        return len(self.depths)


@dataclass
class ScalarCtrl:
    """A scalar control (or fixed parameter if active=False)."""
    name: str
    init: float
    center: float             # prior center (physical units)
    sigma: float              # prior sigma (in internal coords: log if log=True)
    lo: float
    hi: float
    log: bool = False         # optimize in log space (positivity + scale)
    active: bool = True       # False -> held fixed at init, not a control


@dataclass
class KnotCtrl:
    """A time-varying forcing series on knots — either inverted or prescribed.

    values in physical units (T in C for temperature; b in m ice/yr for accum).
    If invert=True the knots are controls with the given prior; else they are
    fixed forcing (prescribed history).
    """
    years: np.ndarray
    init: np.ndarray          # starting/prescribed values (physical)
    center: np.ndarray        # prior centers (physical)
    sigma: float              # prior sigma (C for T; log-space frac for b)
    lo: float                 # bound (physical for T; physical for b)
    hi: float
    log: bool = False         # b-knots use log space
    invert: bool = True
    name: str = "knot"

    def __post_init__(self):
        self.years = np.asarray(self.years, float)
        self.init = np.asarray(self.init, float)
        self.center = np.asarray(self.center, float)


@dataclass
class SiteConfig:
    name: str
    out_dir: str
    tag: str
    # column / numerics
    H_col: float = 130.0
    NZ: int = 100
    stretch_p: float = 2.5
    surface_id: int = 2
    spin_years: float = 2500.0
    dt_years: float = 5.0
    rho_surf: float = 350.0
    rho_ic_deep: float = 820.0    # IC density asymptote (830-903 for warm sites)
    rho_ic_scale: float = 25.0
    present_year: float = 2015.0
    # physics
    conductivity_law: str = "calonne2019"
    deep_cutoff_rho: Optional[float] = None
    beta_enth: float = 5.0e2       # penalty enthalpy surface BC
    beta_w: float = 1.0e7          # penalty velocity surface BC
    n_ice: float = float(np.sqrt(3.18))
    # controls
    scalars: list = field(default_factory=list)     # list[ScalarCtrl]
    T_knots: Optional[KnotCtrl] = None
    b_knots: Optional[KnotCtrl] = None
    b_off_era_year: Optional[float] = None  # if set + b_off active: offset knots > this year
    # observations
    obs: list = field(default_factory=list)          # list[ObsBlock]
    # optimizer
    max_iter: int = 80
    ftol: float = 1e-8
    gtol: float = 1e-7
    maxls: int = 30
