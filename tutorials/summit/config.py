"""tutorials/summit/config.py — build the Summit SiteConfig.

Same pattern as tutorials/southpole/config.py: run.py and every script under
diagnostics/ construct the SAME config by calling build_cfg(), which reads
os.environ at call time. Returns a SimpleNamespace of every builder local;
.cfg is the SiteConfig.

Error model (2026-07-19, the SP recalibration program applied at Summit —
sigma = what the model cannot fit at ANY control setting, measured from our
own series):
  rho  : local rms about a smooth curve on the BINNED composite (residuals
         white at 2-m spacing). Replaces 15+3% (5-8x too big).
  dage : sigma^2 = se^2 + repr^2, repr about the b-knot-resolvable scale
         (16-42 m of core, varies with local knot spacing). Replaces the
         hand-picked 5% floor (~1.7x too small).
  age  : DELETED — GISP2 is layer-counted, so absolute age is a cumulative
         count (errors accumulate, not independent), and with age=0 pinned at
         the surface dage determines age everywhere. 61% of its points sat
         inside dage windows.
Legacy guards restore the archived summit_invert.json error model exactly
(verified J=81.9829): FIRN_SIG_RHO_LEGACY=1 FIRN_SIG_DAGE_LEGACY=1
FIRN_AGE_BLOCK=1.
"""
from __future__ import annotations
import math, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
from firnpack.inverse import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl
from firnpack.models.firn import FirnParameters
from firnpack.constants import year as YEAR_S


def build_cfg():
    HERE = Path(__file__).parent
    DATA = HERE / "data"
    OUT = str(HERE / "output")
    MODE = os.environ.get("FIRN_MODE", "optimize")
    H0 = 150.0
    _p = FirnParameters(); c_i = float(_p.c_i); T_ref = float(_p.T_ref)
    T_FORCE = -28.8   # FirnCover Summit deep-firn mean

    # ---- observations ----
    dens = pd.read_csv(DATA/"summit_density.csv")
    rho_col = [c for c in dens.columns if "rho" in c.lower()][0]
    dro = dens[rho_col].values; dro = dro*1000.0 if np.nanmax(dro) < 5 else dro
    # bin the composite density to ~2 m (mid-firn is clustered; smooths pycnometry scatter)
    edges = np.arange(0, 92, 2.0); dv, dn = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (dens.depth_m.values >= lo) & (dens.depth_m.values < hi)
        if s.sum() >= 1: dv.append(np.mean(dro[s])); dn.append(0.5*(lo+hi))
    rho_d = np.array(dn); rho_v = np.array(dv)
    if os.environ.get("FIRN_SIG_RHO_LEGACY", "0") == "1":
        rho_sig = 15.0 + 0.03*rho_v
        print("rho sigma LEGACY (15 kg/m3 + 3%)")
    else:
        # local rms about a smooth curve on the BINNED series (the assimilated
        # observable; residuals white at 2-m spacing, lag-1 -0.11). The local
        # window handles the composite's two instruments (SUMup core 0-22 m,
        # Fourteau pycnometry 25-90 m) without naming them. NOT the
        # first-difference estimator: the binned profile's real gradient
        # (~10 kg/m3 per bin) contaminates differences at Summit.
        _rres = rho_v - np.polyval(np.polyfit(rho_d, rho_v, 8), rho_d)
        _RLW = 11
        rho_sig = np.array([np.sqrt(np.mean(_rres[max(0,i-_RLW//2):min(len(_rres),i+_RLW//2+1)]**2))
                            for i in range(len(_rres))])
        print(f"rho sigma (data-derived): median {np.median(rho_sig):.1f} kg/m3, "
              f"spread {rho_sig.max()/rho_sig.min():.1f}x (binned lag-1 white; replaces 15+3%)")

    gis = pd.read_csv(DATA/"gisp2_depth_age.csv")
    gis = gis[(gis.depth_m <= H0) & (gis.year_CE <= 2015)].sort_values("depth_m")
    g_d = gis.depth_m.values; g_ageyr = 2015.0 - gis.year_CE.values
    sel = g_ageyr >= 0; g_d, g_ageyr = g_d[sel], g_ageyr[sel]
    def _sub(x, n): return x[::max(1, len(x)//n)]
    dc2, do2, ds2e = [], [], []
    for c in np.arange(6.0, H0-1.0+1e-9, 3.0):
        s = np.abs(g_d - c) <= 1.0
        if s.sum() >= 4:
            A = np.vstack([g_d[s]-c, np.ones(s.sum())]).T
            coef, res, *_ = np.linalg.lstsq(A, g_ageyr[s], rcond=None); nn = s.sum()
            se = math.sqrt(max(float(res[0]) if len(res) else 0.0, 1e-12)/(nn-2)/max(np.sum((g_d[s]-c)**2), 1e-12))
            dc2.append(c); do2.append(coef[0]); ds2e.append(se)
    dc2 = np.array(dc2); do2 = np.array(do2); ds2e = np.array(ds2e)

    B_YEARS = np.array([1000.,1150.,1300.,1450.,1600.,1750.,1850.,1920.,1970.,2015.])
    if os.environ.get("FIRN_SIG_DAGE_LEGACY", "0") == "1":
        ds2 = np.maximum(ds2e, 0.05*np.abs(do2))
        print("dage sigma LEGACY (max(se, 5%))")
    else:
        # sigma^2 = se^2 + repr^2; repr about the b-knot-resolvable scale,
        # which VARIES down the core (local knot spacing / local dage).
        _yr_at = 2015.0 - np.interp(dc2, g_d, g_ageyr)
        _dy = np.diff(B_YEARS)
        _ksp = np.array([_dy[min(max(np.searchsorted(B_YEARS, y)-1, 0), len(_dy)-1)] for y in _yr_at])
        _win = np.clip(_ksp/np.maximum(do2, 0.1), 6.0, 45.0)
        _sm = np.array([np.mean(do2[np.abs(dc2-c) <= _win[i]/2.0]) for i, c in enumerate(dc2)])
        _rr = do2 - _sm
        _DLW = 15
        _repr = np.array([_rr[max(0,i-_DLW//2):min(len(_rr),i+_DLW//2+1)].std()
                          for i in range(len(_rr))])
        ds2 = np.sqrt(ds2e**2 + _repr**2)
        print(f"dage sigma (data-derived): median {np.median(ds2):.3f} yr/m = "
              f"{100*np.median(ds2/np.abs(do2)):.1f}% of value "
              f"(knot window {_win.min():.0f}-{_win.max():.0f} m; replaces 5% floor)")

    # ---- observation EPOCHS (2026-07-19) ---------------------------------------
    # Summit's observations are NOT contemporaneous, and the offsets are large
    # enough to matter at 0.65 m/yr of snow:
    #   GISP2 layers  : counted on a core whose youngest layer is 1989 (the file
    #                   says so: 1.51 m = 1989.0). Scoring them against the 2015
    #                   state misplaces the layer field by ~26 yr of burial
    #                   (~15 m) — a bias the recent b-knots silently absorbed.
    #   GRIP density  : Fourteau pycnometry samples the GRIP core (drilled
    #                   1990-92) -> the firn STATE it measures is 1991's.
    #   FirnCover core: 2017 — beyond present (2015), so it keeps the
    #                   final-state path (the engine tags only pre-present).
    # The composite density block is therefore SPLIT at the instrument seam
    # (bins <= 23 m = FirnCover-only, >= 25 m = GRIP-only; the 22-25 m gap is
    # between instruments) so each part carries its own epoch. sigma stays
    # derived from the COMBINED binned profile (the local estimator already
    # handles the two instruments). Legacy sigma flags reproduce the archived
    # untagged, un-split blocks exactly.
    if os.environ.get("FIRN_SIG_RHO_LEGACY", "0") == "1":
        obs = [ObsBlock("rho", rho_d, rho_v, rho_sig, label="rho")]
    else:
        _sh = rho_d <= 23.5
        obs = [
            ObsBlock("rho", rho_d[_sh], rho_v[_sh], rho_sig[_sh], label="rho",
                     year=2017.0),
            ObsBlock("rho", rho_d[~_sh], rho_v[~_sh], rho_sig[~_sh], label="rho_g",
                     year=1991.0),
        ]
        print(f"rho split at the instrument seam: {int(_sh.sum())} FirnCover pts "
              f"(2017) + {int((~_sh).sum())} GRIP pts (1991, epoch-tagged)")
    _GISP2_YEAR = float(gis.year_CE.max())   # 1989: the core's youngest layer
    if os.environ.get("FIRN_SIG_DAGE_LEGACY", "0") == "1":
        obs.append(ObsBlock("dagedz", dc2, do2, ds2, label="dage"))
    else:
        obs.append(ObsBlock("dagedz", dc2, do2, ds2, label="dage", year=_GISP2_YEAR))
    # absolute-age block DELETED by default (see module docstring)
    if os.environ.get("FIRN_AGE_BLOCK", "0") == "1":
        age_d, age_yr = _sub(g_d, 40), _sub(g_ageyr, 40)
        obs.insert(1, ObsBlock("age", age_d, age_yr*YEAR_S, (10.0+0.03*age_yr)*YEAR_S, label="age"))
        print("age block RESTORED (legacy reproduction mode)")

    # ---- FirnCover compaction-rate block (2026-07-20) --------------------------
    # A DIRECT densification-RATE constraint, independent of the density core:
    # 6 borehole compaction coils (NOT ApRES — physical wire coils) measure the
    # shortening rate of a material firn interval [ztop, zbot]. Model pred =
    # (w@ztop - w@zbot)*yr, the densification strain integrated over the
    # interval (kind "compaction"). Records 2015-2019 straddle the model present
    # (2015) and center ~2017, so final-state path (no epoch tag), consistent
    # with the FirnCover density. sigma is DATA-DERIVED: interannual scatter
    # (year-over-year increments at matched day-of-year -> seasonal removed,
    # 14-19 mm/yr) with a cross-instrument representativeness floor (two coils
    # at ~15.7 m agree to ~16 mm/yr) -- NOT the daily-fit s.e., which is
    # meaningless (daily residuals seasonal, lag-1~1.0). Provenance:
    # data/README.md; the CSV read here is written by
    # diagnostics/derive_compaction_sigma.py.
    # FIRN_COMPACTION=0 drops the block (legacy reproduction).
    if os.environ.get("FIRN_COMPACTION", "1") == "1":
        cmp = pd.read_csv(DATA/"firncover_summit_compaction.csv")
        obs.append(ObsBlock("compaction", cmp.zbot_mean_m.values, cmp.rate_m_yr.values,
                            cmp.sigma_m_yr.values, ztop=cmp.ztop_mean_m.values, label="comp"))
        print(f"compaction block: {len(cmp)} FirnCover coils, intervals "
              f"{cmp.ztop_mean_m.min():.1f}-{cmp.zbot_mean_m.max():.1f} m, "
              f"rates {cmp.rate_m_yr.min():.3f}..{cmp.rate_m_yr.max():.3f} m/yr, "
              f"sigma {1000*cmp.sigma_m_yr.median():.0f} mm/yr median")

    # ---- controls: densification (5, neutral literature prior — SAME as SP) ----
    scalars = [
        ScalarCtrl("hl_k0", 10.79, 10.79, 1.0, 0.5, 500, log=True),
        ScalarCtrl("hl_k1", 570.7, 570.7, 1.0, 10, 50000, log=True),
        ScalarCtrl("hl_Ea1", 10432., 10432., 0.5, 3000, 40000, log=True),
        ScalarCtrl("hl_Ea2", 21875., 21875., 0.5, 5000, 80000, log=True),
        ScalarCtrl("s2_shape", 1.0, 1.0, 0.3, 0.3, 2.0, log=True),
        # fixed (Summit lacks the data to constrain these):
        ScalarCtrl("k_snow_scale", 1.0, 1.0, 1.0, 0.1, 5.0, log=True, active=False),
        ScalarCtrl("k_firn_scale", 1.0, 1.0, 0.15, 0.5, 2.0, log=True, active=False),
        ScalarCtrl("Q_base", 0.0, 0.0, 0.05, -0.2, 0.2, active=False),
    ]

    # ---- accumulation b-knots (invert; Osman ~0.246 flat prior; coarse) ----
    osman = pd.read_csv(DATA/"summit_layer_accum_annual.csv")
    B_CTR = np.full(len(B_YEARS), float(osman.accum_m_iceeq_yr.mean()))   # 0.246 flat
    # T forcing: constant, prescribed (2 knots, not inverted)
    T_YEARS = np.array([1000., 2015.]); T_CTR = np.array([T_FORCE, T_FORCE])

    cfg = SiteConfig(name="Summit", out_dir=OUT, tag=os.environ.get("FIRN_TAG", "summit_invert"),
        H_col=H0, NZ=int(os.environ.get("FIRN_NZ", "120")), stretch_p=2.5,
        spin_years=1500.0, dt_years=float(os.environ.get("FIRN_DT", "5.0")),
        rho_surf=350.0, rho_ic_deep=900.0, rho_ic_scale=15.0,
        conductivity_law="calonne2019", scalars=scalars,
        T_knots=KnotCtrl(T_YEARS, T_CTR, T_CTR, 0.6, -40.0, -20.0, log=False, invert=False, name="Tk"),
        b_knots=KnotCtrl(B_YEARS, B_CTR, B_CTR, 0.25, 0.05, 0.60, log=True, invert=True, name="b"),
        b_off_era_year=None, obs=obs, max_iter=int(os.environ.get("FIRN_MAX_ITER","80")))

    return SimpleNamespace(**{k: v for k, v in locals().items()
                              if not k.startswith("__")})
