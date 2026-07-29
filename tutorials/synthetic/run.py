"""tutorials/synthetic/run.py — Observing System Simulation Experiment (OSSE).

Method-credibility anchor: set a KNOWN truth (densification law + conductivity +
surface-T history + accumulation history), generate synthetic observations from
the forward model with realistic noise, then invert with the SAME engine from a
neutral literature prior and check the MAP recovers the truth.

2026-07-20 REVISION — the OSSE now tests the configuration we actually run:

  TRUTH   : a smooth base history carrying a LARGE old-time (1000-1600)
            excursion (~3 K medieval swing, the sharpest test of the thermal
            null space), PLUS an exponential trend over the last ~200 years:
            warming dT*exp((t-2015)/tau) reaching +3 K at 2015 (tau = 60 yr,
            the observed-warming character) and an accumulation increase
            reaching +30%. Both trend amplitudes are env-tunable for sweeps
            (FIRN_OSSE_DT / FIRN_OSSE_DB / FIRN_OSSE_TAU); the defaults are
            deliberately large. The truth knots sample the exponential densely
            (10-yr post-1800), so the question "can the recovered histories
            track a smooth recent trend?" is real.
  SIGMAS  : the RECALIBRATED error-model magnitudes (borehole-T tens of mK
            depth-dependent, not the old flat 0.2 K; rho 4-13 kg/m3, not
            15+3%; dage ~8.5% of value, not a 5% floor). Generation noise ==
            assimilation sigma, so chi^2/N ~ 1 at truth by construction.
  BLOCKS  : NO absolute-age block (deleted at SP and Summit: layer-counted
            redundancy with dage). Default set is rho + dage + ONE borehole-T
            profile (7-m spacing; the old shallow-RTD second instrument is
            dropped, FIRN_OSSE_TSH=1 restores it) + a FirnCover-style
            compaction-rate block + an ApRES-like differenced-velocity block
            (FIRN_OSSE_COMP=0 / FIRN_OSSE_VEL=0 drop those two).
  LAYOUT  : the SP 17-knot T layout (5-yr spacing toward the present;
            knots_recent_T) and the SP 15-knot b layout — NOT truth's knots,
            so layout error is part of the experiment.
  DEEP    : FIRN_OSSE_DEEP=1 extends the domain into the deep ice (see the
            DEEP block below) so the old climate that advected OUT of the firn
            becomes a genuine reconstruction rather than initialization.

The archived flat-truth OSSE (synthetic_osse.json, old sigmas + age block +
truth-matched 6/8 knots) is superseded; it remains in output/ and git history.

Modes (env FIRN_MODE): "verify" (FD check), "optimize" (default).
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/synthetic/run.py
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from firnpack.inverse import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl, assimilate
from firnpack.models.firn import FirnParameters

HERE = Path(__file__).parent; OUT = str(HERE/"output")
# output/ is gitignored, so it is absent on a fresh checkout, and the truth dump
# below runs before assimilate() would have created it.
Path(OUT).mkdir(parents=True, exist_ok=True)
MODE = os.environ.get("FIRN_MODE", "optimize")
rng = np.random.default_rng(int(os.environ.get("FIRN_SEED", "0")))
_p = FirnParameters(); c_i = float(_p.c_i)
# DEEP mode (FIRN_OSSE_DEEP=1): extend the domain into the deep ice and
# assimilate independent DEEP borehole T + deep depth-age observations, so the
# old climate that advected OUT of the firn (and is preserved in the deep ice)
# becomes a genuine reconstruction rather than initialization. H0/spin/NZ and
# the knot/obs ranges all extend; the shallow (firn-only) OSSE is unchanged.
DEEP = os.environ.get("FIRN_OSSE_DEEP", "0") == "1"
H0 = 300.0 if DEEP else 130.0
SPIN_YR = 3500.0 if DEEP else 2500.0
NZ = 180 if DEEP else 100

# ---- KNOWN TRUTH ----
TRUTH = dict(hl_k0=11.5, hl_k1=540.0, hl_Ea1=10600.0, hl_Ea2=22400.0,
             k_snow_scale=1.15, k_firn_scale=1.0, Q_base=0.045, s2_shape=0.85)
# smooth base with a LARGE old-time (1000-1600) excursion — this is
# deliberate: a big pre-observable-window signal is the sharpest test of the
# thermal null space (if even a 3 K medieval swing is not recovered, the old
# record is unobservable, not merely under-signalled). 1000 cold, 1300 warm,
# 1600 cold; then the recent rise.
_T6Y = np.array([1000.,1300.,1600.,1850.,1950.,2015.])
_T6V = np.array([-53.5,-50.5,-52.5,-51.0,-50.2,-50.7])
_B8Y = np.array([1000.,1200.,1400.,1600.,1800.,1900.,1960.,2015.])
_B8V = np.array([0.076,0.100,0.074,0.098,0.104,0.085,0.104,0.082])
# ...plus LARGER exponential recent trends (env-tunable for sweeps)
DT_EXP = float(os.environ.get("FIRN_OSSE_DT", "3.0"))     # K at 2015
DB_EXP = float(os.environ.get("FIRN_OSSE_DB", "0.30"))    # fractional at 2015
TAU = float(os.environ.get("FIRN_OSSE_TAU", "60.0"))      # e-folding, years
# in DEEP mode the truth carries an OLD (pre-1000 CE) history too, so the deep
# ice has genuine structure to recover; a slow multi-century swing back to
# ~1000 BCE (the deep-ice deposition era of a 300 m column).
_TOLDY = np.array([-1000., -500., 0., 500.]); _TOLDV = np.array([-53.5, -54.5, -53.0, -53.0])
_BOLDY = _TOLDY.copy(); _BOLDV = np.array([0.070, 0.078, 0.070, 0.078])
if DEEP:
    _T6Y = np.concatenate([_TOLDY, _T6Y]); _T6V = np.concatenate([_TOLDV, _T6V])
    _B8Y = np.concatenate([_BOLDY, _B8Y]); _B8V = np.concatenate([_BOLDV, _B8V])
    _recent = np.arange(1800., 2016., 10.0)
    T_YEARS = np.concatenate([[-1000., -500., 0., 500., 1000., 1300., 1600., 1750.], _recent])
else:
    T_YEARS = np.concatenate([[1000., 1300., 1600., 1750.], np.arange(1800., 2016., 10.0)])
B_YEARS = T_YEARS.copy()
_ramp = np.exp((T_YEARS - 2015.0) / TAU)
T_TRUTH = np.interp(T_YEARS, _T6Y, _T6V) + DT_EXP * _ramp
B_TRUTH = np.interp(B_YEARS, _B8Y, _B8V) * (1.0 + DB_EXP * _ramp)
print(f"{'DEEP ' if DEEP else ''}truth: +{DT_EXP} K / +{100*DB_EXP:.0f}% b recent trend "
      f"(tau={TAU:.0f}); knots {T_YEARS.min():.0f}..{T_YEARS.max():.0f} CE; "
      f"H_col={H0:.0f} m spin={SPIN_YR:.0f} yr")

def make_cfg(scalars, T_knots, b_knots, obs, tag, maxit=80):
    return SiteConfig(name="Synthetic", out_dir=OUT, tag=tag,
        H_col=H0, NZ=NZ, spin_years=SPIN_YR, dt_years=5.0, rho_surf=350.0,
        rho_ic_deep=820.0, rho_ic_scale=25.0, conductivity_law="calonne2019",
        scalars=scalars, T_knots=T_knots, b_knots=b_knots, obs=obs,
        max_iter=maxit)

# ---- dynamic vertical strain (ezz) in the truth (FIRN_OSSE_EZZ, /yr) --------
# 0 = pure firn (the archived trend OSSE); nonzero exercises the ezz
# decomposition — the truth velocity then carries dynamic thinning that the
# density/dage blocks are BLIND to, so recovering it REQUIRES the velocity
# block paired with a below-firn ezz constraint (the South Pole mechanism).
EZZ_TRUTH = float(os.environ.get("FIRN_OSSE_EZZ", "0.0"))

# save the truth knots NOW (not just at the end) so forward-only diagnostics
# (knot_observability, reanalysis) can read them while a long inversion runs
json.dump(dict(truth=TRUTH, T_truth=T_TRUTH.tolist(), b_truth=B_TRUTH.tolist(),
               T_years=T_YEARS.tolist(), b_years=B_YEARS.tolist(),
               dT_exp=DT_EXP, db_exp=DB_EXP, tau=TAU, ezz_truth=EZZ_TRUTH),
          open(Path(OUT)/"synthetic_truth.json", "w"), indent=1)

# ---- 1. truth forward -> synthetic profiles ----
truth_scalars = [ScalarCtrl(k, TRUTH[k], TRUTH[k], 1.0, 1e-6, 1e9,
                            log=(k not in ("Q_base",)), active=False) for k in TRUTH]
if EZZ_TRUTH:
    truth_scalars.append(ScalarCtrl("ezz_yr", EZZ_TRUTH, EZZ_TRUTH, 1.0,
                                    -3.0e-4, 2.0e-4, active=False))
    print(f"truth ezz (dynamic strain): {EZZ_TRUTH:.2e} /yr")
truth_T = KnotCtrl(T_YEARS, T_TRUTH, T_TRUTH, 0.6, -60, -40, log=False, invert=False, name="Tk")
truth_b = KnotCtrl(B_YEARS, B_TRUTH, B_TRUTH, 0.2, 0.03, 0.20, log=True, invert=False, name="b")
dummy = [ObsBlock("rho", np.array([50.0]), np.array([600.0]), np.array([1e3]), label="rho")]
truth_cfg = make_cfg(truth_scalars, truth_T, truth_b, dummy, "synthetic_truth_fwd")
tr = assimilate(truth_cfg, mode="forward", verbose=False)
prof = tr["profiles"]; d = np.array(prof["depth"]); rho_t = np.array(prof["rho"])
age_t = np.array(prof["age_yr"]); T_t = np.array(prof["T_C"]); w_t = np.array(prof["w_m_yr"])
o = np.argsort(d); d, rho_t, age_t, T_t, w_t = d[o], rho_t[o], age_t[o], T_t[o], w_t[o]
print(f"truth forward: rho(10)={np.interp(10,d,rho_t):.0f} age(100)={np.interp(100,d,age_t):.0f} "
      f"T(bot)={T_t[-1]:.2f}")

# ---- 2. synthetic observations at the RECALIBRATED sigma levels ----
# rho: SP-like depth-dependent (12-14 kg/m3 in the top 10 m -> 4-7 below)
rho_d = np.linspace(2, 128, 45); rho_o = np.interp(rho_d, d, rho_t)
rho_sig = 4.5 + 9.0 * np.exp(-rho_d / 8.0)
rho_obs = rho_o + rng.normal(0, rho_sig)
# dage: ~8.5% of value (the SP data-derived median), floor 0.15 yr/m
dc, do_, dsg = [], [], []
for cc in np.arange(6.0, H0 - 1.0, 2.0):
    sl = np.interp(cc + 0.5, d, age_t) - np.interp(cc - 0.5, d, age_t)
    dc.append(cc); do_.append(sl); dsg.append(max(0.085 * abs(sl), 0.15))
dc = np.array(dc); do_ = np.array(do_); dsg = np.array(dsg)
dage_obs = do_ + rng.normal(0, dsg)
# borehole T: ONE consistent block — a single instrument's profile through the
# firn (13-125 m, 7-m spacing) at SP's kernel-consistent sigma (depth-dependent:
# SP's realized span is 25-85 mK about a 67 mK median; the profile sampled below
# spans 27-68 mK), extended in DEEP mode straight down the same log into the
# ice (140 m -> H0, 15-m spacing, ~30 mK) where the OLD surface T advected out
# of the firn still lives. That extension is more of the same log, not a second
# instrument. This mirrors the REAL South Pole assimilation, where the only
# temperature PROFILE is the deep SPICEcore borehole log; the near-surface
# signal at SP comes from a separate seasonal-AMPLITUDE observable, NOT a
# second T profile. The earlier split into a deep-borehole block + a
# shallow-RTD block (different instruments, different noise, 0.2 K vs tens of mK)
# was an OSSE simplification and is dropped; FIRN_OSSE_TSH=1 restores the
# shallow-RTD block for the old two-instrument test.
Td_d = np.arange(13.0, 129.0, 7.0)
Td_sig = np.interp(Td_d, [13., 30., 45., 60., 80., 100., 115., 128.],
                   [0.027, 0.027, 0.050, 0.068, 0.060, 0.045, 0.035, 0.055])
if DEEP:
    Tdeep_d = np.arange(140.0, H0 - 1.0, 15.0)
    Td_d = np.concatenate([Td_d, Tdeep_d])
    Td_sig = np.concatenate([Td_sig, np.full(len(Tdeep_d), 0.030)])  # deep log ~30 mK
Td_o = np.interp(Td_d, d, T_t)
Td_obs = Td_o + rng.normal(0, Td_sig)
T_REF = float(_p.T_ref)
obs = [
    ObsBlock("rho", rho_d, rho_obs, rho_sig, label="rho"),
    ObsBlock("dagedz", dc, dage_obs, dsg, label="dage"),
    ObsBlock("enthalpy", Td_d, c_i*(Td_obs+273.15-T_REF), c_i*Td_sig, label="T"),
]
if os.environ.get("FIRN_OSSE_TSH", "0") == "1":
    Ts_d = np.array([2., 4., 6., 8.]); Ts_o = np.interp(Ts_d, d, T_t)
    Ts_obs = Ts_o + rng.normal(0, 0.2, len(Ts_d))
    obs.append(ObsBlock("enthalpy", Ts_d, c_i*(Ts_obs+273.15-T_REF),
                        c_i*np.full(len(Ts_d), 0.2), label="Tsh"))
    print("shallow-RTD (Tsh) block RESTORED (two-instrument test)")
# NO absolute-age block: deleted at SP and Summit (layer-counted redundancy).

# ---- synthetic firn-compaction-rate block (FirnCover coils) -----------------
# A direct densification-RATE observable: material intervals [ztop, zbot] whose
# shortening rate = (w@ztop - w@zbot) is measured. Mirrors the Summit FirnCover
# coils — physical wire coils on material intervals, NOT ApRES (the ApRES
# analog here is the differenced-velocity block below). ~8-12% sigma. Tests
# whether the RATE observable improves recovery of the law/accumulation.
# FIRN_OSSE_COMP=0 drops it (before/after comparison against synthetic_osse_trend).
if os.environ.get("FIRN_OSSE_COMP", "1") == "1":
    comp_zbot = np.array([5., 8., 12., 16., 22., 30.])
    comp_ztop = np.full(len(comp_zbot), 1.0)
    comp_o = np.array([np.interp(zt, d, w_t) - np.interp(zb, d, w_t)
                       for zt, zb in zip(comp_ztop, comp_zbot)])   # m/yr, negative
    comp_sig = np.maximum(0.08*np.abs(comp_o), 0.015)
    comp_obs = comp_o + rng.normal(0, comp_sig)
    obs.append(ObsBlock("compaction", comp_zbot, comp_obs, comp_sig,
                        ztop=comp_ztop, label="comp"))
    print(f"compaction block: {len(comp_zbot)} synthetic coils, "
          f"rates {comp_o.min():.3f}..{comp_o.max():.3f} m/yr, "
          f"sigma {1000*np.median(comp_sig):.0f} mm/yr median")

# ---- synthetic ApRES-like differenced VELOCITY profile ----------------------
# w(z_ref) - w(z) over depth (the dRdt_diff operator, same as South Pole),
# emulating an ApRES range-rate profile. DIFFERENCED, so (like compaction) it
# is blind to the absolute accumulation baseline w_surf -- it constrains the
# deep velocity SHAPE (stage-2 densification + accumulation structure).
# Drawn LAST so the other blocks' noise realizations match the comp/no-comp
# runs byte-for-byte. FIRN_OSSE_VEL=0 drops it.
if os.environ.get("FIRN_OSSE_VEL", "1") == "1":
    v_d = np.array([16., 24., 40., 48., 56., 64., 72., 80., 88., 96., 104., 112.])
    v_ref = 30.0; v_sig = 0.012
    w_r = float(np.interp(v_ref, d, w_t))
    v_o = w_r - np.interp(v_d, d, w_t)          # m/yr
    v_obs = v_o + rng.normal(0, v_sig, len(v_d))
    obs.append(ObsBlock("dRdt_diff", v_d, v_obs, np.full(len(v_d), v_sig),
                        label="v", ref_depth=v_ref))
    print(f"velocity block: {len(v_d)} ApRES-like differenced points "
          f"(ref {v_ref:.0f} m), sigma {1000*v_sig:.0f} mm/yr")

# ---- 3. inversion: neutral prior, SP knot layouts (NOT truth's knots) ----
_kj = json.load(open(HERE.parent/"southpole"/"knots_recent_T.json"))
INV_TY = np.array(_kj["knot_years"], float)     # 17 knots, 5-yr recent spacing
INV_BY = np.array(_kj["b_knot_years"], float)   # 15 knots
if DEEP:
    # extend the inversion layouts back into the deep-ice era so the old
    # climate (now observable via the deep borehole) is a control to recover
    _OLD = np.array([-1000., -500., 0., 500.])
    INV_TY = np.concatenate([_OLD, INV_TY])
    INV_BY = np.concatenate([_OLD, INV_BY])
LIT = dict(hl_k0=10.79, hl_k1=570.7, hl_Ea1=10432.0, hl_Ea2=21875.0)
inv_scalars = [
    ScalarCtrl("hl_k0", LIT["hl_k0"], LIT["hl_k0"], 1.0, 0.5, 500, log=True),
    ScalarCtrl("hl_k1", LIT["hl_k1"], LIT["hl_k1"], 1.0, 10, 50000, log=True),
    ScalarCtrl("hl_Ea1", LIT["hl_Ea1"], LIT["hl_Ea1"], 0.5, 3000, 40000, log=True),
    ScalarCtrl("hl_Ea2", LIT["hl_Ea2"], LIT["hl_Ea2"], 0.5, 5000, 80000, log=True),
    ScalarCtrl("k_snow_scale", 1.0, 1.0, 0.5, 0.1, 5.0, log=True),
    ScalarCtrl("k_firn_scale", 1.0, 1.0, 0.15, 0.5, 2.0, log=True),
    ScalarCtrl("Q_base", 0.0, 0.0, 0.05, -0.2, 0.2),
    ScalarCtrl("s2_shape", 1.0, 1.0, 0.3, 0.3, 2.0, log=True),
]
# ezz DECOMPOSITION (framework tightening for the vertical strain rate):
# free a dynamic-strain control PINNED by a below-firn measurement (tight
# prior = the deep dR/dt gives ezz to ~1e-5, unconfounded by compaction).
# With ezz pinned, the velocity block cleanly separates densification from
# dynamic thinning; WITHOUT it, unmodelled ezz biases the densification law.
# FIRN_OSSE_EZZPIN=1 turns it on (center = the below-firn "measurement").
if os.environ.get("FIRN_OSSE_EZZPIN", "0") == "1":
    _ezz_meas = EZZ_TRUTH + rng.normal(0, 1.0e-5)   # noisy below-firn measurement
    inv_scalars.append(ScalarCtrl("ezz_yr", _ezz_meas, _ezz_meas, 1.0e-5, -3.0e-4, 2.0e-4))
    print(f"ezz control PINNED at below-firn measurement {_ezz_meas:.2e} +- 1e-5 /yr")
T_ctr = np.full(len(INV_TY), -51.5)             # flat neutral prior
# deep-time T-knot prior (framework tightening for the T absolute level):
# the pre-1600 knots are unobservable and, at the loose 0.6 K prior, absorb the
# absolute-level null space (the ~0.5 K cold bias). Tightening them to
# FIRN_OSSE_TPRIOR_DEEP (default 0.6 = unchanged) pins the surface-T datum
# without touching the data-derived observation sigma.
_TSIG_DEEP = float(os.environ.get("FIRN_OSSE_TPRIOR_DEEP", "0.6"))
# In DEEP mode the pre-1600 knots are OBSERVABLE via the deep borehole, so they
# must NOT be pinned — only the very oldest (near/below the domain base, ~pre-
# -500 CE) are still initialization and keep a tight prior.
_pin_before = -500.0 if DEEP else 1600.0
T_sig = np.where(INV_TY <= _pin_before, _TSIG_DEEP, 0.6)
inv_T = KnotCtrl(INV_TY, T_ctr, T_ctr, T_sig, -60, -44, log=False, invert=True, name="Tk")
b_ctr = np.full(len(INV_BY), 0.09)
inv_b = KnotCtrl(INV_BY, b_ctr, b_ctr, 0.20, 0.03, 0.20, log=True, invert=True, name="b")
if _TSIG_DEEP != 0.6:
    print(f"deep-time (<={_pin_before:.0f}) T-knot prior TIGHTENED to {_TSIG_DEEP} K "
          f"({int((INV_TY<=_pin_before).sum())} knots)")
cfg = make_cfg(inv_scalars, inv_T, inv_b, obs, os.environ.get("FIRN_TAG", "synthetic_osse_trend"),
               maxit=int(os.environ.get("FIRN_MAX_ITER", "100")))

if MODE == "verify":
    assimilate(cfg, mode="verify", fd_names=["hl_k0","hl_Ea2","s2_shape","k_snow_scale","b1600","Tk8"])
else:
    r = assimilate(cfg, mode="optimize")
    print("\n=== TRUTH RECOVERY ===")
    for k in TRUTH:
        if k in r["m_map"]:
            print(f"  {k:12s} truth={TRUTH[k]:.4g}  recovered={r['m_map'][k]:.4g}  "
                  f"ratio={r['m_map'][k]/TRUTH[k]:.3f}")
    if EZZ_TRUTH and "ezz_yr" in r["m_map"]:
        print(f"  {'ezz_yr':12s} truth={EZZ_TRUTH:.3e}  recovered={r['m_map']['ezz_yr']:.3e}  "
              f"ratio={r['m_map']['ezz_yr']/EZZ_TRUTH:.3f}")
    json.dump(dict(truth=TRUTH, T_truth=T_TRUTH.tolist(), b_truth=B_TRUTH.tolist(),
                   T_years=T_YEARS.tolist(), b_years=B_YEARS.tolist(),
                   dT_exp=DT_EXP, db_exp=DB_EXP, tau=TAU, ezz_truth=EZZ_TRUTH),
              open(Path(OUT)/"synthetic_truth.json", "w"), indent=1)
    print("  (truth saved for the recovery figure)")
