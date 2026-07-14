"""osse_common.py — shared truth + synthetic-observation generation for the OSSE.

Used by observability.py (observation-subset ablation). Defines ONE truth and
one noise realization (fixed seed) so subset runs differ ONLY in which
observation blocks are included. run.py (the headline OSSE) currently carries
its own copy of this logic at spinup 2500; unify in the cleanup pass.
"""
from __future__ import annotations
import math
import numpy as np
from firn.inverse import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl, assimilate
from firn.models.firn import FirnParameters
from firn.constants import year as YEAR_S

_p = FirnParameters(); C_I = float(_p.c_i); T_REF = float(_p.T_ref)
H0 = 130.0

TRUTH = dict(hl_k0=11.5, hl_k1=540.0, hl_Ea1=10600.0, hl_Ea2=22400.0,
             k_snow_scale=1.15, k_firn_scale=1.0, Q_base=0.045, s2_shape=0.85)
T_YEARS = np.array([1000., 1300., 1600., 1850., 1950., 2015.])
T_TRUTH = np.array([-52.0, -51.5, -51.8, -51.0, -50.4, -50.7])
B_YEARS = np.array([1000., 1200., 1400., 1600., 1800., 1900., 1960., 2015.])
B_TRUTH = np.array([0.088, 0.090, 0.086, 0.091, 0.094, 0.089, 0.097, 0.085])
LIT = dict(hl_k0=10.79, hl_k1=570.7, hl_Ea1=10432.0, hl_Ea2=21875.0)


def truth_profiles(out_dir, spin=1600.0, dt=5.0, NZ=100,
                   T_years=None, T_vals=None, B_years=None, B_vals=None,
                   ezz_truth=0.0, return_w=False):
    """Run the truth forward once; return depth-sorted (d, rho, age_yr, T_C).
    Custom knot arrays override the default smooth truth (temporal-resolution
    experiment uses a decadally-structured truth). ezz_truth adds uniform
    dynamic strain to the truth column (velocity OSSE); default 0 preserves
    the archived observability truth exactly. return_w appends w (m/yr)."""
    Ty = T_YEARS if T_years is None else np.asarray(T_years, float)
    Tv = T_TRUTH if T_vals is None else np.asarray(T_vals, float)
    By = B_YEARS if B_years is None else np.asarray(B_years, float)
    Bv = B_TRUTH if B_vals is None else np.asarray(B_vals, float)
    scal = [ScalarCtrl(k, TRUTH[k], TRUTH[k], 1.0, 1e-6, 1e9,
                       log=(k != "Q_base"), active=False) for k in TRUTH]
    if ezz_truth:
        scal.append(ScalarCtrl("ezz_yr", ezz_truth, ezz_truth, 1.0,
                               -3.0e-4, 2.0e-4, active=False))
    tk = KnotCtrl(Ty, Tv, Tv, 0.6, -60, -40, invert=False, name="Tk")
    bk = KnotCtrl(By, Bv, Bv, 0.2, 0.03, 0.20, log=True, invert=False, name="b")
    dummy = [ObsBlock("rho", np.array([50.0]), np.array([600.0]), np.array([1e3]), label="rho")]
    cfg = SiteConfig(name="SynTruth", out_dir=out_dir, tag="obsv_truth",
                     H_col=H0, NZ=NZ, spin_years=spin, dt_years=dt, rho_surf=350.0,
                     rho_ic_deep=820.0, rho_ic_scale=25.0,
                     conductivity_law="calonne2019", scalars=scal,
                     T_knots=tk, b_knots=bk, obs=dummy)
    r = assimilate(cfg, mode="forward", verbose=False)
    p = r["profiles"]
    d = np.array(p["depth"]); o = np.argsort(d)
    out = (d[o], np.array(p["rho"])[o], np.array(p["age_yr"])[o], np.array(p["T_C"])[o])
    return out + (np.array(p["w_m_yr"])[o],) if return_w else out


def dense_truth(seed=7):
    """Truth WITH decadal structure (temporal-resolution experiment):
    the smooth truth plus seeded AR(1) decadal anomalies — b ±~5% (log),
    T ±~0.3 °C (post-1800 only; deep time stays smooth/unobservable)."""
    rng = np.random.default_rng(seed)
    By = np.arange(1000.0, 2016.0, 10.0)
    base_b = np.interp(By, B_YEARS, np.log(B_TRUTH))
    ar = np.zeros(len(By)); rho_ar = 0.6
    for i in range(1, len(By)):
        ar[i] = rho_ar*ar[i-1] + rng.normal(0, 0.05*math.sqrt(1-rho_ar**2))
    Bv = np.exp(base_b + ar)
    Ty = np.concatenate([np.array([1000., 1300., 1600.]), np.arange(1800.0, 2016.0, 10.0)])
    base_T = np.interp(Ty, T_YEARS, T_TRUTH)
    arT = np.zeros(len(Ty))
    for i in range(1, len(Ty)):
        arT[i] = rho_ar*arT[i-1] + rng.normal(0, 0.3*math.sqrt(1-rho_ar**2))
    arT[Ty < 1800] = 0.0
    Tv = base_T + arT
    return Ty, Tv, By, Bv


def make_obs(d, rho_t, age_t, T_t, which, seed=0, w_t=None, v_sig=0.012):
    """Sample noisy synthetic observation blocks; `which` selects from
    {rho, age, dage, Tdeep, Tsh, v}. Same seed -> same noise across subsets.
    "v" (needs w_t, m/yr on grid d) is an ApRES-like DIFFERENCED vertical
    velocity profile w(z_ref)-w(z), the same dRdt_diff operator as South Pole;
    w_t from a DIFFERENT column than (rho_t,...) emulates non-co-located data."""
    rng = np.random.default_rng(seed)
    # draw ALL noise in a fixed order regardless of subset, so shared blocks
    # carry identical realizations across subsets
    rho_d = np.linspace(2, 128, 45); rho_o = np.interp(rho_d, d, rho_t)
    rho_sig = 15.0 + 0.03 * rho_o; n_rho = rng.normal(0, rho_sig)
    age_d = np.linspace(4, 128, 40); age_o = np.interp(age_d, d, age_t)
    age_sig = 10.0 + 0.03 * age_o; n_age = rng.normal(0, age_sig)
    dc, do_, dsg = [], [], []
    for cc in np.arange(6.0, H0 - 1.0, 2.0):
        sl = np.interp(cc + 0.5, d, age_t) - np.interp(cc - 0.5, d, age_t)
        dc.append(cc); do_.append(sl); dsg.append(max(0.05 * abs(sl), 0.15))
    dc = np.array(dc); do_ = np.array(do_); dsg = np.array(dsg)
    n_dage = rng.normal(0, dsg)
    Td_d = np.linspace(10, 128, 40); Td_o = np.interp(Td_d, d, T_t)
    n_Td = rng.normal(0, 0.2, len(Td_d))
    Ts_d = np.array([2., 4., 6., 8.]); Ts_o = np.interp(Ts_d, d, T_t)
    n_Ts = rng.normal(0, 0.2, len(Ts_d))
    blocks = {
        "rho": ObsBlock("rho", rho_d, rho_o + n_rho, rho_sig, label="rho"),
        "age": ObsBlock("age", age_d, (age_o + n_age) * YEAR_S,
                        (10.0 + 0.03 * age_o) * YEAR_S, label="age"),
        "dage": ObsBlock("dagedz", dc, do_ + n_dage, dsg, label="dage"),
        "Tdeep": ObsBlock("enthalpy", Td_d, C_I * (Td_o + n_Td + 273.15 - T_REF),
                          C_I * np.full(len(Td_d), 0.2), label="Td"),
        "Tsh": ObsBlock("enthalpy", Ts_d, C_I * (Ts_o + n_Ts + 273.15 - T_REF),
                        C_I * np.full(len(Ts_d), 0.2), label="Tsh"),
    }
    # velocity drawn LAST so adding it never perturbs the shared core-block
    # noise realizations of the archived subsets
    if w_t is not None:
        v_d = np.array([16., 24., 40., 48., 56., 64., 72., 80., 88., 96., 104., 112.])
        v_ref = 30.0
        w_i = np.interp(v_d, d, w_t); w_r = float(np.interp(v_ref, d, w_t))
        v_o = (w_r - w_i) + rng.normal(0, v_sig, len(v_d))
        blocks["v"] = ObsBlock("dRdt_diff", v_d, v_o, np.full(len(v_d), v_sig),
                               label="v", ref_depth=v_ref)
    return [blocks[k] for k in which]


def inversion_cfg(obs, out_dir, tag, spin=1600.0, dt=5.0, NZ=100, maxit=40,
                  T_years=None, B_years=None, with_ezz=False):
    """Neutral literature-prior inversion config (cold start). Custom knot
    layouts (temporal-resolution experiment) via T_years/B_years. with_ezz
    frees uniform dynamic strain (SP prior: center 0, sigma 1e-4) — velocity
    OSSE; default False preserves the archived observability configs."""
    Ty = T_YEARS if T_years is None else np.asarray(T_years, float)
    By = B_YEARS if B_years is None else np.asarray(B_years, float)
    scalars = [
        ScalarCtrl("hl_k0", LIT["hl_k0"], LIT["hl_k0"], 1.0, 0.5, 500, log=True),
        ScalarCtrl("hl_k1", LIT["hl_k1"], LIT["hl_k1"], 1.0, 10, 50000, log=True),
        ScalarCtrl("hl_Ea1", LIT["hl_Ea1"], LIT["hl_Ea1"], 0.5, 3000, 40000, log=True),
        ScalarCtrl("hl_Ea2", LIT["hl_Ea2"], LIT["hl_Ea2"], 0.5, 5000, 80000, log=True),
        ScalarCtrl("k_snow_scale", 1.0, 1.0, 0.5, 0.1, 5.0, log=True),
        ScalarCtrl("k_firn_scale", 1.0, 1.0, 0.15, 0.5, 2.0, log=True),
        ScalarCtrl("Q_base", 0.0, 0.0, 0.05, -0.2, 0.2),
        ScalarCtrl("s2_shape", 1.0, 1.0, 0.3, 0.3, 2.0, log=True),
    ]
    if with_ezz:
        scalars.append(ScalarCtrl("ezz_yr", 0.0, 0.0, 1.0e-4, -3.0e-4, 2.0e-4))
    T_ctr = np.full(len(Ty), -51.5)
    b_ctr = np.full(len(By), 0.09)
    return SiteConfig(name=f"Syn[{tag}]", out_dir=out_dir, tag=tag,
        H_col=H0, NZ=NZ, spin_years=spin, dt_years=dt, rho_surf=350.0,
        rho_ic_deep=820.0, rho_ic_scale=25.0, conductivity_law="calonne2019",
        scalars=scalars,
        T_knots=KnotCtrl(Ty, T_ctr, T_ctr, 0.6, -60, -44, invert=True, name="Tk"),
        b_knots=KnotCtrl(By, b_ctr, b_ctr, 0.20, 0.03, 0.20, log=True, invert=True, name="b"),
        obs=obs, max_iter=maxit)
