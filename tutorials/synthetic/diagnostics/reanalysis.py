"""reanalysis.py - OSSE firn height-change attribution, RECOVERED vs TRUTH.

The synthetic analog of the South Pole reanalysis, with the OSSE's decisive
extra: the truth is known, so we run the scenario differencing at BOTH the
recovered MAP and the true forcing/law, and ask whether the inversion recovers
the right height-change drivers - not just the right profiles.

Scenarios (engine forward mode, untaped, ~15 s each):
  CTRL  : T and b held at their year-1000 value
  TONLY : the (recovered or true) T(t) history, b held at b(1000)
  FULL  : the (recovered or true) T(t) AND b(t)

Attribution in the 130 m surface-following column (differencing cancels V_deep
and CTRL drift exactly):
  dh'/dt   = |w_bot|_scenario - |w_bot|_CTRL   (= wbot_CTRL - wbot_scenario)
  T-driven = TONLY - CTRL,   b-driven = FULL - TONLY
  dFAC     = FAC_FULL - FAC_CTRL               (air part; wbot cross-check)
  mass     = cumsum(b_FULL - b_CTRL)           (ice-eq part, baseline-dependent)

Output: output/synthetic_reanalysis.{json,npz} (pure data; plot.py draws it).
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/synthetic/diagnostics/reanalysis.py
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
from firnpack.inverse import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl, assimilate

from firnpack import plot as fp

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "output"
# newest OSSE MAP by mtime (synthetic_osse_comp supersedes _trend); env override.
# Filtered on m_map so the _hessian/_marginals artifacts written under the same
# tag are never mistaken for a MAP.
_mapp = fp.newest_map(OUT, need=("m_map",), env="FIRN_OSSE_MAP",
                      pattern="synthetic_osse*.json")
if _mapp is None:
    raise SystemExit(f"no OSSE MAP in {OUT} - run tutorials/synthetic/run.py first")
print(f"reanalysis MAP: {_mapp.name}")
mp = json.load(open(_mapp))
tr = json.load(open(OUT / "synthetic_truth.json"))

# param sets: recovered MAP vs the known truth (same 8 scalar keys)
REC = mp["m_map"]
TRU = tr["truth"]
REC_TY, REC_TV = np.array(mp["knot_years"], float), np.array(mp["T_knots"], float)
REC_BY, REC_BV = np.array(mp["b_knot_years"], float), np.array(mp["b_knots"], float)
TRU_TY, TRU_TV = np.array(tr["T_years"], float), np.array(tr["T_truth"], float)
TRU_BY, TRU_BV = np.array(tr["b_years"], float), np.array(tr["b_truth"], float)

DUMMY = [ObsBlock("rho", np.array([50.0]), np.array([600.0]), np.array([1e3]), label="rho")]


def scenario(params, Ty, Tv, By, Bv, tag):
    # log-space only for genuinely positive scalars. Q_base is a signed flux, and
    # ezz_yr is a signed strain rate (negative = dynamic thinning), so keying the
    # flag off the name alone raised `math domain error` on any MAP carrying ezz.
    scal = [ScalarCtrl(k, params[k], params[k], 1.0, 1e-6, 1e9,
                       log=(k != "Q_base" and params[k] > 0.0), active=False)
            for k in params]
    cfg = SiteConfig(name="SynRA", out_dir=str(OUT), tag=tag, H_col=130.0, NZ=100,
                     spin_years=2500.0, dt_years=5.0, rho_surf=350.0,
                     rho_ic_deep=820.0, rho_ic_scale=25.0, conductivity_law="calonne2019",
                     scalars=scal,
                     T_knots=KnotCtrl(Ty, Tv, Tv, 0.6, -60, -40, invert=False, name="Tk"),
                     b_knots=KnotCtrl(By, Bv, Bv, 0.2, 0.03, 0.20, log=True, invert=False, name="b"),
                     obs=DUMMY)
    return assimilate(cfg, mode="forward", record_series=True, annotate=False, verbose=False)


# CTRL baseline year. The default (None) pins CTRL at the OLDEST knot, which at
# this layout is year -1000 - the knot the borehole cannot see (0.087 sigma
# response), so its value is set by the prior, not the data. Differencing every
# scenario against a prior-valued baseline puts that error straight into the
# attribution. FIRN_RA_BASE_YEAR=1600 anchors CTRL at the oldest knot the
# observations actually constrain (>=2 sigma response) instead.
_BASE_YEAR = os.environ.get("FIRN_RA_BASE_YEAR")
BASE_YEAR = float(_BASE_YEAR) if _BASE_YEAR else None


def _ctrl_level(years, vals, log=False):
    """The constant CTRL value: oldest knot by default, else the BASE_YEAR value."""
    if BASE_YEAR is None:
        return vals[0]
    if log:   # interpolate accumulation in log space, as the control is optimized
        return float(np.exp(np.interp(BASE_YEAR, years, np.log(vals))))
    return float(np.interp(BASE_YEAR, years, vals))


def attribution(params, Ty, Tv, By, Bv, label):
    """CTRL / TONLY / FULL for one (law, forcing) set -> attribution dict."""
    t0 = time.perf_counter()
    Tc = np.full_like(Tv, _ctrl_level(Ty, Tv))
    Bc = np.full_like(Bv, _ctrl_level(By, Bv, log=True))
    F = scenario(params, Ty, Tv, By, Bv, f"ra_{label}_full")
    T = scenario(params, Ty, Tv, By, Bc, f"ra_{label}_tonly")
    C = scenario(params, Ty, Tc, By, Bc, f"ra_{label}_ctrl")
    sf, st, sc = F["series"], T["series"], C["series"]
    dt = 5.0
    dhdt_full = np.asarray(sc["wbot"]) - np.asarray(sf["wbot"])
    dhdt_T = np.asarray(sc["wbot"]) - np.asarray(st["wbot"])
    hp = np.cumsum(dhdt_full) * dt
    hp_T = np.cumsum(dhdt_T) * dt
    dfac = np.asarray(sf["fac"]) - np.asarray(sc["fac"])
    mass = np.cumsum(np.asarray(sf["b"]) - np.asarray(sc["b"])) * dt
    print(f"  [{label}] h'(2015)={hp[-1]*100:+.1f} cm (T {hp_T[-1]*100:+.1f}, "
          f"b {(hp[-1]-hp_T[-1])*100:+.1f}); FAC {sf['fac'][-1]:.2f} vs {sc['fac'][-1]:.2f} m "
          f"({time.perf_counter()-t0:.0f}s)")
    return dict(years=np.asarray(sf["year"]).tolist(),
                Ts_full=np.asarray(sf["Ts_C"]).tolist(), b_full=np.asarray(sf["b"]).tolist(),
                fac_full=np.asarray(sf["fac"]).tolist(), fac_tonly=np.asarray(st["fac"]).tolist(),
                fac_ctrl=np.asarray(sc["fac"]).tolist(),
                hprime_m=hp.tolist(), hprime_T_m=hp_T.tolist(),
                hprime_b_m=(hp - hp_T).tolist(), dfac_m=dfac.tolist(), mass_m=mass.tolist(),
                snap_depth=F["snap_depth"].tolist(),
                snap_years=[float(v) for v in F["snaps"]["year"]],
                drho=(np.asarray(F["snaps"]["rho"]) - np.asarray(C["snaps"]["rho"])).tolist())


print("OSSE reanalysis - recovered vs truth scenario differencing")
if BASE_YEAR is None:
    print("  CTRL baseline: OLDEST knot (default)")
else:
    print(f"  CTRL baseline: year {BASE_YEAR:.0f}  "
          f"(T rec {_ctrl_level(REC_TY, REC_TV):.2f} vs truth {_ctrl_level(TRU_TY, TRU_TV):.2f} C; "
          f"b rec {_ctrl_level(REC_BY, REC_BV, log=True):.4f} vs truth "
          f"{_ctrl_level(TRU_BY, TRU_BV, log=True):.4f} m/yr)")
rec = attribution(REC, REC_TY, REC_TV, REC_BY, REC_BV, "recovered")
tru = attribution(TRU, TRU_TY, TRU_TV, TRU_BY, TRU_BV, "truth")

# keep the default-baseline product at the canonical name so plot.py is unaffected;
# a re-baselined run writes a suffixed file so the two can be compared side by side.
_sfx = "" if BASE_YEAR is None else f"_base{BASE_YEAR:.0f}"
json.dump(dict(recovered=rec, truth=tru, base_year=BASE_YEAR,
               dT_exp=tr.get("dT_exp"), db_exp=tr.get("db_exp"), tau=tr.get("tau")),
          open(OUT / f"synthetic_reanalysis{_sfx}.json", "w"))
np.savez_compressed(OUT / f"synthetic_reanalysis{_sfx}.npz",
                    depth=np.asarray(rec["snap_depth"]),
                    snap_years=np.asarray(rec["snap_years"]),
                    drho_rec=np.asarray(rec["drho"]), drho_tru=np.asarray(tru["drho"]))
h_r, h_t = rec["hprime_m"][-1] * 100, tru["hprime_m"][-1] * 100
print(f"\nh'(2015): recovered {h_r:+.1f} cm, truth {h_t:+.1f} cm "
      f"(recovered/truth {h_r/h_t:.2f})")
print(f"Saved {OUT/f'synthetic_reanalysis{_sfx}.json'} and .npz")
