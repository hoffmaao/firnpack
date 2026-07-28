"""reanalysis.py — firn-driven surface-height change from the calibrated model.

Scenario-differencing through the SHARED ENGINE's forward mode (untaped, so a
2500-yr scenario is seconds, not minutes), at a given MAP. Three scenarios:

  CTRL  : T and b held at their year-1000 recovered values (spinup climate)
  TONLY : recovered T(t) history, b held at b(1000)
  FULL  : recovered T(t) AND b(t)

Attribution within the 130 m surface-following column (the frame rides on the
surface, so the datum-relative surface rate is V_deep + |w_bot|; differencing
cancels V_deep, the constant ezz, and any CTRL drift exactly):

    dh'/dt = |w_bot|_scenario - |w_bot|_CTRL
    T-driven part  = TONLY - CTRL     (compaction response to the T history)
    b-driven part  = FULL - TONLY     (mass + compaction response to b(t))

Cross-check: the AIR part of h' is dFAC = FAC_scenario - FAC_CTRL (agrees with
the wbot route up to the small air-export anomaly at the base, ~1-rho_bot/rho_i).
The ice-equivalent MASS part of FULL is cumsum(b - b(1000)) and is
baseline-dependent by construction; report it separately, never summed blindly.

Unlike the archived archive/southpole/scripts/reanalysis/sp_reanalysis.py this
runs the ENGINE's own stepper (G_base basal BC, pinned ezz, calonne2019 k,
s2_shape — whatever the config says), so the reanalysis is exactly the physics
the MAP was inverted under. The FULL run's J is checked against the MAP's
stored J_map as a consistency guard.

Outputs (pure data; figures are plot.py's job):
  output/sp_reanalysis.json  — series + headline numbers
  output/sp_reanalysis.npz   — profile snapshots (for the Hovmoller panel)

Env: FIRN_MAP (MAP json in output/, default sp_recal.json); usual FIRN_* knobs.
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/diagnostics/reanalysis.py
"""
from __future__ import annotations
import copy, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent   # the tutorial dir
sys.path.insert(0, str(HERE))
from config import build_cfg
from firnpack.inverse import assimilate

OUT = HERE / "output"
MAP_NAME = os.environ.get("FIRN_MAP", "sp_recal.json")
MAP = json.load(open(OUT / MAP_NAME))
m = MAP["m_map"]

print("=" * 64)
print(f"SP firn reanalysis — engine forward at {MAP_NAME}, CTRL vs TONLY vs FULL")
print(f"  MAP: " + " ".join(f"{k}={v:.4g}" for k, v in m.items()))
print("=" * 64)

_b = build_cfg()
cfg = _b.cfg

def scenario(name, warm):
    t0 = time.perf_counter()
    r = assimilate(cfg, mode="forward", warm=warm, verbose=False,
                   record_series=True, annotate=False)
    s = r["series"]
    print(f"  [{name}] J={r['J']:.4f}  FAC(2015)={s['fac'][-1]:.3f} m  "
          f"|w_bot|={-s['wbot'][-1]:.4f} m/yr  ({time.perf_counter()-t0:.0f}s)")
    return r

# FULL: the MAP as stored. CTRL/TONLY: constant knot histories at the recovered
# year-1000 values (same scalars, so the physics is identical; only forcing
# differs and the differencing isolates the firn's response to it).
warm_full = MAP
warm_tonly = copy.deepcopy(MAP)
warm_tonly["b_knots"] = [MAP["b_knots"][0]] * len(MAP["b_knots"])
warm_ctrl = copy.deepcopy(warm_tonly)
warm_ctrl["T_knots"] = [MAP["T_knots"][0]] * len(MAP["T_knots"])

F = scenario("FULL", warm_full)
T = scenario("TONLY", warm_tonly)
C = scenario("CTRL", warm_ctrl)

jmap = MAP.get("J_map")
if jmap is not None and abs(F["J"] - jmap) > 0.05:
    print(f"  ** WARNING: FULL J={F['J']:.4f} != stored J_map={jmap:.4f} — "
          f"config/env does not match the MAP's error model **")

yr = np.asarray(F["series"]["year"], float)
dt_yr = float(cfg.dt_years)
sf, st, sc = F["series"], T["series"], C["series"]

# dh'/dt = |w_bot|_X - |w_bot|_C = wbot_C - wbot_X  (w negative down)
dhdt_full = np.asarray(sc["wbot"]) - np.asarray(sf["wbot"])
dhdt_T = np.asarray(sc["wbot"]) - np.asarray(st["wbot"])
hprime = np.cumsum(dhdt_full) * dt_yr
hprime_T = np.cumsum(dhdt_T) * dt_yr
hprime_b = hprime - hprime_T
dfac = np.asarray(sf["fac"]) - np.asarray(sc["fac"])
dfac_T = np.asarray(st["fac"]) - np.asarray(sc["fac"])
mass = np.cumsum(np.asarray(sf["b"]) - np.asarray(sc["b"])) * dt_yr  # m ice eq

pre = yr < float(MAP["knot_years"][0])
print(f"\nSanity: pre-{MAP['knot_years'][0]:.0f} anomaly (forcing identical): "
      f"max|dh'/dt| {np.abs(dhdt_full[pre]).max():.2e} m/yr, "
      f"max|dFAC| {np.abs(dfac[pre]).max():.2e} m")

def _at(y): return int(np.argmin(np.abs(yr - y)))
i1900, i1957 = _at(1900.0), _at(1957.0)
headline = dict(
    hprime_2015_cm=float(hprime[-1] * 100),
    hprime_T_2015_cm=float(hprime_T[-1] * 100),
    hprime_b_2015_cm=float(hprime_b[-1] * 100),
    mass_2015_cm=float(mass[-1] * 100),
    dfac_2015_cm=float(dfac[-1] * 100),
    dfac_T_2015_cm=float(dfac_T[-1] * 100),
    hprime_min_cm=float(hprime.min() * 100),
    hprime_min_year=float(yr[int(np.argmin(hprime))]),
    dhdt_1900_2015_mm_yr=float(np.mean(dhdt_full[i1900:]) * 1000),
    dhdt_1957_2015_mm_yr=float(np.mean(dhdt_full[i1957:]) * 1000),
    dhdt_2015_mm_yr=float(dhdt_full[-1] * 1000),
    dhdt_T_2015_mm_yr=float(dhdt_T[-1] * 1000),
    fac_ctrl_2015_m=float(sc["fac"][-1]), fac_full_2015_m=float(sf["fac"][-1]),
)
print("\n===== HEADLINE =====")
print(f"  h'(2015) total     = {headline['hprime_2015_cm']:+.1f} cm "
      f"(T-driven {headline['hprime_T_2015_cm']:+.1f}, "
      f"b-driven {headline['hprime_b_2015_cm']:+.1f}; "
      f"of the b part, ice-eq mass {headline['mass_2015_cm']:+.1f})")
print(f"  air part dFAC      = {headline['dfac_2015_cm']:+.1f} cm "
      f"(T-only dFAC {headline['dfac_T_2015_cm']:+.1f} — "
      f"cross-check vs h'_T)")
print(f"  peak drawdown      = {headline['hprime_min_cm']:+.1f} cm @ "
      f"{headline['hprime_min_year']:.0f}")
print(f"  mean dh'/dt 1900-2015 = {headline['dhdt_1900_2015_mm_yr']:+.2f} mm/yr; "
      f"1957-2015 = {headline['dhdt_1957_2015_mm_yr']:+.2f} mm/yr; "
      f"2015 inst = {headline['dhdt_2015_mm_yr']:+.2f} mm/yr "
      f"(T-only {headline['dhdt_T_2015_mm_yr']:+.2f})")

json.dump(dict(
    map_file=MAP_NAME, map_params=m, J_full=F["J"], J_map_stored=jmap,
    knot_years=MAP["knot_years"], T_knots=MAP["T_knots"],
    b_knot_years=MAP["b_knot_years"], b_knots=MAP["b_knots"],
    dt_years=dt_yr, years=yr.tolist(),
    Ts_full=np.asarray(sf["Ts_C"]).tolist(), b_full=np.asarray(sf["b"]).tolist(),
    Ts_ctrl=np.asarray(sc["Ts_C"]).tolist(), b_ctrl=np.asarray(sc["b"]).tolist(),
    fac_full=np.asarray(sf["fac"]).tolist(), fac_tonly=np.asarray(st["fac"]).tolist(),
    fac_ctrl=np.asarray(sc["fac"]).tolist(),
    wbot_full=np.asarray(sf["wbot"]).tolist(), wbot_tonly=np.asarray(st["wbot"]).tolist(),
    wbot_ctrl=np.asarray(sc["wbot"]).tolist(),
    tmean_full=np.asarray(sf["tmean"]).tolist(), tmean_ctrl=np.asarray(sc["tmean"]).tolist(),
    hprime_m=hprime.tolist(), hprime_T_m=hprime_T.tolist(),
    hprime_b_m=hprime_b.tolist(), dfac_m=dfac.tolist(), dfac_T_m=dfac_T.tolist(),
    mass_m=mass.tolist(), headline=headline,
), open(OUT / "sp_reanalysis.json", "w"), indent=1)
np.savez_compressed(
    OUT / "sp_reanalysis.npz",
    depth=F["snap_depth"], snap_years=F["snaps"]["year"],
    rho_full=F["snaps"]["rho"], rho_ctrl=C["snaps"]["rho"],
    T_full=F["snaps"]["T"], T_ctrl=C["snaps"]["T"],
    w_full=F["snaps"]["w"], w_ctrl=C["snaps"]["w"])
print(f"\nSaved {OUT/'sp_reanalysis.json'} and .npz — figures: run plot.py")
