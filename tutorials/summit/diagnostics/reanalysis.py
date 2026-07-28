"""reanalysis.py — Summit firn structure + height change from the calibrated model.

Scenario-differencing through the shared engine's forward mode (untaped), at a
given MAP. Summit's temperature forcing is CONSTANT (no deep thermal data), so
unlike South Pole there is no T scenario — the firn response here is entirely
accumulation-driven:

  CTRL : b held at the recovered year-1000 value
  FULL : recovered b(t) history

Attribution within the 150 m surface-following column (differencing cancels
V_deep and any CTRL drift exactly):
    dh'/dt = |w_bot|_FULL − |w_bot|_CTRL
    mass   = cumsum(b_FULL − b_CTRL)      (ice-eq part; baseline-dependent)
    dFAC   = FAC_FULL − FAC_CTRL          (the firn-PROCESS/air part)
Firn structure: FAC(t), close-off depth (rho=830) per snapshot, and the
density-anomaly field rho_FULL − rho_CTRL.

The FULL run's J is checked against the MAP's stored J_map (guard against an
env/error-model mismatch — legacy flags must match the MAP's vintage).

Outputs (pure data; figures are plot.py's job):
  output/summit_reanalysis.json / .npz

Env: FIRN_MAP (MAP json in output/, default summit_recal.json); usual FIRN_*.
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/summit/diagnostics/reanalysis.py
"""
from __future__ import annotations
import copy, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from config import build_cfg
from firnpack.inverse import assimilate

OUT = HERE / "output"
MAP_NAME = os.environ.get("FIRN_MAP", "summit_recal.json")
MAP = json.load(open(OUT / MAP_NAME))
m = MAP["m_map"]

print("=" * 64)
print(f"Summit firn reanalysis — engine forward at {MAP_NAME}, CTRL vs FULL "
      "(accumulation-driven; T forcing constant)")
print(f"  MAP: " + " ".join(f"{k}={v:.4g}" for k, v in m.items()))
print("=" * 64)

cfg = build_cfg().cfg

def scenario(name, warm):
    t0 = time.perf_counter()
    r = assimilate(cfg, mode="forward", warm=warm, verbose=False,
                   record_series=True, annotate=False)
    s = r["series"]
    print(f"  [{name}] J={r['J']:.4f}  FAC(2015)={s['fac'][-1]:.3f} m  "
          f"|w_bot|={-s['wbot'][-1]:.4f} m/yr  ({time.perf_counter()-t0:.0f}s)")
    return r

warm_full = MAP
warm_ctrl = copy.deepcopy(MAP)
warm_ctrl["b_knots"] = [MAP["b_knots"][0]] * len(MAP["b_knots"])

F = scenario("FULL", warm_full)
C = scenario("CTRL", warm_ctrl)

jmap = MAP.get("J_map")
if jmap is not None and abs(F["J"] - jmap) > 0.05:
    print(f"  ** WARNING: FULL J={F['J']:.4f} != stored J_map={jmap:.4f} — "
          f"config/env does not match the MAP's error model **")

yr = np.asarray(F["series"]["year"], float)
dt_yr = float(cfg.dt_years)
sf, sc = F["series"], C["series"]

dhdt = np.asarray(sc["wbot"]) - np.asarray(sf["wbot"])   # w negative down
hprime = np.cumsum(dhdt) * dt_yr
dfac = np.asarray(sf["fac"]) - np.asarray(sc["fac"])
mass = np.cumsum(np.asarray(sf["b"]) - np.asarray(sc["b"])) * dt_yr

pre = yr < float(MAP["b_knot_years"][0])
print(f"\nSanity: pre-{MAP['b_knot_years'][0]:.0f} anomaly (forcing identical): "
      f"max|dh'/dt| {np.abs(dhdt[pre]).max():.2e} m/yr, "
      f"max|dFAC| {np.abs(dfac[pre]).max():.2e} m")

# close-off depth per snapshot (firn structure)
snap_z = np.asarray(F["snap_depth"], float)
def closeoff(rhos):
    out = []
    for prof in rhos:
        p = np.asarray(prof)
        out.append(float(np.interp(830.0, p, snap_z)) if p.max() > 830 else float("nan"))
    return np.array(out)
zco_full = closeoff(F["snaps"]["rho"]); zco_ctrl = closeoff(C["snaps"]["rho"])

def _at(y): return int(np.argmin(np.abs(yr - y)))
i1900, i1957 = _at(1900.0), _at(1957.0)
headline = dict(
    hprime_2015_cm=float(hprime[-1] * 100),
    mass_2015_cm=float(mass[-1] * 100),
    dfac_2015_cm=float(dfac[-1] * 100),
    hprime_min_cm=float(hprime.min() * 100),
    hprime_min_year=float(yr[int(np.argmin(hprime))]),
    dhdt_1900_2015_mm_yr=float(np.mean(dhdt[i1900:]) * 1000),
    dhdt_1957_2015_mm_yr=float(np.mean(dhdt[i1957:]) * 1000),
    dhdt_2015_mm_yr=float(dhdt[-1] * 1000),
    fac_ctrl_2015_m=float(sc["fac"][-1]), fac_full_2015_m=float(sf["fac"][-1]),
    zco_full_2015_m=float(zco_full[-1]), zco_ctrl_2015_m=float(zco_ctrl[-1]),
)
print("\n===== HEADLINE =====")
print(f"  h'(2015) = {headline['hprime_2015_cm']:+.1f} cm vs 1000-CE accumulation "
      f"(ice-eq mass {headline['mass_2015_cm']:+.1f}, firn-process/air "
      f"{headline['dfac_2015_cm']:+.1f})")
print(f"  peak {headline['hprime_min_cm']:+.1f} cm @ {headline['hprime_min_year']:.0f}; "
      f"mean dh'/dt 1957-2015 = {headline['dhdt_1957_2015_mm_yr']:+.2f} mm/yr; "
      f"2015 inst = {headline['dhdt_2015_mm_yr']:+.2f} mm/yr")
print(f"  FAC(2015) {headline['fac_full_2015_m']:.2f} vs CTRL {headline['fac_ctrl_2015_m']:.2f} m; "
      f"close-off {headline['zco_full_2015_m']:.1f} vs {headline['zco_ctrl_2015_m']:.1f} m")

json.dump(dict(
    map_file=MAP_NAME, map_params=m, J_full=F["J"], J_map_stored=jmap,
    b_knot_years=MAP["b_knot_years"], b_knots=MAP["b_knots"],
    b_prior_centers=MAP.get("b_prior_centers"),
    dt_years=dt_yr, years=yr.tolist(),
    b_full=np.asarray(sf["b"]).tolist(), b_ctrl=np.asarray(sc["b"]).tolist(),
    fac_full=np.asarray(sf["fac"]).tolist(), fac_ctrl=np.asarray(sc["fac"]).tolist(),
    wbot_full=np.asarray(sf["wbot"]).tolist(), wbot_ctrl=np.asarray(sc["wbot"]).tolist(),
    hprime_m=hprime.tolist(), dfac_m=dfac.tolist(), mass_m=mass.tolist(),
    snap_years=[float(v) for v in F["snaps"]["year"]],
    zco_full_m=zco_full.tolist(), zco_ctrl_m=zco_ctrl.tolist(),
    headline=headline,
), open(OUT / "summit_reanalysis.json", "w"), indent=1)
np.savez_compressed(
    OUT / "summit_reanalysis.npz",
    depth=snap_z, snap_years=np.asarray(F["snaps"]["year"], float),
    rho_full=F["snaps"]["rho"], rho_ctrl=C["snaps"]["rho"],
    w_full=F["snaps"]["w"], w_ctrl=C["snaps"]["w"])
print(f"\nSaved {OUT/'summit_reanalysis.json'} and .npz — figures: run plot.py")
