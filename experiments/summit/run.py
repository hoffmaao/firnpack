"""experiments/summit/run.py — Summit (Greenland) inversion via the shared engine.

Recovers Summit's OWN densification law + accumulation history from density
(composite core) + depth-age (GISP2) + layer-gradient d(age)/dz, under a
prescribed near-isothermal temperature forcing (FirnCover deep-mean -28.8 C;
Summit's 12 m firn-T can't support a deep thermal inversion, so T is an input,
not a target). Same neutral literature-H&L prior as South Pole, so the two
recovered laws are directly comparable — the paper's transferability result.

No velocity/ApRES and no deep borehole T at Summit -> those observables and the
ezz / conductivity / Q_base controls are simply dropped (fixed).

Modes (env FIRN_MODE): "verify" (replay + FD), "optimize" (default).
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> experiments/summit/run.py
"""
from __future__ import annotations
import math, os
from pathlib import Path
import numpy as np, pandas as pd
from firn.inverse import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl, assimilate
from firn.models.firn import FirnParameters
from firn.constants import year as YEAR_S

HERE = Path(__file__).parent; DATA = HERE / "data"; OUT = str(HERE / "results")
MODE = os.environ.get("FIRN_MODE", "optimize")
H0 = 150.0
_p = FirnParameters(); c_i = float(_p.c_i); T_ref = float(_p.T_ref)
T_FORCE = -28.8   # FirnCover Summit deep-firn mean

# ---- observations ----
dens = pd.read_csv(DATA/"summit_density.csv")
rho_col = [c for c in dens.columns if "rho" in c.lower()][0]
dro = dens[rho_col].values; dro = dro*1000.0 if np.nanmax(dro) < 5 else dro
# bin the composite density to ~2 m (mid-firn is clustered; smooths pycnometry scatter)
edges = np.arange(0, 92, 2.0); dc = 0.5*(edges[:-1]+edges[1:]); dv, dn = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (dens.depth_m.values >= lo) & (dens.depth_m.values < hi)
    if s.sum() >= 1: dv.append(np.mean(dro[s])); dn.append(0.5*(lo+hi))
rho_d = np.array(dn); rho_v = np.array(dv); rho_sig = 15.0 + 0.03*rho_v

gis = pd.read_csv(DATA/"gisp2_depth_age.csv")
gis = gis[(gis.depth_m <= H0) & (gis.year_CE <= 2015)].sort_values("depth_m")
g_d = gis.depth_m.values; g_ageyr = 2015.0 - gis.year_CE.values
sel = g_ageyr >= 0; g_d, g_ageyr = g_d[sel], g_ageyr[sel]
# subsample age to ~40 pts; dage slopes from full-res
def _sub(x, n): return x[::max(1, len(x)//n)]
age_d, age_yr = _sub(g_d, 40), _sub(g_ageyr, 40)
dc2, do2, ds2 = [], [], []
for c in np.arange(6.0, H0-1.0+1e-9, 3.0):
    s = np.abs(g_d - c) <= 1.0
    if s.sum() >= 4:
        A = np.vstack([g_d[s]-c, np.ones(s.sum())]).T
        coef, res, *_ = np.linalg.lstsq(A, g_ageyr[s], rcond=None); nn = s.sum()
        se = math.sqrt(max(float(res[0]) if len(res) else 0.0, 1e-12)/(nn-2)/max(np.sum((g_d[s]-c)**2), 1e-12))
        dc2.append(c); do2.append(coef[0]); ds2.append(max(se, 0.05*abs(coef[0])))

obs = [
    ObsBlock("rho", rho_d, rho_v, rho_sig, label="rho"),
    ObsBlock("age", age_d, age_yr*YEAR_S, (10.0+0.03*age_yr)*YEAR_S, label="age"),
    ObsBlock("dagedz", np.array(dc2), np.array(do2), np.array(ds2), label="dage"),
]

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
B_YEARS = np.array([1000.,1150.,1300.,1450.,1600.,1750.,1850.,1920.,1970.,2015.])
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

print(f"Summit inversion: {len(obs[0].obs)} rho, {len(obs[1].obs)} age, {len(obs[2].obs)} dage; "
      f"T forcing {T_FORCE} C; {len(B_YEARS)} b-knots (accum)")
if MODE == "verify":
    assimilate(cfg, mode="verify", fd_names=["hl_Ea2","s2_shape","b1600"])
else:
    assimilate(cfg, mode="optimize")
