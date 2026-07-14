"""experiments/synthetic/run.py — Observing System Simulation Experiment (OSSE).

Method-credibility anchor: set a KNOWN truth (densification law + conductivity +
surface-T history + accumulation history), generate synthetic density / depth-age
/ layer-gradient / borehole-T observations from the forward model with realistic
noise, then invert with the SAME engine from a neutral literature prior and check
the MAP recovers the truth. Adjoint gradient verified by finite differences.

Uses firnpack.inverse (the shared engine) for both the truth-forward and the
inversion — identical machinery to South Pole / Summit.

Modes (env FIRN_MODE): "verify" (truth recovery FD check), "optimize" (default).
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> experiments/synthetic/run.py
"""
from __future__ import annotations
import math, os
from pathlib import Path
import numpy as np
from firnpack.inverse import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl, assimilate
from firnpack.models.firn import FirnParameters
from firnpack.constants import year as YEAR_S

HERE = Path(__file__).parent; OUT = str(HERE/"results")
MODE = os.environ.get("FIRN_MODE", "optimize")
rng = np.random.default_rng(int(os.environ.get("FIRN_SEED", "0")))
_p = FirnParameters(); c_i = float(_p.c_i)
H0 = 130.0

# ---- KNOWN TRUTH (a plausible cold-firn scenario) ----
TRUTH = dict(hl_k0=11.5, hl_k1=540.0, hl_Ea1=10600.0, hl_Ea2=22400.0,
             k_snow_scale=1.15, k_firn_scale=1.0, Q_base=0.045, s2_shape=0.85)
T_YEARS = np.array([1000.,1300.,1600.,1850.,1950.,2015.])
T_TRUTH = np.array([-52.0,-51.5,-51.8,-51.0,-50.4,-50.7])   # a warm-peak-ish history
B_YEARS = np.array([1000.,1200.,1400.,1600.,1800.,1900.,1960.,2015.])
B_TRUTH = np.array([0.088,0.090,0.086,0.091,0.094,0.089,0.097,0.085])

def make_cfg(scalars, T_knots, b_knots, obs, tag, maxit=80):
    return SiteConfig(name="Synthetic", out_dir=OUT, tag=tag,
        H_col=H0, NZ=100, spin_years=2500.0, dt_years=5.0, rho_surf=350.0,
        rho_ic_deep=820.0, rho_ic_scale=25.0, conductivity_law="calonne2019",
        scalars=scalars, T_knots=T_knots, b_knots=b_knots, obs=obs,
        max_iter=maxit)

# ---- 1. truth forward -> synthetic profiles ----
truth_scalars = [ScalarCtrl(k, TRUTH[k], TRUTH[k], 1.0, 1e-6, 1e9,
                            log=(k not in ("Q_base",)), active=False) for k in TRUTH]
truth_T = KnotCtrl(T_YEARS, T_TRUTH, T_TRUTH, 0.6, -60, -40, log=False, invert=False, name="Tk")
truth_b = KnotCtrl(B_YEARS, B_TRUTH, B_TRUTH, 0.2, 0.03, 0.20, log=True, invert=False, name="b")
# a single inert observable so the engine runs (we only need the profiles)
dummy = [ObsBlock("rho", np.array([50.0]), np.array([600.0]), np.array([1e3]), label="rho")]
truth_cfg = make_cfg(truth_scalars, truth_T, truth_b, dummy, "synthetic_truth")
tr = assimilate(truth_cfg, mode="forward")
prof = tr["profiles"]; d = np.array(prof["depth"]); rho_t = np.array(prof["rho"])
age_t = np.array(prof["age_yr"]); T_t = np.array(prof["T_C"])
o = np.argsort(d); d, rho_t, age_t, T_t = d[o], rho_t[o], age_t[o], T_t[o]
print(f"truth forward: rho(10)={np.interp(10,d,rho_t):.0f} age(100)={np.interp(100,d,age_t):.0f} "
      f"T(bot)={T_t[-1]:.2f}")

# ---- 2. sample synthetic observations + realistic noise ----
def samp(depths): return depths
rho_d = np.linspace(2, 128, 45); rho_o = np.interp(rho_d, d, rho_t)
rho_sig = 15.0 + 0.03*rho_o; rho_obs = rho_o + rng.normal(0, rho_sig)
age_d = np.linspace(4, 128, 40); age_o = np.interp(age_d, d, age_t)
age_sig = 10.0 + 0.03*age_o; age_obs = age_o + rng.normal(0, age_sig)
# shallow points (2-8 m, like an RTD string) make near-surface conductivity
# identifiable; deep points (10-128 m) constrain the basal flux + T history
Tt_d = np.concatenate([[2., 4., 6., 8.], np.linspace(10, 128, 40)]); Tt_o = np.interp(Tt_d, d, T_t)
T_obs = Tt_o + rng.normal(0, 0.2, len(Tt_d))
# d(age)/dz from the truth age profile (local slope), noisy
dc, do, dsg = [], [], []
for cc in np.arange(6.0, H0-1.0, 2.0):
    sl = np.interp(cc+0.5, d, age_t) - np.interp(cc-0.5, d, age_t)  # yr/m
    dc.append(cc); do.append(sl + rng.normal(0, 0.05*abs(sl)+0.1)); dsg.append(max(0.05*abs(sl), 0.15))

obs = [
    ObsBlock("rho", rho_d, rho_obs, rho_sig, label="rho"),
    ObsBlock("age", age_d, age_obs*YEAR_S, (10.0+0.03*age_o)*YEAR_S, label="age"),
    ObsBlock("dagedz", np.array(dc), np.array(do), np.array(dsg), label="dage"),
    ObsBlock("enthalpy", Tt_d, c_i*(T_obs+273.15-float(_p.T_ref)), c_i*np.full(len(Tt_d),0.2), label="T"),
]

# ---- 3. inversion config: neutral literature prior, recover truth ----
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
T_ctr = np.interp(T_YEARS, [1000,2015], [-51.5,-51.5])   # flat neutral prior
inv_T = KnotCtrl(T_YEARS, T_ctr, T_ctr, 0.6, -60, -44, log=False, invert=True, name="Tk")
b_ctr = np.full(len(B_YEARS), 0.09)
inv_b = KnotCtrl(B_YEARS, b_ctr, b_ctr, 0.20, 0.03, 0.20, log=True, invert=True, name="b")
cfg = make_cfg(inv_scalars, inv_T, inv_b, obs, "synthetic_osse",
               maxit=int(os.environ.get("FIRN_MAX_ITER","100")))

if MODE == "verify":
    assimilate(cfg, mode="verify", fd_names=["hl_k0","hl_Ea2","s2_shape","k_snow_scale","b1600","Tk3"])
else:
    r = assimilate(cfg, mode="optimize")
    print("\n=== TRUTH RECOVERY ===")
    for k in TRUTH:
        if k in r["m_map"]:
            print(f"  {k:12s} truth={TRUTH[k]:.4g}  recovered={r['m_map'][k]:.4g}  "
                  f"ratio={r['m_map'][k]/TRUTH[k]:.3f}")
    import json
    json.dump(dict(truth=TRUTH, T_truth=T_TRUTH.tolist(), b_truth=B_TRUTH.tolist(),
                   T_years=T_YEARS.tolist(), b_years=B_YEARS.tolist()),
              open(Path(OUT)/"synthetic_truth.json","w"), indent=1)
    print(f"  (truth saved for the recovery figure)")
