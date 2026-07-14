"""tutorials/southpole/run.py — South Pole assimilation via the shared engine.

Reproduces the frozen r8 MAP (sp_joint_r8.json) through firnpack.inverse.assimilate,
validating the engine refactor. Same config as round 8: coarse knots
(knots_r8), corrected borehole-T datum (-5.8), k_snow pinned (USP50 1.29),
de-biased accumulation prior (Buizert climatology 0.096), Calonne conductivity,
per-point chi^2, sigma_dage x2.2.

Modes (env FIRN_MODE): "validate" (default; forward J at the r8 MAP, compare
81.31), "verify" (replay + FD), "optimize" (full run, warm from r8).
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/run.py
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np, pandas as pd
from firnpack.inverse import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl, assimilate
from firnpack.constants import year as YEAR_S

HERE = Path(__file__).parent
DATA = HERE / "data"
SP_RESULTS = HERE / "results"                                # frozen r8 lives here
OUT = str(HERE / "results")
MODE = os.environ.get("FIRN_MODE", "validate")
# Resolution knobs (discretization-sensitivity study): defaults = r8 numerics.
NZ = int(os.environ.get("FIRN_NZ", "100"))
DT = float(os.environ.get("FIRN_DT", "5.0"))
TAG = os.environ.get("FIRN_TAG", "sp_engine")
T_SHIFT = -5.8
H0 = 130.0   # DATA cut: all obs stay <= 130 m regardless of domain depth
# Domain-truncation study (basal BC): H_COL deepens the column so the
# advected/diffused T-history part of the basal gradient is SIMULATED and the
# imposed G is only the quasi-steady deep part; obs cuts stay at H0.
H_COL = float(os.environ.get("FIRN_HCOL", str(H0)))
SPIN = float(os.environ.get("FIRN_SPIN", "2500.0"))

# ---- knot layouts (r8 coarse; override with FIRN_KNOTS=<json> for the
# temporal-sampling probe — file holds {"knot_years": [...], "b_knot_years": [...]}) ----
KNOT_YEARS = np.array([1000.,1300.,1550.,1750.,1850.,1900.,1930.,1955.,1975.,1990.,2000.,2008.,2015.])
KNOT_B_YEARS = np.array([1000.,1100.,1200.,1300.,1400.,1500.,1600.,1700.,1800.,1850.,1900.,1930.,1960.,1990.,2015.])
_kf = os.environ.get("FIRN_KNOTS", "")
if _kf:
    _kj = json.load(open(_kf if os.path.isabs(_kf) else HERE/_kf))
    KNOT_YEARS = np.array(_kj["knot_years"], dtype=float)
    KNOT_B_YEARS = np.array(_kj["b_knot_years"], dtype=float)
    print(f"knot layouts from {_kf}: {len(KNOT_YEARS)} T + {len(KNOT_B_YEARS)} b")

# ---- T prior ramp (hand-drawn 6-knot), datum-shifted ----
_T6Y = np.array([1000.,1300.,1550.,1750.,1900.,2015.]); _T6V = np.array([-45.50,-45.45,-45.35,-45.20,-45.10,-45.00])+T_SHIFT
T_CTR = np.interp(KNOT_YEARS, _T6Y, _T6V)

# ---- b prior centers: Buizert <=1950, de-biased climatology 0.096 >1950 ----
buiz = pd.read_csv(DATA/"buizert2021_spice_accum.csv").query("year_CE>=900")
b_yr, b_ac = buiz.year_CE.values[::-1], buiz.accum.values[::-1]
B_CLIM = 0.096
B_CENTER = np.array([np.interp(y,b_yr,b_ac) if y<=1950 else B_CLIM for y in KNOT_B_YEARS])

# ---- observations (verbatim SP data prep) ----
def _sub(df,n): return df.iloc[::max(1,len(df)//n)]
dens=_sub(pd.read_csv(DATA/"sp19_density.csv").query("depth_m<=@H0"),45)
age=pd.read_csv(DATA/"sp19_depth_age.csv"); age["age_yr"]=2015.0-age.year_CE
agedf=age.query("depth_m<=@H0 and age_yr>=0"); ages=_sub(agedf,40)
bT=_sub(pd.read_csv(DATA/"spicecore_borehole_T.csv").query("depth_m<=@H0"),40)
# ---- ApRES velocity: FIRN_VEL_SITE selects the observable ----
#   "pooled" (default): legacy all-site bin-median absolute velocity (chimera —
#     kept only for r8/r9 reproduction).
#   "<site>" (e.g. x11n6): that site's DIFFERENCED profile v(z)-v(z_ref)
#     (antenna-offset immune) with n(rho) depth REGISTRATION (both pipelines
#     assumed eps_ice=3.18; true depth > reported range in firn).
#   "none": no velocity block (control for the site-selection screen).
VEL_SITE = os.environ.get("FIRN_VEL_SITE", "pooled")
ap=pd.read_csv(DATA/"apres_vertical_velocity_processed.csv")
ap=ap[(ap.range_m<=H0)&ap.v_smooth_m_yr.notna()&(ap.coherence>0.5)]
_vb=np.arange(8.0,H0,8.0); _vm=0.5*(_vb[:-1]+_vb[1:]); ap["_b"]=np.digitize(ap.range_m.values,_vb)
_vr=[(_vm[i],ap.v_smooth_m_yr[ap._b==i+1].median(),ap.v_smooth_m_yr[ap._b==i+1].std(),int((ap._b==i+1).sum())) for i in range(len(_vm))]
_vr=[r for r in _vr if r[3]>=3]
# n(rho) registration map (Kovacs) from the observed SP19 density profile
_zz=np.linspace(0,200,900)
_rho_z=np.interp(_zz,pd.read_csv(DATA/"sp19_density.csv").depth_m,
                 pd.read_csv(DATA/"sp19_density.csv").rho_kgm3*1000.0)
_n_z=1.0+0.845e-3*np.where(_zz<130,_rho_z,917.0)
_NICE=math.sqrt(3.18)
_r_of_z=np.cumsum(_n_z/_NICE)*(_zz[1]-_zz[0])
def _z_true(rng): return float(np.interp(rng,_r_of_z,_zz))
def _n_at(z): return float(np.interp(z,_zz,_n_z))
VEL_SRC = os.environ.get("FIRN_VEL_SRC", "authors")  # authors (pipeline-A 25-m smoothed) | zeising (raw-burst reprocessing)
def site_vel_block_zeising(site, zref_range=30.0, sig_shape=3.5e-3, label="v"):
    # Zeising raw-burst product (proper phase errors; no 25-m smoothing — the
    # pipeline-A smoothing biased firn gradients by up to 4x): 6-m fine windows
    # stepped 2 m -> thin [::3] for independent samples. sigma = phase error
    # (~0.01 mm/yr, negligible) + cross-site shape systematic (3.2 mm/yr rms
    # over the 5 mutually consistent sites, uniform in depth).
    zei = pd.read_csv(DATA/"apres_zeising_processed.csv")
    d = zei[(zei.site==site)&(zei.range_m>=12.0)&(zei.range_m<=112.0)].sort_values("range_m")
    rr = d.range_m.values[::3]; vv = d.dRdt_myr.values[::3]; ee = d.dRdt_err_myr.values[::3]
    if len(rr) < 6: raise ValueError(f"site {site}: only {len(rr)} zeising points")
    vref = float(np.interp(zref_range, rr, vv)); eref = float(np.interp(zref_range, rr, ee))
    keep = np.abs(rr - zref_range) > 3.0
    zt = np.array([_z_true(r) for r in rr[keep]]); zr = _z_true(zref_range)
    # LOCAL-index kinematics (Case & Kingslake 2022): dR/dt = n(z) w(z) / n_ice
    nfac = np.array([_n_at(z)/_NICE for z in zt])
    sig = np.sqrt(ee[keep]**2 + eref**2 + sig_shape**2)
    return ObsBlock("dRdt_diff", zt, vv[keep]-vref, sig, label=label, ref_depth=zr,
                    nfac=nfac, nfac_ref=_n_at(zr)/_NICE)
def site_vel_block(site, zref_range=30.0, sig=0.012, label="v"):
    # per-site data are ~4.2 m spaced and already 25-m smoothed: no rebinning,
    # just thin the native samples to ~8 m
    d=ap[(ap.site==site)&(ap.range_m>=12.0)&(ap.range_m<=112.0)].sort_values("range_m")
    rr=d.range_m.values[::2]; vv=d.v_smooth_m_yr.values[::2]
    if len(rr)<6: raise ValueError(f"site {site}: only {len(rr)} usable points")
    vref=float(np.interp(zref_range,rr,vv))
    keep=np.abs(rr-zref_range)>4.0
    zt=np.array([_z_true(r) for r in rr[keep]]); zr=_z_true(zref_range)
    nfac=np.array([0.5*(_n_at(z)+_n_at(zr))/_NICE for z in zt])
    return ObsBlock("dRdt_diff", zt, vv[keep]-vref, np.full(keep.sum(),sig),
                    label=label, ref_depth=zr, nfac=nfac)
# d(age)/dz slopes at full resolution. Sigma = the STATED per-point estimate
# max(window-fit s.e., 4% floor); NO inflation (Andrew, 2026-07-10: use the
# stated observation uncertainties — the model's inability to fit interannual
# layer scatter should read as an honest ~2σ misfit, not be absorbed into σ).
# NOTE: the archived r8 MAP was produced WITH x2.2 inflation; to reproduce its
# J=81.31 exactly, set FIRN_SIG_DAGE_SCALE=2.2.
_ad,_aa=agedf.depth_m.values,agedf.age_yr.values
SIG_DAGE_SCALE=float(os.environ.get("FIRN_SIG_DAGE_SCALE","1.0"))
dc,do,dsg=[],[],[]
for c in np.arange(6.0,H0-1.0+1e-9,1.0):
    s=np.abs(_ad-c)<=0.6
    if s.sum()>=4:
        A=np.vstack([_ad[s]-c,np.ones(s.sum())]).T; coef,res,*_=np.linalg.lstsq(A,_aa[s],rcond=None); nn=s.sum()
        se=math.sqrt(max(float(res[0]) if len(res) else 0.0,1e-12)/(nn-2)/max(np.sum((_ad[s]-c)**2),1e-12))
        dc.append(c); do.append(coef[0]); dsg.append(max(se,0.04*abs(coef[0]))*SIG_DAGE_SCALE)
P0T_ref=273.15; c_i=2009.0; T_ref=273.15   # match FirnParameters (c_i, T_ref)
from firnpack.models.firn import FirnParameters as _FP
_p=_FP(); c_i=float(_p.c_i); T_ref=float(_p.T_ref)
obs=[
    ObsBlock("rho", dens.depth_m.values, dens.rho_kgm3.values*1000.0, 15.0+0.03*dens.rho_kgm3.values*1000.0, label="rho"),
    ObsBlock("age", ages.depth_m.values, ages.age_yr.values*YEAR_S, (10.0+0.03*ages.age_yr.values)*YEAR_S, label="age"),
    ObsBlock("dagedz", np.array(dc), np.array(do), np.array(dsg), label="dage"),
    ObsBlock("enthalpy", bT.depth_m.values, c_i*(bT.T_C.values+T_SHIFT+273.15-T_ref), c_i*np.full(len(bT),0.2), label="T"),
]
if VEL_SITE == "pooled":
    obs.append(ObsBlock("velocity", np.array([r[0] for r in _vr]), np.array([r[1] for r in _vr]),
               np.maximum(np.array([r[2]/max(r[3],1)**0.5 for r in _vr]),0.01), label="v"))
elif VEL_SITE != "none":
    # "+"-separated multi-site: one differenced block per site (distinct labels;
    # sites are genuinely different columns — never pool across sites)
    _sl = VEL_SITE.split("+")
    _build = site_vel_block_zeising if VEL_SRC == "zeising" else site_vel_block
    for _s in _sl:
        obs.append(_build(_s, label="v" if len(_sl) == 1 else f"v_{_s}"))
        print(f"velocity block[{VEL_SRC}]: site {_s}, differenced + n(rho)-registered "
              f"({obs[-1].n} pts, ref z={obs[-1].ref_depth:.1f} m)")
if os.environ.get("FIRN_SEAS", "0") == "1":
    # USP50 seasonal-amplitude damping IN the inversion (WKB ln-ratio operator
    # through the on-tape k law; replaces the offline k_snow pin — the OSSE
    # matrix shows mean profiles carry no k_snow info, the amplitude does).
    # obs pre-corrected for the WKB-vs-exact operator bias at s*=1.29.
    sa = pd.read_csv(DATA/"usp50_seasonal_lnratio.csv")
    ud = pd.read_csv(DATA/"usp50_density_nicl.csv")
    obs.append(ObsBlock("seas_lnamp", sa.zeff.values, sa.lnr.values, sa.sig.values,
                        label="seas", ref_depth=float(sa.zref.iloc[0]),
                        aux_z=ud.depth.values, aux_val=ud.Density.values))
    print(f"seasonal block: USP50 ln-amplitude ratios ({obs[-1].n} pts, "
          f"ref z={obs[-1].ref_depth:.2f} m), k_snow prior WIDENED")

# ---- controls ----
scalars=[
    ScalarCtrl("hl_k0",10.79,10.79,1.0,0.5,500,log=True),
    ScalarCtrl("hl_k1",570.7,570.7,1.0,10,50000,log=True),
    ScalarCtrl("hl_Ea1",10432.,10432.,0.5,3000,40000,log=True),
    ScalarCtrl("hl_Ea2",21875.,21875.,0.5,5000,80000,log=True),
    (ScalarCtrl("k_snow_scale",1.29,1.0,0.5,0.02,5.0,log=True)     # SEAS: weak prior, damping data ON TAPE
     if os.environ.get("FIRN_SEAS","0")=="1" else
     ScalarCtrl("k_snow_scale",1.29,1.29,0.08,0.02,5.0,log=True)), # USP50 pin (offline)
    ScalarCtrl("k_firn_scale",1.0,1.0,0.15,0.5,2.0,log=True),
    # Basal thermal BC control (Andrew, 2026-07-11): parameterize by the basal
    # TEMPERATURE GRADIENT G_base (K/m, warming-downward positive) — the
    # quantity the borehole data actually constrain at a truncation boundary —
    # rather than the flux Q (which entangles the BC with the conductivity
    # scales; flux is now the DERIVED product q = k*G). FIRN_BASAL=Q restores
    # the r8-era flux control (needed to reproduce archived r8 exactly).
    (ScalarCtrl("Q_base",0.0,0.0,0.05,-0.2,0.2)
     if os.environ.get("FIRN_BASAL","G")=="Q" else
     ScalarCtrl("G_base",-0.005,0.0,0.025,-0.1,0.1)),
    ScalarCtrl("ezz_yr",0.0,0.0,1.0e-4,-3.0e-4,2.0e-4),
    ScalarCtrl("s2_shape",1.0,1.0,0.3,0.3,2.0,log=True),
    ScalarCtrl("b_off",0.0,0.0,0.25,-0.8,0.8),
]
cfg=SiteConfig(name="SouthPole", out_dir=OUT, tag=TAG,
    H_col=H_COL, NZ=NZ, spin_years=SPIN, dt_years=DT, rho_surf=350.0,
    rho_ic_deep=820.0, rho_ic_scale=25.0, conductivity_law="calonne2019",
    scalars=scalars,
    T_knots=KnotCtrl(KNOT_YEARS, T_CTR, T_CTR, 0.6, -48.0+T_SHIFT, -43.0+T_SHIFT, log=False, invert=True, name="Tk"),
    b_knots=KnotCtrl(KNOT_B_YEARS, B_CENTER, B_CENTER, 0.20, 0.03, 0.20, log=True, invert=True, name="b"),
    b_off_era_year=1950.0, obs=obs, max_iter=int(os.environ.get("FIRN_MAX_ITER","80")))

warm = json.load(open(os.environ.get("FIRN_WARM_JSON", str(SP_RESULTS/"sp_joint_r8.json"))))

if MODE=="validate":
    r=assimilate(cfg, mode="forward", warm=warm)
    print(f"\nENGINE forward J at r8 MAP = {r['J']:.4f}  (r8 script J = 81.31, NZ={NZ} dt={DT})")
    print(f"  rms: " + " ".join(f"{k[4:]}={v:.3f}" for k,v in r['diag'].items() if k.startswith('rms_')))
    p=r["profiles"]; d=np.array(p["depth"]); rho=np.array(p["rho"]); ag=np.array(p["age_yr"]); Tp=np.array(p["T_C"])
    zco=float(np.interp(830.0,rho,d)) if rho.max()>830 else float("nan")
    print(f"  derived: close-off(830)={zco:.2f} m  age(100m)={np.interp(100,d,ag):.1f} yr  "
          f"rho(50m)={np.interp(50,d,rho):.1f}  T(bot)={Tp[-1]:.3f} C")
    print(f"  match to r8: {'YES' if abs(r['J']-81.31)<0.5 else 'CHECK'}")
elif MODE=="verify":
    _basal = "Q_base" if os.environ.get("FIRN_BASAL","G")=="Q" else "G_base"
    _fdn = os.environ.get("FIRN_FD_NAMES")
    assimilate(cfg, mode="verify", warm=warm,
               fd_names=(_fdn.split(",") if _fdn else
                         [_basal,"k_firn_scale","s2_shape","b1960","Tk8"]
                         + (["k_snow_scale"] if os.environ.get("FIRN_SEAS","0")=="1" else [])),
               fd_h=float(os.environ.get("FIRN_FD_H", "1e-3")))
elif MODE=="optimize":
    assimilate(cfg, mode="optimize", warm=warm)
