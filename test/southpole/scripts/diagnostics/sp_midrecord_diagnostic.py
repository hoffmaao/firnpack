"""sp_midrecord_diagnostic.py — is the mid-record misfit a LAW problem or a FORCING problem?

Runs the sp_joint_ezz MAP forward once, then organizes the remaining density/age
residuals two ways:
  * vs model density  -> coherent shape = densification-law shape error
  * vs deposition year (via the model's own age(z)) -> coherent centennial
    structure = missing forcing variability (accumulation / T)
and overlays the Buizert-2021 SPICE accumulation reconstruction on the
deposition-year view (model assumes constant bdot = 0.085).

Output: results/sp_midrecord_diagnostic.png + correlation printout.
"""
from __future__ import annotations
import functools, json, os
from pathlib import Path
import numpy as np, pandas as pd
import firedrake as fd
from firedrake.adjoint import stop_annotating
from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.physics.densification import herron_langway as _hl
from firnpack.constants import year as YEAR_S
herron_langway = functools.partial(_hl, smooth=True)

PROC = Path(__file__).parent.parent.parent / "processed"
OUT = Path(__file__).parent.parent.parent / "results"
MAP_FILE = os.environ.get("FIRN_MAP", "sp_joint_ezz.json")
SUFFIX = "" if MAP_FILE == "sp_joint_ezz.json" else "_" + Path(MAP_FILE).stem.replace("sp_joint_", "")
JM = json.load(open(OUT/MAP_FILE)); m = JM["m_map"]
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, BDOT, DT_YEARS, SPIN_YEARS = 350.0, 0.085, 5.0, 2500.0
K_ICE_BASE, BETA = 2.1, 5.0e2
P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)
n_steps = int(SPIN_YEARS/DT_YEARS)
step_years = 2015.0 - (n_steps-1-np.arange(n_steps))*DT_YEARS

dens = pd.read_csv(PROC/"sp19_density.csv").query("depth_m<=@H0")
aged = pd.read_csv(PROC/"sp19_depth_age.csv"); aged["age_yr"]=2015.0-aged.year_CE
aged = aged.query("depth_m<=@H0 and age_yr>=0")
buiz = pd.read_csv(PROC/"buizert2021_spice_accum.csv")

mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0*(1.0-(1.0-fd.SpatialCoordinate(mesh)[0])**STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); Rs = fd.FunctionSpace(mesh, "R", 0)
dx=fd.dx(domain=mesh); ds=fd.ds(domain=mesh); depth=H0-fd.SpatialCoordinate(mesh)[0]; psi=fd.TestFunction(V)
def mk(v,n): f=fd.Function(Rs,name=n); f.assign(float(v)); return f

T_series = np.interp(step_years, np.asarray(JM["knot_years"]), np.asarray(JM["T_knots"]))
if "b_knots" in JM:   # sp_joint_bdot-style MAP: geometric interpolation of b-knots
    b_series = np.exp(np.interp(step_years, np.asarray(JM["b_knot_years"]),
                                np.log(np.asarray(JM["b_knots"]))))
    print(f"time-varying bdot from {MAP_FILE}: [{b_series.min():.4f}, {b_series.max():.4f}] m/yr")
else:
    b_series = np.full(n_steps, BDOT)
_pk = dict(hl_k0_prefactor=mk(m["hl_k0"],"k0"),hl_k1_prefactor=mk(m["hl_k1"],"k1"),
    hl_Ea_stage1=mk(m["hl_Ea1"],"e1"),hl_Ea_stage2=mk(m["hl_Ea2"],"e2"),
    basal_heat_flux_W_m2=mk(m["Q_base"],"Q"),
    hl_stage2_shape=float(m.get("s2_shape", 1.0)))
if "k_snow_scale" in m:   # r5-style MAP: Calonne-2019 conductivity
    _pk.update(conductivity_law="calonne2019",
               k_snow_scale=float(m["k_snow_scale"]),
               k_firn_scale=float(m["k_firn_scale"]))
else:                     # legacy quadratic law
    _pk.update(k_ice=K_ICE_BASE*m["k_factor"])
params=FirnParameters(**_pk)
model=FirnModel(params,densification_rate_fn=herron_langway)
ws0=-BDOT*rho_i/RHO_SURF/YEAR_S
H_f=fd.Function(V);rho_f=fd.Function(V);w_f=fd.Function(V);age_f=fd.Function(V)
H_o=fd.Function(V);rho_o=fd.Function(V);w_o=fd.Function(V);age_o=fd.Function(V);Ts=fd.Function(V)
H_f.assign(c_i*(float(T_series.mean())+273.15-T_ref));H_o.assign(H_f)
rho_f.interpolate(RHO_SURF+(820.0-RHO_SURF)*(1.0-fd.exp(-depth/25.0)));rho_o.assign(rho_f)
w_f.assign(ws0);w_o.assign(w_f);age_f.assign(0.0);age_o.assign(0.0)
ac=mk(BDOT,"ac");rs=mk(RHO_SURF,"rs");ws=mk(ws0,"ws");dt_r=mk(DT_YEARS*YEAR_S,"dt")
bcr=fd.DirichletBC(V,rs,SID);bcw=fd.DirichletBC(V,ws,SID);bca=fd.DirichletBC(V,mk(0.0,"a0"),SID)
bm=ac*rho_i/P0.spy; Ht=fd.TrialFunction(V);rt=fd.TrialFunction(V);wt=fd.TrialFunction(V);at=fd.TrialFunction(V)
sp={"ksp_type":"preonly","pc_type":"lu"}
FH=model.enthalpy_form(Ht,H_o,rho_f,w_f,psi,dt_r)+fd.Constant(BETA)*(Ht-c_i*(Ts-fd.Constant(T_ref)))*psi*ds(SID)
sH=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(FH),fd.rhs(FH),H_f,constant_jacobian=False),solver_parameters=sp)
Te=model.temperature_from_enthalpy(H_f)
Fr,_=model.density_form(rt,rho_o,Te,w_f,None,bm,psi,dt_r)
sR=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(Fr),fd.rhs(Fr),rho_f,bcs=[bcr],constant_jacobian=False),solver_parameters=sp)
dr=herron_langway(rho_f,Te,params=params,bdot=bm)
dw=model.velocity_delta(wt,w_o,rho_f,dr,psi,regularization=1e-3,
                        horizontal_divergence=fd.Constant(-m["ezz_yr"]/YEAR_S))
sW=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(dw),fd.rhs(dw),w_f,bcs=[bcw],constant_jacobian=False),solver_parameters=sp)
Fa=model.age_form(at,age_o,w_f,w_o,psi,dt_r)
sA=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(Fa),fd.rhs(Fa),age_f,bcs=[bca],constant_jacobian=False),solver_parameters=sp)
with stop_annotating():
    for k in range(n_steps):
        Ts.assign(float(T_series[k])+273.15)
        ac.assign(float(b_series[k]))
        ws.assign(-float(b_series[k])*rho_i/RHO_SURF/YEAR_S)
        sH.solve();sR.solve();sW.solve();sA.solve()
        H_o.assign(H_f);rho_o.assign(rho_f);w_o.assign(w_f);age_o.assign(age_f)
xs=mesh.coordinates.dat.data_ro.reshape(-1); d=H0-xs; o=np.argsort(d)
md, mrho = d[o], rho_f.dat.data_ro[o]
mage = age_f.dat.data_ro[o]/YEAR_S

# ---- residuals ----
obs_d = dens.depth_m.values; obs_r = dens.rho_kgm3.values*1000.0
sig_r = 15.0 + 0.03*obs_r
mod_r = np.interp(obs_d, md, mrho)
mod_a_at_r = np.interp(obs_d, md, mage)
res_r = (obs_r - mod_r)/sig_r
depo_r = 2015.0 - mod_a_at_r

obs_ad = aged.depth_m.values; obs_a = aged.age_yr.values
sig_a = 10.0 + 0.03*obs_a
mod_a = np.interp(obs_ad, md, mage)
res_a = (obs_a - mod_a)/sig_a
depo_a = 2015.0 - mod_a

def binmed(x, y, edges):
    c, v = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() >= 3: c.append(0.5*(lo+hi)); v.append(np.median(y[s]))
    return np.array(c), np.array(v)

yr_edges = np.arange(800, 2025, 25.0)
cyr, ryr = binmed(depo_r, res_r, yr_edges)
cya, aya = binmed(depo_a, res_a, yr_edges)
rho_edges = np.arange(360, 860, 20.0)
crho, rrho = binmed(mod_r, res_r, rho_edges)

# Buizert accumulation anomaly on the same year grid
b_yr, b_ac = buiz.year_CE.values, buiz.accum.values
b_on = np.interp(cyr, b_yr[::-1], b_ac[::-1])   # file is BP-ordered (1950 first)
b_anom = (b_on - BDOT)/BDOT
valid = (cyr >= b_yr.min()) & (cyr <= b_yr.max())
cc = np.corrcoef(ryr[valid], b_anom[valid])[0, 1] if valid.sum() > 3 else np.nan
print(f"corr(binned density residual, Buizert accum anomaly) = {cc:+.2f}  (n={valid.sum()} bins)")
print(f"Buizert accum over record: {b_ac.min():.3f}-{b_ac.max():.3f} m/yr vs model constant {BDOT}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, AX = plt.subplots(2, 2, figsize=(13, 9))
ax = AX[0,0]
ax.plot(res_r, obs_d, "k.", ms=3, alpha=0.5)
ax.axvline(0, color="k", lw=0.8); ax.invert_yaxis()
ax.set_title("(a) Density residuals vs depth"); ax.set_xlabel("(obs − model)/σ"); ax.set_ylabel("depth (m)")
ax.grid(alpha=0.3)

ax = AX[0,1]
ax.plot(res_r, mod_r, "k.", ms=3, alpha=0.4)
ax.plot(rrho, crho, "C3-o", ms=4, lw=2, label="25 kg/m³-bin median")
ax.axvline(0, color="k", lw=0.8); ax.invert_yaxis()
ax.axhline(550, color="C0", ls=":", lw=1.5, label="stage transition ρ_m")
ax.set_title("(b) LAW VIEW: density residuals vs model ρ"); ax.set_xlabel("(obs − model)/σ")
ax.set_ylabel("model density (kg m⁻³)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = AX[1,0]
ax.plot(depo_r, res_r, "k.", ms=3, alpha=0.35)
ax.plot(cyr, ryr, "C3-o", ms=4, lw=2, label="25-yr bin median (density)")
ax.plot(cya, aya, "C0-s", ms=4, lw=1.6, label="25-yr bin median (age)")
axt = ax.twinx()
bshow = b_yr >= 780.0   # file extends to 40 kyr BP; show the plotted window only
axt.plot(b_yr[bshow], (b_ac[bshow]-BDOT)/BDOT, "C2-", lw=1.2, alpha=0.8)
axt.set_ylabel("Buizert accum anomaly (frac vs 0.085)", color="C2")
axt.tick_params(axis="y", colors="C2")
ax.set_xlim(800, 2020)
ax.axhline(0, color="k", lw=0.8)
ax.set_title(f"(c) FORCING VIEW: residuals vs deposition year  (corr ρ-res↔accum: {cc:+.2f})")
ax.set_xlabel("deposition year CE (via model age)"); ax.set_ylabel("(obs − model)/σ")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

ax = AX[1,1]
ax.plot(depo_a, res_a, "C0.", ms=3, alpha=0.5)
ax.plot(cya, aya, "C0-s", ms=4, lw=2)
ax.axhline(0, color="k", lw=0.8)
ax.set_title("(d) Age residuals vs deposition year"); ax.set_xlabel("deposition year CE")
ax.set_ylabel("(obs − model)/σ"); ax.grid(alpha=0.3)

bdesc = "b(t) from MAP" if "b_knots" in JM else f"constant bdot = {BDOT} m ice/yr"
fig.suptitle(f"Mid-record misfit diagnosis — {Path(MAP_FILE).stem} MAP ({bdesc})", fontsize=12)
fig.tight_layout()
fig.savefig(OUT/f"sp_midrecord_diagnostic{SUFFIX}.png", dpi=120)
print(f"Saved {OUT/f'sp_midrecord_diagnostic{SUFFIX}.png'}")
