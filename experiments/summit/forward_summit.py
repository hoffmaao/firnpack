"""forward_summit.py — transferability test: does the South-Pole-calibrated firn
law predict Summit, Greenland?

Takes the frozen South Pole densification + conductivity law (results/
sp_joint_r8.json) — physical H&L Arrhenius rates + Calonne conductivity, which
are temperature-independent constants — and runs the forward model under
SUMMIT forcing (T ~ -28.8 C, accumulation ~ 0.246 m ice/yr; 3x warmer-and-
wetter than SP). Compares the prediction to Summit observations:
  * depth-age  vs GISP2 layer-counted timescale
  * firn temperature vs FirnCover RTD string (0.5-11.6 m)
  * density vs a Summit density profile (if experiments/summit/data/summit_density.csv exists)

This is the paper's Summit transferability figure in its simplest form (constant
forcing, steady state). A Summit INVERSION (recover Summit's own law) is the next
step. No adjoint — forward only.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> experiments/summit/forward_summit.py
Env: FIRN_SP_MAP (default sp_joint_r8.json), FIRN_SUMMIT_T_C, FIRN_SUMMIT_B,
     FIRN_SPIN_YEARS (default 1200), FIRN_H (default 130).
"""
from __future__ import annotations
import functools, json, math, os
from pathlib import Path
import numpy as np, pandas as pd
import firedrake as fd
from firedrake.adjoint import stop_annotating
from firn.models.firn import FirnParameters, FirnModel
from firn.physics.densification import herron_langway as _hl
from firn.constants import year as YEAR_S
herron_langway = functools.partial(_hl, smooth=True)

HERE = Path(__file__).parent
DATA = HERE / "data"
SP_RESULTS = HERE.parent.parent / "test/southpole/results"   # frozen SP MAP lives here
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
SP_MAP = os.environ.get("FIRN_SP_MAP", "sp_joint_r8.json")

# ---- Summit forcing (constant, steady-state; from staged data) ----
SUMMIT_T_C = float(os.environ.get("FIRN_SUMMIT_T_C", "-28.8"))   # FirnCover deep firn T
SUMMIT_B = float(os.environ.get("FIRN_SUMMIT_B", "0.246"))       # Osman 2021 mean m ice/yr
H0 = float(os.environ.get("FIRN_H", "130"))
NZ, STRETCH_P, SID = 120, 2.5, 2
RHO_SURF, DT_YEARS = 350.0, 5.0
SPIN_YEARS = float(os.environ.get("FIRN_SPIN_YEARS", "1200"))
K_ICE_BASE, BETA = 2.1, 5.0e2

# ---- transferred SP law ----
J = json.load(open(SP_RESULTS / SP_MAP)); m = J["m_map"]
P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)
_pk = dict(hl_k0_prefactor=m["hl_k0"], hl_k1_prefactor=m["hl_k1"],
           hl_Ea_stage1=m["hl_Ea1"], hl_Ea_stage2=m["hl_Ea2"],
           basal_heat_flux_W_m2=m["Q_base"], hl_stage2_shape=float(m.get("s2_shape", 1.0)))
if "k_snow_scale" in m:
    _pk.update(conductivity_law="calonne2019", k_snow_scale=float(m["k_snow_scale"]),
               k_firn_scale=float(m["k_firn_scale"]))
else:
    _pk.update(k_ice=K_ICE_BASE * m["k_factor"])
params = FirnParameters(**_pk)
model = FirnModel(params, densification_rate_fn=herron_langway)
print("="*64)
print(f"SUMMIT forward transferability test — SP law from {SP_MAP}")
print(f"  law: k0={m['hl_k0']:.2f} k1={m['hl_k1']:.0f} Ea1={m['hl_Ea1']:.0f} "
      f"Ea2={m['hl_Ea2']:.0f} s2={m.get('s2_shape',1):.3f}")
print(f"  Summit forcing: T={SUMMIT_T_C} C, b={SUMMIT_B} m ice/yr, H={H0} m, spin={SPIN_YEARS} yr")
print("="*64)

# ---- mesh + stepper (self-contained forward; pattern from sp_fit_stats.py) ----
mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0*(1.0-(1.0-fd.SpatialCoordinate(mesh)[0])**STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); Rs = fd.FunctionSpace(mesh, "R", 0)
dx = fd.dx(domain=mesh); ds = fd.ds(domain=mesh)
depth = H0-fd.SpatialCoordinate(mesh)[0]; psi = fd.TestFunction(V)
def mk(v, n): f = fd.Function(Rs, name=n); f.assign(float(v)); return f
n_steps = int(SPIN_YEARS/DT_YEARS)
ws0 = -SUMMIT_B*rho_i/RHO_SURF/YEAR_S
H_f=fd.Function(V); rho_f=fd.Function(V); w_f=fd.Function(V); age_f=fd.Function(V)
H_o=fd.Function(V); rho_o=fd.Function(V); w_o=fd.Function(V); age_o=fd.Function(V); Ts=fd.Function(V)
ac=mk(SUMMIT_B,"ac"); rs=mk(RHO_SURF,"rs"); ws=mk(ws0,"ws"); dt_r=mk(DT_YEARS*YEAR_S,"dt")
bcr=fd.DirichletBC(V,rs,SID); bcw=fd.DirichletBC(V,ws,SID); bca=fd.DirichletBC(V,mk(0.0,"a0"),SID)
bm=ac*rho_i/P0.spy
Ht=fd.TrialFunction(V); rt=fd.TrialFunction(V); wt=fd.TrialFunction(V); at=fd.TrialFunction(V)
sp={"ksp_type":"preonly","pc_type":"lu"}
FH=model.enthalpy_form(Ht,H_o,rho_f,w_f,psi,dt_r)+fd.Constant(BETA)*(Ht-c_i*(Ts-fd.Constant(T_ref)))*psi*ds(SID)
sH=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(FH),fd.rhs(FH),H_f,constant_jacobian=False),solver_parameters=sp)
Te=model.temperature_from_enthalpy(H_f)
Fr,_=model.density_form(rt,rho_o,Te,w_f,None,bm,psi,dt_r)
sR=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(Fr),fd.rhs(Fr),rho_f,bcs=[bcr],constant_jacobian=False),solver_parameters=sp)
dr=herron_langway(rho_f,Te,params=params,bdot=bm)
dw=model.velocity_delta(wt,w_o,rho_f,dr,psi,regularization=1e-3)
sW=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(dw),fd.rhs(dw),w_f,bcs=[bcw],constant_jacobian=False),solver_parameters=sp)
Fa=model.age_form(at,age_o,w_f,w_o,psi,dt_r)
sA=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(Fa),fd.rhs(Fa),age_f,bcs=[bca],constant_jacobian=False),solver_parameters=sp)
with stop_annotating():
    H_f.assign(c_i*(SUMMIT_T_C+273.15-T_ref)); H_o.assign(H_f)
    rho_f.interpolate(RHO_SURF+(830.0-RHO_SURF)*(1.0-fd.exp(-depth/15.0))); rho_o.assign(rho_f)
    w_f.assign(ws0); w_o.assign(w_f); age_f.assign(0.0); age_o.assign(0.0)
    Ts.assign(SUMMIT_T_C+273.15)
    for k in range(n_steps):
        sH.solve(); sR.solve(); sW.solve(); sA.solve()
        H_o.assign(H_f); rho_o.assign(rho_f); w_o.assign(w_f); age_o.assign(age_f)
xs=mesh.coordinates.dat.data_ro.reshape(-1); dm=H0-xs; o=np.argsort(dm)
md=dm[o]; mrho=rho_f.dat.data_ro[o]; mage=age_f.dat.data_ro[o]/YEAR_S
mT=model.temperature_from_enthalpy(H_f); mTc=fd.Function(V).interpolate(mT).dat.data_ro[o]-273.15
def at_depth(v, z): return float(np.interp(z, md, v))
z_co = float(np.interp(830.0, mrho, md)) if mrho.max()>830 else float("nan")
print(f"predicted: rho(10m)={at_depth(mrho,10):.0f}, rho(30m)={at_depth(mrho,30):.0f}, "
      f"pore close-off (830) at {z_co:.1f} m; age(100m)={at_depth(mage,100):.0f} yr")

# ---- observations ----
gis = pd.read_csv(DATA/"gisp2_depth_age.csv"); gis["age"]=1950.0-gis.year_CE  # yr BP-ish
ftC = pd.read_csv(DATA/"firncover_summit_firn_T.csv")
dens_path = DATA/"summit_density.csv"
has_dens = dens_path.exists()
if has_dens:
    dens = pd.read_csv(dens_path)
    rho_col = [c for c in dens.columns if "rho" in c.lower() or "dens" in c.lower()][0]
    dpt = dens.depth_m.values; dro = dens[rho_col].values
    if np.nanmax(dro) < 5: dro = dro*1000.0
    age_rms = None

# ---- misfit summary ----
obs_age_d = gis.depth_m.values; obs_age = (1950.0-gis.year_CE.values)   # yr before 1950
# model age is yr since deposition = present age; compare on depth
mod_age_at = np.interp(obs_age_d, md, mage)
sel = obs_age_d <= H0
age_rms_yr = float(np.sqrt(np.nanmean((mod_age_at[sel]-obs_age[sel])**2)))
obs_T_d = ftC.depth_m.values; mod_T_at = np.interp(obs_T_d, md, mTc)
T_rms = float(np.sqrt(np.nanmean((mod_T_at-ftC.T_C.values)**2)))
print(f"MISFIT vs Summit obs:  age RMS = {age_rms_yr:.1f} yr (over 0-{H0:.0f} m),  "
      f"firn-T RMS = {T_rms:.2f} C" + (f",  density available" if has_dens else ",  density MISSING"))

# ---- figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C_M,C_O="#2563EB","#111827"
n=3 if has_dens else 2
fig,AX=plt.subplots(1,n,figsize=(5*n,5.4))
ax=AX[0]
ax.plot(mage,md,"-",color=C_M,lw=2,label="SP-law prediction")
ax.plot(obs_age,obs_age_d,".",color=C_O,ms=3,label="GISP2 layer-counted")
ax.invert_yaxis(); ax.set_ylim(H0,0); ax.set_xlabel("age (yr before 1950)"); ax.set_ylabel("depth (m)")
ax.set_title(f"(a) depth-age (RMS {age_rms_yr:.0f} yr)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax=AX[1]
ax.plot(mTc,md,"-",color=C_M,lw=2,label="prediction")
ax.plot(ftC.T_C,ftC.depth_m,"o",color=C_O,ms=4,label="FirnCover RTD")
ax.invert_yaxis(); ax.set_ylim(20,0); ax.set_xlabel("temperature (C)"); ax.set_ylabel("depth (m)")
ax.set_title(f"(b) firn T (RMS {T_rms:.2f} C)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
if has_dens:
    ax=AX[2]
    ax.plot(mrho,md,"-",color=C_M,lw=2,label="SP-law prediction")
    ax.plot(dro,dpt,".",color=C_O,ms=3,label="Summit density obs")
    ax.axhline(z_co,color="C3",ls=":",lw=1); ax.invert_yaxis(); ax.set_ylim(H0,0)
    ax.set_xlabel("density (kg/m3)"); ax.set_ylabel("depth (m)")
    ax.set_title("(c) density"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle(f"Summit transferability — SP-calibrated law under Summit forcing "
             f"(T={SUMMIT_T_C} C, b={SUMMIT_B} m/yr)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"forward_summit.png", dpi=140)
json.dump(dict(sp_map=SP_MAP, summit_T_C=SUMMIT_T_C, summit_b=SUMMIT_B,
               age_rms_yr=age_rms_yr, T_rms_C=T_rms, close_off_depth_m=z_co,
               has_density=bool(has_dens)), open(OUT/"forward_summit.json","w"), indent=1)
print(f"Saved {OUT/'forward_summit.png'} and .json")
