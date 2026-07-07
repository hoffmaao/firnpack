"""sp_fit_stats.py — replay a joint MAP forward and report per-dataset fit stats.

Rebuilds the EXACT observation operators of the joint assimilation scripts
(Gaussian-kernel scalar misfits, same subsampling / sigmas / ApRES binning /
on-tape antenna reference) and runs the MAP forward once, off-tape. Reports
rms per dataset in sigma units and the chi2 terms under BOTH weightings
(per-dataset 1/N_d, as optimized so far, and per-POINT, for the honest-error
round), plus prior z-scores. Recovers the fit table when a run log is lost;
the reconstructed J validates against the JSON's J.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv python> sp_fit_stats.py
Env: FIRN_MAP=sp_joint_shape.json (default; any m_map-style MAP with T/b knots)
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

PROC = Path(__file__).parent.parent.parent / "processed"
OUT = Path(__file__).parent.parent.parent / "results"
MAP_FILE = os.environ.get("FIRN_MAP", "sp_joint_shape.json")
JM = json.load(open(OUT/MAP_FILE)); m = JM["m_map"]
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF, BDOT0, DT_YEARS = 350.0, 0.085, 5.0
SPIN_YEARS = float(os.environ.get("FIRN_SPIN_YEARS", "2500"))
K_ICE_BASE, BETA, N_ICE = 2.1, 5.0e2, math.sqrt(3.18)
P0 = FirnParameters(); c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)

# ---- priors (verbatim from the assimilation scripts) ----
INIT = dict(hl_k0=10.79, hl_k1=570.7, hl_Ea1=10432.0, hl_Ea2=21875.0,
            k_factor=0.5, Q_base=0.0, ezz_yr=0.0, s2_shape=1.0)
LOGS = {"hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "k_factor", "s2_shape"}
PRIOR_SIG = dict(hl_k0=1.0, hl_k1=1.0, hl_Ea1=0.5, hl_Ea2=0.5, k_factor=1.0,
                 Q_base=0.05, ezz_yr=1.0e-4, s2_shape=0.3)
T_PRIOR_SIG = 0.6
_T6_YEARS = np.array([1000., 1300., 1550., 1750., 1900., 2015.])
_T6_VALS = np.array([-45.50, -45.45, -45.35, -45.20, -45.10, -45.00])

# ---- data (verbatim) ----
def _sub(df, n): return df.iloc[::max(1, len(df)//n)]
dens = _sub(pd.read_csv(PROC/"sp19_density.csv").query("depth_m<=@H0"), 45)
obs_rho_d, obs_rho = dens.depth_m.values, dens.rho_kgm3.values*1000.0
sig_rho = 15.0 + 0.03*obs_rho
age = pd.read_csv(PROC/"sp19_depth_age.csv"); age["age_yr"] = 2015.0-age.year_CE
age = _sub(age.query("depth_m<=@H0 and age_yr>=0"), 40)
obs_age_d, obs_age_s = age.depth_m.values, age.age_yr.values*YEAR_S
sig_age_s = (10.0+0.03*age.age_yr.values)*YEAR_S
# FIRN_T_SHIFT: apply the borehole datum correction (e.g. -5.8 for r5b-style
# MAPs whose T-space is the corrected ~-51 C datum; staged csv is +5.8 off).
T_SHIFT = float(os.environ.get("FIRN_T_SHIFT", "0"))
bT = _sub(pd.read_csv(PROC/"spicecore_borehole_T.csv").query("depth_m<=@H0"), 40)
obs_T_d, obs_T_H = bT.depth_m.values, c_i*(bT.T_C.values+T_SHIFT+273.15-T_ref)
sig_T_H = c_i*np.full(len(obs_T_d), 0.2)
ap = pd.read_csv(PROC/"apres_vertical_velocity_processed.csv")
ap = ap[(ap.range_m<=H0) & ap.v_smooth_m_yr.notna() & (ap.coherence>0.5)]
_vb = np.arange(8.0, H0, 8.0); _vm = 0.5*(_vb[:-1]+_vb[1:]); ap["_b"] = np.digitize(ap.range_m.values, _vb)
_vr = [(_vm[i], ap.v_smooth_m_yr[ap._b==i+1].median(), ap.v_smooth_m_yr[ap._b==i+1].std(),
        int((ap._b==i+1).sum())) for i in range(len(_vm))]
_vr = [r for r in _vr if r[3]>=3]
obs_v_d = np.array([r[0] for r in _vr]); obs_v = np.array([r[1] for r in _vr])
sig_v = np.maximum(np.array([r[2]/max(r[3],1)**0.5 for r in _vr]), 0.01)
N_rho, N_age, N_T, N_v = len(obs_rho_d), len(obs_age_d), len(obs_T_d), len(obs_v_d)

# ---- mesh + kernels (verbatim) ----
mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0*(1.0-(1.0-fd.SpatialCoordinate(mesh)[0])**STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1); Rs = fd.FunctionSpace(mesh, "R", 0)
dx = fd.dx(domain=mesh); ds = fd.ds(domain=mesh)
depth = H0-fd.SpatialCoordinate(mesh)[0]; psi = fd.TestFunction(V); xc = fd.SpatialCoordinate(mesh)[0]
def mk(v, n): f = fd.Function(Rs, name=n); f.assign(float(v)); return f
def make_kernels(depths):
    ks = []
    for d in depths:
        s = float(np.clip(0.04*d+0.5, 0.5, 3.0))
        phi = fd.Function(V).interpolate(fd.exp(-0.5*((xc-(H0-d))/s)**2))
        phi.dat.data[:] /= float(fd.assemble(phi*dx))+1e-30
        ks.append(phi)
    return ks
ker_rho, ker_age, ker_T, ker_v = (make_kernels(obs_rho_d), make_kernels(obs_age_d),
                                  make_kernels(obs_T_d), make_kernels(obs_v_d))

# ---- forcing from the MAP ----
n_steps = int(SPIN_YEARS/DT_YEARS)
step_years = 2015.0 - (n_steps-1-np.arange(n_steps))*DT_YEARS
T_KY, T_K = np.asarray(JM["knot_years"]), np.asarray(JM["T_knots"])
T_series = np.interp(step_years, T_KY, T_K)
if "b_knots" in JM:
    B_KY, B_K = np.asarray(JM["b_knot_years"]), np.asarray(JM["b_knots"])
    b_series = np.exp(np.interp(step_years, B_KY, np.log(B_K)))
else:
    b_series = np.full(n_steps, BDOT0)
T_PRIOR_CTR = np.interp(T_KY, _T6_YEARS, _T6_VALS)

_pk = dict(hl_k0_prefactor=mk(m["hl_k0"],"k0"), hl_k1_prefactor=mk(m["hl_k1"],"k1"),
    hl_Ea_stage1=mk(m["hl_Ea1"],"e1"), hl_Ea_stage2=mk(m["hl_Ea2"],"e2"),
    basal_heat_flux_W_m2=mk(m["Q_base"],"Q"),
    hl_stage2_shape=float(m.get("s2_shape", 1.0)))
if "k_snow_scale" in m:   # r5-style MAP: Calonne-2019 conductivity
    _pk.update(conductivity_law="calonne2019",
               k_snow_scale=float(m["k_snow_scale"]),
               k_firn_scale=float(m["k_firn_scale"]))
else:                     # legacy quadratic law
    _pk.update(k_ice=K_ICE_BASE*m["k_factor"])
params = FirnParameters(**_pk)
model = FirnModel(params, densification_rate_fn=herron_langway)
ws0 = -BDOT0*rho_i/RHO_SURF/YEAR_S
H_f=fd.Function(V); rho_f=fd.Function(V); w_f=fd.Function(V); age_f=fd.Function(V)
H_o=fd.Function(V); rho_o=fd.Function(V); w_o=fd.Function(V); age_o=fd.Function(V); Ts=fd.Function(V)
ac=mk(BDOT0,"ac"); rs=mk(RHO_SURF,"rs"); ws=mk(ws0,"ws"); dt_r=mk(DT_YEARS*YEAR_S,"dt")
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
dw=model.velocity_delta(wt,w_o,rho_f,dr,psi,regularization=1e-3,
                        horizontal_divergence=fd.Constant(-float(m.get("ezz_yr",0.0))/YEAR_S))
sW=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(dw),fd.rhs(dw),w_f,bcs=[bcw],constant_jacobian=False),solver_parameters=sp)
Fa=model.age_form(at,age_o,w_f,w_o,psi,dt_r)
sA=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(Fa),fd.rhs(Fa),age_f,bcs=[bca],constant_jacobian=False),solver_parameters=sp)

with stop_annotating():
    T0 = float(T_PRIOR_CTR.mean())+273.15
    H_f.assign(c_i*(T0-T_ref)); H_o.assign(H_f)
    rho_f.interpolate(RHO_SURF+(820.0-RHO_SURF)*(1.0-fd.exp(-depth/25.0))); rho_o.assign(rho_f)
    w_f.assign(ws0); w_o.assign(w_f); age_f.assign(0.0); age_o.assign(0.0)
    for k in range(n_steps):
        Ts.assign(float(T_series[k])+273.15)
        ac.assign(float(b_series[k]))
        ws.assign(-float(b_series[k])*rho_i/RHO_SURF/YEAR_S)
        sH.solve(); sR.solve(); sW.solve(); sA.solve()
        H_o.assign(H_f); rho_o.assign(rho_f); w_o.assign(w_f); age_o.assign(age_f)

    def sq_sum(field, kernels, obs, sig):
        return sum(((float(fd.assemble(field*phi*dx))-float(o))/float(s))**2
                   for phi, o, s in zip(kernels, obs, sig))
    sq_rho = sq_sum(rho_f, ker_rho, obs_rho, sig_rho)
    sq_age = sq_sum(age_f, ker_age, obs_age_s, sig_age_s)
    sq_T = sq_sum(H_f, ker_T, obs_T_H, sig_T_H)
    w_srf = float(fd.assemble(w_f*ds(SID)))
    sq_v = sum((((float(fd.assemble(w_f*phi*dx))-w_srf)/N_ICE*YEAR_S-float(o))/float(s))**2
               for phi, o, s in zip(ker_v, obs_v, sig_v))

# ---- prior z-scores ----
z2_scal, z_scal = 0.0, {}
for n in [n for n in INIT if n in m]:
    x = math.log(m[n]) if n in LOGS else m[n]
    c = math.log(INIT[n]) if n in LOGS else INIT[n]
    z_scal[n] = (x-c)/PRIOR_SIG[n]; z2_scal += z_scal[n]**2
z_T = (T_K - T_PRIOR_CTR)/T_PRIOR_SIG
z2_T = float(np.sum(z_T**2))
if "b_knots" in JM:
    sig_logB = float(JM.get("sig_logB", 0.15))
    b_ctr = np.asarray(JM["b_prior_centers"])
    z_b = (np.log(B_K)-np.log(b_ctr))/sig_logB
    z2_b = float(np.sum(z_b**2))
else:
    z_b = np.array([]); z2_b = 0.0
N_CTRL = len(z_scal) + len(T_K) + len(z_b)

sqs = dict(rho=sq_rho, age=sq_age, T=sq_T, v=sq_v)
Ns = dict(rho=N_rho, age=N_age, T=N_T, v=N_v)
J_data_perds = sum(0.5*sqs[k]/Ns[k] for k in sqs)
J_prior_perds = 0.5*(z2_scal+z2_T+z2_b)/N_CTRL
J_data_pp = sum(0.5*sqs[k] for k in sqs)
J_prior_pp = 0.5*(z2_scal+z2_T+z2_b)

print("="*64)
print(f"FIT STATS @ {MAP_FILE}  ({N_CTRL} ctrls; spinup {SPIN_YEARS:.0f} yr, dt {DT_YEARS})")
print("="*64)
for k in sqs:
    print(f"  {k:4s}: rms={math.sqrt(sqs[k]/Ns[k]):.3f} sigma  (N={Ns[k]}, chi2={sqs[k]:.1f})")
print(f"  per-DATASET weighting: J_data={J_data_perds:.4f} + J_prior={J_prior_perds:.4f}"
      f" = {J_data_perds+J_prior_perds:.4f}  (JSON J={JM['J']:.4f})")
print(f"  per-POINT   weighting: J_data={J_data_pp:.2f} + J_prior={J_prior_pp:.2f}"
      f" = {J_data_pp+J_prior_pp:.2f}")
print("  prior z (scalars): " + " ".join(f"{n}={z_scal[n]:+.2f}" for n in z_scal))
print(f"  prior z: |T| max {np.abs(z_T).max():.2f}, |logb| max {np.abs(z_b).max() if z_b.size else 0:.2f}")
json.dump(dict(map_file=MAP_FILE, rms={k: math.sqrt(sqs[k]/Ns[k]) for k in sqs},
               chi2=sqs, N=Ns, J_data_per_dataset=J_data_perds, J_prior_per_dataset=J_prior_perds,
               J_json=JM["J"], J_data_per_point=J_data_pp, J_prior_per_point=J_prior_pp,
               z_scalars=z_scal, z_T=z_T.tolist(), z_logb=z_b.tolist()),
          open(OUT/f"sp_fit_stats_{Path(MAP_FILE).stem}.json","w"), indent=1)
print(f"Saved {OUT/f'sp_fit_stats_{Path(MAP_FILE).stem}.json'}")
