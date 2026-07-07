"""sp_assimilate.py  —  South Pole firn data assimilation (clean rebuild)

Derive the densification physics by assimilating REAL South Pole data:
  * SP19 density profile
  * SP19 depth-age
  * SPICE borehole temperature
  * (ApRES vertical velocity — added once rho+age+T is validated)

Forward model = Evan Cummings (H, rho, w, age) firn column + Herron-Langway
densification (firn.models.firn + firn.physics.densification).  Verified in
forward_basics_diagnostic.py to reproduce SP19 age + velocity exactly and
density to within a uniform +35..65 kg/m3 high bias at literature params.

This script inverts the H&L parameters (and basal heat flux) to remove that
bias and fit all data jointly.

Design (lessons baked in):
  * CONSTANT forcing at the borehole temperature (-45.5 C) — the firn is
    near-isothermal; -50 C (Buizert cloud temp) over-suppresses densification.
  * SPINUP >= ~2500 yr on tape (transit time ~1200 yr) so the deep obs are
    reachable by the adjoint.  dt=5 yr (validated == dt=1 yr steady state)
    -> ~500 steps on tape.
  * TAPE-REBUILD adjoint: clear + re-annotate the tape every J/grad eval.
    Fixes the pyadjoint replay bug that silently corrupts time-varying tapes.
  * log-space controls + auto gradient scaling + scipy L-BFGS-B.
  * interpolate-then-assemble objective terms (adjoint-safe).

Env:
  FIRN_VERIFY_ONLY=1   stop after reproducibility + Taylor test
  FIRN_MAX_ITER=N      L-BFGS-B iterations (default 40)
  FIRN_SPIN_YEARS=Y    spinup years (default 2500)
"""
from __future__ import annotations
import functools, json, math, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize

import firedrake as fd
from firedrake.adjoint import (
    Control, continue_annotation, pause_annotation, stop_annotating,
    get_working_tape)
from pyadjoint import compute_gradient
from firn.models.firn import FirnParameters, FirnModel
from firn.solvers.firn_solver import FirnColumnSolver
from firn.physics.densification import herron_langway as _hl
from firn.constants import year as YEAR_S

herron_langway = functools.partial(_hl, smooth=True)
sys.stdout.reconfigure(line_buffering=True)

_HERE = Path(__file__).parent
PROC = _HERE.parent.parent / "processed"
OUT = _HERE.parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
TAG = "sp_assim_hl_rho_age_T"

# ---------------- configuration ----------------
H0, NZ, STRETCH_P, SID = 130.0, 100, 2.5, 2
RHO_SURF = 350.0
T_SURF_C = -45.5
BDOT = 0.085
DT_YEARS = 5.0
SPIN_YEARS = float(os.environ.get("FIRN_SPIN_YEARS", "2500"))
MAX_ITER = int(os.environ.get("FIRN_MAX_ITER", "40"))
VERIFY_ONLY = os.environ.get("FIRN_VERIFY_ONLY", "0") == "1"

# controls: literature H&L + basal heat flux
INITIAL = dict(hl_k0=11.0, hl_k1=575.0, hl_Ea1=10160.0, hl_Ea2=21400.0,
               Q_base=0.0)
BOUNDS = dict(hl_k0=(0.5, 500.0), hl_k1=(10.0, 50000.0),
              hl_Ea1=(3000.0, 40000.0), hl_Ea2=(5000.0, 80000.0),
              Q_base=(-0.2, 0.2))
LOG = {"hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2"}
PRIOR_SIG = dict(hl_k0=1.0, hl_k1=1.0, hl_Ea1=0.5, hl_Ea2=0.5, Q_base=0.05)
NAMES = ["hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "Q_base"]
N_CTRL = len(NAMES)

# data weights (env-configurable for isolation tests)
W_RHO = float(os.environ.get("FIRN_W_RHO", "1.0"))
W_AGE = float(os.environ.get("FIRN_W_AGE", "1.0"))
W_T = float(os.environ.get("FIRN_W_T", "1.0"))
W_V = float(os.environ.get("FIRN_W_V", "0.0"))   # ApRES velocity (off until rho+age+T validated)

print("=" * 64)
print(f"SP ASSIMILATION: H&L, rho+age+T, constant T={T_SURF_C}C")
print(f"  spinup={SPIN_YEARS:.0f} yr @ dt={DT_YEARS} yr = "
      f"{int(SPIN_YEARS/DT_YEARS)} steps, max_iter={MAX_ITER}")
print("=" * 64)

def mk(R, v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f

# ---------------- load data ----------------
P0 = FirnParameters()
c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)

def _sub(df, n):
    return df.iloc[:: max(1, len(df) // n)]

dens = pd.read_csv(PROC / "sp19_density.csv"); dens = dens[dens.depth_m <= H0]
dens = _sub(dens, 45)
obs_rho_d = dens.depth_m.values
obs_rho = dens.rho_kgm3.values * 1000.0
sig_rho = 15.0 + 0.03 * obs_rho                      # kg/m3

age = pd.read_csv(PROC / "sp19_depth_age.csv")
age["age_yr"] = 2015.0 - age.year_CE
age = age[(age.depth_m <= H0) & (age.age_yr >= 0)]
age = _sub(age, 40)
obs_age_d = age.depth_m.values
obs_age_s = age.age_yr.values * YEAR_S
sig_age_s = (10.0 + 0.03 * age.age_yr.values) * YEAR_S  # yr -> s (loose enough)

bT = pd.read_csv(PROC / "spicecore_borehole_T.csv"); bT = bT[bT.depth_m <= H0]
bT = _sub(bT, 30)
obs_T_d = bT.depth_m.values
obs_T_H = c_i * (bT.T_C.values + 273.15 - T_ref)        # enthalpy
sig_T_H = c_i * np.full(len(obs_T_d), 2.0)              # 2 C structural

# ApRES vertical velocity (binned across sites).  Forward operator:
#   v_apres(z) = (w(z) - w_surf) / n_ice    [m/yr]   (dR/dt compaction memo)
N_ICE = math.sqrt(3.18)
ap = pd.read_csv(PROC / "apres_vertical_velocity_processed.csv")
ap = ap[(ap.range_m <= H0) & ap.v_smooth_m_yr.notna() & (ap.coherence > 0.5)]
_vb = np.arange(8.0, H0, 8.0)
_vm = 0.5 * (_vb[:-1] + _vb[1:])
ap["_b"] = np.digitize(ap.range_m.values, _vb)
_vrows = [(_vm[i], ap.v_smooth_m_yr[ap._b == i + 1].median(),
           ap.v_smooth_m_yr[ap._b == i + 1].std(), int((ap._b == i + 1).sum()))
          for i in range(len(_vm))]
_vrows = [r for r in _vrows if r[3] >= 3]
obs_v_d = np.array([r[0] for r in _vrows])
obs_v = np.array([r[1] for r in _vrows])                 # m/yr
sig_v = np.maximum(np.array([r[2] / max(r[3], 1) ** 0.5 for r in _vrows]), 0.01)

N_rho, N_age, N_T, N_v = len(obs_rho_d), len(obs_age_d), len(obs_T_d), len(obs_v_d)
print(f"  obs: {N_rho} rho, {N_age} age, {N_T} T, {N_v} velocity")

# ---------------- mesh / spaces ----------------
mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xi = fd.SpatialCoordinate(mesh)[0]
x_phys = H0 * (1.0 - (1.0 - xi) ** STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space())
                        .interpolate(fd.as_vector([x_phys])))
V = fd.FunctionSpace(mesh, "CG", 1)
R = fd.FunctionSpace(mesh, "R", 0)
dx = fd.dx(domain=mesh)
depth = H0 - fd.SpatialCoordinate(mesh)[0]

# ---------------- observation operators (Gaussian kernels, off-tape) ----------------
# Memory rule: VOM point-sampling and direct r*r assembly BOTH break the
# adjoint (order-1 Taylor).  Use normalised Gaussian kernels and compute each
# prediction as a SCALAR linear assemble  pred_i = assemble(field * phi_i * dx)
# so the objective is scalar AdjFloat arithmetic (adjoint-safe, order-2).
xcoord = fd.SpatialCoordinate(mesh)[0]

def make_kernels(depths):
    """List of normalised CG1 Gaussian kernels centred at each obs depth."""
    ks = []
    with stop_annotating():
        for d in depths:
            xi_c = H0 - float(np.clip(d, 0.0, H0))
            s = float(np.clip(0.04 * d + 0.5, 0.5, 3.0))   # wider at depth
            phi = fd.Function(V)
            phi.interpolate(fd.exp(-0.5 * ((xcoord - xi_c) / s) ** 2))
            nrm = float(fd.assemble(phi * dx)) + 1e-30
            phi.dat.data[:] /= nrm
            ks.append(phi)
    return ks

with stop_annotating():
    dlen = float(fd.assemble(fd.Constant(1.0) * dx)) + 1e-30
ker_rho = make_kernels(obs_rho_d)
ker_age = make_kernels(obs_age_d)
ker_T = make_kernels(obs_T_d)
ker_v = make_kernels(obs_v_d)
_prior = fd.Function(V, name="prior")

def misfit(field, kernels, obs, sig, weight, Nrm):
    """Scalar chi-square misfit via Gaussian-kernel predictions (adjoint-safe)."""
    J = 0.0
    sq = 0.0
    for phi, o, s in zip(kernels, obs, sig):
        pred = fd.assemble(field * phi * dx)
        r = (pred - float(o)) / float(s)
        J = J + r * r
        with stop_annotating():
            sq += float(r) ** 2
    misfit._last = (sq / len(kernels)) ** 0.5   # RMS residual in sigma units
    return weight * 0.5 / Nrm * J

def misfit_vel(w_field, w_surf_val):
    """ApRES velocity misfit: v_pred(z) = (w(z) - w_surf)/n_ice  [m/yr]."""
    J = 0.0; sq = 0.0
    for phi, o, s in zip(ker_v, obs_v, sig_v):
        wbar = fd.assemble(w_field * phi * dx)              # m/s
        pred = (wbar - w_surf_val) / N_ICE * YEAR_S         # m/yr
        r = (pred - float(o)) / float(s)
        J = J + r * r
        with stop_annotating():
            sq += float(r) ** 2
    misfit_vel._last = (sq / max(len(ker_v), 1)) ** 0.5
    return W_V * 0.5 / N_v * J

# ---------------- state ----------------
H_f = fd.Function(V, name="H"); rho_f = fd.Function(V, name="rho")
w_f = fd.Function(V, name="w"); age_f = fd.Function(V, name="age")

# ---------------- controls ----------------
log_k0 = mk(R, math.log(INITIAL["hl_k0"]), "lk0")
log_k1 = mk(R, math.log(INITIAL["hl_k1"]), "lk1")
log_Ea1 = mk(R, math.log(INITIAL["hl_Ea1"]), "lEa1")
log_Ea2 = mk(R, math.log(INITIAL["hl_Ea2"]), "lEa2")
Q_base = mk(R, INITIAL["Q_base"], "Qb")
ctrl_fns = [log_k0, log_k1, log_Ea1, log_Ea2, Q_base]

x0 = np.array([float(c.dat.data_ro[0]) for c in ctrl_fns])
lb = np.array([math.log(BOUNDS[n][0]) if n in LOG else BOUNDS[n][0] for n in NAMES])
ub = np.array([math.log(BOUNDS[n][1]) if n in LOG else BOUNDS[n][1] for n in NAMES])

# ---------------- model / solver / BCs ----------------
params = FirnParameters(
    hl_k0_prefactor=fd.exp(log_k0), hl_k1_prefactor=fd.exp(log_k1),
    hl_Ea_stage1=fd.exp(log_Ea1), hl_Ea_stage2=fd.exp(log_Ea2),
    basal_heat_flux_W_m2=Q_base)
model = FirnModel(params, densification_rate_fn=herron_langway)
solver = FirnColumnSolver(model, surface_id=SID)

T_surf_K = T_SURF_C + 273.15
ws0 = -BDOT * rho_i / RHO_SURF / YEAR_S
Ts = mk(R, T_surf_K, "Ts")
Hs = mk(R, c_i * (T_surf_K - T_ref), "Hs")
ac = mk(R, BDOT, "ac")
rs = mk(R, RHO_SURF, "rs")
ws = mk(R, ws0, "ws")
dt_r = mk(R, DT_YEARS * YEAR_S, "dt")
bc_H = fd.DirichletBC(V, Hs, SID)
bc_rho = fd.DirichletBC(V, rs, SID)
bc_w = fd.DirichletBC(V, ws, SID)
bc_age = fd.DirichletBC(V, mk(R, 0.0, "a0"), SID)

n_steps = int(SPIN_YEARS / DT_YEARS)

# ---------------- forward ----------------
RHO_IC_CAP = 820.0   # cap initial density well below rho_i so the density_form
                     # soft barrier max_value(rho-rho_i,0) never activates (its
                     # kink makes J non-differentiable -> breaks the adjoint).
def forward():
    H_f.assign(c_i * (T_surf_K - T_ref))
    rho_f.interpolate(RHO_SURF + (RHO_IC_CAP - RHO_SURF) * (1.0 - fd.exp(-depth / 25.0)))
    w_f.assign(ws0)
    age_f.assign(0.0)

    for _ in range(n_steps):
        solver.prognostic_solve(
            enthalpy=H_f, density=rho_f, firn_velocity=w_f, dt=dt_r,
            accumulation=ac, surface_density=rs,
            boundary_conditions=[bc_H, bc_rho, bc_w],
            surface_temperature=Ts, enthalpy_bc_constant=Hs,
            age=age_f, age_boundary_condition=bc_age)

    J_rho = misfit(rho_f, ker_rho, obs_rho, sig_rho, W_RHO, N_rho)
    rms_rho = misfit._last
    J_age = misfit(age_f, ker_age, obs_age_s, sig_age_s, W_AGE, N_age)
    rms_age = misfit._last
    J_T = misfit(H_f, ker_T, obs_T_H, sig_T_H, W_T, N_T)
    rms_T = misfit._last
    J_v = misfit_vel(w_f, ws0) if W_V > 0 else 0.0
    rms_v = misfit_vel._last if W_V > 0 else 0.0
    J = J_rho + J_age + J_T + J_v

    J_prior = 0.0
    for c, nm in zip(ctrl_fns, NAMES):
        ref = math.log(float(INITIAL[nm])) if nm in LOG else float(INITIAL[nm])
        _prior.interpolate((c - fd.Constant(ref)) / fd.Constant(PRIOR_SIG[nm]))
        J_prior = J_prior + (1.0 / N_CTRL) * 0.5 * fd.assemble(_prior * _prior * dx) / dlen
    J = J + J_prior

    with stop_annotating():
        forward._diag = dict(
            J_rho=float(J_rho), J_age=float(J_age), J_T=float(J_T),
            J_v=float(J_v), J_prior=float(J_prior),
            rms_rho=rms_rho, rms_age=rms_age, rms_T=rms_T, rms_v=rms_v,
            rho_max=float(rho_f.dat.data_ro.max()),
            age_base=float(age_f.dat.data_ro.max()) / YEAR_S)
    return J

# ---------------- tape-rebuild eval ----------------
tape = get_working_tape()
n_eval = [0]; J_hist = []

def eval_J_and_grad(xs):
    x = np.clip(x0 + xs * scale, lb, ub)
    t0 = time.perf_counter()
    tape.clear_tape(); continue_annotation()
    for c, v in zip(ctrl_fns, x):
        c.assign(float(v))
    try:
        J = forward()
        controls = [Control(c) for c in ctrl_fns]
        dJ = compute_gradient(J, controls)
        J_val = float(J)
        grad_raw = np.array([float(g.dat.data_ro[0]) * dlen for g in dJ])
        grad_s = grad_raw * scale
    except Exception as e:
        print(f"  *** eval failed: {e}")
        J_val = 1e8; grad_s = xs * 100.0
    pause_annotation()
    n_eval[0] += 1; J_hist.append(J_val)
    d = forward._diag
    mp = {n: (math.exp(x[i]) if n in LOG else x[i]) for i, n in enumerate(NAMES)}
    print(f"  [{n_eval[0]:03d}] J={J_val:.4f} (rho={d['J_rho']:.2f} "
          f"age={d['J_age']:.2f} T={d['J_T']:.2f} v={d['J_v']:.2f} pri={d['J_prior']:.3f}) "
          f"rho_max={d['rho_max']:.0f} age_b={d['age_base']:.0f} "
          f"|g|={np.linalg.norm(grad_s):.1e} ({time.perf_counter()-t0:.0f}s)")
    print(f"        k0={mp['hl_k0']:.2f} k1={mp['hl_k1']:.1f} "
          f"Ea1={mp['hl_Ea1']:.0f} Ea2={mp['hl_Ea2']:.0f} Q={mp['Q_base']:.4f}")
    return J_val, grad_s

# ---------------- reproducibility ----------------
scale = np.ones(N_CTRL)
print("\nReproducibility (two identical forward runs):")
def _run_once():
    tape.clear_tape(); continue_annotation()
    for c, v in zip(ctrl_fns, x0): c.assign(float(v))
    J = float(forward()); pause_annotation(); return J
J1 = _run_once(); J2 = _run_once()
print(f"  J1={J1:.8f}  J2={J2:.8f}  match={abs(J1-J2)<1e-10}")
d = forward._diag
print(f"  J: rho={d['J_rho']:.3f} age={d['J_age']:.3f} T={d['J_T']:.3f}  "
      f"(RMS in sigma: rho={d['rms_rho']:.2f} age={d['rms_age']:.2f} T={d['rms_T']:.2f})")
print(f"  rho_max={d['rho_max']:.0f}  age_base={d['age_base']:.0f}")

# ---------------- Taylor test ----------------
print("\nTaylor test:")
tape.clear_tape(); continue_annotation()
for c, v in zip(ctrl_fns, x0): c.assign(float(v))
J0f = forward()
dJ0 = compute_gradient(J0f, [Control(c) for c in ctrl_fns])
pause_annotation()
J0 = float(J0f)
g = np.array([float(gi.dat.data_ro[0]) * dlen for gi in dJ0])
print(f"  J0={J0:.6f} |g|={np.linalg.norm(g):.3e}")
print("  per-ctrl dJ: " + " ".join(f"{n}={g[i]:.2e}" for i, n in enumerate(NAMES)))
rng = np.random.default_rng(0)
dd = rng.standard_normal(N_CTRL) * 0.01
gd = g @ dd
prev = None
for eps in [0.1, 0.05, 0.025, 0.0125]:
    xp = np.clip(x0 + eps * dd, lb, ub)
    tape.clear_tape(); continue_annotation()
    for c, v in zip(ctrl_fns, xp): c.assign(float(v))
    J1 = float(forward()); pause_annotation()
    r = abs(J1 - J0 - eps * gd)
    o = f"order={math.log(prev/r)/math.log(2):.2f}" if prev and r > 0 else ""
    print(f"  eps={eps:.4f} rem={r:.3e} {o}")
    prev = r

if os.environ.get("FIRN_FD_CHECK", "0") == "1":
    print("\nFinite-difference gradient check (central, h=1e-2):")
    h = 1e-2
    def Jat(xv):
        tape.clear_tape(); continue_annotation()
        for c, v in zip(ctrl_fns, xv): c.assign(float(v))
        Jv = float(forward()); pause_annotation(); return Jv
    for i, nm in enumerate(NAMES):
        ep = x0.copy(); ep[i] += h
        em = x0.copy(); em[i] -= h
        fd_i = (Jat(ep) - Jat(em)) / (2 * h)
        ratio = g[i] / fd_i if abs(fd_i) > 1e-12 else float('nan')
        print(f"  {nm:10s} adjoint={g[i]:+.4e}  FD={fd_i:+.4e}  ratio={ratio:+.4f}")

if VERIFY_ONLY:
    print("\nVERIFY_ONLY set — stopping before optimization.")
    sys.exit(0)

# ---------------- optimize ----------------
scale = 1.0 / np.maximum(np.abs(g), 1e-8)     # auto-scale: |dJ/dx_s| ~ 1
lb_s = (lb - x0) / scale; ub_s = (ub - x0) / scale
print(f"\nL-BFGS-B (auto-scaled, max {MAX_ITER}):")
n_eval[0] = 0; J_hist.clear()
res = sp_minimize(eval_J_and_grad, np.zeros(N_CTRL), jac=True, method="L-BFGS-B",
                  bounds=list(zip(lb_s, ub_s)),
                  options={"maxiter": MAX_ITER, "ftol": 1e-8, "gtol": 1e-7,
                           "maxfun": MAX_ITER * 3, "maxls": 30})
x_map = x0 + res.x * scale
m_map = {n: (math.exp(x_map[i]) if n in LOG else x_map[i]) for i, n in enumerate(NAMES)}
print(f"\n{res.message}\n  {res.nit} iters, {res.nfev} evals, J_map={res.fun:.4f}")
print("MAP:")
for n in NAMES:
    print(f"  {n:10s} = {m_map[n]:.4e}  (init {INITIAL[n]:.4e})")

with open(OUT / f"{TAG}.json", "w") as f:
    json.dump(dict(J_map=res.fun, J_hist=J_hist, m_map=m_map, INITIAL=INITIAL,
                   T_SURF_C=T_SURF_C, BDOT=BDOT, SPIN_YEARS=SPIN_YEARS,
                   message=str(res.message), nit=int(res.nit)), f, indent=2)
print(f"Saved {OUT / (TAG + '.json')}")

# ---------------- profiles + plot (initial vs MAP vs data) ----------------
def run_and_profiles(xvals):
    with stop_annotating():
        for c, v in zip(ctrl_fns, xvals): c.assign(float(v))
        forward()
        xs = mesh.coordinates.dat.data_ro.reshape(-1)
        d = H0 - xs; o = np.argsort(d)
        Tp = fd.Function(V).interpolate(H_f / c_i + T_ref)
        return dict(d=d[o], rho=rho_f.dat.data_ro[o], age=age_f.dat.data_ro[o] / YEAR_S,
                    w=w_f.dat.data_ro[o] * YEAR_S, T=Tp.dat.data_ro[o] - 273.15)

pi = run_and_profiles(x0)
pm = run_and_profiles(x_map)
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 7), sharey=True)
    ax[0].plot(obs_rho, obs_rho_d, "k.", ms=4, label="SP19")
    ax[0].plot(pi["rho"], pi["d"], "C1--", lw=1.5, label="initial")
    ax[0].plot(pm["rho"], pm["d"], "C0-", lw=2, label="MAP")
    ax[0].set_xlabel("density (kg/m³)"); ax[0].set_ylabel("depth (m)"); ax[0].set_title("Density")
    ax[1].plot(obs_age_s / YEAR_S, obs_age_d, "k.", ms=4, label="SP19")
    ax[1].plot(pi["age"], pi["d"], "C1--", lw=1.5); ax[1].plot(pm["age"], pm["d"], "C0-", lw=2)
    ax[1].set_xlabel("age (yr)"); ax[1].set_title("Age")
    ws0_yr = ws0 * YEAR_S
    ax[2].plot(obs_v, obs_v_d, "k.", ms=5, label="ApRES")
    ax[2].plot((pi["w"] - ws0_yr) / N_ICE, pi["d"], "C1--", lw=1.5)
    ax[2].plot((pm["w"] - ws0_yr) / N_ICE, pm["d"], "C0-", lw=2)
    ax[2].set_xlabel("(w-w_s)/n_ice (m/yr)"); ax[2].set_title("Velocity (ApRES op)")
    ax[3].plot(bT.T_C.values, obs_T_d, "k.", ms=4, label="SPICE")
    ax[3].plot(pi["T"], pi["d"], "C1--", lw=1.5); ax[3].plot(pm["T"], pm["d"], "C0-", lw=2)
    ax[3].set_xlabel("T (°C)"); ax[3].set_title("Temperature")
    for a in ax: a.invert_yaxis(); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(f"SP assimilation (H&L, rho+age+T): J {J_hist[0]:.1f} -> {res.fun:.1f}")
    fig.tight_layout(); fig.savefig(OUT / f"{TAG}.png", dpi=110)
    print(f"Saved {OUT / (TAG + '.png')}")
except Exception as e:
    print(f"plot failed: {e}")
print("Done.")
