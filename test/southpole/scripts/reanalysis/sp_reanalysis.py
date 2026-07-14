"""sp_reanalysis.py — South Pole firn reanalysis from the calibrated joint model.

Forward-only (NO adjoint machinery): runs the MAP model from results/sp_joint.json
under two climate scenarios and diagnoses the surface-height change attributable
to the firn's response to the recovered surface-temperature history.

  CTRL : T = T(1000 CE) forever   (the spinup climate — "no climate change")
  FULL : recovered piecewise-linear T-knot history (1000-2015 CE, clamped outside)

Attribution (fixed 130 m surface-following column, constant accumulation,
steady deep-ice dynamics):
    dh'/dt = |w_bot|_FULL - |w_bot|_CTRL
The frame rides on the surface, so the datum-relative surface velocity is
V_deep + |w_bot(t)| with V_deep set by the (steady) column below 130 m; the
scenario difference isolates the firn response above 130 m exactly.
Cross-check: h'(t) ≈ ΔFAC(t) = FAC_FULL - FAC_CTRL, up to the small air-export
anomaly at the column base (1 - rho_bot/rho_i ≈ 8%).

Outputs: results/sp_reanalysis.json / .npz / .png
Env: FIRN_SPIN_YEARS (default 2500), FIRN_DT_YEARS (default 5).
Run: PYTHONPATH=src OMP_NUM_THREADS=1 /home/andrew/venv-firedrake-2026/bin/python sp_reanalysis.py
"""
from __future__ import annotations
import functools, json, os, sys, time
from pathlib import Path
import numpy as np
import firedrake as fd
from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.physics.densification import herron_langway as _hl
from firnpack.constants import year as YEAR_S
herron_langway = functools.partial(_hl, smooth=True)
sys.stdout.reconfigure(line_buffering=True)

OUT = Path(__file__).parent.parent.parent / "results"
TAG = os.environ.get("FIRN_TAG", "sp_reanalysis")
H0, STRETCH_P, SID = 130.0, 2.5, 2
NZ = int(os.environ.get("FIRN_NZ", "100"))
RHO_SURF, BDOT = 350.0, 0.085
DT_YEARS = float(os.environ.get("FIRN_DT_YEARS", "5.0"))
SPIN_YEARS = float(os.environ.get("FIRN_SPIN_YEARS", "2500"))
K_ICE_BASE, BETA = 2.1, 5.0e2
SNAP_EVERY = 5  # profile snapshot cadence (steps) -> 25 yr at dt=5

# ---- MAP parameters from the joint inversion ----
MAP_FILE = os.environ.get("FIRN_MAP", "sp_joint.json")
MAP = json.load(open(OUT / MAP_FILE))
m = MAP["m_map"]
EZZ_YR = float(m.get("ezz_yr", 0.0))   # dynamic vertical strain (sp_joint_ezz MAP)
KNOT_YEARS = np.array(MAP["knot_years"])
T_KNOTS = np.array(MAP["T_knots"])
HAS_BT = "b_knots" in MAP              # r5-style MAP: time-varying accumulation
if HAS_BT:
    B_KY, B_K = np.array(MAP["b_knot_years"]), np.array(MAP["b_knots"])
    BDOT0 = float(B_K[0])              # spinup / CTRL accumulation = b(1000)
else:
    BDOT0 = BDOT
print("=" * 64)
print(f"SP firn reanalysis — forward MAP model ({MAP_FILE}), CTRL vs FULL forcing"
      + (" (T + b(t))" if HAS_BT else " (T only)"))
_kd = (f"ks={m['k_snow_scale']:.3f} kf2={m['k_firn_scale']:.3f}" if "k_snow_scale" in m
       else f"kf={m['k_factor']:.3f}")
print(f"  MAP: k0={m['hl_k0']:.2f} k1={m['hl_k1']:.1f} Ea1={m['hl_Ea1']:.0f} "
      f"Ea2={m['hl_Ea2']:.0f} {_kd} Qb={m['Q_base']:.4f} ezz={EZZ_YR:.2e}/yr")
print(f"  T-knots: " + " ".join(f"{y:.0f}:{t:.2f}" for y, t in zip(KNOT_YEARS, T_KNOTS)))
print(f"  spinup={SPIN_YEARS:.0f}yr dt={DT_YEARS} = {int(SPIN_YEARS/DT_YEARS)} steps")
print("=" * 64)

_pk = dict(
    hl_k0_prefactor=m["hl_k0"], hl_k1_prefactor=m["hl_k1"],
    hl_Ea_stage1=m["hl_Ea1"], hl_Ea_stage2=m["hl_Ea2"],
    basal_heat_flux_W_m2=m["Q_base"],
    hl_stage2_shape=float(m.get("s2_shape", 1.0)))
if "k_snow_scale" in m:   # r5-style MAP: Calonne-2019 conductivity
    _pk.update(conductivity_law="calonne2019",
               k_snow_scale=float(m["k_snow_scale"]),
               k_firn_scale=float(m["k_firn_scale"]))
else:                     # legacy quadratic law
    _pk.update(k_ice=K_ICE_BASE * m["k_factor"])
params = FirnParameters(**_pk)
c_i, T_ref, rho_i = float(params.c_i), float(params.T_ref), float(params.rho_i)
model = FirnModel(params, densification_rate_fn=herron_langway)

# ---- mesh / spaces (identical to sp_joint_assimilate.py) ----
mesh = fd.IntervalMesh(NZ, 0.0, 1.0)
xphys = H0 * (1.0 - (1.0 - fd.SpatialCoordinate(mesh)[0]) ** STRETCH_P)
mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space()).interpolate(fd.as_vector([xphys])))
V = fd.FunctionSpace(mesh, "CG", 1)
R = fd.FunctionSpace(mesh, "R", 0)
dx = fd.dx(domain=mesh); ds = fd.ds(domain=mesh)
xc = fd.SpatialCoordinate(mesh)[0]
depth = H0 - xc
psi = fd.TestFunction(V)
dlen = float(fd.assemble(fd.Constant(1.0) * dx))

xs = fd.Function(V).interpolate(xc).dat.data_ro.copy()
i_bot = int(np.argmin(xs)); i_srf = int(np.argmax(xs))
order = np.argsort(xs)              # bottom -> surface
z_sorted = xs[order]                # physical height coordinate
depth_sorted = H0 - z_sorted        # depth below surface

# ---- stepper (cached solvers, penalty enthalpy BC — as in the joint script) ----
def mk(v, n):
    f = fd.Function(R, name=n); f.assign(float(v)); return f

ws0 = -BDOT0 * rho_i / RHO_SURF / YEAR_S
H_f = fd.Function(V, name="H"); rho_f = fd.Function(V, name="rho")
w_f = fd.Function(V, name="w"); age_f = fd.Function(V, name="age")
H_o = fd.Function(V); rho_o = fd.Function(V); w_o = fd.Function(V); age_o = fd.Function(V)
Ts_eff = fd.Function(V)
ac = mk(BDOT0, "ac"); rs = mk(RHO_SURF, "rs"); ws = mk(ws0, "ws"); dt_r = mk(DT_YEARS * YEAR_S, "dt")
bc_rho = fd.DirichletBC(V, rs, SID); bc_w = fd.DirichletBC(V, ws, SID)
bc_age = fd.DirichletBC(V, mk(0.0, "a0"), SID)
bdot_mass = ac * rho_i / params.spy
sp_lin = {"ksp_type": "preonly", "pc_type": "lu"}
Ht = fd.TrialFunction(V); rt = fd.TrialFunction(V); wt = fd.TrialFunction(V); at = fd.TrialFunction(V)
F_H = model.enthalpy_form(Ht, H_o, rho_f, w_f, psi, dt_r) \
    + fd.Constant(BETA) * (Ht - c_i * (Ts_eff - fd.Constant(T_ref))) * psi * ds(SID)
solv_H = fd.LinearVariationalSolver(
    fd.LinearVariationalProblem(fd.lhs(F_H), fd.rhs(F_H), H_f, constant_jacobian=False),
    solver_parameters=sp_lin)
T_expr = model.temperature_from_enthalpy(H_f)
F_rho, _ = model.density_form(rt, rho_o, T_expr, w_f, None, bdot_mass, psi, dt_r)
solv_rho = fd.LinearVariationalSolver(
    fd.LinearVariationalProblem(fd.lhs(F_rho), fd.rhs(F_rho), rho_f, bcs=[bc_rho], constant_jacobian=False),
    solver_parameters=sp_lin)
drhodt_expr = herron_langway(rho_f, T_expr, params=params, bdot=bdot_mass)
# div_h = -ezz/YEAR_S; a constant ezz cancels EXACTLY in the CTRL-FULL height
# differencing (density-independent w contribution), but keeps w consistent
# with the sp_joint_ezz calibration.
delta_w = model.velocity_delta(wt, w_o, rho_f, drhodt_expr, psi, regularization=1e-3,
                               horizontal_divergence=fd.Constant(-EZZ_YR / YEAR_S))
solv_w = fd.LinearVariationalSolver(
    fd.LinearVariationalProblem(fd.lhs(delta_w), fd.rhs(delta_w), w_f, bcs=[bc_w], constant_jacobian=False),
    solver_parameters=sp_lin)
F_age = model.age_form(at, age_o, w_f, w_o, psi, dt_r)
solv_age = fd.LinearVariationalSolver(
    fd.LinearVariationalProblem(fd.lhs(F_age), fd.rhs(F_age), age_f, bcs=[bc_age], constant_jacobian=False),
    solver_parameters=sp_lin)

# ---- timeline / forcing ----
n_steps = int(SPIN_YEARS / DT_YEARS)
step_years = 2015.0 - (n_steps - 1 - np.arange(n_steps)) * DT_YEARS
Ts_full = np.interp(step_years, KNOT_YEARS, T_KNOTS)   # clamps outside knots
Ts_ctrl = np.full(n_steps, T_KNOTS[0])
if HAS_BT:
    b_full = np.exp(np.interp(step_years, B_KY, np.log(B_K)))
else:
    b_full = np.full(n_steps, BDOT0)
b_ctrl = np.full(n_steps, BDOT0)

def run_scenario(name, ts_C, b_series):
    t0 = time.perf_counter()
    T0 = float(T_KNOTS.mean()) + 273.15
    H_f.assign(c_i * (T0 - T_ref)); H_o.assign(H_f)
    rho_f.interpolate(RHO_SURF + (820.0 - RHO_SURF) * (1.0 - fd.exp(-depth / 25.0))); rho_o.assign(rho_f)
    w_f.assign(ws0); w_o.assign(w_f); age_f.assign(0.0); age_o.assign(0.0)
    fac = np.zeros(n_steps); wbot = np.zeros(n_steps); tmean = np.zeros(n_steps)
    tbot = np.zeros(n_steps); rhomax = np.zeros(n_steps)
    snaps = {"year": [], "rho": [], "T": [], "w": [], "age": []}
    for k in range(n_steps):
        Ts_eff.assign(float(ts_C[k]) + 273.15)
        ac.assign(float(b_series[k]))
        ws.assign(-float(b_series[k]) * rho_i / RHO_SURF / YEAR_S)
        solv_H.solve(); solv_rho.solve(); solv_w.solve(); solv_age.solve()
        H_o.assign(H_f); rho_o.assign(rho_f); w_o.assign(w_f); age_o.assign(age_f)
        fac[k] = float(fd.assemble((1.0 - rho_f / rho_i) * dx))
        wbot[k] = float(w_f.dat.data_ro[i_bot]) * YEAR_S          # m/yr, negative down
        tmean[k] = float(fd.assemble(H_f * dx)) / dlen / c_i + T_ref - 273.15
        tbot[k] = float(H_f.dat.data_ro[i_bot]) / c_i + T_ref - 273.15
        rhomax[k] = float(rho_f.dat.data_ro.max())
        if k % SNAP_EVERY == 0 or k == n_steps - 1:
            snaps["year"].append(step_years[k])
            snaps["rho"].append(rho_f.dat.data_ro[order].copy())
            snaps["T"].append(H_f.dat.data_ro[order].copy() / c_i + T_ref - 273.15)
            snaps["w"].append(w_f.dat.data_ro[order].copy() * YEAR_S)
            snaps["age"].append(age_f.dat.data_ro[order].copy() / YEAR_S)
        if (k + 1) % 100 == 0:
            print(f"  [{name}] step {k+1}/{n_steps} (yr {step_years[k]:.0f}) "
                  f"FAC={fac[k]:.3f} wbot={wbot[k]:.4f} rho_max={rhomax[k]:.0f}")
    print(f"  [{name}] done in {time.perf_counter()-t0:.0f}s — FAC(2015)={fac[-1]:.3f} m, "
          f"|w_bot|={-wbot[-1]:.4f} m/yr, T=[{tmean[-1]:.2f} mean]")
    return dict(fac=fac, wbot=wbot, tmean=tmean, tbot=tbot, rhomax=rhomax,
                snaps={kk: np.array(vv) for kk, vv in snaps.items()})

print("\nCTRL scenario (constant T = %.2f C, b = %.4f):" % (T_KNOTS[0], BDOT0))
C = run_scenario("CTRL", Ts_ctrl, b_ctrl)
print("\nFULL scenario (recovered T%s history):" % (" + b(t)" if HAS_BT else ""))
F = run_scenario("FULL", Ts_full, b_full)

# ---- attribution diagnostics ----
dt_yr = DT_YEARS
# dh'/dt = |w_bot|_F - |w_bot|_C = wbot_C - wbot_F  (w negative down)
# With b(t), h' contains the ice-equivalent MASS part (cumsum of b anomaly,
# baseline-choice-dependent) PLUS the air part; dFAC isolates the air.
dhdt = C["wbot"] - F["wbot"]                       # m/yr
hprime = np.cumsum(dhdt) * dt_yr                   # m
dfac = F["fac"] - C["fac"]                         # m of air
mass = np.cumsum(b_full - b_ctrl) * dt_yr          # m ice eq (0 if constant b)
pre = step_years < 1000.0
print("\nSanity: max |anomaly| pre-1000 CE (forcing identical): "
      f"dh'/dt {np.abs(dhdt[pre]).max():.2e} m/yr, dFAC {np.abs(dfac[pre]).max():.2e} m")
print(f"CTRL steady-state drift, FAC(2015)-FAC(1000): "
      f"{C['fac'][-1] - C['fac'][np.searchsorted(step_years, 1000.0)]:.2e} m")

def _at(yr):
    return int(np.argmin(np.abs(step_years - yr)))
i1900, i1957, i2015 = _at(1900), _at(1957), _at(2015)
headline = dict(
    hprime_2015_cm=float(hprime[-1] * 100),
    mass_2015_cm=float(mass[-1] * 100),
    dfac_2015_cm=float(dfac[-1] * 100),
    hprime_min_cm=float(hprime.min() * 100),
    hprime_min_year=float(step_years[int(np.argmin(hprime))]),
    dhdt_1900_2015_mm_yr=float(np.mean(dhdt[i1900:]) * 1000),
    dhdt_1957_2015_mm_yr=float(np.mean(dhdt[i1957:]) * 1000),
    dhdt_2015_mm_yr=float(dhdt[-1] * 1000),
    fac_ctrl_2015_m=float(C["fac"][-1]), fac_full_2015_m=float(F["fac"][-1]),
    tmean_full_2015=float(F["tmean"][-1]), tmean_ctrl_2015=float(C["tmean"][-1]),
)
print("\n===== HEADLINE =====")
print(f"  Surface-height anomaly h'(2015)  = {headline['hprime_2015_cm']:+.1f} cm "
      f"(mass {headline['mass_2015_cm']:+.1f}, air/dFAC {headline['dfac_2015_cm']:+.1f})")
print(f"  Peak drawdown {headline['hprime_min_cm']:+.1f} cm @ {headline['hprime_min_year']:.0f}")
print(f"  Mean firn dh'/dt 1900-2015 = {headline['dhdt_1900_2015_mm_yr']:+.2f} mm/yr")
print(f"  Mean firn dh'/dt 1957-2015 = {headline['dhdt_1957_2015_mm_yr']:+.2f} mm/yr")
print(f"  Instantaneous 2015 rate    = {headline['dhdt_2015_mm_yr']:+.2f} mm/yr")

# ---- save ----
OUT.mkdir(exist_ok=True)
json.dump(dict(
    map_params=m, knot_years=KNOT_YEARS.tolist(), T_knots=T_KNOTS.tolist(),
    dt_years=DT_YEARS, spin_years=SPIN_YEARS,
    years=step_years.tolist(), Ts_full=Ts_full.tolist(), Ts_ctrl=Ts_ctrl.tolist(),
    fac_ctrl=C["fac"].tolist(), fac_full=F["fac"].tolist(),
    wbot_ctrl=C["wbot"].tolist(), wbot_full=F["wbot"].tolist(),
    tmean_ctrl=C["tmean"].tolist(), tmean_full=F["tmean"].tolist(),
    b_full=b_full.tolist(), b_ctrl=b_ctrl.tolist(),
    hprime_m=hprime.tolist(), dfac_m=dfac.tolist(), mass_m=mass.tolist(),
    headline=headline,
), open(OUT / f"{TAG}.json", "w"), indent=1)
np.savez_compressed(
    OUT / f"{TAG}.npz",
    depth=depth_sorted, snap_years=C["snaps"]["year"],
    rho_ctrl=C["snaps"]["rho"], rho_full=F["snaps"]["rho"],
    T_ctrl=C["snaps"]["T"], T_full=F["snaps"]["T"],
    w_ctrl=C["snaps"]["w"], w_full=F["snaps"]["w"],
    age_ctrl=C["snaps"]["age"], age_full=F["snaps"]["age"])
print(f"\nSaved {OUT/(TAG+'.json')} and .npz")

# ---- plot ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

show = step_years >= 1000.0
yr = step_years[show]
fig = plt.figure(figsize=(12, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.32, wspace=0.25)

ax = fig.add_subplot(gs[0, 0])
ax.plot(yr, Ts_full[show], "C3-", lw=1.5, label="FULL (recovered)")
ax.plot(KNOT_YEARS, T_KNOTS, "C3o", ms=5)
ax.axhline(T_KNOTS[0], color="k", ls="--", lw=1, label="CTRL (const)")
ax.set_ylabel("surface T (°C)"); ax.set_title("(a) Surface-temperature forcing")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ax.plot(yr, F["tmean"][show], "C3-", lw=1.5, label="FULL")
ax.plot(yr, C["tmean"][show], "k--", lw=1, label="CTRL")
ax.set_ylabel("column-mean firn T (°C)"); ax.set_title("(b) Firn thermal memory")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 0])
ax.plot(yr, F["fac"][show], "C3-", lw=1.5, label="FULL")
ax.plot(yr, C["fac"][show], "k--", lw=1, label="CTRL")
ax.set_ylabel("FAC 0–130 m (m of air)"); ax.set_title("(c) Firn air content")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
ax.plot(yr, hprime[show] * 100, "C0-", lw=2, label=r"$h'$ from $|w_{bot}|$ anomaly")
if HAS_BT:
    ax.plot(yr, mass[show] * 100, "C2:", lw=1.5, label="ice-eq mass part")
ax.plot(yr, dfac[show] * 100, "C1--", lw=1.5,
        label=r"$\Delta$FAC (air part)" if HAS_BT else r"$\Delta$FAC cross-check")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("surface-height anomaly (cm)")
ax.set_title("(d) Firn-driven surface-height anomaly")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.annotate(f"2015: {headline['hprime_2015_cm']:+.1f} cm\n"
            f"1957–2015: {headline['dhdt_1957_2015_mm_yr']:+.2f} mm/yr",
            xy=(0.03, 0.05), xycoords="axes fraction", fontsize=9,
            bbox=dict(boxstyle="round", fc="w", alpha=0.8))

ax = fig.add_subplot(gs[2, :])
sy = np.array(C["snaps"]["year"]); ssel = sy >= 1000.0
drho = (F["snaps"]["rho"] - C["snaps"]["rho"])[ssel]
vmax = max(np.abs(drho).max(), 1e-6)
pc = ax.pcolormesh(sy[ssel], depth_sorted, drho.T, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, shading="nearest")
ax.invert_yaxis(); ax.set_xlabel("year CE"); ax.set_ylabel("depth (m)")
ax.set_title(r"(e) Density anomaly $\rho_{FULL}-\rho_{CTRL}$ (kg m$^{-3}$)")
fig.colorbar(pc, ax=ax, pad=0.01)

fig.suptitle("South Pole firn reanalysis — MAP joint model, firn response to recovered T history",
             fontsize=12, y=0.995)
fig.savefig(OUT / f"{TAG}.png", dpi=140, bbox_inches="tight")
print(f"Saved {OUT/(TAG+'.png')}")
print("Done.")
