"""stage_seasonal_obs.py — validate the in-engine WKB seasonal-damping operator
against the exact complex harmonic BVP, and stage the USP50 observation block.

The engine predicts ln A(z_j)/A(ref) = -int sqrt(w rho c / 2k) dz on the coarse
solver mesh (CG1, dz ~1.3 m, erf windows sw=0.6). Here we compute, at the
reference config (s_snow*=1.29, k_firn*=0.987, T*=222.25 K, USP50 density):
    exact   : tridiagonal complex BVP, fine grid (the offline-fit operator)
    wkb_fine: damping integral, fine grid (WKB error alone)
    mimic   : damping integral exactly as the engine discretizes it
Stage obs = measured lnr + (mimic - exact)|_ref-config  (operator-bias
correction; residual is second order in s away from s*), sigma = sqrt(0.075^2
+ (0.2*bias)^2). Also refits s_snow with the exact operator as a cross-check
against the offline 1.291.

Output: data/usp50_seasonal_lnratio.csv
Run: <venv-python> tutorials/southpole/diagnostics/stage_seasonal_obs.py   (numpy only)
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent  # the tutorial dir, not diagnostics/
FIT = json.load(open(HERE / "output/usp50_k_snow_fit.json"))
RHO_I, C_I, T_K = 917.0, 2009.0, 222.25
KF, SSTAR = 0.987, FIT["s_snow"]
YEAR_S = 365.25 * 86400.0
OMEGA = 2.0 * math.pi / YEAR_S
Z_REF, A_MIN = 1.23, 0.05

ud = pd.read_csv(HERE / "data/usp50_density_nicl.csv")
zd, rd = ud.depth.values, ud.Density.values
rho_of = lambda z: np.interp(z, zd, rd, left=rd[0], right=rd[-1])

def k_cal(rho, s):
    k_sn = 0.024 - 1.23e-4 * rho + 2.5e-6 * rho**2
    k_fi = 2.107 + 3.618e-3 * (rho - RHO_I)
    th = 1.0 / (1.0 + np.exp(-2.0 * 0.02 * (rho - 450.0)))
    return math.exp(-5.7e-3 * (T_K - 270.15)) * ((1 - th) * s * k_sn + th * KF * k_fi)

sens = [s for s in FIT["sensors"] if s["zeff"] > Z_REF and s["A"] >= A_MIN]
A_ref = [s["A"] for s in FIT["sensors"] if abs(s["zeff"] - Z_REF) < 0.01][0]
ze = np.array([s["zeff"] for s in sens])
lnr_meas = np.log(np.array([s["A"] for s in sens]) / A_ref)

def exact_lnA(s, zq):
    z = np.arange(0.0, 45.0 + 1e-9, 0.01)
    k = k_cal(rho_of(z), s); rc = rho_of(z) * C_I
    n = len(z); h = z[1] - z[0]
    kh = 0.5 * (k[1:] + k[:-1])           # k at half nodes
    lo = kh[:-1] / h**2; up = kh[1:] / h**2
    di = -(kh[:-1] + kh[1:]) / h**2 - 1j * OMEGA * rc[1:-1]
    A = np.zeros((3, n - 2), complex)
    A[0, 1:] = up[:-1]; A[1, :] = di; A[2, :-1] = lo[1:]
    b = np.zeros(n - 2, complex); b[0] = -lo[0] * 1.0
    from scipy.linalg import solve_banded
    T = np.empty(n, complex); T[0] = 1.0; T[-1] = 0.0
    T[1:-1] = solve_banded((1, 1), A, b)
    return np.interp(zq, z, np.log(np.abs(T)))

def wkb_fine_lnr(s, zq):
    z = np.arange(Z_REF, 20.0, 0.005)
    q = np.sqrt(OMEGA * rho_of(z) * C_I / (2.0 * k_cal(rho_of(z), s)))
    Q = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(z))])
    return -np.interp(zq, z, Q)

def mimic_lnr(s, zq, H=130.0, NZ=100, sw=0.6):
    zn = np.linspace(0.0, H, NZ + 1)                     # solver-mesh node depths
    qn = np.sqrt(OMEGA * rho_of(zn) * C_I / (2.0 * k_cal(rho_of(zn), s)))
    from scipy.special import erf
    zf = np.arange(0.0, 30.0, 0.002)                     # fine grid for exact CG1 product
    qf = np.interp(zf, zn, qn)                           # piecewise-linear q
    out = []
    for zj in zq:
        wn = 0.5 * (erf((zj - zn) / (math.sqrt(2) * sw)) - erf((Z_REF - zn) / (math.sqrt(2) * sw)))
        wf = np.interp(zf, zn, wn)                       # piecewise-linear window
        out.append(-np.trapz(qf * wf, zf))
    return np.array(out)

ex = exact_lnA(SSTAR, ze) - exact_lnA(SSTAR, np.array([Z_REF]))[0]
wk = wkb_fine_lnr(SSTAR, ze)
mi = mimic_lnr(SSTAR, ze)
bias = mi - ex

# cross-check: refit s with the exact operator (should reproduce offline 1.291)
ss = np.linspace(0.9, 1.8, 181)
cost = [np.sum((exact_lnA(s, ze) - exact_lnA(s, np.array([Z_REF]))[0] - lnr_meas) ** 2) for s in ss]
s_fit = ss[int(np.argmin(cost))]

print(f"{'zeff':>6s} {'lnr_meas':>9s} {'exact':>8s} {'wkbfine':>8s} {'mimic':>8s} {'bias':>7s}")
for i in range(len(ze)):
    print(f"{ze[i]:6.2f} {lnr_meas[i]:9.3f} {ex[i]:8.3f} {wk[i]:8.3f} {mi[i]:8.3f} {bias[i]:+7.3f}")
print(f"\nWKB-alone error (fine): max {np.max(np.abs(wk-ex)):.3f}; "
      f"engine-mimic bias: max {np.max(np.abs(bias)):.3f} (corrected at staging)")
print(f"exact-operator refit s_snow = {s_fit:.3f}  (offline joint fit: {SSTAR:.3f})")

sig = np.sqrt(0.075**2 + (0.2 * bias)**2)
out = pd.DataFrame(dict(zeff=ze, lnr=lnr_meas + bias, sig=sig, zref=Z_REF,
                        lnr_meas=lnr_meas, op_bias=bias))
out.to_csv(HERE / "data/usp50_seasonal_lnratio.csv", index=False)
print(f"Staged {HERE/'data/usp50_seasonal_lnratio.csv'} ({len(out)} sensors)")
