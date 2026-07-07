"""sp_uq_combine_pp.py — Laplace posterior + honest error bars (per-point round).

Combines results/sp_uq_hessian_pp.json (46x46 FD Hessian of the PER-POINT
posterior at its MAP) with results/sp_uq_sensitivity_pp.json (dq/dx of the
attribution) into Sigma = H^{-1}, marginals, softest modes, and 1-sigma bars
on q = [h'(2015), rate 1957-2015, dFAC(2015)] — plus the eigenmode breakdown
of var(h') (which parameter combinations carry the uncertainty).

numpy only. Output: results/sp_uq_result_pp.json + console summary.
Run: /home/andrew/venv-firedrake-2026/bin/python sp_uq_combine_pp.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent.parent.parent / "results"
Hd = json.load(open(OUT / "sp_uq_hessian_pp.json"))
Sd = json.load(open(OUT / "sp_uq_sensitivity_pp.json"))
names = Hd["names"]
assert names == Sd["names"], "control ordering mismatch"
N = len(names)
LOGS = set(Hd["logs"])
N_SCAL = N - Hd["n_knot"] - Hd["n_b"]
LOG_COORD = LOGS | set(names[N_SCAL+Hd["n_knot"]:])   # b-knots are log-space too
x_star = np.array(Hd["x_star"])
H = np.array(Hd["H_sym"])
S = np.array(Sd["S"])          # (3, N)
q0 = np.array(Sd["q_base"])
q_names = Sd["q_names"]
scale = np.array(Hd["scale"])

# work in prior-sigma coords for conditioning, convert back for marginals
Hs = H * np.outer(scale, scale)
lam, U = np.linalg.eigh(Hs)
print("Hessian eigenvalues (prior-sigma coords):")
print("  " + " ".join(f"{v:.2e}" for v in lam))
eps = 1e-8 * lam.max()
n_clip = int((lam < eps).sum())
if n_clip:
    print(f"WARNING: clipping {n_clip} eigenvalue(s) below {eps:.2e}")
lam_c = np.maximum(lam, eps)
Sigma_s = (U / lam_c) @ U.T                    # posterior cov in sigma coords
Sigma = Sigma_s * np.outer(scale, scale)       # native internal coords

print(f"\nFD asymmetry (quality): {Hd['asym']:.2e}   |g(x*)*sigma| = "
      f"{np.linalg.norm(np.array(Hd['g0'])*scale):.2e}")
print(f"J(x*) = {Hd['J0']:.6f} vs saved MAP {Hd['J_map_saved']:.6f}")

sig = np.sqrt(np.diag(Sigma))
print("\n--- marginal posteriors (Laplace, per-point error model) ---")
for i, n in enumerate(names):
    if n in LOG_COORD:
        med = math.exp(x_star[i]); fac = math.exp(sig[i])
        print(f"  {n:9s} = {med:.4g}  x/ {fac:.3f}   (68%: [{med/fac:.4g}, {med*fac:.4g}])")
    else:
        print(f"  {n:9s} = {x_star[i]:+.4f} +/- {sig[i]:.4f}")

i_soft = int(np.argmin(lam))
v = U[:, i_soft]; v = v / np.abs(v).max()
print("\nSoftest Hessian mode (prior-sigma coords):")
print("  " + "  ".join(f"{n}:{v[i]:+.2f}" for i, n in enumerate(names) if abs(v[i]) > 0.15))

cov_q = S @ Sigma @ S.T
sig_q = np.sqrt(np.diag(cov_q))
sc = [100.0, 1000.0, 100.0]; unit = ["cm", "mm/yr", "cm"]
print("\n--- attribution headline with 1-sigma (per-point posterior) ---")
res = {}
for k, qn in enumerate(q_names):
    print(f"  {qn:22s} = {q0[k]*sc[k]:+.2f} +/- {sig_q[k]*sc[k]:.2f} {unit[k]}")
    res[qn] = dict(value=float(q0[k]), sigma=float(sig_q[k]))

# eigenmode breakdown of var(h'): var = sum_m (S_h . u_m)^2 / lam_m  (sigma coords)
Ss = S * scale[None, :]                        # dq/d(sigma-coord)
proj = Ss[0, :] @ U                            # (N,) mode projections for h'
contrib = proj**2 / lam_c
order = np.argsort(contrib)[::-1]
tot = contrib.sum()
print("\n--- var(h') eigenmode breakdown (top 5) ---")
top_modes = []
for m in order[:5]:
    u = U[:, m]; u = u / np.abs(u).max()
    members = "  ".join(f"{names[i]}:{u[i]:+.2f}" for i in range(N) if abs(u[i]) > 0.3)
    print(f"  lam={lam_c[m]:.2e}  {100*contrib[m]/tot:5.1f}%  {members}")
    top_modes.append(dict(lam=float(lam_c[m]), frac=float(contrib[m]/tot),
                          mode=U[:, m].tolist()))

json.dump(dict(names=names, x_star=x_star.tolist(), sigma_marginal=sig.tolist(),
               Sigma=Sigma.tolist(), eig_sigma_coords=lam.tolist(), n_clipped=n_clip,
               softest_mode=U[:, i_soft].tolist(), headline=res,
               var_h_top_modes=top_modes,
               corr_note="Per-point chi^2 posterior (empirical N_eff ~= N); "
                         "representation error still in the generous sigmas."),
          open(OUT / "sp_uq_result_pp.json", "w"), indent=1)
print(f"\nSaved {OUT/'sp_uq_result_pp.json'}")
print("Done.")
