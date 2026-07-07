"""sp_uq_combine.py — Laplace posterior + error bars on the reanalysis headline.

Combines results/sp_uq_hessian.json (12x12 FD Hessian of the joint posterior at
the MAP, internal coords) with results/sp_uq_sensitivity.json (dq/dx of the
attribution diagnostics) into:

  Sigma = H_sym^{-1}  (eigenvalue-clipped if needed)
  sigma_q = sqrt(S Sigma S^T)  for q = [h'(2015), rate 1957-2015, dFAC(2015)]
  per-parameter marginal posteriors (log-normal for the LOGS params)
  softest Hessian mode (the least-constrained parameter combination)

numpy only — no Firedrake. Output: results/sp_uq_result.json + console summary.
Run: /home/andrew/venv-firedrake-2026/bin/python sp_uq_combine.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent.parent.parent / "results"
Hd = json.load(open(OUT / "sp_uq_hessian.json"))
Sd = json.load(open(OUT / "sp_uq_sensitivity.json"))
names = Hd["names"]
assert names == Sd["names"], "control ordering mismatch"
N = len(names)
LOGS = set(Hd["logs"])
x_star = np.array(Hd["x_star"])
H = np.array(Hd["H_sym"])
S = np.array(Sd["S"])          # (3, N)
q0 = np.array(Sd["q_base"])    # base attribution values
q_names = Sd["q_names"]

lam, U = np.linalg.eigh(H)
print("Hessian eigenvalues:", " ".join(f"{v:.3e}" for v in lam))
eps = 1e-8 * lam.max()
n_clip = int((lam < eps).sum())
if n_clip:
    print(f"WARNING: clipping {n_clip} eigenvalue(s) below {eps:.2e}")
lam_c = np.maximum(lam, eps)
Sigma = (U / lam_c) @ U.T

print(f"\nFD asymmetry (quality): {Hd['asym']:.2e}   |g(x*)| = {np.linalg.norm(Hd['g0']):.2e}")
print(f"J(x*) = {Hd['J0']:.6f} vs saved MAP {Hd['J_map_saved']:.6f}")

sig = np.sqrt(np.diag(Sigma))
print("\n--- marginal posteriors (Laplace) ---")
for i, n in enumerate(names):
    if n in {"hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "k_factor"}:
        med = math.exp(x_star[i]); fac = math.exp(sig[i])
        print(f"  {n:9s} = {med:.4g}  x/ {fac:.3f}   (68%: [{med/fac:.4g}, {med*fac:.4g}])")
    else:
        print(f"  {n:9s} = {x_star[i]:+.4f} +/- {sig[i]:.4f}")

# correlation of the softest modes
i_soft = int(np.argmin(lam))
v = U[:, i_soft]
v = v / np.abs(v).max()
print("\nSoftest Hessian mode (least-constrained combination):")
print("  " + "  ".join(f"{n}:{v[i]:+.2f}" for i, n in enumerate(names) if abs(v[i]) > 0.15))

cov_q = S @ Sigma @ S.T
sig_q = np.sqrt(np.diag(cov_q))
scale = [100.0, 1000.0, 100.0]; unit = ["cm", "mm/yr", "cm"]
print("\n--- reanalysis headline with 1-sigma (parameter posterior only) ---")
res = {}
for k, qn in enumerate(q_names):
    print(f"  {qn:22s} = {q0[k]*scale[k]:+.2f} +/- {sig_q[k]*scale[k]:.2f} {unit[k]}")
    res[qn] = dict(value=float(q0[k]), sigma=float(sig_q[k]))

json.dump(dict(names=names, x_star=x_star.tolist(), sigma_marginal=sig.tolist(),
               Sigma=Sigma.tolist(), eig=lam.tolist(), n_clipped=n_clip,
               softest_mode=U[:, i_soft].tolist(), headline=res,
               corr_note="Sigma is the Laplace posterior of the normalized joint J "
                         "(per-dataset 1/N chi2 -> sigma_d inflated by sqrt(N_d); conservative)"),
          open(OUT / "sp_uq_result.json", "w"), indent=1)
print(f"\nSaved {OUT/'sp_uq_result.json'}")
print("Done.")
