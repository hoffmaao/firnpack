"""uq_laplace.py - Laplace posterior marginals from an engine Hessian.

Reads output/<tag>_hessian.json (written by FIRN_MODE=hessian) and reports,
per control: the posterior marginal sigma, the prior sigma, and the shrinkage
f = sigma_post/sigma_prior (f ~ 1 -> prior-bound, data say nothing; f << 1 ->
data-constrained). Log-controls are reported as RELATIVE (%) via the log-space
sigma; T-knots in K.

J includes the prior, so H is the full posterior precision; Sigma = inv(H).
The known caveat from the archived UQ ([[sp-pp-uq-verdict]]) applies: the
DIAGONAL of Sigma understates joint uncertainty when controls trade - the
correlation table at the bottom shows the biggest trades.

Env: FIRN_HESS (default sp_recal2_hessian.json).
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/diagnostics/uq_laplace.py
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "output"
HP = OUT / os.environ.get("FIRN_HESS", "sp_recal2_hessian.json")
d = json.load(open(HP))
names = d["names"]; H = np.array(d["H"]); pri = np.array(d["prior_sigma"])
islog = np.array(d["islog"], bool); x0 = np.array(d["x0"])
n = len(names)
print(f"{HP.name}: {n}x{n} block at J0={d['J0']:.4f} (tag {d['tag']})")

# Work in PRIOR-SCALED coordinates (x/sigma_prior): J includes the prior, so
# the scaled Hessian is ~identity + data information and its spectrum is tame
# (the raw spectrum spans ~1e10 because the ezz pin's precision dwarfs
# everything - any absolute ridge chosen against ev.max() there flattens every
# other marginal, which is an artifact, not a posterior).
D = np.diag(pri)
Hs = D @ H @ D
ev, U = np.linalg.eigh(Hs)
print(f"scaled eigenvalues: min {ev.min():.3e}, max {ev.max():.3e}"
      + ("" if ev.min() > 0 else
         f"  ** {int((ev <= 0).sum())} non-positive (iteration-capped MAP / FD noise) **"))
# floor tiny/negative eigenvalues: sigma_post along such a direction is capped
# at (1/sqrt(floor)) * sigma_prior instead of exploding/imaginary
FLOOR = 0.05
nfl = int((ev < FLOOR).sum())
if nfl:
    print(f"  ({nfl} eigenvalues floored at {FLOOR} -> sigma capped at "
          f"{1/np.sqrt(FLOOR):.1f}x prior along those directions)")
Ss = (U / np.maximum(ev, FLOOR)) @ U.T
S = D @ Ss @ D          # back to internal coordinates
sig = np.sqrt(np.diag(S))

print(f"\n{'control':13s} {'MAP':>10s} {'sig_post':>9s} {'sig_prior':>9s} {'f':>5s}")
for i, nm in enumerate(names):
    f = sig[i] / pri[i]
    if islog[i]:
        mapv = float(np.exp(x0[i]))
        post = f"{100*(np.exp(sig[i])-1):.1f}%"; prio = f"{100*(np.exp(pri[i])-1):.0f}%"
    else:
        mapv = float(x0[i])
        post = f"{sig[i]:.3f}"; prio = f"{pri[i]:.2f}"
    flag = "  <- prior-bound" if f > 0.85 else ("  <- data-tight" if f < 0.3 else "")
    print(f"{nm:13s} {mapv:10.4g} {post:>9s} {prio:>9s} {f:5.2f}{flag}")

C = S / np.outer(sig, sig)
iu = np.triu_indices(n, 1)
big = np.argsort(-np.abs(C[iu]))[:8]
print("\nbiggest posterior correlations (trades):")
for k in big:
    i, j = iu[0][k], iu[1][k]
    print(f"  {names[i]:12s} <-> {names[j]:12s} r = {C[i,j]:+.2f}")

out = dict(names=names, sigma_post=sig.tolist(), sigma_prior=pri.tolist(),
           f=(sig/pri).tolist(), islog=islog.tolist(), x0=x0.tolist(),
           eig_min=float(ev.min()), corr=C.tolist(), source=HP.name)
json.dump(out, open(OUT / (HP.stem + "_marginals.json"), "w"), indent=1)
print(f"\nwrote {OUT/(HP.stem + '_marginals.json')}")
