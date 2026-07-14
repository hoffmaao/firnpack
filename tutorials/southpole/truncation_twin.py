"""truncation_twin.py — is the fixed-G basal BC at 130 m adequate?

Andrew (2026-07-13): the true gradient at 130 m = quasi-steady geothermal part
+ a transient part from the surface T-history advected/diffused down; a fixed
G clamps the latter. Analytic scale: +/-0.5 C centennial anomalies induce
~1 mK/m transient gradients at 130 m — same order as the fitted G (-4.4 mK/m).

Twin: forward at the SAME MAP (sp_r10v_pair) with
  A: H_col=130 m (baseline; G imposed AT the data boundary — clamped), and
  B: H_col=300 m (NZ 230 keeps dz; spin 3200 > thermal equilibration
     300^2/kappa ~ 2400 yr; same G imposed at 300 m, where the forcing
     window's transients are negligible — the 130-m gradient EMERGES).
Report: max |dT| over the data range, per-block rms changes, and the emergent
130-m gradient in B vs the imposed G.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> experiments/southpole/truncation_twin.py
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; R = HERE / "results"
from firnpack.inverse import assimilate

MAP_JSON = os.environ.get("FIRN_MAP_JSON", str(R / "sp_r10v_pair.json"))
warm = json.load(open(MAP_JSON))
G_MAP = warm["m_map"]["G_base"]

def forward_at(hcol, nz, spin):
    os.environ.update(FIRN_VEL_SITE="x17s2+x11n0", FIRN_HCOL=str(hcol),
                      FIRN_NZ=str(nz), FIRN_SPIN=str(spin))
    ns = {"__file__": str(HERE / "run.py"), "__name__": "cfgbuild"}
    src = open(HERE / "run.py").read().split("warm = json.load")[0]
    exec(src, ns)
    return assimilate(ns["cfg"], mode="forward", warm=warm, verbose=False)

out = {}
for tag, (h, nz, sp) in dict(A=(130.0, 100, 2500.0), B=(300.0, 230, 3200.0)).items():
    r = forward_at(h, nz, sp)
    p = r["profiles"]
    out[tag] = dict(H=h, J=r["J"], diag=r["diag"],
                    depth=p["depth"], T_C=p["T_C"], rho=p["rho"])
    print(f"{tag} (H={h:.0f}): J={r['J']:.3f} " +
          " ".join(f"{k[4:]}={v:.2f}" for k, v in r["diag"].items() if k.startswith("rms_")))

dA, TA = np.array(out["A"]["depth"]), np.array(out["A"]["T_C"])
dB, TB = np.array(out["B"]["depth"]), np.array(out["B"]["T_C"])
sel = dA <= 130.0
TB_on_A = np.interp(dA[sel], dB, TB)
dT = TB_on_A - TA[sel]
print(f"\nT(z<=130) difference (B - A): max|dT|={np.max(np.abs(dT))*1000:.1f} mK "
      f"at z={dA[sel][np.argmax(np.abs(dT))]:.0f} m; rms={np.sqrt(np.mean(dT**2))*1000:.1f} mK")
for zq in (60, 100, 125):
    print(f"  dT({zq:3d} m) = {np.interp(zq, dA[sel], dT)*1000:+.1f} mK")

def grad_at(d, T, z0, half=6.0):
    s = np.abs(d - z0) <= half
    A = np.vstack([d[s] - z0, np.ones(s.sum())]).T
    return np.linalg.lstsq(A, T[s], rcond=None)[0][0]

gA = grad_at(dA, TA, 128.0); gB = grad_at(dB, TB, 130.0)
print(f"\ngradient near 130 m: A (clamped) {gA*1e3:+.2f} mK/m | "
      f"B (emergent) {gB*1e3:+.2f} mK/m | imposed G_MAP {G_MAP*1e3:+.2f} mK/m")
print(f"transient part (B_emergent - imposed): {(gB-G_MAP)*1e3:+.2f} mK/m")
json.dump(out, open(R / "truncation_twin.json", "w"))
print(f"Saved {R/'truncation_twin.json'}")
