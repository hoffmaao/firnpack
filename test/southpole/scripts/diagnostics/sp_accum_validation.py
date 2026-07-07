"""sp_accum_validation.py — is the r6 1940s-50s accumulation spike REAL or a
Buizert/ERA5 prior-seam artifact?

Model-free arbiter: the raw SP19 annual-layer thickness crosses 1950
CONTINUOUSLY (no seam), whereas the priors have a Buizert(<=1950)/ERA5(>1950)
discontinuity there. Apparent ice-equivalent accumulation from the raw layers:

    b_app(yr) = lambda(z) * rho(z)/rho_ice / thinning
    lambda(z) = local annual-layer thickness  (d depth / d age, native 1-yr)
    thinning  = exp(-|ezz| * age)              (~0.5% at 20 m; negligible)

If b_app is smooth across 1950 but the MAP b(t) spikes there, the spike is
prior-induced. If b_app itself thickens at 1945-50, the accumulation high is
real (data-driven).

Overlays: raw b_app, r6 MAP b(t), Buizert prior (<=1950), ERA5 annual (>1940),
and (if staged) the Kahle-2020 SPICEcore reconstruction.

Output: results/sp_accum_validation.{png,json}
Run: /home/andrew/venv-firedrake-2026/bin/python sp_accum_validation.py  (numpy only)
Env: FIRN_MAP (default sp_joint_r6.json)
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent.parent
PROC = BASE / "processed"
OUT = BASE / "results"
RHO_I = 917.0
MAP_FILE = os.environ.get("FIRN_MAP", "sp_joint_r6.json")

MAP = json.load(open(OUT / MAP_FILE))
BKY, BK = np.array(MAP["b_knot_years"]), np.array(MAP["b_knots"])
BCTR = np.array(MAP.get("b_prior_centers", BK))
EZZ = abs(float(MAP["m_map"].get("ezz_yr", 0.0)))

# ---- raw SP19 layer thickness -> apparent ice-eq accumulation ----
a = pd.read_csv(PROC / "sp19_depth_age.csv")
a = a[(a.depth_m <= 130) & (a.year_CE <= 2015)].sort_values("depth_m").reset_index(drop=True)
z, yr = a.depth_m.values, a.year_CE.values
age = 2015.0 - yr
# local annual layer thickness via centered gradient of depth wrt age
order = np.argsort(age)
age_s, z_s = age[order], z[order]
lam = np.gradient(z_s, age_s)                      # m (firn) per yr
# density at layer depth
d = pd.read_csv(PROC / "sp19_density.csv")
rho = np.interp(z_s, d.depth_m.values, d.rho_kgm3.values * 1000.0)
thinning = np.exp(-EZZ * age_s)                    # ice-dynamic strain (tiny here)
b_app = lam * rho / RHO_I / thinning               # m ice / yr
dep_year = 2015.0 - age_s
# smooth (5-yr running median) for display; keep raw too
def runmed(x, w=5):
    return np.array([np.median(x[max(0, i-w//2):i+w//2+1]) for i in range(len(x))])
b_app_sm = runmed(b_app, 5)

# ---- MAP b(t) on a dense year grid ----
yy = np.linspace(1850, 2015, 400)
b_map = np.exp(np.interp(yy, BKY, np.log(BK)))

# ---- Buizert prior (raw file) + ERA5 annual ----
buiz = pd.read_csv(PROC / "buizert2021_spice_accum.csv").query("year_CE>=1850 and year_CE<=1950")
era = pd.read_csv(PROC / "era5_monthly_point.csv")
era_ann = era.groupby("year")["net_accum_m_iceeq_month"].sum()
era_ann = era_ann[(era_ann.index >= 1941) & (era_ann.index <= 2015)]

# ---- optional Kahle reconstruction ----
kahle = None
for cand in ["kahle2020_spice_accum.csv", "Site_Reconstructions_Temperature_AccumulationRate.txt"]:
    p = PROC / cand
    if p.exists():
        try:
            k = pd.read_csv(p, sep=None, engine="python", comment="#")
            yc = [c for c in k.columns if "year" in c.lower() or "age" in c.lower()][0]
            ac = [c for c in k.columns if "accum" in c.lower() or "acc" in c.lower()][0]
            kahle = (k[yc].values, k[ac].values)
        except Exception as e:
            print(f"(Kahle file {cand} present but unparsed: {e})")
        break

# ---- quantify: MAP vs raw at the spike; smoothness of raw across 1950 ----
def at(arr_y, arr_v, y):
    return float(np.interp(y, arr_y, arr_v))
spike_map = at(yy, b_map, 1950)
base_map = 0.5 * (at(yy, b_map, 1925) + at(yy, b_map, 1970))
# raw apparent accumulation is depth-ordered; interp on deposition year (ascending)
oy = np.argsort(dep_year)
dy, ba = dep_year[oy], b_app_sm[oy]
spike_raw = at(dy, ba, 1950)
base_raw = 0.5 * (at(dy, ba, 1925) + at(dy, ba, 1970))
# roughness of raw across the seam: does 1950 stick out above its 1940-1960 neighbours?
win = (dy >= 1940) & (dy <= 1960)
raw_1950_z = (at(dy, ba, 1950) - np.mean(ba[win])) / (np.std(ba[win]) + 1e-9)

print("=" * 64)
print(f"ACCUM SPIKE VALIDATION — {MAP_FILE}")
print("=" * 64)
print(f"MAP b(t):  b(1950)={spike_map:.4f}, baseline(1925/1970 avg)={base_map:.4f}"
      f"  -> spike +{100*(spike_map/base_map-1):.0f}%")
print(f"RAW b_app: b(1950)={spike_raw:.4f}, baseline={base_raw:.4f}"
      f"  -> spike +{100*(spike_raw/base_raw-1):.0f}%")
if raw_1950_z > 1.0:
    verdict = "RAW SPIKES UP at 1950 too -> spike is REAL"
elif raw_1950_z < -0.5:
    verdict = "RAW is a LOCAL LOW at 1950 (opposite of MAP) -> MAP spike is ARTIFACT"
else:
    verdict = "RAW is SMOOTH across 1950 -> MAP spike is prior-induced ARTIFACT"
print(f"RAW 1950 z-score within 1940-1960: {raw_1950_z:+.2f} sigma  ({verdict})")
mean_raw = float(np.mean(ba[(dy >= 1900) & (dy <= 2010)]))
print(f"raw b_app mean 1900-2010 = {mean_raw:.4f} m ice/yr; stake-farm band ~0.085-0.093")

# ---- figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C_MAP, C_RAW, C_BUIZ, C_ERA, C_KAH = "#2563EB", "#111827", "#EA580C", "#6B7280", "#059669"
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(dep_year, b_app, ".", color=C_RAW, ms=2.5, alpha=0.35, label="raw layer b_app (annual)")
ax.plot(dy, ba, "-", color=C_RAW, lw=2, label="raw b_app (5-yr median)")
ax.plot(yy, b_map, "-", color=C_MAP, lw=2.2, label=f"r6 MAP b(t)")
ax.plot(BKY[BKY >= 1850], BK[BKY >= 1850], "o", color=C_MAP, ms=4)
ax.plot(buiz.year_CE, buiz.accum, "--", color=C_BUIZ, lw=1.5, label="Buizert prior (≤1950)")
ax.plot(era_ann.index, era_ann.values, "-", color=C_ERA, lw=1, alpha=0.6, label="ERA5 annual (>1940)")
if kahle is not None:
    ky, kv = kahle
    sel = (ky >= 1850) & (ky <= 2015)
    ax.plot(ky[sel], kv[sel], "-", color=C_KAH, lw=1.6, label="Kahle 2020 recon")
ax.axvline(1950, color=C_BUIZ, ls=":", lw=1.2, alpha=0.7)
ax.text(1950.5, ax.get_ylim()[1]*0.97, "Buizert→ERA5\nprior seam", fontsize=8,
        color=C_BUIZ, va="top")
ax.axhspan(0.085, 0.093, color=C_ERA, alpha=0.08, lw=0)
ax.set_xlabel("deposition year CE"); ax.set_ylabel("accumulation (m ice eq / yr)")
ax.set_xlim(1850, 2015); ax.set_ylim(0.04, 0.16)
ax.set_title(f"Is the {MAP_FILE.replace('sp_joint_','').replace('.json','')} 1940s–50s "
             f"accumulation spike real? — raw SP19 layers cross 1950 with no seam")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "sp_accum_validation.png", dpi=140)
json.dump(dict(map_file=MAP_FILE, spike_map=spike_map, base_map=base_map,
               spike_raw=spike_raw, base_raw=base_raw, raw_1950_zscore=raw_1950_z,
               raw_mean_1900_2010=mean_raw, has_kahle=kahle is not None),
          open(OUT / "sp_accum_validation.json", "w"), indent=1)
print(f"Saved {OUT/'sp_accum_validation.png'} and .json")
