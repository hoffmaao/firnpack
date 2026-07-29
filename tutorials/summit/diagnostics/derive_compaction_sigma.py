"""Defensible sigma for the Summit FirnCover compaction rates.

The daily-fit standard error is unusable (daily residuals are seasonal,
lag-1 ~ 1.0 -- see stage_firncover_compaction.py, which stages it as evidence).
Derive the assimilation sigma from:
  - interannual scatter: year-over-year compaction increments at matched
    day-of-year (removes the seasonal cycle) -> interannual sd of the rate;
  - cross-instrument representativeness: two instruments (30, 33) sit at
    nearly the same depth (~15.7 m) -> their rate difference is a direct
    site-repr estimate;
  - the material-interval-vs-model-column approximation (top = install
    surface ~ model surface) is small (top 1 m compacts negligibly) but
    folded into a floor.

This is the script that WRITES the assimilated file: it emits
data/firncover_summit_compaction.csv with `instrument_ID, install,
ztop_mean_m, zbot_mean_m, record_years, rate_m_yr, sigma_m_yr` -- exactly the
columns tutorials/summit/config.py reads for the `compaction` obs block.

Source HDF5: DataONE doi:10.18739/A25X25D7M (FirnCoverData_2_0_2021_07_30.h5);
point FIRNCOVER_H5 at it, or drop it at the default path below. See
data/README.md for the re-download URL.

Run: /home/andrew/venv-firedrake-2026/bin/python \
       tutorials/summit/diagnostics/derive_compaction_sigma.py
"""
import os
from pathlib import Path

import h5py
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
H5 = Path(os.environ.get("FIRNCOVER_H5",
                         DATA / "raw" / "FirnCoverData_2_0_2021_07_30.h5"))
if not H5.exists():
    raise SystemExit(f"FirnCover HDF5 not found at {H5} — set FIRNCOVER_H5 "
                     f"or download it there (see {DATA/'README.md'})")
OUT = DATA / "firncover_summit_compaction.csv"

f = h5py.File(H5)
comp = pd.DataFrame(np.array(f["FirnCover/Compaction_Daily"]))
meta = pd.DataFrame(np.array(f["FirnCover/Compaction_Instrument_Metadata"]))
for df in (comp, meta):
    for c in df.columns:
        if df[c].dtype == object: df[c] = df[c].str.decode("utf-8", errors="ignore")
su = meta[meta.sitename.str.contains("Summit", case=False)]

rows = []
for _, m in su.iterrows():
    iid = m.instrument_ID
    d = comp[comp.instrument_id == iid].copy()
    t = pd.to_datetime(d["daynumber_YYYYMMDD"].astype(str).str.split(".").str[0],
                       format="%Y%m%d", errors="coerce")
    L = d["compaction_borehole_length_m"].astype(float).values
    zt = d["borehole_depth_top_m"].astype(float).abs().values
    zb = d["borehole_depth_bottom_m"].astype(float).abs().values
    ok = np.isfinite(L) & t.notna().values
    t, L, zt, zb = t[ok], L[ok], zt[ok], zb[ok]
    if len(L) < 100:
        print(f"  inst {iid}: only {len(L)} usable days, skip"); continue
    ty = (t - t.iloc[0]).dt.days.values / 365.25
    # full-record linear rate
    A = np.vstack([ty, np.ones_like(ty)]).T
    rate = np.linalg.lstsq(A, L, rcond=None)[0][0]
    # interannual increments: match points ~365 days apart
    dvals = []
    tv = ty
    for i in range(len(tv)):
        j = np.searchsorted(tv, tv[i] + 1.0)
        if j < len(tv) and abs(tv[j] - tv[i] - 1.0) < 0.05:
            dvals.append(L[j] - L[i])   # 1-yr compaction (m/yr, negative)
    dvals = np.array(dvals)
    inter_sd = float(np.std(dvals)) if len(dvals) > 2 else np.nan
    rows.append(dict(iid=int(iid), install=str(m.installation_daynumber_YYYYMMDD),
                     zt=float(np.nanmean(zt)), zb=float(np.nanmean(zb)),
                     span=float(ty[-1]), rate=float(rate),
                     n_ann=len(dvals), inter_sd=inter_sd))
    print(f"  inst {iid}: zbot {rows[-1]['zb']:.1f} m  rate {rate*1000:+.1f} mm/yr  "
          f"span {ty[-1]:.1f} yr  interann sd {1000*inter_sd:.1f} mm/yr (n={len(dvals)})")

if not rows:
    raise SystemExit(f"no Summit instrument in {H5} has a usable record — nothing to write")
R = pd.DataFrame(rows)
# cross-instrument repr from the ~same-depth pair. Absent at another site (or
# under a filtered/updated record) the floor falls back to interannual scatter
# alone rather than failing.
same = R[(R.zb > 15) & (R.zb < 17)]
pair_diff = 0.0
if len(same) >= 2:
    pair_diff = float(np.abs(np.diff(same.rate.values)).max())
    print(f"\ncross-instrument (2 @ ~15.7 m): rates differ by {1000*pair_diff:.1f} mm/yr "
          f"-> repr ~ {1000*pair_diff/np.sqrt(2):.1f} mm/yr")
else:
    print(f"\ncross-instrument: no ~same-depth pair ({len(same)} instrument(s) at "
          f"15-17 m) -> repr floor from interannual scatter only")
med_inter = np.nanmedian(R.inter_sd) if np.isfinite(R.inter_sd).any() else 0.0
print(f"median interannual sd: {1000*med_inter:.1f} mm/yr")
# proposed sigma: max(interannual sd, repr floor, 8% of |rate|)
floor = max(med_inter, pair_diff/np.sqrt(2))
R["sigma"] = np.maximum.reduce([
    np.where(np.isfinite(R.inter_sd), R.inter_sd, floor),
    np.full(len(R), floor),
    0.08*np.abs(R.rate)])
print("\nproposed per-instrument sigma (m/yr):")
for _, r in R.iterrows():
    print(f"  zbot {r.zb:5.1f} m: rate {r.rate*1000:+.1f}  sigma {r.sigma*1000:.1f} mm/yr "
          f"({100*r.sigma/abs(r.rate):.0f}%)")

out = pd.DataFrame(dict(
    instrument_ID=R.iid.astype(int), install=R.install,
    ztop_mean_m=R.zt.round(6), zbot_mean_m=R.zb.round(6),
    record_years=R.span.round(2), rate_m_yr=R.rate.round(6),
    sigma_m_yr=R.sigma.round(6))).sort_values("zbot_mean_m")
_bad = ~(np.isfinite(out.rate_m_yr) & (out.sigma_m_yr > 0))
if _bad.any():
    raise SystemExit(f"refusing to write {OUT}: non-finite rate or non-positive sigma for "
                     f"instrument(s) {out.instrument_ID[_bad].tolist()}")
out.to_csv(OUT, index=False)
print(f"\nwrote {OUT}")
