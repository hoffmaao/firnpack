"""Stage the raw Summit compaction-rate fits from the FirnCover HDF5.

Each instrument measures the length of a MATERIAL interval: anchor at the
borehole bottom, coil at the install-date surface (which then buries). This
script stages the per-instrument RATE columns and the fit diagnostics:
  rate_m_yr    : linear fit of daily length vs time over the record
  rate_se_m_yr : formal slope s.e. from the daily residuals, with an AR(1)
                 independence correction
  resid_sd_mm  : residual scatter about the fit
  lag1         : lag-1 autocorrelation of the daily residuals
  ztop/zbot    : record means of the tracked material-interval endpoints

rate_se_m_yr is staged as EVIDENCE, not as an assimilation sigma: lag1 comes
out at ~1.0 (the daily residuals are a seasonal cycle, not noise), so the
formal s.e. is meaningless and is rejected. The sigma that is actually
assimilated is derived separately -- see derive_compaction_sigma.py, which
writes data/firncover_summit_compaction.csv.

Source HDF5: DataONE doi:10.18739/A25X25D7M (FirnCoverData_2_0_2021_07_30.h5);
point FIRNCOVER_H5 at it, or drop it at the default path below. See
data/README.md for the re-download URL.

Run: /home/andrew/venv-firedrake-2026/bin/python \
       tutorials/summit/diagnostics/stage_firncover_compaction.py
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
OUT = DATA / "firncover_summit_compaction_raw.csv"

f = h5py.File(H5)
comp = pd.DataFrame(np.array(f["FirnCover/Compaction_Daily"]))
meta = pd.DataFrame(np.array(f["FirnCover/Compaction_Instrument_Metadata"]))
for df in (comp, meta):
    for c in df.columns:
        if df[c].dtype == object: df[c] = df[c].str.decode("utf-8", errors="ignore")
print("Compaction_Daily columns:", comp.columns.tolist())
su_ids = meta[meta.sitename.str.contains("Summit", case=False)].instrument_ID.tolist()
print("Summit instruments:", su_ids)

rows = []
for iid in su_ids:
    m = meta[meta.instrument_ID == iid].iloc[0]
    d = comp[comp.instrument_id == iid].copy()
    if len(d) < 100:
        print(f"  inst {iid}: only {len(d)} days, skip"); continue
    # find the length column
    t = pd.to_datetime(d["daynumber_YYYYMMDD"].astype(str).str.split(".").str[0],
                       format="%Y%m%d", errors="coerce")
    L = d["compaction_borehole_length_m"].astype(float).values
    ztop = d["borehole_depth_top_m"].astype(float).values
    zbot = d["borehole_depth_bottom_m"].astype(float).values
    ok = np.isfinite(L) & t.notna().values
    t, L, ztop, zbot = t[ok], L[ok], ztop[ok], zbot[ok]
    ty = (t - t.iloc[0]).dt.days.values / 365.25
    # robust-ish linear fit
    A = np.vstack([ty, np.ones_like(ty)]).T
    coef, *_ = np.linalg.lstsq(A, L, rcond=None)
    resid = L - A @ coef
    # AR(1) effective sample size for the slope error
    r1 = np.corrcoef(resid[:-1], resid[1:])[0, 1] if len(resid) > 2 else 0.0
    neff = len(L) * (1 - r1) / (1 + r1) if r1 < 1 else 2.0
    se = np.sqrt(np.var(resid, ddof=2) / max(neff - 2, 1) / max(np.var(ty) * len(ty), 1e-12) * len(ty) / max(neff, 2))
    span = ty[-1]
    rows.append(dict(instrument_ID=int(iid), install=str(m.installation_daynumber_YYYYMMDD),
                     z_bottom_m=float(-m.borehole_bottom_from_surface_m),
                     record_years=round(float(span), 2), n_days=len(L),
                     L0_m=float(-m.borehole_initial_length_m),
                     ztop_mean_m=float(np.nanmean(np.abs(ztop))),
                     zbot_mean_m=float(np.nanmean(np.abs(zbot))),
                     rate_m_yr=float(coef[0]), rate_se_m_yr=float(se),
                     resid_sd_mm=float(np.std(resid, ddof=2) * 1000),
                     lag1=float(r1)))
    print(f"    z_bot {rows[-1]['z_bottom_m']:.1f} m: rate {coef[0]*1000:+.1f} mm/yr "
          f"over {span:.1f} yr (resid sd {rows[-1]['resid_sd_mm']:.1f} mm, lag1 {r1:+.2f}, "
          f"se {se*1000:.2f} mm/yr)")

out = pd.DataFrame(rows).sort_values("zbot_mean_m")
out.to_csv(OUT, index=False)
print(f"\nwrote {OUT} (raw fits; lag1 ~ 1 => rate_se_m_yr is NOT the "
      f"assimilation sigma — see derive_compaction_sigma.py)")
