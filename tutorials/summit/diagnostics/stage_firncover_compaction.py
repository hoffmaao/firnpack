"""Extract Summit compaction-rate observations from the FirnCover HDF5.

Each instrument measures the length of a MATERIAL interval: anchor at the
borehole bottom, coil at the install-date surface (which then buries). The
observable staged here is the mean shortening rate over the record, with:
  rate  : robust linear fit of daily length vs time
  sig   : residual scatter about the fit (Zeising's method — the stated
          instrument precision is optimistic; the scatter carries wind
          pumping, thermal cycling, wire artifacts), with an AR(1)
          independence correction on the daily residuals
  depth : mean top/bottom depths of the material interval over the record
          (top = burial/2 estimated from the site's accumulation)
"""
import h5py
import numpy as np, pandas as pd

f = h5py.File("/home/andrew/.claude/jobs/f7fe0a6b/tmp/FirnCoverData_2_0_2021_07_30.h5")
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

out = pd.DataFrame(rows)
out.to_csv("/home/andrew/.claude/jobs/f7fe0a6b/tmp/firncover_summit_compaction.csv", index=False)
print("\nwrote tmp/firncover_summit_compaction.csv")
