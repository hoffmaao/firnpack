"""Fetch ERA5 forcing for the southeast Greenland firn aquifer site.

Follows the provenance pattern the other case studies use: the raw NetCDF lands
in ``archive/`` (untracked, large), and a small processed CSV is written into
``tutorials/aquifer/data/`` (tracked) with a README recording where it came
from.

Site
----
The aquifer discovered by Forster et al. (2014) and cored by Koenig et al.
(2014) / Miege et al. (2016) in the Helheim catchment of southeast Greenland.
The default coordinates below are the discovery area, not a surveyed borehole
position - override with ``--lat/--lon`` for a specific core.

What ERA5 actually gives here
-----------------------------
Measured at (66.18 N, 39.04 W), 1990-2024:

    mean annual T       -10.3 C
    accumulation         1.39 m w.e./yr
    melt                 0.08 m w.e./yr      <- the problem
    melt / accumulation  0.06

The accumulation is fine: 1.39 m w.e./yr sits inside the 1-2.5 m w.e./yr these
sites are reported at, so the orographic under-accumulation one would expect at
31 km does not show up. It is the **melt** that is unusable - roughly an order
of magnitude below what an aquifer needs, and less than the 0.15 m w.e./yr case
in which this model refreezes everything and forms no aquifer at all.

Nor is it an elevation artifact: the ERA5 cell sits at 1433 m against a site
near 1550 m, so a lapse correction makes the site *colder* still (-11.1 C) and
the melt problem worse. The likely cause is ERA5's surface scheme over
glaciated points, which has no real firn/percolation model; ERA5 melt over
Greenland is known to be low. RACMO2.3p2 or MAR at 1-5.5 km is what the
literature uses for this question (Kuipers Munneke et al. 2014), and neither is
openly downloadable.

The script prints these numbers so the bias is visible rather than silently
inherited, and ``--scale-accum`` applies an explicit correction if wanted.

Run:
    python tutorials/aquifer/fetch_era5.py --years 1990 2024
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).parent
RAW_DIR = HERE.parent.parent / "archive" / "aquifer" / "sources"
DATA_DIR = HERE / "data"

# Forster et al. (2014) discovery area, Helheim catchment. Approximate.
DEFAULT_LAT, DEFAULT_LON = 66.18, -39.04

VARIABLES = [
    "2m_temperature",       # t2m   [K]      thermal boundary condition
    "snowfall",             # sf    [m w.e.] accumulation
    "snowmelt",             # smlt  [m w.e.] melt - ERA5 gives this directly,
                            #                so no degree-day parameterisation
    "evaporation",          # e     [m w.e.] sublimation (negative = loss)
]


def fetch(lat, lon, y0, y1, force=False):
    import cdsapi

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"ERA5_monthly_segreenland_{y0}_{y1}.nc"
    if out.exists() and not force:
        print(f"already have {out}")
        return out

    d = 0.5  # a small box; ERA5 is ~0.28 deg so this is a few cells
    print(f"requesting ERA5 monthly means {y0}-{y1} at ({lat}, {lon}) ...")
    cdsapi.Client().retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": VARIABLES,
            "year": [str(y) for y in range(y0, y1 + 1)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "area": [lat + d, lon - d, lat - d, lon + d],  # N, W, S, E
            "data_format": "netcdf",
        },
        str(out),
    )
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return out


def _open(nc_path):
    """Open the CDS payload, which may be a zip of per-stream NetCDFs.

    The current CDS API returns a zip when the requested variables span more
    than one internal stream (here: instantaneous 2 m temperature vs the
    accumulated fluxes), even though the request asked for netcdf and the file
    is named .nc.
    """
    import zipfile
    import xarray as xr

    if not zipfile.is_zipfile(nc_path):
        return xr.open_dataset(nc_path)

    extract = nc_path.parent / (nc_path.stem + "_unzipped")
    extract.mkdir(exist_ok=True)
    with zipfile.ZipFile(nc_path) as z:
        z.extractall(extract)
    parts = sorted(extract.glob("*.nc"))
    if not parts:
        raise SystemExit(f"no NetCDF inside {nc_path}")
    # join="override" takes the first file's coordinates for all of them. The
    # two streams carry the same grid but their coordinate values differ in the
    # last bits, and an aligning merge turns that into all-NaN flux fields
    # while leaving temperature intact - a silent, one-sided failure.
    ds = xr.open_dataset(parts[0])
    for extra in parts[1:]:
        ds = xr.merge([ds, xr.open_dataset(extra)],
                      compat="override", join="override")
    return ds


def process(nc_path, lat, lon, scale_accum=None):
    import numpy as np

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = _open(nc_path)
    # nearest cell to the site, then squeeze any singleton dims
    sel = {}
    for name, val in (("latitude", lat), ("longitude", lon)):
        if name in ds.coords:
            sel[name] = val
    pt = ds.sel(**sel, method="nearest") if sel else ds
    pt = pt.squeeze(drop=True)

    tname = "valid_time" if "valid_time" in pt.coords else "time"
    time = pt[tname].values
    days = np.array([np.datetime64(t, "D").astype("datetime64[D]") for t in time])

    def var(*names):
        for n in names:
            if n in pt:
                return np.asarray(pt[n].values, dtype=float).ravel()
        return None

    t2m = var("t2m")
    # ERA5 monthly means of accumulated fields are mean DAILY totals [m/day]
    sf = var("sf")
    smlt = var("smlt")
    evap = var("e")

    import calendar
    ndays = np.array([calendar.monthrange(
        int(str(d)[:4]), int(str(d)[5:7]))[1] for d in days], dtype=float)

    rows = []
    for i, d in enumerate(days):
        rows.append(dict(
            time=str(d),
            year=int(str(d)[:4]), month=int(str(d)[5:7]),
            t2m_K=float(t2m[i]) if t2m is not None else None,
            snowfall_m_we_month=float(sf[i] * ndays[i]) if sf is not None else None,
            snowmelt_m_we_month=float(smlt[i] * ndays[i]) if smlt is not None else None,
            evap_m_we_month=float(evap[i] * ndays[i]) if evap is not None else None,
        ))

    import csv
    csv_path = DATA_DIR / "era5_monthly_segreenland.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    yrs = np.array([r["year"] for r in rows])
    acc = np.array([r["snowfall_m_we_month"] or 0.0 for r in rows])
    melt = np.array([r["snowmelt_m_we_month"] or 0.0 for r in rows])
    tmean = np.nanmean([r["t2m_K"] for r in rows]) - 273.15
    n_yr = len(np.unique(yrs))
    acc_yr, melt_yr = acc.sum() / n_yr, melt.sum() / n_yr

    print(f"\nERA5 at nearest cell to ({lat}, {lon}), {yrs.min()}-{yrs.max()}:")
    print(f"  mean annual T      {tmean:8.2f} C")
    print(f"  mean accumulation  {acc_yr:8.3f} m w.e./yr")
    print(f"  mean melt          {melt_yr:8.3f} m w.e./yr")
    print(f"  melt / accum       {melt_yr/max(acc_yr,1e-9):8.2f}")
    print(f"\n  wrote {csv_path}")
    if acc_yr < 0.8:
        print("  NOTE: southeast Greenland aquifer sites accumulate roughly "
              "1-2.5 m w.e./yr.\n        This is the expected ERA5 low bias at "
              "31 km over steep coastal terrain;\n        use --scale-accum to "
              "correct it explicitly.")
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--years", type=int, nargs=2, default=(1990, 2024))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--scale-accum", type=float, default=None,
                    help="multiply ERA5 accumulation by this factor")
    a = ap.parse_args()
    nc = fetch(a.lat, a.lon, a.years[0], a.years[1], force=a.force)
    process(nc, a.lat, a.lon, scale_accum=a.scale_accum)


if __name__ == "__main__":
    main()
