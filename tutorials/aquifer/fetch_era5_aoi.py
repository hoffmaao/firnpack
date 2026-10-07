"""3-hourly ERA5 for a surface energy balance over the SE Greenland aquifer AOI.

What this fetches and why
-------------------------
Everything the surface energy balance in :mod:`firnpack.surface_energy` needs,
plus ERA5's own melt so the two can be compared rather than one silently
trusted.

The *downward* radiation components are fetched, not the net ones, because the
whole point is to apply our own albedo and our own skin temperature: ERA5's net
fluxes already carry its surface scheme's answers for both. ``forecast_albedo``
comes along as an independent check on the albedo parameterisation, which is
otherwise the largest free knob in the balance.

  group A   2m_temperature                    skin-temperature solve, Q_H
            2m_dewpoint_temperature           -> specific humidity, Q_L
            surface_pressure                  air density, humidity
            snowfall                          accumulation, albedo reset
            total_precipitation               rain = tp - sf, gives Q_R

  group B   surface_solar_radiation_downwards SW_in
            surface_thermal_radiation_downwards LW_in
            10m_u_component_of_wind           }  wind speed -> turbulent
            10m_v_component_of_wind           }  exchange
            snowmelt                          ERA5's own melt, for comparison
            forecast_albedo                   independent albedo

Why two groups, and why 3-hourly rather than hourly
---------------------------------------------------
CDS prices a request as timesteps x variables and refuses anything much over
about 20000 fields for this dataset - measured, not assumed: 2 variables x 8784
hourly steps (17568) is accepted, 3 x 8784 (26280) and 5 x 8760 (43800) are
both refused with "cost limits exceeded".

At hourly resolution that caps a request at two variables, so eleven variables
over 85 years would be 510 separate queued requests. Three-hourly sampling
divides the field count by three and brings it to 170: group A is 5 x 2920 =
14600 fields and group B is 6 x 2920 = 17520, both inside the measured limit.

Three-hourly keeps eight samples a day, which still resolves the diurnal melt
cycle - the thing monthly means destroy, and the whole reason for going to
sub-monthly forcing. It is also the resolution regional models are commonly
archived at.

Why sub-monthly, and why an area
--------------------------------
July's monthly *mean* 2 m temperature at the core site is -1.0 C, so anything
keyed on monthly means sees almost no melt - ERA5's own monthly ``snowmelt``
gives 0.08 m w.e./yr where an aquifer needs of order 0.5. The melt lives in
individual warm hours. Area-averaging is *not* what fixes this: on the monthly
product every elevation band from 800 to 2000 m returns 0.09-0.15 m w.e./yr.
Averaging is done because the aquifer is a belt rather than a point, and
because the two published core sites land in ERA5 cells whose elevations
bracket the true ones from opposite sides.

Resumability
------------
One request per group per year, skipped when its file exists. That is 2 x 85
requests through the CDS queue, so this is a background job measured in hours;
re-run until it stops printing "requesting". ``--reduce`` works on whatever has
arrived so far.

Run:
    python tutorials/aquifer/fetch_era5_aoi.py                 # fetch + reduce
    python tutorials/aquifer/fetch_era5_aoi.py --reduce        # reduce only
    python tutorials/aquifer/fetch_era5_aoi.py --years 1990 2024
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).parent
RAW = HERE.parent.parent / "archive" / "aquifer" / "sources"
DATA = HERE / "data"

# Box around the southeast Greenland aquifer belt; contains both core sites.
AOI = dict(north=67.0, west=-40.5, south=65.5, east=-38.0)
ELEV_BAND = (1200.0, 1800.0)     # m; the band the observed aquifers occupy

# Exclude any cell that is not wholly glacier ice.
#
# ERA5 is ~31 km, so a single cell can straddle the ice sheet and a nunatak or
# a coastal mountain, and its 2 m temperature, albedo and radiation are then a
# mixture of ice and rock. Rock warms far more in summer, so including such a
# cell biases the melt forcing high for reasons that have nothing to do with
# firn.
#
# ERA5's own climatological albedo is the discriminator: it assigns a fixed
# 0.85 over permanent ice, and a cell reads lower only if some non-ice surface
# is mixed in. Over the full record in late summer (Aug-Sep, when rock is most
# exposed) the 22 cells in the elevation band split cleanly: 18 sit at
# 0.848-0.850, and four fall away at 0.749, 0.764, 0.793 and 0.808 - all four
# in the same northeast corner against the mountains.
#
# The cut is at 0.845, just under the permanent-ice value, because the brief is
# to exclude a cell with *any* rock in it rather than one that is mostly rock.
# A looser 0.80 kept the 0.808 cell, which is 5 per cent short of pure ice and
# so is partly rock by exactly this test. The gap between 0.808 and 0.848 is
# wide enough that the cut is not sensitive to where in it the line is drawn.
ICE_ALBEDO_MIN = 0.845
ICE_MASK_MONTHS = (8, 9)

YEAR0, YEAR1 = 1940, 2024
G0 = 9.80665

GROUPS = {
    "A": ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure",
          "snowfall", "total_precipitation"],
    "B": ["surface_solar_radiation_downwards",
          "surface_thermal_radiation_downwards",
          "10m_u_component_of_wind", "10m_v_component_of_wind",
          "snowmelt", "forecast_albedo"],
}

# ERA5 short names, by group, in the same order.
SHORT = {
    "A": ["t2m", "d2m", "sp", "sf", "tp"],
    "B": ["ssrd", "strd", "u10", "v10", "smlt", "fal"],
}

# ERA5 accumulated fields carry the accumulation over the PRECEDING HOUR, so
# sampling every third hour and summing returns a third of the true total. The
# 2012 test made that visible: 0.355 m w.e./yr of snowfall against 1.061 from
# the full hourly record, and 0.030 of ERA5 melt against 0.09 from the monthly
# product - both exactly a factor of three low. Each sample is therefore taken
# to represent its 3-hour block and scaled accordingly.
#
# That is unbiased but noisy for an intermittent field like snowfall: one hour
# in three is observed, so a single year's total carries sampling error even
# though a multi-decade mean does not. Instantaneous fields (temperature, wind,
# pressure, albedo) need no such scaling - their sample mean is already the
# right estimate.
ACCUMULATED = {"sf", "tp", "smlt", "ssrd", "strd"}
SAMPLE_HOURS = 3.0


def _client():
    import cdsapi
    return cdsapi.Client()


def _open_parts(path):
    """Return the NetCDF streams inside a CDS payload, unmerged.

    They are deliberately not merged here. CDS splits a request across streams
    (instantaneous vs accumulated), and for some years those streams carry a
    different number of time steps; merging them with an aligning join then
    fails outright ("cannot align objects ... that don't have the same size").
    Each stream is reduced on its own time axis and the results are joined by
    timestamp, which sidesteps the alignment entirely.
    """
    import xarray as xr
    if not zipfile.is_zipfile(path):
        return [xr.open_dataset(path)]
    out = path.parent / (path.stem + "_un")
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(out)
    return [xr.open_dataset(p) for p in sorted(out.glob("*.nc"))]


def _open(path):
    """Single merged dataset, for callers that know the streams align."""
    parts = _open_parts(path)
    if len(parts) == 1:
        return parts[0]
    import xarray as xr
    return xr.merge(parts, compat="override", join="override")


def fetch_orography():
    out = RAW / "ERA5_orog_aoi.nc"
    if out.exists():
        return out
    RAW.mkdir(parents=True, exist_ok=True)
    _client().retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis", "variable": ["geopotential"],
        "year": "2020", "month": "01", "day": "01", "time": "00:00",
        "area": [AOI["north"], AOI["west"], AOI["south"], AOI["east"]],
        "data_format": "netcdf"}, str(out))
    return out


def fetch(group, year):
    out = RAW / f"ERA5_seb_aoi_{group}_{year}.nc"
    if out.exists():
        return out
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"  requesting {group} {year} ...", flush=True)
    _client().retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis",
        "variable": GROUPS[group],
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
        "area": [AOI["north"], AOI["west"], AOI["south"], AOI["east"]],
        "data_format": "netcdf",
    }, str(out))
    print(f"  wrote {out.name} ({out.stat().st_size/1e6:.0f} MB)", flush=True)
    return out


def _largest_contiguous(mask):
    """Keep only the largest 4-connected group of selected cells.

    The albedo test removes a cell whose own late-summer albedo betrays bare
    rock, but not one that merely sits in the same marginal zone and happens
    to clear the threshold. The belt is a connected band along the flow
    divide, so an isolated cell with no retained neighbour is an outlier
    rather than part of it: -38.00 E, 66.75 N (1660 m, albedo 0.850 against a
    0.845 threshold) sat immediately east of the excluded rock row with its
    nearest retained neighbour 0.75 degrees away, and contributed marginal
    ice to the average. Requiring connectivity drops it, and drops the next
    such cell without another hand-edit.
    """
    lab = np.zeros(mask.shape, dtype=int)
    groups = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i, j] or lab[i, j]:
                continue
            groups.append(0)
            stack = [(i, j)]
            lab[i, j] = len(groups)
            while stack:
                a, b = stack.pop()
                groups[-1] += 1
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if (0 <= x < mask.shape[0] and 0 <= y < mask.shape[1]
                            and mask[x, y] and not lab[x, y]):
                        lab[x, y] = len(groups)
                        stack.append((x, y))
    if not groups:
        return mask
    keep = int(np.argmax(groups)) + 1
    return lab == keep


def _ice_mask(z):
    """Cells that are wholly glacier ice, by late-summer ERA5 albedo.

    Returns None when no group-B file carries albedo yet, in which case the
    caller falls back to the elevation band alone.
    """
    import xarray as xr
    fields = []
    for path in sorted(RAW.glob("ERA5_seb_aoi_B_*.nc")):
        for ds in _open_parts(path):
            if "fal" not in ds:
                continue
            tname = "valid_time" if "valid_time" in ds.coords else "time"
            sel = ds["fal"].sel(
                {tname: ds[tname].dt.month.isin(list(ICE_MASK_MONTHS))})
            if sel.sizes.get(tname, 0):
                fields.append(sel.mean(tname))
    if not fields:
        return None
    fal = xr.concat(fields, dim="_y").mean("_y")
    fal = fal.sel(latitude=z.latitude, longitude=z.longitude, method="nearest")
    return fal >= ICE_ALBEDO_MIN


def reduce_all():
    """Area-average both groups over the elevation-masked AOI and join."""
    import pandas as pd

    orog = _open(fetch_orography()).squeeze(drop=True)
    z = orog["z"] / G0
    mask = (z >= ELEV_BAND[0]) & (z <= ELEV_BAND[1])
    if int(mask.sum()) == 0:
        raise SystemExit(f"elevation band {ELEV_BAND} selects no cells")

    ice = _ice_mask(z)
    if ice is not None:
        dropped = int((mask & ~ice).sum())
        mask = mask & ice
        if dropped:
            print(f"  excluded {dropped} cell(s) with bare rock "
                  f"(late-summer albedo < {ICE_ALBEDO_MIN})")

    # and drop cells that survive the albedo test but sit apart from the belt
    connected = _largest_contiguous(mask.values)
    n_iso = int(mask.sum()) - int(connected.sum())
    if n_iso:
        print(f"AOI: dropping {n_iso} isolated cell(s) not connected to the belt")
    mask = mask & xr.DataArray(connected, coords=mask.coords, dims=mask.dims)

    w = np.cos(np.deg2rad(orog["latitude"])).broadcast_like(z).where(mask)
    print(f"AOI: {int(mask.sum())} cells, {ELEV_BAND[0]:.0f}-{ELEV_BAND[1]:.0f} m"
          f" (mean {float(z.where(mask).mean()):.0f} m)")

    per_group = {}
    for group, shorts in SHORT.items():
        frames = []
        n_files = 0
        for path in sorted(RAW.glob(f"ERA5_seb_aoi_{group}_*.nc")):
            n_files += 1
            for ds in _open_parts(path):
                tname = "valid_time" if "valid_time" in ds.coords else "time"
                sub = ds.sel(latitude=z.latitude, longitude=z.longitude)
                cols = {"time": ds[tname].values}
                for name in shorts:
                    if name in sub:
                        cols[name] = sub[name].where(mask).weighted(
                            w.fillna(0)).mean(("latitude", "longitude")).values
                if len(cols) > 1:
                    frames.append(pd.DataFrame(cols))
        if frames:
            # streams within a year share timestamps but carry different
            # variables, so group by time and take the first non-null of each
            merged = pd.concat(frames).groupby("time", as_index=False).first()
            per_group[group] = merged
            print(f"  group {group}: {len(merged)} steps from {n_files} files")

    if not per_group:
        raise SystemExit("no chunks downloaded yet")
    df = None
    for g in sorted(per_group):
        df = per_group[g] if df is None else df.merge(per_group[g], on="time",
                                                      how="outer")
    df = df.sort_values("time")
    # Scale the accumulated fields from a 1-hour accumulation to the 3-hour
    # block each sample stands for (see ACCUMULATED above), so downstream code
    # never has to know the sampling interval.
    for name in ACCUMULATED:
        if name in df:
            df[name] = df[name] * SAMPLE_HOURS
    df["sample_hours"] = SAMPLE_HOURS
    DATA.mkdir(parents=True, exist_ok=True)
    # .gz: pandas compresses and decompresses by extension, and the raw CSV is
    # 48 MB of mostly float noise against 20 MB compressed
    out = DATA / "era5_hourly_aoi_seb.csv.gz"
    df.to_csv(out, index=False)

    n_yr = max(len(np.unique(pd.DatetimeIndex(df["time"]).year)), 1)
    print(f"\n{df['time'].iloc[0]} .. {df['time'].iloc[-1]}  ({n_yr} years, "
          f"{len(df)} hours)")
    have = [c for c in df.columns if c != "time"]
    print(f"  variables: {', '.join(have)}")
    if "t2m" in df:
        T = df["t2m"].values - 273.15
        print(f"  mean T                {np.nanmean(T):8.2f} C")
    # The accumulated columns were already scaled to 3-hour block totals above,
    # so these must not scale again: sum them straight for totals, and divide
    # by the block length in seconds for a mean flux.
    block_s = SAMPLE_HOURS * 3600.0
    if "sf" in df:
        print(f"  snowfall              {np.nansum(df['sf'])/n_yr:8.3f} m w.e./yr")
    if "smlt" in df:
        print(f"  ERA5 snowmelt         {np.nansum(df['smlt'])/n_yr:8.3f} m w.e./yr")
    if "ssrd" in df:
        print(f"  mean SW down          {np.nanmean(df['ssrd'])/block_s:8.1f} W/m2")
    if "strd" in df:
        print(f"  mean LW down          {np.nanmean(df['strd'])/block_s:8.1f} W/m2")
    if "fal" in df:
        print(f"  ERA5 albedo           {np.nanmean(df['fal']):8.3f}")
    print(f"\n  wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduce", action="store_true")
    ap.add_argument("--years", type=int, nargs=2, default=(YEAR0, YEAR1))
    a = ap.parse_args()
    if not a.reduce:
        fetch_orography()
        for year in range(a.years[0], a.years[1] + 1):
            for group in GROUPS:
                fetch(group, year)
    reduce_all()


if __name__ == "__main__":
    main()
