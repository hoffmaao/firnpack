"""The averaging AOI drawn on MODIS true-colour imagery of SE Greenland.

Shows what the forcing is actually averaged over: the ERA5 cells inside the
1200-1800 m elevation band, against the real ice sheet, its margin and the
fjord system that drains it. Useful as a sanity check that the band picks out
the percolation zone rather than bare rock, sea ice or the high interior.

Imagery is MODIS Terra corrected reflectance from NASA GIBS, which needs no
credentials. The default date is 2012-07-12: nearly cloud-free over the belt,
and 2012 is both the extreme melt year and the year validated against the full
hourly record.

Run: python tutorials/aquifer/diagnostics/map_aoi_optical.py
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FIGURES

HERE = Path(__file__).parent.parent   # the tutorial directory
RAW = HERE.parent.parent / "archive" / "aquifer" / "sources"

AOI = dict(north=67.0, west=-40.5, south=65.5, east=-38.0)
ELEV_BAND = (1200.0, 1800.0)
G0 = 9.80665
ICE_ALBEDO_MIN = 0.845      # keep in step with fetch_era5_aoi.ICE_ALBEDO_MIN
DATE = "2012-07-12"

# Wide frame for context, and the AOI itself for the inset.
WIDE = dict(west=-42.5, east=-36.5, south=64.8, north=67.6)

SITES = {
    "PFA-13 core": dict(lat=66.18, lon=-39.04),
    "Helheim survey": dict(lat=66.36, lon=-39.33),
}

INK, MUTED = "#12120f", "#f5f5f2"
AOI_COLOR, CELL_COLOR, ROCK_COLOR = "#ff4d2e", "#ffd23f", "#7b2d8e"

GIBS = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?"
        "SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
        "&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&SRS=EPSG:4326"
        "&BBOX={west},{south},{east},{north}&WIDTH={w}&HEIGHT={h}"
        "&FORMAT=image/jpeg&TIME={date}")


def imagery(box, date=DATE, w=1400, h=1000):
    RAW.mkdir(parents=True, exist_ok=True)
    tag = f"{box['west']}_{box['east']}_{box['south']}_{box['north']}_{date}"
    path = RAW / f"MODIS_{tag.replace('.', 'p').replace('-', 'm')}.jpg"
    if not path.exists():
        url = GIBS.format(w=w, h=h, date=date, **box)
        urllib.request.urlretrieve(url, path)
    return plt.imread(path)


def orography():
    src = RAW / "ERA5_orog_aoi.nc"
    if not src.exists():
        raise SystemExit(f"missing {src}; run fetch_era5_aoi.py first")
    if zipfile.is_zipfile(src):
        out = src.parent / (src.stem + "_un")
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(out)
        src = sorted(out.glob("*.nc"))[0]
    ds = xr.open_dataset(src).squeeze(drop=True)
    return ds["z"] / G0


def _ice_fraction_mask(z):
    """Cells ERA5 treats as permanent ice, by late-summer albedo.

    Mirrors fetch_era5_aoi so the figure shows the cells actually averaged
    rather than a separate, possibly divergent, definition.
    """
    import xarray as xr
    sys.path.insert(0, str(HERE))
    import fetch_era5_aoi as F
    fields = []
    for path in sorted(RAW.glob("ERA5_seb_aoi_B_*.nc")):
        for ds in F._open_parts(path):
            if "fal" not in ds:
                continue
            t = "valid_time" if "valid_time" in ds.coords else "time"
            sel = ds["fal"].sel({t: ds[t].dt.month.isin([8, 9])})
            if sel.sizes.get(t, 0):
                fields.append(sel.mean(t))
    if not fields:
        return np.ones(z.shape, bool)
    fal = xr.concat(fields, dim="_y").mean("_y").sel(
        latitude=z.latitude, longitude=z.longitude, method="nearest")
    return (fal >= ICE_ALBEDO_MIN).values


def _frame(ax, box, img):
    ax.imshow(img, extent=[box["west"], box["east"], box["south"], box["north"]],
              origin="upper", aspect="auto", interpolation="bilinear")
    ax.set_xlim(box["west"], box["east"])
    ax.set_ylim(box["south"], box["north"])
    ax.tick_params(colors=INK, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#9a9a94")


def _sites(ax, fontsize=9, offsets=((12, 14), (12, -24))):
    lat_max = max(v["lat"] for v in SITES.values())
    for name, s in SITES.items():
        off = offsets[0] if s["lat"] >= lat_max else offsets[1]
        ax.plot(s["lon"], s["lat"], "o", ms=8, mfc="white", mec=INK, mew=1.8,
                zorder=8)
        ax.annotate(name, (s["lon"], s["lat"]), xytext=off,
                    textcoords="offset points", fontsize=fontsize, color=INK,
                    zorder=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                              alpha=0.85))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    z = orography()
    lon = z["longitude"].values
    lat = z["latitude"].values
    dlon = abs(float(lon[1] - lon[0])) if lon.size > 1 else 0.25
    dlat = abs(float(lat[1] - lat[0])) if lat.size > 1 else 0.25
    band = ((z >= ELEV_BAND[0]) & (z <= ELEV_BAND[1])).values
    ice = _ice_fraction_mask(z)
    mask = band & ice            # kept in the average
    rock = band & ~ice           # in the elevation band but rock-contaminated

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6))

    # --- context ---
    ax = axes[0]
    _frame(ax, WIDE, imagery(WIDE))
    ax.add_patch(Rectangle((AOI["west"], AOI["south"]),
                           AOI["east"] - AOI["west"], AOI["north"] - AOI["south"],
                           fill=False, ec=AOI_COLOR, lw=2.2, zorder=6))
    ax.annotate("averaging AOI", (AOI["west"], AOI["north"]), xytext=(4, 6),
                textcoords="offset points", fontsize=10, color=AOI_COLOR,
                weight="bold", zorder=9)
    _sites(ax, fontsize=8.5)
    ax.set_title(f"Southeast Greenland, MODIS true colour {DATE}",
                 fontsize=11, color=INK, loc="left")
    ax.set_xlabel("longitude", fontsize=9, color=INK)
    ax.set_ylabel("latitude", fontsize=9, color=INK)

    # --- the AOI itself, with the cells actually averaged ---
    ax = axes[1]
    pad = 0.35
    zoom = dict(west=AOI["west"] - pad, east=AOI["east"] + pad,
                south=AOI["south"] - pad, north=AOI["north"] + pad)
    _frame(ax, zoom, imagery(zoom, w=1200, h=1000))
    n = n_rock = 0
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            if mask[i, j]:
                n += 1
                fc, ec, al = CELL_COLOR, CELL_COLOR, 0.30
            elif rock[i, j]:
                n_rock += 1
                fc, ec, al = ROCK_COLOR, ROCK_COLOR, 0.42
            else:
                continue
            ax.add_patch(Rectangle((lo - dlon / 2, la - dlat / 2), dlon, dlat,
                                   fc=fc, ec=ec, alpha=al, lw=0.5, zorder=5))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=CELL_COLOR, alpha=0.45, label=f"averaged ({n})"),
                       Patch(fc=ROCK_COLOR, alpha=0.5,
                             label=f"excluded, bare rock ({n_rock})")],
              loc="lower left", fontsize=8.5, framealpha=0.9)
    ax.add_patch(Rectangle((AOI["west"], AOI["south"]),
                           AOI["east"] - AOI["west"], AOI["north"] - AOI["south"],
                           fill=False, ec=AOI_COLOR, lw=2.2, zorder=6))
    _sites(ax)
    ax.set_title(f"{n} ice cells averaged, {n_rock} excluded for bare rock "
                 f"({ELEV_BAND[0]:.0f}-{ELEV_BAND[1]:.0f} m band)",
                 fontsize=11, color=INK, loc="left")
    ax.set_xlabel("longitude", fontsize=9, color=INK)

    fig.suptitle("What the aquifer forcing is averaged over", fontsize=13,
                 color=INK, y=0.99)
    fig.tight_layout()
    out = FIGURES / "aquifer_aoi_optical.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"averaged: {n} cells   excluded as rock: {n_rock}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
