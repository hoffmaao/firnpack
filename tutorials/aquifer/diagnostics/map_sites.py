"""Map the southeast Greenland firn aquifer sites against ERA5 melt.

Answers two questions before any column is run:

1. **Where exactly are the observed aquifers?** Coordinates below are from the
   literature, not from memory - see SITES.
2. **Does ERA5 put melt where the aquifers are?** The column runs need a melt
   forcing, and ERA5's melt at the core site came out an order of magnitude
   below what an aquifer needs. Mapping the regional field shows whether that
   is a local sampling artifact or the whole field being low.

Reads the regional ERA5 extraction from ``archive/`` and writes a figure to
``figures/`` plus a small site table to ``data/``.

Run: python tutorials/aquifer/diagnostics/map_sites.py
"""
from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent.parent   # the tutorial directory
RAW = HERE.parent.parent / "archive" / "aquifer" / "sources"
FIGURES = HERE / "figures"
DATA = HERE / "data"

G0 = 9.80665

# Published site positions.
#   PFA-13   Koenig et al. (2014), the discovery core: water found 12-37 m down.
#   Helheim  Montgomery et al. (2017, Front. Earth Sci.) seismic survey, upper
#            Helheim: water table 10-20 m, aquifer base 27.7 +/- 2.9 m,
#            thickness 11.5 +/- 5.5 m.
SITES = {
    "PFA-13 core\n(Koenig 2014)": dict(lat=66.18, lon=-39.04, elev=1563,
                                       wt=(12.0, 37.0)),
    "Helheim survey\n(Montgomery 2017)": dict(lat=66.36, lon=-39.33, elev=1518,
                                              wt=(10.0, 27.7)),
}

MELT = LinearSegmentedColormap.from_list(
    "melt", ["#f7f7f5", "#f6dfc9", "#f0b183", "#e07b3c", "#a8481a"])
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#d8d8d3"


def open_regional():
    src = RAW / "ERA5_monthly_SEG_region_1990_2024.nc"
    if not src.exists():
        raise SystemExit(f"missing {src}; run the regional fetch first")
    if not zipfile.is_zipfile(src):
        return xr.open_dataset(src)
    out = src.parent / (src.stem + "_un")
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(out)
    parts = sorted(out.glob("*.nc"))
    ds = xr.open_dataset(parts[0])
    for extra in parts[1:]:
        ds = xr.merge([ds, xr.open_dataset(extra)],
                      compat="override", join="override")
    return ds


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    ds = open_regional()
    tname = "valid_time" if "valid_time" in ds.coords else "time"

    days = ds[tname].dt.days_in_month
    # ERA5 monthly means of accumulated fields are mean daily rates
    melt = (ds["smlt"] * days).groupby(ds[tname].dt.year).sum().mean("year")
    accum = (ds["sf"] * days).groupby(ds[tname].dt.year).sum().mean("year")
    t2m = ds["t2m"].mean(tname) - 273.15
    elev = ds["z"].isel({tname: 0}) / G0 if "z" in ds else None

    lon = ds["longitude"].values
    lat = ds["latitude"].values

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), sharey=True)

    for ax, field, label, cmap, vmax in (
        (axes[0], melt, "mean annual melt (m w.e. yr$^{-1}$)", MELT, None),
        (axes[1], accum, "mean annual snowfall (m w.e. yr$^{-1}$)", "Blues", None),
    ):
        im = ax.pcolormesh(lon, lat, field.values, cmap=cmap, shading="auto",
                           vmin=0.0, vmax=vmax)
        if elev is not None:
            ev = elev.values
            # Grey out everything at or below sea level. ERA5's ocean cells
            # otherwise carry a dense tangle of 0 m contours that reads as
            # structure and obscures the ice-sheet gradient the map is about.
            ax.contourf(lon, lat, (ev <= 1.0).astype(float), levels=[0.5, 1.5],
                        colors=["#eceae6"], zorder=3)
            cs = ax.contour(lon, lat, np.where(ev > 1.0, ev, np.nan),
                            levels=np.arange(400, 3400, 400),
                            colors=MUTED, linewidths=0.6, alpha=0.75, zorder=4)
            ax.clabel(cs, fmt="%d", fontsize=7, colors=MUTED)
        # Each label is pushed AWAY from the other site: PFA-13 is the
        # southern point so its label goes down, Helheim's goes up. Offsetting
        # them the other way round put each label against the other's marker,
        # which reads as mislabelling.
        offsets = {True: (12, 16), False: (12, -24)}   # keyed on "is northern"
        lat_max = max(v["lat"] for v in SITES.values())
        for name, s in SITES.items():
            off = offsets[s["lat"] >= lat_max]
            ax.plot(s["lon"], s["lat"], "o", ms=9, mfc="white",
                    mec=INK, mew=2.0, zorder=6)  # noqa: E128
            ax.annotate(name, (s["lon"], s["lat"]), xytext=off,
                        textcoords="offset points", fontsize=8.5, color=INK,
                        zorder=7,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="none", alpha=0.82))
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(label, fontsize=9, color=MUTED)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_visible(False)
        ax.set_xlabel("longitude", fontsize=9, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    axes[0].set_ylabel("latitude", fontsize=9, color=MUTED)
    fig.suptitle("ERA5 1990-2024 over southeast Greenland, with the observed "
                 "firn aquifer sites", fontsize=12, color=INK)
    fig.tight_layout()
    out = FIGURES / "aquifer_site_map.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)

    rows = []
    print(f"{'site':<22} {'lat':>7} {'lon':>8} {'elev':>6} {'ERA5 z':>7} "
          f"{'T':>7} {'accum':>7} {'melt':>7}")
    for name, s in SITES.items():
        pick = dict(latitude=s["lat"], longitude=s["lon"])
        m = float(melt.sel(**pick, method="nearest"))
        a = float(accum.sel(**pick, method="nearest"))
        t = float(t2m.sel(**pick, method="nearest"))
        z = float(elev.sel(**pick, method="nearest")) if elev is not None else np.nan
        flat = name.replace("\n", " ")
        print(f"{flat:<22} {s['lat']:7.2f} {s['lon']:8.2f} {s['elev']:6.0f} "
              f"{z:7.0f} {t:7.2f} {a:7.3f} {m:7.3f}")
        rows.append(dict(site=flat, lat=s["lat"], lon=s["lon"],
                         site_elev_m=s["elev"], era5_elev_m=round(z, 1),
                         era5_T_C=round(t, 2), era5_accum_m_we_yr=round(a, 3),
                         era5_melt_m_we_yr=round(m, 3),
                         obs_water_table_top_m=s["wt"][0],
                         obs_water_table_base_m=s["wt"][1]))

    csv_path = DATA / "aquifer_sites.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    mmax = float(melt.max())
    print(f"\nregional ERA5 melt: max {mmax:.3f} m w.e./yr over the domain")
    print(f"  the sites sit at {rows[0]['era5_melt_m_we_yr']:.3f} and "
          f"{rows[1]['era5_melt_m_we_yr']:.3f} m w.e./yr")
    print(f"Saved {out}\nSaved {csv_path}")


if __name__ == "__main__":
    main()
