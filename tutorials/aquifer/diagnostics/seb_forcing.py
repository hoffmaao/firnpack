"""Turn the ERA5 AOI extraction into a melt and accumulation forcing.

Runs :class:`firnpack.surface_energy.SurfaceEnergyBalance` over the reduced
ERA5 record and reports the annual energy and mass budget, alongside ERA5's own
``snowmelt`` and the degree-day estimate this replaces. Those two comparisons
are the point: a surface energy balance is only worth the extra machinery if it
lands somewhere defensible between a reanalysis melt scheme known to run low
over Greenland and a degree-day factor that can be tuned to any answer.

Coupling caveat
---------------
``T_firn`` and ``k_eff`` belong to the firn column. Run standalone, this script
prescribes them, so the conductive term is approximate and the melt carries
that approximation. Sensitivity to ``T_firn`` is printed for exactly that
reason. Once the Neumann coupling is wired the column supplies both and this
becomes a diagnostic rather than a forcing generator.

Run: python tutorials/aquifer/diagnostics/seb_forcing.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

HERE = Path(__file__).parent.parent   # the tutorial directory
DATA = HERE / "data"

sys.path.insert(0, str(HERE.parent.parent / "src"))
from firnpack.surface_energy import (  # noqa: E402
    SurfaceEnergyBalance, SurfaceEnergyParameters,
    specific_humidity_from_dewpoint,
)

SECONDS_PER_YEAR = 365.25 * 86400.0

# The aged-firn albedo floor the tracked data/aquifer_annual_forcing.csv was
# built at, and the default here so the command in data/README.md reproduces
# that file. The floor is the dominant control on the melt - at the library
# default of 0.60 every melt value in the table roughly doubles and the melt
# trend the tutorial quotes moves from +23% to +19% - so it is recorded as a
# column in the table rather than left implicit in whoever ran the script.
ANNUAL_TABLE_FLOOR = 0.72


def load(csv_path=None):
    csv_path = Path(csv_path or DATA / "era5_hourly_aoi_seb.csv.gz")
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}; run fetch_era5_aoi.py first")
    df = pd.read_csv(csv_path, parse_dates=["time"])
    step_h = float(df["sample_hours"].iloc[0]) if "sample_hours" in df else 3.0
    block_s = step_h * 3600.0

    out = dict(
        time=pd.DatetimeIndex(df["time"]),
        block_s=block_s,
        # accumulated fields are already scaled to the block by the fetcher
        sw_in=df["ssrd"].values / block_s,          # W m^-2
        lw_in=df["strd"].values / block_s,          # W m^-2
        T_air=df["t2m"].values,                     # K
        pressure=df["sp"].values,                   # Pa
        wind=np.hypot(df["u10"].values, df["v10"].values),
        snowfall=df["sf"].values,                   # m w.e. per block
        precip=df["tp"].values if "tp" in df else None,
        era5_melt=df["smlt"].values if "smlt" in df else None,
        era5_albedo=df["fal"].values if "fal" in df else None,
    )
    out["q_air"] = specific_humidity_from_dewpoint(df["d2m"].values,
                                                   df["sp"].values)
    return out


def model_albedo(seb, forcing):
    """Our own ageing albedo, reset by snowfall above a threshold.

    Deliberate duplicate of the ``albedo == "model"`` branch of
    ``firnpack.aquifer.ReanalysisSite.__init__``, which applies the identical
    rule. Kept separate because this diagnostic must keep running without the
    column. The two must agree - this script generates the melt record the
    ERA5 runs are judged against, so a drift would make the audit table and
    the runs disagree about the forcing while both still look right.
    """
    p = seb.params
    sf = forcing["snowfall"]
    step_days = forcing["block_s"] / 86400.0
    days = np.empty(sf.size)
    since = 30.0
    for i, s in enumerate(sf):
        since = 0.0 if s >= p.fresh_snow_m_we else since + step_days
        days[i] = since
    return seb.albedo(days)


def run(forcing, seb=None, T_firn=270.0, k_eff=0.5, albedo="era5"):
    seb = seb or SurfaceEnergyBalance()
    n = forcing["T_air"].size
    if albedo == "era5":
        if forcing["era5_albedo"] is None:
            raise ValueError("albedo='era5' needs a 'fal' column in the forcing")
        alb = forcing["era5_albedo"]
    elif albedo == "model":
        alb = model_albedo(seb, forcing)
    else:
        raise ValueError(f"albedo must be 'model' or 'era5', not {albedo!r}")

    T_s, melt, Q_C, fx = seb.solve(
        sw_in=forcing["sw_in"], lw_in=forcing["lw_in"], T_air=forcing["T_air"],
        wind=forcing["wind"], pressure=forcing["pressure"],
        q_air=forcing["q_air"], T_firn=np.full(n, T_firn),
        k_eff=np.full(n, k_eff), albedo=alb)
    # fx already carries T_s and Q_C from the solve, so merge rather than
    # pass them again as keywords.
    return {**fx, "melt": melt, "albedo": alb}


def report(forcing, res, label=""):
    """Annual means over the years that are actually in the record.

    A missing year has no rows at all and a present year can carry a few
    blank cells, so the year count comes from rows with data and plain means
    would otherwise be NaN or diluted.
    """
    block_s = forcing["block_s"]
    have = np.isfinite(forcing["T_air"]) & np.isfinite(forcing["sw_in"])
    n_yr = max(len(np.unique(forcing["time"].year[have])), 1)
    melt_yr = np.nansum(res["melt"]) * block_s / n_yr
    print(f"\n--- {label} ({n_yr} years with data) ---")
    print(f"  melt (our SEB)        {melt_yr:8.3f} m w.e./yr")
    if forcing["era5_melt"] is not None:
        print(f"  melt (ERA5 snowmelt)  "
              f"{np.nansum(forcing['era5_melt'])/n_yr:8.3f} m w.e./yr")
    print(f"  snowfall              {np.nansum(forcing['snowfall'])/n_yr:8.3f} m w.e./yr")
    print(f"  mean albedo           {np.nanmean(res['albedo']):8.3f}")
    print(f"  melting fraction      "
          f"{100.0*np.mean(res['melt'][have] > 0):8.1f} % of steps")
    print("  mean fluxes (W/m2):   "
          f"SWnet {np.nanmean(res['SW_net']):6.1f}  LWnet {np.nanmean(res['LW_net']):6.1f}"
          f"  QH {np.nanmean(res['Q_H']):6.1f}  QL {np.nanmean(res['Q_L']):6.1f}"
          f"  QC {np.nanmean(res['Q_C']):6.1f}  QM {np.nanmean(res['Q_M']):6.1f}")
    return melt_yr


def annual_table(forcing, res, albedo_floor):
    """Per-year melt (ours and ERA5's), snowfall and mean temperature.

    Only years with a complete forcing record are kept: a year with a
    partial download would report a fraction of its melt as if it were the
    whole, and plot_history draws the gaps deliberately.

    ``albedo_floor`` is recorded as a column because the melt scales with
    it and the table is a tracked artifact read by plot_history.py.
    """
    df = pd.DataFrame({
        "year": forcing["time"].year,
        "seb_melt_m_we": res["melt"] * forcing["block_s"],
        "era5_melt_m_we": (forcing["era5_melt"] if forcing["era5_melt"] is not None
                           else np.nan),
        "snowfall_m_we": forcing["snowfall"],
        "T_C": forcing["T_air"] - 273.15,
        "_ok": np.isfinite(forcing["T_air"]) & np.isfinite(forcing["sw_in"]),
    })
    complete = df.groupby("year")["_ok"].all()
    df = df[df["year"].map(complete)]
    out = df.groupby("year").agg(seb_melt_m_we=("seb_melt_m_we", "sum"),
                                 era5_melt_m_we=("era5_melt_m_we", "sum"),
                                 snowfall_m_we=("snowfall_m_we", "sum"),
                                 T_C=("T_C", "mean")).reset_index()
    out["albedo_floor"] = float(albedo_floor)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--floor", type=float, default=ANNUAL_TABLE_FLOOR,
                    help="aged-firn albedo floor for the annual table "
                         f"(default: {ANNUAL_TABLE_FLOOR}, the floor the "
                         "tracked data/aquifer_annual_forcing.csv was built "
                         "at; the melt in that table scales with it)")
    a = ap.parse_args()

    f = load(a.csv)
    print(f"{f['time'][0]} .. {f['time'][-1]}  ({f['time'].size} steps of "
          f"{f['block_s']/3600:.0f} h)")
    print(f"  mean T {np.nanmean(f['T_air'])-273.15:.2f} C, "
          f"wind {np.nanmean(f['wind']):.1f} m/s, "
          f"SW down {np.nanmean(f['sw_in']):.1f} W/m2, "
          f"LW down {np.nanmean(f['lw_in']):.1f} W/m2")

    seb = SurfaceEnergyBalance(SurfaceEnergyParameters(albedo_firn=a.floor))
    report(f, run(f, seb, albedo="era5"), "ERA5 albedo")
    res = run(f, seb, albedo="model")
    report(f, res, f"model ageing albedo (floor {seb.params.albedo_firn:.2f})")
    table = annual_table(f, res, seb.params.albedo_firn)
    out = DATA / "aquifer_annual_forcing.csv"
    table.to_csv(out, index=False, float_format="%.6g")
    print(f"\nannual table: {len(table)} complete years -> {out}")

    print("\nsensitivity to the prescribed firn temperature "
          "(the term the column will supply):")
    for T_firn in (265.0, 270.0, 272.0, 273.15):
        r = run(f, seb, T_firn=T_firn, albedo="era5")
        have = np.isfinite(f["T_air"]) & np.isfinite(f["sw_in"])
        m = np.nansum(r["melt"]) * f["block_s"] / max(len(np.unique(f["time"].year[have])), 1)
        print(f"  T_firn = {T_firn-273.15:+6.2f} C  ->  melt {m:6.3f} m w.e./yr")


if __name__ == "__main__":
    main()
