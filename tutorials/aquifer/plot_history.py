"""Melt history over the aquifer AOI, 1940 onward.

Reads ``data/aquifer_annual_forcing.csv`` (written by the SEB run) and draws the
annual melt record against ERA5's own melt and the mean temperature.

Gaps are drawn as gaps. The ERA5 archive is still downloading, so whole
stretches of years are simply absent; joining across them would invent a trend
line through data that does not exist, and a decadal mean built from three
surviving years of a decade is not a decadal mean.

Run: python tutorials/aquifer/plot_history.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURES

HERE = Path(__file__).parent
DATA = HERE / "data"

SEB_C, ERA5_C = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#e3e3df"
AQUIFER_MELT = 0.5      # m w.e./yr, the order an aquifer is thought to need


def _tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _segments(years, values):
    """Split into runs of consecutive years so gaps are not bridged."""
    out, run_y, run_v = [], [], []
    for y, v in zip(years, values):
        if run_y and y != run_y[-1] + 1:
            out.append((run_y, run_v))
            run_y, run_v = [], []
        run_y.append(y)
        run_v.append(v)
    if run_y:
        out.append((run_y, run_v))
    return out


def main():
    path = DATA / "aquifer_annual_forcing.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    df = pd.read_csv(path).sort_values("year")
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})

    for ys, vs in _segments(df["year"].values, df["seb_melt_m_we"].values):
        ax.plot(ys, vs, color=SEB_C, lw=1.8, solid_capstyle="round")
    for ys, vs in _segments(df["year"].values, df["era5_melt_m_we"].values):
        ax.plot(ys, vs, color=ERA5_C, lw=1.5, solid_capstyle="round")
    ax.axhline(AQUIFER_MELT, color=MUTED, lw=1.0, ls="--")
    ax.annotate("melt an aquifer is thought to need", xy=(df["year"].min(), AQUIFER_MELT),
                xytext=(4, 5), textcoords="offset points", fontsize=8.5, color=MUTED)

    # Label where the two series are furthest apart, not at the final year:
    # they converge near zero at the end of each block, so end-labels land on
    # top of each other and on the footnote.
    gap = (df["seb_melt_m_we"] - df["era5_melt_m_we"]).abs()
    k = int(gap.idxmax())
    row = df.loc[k]
    ax.annotate("firnpack SEB", xy=(row["year"], row["seb_melt_m_we"]),
                xytext=(6, 8), textcoords="offset points", fontsize=9,
                color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                     ec="none", alpha=0.85))
    ax.annotate("ERA5 snowmelt", xy=(row["year"], row["era5_melt_m_we"]),
                xytext=(6, -14), textcoords="offset points", fontsize=9,
                color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                     ec="none", alpha=0.85))
    ax.set_ylabel("annual melt (m w.e. yr$^{-1}$)", fontsize=9, color=MUTED)
    ax.set_title("Meltwater supply to the southeast Greenland aquifer belt\n"
                 "18 ice-only ERA5 cells, bare-rock cells excluded",
                 fontsize=12, color=INK, loc="left")
    _tidy(ax)
    ax.margins(x=0.10)

    for ys, vs in _segments(df["year"].values, df["T_C"].values):
        ax2.plot(ys, vs, color=INK, lw=1.5, solid_capstyle="round")
    ax2.set_ylabel("mean annual T (C)", fontsize=9, color=MUTED)
    ax2.set_xlabel("year", fontsize=9, color=MUTED)
    _tidy(ax2)

    # shade the years still missing from the archive
    present = set(df["year"].values)
    allyrs = range(int(df["year"].min()), int(df["year"].max()) + 1)
    for y in allyrs:
        if y not in present:
            for a in (ax, ax2):
                a.axvspan(y - 0.5, y + 0.5, color="#f0efec", zorder=0, lw=0)
    ax2.annotate("shaded = not yet downloaded", xy=(0.01, 0.05),
                 xycoords="axes fraction", ha="left", fontsize=8.5, color=MUTED)

    fig.tight_layout()
    out = FIGURES / "aquifer_melt_history.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)

    early = df[df["year"] < 1980]
    late = df[df["year"] >= 1990]
    print(f"{'period':<14} {'years':>6} {'T(C)':>7} {'snow':>7} {'SEB melt':>9}")
    for name, sub in (("pre-1980", early), ("1990 on", late)):
        print(f"{name:<14} {len(sub):6d} {sub['T_C'].mean():7.2f} "
              f"{sub['snowfall_m_we'].mean():7.3f} {sub['seb_melt_m_we'].mean():9.3f}")
    print(f"\nchange in melt: {100*(late['seb_melt_m_we'].mean()/early['seb_melt_m_we'].mean()-1):+.0f}%")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
