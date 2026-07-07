#!/usr/bin/env python3
"""Compute summary melt rate for a given ApRES site and report measurement position.

This is meant to be a small post-processing helper after you have produced
`pair_results.csv` for each station.

Run:
  python site_melt_summary.py GA01

Expected repo layout (relative to this script):
  data/<SITE>/.../*.DAT (or .dat)
  results/<SITE>/pair_results.csv   (preferred)

Fallbacks:
  - If results/<SITE>/pair_results.csv is missing, will look for pair_results.csv
    in the repo root.
  - If coordinates aren't present in config.ini/CONFIG.INI, will attempt to read
    Latitude/Longitude from the first .DAT header.

Outputs:
  Prints a one-line summary and writes results/<SITE>/site_summary.csv
"""

from __future__ import annotations

import csv
import glob
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Optional manual fallback: if you have known waypoint coordinates (e.g., from a
# field spreadsheet), you can populate them here.
# Format: {"GA01": (-77.000000, 165.000000), ...}
SITE_COORDS: dict[str, tuple[float, float]] = {
    # "GA01": (-77.000000, 165.000000),
}


@dataclass
class SiteSummary:
    site: str
    latitude: Optional[float]
    longitude: Optional[float]
    melt_rate_m_per_year_median: Optional[float]
    melt_rate_m_per_year_mean: Optional[float]
    n_pairs: int
    csv_path: Optional[str]
    coord_source: str


def _find_repo_root(start: Path) -> Path:
    """Heuristic: repo root is parent that contains 'data' dir."""
    p = start.resolve()
    for _ in range(6):
        if (p / "data").is_dir():
            return p
        p = p.parent
    # fallback: script's parent
    return start.resolve().parent


def _read_text_file(path: Path, max_bytes: int = 200_000) -> str:
    with path.open("rb") as f:
        b = f.read(max_bytes)
    # DAT headers are ASCII; ignore any binary after header
    return b.decode("utf-8", errors="ignore")


def _extract_latlon_from_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    # Supports header style: Latitude=..., Longitude=...
    lat = None
    lon = None

    m = re.search(r"\bLatitude\s*=\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        try:
            lat = float(m.group(1))
        except ValueError:
            lat = None

    m = re.search(r"\bLongitude\s*=\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        try:
            lon = float(m.group(1))
        except ValueError:
            lon = None

    # Some configs use lat/lon keywords
    if lat is None:
        m = re.search(r"\b(lat|latitude)\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if m:
            try:
                lat = float(m.group(2))
            except ValueError:
                lat = None

    if lon is None:
        m = re.search(r"\b(lon|long|longitude)\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if m:
            try:
                lon = float(m.group(2))
            except ValueError:
                lon = None

    # Treat 0/0 as "missing" (common when GPS is off)
    if lat is not None and abs(lat) < 1e-9:
        lat = None
    if lon is not None and abs(lon) < 1e-9:
        lon = None

    return lat, lon


def find_site_coordinates(site_dir: Path) -> Tuple[Optional[float], Optional[float], str]:
    """Find site coordinates from config.ini/CONFIG.INI or from the first DAT header."""

    # 0) Manual fallback table
    site_name = site_dir.name
    if site_name in SITE_COORDS:
        lat, lon = SITE_COORDS[site_name]
        return float(lat), float(lon), "manual_table"

    # 1) Search INI files
    ini_paths = []
    ini_paths.extend(site_dir.rglob("config.ini"))
    ini_paths.extend(site_dir.rglob("CONFIG.INI"))

    for ini in sorted(set(ini_paths)):
        try:
            txt = _read_text_file(ini)
        except OSError:
            continue
        lat, lon = _extract_latlon_from_text(txt)
        if lat is not None and lon is not None:
            return lat, lon, f"ini:{ini.relative_to(site_dir)}"

    # 2) Fallback: read first DAT header
    dat_files = sorted(site_dir.rglob("*.DAT")) + sorted(site_dir.rglob("*.dat"))
    for dat in dat_files:
        try:
            head = _read_text_file(dat)
        except OSError:
            continue
        # For RMB files, header is between *** Burst Header *** and *** End Header ***
        if "*** Burst Header ***" in head:
            # truncate to end header
            end_idx = head.find("*** End Header ***")
            if end_idx != -1:
                head = head[: end_idx + len("*** End Header ***")]
        lat, lon = _extract_latlon_from_text(head)
        if lat is not None and lon is not None:
            return lat, lon, f"dat_header:{dat.relative_to(site_dir)}"

    return None, None, "not_found"


def find_site_coordinates_from_preprocessed(results_site_dir: Path) -> Tuple[Optional[float], Optional[float], str]:
    """Try to read lat/lon from saved preprocessed .npz caches.

    This is usually faster than scanning the raw .DAT headers, and it keeps the
    coordinate-handling consistent with the main processing pipeline.
    """

    pre_dir = Path(results_site_dir) / "preprocessed"
    if not pre_dir.exists():
        return None, None, "not_found"

    lat_vals: list[float] = []
    lon_vals: list[float] = []

    for npz in sorted(pre_dir.glob("*.npz")):
        try:
            with np.load(npz, allow_pickle=False) as z:
                if "latitude_deg" not in z or "longitude_deg" not in z:
                    continue
                lat = float(np.asarray(z["latitude_deg"]).ravel()[0])
                lon = float(np.asarray(z["longitude_deg"]).ravel()[0])
        except Exception:
            continue

        # Treat NaNs and obvious placeholders as missing.
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue
        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            continue
        if not (-90.0 <= lat <= 90.0):
            continue
        if not (-180.0 <= lon <= 180.0):
            continue

        lat_vals.append(lat)
        lon_vals.append(lon)

    if not lat_vals or not lon_vals:
        return None, None, "not_found"

    return _median(lat_vals), _median(lon_vals), "preprocessed_npz"


def _read_melt_rates(csv_path: Path) -> Tuple[list[float], str]:
    """Read melt rates from a pair_results.csv.

    Returns list of melt_rate_m_per_year values. Accepts a few common column names.
    """
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        # candidate columns
        candidates = [
            # Preferred: post-filtered series written by the Python driver
            "melt_rate_m_per_year_medfilt",
            "melt_rate_m_per_year",
            "melt_rate",
            "melt_rate_myr",
            "melt_rate_m_per_yr",
        ]
        col = None
        for c in candidates:
            if c in fieldnames:
                col = c
                break
        if col is None:
            raise ValueError(
                f"Could not find a melt-rate column in {csv_path}. "
                f"Found columns: {fieldnames}"
            )

        vals: list[float] = []
        for row in reader:
            v = row.get(col, "")
            if v is None:
                continue
            v = str(v).strip()
            if v == "" or v.lower() in {"nan", "none"}:
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue

    return vals, col


def _median(x: list[float]) -> float:
    x_sorted = sorted(x)
    n = len(x_sorted)
    mid = n // 2
    if n % 2 == 1:
        return x_sorted[mid]
    return 0.5 * (x_sorted[mid - 1] + x_sorted[mid])


def summarize_site(site: str, repo_root: Path) -> SiteSummary:
    data_dir = repo_root / "data" / site
    if not data_dir.exists():
        raise FileNotFoundError(f"Could not find site directory: {data_dir}")

    # locate pair_results.csv
    preferred = repo_root / "results" / site / "pair_results.csv"
    fallback = repo_root / "pair_results.csv"

    csv_path = None
    if preferred.exists():
        csv_path = preferred
    elif fallback.exists():
        csv_path = fallback

    melt_vals: list[float] = []
    if csv_path is not None:
        melt_vals, _ = _read_melt_rates(csv_path)

    # Prefer coordinates from the processing pipeline's cached preprocessed profiles
    # (if they exist), then fall back to scanning data/ for INI / DAT headers.
    lat, lon, src = find_site_coordinates_from_preprocessed(repo_root / "results" / site)
    if lat is None or lon is None:
        lat, lon, src = find_site_coordinates(data_dir)

    if melt_vals:
        median = _median(melt_vals)
        mean = sum(melt_vals) / len(melt_vals)
    else:
        median = None
        mean = None

    return SiteSummary(
        site=site,
        latitude=lat,
        longitude=lon,
        melt_rate_m_per_year_median=median,
        melt_rate_m_per_year_mean=mean,
        n_pairs=len(melt_vals),
        csv_path=str(csv_path) if csv_path is not None else None,
        coord_source=src,
    )


def write_summary_csv(summary: SiteSummary, repo_root: Path) -> Path:
    out_dir = repo_root / "results" / summary.site
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "site_summary.csv"

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "site",
                "latitude",
                "longitude",
                "coord_source",
                "melt_rate_m_per_year_median",
                "melt_rate_m_per_year_mean",
                "n_pairs",
                "pair_results_csv",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "site": summary.site,
                "latitude": summary.latitude,
                "longitude": summary.longitude,
                "coord_source": summary.coord_source,
                "melt_rate_m_per_year_median": summary.melt_rate_m_per_year_median,
                "melt_rate_m_per_year_mean": summary.melt_rate_m_per_year_mean,
                "n_pairs": summary.n_pairs,
                "pair_results_csv": summary.csv_path,
            }
        )

    return out_path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python site_melt_summary.py <SITE>, e.g. GA01")
        return 2

    site = sys.argv[1].strip()
    script_path = Path(__file__).resolve()
    repo_root = _find_repo_root(script_path)

    try:
        summary = summarize_site(site, repo_root)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    out_csv = write_summary_csv(summary, repo_root)

    # Print a concise human-readable summary
    latlon = (
        f"{summary.latitude:.6f}, {summary.longitude:.6f}"
        if summary.latitude is not None and summary.longitude is not None
        else "(lat/lon not found)"
    )

    if summary.melt_rate_m_per_year_median is None:
        melt_txt = "(no melt-rate values found)"
    else:
        melt_txt = (
            f"median={summary.melt_rate_m_per_year_median:.4f} m/yr, "
            f"mean={summary.melt_rate_m_per_year_mean:.4f} m/yr, n={summary.n_pairs}"
        )

    print(
        f"{summary.site}: position {latlon} [{summary.coord_source}] | melt rate {melt_txt}"
    )
    print(f"Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
