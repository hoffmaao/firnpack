"""Serialize preprocessed profiles and pair results.

We keep the on-disk format intentionally simple:
- Preprocessed profiles: .npz (numpy savez_compressed)
- Pair results: .csv (via pandas if available, otherwise python csv)
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from .preprocess import PreprocessedProfile
from .strain import BedShift, CoarseAlignment, FineAlignment, StrainFit, StrainMeltResult


def save_preprocessed_npz(profile: PreprocessedProfile, outpath: Path) -> None:
    """Save a PreprocessedProfile to an .npz file."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Optional sub-band spectra (may be None for backwards compatibility)
    extra = {}
    if getattr(profile, "spec_cor_subbands", None) is not None:
        try:
            sb = np.asarray(profile.spec_cor_subbands)
            extra.update(
                {
                    "subband_edges_hz": np.asarray(profile.subband_edges_hz, dtype=float)
                    if profile.subband_edges_hz is not None
                    else np.zeros((0, 2), dtype=float),
                    "subband_fc_hz": np.asarray(profile.subband_fc_hz, dtype=float)
                    if profile.subband_fc_hz is not None
                    else np.zeros((0,), dtype=float),
                    "subband_spec_cor_real": np.real(sb),
                    "subband_spec_cor_imag": np.imag(sb),
                }
            )
        except Exception:
            # If something goes wrong, silently drop sub-band fields
            extra = {}

    np.savez_compressed(
        outpath,
        file_path=str(profile.file_path),
        name=profile.name,
        timestamp=profile.timestamp.isoformat(),
        gps_on=np.array([-1 if profile.gps_on is None else int(profile.gps_on)]),
        latitude_deg=np.array([np.nan if profile.latitude_deg is None else float(profile.latitude_deg)]),
        longitude_deg=np.array([np.nan if profile.longitude_deg is None else float(profile.longitude_deg)]),
        range_coarse_m=profile.range_coarse_m,
        range_fine_m=profile.range_fine_m,
        range_m=profile.range_m,
        spec_raw_real=np.real(profile.spec_raw),
        spec_raw_imag=np.imag(profile.spec_raw),
        spec_cor_real=np.real(profile.spec_cor),
        spec_cor_imag=np.imag(profile.spec_cor),
        phase_std_error_rad=profile.phase_std_error_rad,
        range_error_m=profile.range_error_m,
        bed_depth_m=np.array(profile.bed_depth_m),
        bed_index=np.array(profile.bed_index),
        noise_depth_m=np.array(profile.noise_depth_m),
        fs_hz=np.array(profile.fs_hz),
        f0_hz=np.array(profile.f0_hz),
        f1_hz=np.array(profile.f1_hz),
        B_hz=np.array(profile.B_hz),
        fc_hz=np.array(profile.fc_hz),
        K_rad_s2=np.array(profile.K_rad_s2),
        ci_m_s=np.array(profile.ci_m_s),
        lambdac_m=np.array(profile.lambdac_m),
        er_ice=np.array(profile.er_ice),
        pad_factor=np.array(profile.pad_factor),
        **extra,
    )


def load_preprocessed_npz(path: Path) -> PreprocessedProfile:
    """Load a PreprocessedProfile from an .npz file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        spec_raw = z["spec_raw_real"] + 1j * z["spec_raw_imag"]
        spec_cor = z["spec_cor_real"] + 1j * z["spec_cor_imag"]

        # Backwards compatible optional fields
        gps_on = None
        lat = None
        lon = None
        if "gps_on" in z:
            try:
                gps_val = int(np.asarray(z["gps_on"]).ravel()[0])
                gps_on = None if gps_val < 0 else gps_val
            except Exception:
                gps_on = None
        if "latitude_deg" in z:
            try:
                v = float(np.asarray(z["latitude_deg"]).ravel()[0])
                lat = None if np.isnan(v) else v
            except Exception:
                lat = None
        if "longitude_deg" in z:
            try:
                v = float(np.asarray(z["longitude_deg"]).ravel()[0])
                lon = None if np.isnan(v) else v
            except Exception:
                lon = None
        # Optional sub-band fields
        sb_edges = None
        sb_fc = None
        sb_spec = None
        if "subband_spec_cor_real" in z and "subband_spec_cor_imag" in z:
            try:
                sb_spec = z["subband_spec_cor_real"] + 1j * z["subband_spec_cor_imag"]
                if sb_spec.size == 0:
                    sb_spec = None
            except Exception:
                sb_spec = None
        if "subband_edges_hz" in z:
            try:
                v = np.asarray(z["subband_edges_hz"], dtype=float)
                sb_edges = None if v.size == 0 else v
            except Exception:
                sb_edges = None
        if "subband_fc_hz" in z:
            try:
                v = np.asarray(z["subband_fc_hz"], dtype=float)
                sb_fc = None if v.size == 0 else v
            except Exception:
                sb_fc = None

        return PreprocessedProfile(
            file_path=Path(str(z["file_path"])),
            name=str(z["name"]),
            timestamp=datetime.fromisoformat(str(z["timestamp"])),
            gps_on=gps_on,
            latitude_deg=lat,
            longitude_deg=lon,
            range_coarse_m=z["range_coarse_m"],
            range_fine_m=z["range_fine_m"],
            range_m=z["range_m"],
            spec_raw=spec_raw,
            spec_cor=spec_cor,
            phase_std_error_rad=z["phase_std_error_rad"],
            range_error_m=z["range_error_m"],
            bed_depth_m=float(z["bed_depth_m"]),
            bed_index=int(z["bed_index"]),
            noise_depth_m=float(z["noise_depth_m"]),
            fs_hz=float(z["fs_hz"]),
            f0_hz=float(z["f0_hz"]),
            f1_hz=float(z["f1_hz"]),
            B_hz=float(z["B_hz"]),
            fc_hz=float(z["fc_hz"]),
            K_rad_s2=float(z["K_rad_s2"]),
            ci_m_s=float(z["ci_m_s"]),
            lambdac_m=float(z["lambdac_m"]),
            er_ice=float(z["er_ice"]),
            pad_factor=int(z["pad_factor"]),
            subband_edges_hz=sb_edges,
            subband_fc_hz=sb_fc,
            spec_cor_subbands=sb_spec,
        )


def save_pair_results_csv(results: List[StrainMeltResult], outpath: Path) -> None:
    """Write a list of StrainMeltResult rows to CSV."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Avoid hard dependency on pandas.
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame([asdict(r) for r in results])
        df.to_csv(outpath, index=False)
        return
    except Exception:
        pass

    import csv

    rows = [asdict(r) for r in results]
    if not rows:
        return

    with outpath.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            # Ensure datetimes serialize nicely
            row = dict(row)
            row["t1"] = row["t1"].isoformat()
            row["t2"] = row["t2"].isoformat()
            w.writerow(row)


def save_pair_diagnostics_npz(
    result: StrainMeltResult,
    coarse: CoarseAlignment,
    fine: FineAlignment,
    fit: StrainFit,
    bed: BedShift,
    outpath: Path,
) -> None:
    """Save full per-pair diagnostic outputs to an .npz.

    This is optional, but it makes it easy to re-plot or do further
    analysis without re-running the full solver.
    """

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        outpath,
        # Result metadata
        station=result.station,
        t1=result.t1.isoformat(),
        t2=result.t2.isoformat(),
        dt_days=result.dt_days,
        vsr_per_year=result.vsr_per_year,
        vsr_err_per_year=result.vsr_err_per_year,
        surface_compaction_m=result.surface_compaction_m,
        surface_compaction_err_m=result.surface_compaction_err_m,
        bed_depth_m=result.bed_depth_m,
        bed_shift_m=result.bed_shift_m,
        bed_shift_err_m=result.bed_shift_err_m,
        melt_m=result.melt_m,
        melt_err_m=result.melt_err_m,
        melt_rate_m_per_year=result.melt_rate_m_per_year,
        melt_rate_err_m_per_year=result.melt_rate_err_m_per_year,
        # Additional melt diagnostics
        bed_shift_amp_m=result.bed_shift_amp_m,
        melt_m_amp=result.melt_m_amp,
        melt_rate_m_per_year_amp=result.melt_rate_m_per_year_amp,
        melt_rate_phase_minus_amp_m_per_year=result.melt_rate_phase_minus_amp_m_per_year,
        melt_rate_phase_minus_amp_abs_m_per_year=result.melt_rate_phase_minus_amp_abs_m_per_year,
        melt_rate_subband0_m_per_year=result.melt_rate_subband0_m_per_year,
        melt_rate_subband1_m_per_year=result.melt_rate_subband1_m_per_year,
        melt_rate_subband2_m_per_year=result.melt_rate_subband2_m_per_year,
        melt_rate_subband_std_m_per_year=result.melt_rate_subband_std_m_per_year,
        melt_rate_subband_range_m_per_year=result.melt_rate_subband_range_m_per_year,
        # Coarse alignment
        coarse_range_m=coarse.range_m,
        coarse_lag_bins=coarse.lag_bins,
        coarse_dh_m=coarse.dh_m,
        coarse_amp_cor=coarse.amp_cor,
        coarse_amp_cor_prom=coarse.amp_cor_prom,
        coarse_is_good=coarse.is_good,
        coarse_poly_coeff=coarse.poly_coeff,
        coarse_rangeind=coarse.rangeind if coarse.rangeind is not None else np.array([]),
        coarse_ampcor_matrix=coarse.ampcor_matrix if coarse.ampcor_matrix is not None else np.array([]),
        coarse_lags_vec=coarse.lags_vec if coarse.lags_vec is not None else np.array([]),
        # Fine alignment
        fine_range_m=fine.range_m,
        fine_lag_bins=fine.lag_bins,
        fine_amp_cor=fine.amp_cor,
        fine_coherence_real=np.real(fine.coherence),
        fine_coherence_imag=np.imag(fine.coherence),
        fine_phase_cor=fine.phase_cor,
        fine_pe=fine.pe,
        fine_pse=fine.pse,
        fine_dh_m=fine.dh_m,
        fine_dhe_m=fine.dhe_m,
        fine_selected_lag_index=fine.selected_lag_index if fine.selected_lag_index is not None else np.array([]),
        fine_initial_lag_index=fine.initial_lag_index if fine.initial_lag_index is not None else np.array([]),
        fine_num_wavelength_error=np.array([-1 if fine.num_wavelength_error is None else fine.num_wavelength_error]),
        fine_ampcor_matrix=fine.ampcor_matrix if fine.ampcor_matrix is not None else np.array([]),
        fine_cor_matrix_real=np.real(fine.cor_matrix) if fine.cor_matrix is not None else np.array([]),
        fine_cor_matrix_imag=np.imag(fine.cor_matrix) if fine.cor_matrix is not None else np.array([]),
        fine_lags_vec=fine.lags_vec if fine.lags_vec is not None else np.array([]),
        # Fit
        fit_mid_range_m=np.array([fit.mid_range_m]),
        fit_m0=np.array([fit.m0]),
        fit_m1=np.array([fit.m1]),
        fit_cov=fit.cov,
        fit_intercept0_m=np.array([fit.intercept0_m]),
        fit_intercept0_err_m=np.array([fit.intercept0_err_m]),
        fit_slope=np.array([fit.slope]),
        fit_slope_err=np.array([fit.slope_err]),
        fit_vsr_per_year=np.array([fit.vsr_per_year]),
        fit_vsr_err_per_year=np.array([fit.vsr_err_per_year]),
        fit_used_mask=fit.used_mask,
        fit_r2=np.array([fit.r2]),
        # Bed shift
        bed_method=bed.method,
        bed_dh_m=np.array([bed.dh_m]),
        bed_dhe_m=np.array([bed.dhe_m]),
        bed_lag_bins=np.array([bed.lag_bins]),
        bed_phase_rad=np.array([bed.phase_rad]),
        bed_coherence_real=np.array([np.real(bed.coherence)]),
        bed_coherence_imag=np.array([np.imag(bed.coherence)]),
    )
