"""Strain and melt estimation (adapted from MATLAB func_strain_rates.m).

This module implements the key pieces of the Craig Stewart / ApRES workflow:

- Coarse alignment via amplitude cross-correlation (MATLAB: func_align_coarse)
- Fine alignment via complex cross-correlation (MATLAB: func_align_fine)
- Linear fit of depth-dependent displacement to infer vertical strain rate
  (MATLAB: func_fit_ice / menke_fit)
- Bed shift and basal melt rate (MATLAB: ct_fmcw_melt bed shift + melt calc)

The implementation here is intentionally conservative and transparent.
It is designed to be "diagnosable" with saved figures rather than to be a
fully-automated black box.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import ProcessingConfig
from .preprocess import PreprocessedProfile
from .xcorr import XCorrResult, fmcw_xcorr


@dataclass
class CoarseAlignment:
    # 1D summary (best correlation per depth window)
    range_m: np.ndarray
    lag_bins: np.ndarray
    dh_m: np.ndarray
    amp_cor: np.ndarray
    amp_cor_prom: np.ndarray
    is_good: np.ndarray

    # Polynomial/robust fit of lags(range)
    poly_coeff: np.ndarray  # numpy.poly1d-style, highest power first

    # 2D fields (depth_window x lag) for diagnostics (optional but MATLAB-like)
    rangeind: np.ndarray | None = None
    ampcor_matrix: np.ndarray | None = None
    lags_vec: np.ndarray | None = None


@dataclass
class FineAlignment:
    range_m: np.ndarray
    lag_bins: np.ndarray
    amp_cor: np.ndarray
    coherence: np.ndarray  # complex coherent correlation at chosen lag
    phase_cor: np.ndarray  # |cc|/ic at chosen lag (MATLAB AF.phaseCor)
    pe: np.ndarray
    pse: np.ndarray
    dh_m: np.ndarray
    dhe_m: np.ndarray

    # For smart-unwrap diagnostics
    selected_lag_index: np.ndarray | None = None
    initial_lag_index: np.ndarray | None = None
    num_wavelength_error: int | None = None

    # Optional full matrices (depth_window x lag)
    ampcor_matrix: np.ndarray | None = None
    cor_matrix: np.ndarray | None = None
    lags_vec: np.ndarray | None = None


@dataclass
class StrainFit:
    mid_range_m: float
    m0: float  # intercept at mid-range for de-meaned design
    m1: float  # slope (dh per meter) for de-meaned design
    cov: np.ndarray  # 2x2 covariance of [m0,m1]

    intercept0_m: float  # intercept at 0 m depth
    intercept0_err_m: float
    slope: float
    slope_err: float

    vsr_per_year: float
    vsr_err_per_year: float

    used_mask: np.ndarray  # mask on fine alignment points used
    r2: float


@dataclass
class BedShift:
    method: str
    dh_m: float
    dhe_m: float
    lag_bins: int
    phase_rad: float
    coherence: complex


@dataclass
class StrainMeltResult:
    station: str
    t1: datetime
    t2: datetime
    dt_days: float

    # Strain
    vsr_per_year: float
    vsr_err_per_year: float
    surface_compaction_m: float
    surface_compaction_err_m: float

    # Bed
    bed_depth_m: float
    bed_shift_m: float
    bed_shift_err_m: float

    # Melt
    melt_m: float
    melt_err_m: float
    melt_rate_m_per_year: float
    melt_rate_err_m_per_year: float

    # ------------------------------------------------------------
    # Additional diagnostics for basal complexity / accretion
    # ------------------------------------------------------------
    # "Amplitude" method: estimate bed motion from the picked bed depth
    # (i.e., the depth of the dominant amplitude peak) and convert to melt by
    # subtracting the strain-predicted displacement.
    bed_shift_amp_m: float = float("nan")
    melt_m_amp: float = float("nan")
    melt_rate_m_per_year_amp: float = float("nan")

    # Vaňková-style disagreement between phase- and amplitude-derived melt.
    melt_rate_phase_minus_amp_m_per_year: float = float("nan")
    melt_rate_phase_minus_amp_abs_m_per_year: float = float("nan")

    # Frequency dependence tests: phase-derived melt using sub-bands of the chirp.
    # (Only first three are stored explicitly; std/range are computed over all
    # available sub-bands.)
    melt_rate_subband0_m_per_year: float = float("nan")
    melt_rate_subband1_m_per_year: float = float("nan")
    melt_rate_subband2_m_per_year: float = float("nan")
    melt_rate_subband_std_m_per_year: float = float("nan")
    melt_rate_subband_range_m_per_year: float = float("nan")

    # --- Fit diagnostics (helps QC poor strain-rate profiles)
    n_fit_points: int = 0
    fit_r2: float = float("nan")
    mean_coherence_used: float = float("nan")

    # --- Optional post-processing smoothing (filled later in the driver)
    vsr_per_year_medfilt: Optional[float] = None
    melt_rate_m_per_year_medfilt: Optional[float] = None


def _dr_from_profile(p: PreprocessedProfile) -> float:
    r = p.range_coarse_m
    if r.size < 2:
        raise ValueError("Range axis too short")
    return float(np.mean(np.diff(r)))


def _local_peaks(x: np.ndarray) -> np.ndarray:
    """Indices of (simple) local maxima, excluding endpoints.

    This is a lightweight replacement for MATLAB's findpeaks on smooth
    correlation curves.
    """

    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 3:
        return np.asarray([], dtype=int)
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1


def _ampcor_prominence(ampcor: np.ndarray) -> float:
    """Approximate MATLAB's peak-prominence metric used in func_align_coarse.

    MATLAB computes:
        cpk = findpeaks(AMPCOR,'sortstr','descend','npeaks',2)
        prom = cpk(1) - cpk(2)

    We mimic that by finding local maxima and taking the top two.
    """

    ampcor = np.asarray(ampcor, dtype=float).reshape(-1)
    pks = _local_peaks(ampcor)
    if pks.size == 0:
        return 0.0
    vals = np.sort(ampcor[pks])[::-1]
    if vals.size == 1:
        return 1.0
    return float(vals[0] - vals[1])


def _robust_linear_fit_huber(x: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Robust linear fit y = b0 + b1*x using Huber IRLS.

    Returns coefficients [b1, b0] so they can be used with np.polyval.
    """

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have same length")
    if x.size < 3:
        raise ValueError("Need at least 3 points for robust fit")

    A = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)  # [b0, b1]

    # Huber tuning constant (approximately 95% efficiency for Gaussian)
    c = 1.345

    for _ in range(max_iter):
        resid = y - (A @ beta)
        # robust scale estimate
        med = float(np.median(resid))
        s = 1.4826 * float(np.median(np.abs(resid - med)))
        s = max(s, 1e-12)
        u = resid / (c * s)
        w = np.ones_like(u)
        mask = np.abs(u) > 1.0
        w[mask] = 1.0 / np.abs(u[mask])

        # Weighted least squares
        Aw = A * w[:, None]
        yw = y * w
        beta_new, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        if np.linalg.norm(beta_new - beta) <= tol * np.linalg.norm(beta):
            beta = beta_new
            break
        beta = beta_new

    b0, b1 = float(beta[0]), float(beta[1])
    return np.asarray([b1, b0], dtype=float)


def _coarse_alignment(p1: PreprocessedProfile, p2: PreprocessedProfile, cfg: ProcessingConfig) -> CoarseAlignment:
    """Coarse alignment (amplitude xcorr).

    This is a direct translation of MATLAB `func_align_coarse.m`:
    - cross-correlate large chunks in amplitude to estimate integer-bin lags
    - quality checks: maxlag edge + peak prominence
    - robust linear fit of lag vs depth (polyval-ready coefficients)
    """

    r = p1.range_coarse_m
    dr = _dr_from_profile(p1)

    max_depth = min(p1.bed_depth_m, p2.bed_depth_m) - 20.0
    start_depth = float(cfg.min_depth_m)
    # MATLAB uses (fit.maxDepth - 2*coarseChunkWidth) for the binStart end.
    end_depth = max_depth - 2.0 * cfg.coarse_chunk_width_m
    if end_depth <= start_depth:
        raise ValueError("Not enough depth for coarse alignment")

    bin_start = np.arange(start_depth, end_depth + 1e-9, cfg.coarse_step_m)

    # Preallocate
    lags = None
    nwin = bin_start.size

    # Store full xcorr curves (optional, but useful for plotting/debug)
    rangeind = []
    ampcor_matrix = []

    ac_range = np.full(nwin, np.nan, dtype=float)
    ac_lag = np.full(nwin, np.nan, dtype=float)
    ac_dh = np.full(nwin, np.nan, dtype=float)
    ac_amp = np.full(nwin, np.nan, dtype=float)
    ac_prom = np.full(nwin, 0.0, dtype=float)

    for ii, z0 in enumerate(bin_start):
        z1 = z0 + cfg.coarse_chunk_width_m
        # MATLAB uses < max(depthRange) for the upper bound
        fi = np.where((r >= z0) & (r < z1))[0]
        if fi.size < 5:
            continue

        res = fmcw_xcorr(p1.spec_cor, p2.spec_cor, fi, cfg.maxlag_bins)
        if lags is None:
            lags = res.lags

        rangeind.append(res.iw)
        ampcor_matrix.append(res.ic)

        mci = int(np.argmax(res.ic))
        best = float(res.ic[mci])

        # Edge-of-search check: if best lag hits maxlag boundary, reject.
        if mci == 0 or mci == res.ic.size - 1:
            best = 0.0

        prom = _ampcor_prominence(res.ic)

        # Effective range at the best lag (MATLAB: interp1(..., RANGEIND))
        rr = np.interp(res.iw, np.arange(r.size, dtype=float), r)
        ac_range[ii] = float(rr[mci])
        ac_lag[ii] = float(res.lags[mci])
        ac_dh[ii] = float(dr * res.lags[mci])
        ac_amp[ii] = best
        ac_prom[ii] = float(prom)

    # Stack optional matrices
    rangeind_mat = None
    ampcor_mat = None
    lags_vec = None
    if len(rangeind) > 0 and lags is not None:
        rangeind_mat = np.vstack([np.asarray(v, dtype=float) for v in rangeind])
        ampcor_mat = np.vstack([np.asarray(v, dtype=float) for v in ampcor_matrix])
        lags_vec = np.asarray(lags, dtype=int)

    # Fit depth window (MATLAB has HandleFitCoarse; we use the common 'calc' behavior)
    fit_start = 0.0
    fit_end = min(float(cfg.bed_search_max_m or cfg.max_range_m), float(min(p1.noise_depth_m, p2.noise_depth_m)))

    is_good = (
        np.isfinite(ac_lag)
        & np.isfinite(ac_range)
        & (ac_amp >= cfg.min_ampcor)
        & (ac_prom >= cfg.min_ampcor_prom)
        & (ac_range >= fit_start)
        & (ac_range <= fit_end)
        & (ac_range <= min(p1.noise_depth_m, p2.noise_depth_m))
    )

    if np.sum(is_good) < 3:
        raise ValueError("Too few good coarse-alignment points")

    # Robust fit (linear by default, like MATLAB robustfit)
    if cfg.smooth_coarse_poly_order <= 1:
        poly_coeff = _robust_linear_fit_huber(ac_range[is_good], ac_lag[is_good])
    else:
        poly_coeff = np.polyfit(ac_range[is_good], ac_lag[is_good], deg=cfg.smooth_coarse_poly_order)

    return CoarseAlignment(
        range_m=ac_range,
        lag_bins=ac_lag,
        dh_m=ac_dh,
        amp_cor=ac_amp,
        amp_cor_prom=ac_prom,
        is_good=is_good,
        poly_coeff=poly_coeff,
        rangeind=rangeind_mat,
        ampcor_matrix=ampcor_mat,
        lags_vec=lags_vec,
    )


def _fine_alignment(p1: PreprocessedProfile, p2: PreprocessedProfile, cfg: ProcessingConfig, coarse: CoarseAlignment) -> FineAlignment:
    """Fine alignment (complex xcorr).

    Direct translation of MATLAB `func_align_fine.m`.
    """

    r = p1.range_coarse_m
    dr = _dr_from_profile(p1)

    max_depth = min(p1.bed_depth_m, p2.bed_depth_m) - 20.0
    start_depth = float(cfg.min_depth_m)
    end_depth = max_depth - cfg.coarse_chunk_width_m
    if end_depth <= start_depth:
        raise ValueError("Not enough depth for fine alignment")

    bin_start = np.arange(start_depth, end_depth + 1e-9, cfg.fine_step_m)
    nwin = bin_start.size

    # Preallocate matrices (depth_window x lag)
    lags_vec = None
    ampcor_mat = []
    cor_mat = []
    pe_mat = []
    pse_mat = []
    rangeind_mat = []

    # Initial best-lag indices per window
    mci = np.full(nwin, -1, dtype=int)
    amp_best = np.full(nwin, np.nan, dtype=float)
    cor_best = np.full(nwin, np.nan + 1j * np.nan, dtype=complex)
    range_best = np.full(nwin, np.nan, dtype=float)

    for ii, z0 in enumerate(bin_start):
        z1 = z0 + cfg.fine_chunk_width_m
        bin_depth = 0.5 * (z0 + z1)

        fi = np.where((r >= z0) & (r < z1))[0]
        if fi.size < 5:
            continue

        res = fmcw_xcorr(
            p1.spec_cor,
            p2.spec_cor,
            fi,
            cfg.maxlag_bins,
            fe=p1.phase_std_error_rad,
            ge=p2.phase_std_error_rad,
            pad_factor=cfg.pad_factor,
        )
        if lags_vec is None:
            lags_vec = np.asarray(res.lags, dtype=int)

        # Convert rangeind -> range (MATLAB: interp1(1:N,rangeCoarse,RANGEIND))
        rr = np.interp(res.iw, np.arange(r.size, dtype=float), r)
        rangeind_mat.append(rr)

        ampcor_mat.append(res.ic)
        cor_mat.append(res.cc)
        pe_mat.append(res.pe)
        pse_mat.append(res.pse)

        # Choose lag index either from coarse offset or from fine amplitude peak.
        if cfg.use_coarse_offset and np.any(np.isfinite(coarse.poly_coeff)):
            if cfg.do_poly_smooth_coarse_offset:
                pred_lag = float(np.polyval(coarse.poly_coeff, bin_depth))
            else:
                # Interpolate coarse dh (converted to bins)
                pred_lag = float(
                    np.interp(bin_depth, coarse.range_m, coarse.lag_bins, left=coarse.lag_bins[0], right=coarse.lag_bins[-1])
                )
            idx = int(np.argmin(np.abs(res.lags - pred_lag)))
        else:
            idx = int(np.argmax(res.ic))

        mci[ii] = idx
        amp_best[ii] = float(res.ic[idx])
        cor_best[ii] = complex(res.cc[idx])
        range_best[ii] = float(rr[idx])

    if lags_vec is None or len(ampcor_mat) == 0:
        raise ValueError("No valid fine-alignment windows")

    ampcor_mat_arr = np.vstack([np.asarray(v, dtype=float) for v in ampcor_mat])
    cor_mat_arr = np.vstack([np.asarray(v, dtype=complex) for v in cor_mat])
    pe_mat_arr = np.vstack([np.asarray(v, dtype=float) for v in pe_mat])
    pse_mat_arr = np.vstack([np.asarray(v, dtype=float) for v in pse_mat])
    range_mat_arr = np.vstack([np.asarray(v, dtype=float) for v in rangeind_mat])

    phasecor_mat_arr = np.abs(cor_mat_arr) / np.maximum(ampcor_mat_arr, 1e-12)

    # Smart unwrap (optional)
    selected_idx = mci.copy()
    num_wavelength_error = None
    if cfg.do_smart_unwrap:
        dphi = 2.0 * np.pi * p1.fc_hz / (p1.B_hz * float(p1.pad_factor))
        nwrap = int(np.round(2.0 * np.pi / dphi)) if dphi > 0 else 0

        # mci_pm: lag index adjusted to minimize phase diff near mci
        mci_pm = mci.copy()
        for ii in range(nwin):
            if mci[ii] < 0:
                continue
            ang = float(np.angle(cor_best[ii]))
            mci_pm[ii] = int(mci[ii] - np.fix(ang / dphi))
            if nwrap > 0:
                if mci_pm[ii] < 0:
                    mci_pm[ii] += nwrap
                if mci_pm[ii] >= lags_vec.size:
                    mci_pm[ii] -= nwrap

        # Pick starting depth index (MATLAB: cfg.HandleAF)
        # The MATLAB code selects the strongest-amplitude reflector within the
        # usable range, unless the user forces a depth.
        col_fi = np.where(
            np.isfinite(range_best)
            & (range_best >= start_depth)
            & (range_best <= end_depth)
        )[0]
        if col_fi.size == 0:
            col_fi = np.where(np.isfinite(range_best))[0]

        if cfg.handle_af == "use" and cfg.use_af_depth_m is not None and col_fi.size:
            # Nearest depth to the user-selected start depth.
            si = int(col_fi[np.argmin(np.abs(range_best[col_fi] - float(cfg.use_af_depth_m)))])
        else:
            # Default: max amplitude correlation over the valid depth range.
            if col_fi.size:
                si = int(col_fi[np.nanargmax(amp_best[col_fi])])
            else:
                si = int(np.nanargmax(amp_best))

        # mci_pmu: chosen lag-index path after phase-minimum tracking
        mci_pmu = np.full_like(mci_pm, -1)
        mci_pmu[si] = int(mci_pm[si] if mci_pm[si] >= 0 else mci[si])

        def _phase_minima_indices(row_cc: np.ndarray) -> np.ndarray:
            """Indices of local minima of |phase| (MATLAB: findpeaks(-abs(angle)))."""

            absph = np.abs(np.angle(row_cc))
            if absph.size < 3:
                return np.asarray([int(np.argmin(absph))], dtype=int)
            mins = np.where((absph[1:-1] <= absph[:-2]) & (absph[1:-1] <= absph[2:]))[0] + 1
            if mins.size == 0:
                mins = np.asarray([int(np.argmin(absph))], dtype=int)
            return mins

        def _choose_minima_with_tiebreak(pki: np.ndarray, ref_series: np.ndarray, ref_index: int, step: int) -> int:
            """Choose the pki closest to ref_series[ref_index], with MATLAB-like tie-breaking.

            Parameters
            ----------
            pki
                Candidate indices (into lag axis).
            ref_series
                Already-filled chosen indices (mci_pmu).
            ref_index
                Index to use as the primary reference.
            step
                Direction for tie-breaking (-1 looks backward, +1 looks forward).
            """

            counter = 1
            ref_val = int(ref_series[ref_index])
            # Primary tie set
            d = np.abs(ref_val - pki)
            fi = np.where(d == np.min(d))[0]

            # MATLAB breaks ties by looking further along the already-filled path.
            while fi.size > 1:
                counter += 1
                j = ref_index + step * (counter - 1)
                if j < 0 or j >= ref_series.size:
                    break
                if ref_series[j] < 0:
                    continue
                ref_val = int(ref_series[j])
                d = np.abs(ref_val - pki)
                fi = np.where(d == np.min(d))[0]

            # Final choice (MATLAB: [~,bpki] = min(abs(ref_val - pki)))
            return int(pki[np.argmin(np.abs(ref_val - pki))])

        # Forward pass (increasing depth)
        for ii in range(si + 1, nwin):
            pki = _phase_minima_indices(cor_mat_arr[ii, :])
            # Choose closest to previous, tie-broken using earlier points
            mci_pmu[ii] = _choose_minima_with_tiebreak(pki, mci_pmu, ref_index=ii - 1, step=-1)

            # MATLAB safety: if we jumped >=2 bins, retry using the NEXT depth window's minima.
            if abs(mci_pmu[ii] - mci_pmu[ii - 1]) >= 2 and ii < (nwin - 1):
                pki2 = _phase_minima_indices(cor_mat_arr[ii + 1, :])
                mci_pmu[ii] = _choose_minima_with_tiebreak(pki2, mci_pmu, ref_index=ii - 1, step=-1)

        # Backward pass (decreasing depth)
        for ii in range(si - 1, -1, -1):
            pki = _phase_minima_indices(cor_mat_arr[ii, :])
            mci_pmu[ii] = _choose_minima_with_tiebreak(pki, mci_pmu, ref_index=ii + 1, step=+1)

        selected_idx = mci_pmu
        num_wavelength_error = int(np.sum((mci_pm >= 0) & (selected_idx >= 0) & (mci_pm != selected_idx)))
    else:
        mci_pm = None

    # Extract values at chosen offsets
    lag_bins = np.full(nwin, np.nan, dtype=float)
    coherence = np.full(nwin, np.nan + 1j * np.nan, dtype=complex)
    phase_cor = np.full(nwin, np.nan, dtype=float)
    pe = np.full(nwin, np.nan, dtype=float)
    pse = np.full(nwin, np.nan, dtype=float)

    for ii in range(nwin):
        idx = selected_idx[ii] if selected_idx[ii] >= 0 else mci[ii]
        if idx < 0:
            continue
        lag_bins[ii] = float(lags_vec[idx])
        coherence[ii] = complex(cor_mat_arr[ii, idx])
        phase_cor[ii] = float(phasecor_mat_arr[ii, idx])
        pe[ii] = float(pe_mat_arr[ii, idx])
        pse[ii] = float(pse_mat_arr[ii, idx])
        range_best[ii] = float(range_mat_arr[ii, idx])
        amp_best[ii] = float(ampcor_mat_arr[ii, idx])

    lagdh = lag_bins * dr
    phasedh = -np.angle(coherence) * (p1.lambdac_m / (4.0 * np.pi))
    dh = lagdh + phasedh
    dhe = pse * (p1.lambdac_m / (4.0 * np.pi))

    return FineAlignment(
        range_m=range_best,
        lag_bins=lag_bins,
        amp_cor=amp_best,
        coherence=coherence,
        phase_cor=phase_cor,
        pe=pe,
        pse=pse,
        dh_m=dh,
        dhe_m=dhe,
        selected_lag_index=selected_idx if cfg.do_smart_unwrap else None,
        initial_lag_index=mci,
        num_wavelength_error=num_wavelength_error,
        ampcor_matrix=ampcor_mat_arr,
        cor_matrix=cor_mat_arr,
        lags_vec=lags_vec,
    )


def _interp_nearest(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """1D nearest-neighbour interpolation.

    MATLAB equivalent: interp1(...,'nearest','extrap') for monotonically
    increasing xp.
    """

    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    if xp.size == 0:
        return np.full_like(x, np.nan, dtype=float)

    # Ensure xp is sorted
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]

    idx = np.searchsorted(xp, x, side="left")
    idx = np.clip(idx, 0, xp.size - 1)

    # Compare with left neighbour where possible
    left = np.clip(idx - 1, 0, xp.size - 1)
    right = idx
    choose_left = np.abs(x - xp[left]) <= np.abs(x - xp[right])
    out_idx = np.where(choose_left, left, right)
    return fp[out_idx]


def menke_fit(G: np.ndarray, y: np.ndarray, err: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted least-squares fit following Menke (1984) / MATLAB menke_fit.m.

    Parameters
    ----------
    G
        Design / predictor matrix (N x P). Include a column of ones if you
        want an intercept.
    y
        Data vector (N,).
    err
        Error estimates on each y (scalar or length N). If None, unweighted.

    Returns
    -------
    M, Me, cov
        Coefficients, 1-sigma errors, and covariance matrix.
    """

    G = np.asarray(G, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if G.ndim != 2:
        raise ValueError("G must be 2D")
    if G.shape[0] != y.size:
        raise ValueError("G and y length mismatch")

    if err is None:
        W = np.eye(y.size)
    else:
        err = np.asarray(err, dtype=float).reshape(-1)
        if err.size == 1:
            err = np.full_like(y, float(err), dtype=float)
        if err.size != y.size:
            raise ValueError("err must be scalar or length N")
        w = np.where(np.isfinite(err) & (err > 0), 1.0 / (err**2), 1.0)
        W = np.diag(w)

    GTWG = G.T @ W @ G
    GTWy = G.T @ W @ y
    M = np.linalg.solve(GTWG, GTWy)

    # Residual-based covariance estimate (matches MATLAB's default branch)
    resid = y - (G @ M)
    # cov(resid) for vector -> variance with (N-1) normalization
    var = float(np.var(resid, ddof=1)) if resid.size > 1 else 0.0
    N = float(y.size)
    P = float(G.shape[1])
    dof = max(1.0, N - P)
    cov = var * ((N - 1.0) / dof) * np.linalg.inv(G.T @ G)
    Me = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return M, Me, cov


def _strain_fit(
    p1: PreprocessedProfile,
    p2: PreprocessedProfile,
    fine: FineAlignment,
    coarse: CoarseAlignment,
    cfg: ProcessingConfig,
    dt_days: float,
) -> StrainFit:
    """Fit vertical strain from the depth-dependent displacement.

    MATLAB reference: func_fit_ice.m (via func_fit_calc).

    We fit a linear model to fine-alignment displacements:

        dh(z) = m0 + m1 * (z - z_mid)

    where m1 is the vertical strain (per meter). The strain rate is
    m1 * daysPerYear / dt_days.
    """

    # --- Fit depth limits
    max_depth = min(p1.bed_depth_m, p2.bed_depth_m) - 20.0
    fit_end = min(max_depth, p1.noise_depth_m, p2.noise_depth_m)
    fit_start = float(cfg.firn_depth_m)

    # --- Interpolate coarse amplitude correlation onto fine ranges (nearest)
    cg = np.isfinite(coarse.range_m) & np.isfinite(coarse.amp_cor)
    coarse_amp_interp = _interp_nearest(fine.range_m, coarse.range_m[cg], coarse.amp_cor[cg])

    # --- Select good reflectors
    coh_mag = np.abs(fine.coherence)
    used = (
        np.isfinite(fine.range_m)
        & np.isfinite(fine.dh_m)
        & np.isfinite(fine.dhe_m)
        & np.isfinite(coh_mag)
        & (fine.range_m >= fit_start)
        & (fine.range_m <= fit_end)
        & (coh_mag >= cfg.min_cohere_fine)
        & (coarse_amp_interp >= cfg.min_cohere_coarse)
    )

    if cfg.min_cohere_phase is not None:
        used = used & np.isfinite(fine.phase_cor) & (fine.phase_cor >= float(cfg.min_cohere_phase))

    n_used = int(np.sum(used))
    if n_used < cfg.min_points_to_fit:
        raise ValueError(f"Not enough coherent points to fit strain (have {n_used})")

    x = fine.range_m[used]
    y = fine.dh_m[used]
    sigma = fine.dhe_m[used]

    mid = float(np.mean(x))
    G = np.column_stack([np.ones_like(x), x - mid])

    if cfg.fit_method == "robust":
        # Robust IRLS on the de-meaned predictor (ignoring sigma weights)
        A = G
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        c = 1.345
        for _ in range(50):
            resid = y - A @ beta
            med = float(np.median(resid))
            s = 1.4826 * float(np.median(np.abs(resid - med)))
            s = max(s, 1e-12)
            u = resid / (c * s)
            w = np.ones_like(u)
            mask = np.abs(u) > 1.0
            w[mask] = 1.0 / np.abs(u[mask])
            Aw = A * w[:, None]
            yw = y * w
            beta_new, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
            if np.linalg.norm(beta_new - beta) <= 1e-8 * max(1.0, np.linalg.norm(beta)):
                beta = beta_new
                break
            beta = beta_new
        M = beta
        # Approximate covariance from residual variance (unweighted)
        resid = y - A @ M
        var = float(np.var(resid, ddof=1)) if resid.size > 1 else 0.0
        cov = var * np.linalg.inv(A.T @ A)
        Me = np.sqrt(np.maximum(np.diag(cov), 0.0))
    else:
        # Menke weighted least squares
        M, Me, cov = menke_fit(G, y, sigma)

    # Optional intercept shift relative to firn depth (MATLAB fitMethod_shift)
    if cfg.fit_method_shift:
        a = float(cfg.firn_depth_m) - mid
        A = np.asarray([[0.0, -a], [0.0, 1.0]], dtype=float)
        M = A @ M
        cov = A @ cov @ A.T
        Me = np.sqrt(np.maximum(np.diag(cov), 0.0))

    m0 = float(M[0])
    m1 = float(M[1])

    # Convert to intercept at zero depth: dh = intercept0 + slope*z
    intercept0 = m0 - mid * m1

    # Propagate covariance under intercept0 = m0 - mid*m1
    var0 = float(cov[0, 0])
    var1 = float(cov[1, 1])
    cov01 = float(cov[0, 1])
    var_intercept0 = var0 + (mid**2) * var1 - 2.0 * mid * cov01
    intercept0_err = float(np.sqrt(max(var_intercept0, 0.0)))

    slope = m1
    slope_err = float(np.sqrt(max(var1, 0.0)))

    vsr = slope * cfg.daysPerYear / dt_days
    vsr_err = slope_err * cfg.daysPerYear / dt_days

    # MATLAB reports corrcoef(range(gi),dh(gi))^2 as an R^2-like metric
    if x.size >= 2 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
        rxy = float(np.corrcoef(x, y)[0, 1])
        r2 = rxy * rxy
    else:
        r2 = float("nan")

    return StrainFit(
        mid_range_m=mid,
        m0=m0,
        m1=m1,
        cov=cov,
        intercept0_m=float(intercept0),
        intercept0_err_m=float(intercept0_err),
        slope=float(slope),
        slope_err=float(slope_err),
        vsr_per_year=float(vsr),
        vsr_err_per_year=float(vsr_err),
        used_mask=used,
        r2=float(r2),
    )


def _bed_shift_xcorr(p1: PreprocessedProfile, p2: PreprocessedProfile, cfg: ProcessingConfig) -> BedShift:
    """Bed shift via complex xcorr (translation of ct_fmcw_melt bedshift 'xcorr')."""
    dr = _dr_from_profile(p1)

    # Bed indices
    bn1 = int(p1.bed_index)
    bn2 = int(p2.bed_index)

    bed_coarse_1 = float(p1.range_coarse_m[bn1])

    win_lo, win_hi = cfg.xcor_bed_win_m
    fi = np.where(
        (p1.range_coarse_m >= min(bed_coarse_1 + win_lo, bed_coarse_1 + win_hi))
        & (p1.range_coarse_m <= max(bed_coarse_1 + win_lo, bed_coarse_1 + win_hi))
    )[0]

    if fi.size < 5:
        raise ValueError("Bed window too small / outside range")

    coarse_shift_m = float(p2.range_coarse_m[bn2] - p1.range_coarse_m[bn1])

    # Search range: coarse +/- N wavelengths
    wl = float(p1.lambdac_m)
    sr_lo = coarse_shift_m - cfg.bed_search_wavelength_margin * wl
    sr_hi = coarse_shift_m + cfg.bed_search_wavelength_margin * wl

    maxlag = (int(np.floor(sr_lo / dr)), int(np.ceil(sr_hi / dr)))

    res = fmcw_xcorr(
        p1.spec_cor,
        p2.spec_cor,
        fi,
        maxlag,
        fe=p1.phase_std_error_rad,
        ge=p2.phase_std_error_rad,
        pad_factor=p1.pad_factor,
    )

    # Choose lag that maximizes coherent correlation magnitude.
    mci = int(np.argmax(np.abs(res.cc)))
    cc = complex(res.cc[mci])
    lag = int(res.lags[mci])
    pse = float(res.pse[mci]) if np.isfinite(res.pse[mci]) else np.nan

    lagdh = lag * dr
    phasedh = -np.angle(cc) * (wl / (4.0 * np.pi))

    dh = float(lagdh + phasedh)
    dhe = float(pse * (wl / (4.0 * np.pi))) if np.isfinite(pse) else float("nan")

    return BedShift(
        method="xcorr",
        dh_m=dh,
        dhe_m=dhe,
        lag_bins=lag,
        phase_rad=float(np.angle(cc)),
        coherence=cc,
    )


def _bed_shift_xcorr_on_spectrum(
    range_axis_m: np.ndarray,
    spec1: np.ndarray,
    spec2: np.ndarray,
    bed_depth_m: float,
    coarse_shift_m: float,
    wl_m: float,
    cfg: ProcessingConfig,
) -> BedShift:
    """Bed shift via complex xcorr on precomputed spectra.

    This is used for sub-band / frequency dependence tests where we do not
    necessarily have phase-error estimates (so dhe_m may be NaN).
    """

    range_axis_m = np.asarray(range_axis_m, dtype=float).reshape(-1)
    spec1 = np.asarray(spec1).reshape(-1)
    spec2 = np.asarray(spec2).reshape(-1)
    dr = float(np.mean(np.diff(range_axis_m)))

    win_lo, win_hi = cfg.xcor_bed_win_m
    fi = np.where(
        (range_axis_m >= min(bed_depth_m + win_lo, bed_depth_m + win_hi))
        & (range_axis_m <= max(bed_depth_m + win_lo, bed_depth_m + win_hi))
    )[0]

    if fi.size < 5:
        raise ValueError("Bed window too small / outside range")

    # Restrict to indices where both spectra are finite
    good = np.isfinite(np.real(spec1[fi])) & np.isfinite(np.real(spec2[fi]))
    fi = fi[good]
    if fi.size < 5:
        raise ValueError("Not enough finite samples in bed window")

    sr_lo = float(coarse_shift_m) - cfg.bed_search_wavelength_margin * float(wl_m)
    sr_hi = float(coarse_shift_m) + cfg.bed_search_wavelength_margin * float(wl_m)
    maxlag = (int(np.floor(sr_lo / dr)), int(np.ceil(sr_hi / dr)))

    res = fmcw_xcorr(spec1, spec2, fi, maxlag)
    mci = int(np.argmax(np.abs(res.cc)))
    cc = complex(res.cc[mci])
    lag = int(res.lags[mci])

    lagdh = lag * dr
    phasedh = -np.angle(cc) * (float(wl_m) / (4.0 * np.pi))
    dh = float(lagdh + phasedh)

    return BedShift(
        method="xcorr_subband",
        dh_m=dh,
        dhe_m=float("nan"),
        lag_bins=lag,
        phase_rad=float(np.angle(cc)),
        coherence=cc,
    )


# -----------------------------
# Public MATLAB-like wrappers
# -----------------------------


def align_coarse(p1: PreprocessedProfile, p2: PreprocessedProfile, cfg: ProcessingConfig) -> CoarseAlignment:
    """Public wrapper for coarse alignment (MATLAB: func_align_coarse)."""

    return _coarse_alignment(p1, p2, cfg.validate())


def align_fine(p1: PreprocessedProfile, p2: PreprocessedProfile, cfg: ProcessingConfig, coarse: CoarseAlignment) -> FineAlignment:
    """Public wrapper for fine alignment (MATLAB: func_align_fine)."""

    return _fine_alignment(p1, p2, cfg.validate(), coarse)


def fit_ice(
    p1: PreprocessedProfile,
    p2: PreprocessedProfile,
    cfg: ProcessingConfig,
    fine: FineAlignment,
    coarse: CoarseAlignment,
    dt_days: float,
) -> StrainFit:
    """Public wrapper for strain fit (MATLAB: func_fit_ice)."""

    return _strain_fit(p1, p2, fine, coarse, cfg.validate(), dt_days)


def strain_melt_between_profiles(
    p1: PreprocessedProfile,
    p2: PreprocessedProfile,
    cfg: ProcessingConfig,
    figures_dir: Optional[Path] = None,
) -> Tuple[StrainMeltResult, CoarseAlignment, FineAlignment, StrainFit, BedShift]:
    """Compute strain rate and basal melt between two consecutive profiles."""
    cfg = cfg.validate()

    # Time difference in days
    dt_days = (p2.timestamp - p1.timestamp).total_seconds() / 86400.0
    if dt_days <= 0:
        raise ValueError(f"Non-positive dt_days between profiles: {dt_days}")

    coarse = _coarse_alignment(p1, p2, cfg)
    fine = _fine_alignment(p1, p2, cfg, coarse)
    fit = _strain_fit(p1, p2, fine, coarse, cfg, dt_days)

    bed_depth = float(p1.bed_depth_m)

    if cfg.do_melt_estimate:
        bed = _bed_shift_xcorr(p1, p2, cfg)

        # Predicted bed displacement due to strain: y = m0 + m1*(z-mid)
        z = bed_depth
        dz = z - fit.mid_range_m
        var_pred = fit.cov[0, 0] + 2.0 * dz * fit.cov[0, 1] + (dz**2) * fit.cov[1, 1]
        dh_strain = float(fit.m0 + fit.m1 * dz)
        dh_strain_err = float(np.sqrt(max(var_pred, 0.0)))

        melt = -(bed.dh_m - dh_strain)
        melt_err = float(np.sqrt(bed.dhe_m**2 + dh_strain_err**2))

        melt_rate = melt * cfg.daysPerYear / dt_days
        melt_rate_err = melt_err * cfg.daysPerYear / dt_days

        bed_shift_m = bed.dh_m
        bed_shift_err = bed.dhe_m

        # -----------------------------
        # Vaňková-style phase vs amplitude disagreement
        # -----------------------------
        # Amplitude method: track the picked bed depth (from preprocessing)
        # and convert to melt by subtracting strain-predicted displacement.
        bed_shift_amp_m = float(p2.bed_depth_m - p1.bed_depth_m)
        melt_amp = float(-(bed_shift_amp_m - dh_strain))
        melt_rate_amp = float(melt_amp * cfg.daysPerYear / dt_days)
        melt_rate_diff = float(melt_rate - melt_rate_amp)
        melt_rate_diff_abs = float(abs(melt_rate_diff))

        # -----------------------------
        # Sub-band / frequency dependence tests
        # -----------------------------
        sub_melt_rates = []
        try:
            if (
                getattr(p1, "spec_cor_subbands", None) is not None
                and getattr(p2, "spec_cor_subbands", None) is not None
                and getattr(p1, "subband_fc_hz", None) is not None
                and getattr(p2, "subband_fc_hz", None) is not None
            ):
                sb1 = np.asarray(p1.spec_cor_subbands)
                sb2 = np.asarray(p2.spec_cor_subbands)
                fc1 = np.asarray(p1.subband_fc_hz, dtype=float)
                fc2 = np.asarray(p2.subband_fc_hz, dtype=float)
                n_sb = int(min(sb1.shape[0], sb2.shape[0], fc1.size, fc2.size))

                for i in range(n_sb):
                    # Use the mean center frequency across the pair
                    fc = float(0.5 * (fc1[i] + fc2[i]))
                    if not np.isfinite(fc) or fc <= 0:
                        sub_melt_rates.append(float("nan"))
                        continue

                    wl = float(p1.ci_m_s / fc)
                    bsub = _bed_shift_xcorr_on_spectrum(
                        p1.range_coarse_m,
                        sb1[i],
                        sb2[i],
                        bed_depth_m=float(bed_depth),
                        coarse_shift_m=float(dh_strain),
                        wl_m=wl,
                        cfg=cfg,
                    )
                    melt_sub = float(-(bsub.dh_m - dh_strain))
                    sub_melt_rates.append(float(melt_sub * cfg.daysPerYear / dt_days))
        except Exception:
            # Diagnostics only: don't let this break the core solver
            sub_melt_rates = []

    else:
        bed = BedShift(method="none", dh_m=float("nan"), dhe_m=float("nan"), lag_bins=0, phase_rad=float("nan"), coherence=complex("nan"))
        melt = float("nan")
        melt_err = float("nan")
        melt_rate = float("nan")
        melt_rate_err = float("nan")
        bed_shift_m = float("nan")
        bed_shift_err = float("nan")

        bed_shift_amp_m = float("nan")
        melt_amp = float("nan")
        melt_rate_amp = float("nan")
        melt_rate_diff = float("nan")
        melt_rate_diff_abs = float("nan")
        sub_melt_rates = []

    # Fit QC metrics (useful for filtering poor-quality pairs)
    n_fit_points = int(np.sum(fit.used_mask))
    if n_fit_points > 0:
        try:
            mean_cohere_used = float(np.nanmean(np.abs(fine.coherence[fit.used_mask])))
        except Exception:
            mean_cohere_used = float("nan")
    else:
        mean_cohere_used = float("nan")

    result = StrainMeltResult(
        station=cfg.station,
        t1=p1.timestamp,
        t2=p2.timestamp,
        dt_days=float(dt_days),
        vsr_per_year=float(fit.vsr_per_year),
        vsr_err_per_year=float(fit.vsr_err_per_year),
        surface_compaction_m=float(fit.intercept0_m),
        surface_compaction_err_m=float(fit.intercept0_err_m),
        bed_depth_m=float(bed_depth),
        bed_shift_m=float(bed_shift_m),
        bed_shift_err_m=float(bed_shift_err),
        melt_m=float(melt),
        melt_err_m=float(melt_err),
        melt_rate_m_per_year=float(melt_rate),
        melt_rate_err_m_per_year=float(melt_rate_err),
        bed_shift_amp_m=float(bed_shift_amp_m),
        melt_m_amp=float(melt_amp),
        melt_rate_m_per_year_amp=float(melt_rate_amp),
        melt_rate_phase_minus_amp_m_per_year=float(melt_rate_diff),
        melt_rate_phase_minus_amp_abs_m_per_year=float(melt_rate_diff_abs),
        melt_rate_subband0_m_per_year=float(sub_melt_rates[0]) if len(sub_melt_rates) > 0 else float("nan"),
        melt_rate_subband1_m_per_year=float(sub_melt_rates[1]) if len(sub_melt_rates) > 1 else float("nan"),
        melt_rate_subband2_m_per_year=float(sub_melt_rates[2]) if len(sub_melt_rates) > 2 else float("nan"),
        melt_rate_subband_std_m_per_year=float(np.nanstd(sub_melt_rates)) if len(sub_melt_rates) > 0 else float("nan"),
        melt_rate_subband_range_m_per_year=float(np.nanmax(sub_melt_rates) - np.nanmin(sub_melt_rates))
        if len(sub_melt_rates) > 0
        else float("nan"),
        n_fit_points=n_fit_points,
        fit_r2=float(fit.r2),
        mean_coherence_used=float(mean_cohere_used),
    )

    return result, coarse, fine, fit, bed
