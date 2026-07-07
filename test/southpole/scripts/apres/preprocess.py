"""Preprocess raw ApRES bursts into phase-sensitive range profiles.

This module ports the *preprocessing* part of the MATLAB workflow:

- func_preprocess.m (options 1..10)
- func_load.m / func_file_format.m + LoadBurstRMB*.m (handled in io.py)
- fmcw_burst_split_by_att.m
- func_cull_freq.m
- func_icethickness.m (here: automated options only)
- func_bad_chirps.m + fmcw_cull_bad_corr.m
- func_depth_conversion.m
- func_correlation.m
- func_estimate_error.m
- fmcw_burst_mean.m
- fmcw_range_vel.m (Brennan et al. 2013 formulation)

The output of this stage is a `PreprocessedProfile` which is what the strain
and basal-melt solvers consume.

Design notes
------------
* The MATLAB code relies on a global `cfg` and stores processing history
  inside the vdat struct. Here we keep things explicit via `ProcessingConfig`.
* We do *not* implement interactive bed picking. For automation we support:
  - `ice_thickness_method = 'max'` (default)
  - `ice_thickness_method = 'use'`

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .config import ProcessingConfig
from .io import ApRESBurst, load_apres_burst


# ----------------------------
# Public dataclasses
# ----------------------------


@dataclass
class PreprocessedProfile:
    """Single time-stamped range profile ready for strain processing."""

    file_path: Path
    name: str
    timestamp: "datetime"

    # Optional position metadata (from .DAT header; often missing)
    gps_on: Optional[int]
    latitude_deg: Optional[float]
    longitude_deg: Optional[float]

    # Range axes
    range_coarse_m: np.ndarray  # (nr,)
    range_fine_m: np.ndarray  # (nr,)
    range_m: np.ndarray  # (nr,)

    # Spectra (complex)
    spec_raw: np.ndarray  # (nr,)
    spec_cor: np.ndarray  # (nr,)

    # Error estimates
    phase_std_error_rad: np.ndarray  # (nr,)
    range_error_m: np.ndarray  # (nr,)

    # Picks / diagnostics
    bed_depth_m: float
    bed_index: int
    noise_depth_m: float

    # Radar parameters
    fs_hz: float
    f0_hz: float
    f1_hz: float
    B_hz: float
    fc_hz: float
    K_rad_s2: float
    ci_m_s: float
    lambdac_m: float
    er_ice: float
    pad_factor: int

    # Optional: sub-band spectra interpolated onto `range_coarse_m`
    # Shape: (n_subbands, nr)
    subband_edges_hz: Optional[np.ndarray] = None
    subband_fc_hz: Optional[np.ndarray] = None
    spec_cor_subbands: Optional[np.ndarray] = None


# ----------------------------
# Utility helpers (MATLAB ports)
# ----------------------------


def _stable_unique_complex(x: np.ndarray) -> np.ndarray:
    """Stable unique for complex arrays (preserve first-seen order)."""

    out = []
    seen = set()
    for v in x.tolist():
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return np.asarray(out, dtype=x.dtype)


def burst_subset(burst: ApRESBurst, chirplist: Sequence[int]) -> ApRESBurst:
    """Keep only the requested chirps (0-indexed) from a burst.

    MATLAB equivalent: fmcw_burst_subset.m
    """

    chirplist = np.asarray(list(chirplist), dtype=int)
    burst.vif = burst.vif[chirplist, :]
    burst.chirp_num = burst.chirp_num[chirplist]
    burst.chirp_att = burst.chirp_att[chirplist]
    burst.chirps_in_burst = int(burst.vif.shape[0])

    att_set_list = _stable_unique_complex(burst.chirp_att)
    burst.attenuator_1_db = np.real(att_set_list).astype(float)
    burst.attenuator_2_db = np.imag(att_set_list).astype(float)

    # Reset chirp numbers to sequential (MATLAB does an idiosyncratic ceil(.../ii))
    burst.chirp_num = np.arange(burst.chirps_in_burst, dtype=int) + 1
    return burst


def burst_split_by_att(burst: ApRESBurst, att_index: int = 1) -> ApRESBurst:
    """Select chirps for one attenuator setting.

    MATLAB equivalent: fmcw_burst_split_by_att.m

    Parameters
    ----------
    att_index:
        1-indexed choice of which attenuator group to keep.
    """

    att_set_list = _stable_unique_complex(burst.chirp_att)
    if att_set_list.size == 0:
        return burst

    att_index = int(att_index)
    if att_index < 1 or att_index > att_set_list.size:
        att_index = 1

    target = att_set_list[att_index - 1]
    chirplist = np.where(burst.chirp_att == target)[0]
    if chirplist.size == 0:
        return burst

    return burst_subset(burst, chirplist)


def burst_mean(burst: ApRESBurst) -> ApRESBurst:
    """Average across chirps (requires a single attenuator setting).

    MATLAB equivalent: fmcw_burst_mean.m
    """

    if burst.vif.ndim != 2:
        raise ValueError("burst.vif must be 2D")

    att_unique = _stable_unique_complex(burst.chirp_att)
    if att_unique.size > 1:
        raise ValueError("Trying to average across multiple attenuator settings")

    if burst.vif.shape[0] > 1:
        burst.vif = np.mean(burst.vif, axis=0, keepdims=True)
        burst.chirps_in_burst = 1
        burst.chirp_num = np.asarray([1], dtype=int)
        burst.chirp_att = att_unique[:1] if att_unique.size else np.asarray([0.0 + 0.0j])
    return burst


def cull_frequency_range(burst: ApRESBurst, frange_hz: Tuple[float, float]) -> ApRESBurst:
    """Crop the time series to a specified frequency range.

    MATLAB equivalent: func_cull_freq.m (fmcw_cull_freq2)

    Notes
    -----
    * Frequency is computed from the chirp law f(t) = f0 + t*K/(2*pi).
    * Updates `burst.f0_hz` and all derived parameters.
    """

    fr0, fr1 = float(frange_hz[0]), float(frange_hz[1])
    if fr1 <= fr0:
        raise ValueError("frange_hz must be (f_low, f_high) with f_high > f_low")

    N = int(burst.samples_per_chirp)
    t = burst.dt_s * np.arange(N, dtype=float)
    f = burst.f0_hz + t * burst.K_rad_s2 / (2.0 * np.pi)

    # Clamp to available
    fr0 = max(fr0, float(f[0]))
    fr1 = min(fr1, float(f[-1]))
    if fr1 <= fr0:
        raise ValueError("Requested frange_hz leaves no data")

    sample_no = np.arange(1, N + 1, dtype=float)
    n1 = int(np.round(np.interp(fr0, f, sample_no)))
    n2 = int(np.round(np.interp(fr1, f, sample_no)))

    if n1 == n2:
        raise ValueError("No data left in frequency range")

    # Convert to Python slicing (0-indexed, end exclusive)
    start = max(n1 - 1, 0)
    end = min(max(n2, start + 1), N)

    burst.vif = burst.vif[:, start:end]
    burst.samples_per_chirp = int(burst.vif.shape[1])

    # Update f0 and derived parameters
    burst.f0_hz = float(f[start])

    dt_s, T_s, f1_hz, B_hz, fc_hz, ci_m_s, lambdac_m = _derive_parameters(
        samples_per_chirp=burst.samples_per_chirp,
        fs_hz=burst.fs_hz,
        f0_hz=burst.f0_hz,
        K_rad_s2=burst.K_rad_s2,
        er_ice=burst.er_ice,
    )
    burst.dt_s = dt_s
    burst.T_s = T_s
    burst.f1_hz = f1_hz
    burst.B_hz = B_hz
    burst.fc_hz = fc_hz
    burst.ci_m_s = ci_m_s
    burst.lambdac_m = lambdac_m

    return burst


def _derive_parameters(
    *,
    samples_per_chirp: int,
    fs_hz: float,
    f0_hz: float,
    K_rad_s2: float,
    er_ice: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """Derive radar parameters (MATLAB fmcw_derive_parameters2)."""

    N = int(samples_per_chirp)
    dt_s = 1.0 / float(fs_hz)
    T_s = (N - 1) / float(fs_hz)
    f1_hz = float(f0_hz + T_s * K_rad_s2 / (2.0 * np.pi))
    B_hz = float((N / float(fs_hz)) * (K_rad_s2 / (2.0 * np.pi)))
    fc_hz = 0.5 * (float(f0_hz) + float(f1_hz))
    ci_m_s = 299_792_458.0 / np.sqrt(float(er_ice))
    lambdac_m = ci_m_s / fc_hz
    return float(dt_s), float(T_s), float(f1_hz), float(B_hz), float(fc_hz), float(ci_m_s), float(lambdac_m)


def fmcw_phase2range(phi_rad: np.ndarray, lambdac_m: float, Rcoarse_m: np.ndarray, K_rad_s2: float, ci_m_s: float) -> np.ndarray:
    """Convert phase (rad) to a fine range offset (m).

    MATLAB equivalent: fmcw_phase2range.m

    Implements the full equation used by Craig Stewart:
        Rfine = phi / (4*pi/lambda - 4*Rcoarse*K/ci^2)

    Parameters
    ----------
    phi_rad:
        Phase in radians (array).
    lambdac_m:
        Centre wavelength (m).
    Rcoarse_m:
        Coarse range bin centre (m), same shape as phi_rad or broadcastable.
    K_rad_s2:
        Chirp gradient in rad/s^2.
    ci_m_s:
        Wave speed in ice (m/s).
    """

    denom = (4.0 * np.pi / float(lambdac_m)) - (4.0 * np.asarray(Rcoarse_m) * float(K_rad_s2) / (float(ci_m_s) ** 2))
    return np.asarray(phi_rad) / denom


def depth_conversion_range_coarse(burst: ApRESBurst, pad_factor: int, max_range_m: float) -> np.ndarray:
    """Compute the coarse range axis (m).

    MATLAB equivalent: func_depth_conversion.m

    Returns the *cropped* axis up to `max_range_m`.
    """

    N = int(burst.samples_per_chirp)
    nf = int(np.round((pad_factor * N) / 2.0 - 0.5))
    n = np.arange(nf, dtype=float)

    Rcoarse = n * burst.ci_m_s / (2.0 * burst.B_hz * pad_factor)

    # Crop
    keep = np.where(Rcoarse <= float(max_range_m))[0]
    if keep.size == 0:
        return Rcoarse[:1]
    return Rcoarse[: keep[-1] + 1]


def fmcw_range_vel(
    vif: np.ndarray,
    burst: ApRESBurst,
    range_coarse_m: np.ndarray,
    pad_factor: int,
    winfun: str = "blackman",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Phase-sensitive FMCW processing (Brennan et al. 2013 style).

    MATLAB equivalent: fmcw_range_vel.m

    Parameters
    ----------
    vif:
        Time-domain IF, shape (nchirps, N) or (N,).
    range_coarse_m:
        Coarse range bin centres (already cropped).
    pad_factor:
        FFT zero-padding / interpolation factor.
    winfun:
        Window function name. Currently supports: "blackman" (default).

    Returns
    -------
    spec_raw:
        Complex spectrum (positive frequencies) for each chirp, shape (nchirps, nr).
    spec_cor:
        Phase-referenced spectrum, shape (nchirps, nr).
    range_fine:
        Fine range offsets (m), shape (nchirps, nr).
    range_total:
        Total range (m) = range_coarse + range_fine, shape (nchirps, nr).
    """

    vif = np.asarray(vif)
    if vif.ndim == 1:
        vif = vif[None, :]
    if vif.ndim != 2:
        raise ValueError("vif must be 1D or 2D")

    nch, N = vif.shape
    p = int(pad_factor)
    if p <= 0:
        raise ValueError("pad_factor must be positive")

    # MATLAB defines nf from p*N and then crops. Here we take nf from the
    # provided (cropped) range axis so indexing stays consistent.
    nr = int(np.asarray(range_coarse_m).size)
    n = np.arange(nr, dtype=float)

    # Window
    if winfun.lower() == "blackman":
        win = np.blackman(N).astype(float)
    else:
        raise ValueError(f"Unsupported winfun '{winfun}'. Use 'blackman'.")

    win_rms = np.sqrt(np.mean(win**2))
    if win_rms == 0:
        win_rms = 1.0

    # Phase reference (MATLAB eq. 17)
    fc = float(burst.fc_hz)
    B = float(burst.B_hz)
    K = float(burst.K_rad_s2)
    phiref = (2.0 * np.pi * fc * n / (B * p)) - (K * (n**2) / (2.0 * (B**2) * (p**2)))
    comp = np.exp(-1j * phiref)

    # Time shift prior to FFT to measure phase at t=T/2
    if N % 2 == 1:
        xn = int(0.5 * (N - 1))
    else:
        xn = int(0.5 * N)

    spec_raw = np.zeros((nch, nr), dtype=np.complex128)
    spec_cor = np.zeros((nch, nr), dtype=np.complex128)

    nfft = p * N
    for ii in range(nch):
        x = vif[ii, :].astype(float)
        x = x - np.mean(x)
        xw = win * x

        vifpad = np.zeros(nfft, dtype=float)
        vifpad[:N] = xw
        vifpad = np.roll(vifpad, -xn)

        fftvif = (np.sqrt(2.0 * p) / float(nfft)) * np.fft.fft(vifpad)
        fftvif = fftvif / win_rms

        s = fftvif[:nr]
        spec_raw[ii, :] = s
        spec_cor[ii, :] = comp * s

    # Fine range and total range
    Rcoarse = np.asarray(range_coarse_m, dtype=float)
    Rcoarse2 = np.tile(Rcoarse[None, :], (nch, 1))
    range_fine = fmcw_phase2range(np.angle(spec_cor), burst.lambdac_m, Rcoarse2, burst.K_rad_s2, burst.ci_m_s)
    range_total = Rcoarse2 + range_fine

    return spec_raw, spec_cor, range_fine, range_total


def cull_bad_chirps_corr(burst: ApRESBurst, cfg: ProcessingConfig) -> ApRESBurst:
    """Cull contaminated chirps using correlation between chirps.

    MATLAB equivalent: fmcw_cull_bad_corr.m

    Notes
    -----
    The MATLAB threshold uses:
        limit = mean(top 75%) - 3*std(top 75%)
    where "top 75%" is implemented by discarding the lowest quartile of the
    per-chirp mean-correlation distribution.

    We keep the same logic, but expose the sigma multiplier as
    `cfg.bad_chirps_sigma`.
    """

    v = burst.vif
    nch = int(burst.chirps_in_burst)

    if nch <= 1:
        return burst

    # Correlation matrix (nch x nch)
    corr_all = np.corrcoef(v)

    # corr_all contains correlation between rows when v is 2D where each row is an observation? Wait:
    # np.corrcoef with rowvar=True (default) treats rows as variables, columns as observations.
    # We want corr across chirps -> each chirp is a variable, so rowvar=True works.
    corr_mean = np.mean(corr_all, axis=1)

    sort_corr = np.sort(corr_mean)
    q = int(np.round(len(sort_corr) * 0.25))
    tail = sort_corr[q:] if q < len(sort_corr) else sort_corr
    if tail.size == 0:
        tail = sort_corr

    limit = float(np.mean(tail) - cfg.bad_chirps_sigma * np.std(tail, ddof=0))
    noisey = corr_mean < limit

    # MATLAB forces at least one "bad" if none found (to keep later logic happy)
    if np.sum(noisey) == 0:
        noisey[0] = True

    good_idx = np.where(~noisey)[0]
    if good_idx.size == 0:
        # As a last resort keep the best-correlated chirp
        good_idx = np.array([int(np.argmax(corr_mean))], dtype=int)

    return burst_subset(burst, good_idx)


def cull_bad_chirps_rms(burst: ApRESBurst, cfg: ProcessingConfig) -> ApRESBurst:
    """RMS-based cull for very large bursts (MATLAB fmcw_cull_bad2).

    In your MATLAB workflow, `func_bad_chirps.m` switches to
    `fmcw_cull_bad2(vdat,'chirp','std')` when the burst is large.

    We implement that same logic here:
      1) De-mean each chirp (meanType='chirp')
      2) Compute per-chirp RMS power `p`
      3) Compute mean/std of the *best half* (lowest p)
      4) Mark chirps as bad if p > mean_best + std_best

    Notes
    -----
    - This function expects the burst to already have a single attenuator
      setting (as in the MATLAB pipeline which calls
      fmcw_burst_split_by_att before bad-chirp removal).
    """

    v = np.asarray(burst.vif)
    if v.ndim != 2 or v.shape[0] <= 1:
        return burst

    # MATLAB sanity checks
    if _stable_unique_complex(np.asarray(burst.chirp_att)).size > 1:
        raise ValueError("Multiple attenuator settings present; split by attenuator before culling bad chirps")

    nch, ns = v.shape
    if nch < 5:
        return burst

    # meanType = 'chirp'
    mvc = np.mean(v, axis=1, keepdims=True)
    p = np.sqrt(np.mean((v - mvc) ** 2, axis=1))

    # fitType = 'std' : mean/std of best half
    n_best = int(np.round(nch * 0.5))
    n_best = max(n_best, 1)
    best = np.sort(p)[:n_best]
    mean_p = float(np.mean(best))
    std_p = float(np.std(best, ddof=1)) if best.size >= 2 else 0.0

    noisey = p > (mean_p + std_p)
    good_idx = np.where(~noisey)[0]
    if good_idx.size == 0:
        good_idx = np.array([int(np.argmin(p))], dtype=int)

    return burst_subset(burst, good_idx)


def func_bad_chirps(burst: ApRESBurst, cfg: ProcessingConfig) -> ApRESBurst:
    """Apply the selected bad-chirp removal method.

    MATLAB equivalent: func_bad_chirps.m
    """

    method = getattr(cfg, "bad_chirps_method", "BadChirps")

    if method == "nChirps":
        # Drop first N chirps
        n = int(cfg.delete_first_chirps)
        if n <= 0:
            return burst
        keep = np.arange(burst.chirps_in_burst, dtype=int)
        keep = keep[n:]
        if keep.size == 0:
            keep = np.array([0], dtype=int)
        return burst_subset(burst, keep)

    # Default: "BadChirps" branch
    if burst.chirps_in_burst > cfg.max_chirps_correlation:
        return cull_bad_chirps_rms(burst, cfg)
    return cull_bad_chirps_corr(burst, cfg)


def estimate_noise_depth(
    burst: ApRESBurst,
    cfg: ProcessingConfig,
    *,
    range_coarse_m: np.ndarray,
    bed_depth_m: Optional[float] = None,
) -> float:
    """Estimate a depth beyond which coherence is too low ("noise level").

    MATLAB equivalent: func_correlation.m (the *noise* part)

    This uses the correlation between the amplitude spectra of the mean of
    the first half of chirps and the mean of the second half.
    """

    nch = int(burst.chirps_in_burst)
    if nch < 2:
        return float(bed_depth_m) if bed_depth_m is not None else 0.0

    # Split into two halves (MATLAB: first ceil(n/2), last floor(n/2))
    n1 = int(np.ceil(nch / 2.0))
    n2 = int(np.floor(nch / 2.0))

    b1 = ApRESBurst(**{**burst.__dict__})  # shallow copy
    b2 = ApRESBurst(**{**burst.__dict__})

    b1.vif = burst.vif[:n1, :]
    b1.chirp_num = burst.chirp_num[:n1]
    b1.chirp_att = burst.chirp_att[:n1]
    b1.chirps_in_burst = int(b1.vif.shape[0])

    b2.vif = burst.vif[nch - n2 :, :]
    b2.chirp_num = burst.chirp_num[nch - n2 :]
    b2.chirp_att = burst.chirp_att[nch - n2 :]
    b2.chirps_in_burst = int(b2.vif.shape[0])

    b1 = burst_mean(b1)
    b2 = burst_mean(b2)

    _, spec1, _, _ = fmcw_range_vel(b1.vif, b1, range_coarse_m, cfg.pad_factor)
    _, spec2, _, _ = fmcw_range_vel(b2.vif, b2, range_coarse_m, cfg.pad_factor)

    a1 = np.abs(spec1[0])
    a2 = np.abs(spec2[0])

    depth = np.asarray(range_coarse_m)

    win = int(10 * cfg.pad_factor)
    dlength = int(depth.size - win)

    corr = np.zeros(depth.size, dtype=float)
    for offset in range(win + 1, dlength):
        s1 = a1[offset - win : offset + win + 1]
        s2 = a2[offset - win : offset + win + 1]
        if s1.size < 2 or s2.size < 2:
            continue
        R = np.corrcoef(s1, s2)
        corr[offset] = float(R[0, 1])

    # Moving average
    window_MA = int(20 * cfg.pad_factor)
    corr_ma = np.zeros_like(corr)
    for offset in range(window_MA + win + 1, dlength - window_MA):
        corr_ma[offset] = float(np.mean(corr[offset - window_MA : offset + window_MA + 1]))

    # First depth where correlation drops below limit
    fi = np.where((corr_ma <= cfg.correlation_limit) & (corr_ma != 0))[0]
    if fi.size >= 1:
        noise_depth = float(depth[fi[0]])
    else:
        noise_depth = float(bed_depth_m) if bed_depth_m is not None else 0.0

    if bed_depth_m is not None:
        noise_depth = min(noise_depth, float(bed_depth_m))

    return float(noise_depth)


def estimate_phase_error(
    burst: ApRESBurst,
    cfg: ProcessingConfig,
    *,
    range_coarse_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate phase standard error and convert to range error.

    MATLAB equivalent: func_estimate_error.m
    """

    # Mean burst
    bmean = ApRESBurst(**{**burst.__dict__})
    bmean = burst_mean(bmean)

    _, spec_cor_mean, _, _ = fmcw_range_vel(bmean.vif, bmean, range_coarse_m, cfg.pad_factor)
    _, spec_cor_all, _, _ = fmcw_range_vel(burst.vif, burst, range_coarse_m, cfg.pad_factor)

    F = spec_cor_all  # (nchirps, nr)
    # MATLAB std uses N-1 (ddof=1)
    phase_std_dev = np.std(F, axis=0, ddof=1) / (np.sqrt(2.0) * np.abs(spec_cor_mean[0]))

    # Standard error of mean shot (MATLAB uses sqrt(n-1))
    denom = max(float(burst.chirps_in_burst - 1), 1.0)
    phase_std_error = phase_std_dev / np.sqrt(denom)

    range_error = fmcw_phase2range(phase_std_error, bmean.lambdac_m, range_coarse_m, bmean.K_rad_s2, bmean.ci_m_s)
    return np.asarray(phase_std_error, dtype=float), np.asarray(range_error, dtype=float)


def estimate_ice_thickness(
    burst: ApRESBurst,
    cfg: ProcessingConfig,
    *,
    range_coarse_m: np.ndarray,
    spec_cor_mean: np.ndarray,
) -> Tuple[int, float]:
    """Estimate ice thickness / bed depth.

    MATLAB equivalent: func_icethickness.m (automated branches only)

    Returns
    -------
    bed_index, bed_depth_m
    """

    method = getattr(cfg, "ice_thickness_method", "max")

    if method == "use":
        H = float(getattr(cfg, "ice_thickness_use_m", np.nan))
        if not np.isfinite(H):
            raise ValueError("ice_thickness_method='use' but ice_thickness_use_m is not set")
        idx = int(np.searchsorted(range_coarse_m, H, side="left"))
        idx = min(max(idx, 0), range_coarse_m.size - 1)
        return idx, float(range_coarse_m[idx])

    # Default: 'max'
    min_m = float(getattr(cfg, "ice_thickness_min_m", cfg.bed_search_min_m))
    idx_search = range_coarse_m > min_m
    if not np.any(idx_search):
        idx_search = slice(None)

    amp = np.abs(spec_cor_mean)
    amp_search = amp[idx_search]
    if amp_search.size == 0:
        idx = int(np.argmax(amp))
    else:
        local = int(np.argmax(amp_search))
        idx0 = int(np.where(idx_search)[0][0]) if not isinstance(idx_search, slice) else 0
        idx = idx0 + local

    idx = min(max(idx, 0), range_coarse_m.size - 1)
    return int(idx), float(range_coarse_m[idx])


# ----------------------------
# Main entry point
# ----------------------------


def preprocess_file(path: Path, cfg: ProcessingConfig, *, figures_dir: Optional[Path] = None) -> PreprocessedProfile:
    """Preprocess one raw ApRES file into a range profile.

    This is the Python analogue of calling `func_preprocess` in MATLAB with
    options [1,2,3,4,5,6,7,8,9,10] (but with automated-only bed picking).

    Parameters
    ----------
    path:
        Path to the raw .DAT/.dat file.
    cfg:
        Processing configuration.
    figures_dir:
        Directory for optional diagnostic plots (created if needed). The
        plotting functions live in `apres.plotting`.

    Returns
    -------
    PreprocessedProfile
    """

    if not getattr(cfg, "_validated", False):
        cfg.validate()

    path = Path(path)

    # 1. Load
    burst = load_apres_burst(
        path,
        burst=1,
        samples_per_chirp=cfg.samples_per_chirp,
        fs_hz=cfg.sampling_frequency_hz,
        f0_hz=cfg.f0_hz,
        K_rad_s2=cfg.K_rad_s2,
        er_ice=cfg.relative_permittivity_ice,
        use_impdar=cfg.use_impdar_reader,
    )

    # 2. Split by attenuator (keep one group)
    burst = burst_split_by_att(burst, getattr(cfg, "attenuator_index", 1))

    # 3. Cull frequency range (optional)
    frange = getattr(cfg, "frequency_range_hz", None)
    if frange is not None:
        burst = cull_frequency_range(burst, frange)

    # 4. (Ice thickness step exists in MATLAB mainly to tune max_range.
    #     We do automated bed picking later after range conversion.)

    # 5. Bad chirps
    burst = func_bad_chirps(burst, cfg)

    # 6. Depth conversion (coarse axis)
    range_coarse = depth_conversion_range_coarse(burst, cfg.pad_factor, cfg.max_range_m)

    # 7. Correlation-based noise depth (needs range processing on halves)
    # We'll compute later once we have a bed pick (so noise is capped at bed).

    # 9. Burst mean (after culling bad chirps)
    burst_meaned = ApRESBurst(**{**burst.__dict__})
    burst_meaned = burst_mean(burst_meaned)

    # 10. Range processing
    spec_raw_mean, spec_cor_mean, range_fine_mean, range_mean = fmcw_range_vel(
        burst_meaned.vif,
        burst_meaned,
        range_coarse,
        cfg.pad_factor,
        winfun=getattr(cfg, "winfun", "blackman"),
    )

    # Flatten chirp dimension (meaned burst has nch=1)
    spec_raw_mean = spec_raw_mean[0]
    spec_cor_mean = spec_cor_mean[0]
    range_fine_mean = range_fine_mean[0]
    range_mean = range_mean[0]

    # 4 (continued): bed / ice thickness pick
    bed_idx, bed_depth = estimate_ice_thickness(
        burst_meaned,
        cfg,
        range_coarse_m=range_coarse,
        spec_cor_mean=spec_cor_mean,
    )

    # 7 (now): noise depth
    noise_depth = estimate_noise_depth(burst, cfg, range_coarse_m=range_coarse, bed_depth_m=bed_depth)

    # 8. Phase error
    phase_std_error, range_error = estimate_phase_error(burst, cfg, range_coarse_m=range_coarse)

    # Ensure float arrays
    phase_std_error = np.asarray(phase_std_error, dtype=float)
    range_error = np.asarray(range_error, dtype=float)

    # ----------------------------------------------
    # Optional: sub-band spectra for frequency-dependence diagnostics
    # ----------------------------------------------
    subband_edges_hz = None
    subband_fc_hz = None
    spec_cor_subbands = None

    if getattr(cfg, "subbands_enabled", False) and getattr(cfg, "subband_fracs", None):
        try:
            f0_full = float(burst_meaned.f0_hz)
            f1_full = float(burst_meaned.f1_hz)
            bw_full = float(f1_full - f0_full)
            if bw_full > 0:
                edges = []
                fcs = []
                specs = []

                for (a, b) in cfg.subband_fracs:
                    a = float(a)
                    b = float(b)
                    f_low = f0_full + a * bw_full
                    f_high = f0_full + b * bw_full

                    # Work on a copy of the meaned burst so we don't mutate
                    # the full-band burst used above.
                    sub = ApRESBurst(**{**burst_meaned.__dict__})
                    sub.vif = np.asarray(burst_meaned.vif).copy()

                    sub = cull_frequency_range(sub, (f_low, f_high))
                    range_coarse_sub = depth_conversion_range_coarse(sub, cfg.pad_factor, cfg.max_range_m)

                    _, spec_cor_sub, _, _ = fmcw_range_vel(
                        sub.vif,
                        sub,
                        range_coarse_sub,
                        cfg.pad_factor,
                        winfun=getattr(cfg, "winfun", "blackman"),
                    )

                    # Meaned burst has nch=1
                    spec_cor_sub = np.asarray(spec_cor_sub[0])

                    # Interpolate onto the full-band range axis so downstream
                    # bed-window indexing is consistent.
                    re = np.interp(range_coarse, range_coarse_sub, np.real(spec_cor_sub), left=np.nan, right=np.nan)
                    im = np.interp(range_coarse, range_coarse_sub, np.imag(spec_cor_sub), left=np.nan, right=np.nan)
                    specs.append(re + 1j * im)
                    edges.append([f_low, f_high])
                    fcs.append(float(sub.fc_hz))

                if specs:
                    spec_cor_subbands = np.stack(specs, axis=0)
                    subband_edges_hz = np.asarray(edges, dtype=float)
                    subband_fc_hz = np.asarray(fcs, dtype=float)
        except Exception:
            # Keep preprocessing robust: if sub-band processing fails for any
            # reason we simply omit these diagnostics.
            subband_edges_hz = None
            subband_fc_hz = None
            spec_cor_subbands = None

    # Package
    return PreprocessedProfile(
        file_path=path,
        name=path.stem,
        timestamp=burst.timestamp,
        gps_on=getattr(burst, "gps_on", None),
        latitude_deg=getattr(burst, "latitude_deg", None),
        longitude_deg=getattr(burst, "longitude_deg", None),
        range_coarse_m=range_coarse,
        range_fine_m=range_fine_mean,
        range_m=range_mean,
        spec_raw=spec_raw_mean,
        spec_cor=spec_cor_mean,
        phase_std_error_rad=phase_std_error,
        range_error_m=range_error,
        bed_depth_m=float(bed_depth),
        bed_index=int(bed_idx),
        noise_depth_m=float(noise_depth),
        fs_hz=float(burst.fs_hz),
        f0_hz=float(burst.f0_hz),
        f1_hz=float(burst.f1_hz),
        B_hz=float(burst.B_hz),
        fc_hz=float(burst.fc_hz),
        K_rad_s2=float(burst.K_rad_s2),
        ci_m_s=float(burst.ci_m_s),
        lambdac_m=float(burst.lambdac_m),
        er_ice=float(burst.er_ice),
        pad_factor=int(cfg.pad_factor),
        subband_edges_hz=subband_edges_hz,
        subband_fc_hz=subband_fc_hz,
        spec_cor_subbands=spec_cor_subbands,
    )
