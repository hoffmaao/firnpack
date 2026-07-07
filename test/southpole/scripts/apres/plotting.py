"""Plotting helpers for diagnosing the ApRES strain/melt solver.

The original MATLAB workflow produces many figures. For the Python port we
keep a small set of high-value diagnostics and save them to disk so the
pipeline can run headlessly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import matplotlib

# Headless backend so scripts can run on servers / in batch
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import ProcessingConfig
from .preprocess import PreprocessedProfile
from .strain import BedShift, CoarseAlignment, FineAlignment, StrainFit, StrainMeltResult


def _db(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    x = np.asarray(x)
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def _apply_style(cfg: ProcessingConfig) -> None:
    plt.rcParams.update(
        {
            "font.size": cfg.font_size,
            "lines.linewidth": cfg.line_width,
            "figure.dpi": cfg.figure_dpi,
        }
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _local_maxima(y: np.ndarray) -> np.ndarray:
    """Return indices of simple 1D local maxima.

    This is intentionally lightweight (no scipy dependency). It is used for
    basal multi-peak metrics where we simply want to know if there are multiple
    comparably-strong peaks in a small window around the bed.
    """

    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1


def basal_metrics_from_profile(p: PreprocessedProfile, cfg: ProcessingConfig) -> Dict[str, float]:
    """Compute per-acquisition basal-return diagnostics.

    Metrics are intended for diagnosing basal complexity (multiple peaks,
    trailing energy) that can cause the bed tracker to hop between reflectors.

    Returns a dict with NaNs for values that cannot be computed.
    """

    cfg = cfg.validate()

    bed_depth = float(p.bed_depth_m)
    bed_idx = int(p.bed_index)
    rng = np.asarray(p.range_m, dtype=float)
    amp = np.abs(np.asarray(p.spec_cor))

    out: Dict[str, float] = {
        "bed_depth_m": bed_depth,
        "bed_peak_amp_db": float("nan"),
        "basal_window_mean_amp_db": float("nan"),
        "peak1_depth_m": float("nan"),
        "peak2_depth_m": float("nan"),
        "peak1_amp_db": float("nan"),
        "peak2_amp_db": float("nan"),
        "peak_ratio_db": float("nan"),
        "peak_separation_m": float("nan"),
    }

    # --- Basal peak amplitude at the picked bed
    if 0 <= bed_idx < amp.size:
        out["bed_peak_amp_db"] = float(_db(amp[bed_idx]))

    # --- Basal window mean amplitude (linear-mean, then to dB)
    L = float(cfg.basal_window_mean_len_m)
    if L > 0:
        m = (rng >= bed_depth) & (rng <= bed_depth + L)
        if np.any(m):
            out["basal_window_mean_amp_db"] = float(_db(np.nanmean(amp[m])))

    # --- Multi-peak metrics in a window around the bed
    above = float(cfg.basal_peak_window_above_m)
    below = float(cfg.basal_peak_window_below_m)
    m2 = (rng >= bed_depth - above) & (rng <= bed_depth + below)
    if np.sum(m2) >= 5:
        rng_w = rng[m2]
        amp_w = amp[m2]
        # Identify local maxima; if none, fall back to the global maximum
        pk = _local_maxima(amp_w)
        if pk.size == 0:
            pk = np.array([int(np.nanargmax(amp_w))], dtype=int)

        # Sort peaks by amplitude
        pk_sorted = pk[np.argsort(amp_w[pk])[::-1]]
        if pk_sorted.size >= 1:
            i1 = int(pk_sorted[0])
            out["peak1_depth_m"] = float(rng_w[i1])
            out["peak1_amp_db"] = float(_db(amp_w[i1]))
        if pk_sorted.size >= 2:
            i2 = int(pk_sorted[1])
            out["peak2_depth_m"] = float(rng_w[i2])
            out["peak2_amp_db"] = float(_db(amp_w[i2]))
            out["peak_ratio_db"] = float(out["peak1_amp_db"] - out["peak2_amp_db"])
            out["peak_separation_m"] = float(abs(out["peak1_depth_m"] - out["peak2_depth_m"]))

    return out


def savefig(fig: plt.Figure, outpath: Path, cfg: ProcessingConfig) -> None:
    _ensure_parent(outpath)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_preprocess_profile(p: PreprocessedProfile, outpath: Path, cfg: ProcessingConfig) -> None:
    """Amplitude vs range for a single profile (bed pick + noise depth)."""
    cfg = cfg.validate()
    _apply_style(cfg)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p.range_m, _db(p.spec_cor))
    ax.axvline(p.bed_depth_m, linestyle="--")
    ax.axvline(p.noise_depth_m, linestyle=":")

    ax.set_xlabel("Range / depth (m)")
    ax.set_ylabel("Amplitude (dB, arbitrary)")
    ax.set_title(f"{cfg.station} | {p.name} | {p.timestamp.isoformat()}")
    ax.grid(True, alpha=0.3)

    ax.text(
        0.02,
        0.02,
        f"bed ≈ {p.bed_depth_m:.1f} m\nnoise ≈ {p.noise_depth_m:.1f} m",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
    )

    savefig(fig, outpath, cfg)


def plot_pair_diagnostics(
    p1: PreprocessedProfile,
    p2: PreprocessedProfile,
    result: StrainMeltResult,
    coarse: CoarseAlignment,
    fine: FineAlignment,
    fit: StrainFit,
    bed: BedShift,
    outpath: Path,
    cfg: ProcessingConfig,
) -> None:
    """Multi-panel figure diagnosing a single profile pair."""
    cfg = cfg.validate()
    _apply_style(cfg)

    fig, axs = plt.subplots(3, 1, figsize=(7, 10))

    # (1) Amplitude overlay
    axs[0].plot(p1.range_m, _db(p1.spec_cor), label=p1.name)
    axs[0].plot(p2.range_m, _db(p2.spec_cor), label=p2.name)
    axs[0].axvline(p1.bed_depth_m, linestyle="--")
    axs[0].set_xlabel("Range / depth (m)")
    axs[0].set_ylabel("Amplitude (dB)")
    axs[0].set_title(
        f"{cfg.station}: {p1.timestamp.date().isoformat()} → {p2.timestamp.date().isoformat()} (Δt={result.dt_days:.2f} d)"
    )
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best", fontsize=max(8, cfg.font_size - 2))

    # (2) Displacement profile + fit
    axs[1].plot(fine.range_m, fine.dh_m, label="dh (fine)")
    # highlight used points
    used = fit.used_mask
    if np.any(used):
        axs[1].plot(fine.range_m[used], fine.dh_m[used], linestyle="None", marker="o", label="used")

    # Fit line over same depth span as fine grid
    z = fine.range_m
    zmid = fit.mid_range_m
    dh_fit = fit.m0 + fit.m1 * (z - zmid)
    axs[1].plot(z, dh_fit, label="fit")

    axs[1].axvline(p1.bed_depth_m, linestyle="--")
    axs[1].set_xlabel("Depth (m)")
    axs[1].set_ylabel("Displacement dh (m)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best", fontsize=max(8, cfg.font_size - 2))

    # (3) Coherence / correlation
    axs[2].plot(fine.range_m, np.abs(fine.coherence), label="|coherence|")
    axs[2].axhline(cfg.min_cohere_fine, linestyle=":")
    axs[2].axvline(p1.bed_depth_m, linestyle="--")
    axs[2].set_xlabel("Depth (m)")
    axs[2].set_ylabel("|coherence| (0..1)")
    axs[2].grid(True, alpha=0.3)

    # Extra diagnostics (Vaňková-style)
    extra_lines = ""
    if np.isfinite(result.melt_rate_m_per_year_amp):
        extra_lines += f"melt (amp) = {result.melt_rate_m_per_year_amp:.3f} m/yr\n"
    if np.isfinite(result.melt_rate_phase_minus_amp_m_per_year):
        extra_lines += f"phase-amp = {result.melt_rate_phase_minus_amp_m_per_year:.3f} m/yr\n"
    if np.isfinite(result.melt_rate_subband_std_m_per_year):
        extra_lines += f"subband std = {result.melt_rate_subband_std_m_per_year:.3f} m/yr\n"

    txt = (
        f"vsr = {result.vsr_per_year:.3e} ± {result.vsr_err_per_year:.1e} /yr\n"
        f"melt (phase) = {result.melt_rate_m_per_year:.3f} ± {result.melt_rate_err_m_per_year:.3f} m/yr\n"
        + extra_lines
        + f"bed shift = {result.bed_shift_m:.3f} ± {result.bed_shift_err_m:.3f} m\n"
        + f"fit: n={result.n_fit_points:d}, R^2={result.fit_r2:.2f}, mean(|coh|)={result.mean_coherence_used:.3f}"
    )
    axs[2].text(0.02, 0.02, txt, transform=axs[2].transAxes, va="bottom", ha="left")

    savefig(fig, outpath, cfg)


def plot_timeseries(
    results: List[StrainMeltResult],
    outpath: Path,
    cfg: ProcessingConfig,
    profiles: Optional[List[PreprocessedProfile]] = None,
) -> None:
    """Time-series plots.

    By default this produces a two-panel time series (strain + melt). If
    `profiles` is provided, extra basal-return diagnostics are plotted:

    - bed depth
    - basal peak amplitude
    - basal window mean amplitude
    - multi-peak metrics (peak ratio and peak separation)
    """

    if len(results) == 0:
        return

    cfg = cfg.validate()
    _apply_style(cfg)

    # Pairwise results (rates)
    t_pair = [r.t2 for r in results]
    vsr = np.array([r.vsr_per_year for r in results], dtype=float)
    melt = np.array([r.melt_rate_m_per_year for r in results], dtype=float)

    def _rolling_median_nan(x: np.ndarray, window: int) -> np.ndarray:
        """Centered rolling median with NaN-safe behaviour."""

        x = np.asarray(x, dtype=float)
        n = x.size
        if window <= 1 or n == 0:
            return x.copy()
        half = window // 2
        out = np.full(n, np.nan, dtype=float)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            out[i] = float(np.nanmedian(x[lo:hi]))
        return out

    # Optional median filtering for display
    vsr_filt = None
    melt_filt = None
    if cfg.timeseries_median_filter_enabled and cfg.timeseries_median_filter_window > 1:
        w = int(cfg.timeseries_median_filter_window)

        # Prefer values already computed in the driver (if present)
        vsr_attr = np.array(
            [np.nan if r.vsr_per_year_medfilt is None else float(r.vsr_per_year_medfilt) for r in results], dtype=float
        )
        melt_attr = np.array(
            [
                np.nan
                if r.melt_rate_m_per_year_medfilt is None
                else float(r.melt_rate_m_per_year_medfilt)
                for r in results
            ],
            dtype=float,
        )
        vsr_filt = vsr_attr if np.isfinite(vsr_attr).any() else _rolling_median_nan(vsr, w)
        melt_filt = melt_attr if np.isfinite(melt_attr).any() else _rolling_median_nan(melt, w)

    def _robust_ylim(y: np.ndarray) -> tuple[float, float] | None:
        """Return robust (lo, hi) y-limits based on percentiles + padding."""

        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return None
        if y.size >= 5:
            p_lo, p_hi = cfg.timeseries_plot_robust_percentiles
            lo = float(np.nanpercentile(y, p_lo))
            hi = float(np.nanpercentile(y, p_hi))
        else:
            lo = float(np.nanmin(y))
            hi = float(np.nanmax(y))

        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None

        if hi == lo:
            span = abs(lo) if lo != 0 else 1.0
            lo -= 0.5 * span
            hi += 0.5 * span
        else:
            span = hi - lo

        pad = float(cfg.timeseries_plot_robust_pad_frac) * span
        return lo - pad, hi + pad

    # -----------------------------------------
    # Compute additional basal diagnostics (per acquisition)
    # -----------------------------------------
    acq_t = None
    bed_depth = None
    bed_peak_amp_db = None
    basal_mean_amp_db = None
    peak_ratio_db = None
    peak_sep_m = None
    peak1_depth = None
    peak2_depth = None
    peak1_amp_db = None
    peak2_amp_db = None

    if profiles is not None and len(profiles) > 0:
        profiles = sorted(profiles, key=lambda p: p.timestamp)
        acq_t = [p.timestamp for p in profiles]
        metrics = [basal_metrics_from_profile(p, cfg) for p in profiles]

        bed_depth = np.array([m["bed_depth_m"] for m in metrics], dtype=float)
        bed_peak_amp_db = np.array([m["bed_peak_amp_db"] for m in metrics], dtype=float)
        basal_mean_amp_db = np.array([m["basal_window_mean_amp_db"] for m in metrics], dtype=float)
        peak_ratio_db = np.array([m["peak_ratio_db"] for m in metrics], dtype=float)
        peak_sep_m = np.array([m["peak_separation_m"] for m in metrics], dtype=float)

        peak1_depth = np.array([m["peak1_depth_m"] for m in metrics], dtype=float)
        peak2_depth = np.array([m["peak2_depth_m"] for m in metrics], dtype=float)
        peak1_amp_db = np.array([m["peak1_amp_db"] for m in metrics], dtype=float)
        peak2_amp_db = np.array([m["peak2_amp_db"] for m in metrics], dtype=float)

        # Optional median filtering for diagnostics as well (for readability)
        if cfg.timeseries_median_filter_enabled and cfg.timeseries_median_filter_window > 1:
            w = int(cfg.timeseries_median_filter_window)
            bed_depth_f = _rolling_median_nan(bed_depth, w)
            bed_peak_amp_db_f = _rolling_median_nan(bed_peak_amp_db, w)
            basal_mean_amp_db_f = _rolling_median_nan(basal_mean_amp_db, w)
            peak_ratio_db_f = _rolling_median_nan(peak_ratio_db, w)
            peak_sep_m_f = _rolling_median_nan(peak_sep_m, w)
            peak1_depth_f = _rolling_median_nan(peak1_depth, w)
            peak2_depth_f = _rolling_median_nan(peak2_depth, w)
            peak1_amp_db_f = _rolling_median_nan(peak1_amp_db, w)
            peak2_amp_db_f = _rolling_median_nan(peak2_amp_db, w)
        else:
            bed_depth_f = bed_peak_amp_db_f = basal_mean_amp_db_f = peak_ratio_db_f = peak_sep_m_f = None
            peak1_depth_f = peak2_depth_f = peak1_amp_db_f = peak2_amp_db_f = None

    # -----------------------------------------
    # Plot
    # -----------------------------------------
    if profiles is None or acq_t is None:
        nrows = 2
    else:
        nrows = 5

    fig, axs = plt.subplots(nrows, 1, figsize=(9, 3.0 * nrows), sharex=True)
    if nrows == 1:
        axs = [axs]

    # --- (1) Strain rate
    if cfg.timeseries_plot_raw_as_scatter:
        axs[0].scatter(t_pair, vsr, label="raw", alpha=cfg.timeseries_plot_raw_alpha)
    else:
        axs[0].plot(t_pair, vsr, marker="o", label="raw", alpha=cfg.timeseries_plot_raw_alpha)
    if vsr_filt is not None:
        axs[0].plot(t_pair, vsr_filt, label=f"median({cfg.timeseries_median_filter_window})")
    axs[0].set_ylabel("Vertical strain rate (/yr)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best", fontsize=max(8, cfg.font_size - 2))

    if cfg.timeseries_plot_use_robust_ylim:
        y_for_lim = (
            np.asarray(vsr_filt, dtype=float)
            if (cfg.timeseries_plot_use_filtered_for_ylim and vsr_filt is not None)
            else vsr
        )
        lims = _robust_ylim(y_for_lim)
        if lims is not None:
            axs[0].set_ylim(*lims)

    # --- (2) Melt rate
    if cfg.timeseries_plot_raw_as_scatter:
        axs[1].scatter(t_pair, melt, label="raw", alpha=cfg.timeseries_plot_raw_alpha)
    else:
        axs[1].plot(t_pair, melt, marker="o", label="raw", alpha=cfg.timeseries_plot_raw_alpha)
    if melt_filt is not None:
        axs[1].plot(t_pair, melt_filt, label=f"median({cfg.timeseries_median_filter_window})")
    axs[1].set_ylabel("Basal melt rate (m/yr)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best", fontsize=max(8, cfg.font_size - 2))

    if cfg.timeseries_plot_use_robust_ylim:
        y_for_lim = (
            np.asarray(melt_filt, dtype=float)
            if (cfg.timeseries_plot_use_filtered_for_ylim and melt_filt is not None)
            else melt
        )
        lims = _robust_ylim(y_for_lim)
        if lims is not None:
            axs[1].set_ylim(*lims)
            lo, hi = lims
            n_clip = int(np.sum((melt < lo) | (melt > hi)))
            if n_clip > 0:
                axs[1].text(
                    0.02,
                    0.95,
                    f"{n_clip} outlier(s) outside y-limits",
                    transform=axs[1].transAxes,
                    va="top",
                    ha="left",
                    fontsize=max(8, cfg.font_size - 2),
                )

    if nrows > 2 and acq_t is not None:
        # --- (3) Bed depth
        axs[2].scatter(acq_t, bed_depth, label="picked bed", alpha=0.7)
        axs[2].scatter(acq_t, peak1_depth, label="peak1", alpha=0.5)
        axs[2].scatter(acq_t, peak2_depth, label="peak2", alpha=0.5)
        if bed_depth_f is not None:
            axs[2].plot(acq_t, bed_depth_f, label=f"median({cfg.timeseries_median_filter_window})")
        if peak1_depth_f is not None:
            axs[2].plot(acq_t, peak1_depth_f, label=f"peak1 median({cfg.timeseries_median_filter_window})")
        if peak2_depth_f is not None:
            axs[2].plot(acq_t, peak2_depth_f, label=f"peak2 median({cfg.timeseries_median_filter_window})")
        axs[2].set_ylabel("Bed depth (m)")
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc="best", fontsize=max(8, cfg.font_size - 2))

        # --- (4) Basal amplitude metrics
        axs[3].scatter(acq_t, bed_peak_amp_db, label="picked bed", alpha=0.7)
        axs[3].scatter(acq_t, peak1_amp_db, label="peak1", alpha=0.6)
        axs[3].scatter(acq_t, peak2_amp_db, label="peak2", alpha=0.6)
        axs[3].scatter(acq_t, basal_mean_amp_db, label=f"mean[{cfg.basal_window_mean_len_m:.0f} m]", alpha=0.7)
        if bed_peak_amp_db_f is not None:
            axs[3].plot(acq_t, bed_peak_amp_db_f, label=f"bed peak median({cfg.timeseries_median_filter_window})")
        if peak1_amp_db_f is not None:
            axs[3].plot(acq_t, peak1_amp_db_f, label=f"peak1 median({cfg.timeseries_median_filter_window})")
        if peak2_amp_db_f is not None:
            axs[3].plot(acq_t, peak2_amp_db_f, label=f"peak2 median({cfg.timeseries_median_filter_window})")
        if basal_mean_amp_db_f is not None:
            axs[3].plot(
                acq_t,
                basal_mean_amp_db_f,
                label=f"mean median({cfg.timeseries_median_filter_window})",
            )
        axs[3].set_ylabel("Basal amplitude (dB)")
        axs[3].grid(True, alpha=0.3)
        axs[3].legend(loc="best", fontsize=max(8, cfg.font_size - 2))

        # --- (5) Multi-peak metrics (ratio + separation)
        ax = axs[4]
        ax2 = ax.twinx()

        ax.scatter(acq_t, peak_ratio_db, label="peak ratio (dB)", alpha=0.7)
        if peak_ratio_db_f is not None:
            ax.plot(acq_t, peak_ratio_db_f, label=f"ratio median({cfg.timeseries_median_filter_window})")
        ax.set_ylabel("Peak1 - Peak2 (dB)")

        ax2.scatter(acq_t, peak_sep_m, label="peak separation (m)", alpha=0.5)
        if peak_sep_m_f is not None:
            ax2.plot(acq_t, peak_sep_m_f, label=f"sep median({cfg.timeseries_median_filter_window})")
        ax2.set_ylabel("Peak separation (m)")

        ax.grid(True, alpha=0.3)

        # Combined legend
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=max(8, cfg.font_size - 2))

    axs[-1].set_xlabel("Time")
    fig.suptitle(f"{cfg.station}: strain + melt + basal diagnostics")
    savefig(fig, outpath, cfg)


def plot_melt_method_diagnostics(results: List[StrainMeltResult], outpath: Path, cfg: ProcessingConfig) -> None:
    """Plot diagnostics comparing phase- and amplitude-derived melt, and
    sub-band frequency dependence.

    This follows the spirit of Vaňková et al. (2021): when basal returns are
    complex (e.g., intermittently accreted ice), phase-derived melt and
    amplitude-derived thickness change can disagree, and phase-derived melt
    can become frequency dependent.
    """

    if len(results) == 0:
        return

    cfg = cfg.validate()
    _apply_style(cfg)

    t = [r.t2 for r in results]
    melt_phase = np.array([r.melt_rate_m_per_year for r in results], dtype=float)
    melt_amp = np.array([r.melt_rate_m_per_year_amp for r in results], dtype=float)
    diff = np.array([r.melt_rate_phase_minus_amp_m_per_year for r in results], dtype=float)
    absdiff = np.array([r.melt_rate_phase_minus_amp_abs_m_per_year for r in results], dtype=float)

    sub0 = np.array([r.melt_rate_subband0_m_per_year for r in results], dtype=float)
    sub1 = np.array([r.melt_rate_subband1_m_per_year for r in results], dtype=float)
    sub2 = np.array([r.melt_rate_subband2_m_per_year for r in results], dtype=float)
    sub_std = np.array([r.melt_rate_subband_std_m_per_year for r in results], dtype=float)

    def _rolling_median_nan(x: np.ndarray, window: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.size
        if window <= 1 or n == 0:
            return x.copy()
        half = window // 2
        out = np.full(n, np.nan, dtype=float)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            out[i] = float(np.nanmedian(x[lo:hi]))
        return out

    def _robust_ylim(y: np.ndarray) -> tuple[float, float] | None:
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return None
        if y.size >= 5:
            p_lo, p_hi = cfg.timeseries_plot_robust_percentiles
            lo = float(np.nanpercentile(y, p_lo))
            hi = float(np.nanpercentile(y, p_hi))
        else:
            lo = float(np.nanmin(y))
            hi = float(np.nanmax(y))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        if hi == lo:
            span = abs(lo) if lo != 0 else 1.0
            lo -= 0.5 * span
            hi += 0.5 * span
        else:
            span = hi - lo
        pad = float(cfg.timeseries_plot_robust_pad_frac) * span
        return lo - pad, hi + pad

    # Optional median filtering for display
    w = int(cfg.timeseries_median_filter_window)
    do_filt = cfg.timeseries_median_filter_enabled and w > 1
    melt_phase_f = _rolling_median_nan(melt_phase, w) if do_filt else None
    melt_amp_f = _rolling_median_nan(melt_amp, w) if do_filt else None
    diff_f = _rolling_median_nan(diff, w) if do_filt else None
    absdiff_f = _rolling_median_nan(absdiff, w) if do_filt else None
    sub0_f = _rolling_median_nan(sub0, w) if do_filt else None
    sub1_f = _rolling_median_nan(sub1, w) if do_filt else None
    sub2_f = _rolling_median_nan(sub2, w) if do_filt else None
    substd_f = _rolling_median_nan(sub_std, w) if do_filt else None

    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # --- (1) Phase vs amplitude melt
    axs[0].scatter(t, melt_phase, label="phase", alpha=cfg.timeseries_plot_raw_alpha)
    axs[0].scatter(t, melt_amp, label="amplitude", alpha=cfg.timeseries_plot_raw_alpha)
    if melt_phase_f is not None:
        axs[0].plot(t, melt_phase_f, label=f"phase median({w})")
    if melt_amp_f is not None:
        axs[0].plot(t, melt_amp_f, label=f"amp median({w})")
    axs[0].set_ylabel("Melt rate (m/yr)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best", fontsize=max(8, cfg.font_size - 2))
    if cfg.timeseries_plot_use_robust_ylim:
        y_for_lim = melt_phase_f if (cfg.timeseries_plot_use_filtered_for_ylim and melt_phase_f is not None) else melt_phase
        lims = _robust_ylim(y_for_lim)
        if lims is not None:
            axs[0].set_ylim(*lims)

    # --- (2) Disagreement
    axs[1].scatter(t, diff, label="phase - amp", alpha=cfg.timeseries_plot_raw_alpha)
    axs[1].scatter(t, absdiff, label="|phase - amp|", alpha=cfg.timeseries_plot_raw_alpha)
    if diff_f is not None:
        axs[1].plot(t, diff_f, label=f"(phase-amp) median({w})")
    if absdiff_f is not None:
        axs[1].plot(t, absdiff_f, label=f"|.| median({w})")
    axs[1].set_ylabel("Disagreement (m/yr)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best", fontsize=max(8, cfg.font_size - 2))
    if cfg.timeseries_plot_use_robust_ylim:
        y_for_lim = absdiff_f if (cfg.timeseries_plot_use_filtered_for_ylim and absdiff_f is not None) else absdiff
        lims = _robust_ylim(y_for_lim)
        if lims is not None:
            axs[1].set_ylim(*lims)

    # --- (3) Sub-band melt rates
    axs[2].scatter(t, sub0, label="subband0", alpha=cfg.timeseries_plot_raw_alpha)
    axs[2].scatter(t, sub1, label="subband1", alpha=cfg.timeseries_plot_raw_alpha)
    axs[2].scatter(t, sub2, label="subband2", alpha=cfg.timeseries_plot_raw_alpha)
    if sub0_f is not None:
        axs[2].plot(t, sub0_f, label=f"sub0 median({w})")
    if sub1_f is not None:
        axs[2].plot(t, sub1_f, label=f"sub1 median({w})")
    if sub2_f is not None:
        axs[2].plot(t, sub2_f, label=f"sub2 median({w})")

    # Also show sub-band spread as a secondary axis
    ax2 = axs[2].twinx()
    ax2.scatter(t, sub_std, label="subband std", alpha=0.3)
    if substd_f is not None:
        ax2.plot(t, substd_f, label=f"std median({w})")
    ax2.set_ylabel("Subband std (m/yr)")

    axs[2].set_ylabel("Subband melt (m/yr)")
    axs[2].grid(True, alpha=0.3)
    # Combined legend
    h1, l1 = axs[2].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axs[2].legend(h1 + h2, l1 + l2, loc="best", fontsize=max(8, cfg.font_size - 2))
    axs[2].set_xlabel("Time")

    fig.suptitle(f"{cfg.station}: melt method diagnostics")
    savefig(fig, outpath, cfg)
