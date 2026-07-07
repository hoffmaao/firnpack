"""Configuration defaults for the McMurdo ApRES processing workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class ProcessingConfig:
    """Processing configuration.

    This merges the intent of the MATLAB config files:
      - config_pRES_preprocess.m
      - config_pRES_strain.m

    It is not a 1:1 port of every MATLAB switch, but it contains the
    parameters actually used by the Python implementation.
    """

    # --------------------
    # Dataset / paths
    # --------------------
    station: str = "GA04"

    # The driver script resolves these relative to the repository root.
    data_dir: Path | None = None
    figures_dir: Path | None = None
    results_dir: Path | None = None

    # Optional: use ImpDAR's ApRES reader (impdar.lib.ApresData) if installed.
    use_impdar_reader: bool = False

    # Constant for converting day-based intervals to per-year rates
    daysPerYear: float = 365.25

    # --------------------
    # Preprocess settings
    # --------------------
    samples_per_chirp: int = 40000
    pad_factor: int = 8
    max_range_m: float = 1500.0

    # Radar parameters (defaults consistent with common ApRES deployments).
    # These can be overridden from the burst header if present.
    sampling_frequency_hz: float = 40000.0
    f0_hz: float = 2.0e8
    # Chirp gradient in rad/s^2 (K/(2*pi) is in Hz/s). 2e8 Hz/s is common.
    K_rad_s2: float = 2.0e8 * 2.0 * 3.141592653589793
    relative_permittivity_ice: float = 3.18

    # Split-by-attenuator choice (MATLAB `att` argument to fmcw_burst_split_by_att)
    attenuator_index: int = 1

    # Optional frequency cropping (MATLAB func_cull_freq / fmcw_cull_freq2)
    # Set to None to disable.
    frequency_range_hz: Tuple[float, float] | None = None

    # Window function for range processing (MATLAB winFun). We currently
    # implement only 'blackman', but keep this as a config for parity.
    winfun: str = "blackman"

    # Bad chirp culling
    bad_chirps_method: str = "BadChirps"  # 'BadChirps' or 'nChirps'
    delete_first_chirps: int = 1
    max_chirps_correlation: int = 200
    bad_chirps_sigma: float = 3.0

    # Correlation-based noise-depth detection
    correlation_limit: float = 0.65

    # Ice thickness / bed picking (MATLAB func_icethickness automated branches)
    ice_thickness_method: str = "max"  # 'max' or 'use'
    ice_thickness_use_m: float | None = None
    ice_thickness_min_m: float = 50.0

    # Bed picking (used by the Python workflow for diagnostics / melt)
    # (These are largely redundant with the ice_thickness_* fields but kept
    # for backwards compatibility with earlier iterations of the script.)
    bed_search_min_m: float = 50.0
    bed_search_max_m: float | None = None  # None -> use max_range_m

    # Basal echo diagnostics (time-series plots)
    # ------------------------------------------
    # These settings add additional per-acquisition metrics that help
    # diagnose basal-return complexity (e.g., multiple peaks, trailing
    # energy) which can be associated with accretionary/marine ice or
    # reflector switching.
    #
    # Basal-window mean amplitude is computed over [bed, bed+L].
    basal_window_mean_len_m: float = 50.0

    # Multi-peak metrics are computed in a window around the picked bed
    # depth: [bed - above, bed + below].
    basal_peak_window_above_m: float = 10.0
    basal_peak_window_below_m: float = 50.0

    # --------------------
    # Sub-band / frequency dependence diagnostics
    # --------------------
    # Vaňková et al. (2021) show that melt inferred from the complex-phase
    # method can disagree with amplitude-based thickness change when the
    # basal return is affected by intermittently accreted ice / reflector
    # complexity. A useful diagnostic is to re-compute bed displacement using
    # different sub-bands of the FMCW chirp (i.e., different center
    # frequencies) and check whether the inferred melt rate is frequency-
    # dependent.
    #
    # Implementation here: during preprocessing we compute additional
    # phase-sensitive range profiles for a small number of fractional
    # sub-bands, interpolated onto the full-band range axis. During strain/melt
    # solving we compute a bed-shift/melt-rate for each sub-band.
    subbands_enabled: bool = True
    # Fractions of the available chirp bandwidth to use as sub-bands.
    # Example: three equal-width sub-bands.
    subband_fracs: Tuple[Tuple[float, float], ...] = (
        (0.00, 0.33),
        (0.33, 0.67),
        (0.67, 1.00),
    )

    # --------------------
    # Strain / melt settings
    # --------------------
    min_depth_m: float = 20.0

    # Coarse alignment (amplitude xcorr)
    maxlag_bins: Tuple[int, int] = (-100, 450)
    coarse_chunk_width_m: float = 10.0
    coarse_step_m: float = 2.0
    min_ampcor: float = 0.9
    min_ampcor_prom: float = 0.05

    # Fine alignment (complex xcorr)
    fine_chunk_width_m: float = 6.0
    fine_step_m: float = 2.0
    use_coarse_offset: bool = True
    # In the MATLAB code, this is effectively linear (robustfit). We keep
    # it configurable but default to 1 for parity.
    smooth_coarse_poly_order: int = 1

    # Optional: follow phase-minimum path to reduce half-wavelength ambiguity
    # (MATLAB: cfg.doSmartUnwrap)
    # Ole's config defaults this ON, and it improves robustness when integer-bin
    # ambiguity exists in the complex cross-correlation.
    do_smart_unwrap: bool = True

    # Coarse-offset smoothing control (MATLAB: cfg.doPolySmoothCoarseOffset)
    do_poly_smooth_coarse_offset: bool = True

    # How to pick the "bulk" alignment depth when doing smart unwrap
    # (MATLAB: cfg.HandleAF)
    handle_af: str = "calc"  # 'calc' or 'use'
    use_af_depth_m: float | None = None

    # Fit (vertical strain)
    firn_depth_m: float = 100.0
    min_cohere_coarse: float = 0.875
    min_cohere_fine: float = 0.875
    min_points_to_fit: int = 3

    # Fit method (MATLAB: cfg.fitMethod)
    fit_method: str = "menke"  # 'menke' (weighted LS) or 'robust'

    # Optional: shift intercept relative to firn depth (MATLAB: cfg.fitMethod_shift)
    fit_method_shift: bool = False

    # Optional: require phase correlation quality (MATLAB: cfg.minCoherePhase)
    min_cohere_phase: float | None = None

    # Bed shift / melt
    do_melt_estimate: bool = True
    bed_shift_method: str = "xcorr"  # 'xcorr' or 'rangeDiff'
    xcor_bed_win_m: Tuple[float, float] = (-2.0, 1.0)
    range_bed_win_m: Tuple[float, float] = (-1.0, 0.0)
    bed_search_wavelength_margin: float = 1.0  # +/- N wavelengths around coarse bed shift

    # --------------------
    # Plotting
    # --------------------
    # Simple median filtering of the final strain/melt-rate time series.
    # This is a *post-processing* smoother for display and station summaries
    # (it does not change the per-pair diagnostics).
    timeseries_median_filter_enabled: bool = True
    # Window length in samples (must be odd). 5 works well for 30–60 min burst
    # cadences where occasional spurious pairs can occur.
    timeseries_median_filter_window: int = 5

    # Time-series plotting cosmetics
    # ----------------------------
    # For stations with very short profile-to-profile time steps, annualized
    # melt/strain rates can show occasional large outliers that dominate the
    # y-axis scale. These settings keep the plots readable while still showing
    # the underlying raw points.
    timeseries_plot_use_robust_ylim: bool = True
    # Percentiles used to set y-limits (e.g., (2,98) clips extreme outliers).
    timeseries_plot_robust_percentiles: Tuple[float, float] = (2.0, 98.0)
    # Fractional padding added on top of the percentile span.
    timeseries_plot_robust_pad_frac: float = 0.10
    # Prefer the median-filtered series for determining y-limits, if available.
    timeseries_plot_use_filtered_for_ylim: bool = True
    # Plot raw values as scatter points (instead of a connected line).
    timeseries_plot_raw_as_scatter: bool = True
    # Alpha transparency for raw scatter points.
    timeseries_plot_raw_alpha: float = 0.50

    save_figures: bool = True
    show_figures: bool = False  # keep False for command-line runs
    figure_dpi: int = 150

    # Matplotlib formatting
    font_size: int = 12
    line_width: float = 1.5

    # --------------------
    # Internals
    # --------------------
    _validated: bool = field(default=False, init=False, repr=False)

    def validate(self) -> "ProcessingConfig":
        """Validate and finalize config."""

        if self.samples_per_chirp <= 0:
            raise ValueError("samples_per_chirp must be positive")
        if self.pad_factor <= 0:
            raise ValueError("pad_factor must be positive")
        if self.max_range_m <= 0:
            raise ValueError("max_range_m must be positive")
        if not (0.0 < self.correlation_limit <= 1.0):
            raise ValueError("correlation_limit must be in (0,1]")

        if self.frequency_range_hz is not None:
            f0, f1 = self.frequency_range_hz
            if f1 <= f0:
                raise ValueError("frequency_range_hz must be (f_low, f_high) with f_high > f_low")

        if self.bed_search_max_m is None:
            self.bed_search_max_m = float(self.max_range_m)
        if self.bed_search_min_m >= (self.bed_search_max_m or self.max_range_m):
            raise ValueError("bed_search_min_m must be < bed_search_max_m")

        if self.basal_window_mean_len_m <= 0:
            raise ValueError("basal_window_mean_len_m must be positive")
        if self.basal_peak_window_above_m < 0 or self.basal_peak_window_below_m <= 0:
            raise ValueError("basal_peak_window_above_m must be >=0 and basal_peak_window_below_m must be >0")

        if self.subbands_enabled:
            if not self.subband_fracs:
                raise ValueError("subband_fracs must be a non-empty sequence when subbands_enabled=True")
            for (a, b) in self.subband_fracs:
                a = float(a)
                b = float(b)
                if not (0.0 <= a < b <= 1.0):
                    raise ValueError(
                        "Each (start,end) in subband_fracs must satisfy 0.0 <= start < end <= 1.0"
                    )

        if self.bed_shift_method not in {"xcorr", "rangeDiff"}:
            raise ValueError("bed_shift_method must be 'xcorr' or 'rangeDiff'")

        if self.ice_thickness_method not in {"max", "use"}:
            raise ValueError("ice_thickness_method must be 'max' or 'use'")
        if self.ice_thickness_method == "use":
            if self.ice_thickness_use_m is None:
                raise ValueError("ice_thickness_use_m must be set when ice_thickness_method='use'")

        if self.bad_chirps_method not in {"BadChirps", "nChirps"}:
            raise ValueError("bad_chirps_method must be 'BadChirps' or 'nChirps'")

        if self.winfun.lower() != "blackman":
            raise ValueError("Only winfun='blackman' is implemented currently")

        if self.handle_af not in {"calc", "use"}:
            raise ValueError("handle_af must be 'calc' or 'use'")
        if self.handle_af == "use" and self.use_af_depth_m is None:
            raise ValueError("use_af_depth_m must be set when handle_af='use'")

        if self.fit_method not in {"menke", "robust"}:
            raise ValueError("fit_method must be 'menke' or 'robust'")

        if self.timeseries_median_filter_window < 1:
            raise ValueError("timeseries_median_filter_window must be >= 1")
        if self.timeseries_median_filter_window % 2 == 0:
            raise ValueError("timeseries_median_filter_window must be odd (e.g., 5)")

        if self.timeseries_plot_use_robust_ylim:
            p_lo, p_hi = self.timeseries_plot_robust_percentiles
            if not (0.0 <= float(p_lo) < float(p_hi) <= 100.0):
                raise ValueError("timeseries_plot_robust_percentiles must satisfy 0<=p_lo<p_hi<=100")
            if not (0.0 <= float(self.timeseries_plot_robust_pad_frac) <= 2.0):
                raise ValueError("timeseries_plot_robust_pad_frac must be between 0 and 2")
            if not (0.0 <= float(self.timeseries_plot_raw_alpha) <= 1.0):
                raise ValueError("timeseries_plot_raw_alpha must be between 0 and 1")

        self._validated = True
        return self

    @property
    def bed_search_range_m(self) -> Tuple[float, float]:
        return (float(self.bed_search_min_m), float(self.bed_search_max_m or self.max_range_m))
