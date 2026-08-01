# South Pole observation data - provenance

Site: South Pole (SPICEcore / SP19), 89.99°S. Curated observation CSVs for the
FirnGrain South Pole assimilation. All small; tracked in git for reproducibility.

| File | Contents | Source |
|------|----------|--------|
| `sp19_density.csv` | `depth_m, rho_kgm3` (ρ in g/cm³ → ×1000) | SPICEcore SP19 density; Stevens et al. 2023 (usap-dc 601680) |
| `sp19_depth_age.csv` | `depth_m, age, age_yrBP, year_CE` | SP19 depth–age scale (Winski et al. 2019 / Kahle et al. 2021) |
| `spicecore_borehole_T.csv` | `depth_m, T_C`, 13–130 m | SPICEcore borehole temperature (upstream) |
| `apres_zeising_processed.csv` | Per-site ApRES range rate dR/dt vs range with phase errors, plus the full-column Menke vertical strain rate (`site, range_m, dh_m, dhe_m, dRdt_myr, dRdt_err_myr, coherence, dt_days, vsr_per_year, vsr_err_per_year`). 7 sites, 12.7–863.9 m | Raw `.DAT` bursts from Hills et al. 2022 (usap-dc 601503), reprocessed through the Zeising ApRES pipeline (a port of the MATLAB reference `func_fit_ice.m`) via `archive/southpole/scripts/southpole_apres_zeising.py` + `apres/`. Sub-burst phase standard errors; Menke weighted least squares with residual-based covariance. Site-to-directory mapping from that dataset's `Acquisition_MetaData.txt` |
| `apres_vertical_velocity_processed.csv` | ApRES vertical velocity vs range. **Superseded**, see caveat 3 | Stevens et al. 2022 (usap-dc 601525); Zeising processing (pipeline A) |
| `apres_site_locations.csv` | `site, latitude, longitude` for the ApRES array | Hills et al. 2022 ApRES acquisition metadata (usap-dc 601503, `raw/Acquisition_MetaData.txt`) |
| `buizert2021_spice_accum.csv` | `age_yrBP, accum, year_CE` | Buizert et al. 2021 SPICE accumulation reconstruction (NOAA Paleo) |
| `buizert2021_spice_temp.csv` | δ¹⁸O-derived cloud temperature | Buizert et al. 2021 (NOAA Paleo) |
| `era5_monthly_point.csv` | `year, month, t2m_K, net_accum_m_iceeq_month` | ERA5 monthly, SP gridpoint (Copernicus CDS) |

## ⚠ Three data caveats baked into the analysis

1. **Borehole-T datum (+5.8 °C error).** `spicecore_borehole_T.csv` reads
   −45.5 °C at depth, but the raw USP50 thermistors (and canonical SP 10-m firn
   T) are ≈ −51 °C - the Buizert cloud→surface offset (+5.8) was applied to this
   series in staging. The file is kept as-is for provenance; the assimilation
   applies `FIRN_T_SHIFT=-5.8` to correct it coherently. Independently confirmed
   against USP50 (601525) thermistors. **Absolute temperatures are on the
   ~−51 °C datum; anomalies/shapes are datum-independent.**

2. **ERA5 accumulation is ~27 % low.** ERA5 net accumulation (P−E) at SP
   (~0.073 m ice/yr) is biased low vs Buizert-era climatology (0.096), the raw
   SP19 layer thickness (~0.093), and stake farms (0.085–0.093) - the known
   ERA5 plateau sublimation/dry bias. The accumulation prior uses the Buizert
   climatology, not ERA5 (`FIRN_B_CLIM`).

3. **Use the Zeising raw-burst ApRES product, not pipeline A.**
   `apres_vertical_velocity_processed.csv` (pipeline A) is 25-m smoothed, which
   biased firn gradients by up to 4× and inverted the site ranking.
   `apres_zeising_processed.csv` reprocesses the raw bursts with proper phase
   errors and no smoothing, and is the default (`FIRN_VEL_SRC=zeising`). It is
   load-bearing in two places: it is the sole source of the **ezz pin** (fit
   below close-off, 250–500 m) and of the **per-site velocity sigma**. The
   pipeline-A file is kept because it is the provenance record for the archived
   r8/r9 MAPs, which the legacy guards (`FIRN_VEL_SRC=authors`) reproduce, but
   do not reach for it for new work.
