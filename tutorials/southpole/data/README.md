# South Pole observation data — provenance

Site: South Pole (SPICEcore / SP19), 89.99°S. Curated observation CSVs for the
FirnGrain South Pole assimilation. All small; tracked in git for reproducibility.

| File | Contents | Source |
|------|----------|--------|
| `sp19_density.csv` | `depth_m, rho_kgm3` (ρ in g/cm³ → ×1000) | SPICEcore SP19 density; Stevens et al. 2023 (usap-dc 601680) |
| `sp19_depth_age.csv` | `depth_m, age, age_yrBP, year_CE` | SP19 depth–age scale (Winski et al. 2019 / Kahle et al. 2021) |
| `spicecore_borehole_T.csv` | `depth_m, T_C`, 13–130 m | SPICEcore borehole temperature (upstream) |
| `apres_vertical_velocity_processed.csv` | ApRES vertical velocity vs range | Stevens et al. 2022 (usap-dc 601525); Zeising processing |
| `buizert2021_spice_accum.csv` | `age_yrBP, accum, year_CE` | Buizert et al. 2021 SPICE accumulation reconstruction (NOAA Paleo) |
| `buizert2021_spice_temp.csv` | δ¹⁸O-derived cloud temperature | Buizert et al. 2021 (NOAA Paleo) |
| `era5_monthly_point.csv` | `year, month, t2m_K, net_accum_m_iceeq_month` | ERA5 monthly, SP gridpoint (Copernicus CDS) |

## ⚠ Two data caveats baked into the analysis

1. **Borehole-T datum (+5.8 °C error).** `spicecore_borehole_T.csv` reads
   −45.5 °C at depth, but the raw USP50 thermistors (and canonical SP 10-m firn
   T) are ≈ −51 °C — the Buizert cloud→surface offset (+5.8) was applied to this
   series in staging. The file is kept as-is for provenance; the assimilation
   applies `FIRN_T_SHIFT=-5.8` to correct it coherently. Independently confirmed
   against USP50 (601525) thermistors. **Absolute temperatures are on the
   ~−51 °C datum; anomalies/shapes are datum-independent.**

2. **ERA5 accumulation is ~27 % low.** ERA5 net accumulation (P−E) at SP
   (~0.073 m ice/yr) is biased low vs Buizert-era climatology (0.096), the raw
   SP19 layer thickness (~0.093), and stake farms (0.085–0.093) — the known
   ERA5 plateau sublimation/dry bias. The accumulation prior uses the Buizert
   climatology, not ERA5 (`FIRN_B_CLIM`).
