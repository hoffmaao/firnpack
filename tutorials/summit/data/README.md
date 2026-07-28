# Summit (Greenland) observation data — provenance & acquisition status

Site: Summit, Greenland (72.6°N, 38.5°W, ~3200 m). Warm (~−30 °C), high
accumulation (~0.23 m ice/yr) — the contrasting regime to South Pole.

## Present

| File | Contents | Source |
|------|----------|--------|
| `summit_density.csv` | `depth_m, rho_kgm3`, 0.055–89.87 m (411 pts) | **Composite** — FirnCover GISP2 core (0–22 m, via SUMup 2022) + Fourteau et al. GRIP gas-pycnometry (25–90 m, through pore close-off). See provenance below. |
| `gisp2_depth_age.csv` | `depth_m, year_CE, age_yrBP`, 1.5–150 m (518 pts, ~0.27 m) | GISP2 layer-counted timescale, NOAA Paleo `countage-noaa.txt`. Accessed 2026-07-08. |
| `firncover_summit_firn_T.csv` | `depth_m, T_C, T_seasonal_std_C`, 1.1–11.6 m (18) | FirnCover Summit RTD string, 2015–2019 mean; DataONE `doi:10.18739/A25X25D7M` (`FirnCoverData_2_0_2021_07_30.h5`). Deep-mean −28.8 °C. **Shallow** (≤12 m). |
| `summit_layer_accum_annual.csv` | `year, accum_kgm2yr, accum_m_iceeq_yr`, 1743–1939 | Osman et al. 2021 annual layers (NOAA Paleo `summit2021accum.txt`); mean 0.246 m ice/yr |
| `summit_lmr_seasonal_point.csv` | `year, season_month, tas_anom_K`, from 800 CE | Last Millennium Reanalysis (Seasonal) — temperature **forcing**, not an observation |

**Status: the densification observable set is COMPLETE** — density + depth–age
+ accumulation + a shallow near-surface firn-T constraint. Configured as a
densification + accumulation study with temperature as a shallow constraint +
forcing (there is no deep borehole-T and no ApRES velocity at Summit — the
framework treats both as optional and drops them).

| `firncover_summit_compaction.csv` | Per-instrument mean firn compaction rate: 6 instruments, material intervals from the install surface to 4.2–22.0 m, records 1.1–3.4 yr (2015–2019), rates −127…−302 mm/yr | FirnCover compaction coils at Summit (same HDF5 as the firn-T: DataONE `doi:10.18739/A25X25D7M`, `FirnCoverData_2_0_2021_07_30.h5`, `Compaction_Daily` + `Compaction_Instrument_Metadata`; ESSD paper doi:10.5194/essd-14-955-2022). Extracted by `diagnostics/stage_firncover_compaction.py`; re-download via `https://cn.dataone.org/cn/v2/resolve/urn:uuid:1b044917-f7d9-46ac-b2cf-6217e49b4d38`. Columns `instrument_ID, install, ztop_mean_m, zbot_mean_m, record_years, rate_m_yr, sigma_m_yr`. **sigma_m_yr is DATA-DERIVED** (not the daily-fit s.e., which is meaningless — daily residuals seasonal, lag-1 ≈ 1.0): interannual scatter of year-over-year increments at matched day-of-year (14-19 mm/yr, seasonal removed) with a cross-instrument representativeness floor (two coils at ~15.7 m agree to ~16 mm/yr); ~8-12% of rate. Derived by `diagnostics/derive_compaction_sigma.py`; assimilated via the engine's `compaction` obs kind (interval shortening w@top − w@bot). The measured interval is MATERIAL (top = buried install surface; `ztop_mean_m`/`zbot_mean_m` are record means of the tracked endpoint depths) |

## Remaining gap (optional, not blocking)

| Need | Status | Notes |
|------|--------|-------|
| Deep borehole T (`depth_m, T_C`, to ~100 m) | not staged | GISP2-D log (Clow) is archived decimated to 100 m spacing — too coarse for firn. Full-res ~20 cm Clow log needed for a deep thermal inversion; the shallow FirnCover string suffices for the near-surface T constraint. |
| ApRES velocity | unavailable | dropped (optional observable). |

## Provenance — `summit_density.csv` (density, acquired 2026-07-08)

Columns `depth_m, rho_kgm3` (density in **kg/m³**), sorted by depth ascending,
411 points spanning **0.055–89.87 m**, density **285→873 kg/m³** (surface ~331;
crosses the ~550 kg/m³ critical density at ~11.5 m; crosses pore close-off
~830 kg/m³ at **72.65 m** — consistent with Summit's known ~72–80 m close-off).
5 m-binned means rise monotonically (377→469→530→566→585→609→680→719→752→779→
797→818→829→846→855 kg/m³). No single open dataset spans the full Summit firn
column, so this is a **composite of two Summit-dome cores**:

- **0.055–22.12 m** (215 pts, ~0.1 m spacing): FirnCover Summit core at the
  **GISP2** site (72.578 °N, 38.469 °W, 2017; MacFerrin et al. FirnCover,
  method = ice-core section), extracted from the **SUMup 2022 snow-density
  compilation** (`Profile 700`, `Citation 46`). Source: SUMup, Arctic Data
  Center / DataONE, DOI **10.18739/A24Q7QR58**, data object
  `SUMup_density_2022.csv` = `urn:uuid:af7b851c-18ec-45a3-a5c8-8fcb98f62665`
  (resolve via `https://cn.dataone.org/cn/v2/resolve/urn:uuid:...`; density
  already in kg/m³). Accessed 2026-07-08.
- **25.03–89.87 m** (196 pts): **Fourteau et al. (2019)** gas-pycnometry firn
  density at the **GRIP** site (72.58 °N, 37.64 °W). PANGAEA DOI
  **10.1594/PANGAEA.907675** (series 907678), CC-BY-4.0; download
  `https://doi.pangaea.de/10.1594/PANGAEA.907675?format=textfile`. Density
  column is g/cm³ → **multiplied by 1000** to kg/m³ here. Paper: Fourteau et
  al. (2020), *Historical porosity data in polar firn*, Earth Syst. Sci. Data
  12, 1171–1179, doi:10.5194/essd-12-1171-2020. Accessed 2026-07-08.

**Caveats.** (1) Composite of two sites: GISP2 (surface) and GRIP (deep) are
~28 km apart on the Summit dome, same latitude, same accumulation (~0.23 m
ice/yr) and mean-annual T (~−31 °C); the 22→25 m handoff is continuous
(605→600 kg/m³). (2) **Mid-firn 25–62 m is clustered** (samples at ~25, 37, 47,
51, 62 m with gaps up to ~12 m); the surface (≤22 m) and the close-off zone
(64–90 m) are densely sampled (sub-metre). (3) Fourteau pycnometry is on
discrete samples → real layer-scale scatter (±~20–30 kg/m³ within a metre);
bin/smooth if a monotonic profile is required. (4) FirnCover's redundant 7.65–
7.85 m Fourteau cluster was dropped in favour of the denser GISP2 surface core.
Both raw sources are re-downloadable from the DOIs/URLs above (SUMup CSV via the
DataONE `urn:uuid`; Fourteau via the PANGAEA `?format=textfile` URL); the SUMup
Summit subset is `Profile==700` filtered on `Latitude∈[72.5,72.6]`,
`Longitude∈[-38.5,-38.4]`.

### Bonus notes (not staged)
- **Borehole temperature:** GISP2-D borehole T (Clow & Gundestrup) is at NOAA
  Paleo (`.../summit/gisp2/physical/tempertr.txt`) but the archived copy is
  **decimated to 100 m spacing** (first point 100 m = −31.4 °C) — too coarse
  for a firn (0–100 m) temperature profile. The full-resolution (~20 cm) Clow
  log would need to be sourced elsewhere. A fine firn-T *time series* exists in
  the literature (Cambridge, *"Reconstructing thermal properties of firn at
  Summit, Greenland, from a temperature profile time series"*).
- **Accumulation:** GISP2 `.../physical/accum.txt` (Alley layer-based) is an
  additional Summit accumulation series if the Osman 2021 record needs
  extending.
