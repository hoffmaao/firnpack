# Summit (Greenland) observation data — provenance & acquisition status

Site: Summit, Greenland (72.6°N, 38.5°W, ~3200 m). Warm (~−30 °C), high
accumulation (~0.23 m ice/yr) — the contrasting regime to South Pole.

## Present

| File | Contents | Source |
|------|----------|--------|
| `summit_layer_accum_annual.csv` | `year, accum_kgm2yr, accum_m_iceeq_yr`, 1743–1939 | Osman et al. 2021 annual layers (NOAA Paleo `summit2021accum.txt`) |
| `summit_lmr_seasonal_point.csv` | `year, season_month, tas_anom_K`, from 800 CE | Last Millennium Reanalysis (Seasonal) — temperature **forcing**, not an observation |

Climate forcing (accumulation + temperature) is essentially in place, built by
`test/summit/scripts/download_climate.py` (LMR pre-1940 + ERA5 post-1940 temp;
Osman 2021 pre-1940 + ERA5 post-1940 accum).

## ⚠ MISSING — the three in-situ constraints that drive the assimilation

Summit currently has **no density, depth–age, or borehole temperature**. These
must be acquired and processed into this directory (matching the South Pole CSV
formats) before a Summit assimilation can run. Targets:

| Need | Candidate source | Notes |
|------|------------------|-------|
| **Density** `depth_m, rho_kgm3` | FirnCover Summit site (MacFerrin et al.); or GISP2/GRIP density; Hawley Summit neutron-probe density | FirnCover DOI 10.18739/A25X25D7M currently staged as **metadata only** — the data bytes need fetching |
| **Depth–age** `depth_m, year_CE` | GISP2 or GRIP layer-counted timescale (NOAA Paleo) | annual-layer-counted; deep, well-dated |
| **Borehole T** `depth_m, T_C` | GISP2 borehole temperature (Cuffey & Clow 1997); GRIP borehole T | classic paleothermometry profile |
| ApRES velocity | likely unavailable at Summit | the framework treats observables as optional — this term is simply dropped |

Acquisition is tracked as a task; it is the main schedule risk for the paper's
third test case. Provenance for each acquired file should be added to this
table with its DOI/URL and access date.
