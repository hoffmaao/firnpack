# McMurdo ApRES strain + basal melt processing (Python)

This is a Python translation/adaptation of the MATLAB workflow you shared
(`Pr_pRES_preprocess.m`, `Pr_pRES_strain.m`, plus functions in `Codes.zip`).

## Run

From the repository root:

```bash
python scripts/mcmurdo_apres_meltrates.py
```

The script assumes this directory layout:

```
<repo>/data/GA04/**/*.DAT  (or .dat)
<repo>/scripts/mcmurdo_apres_meltrates.py
<repo>/figures/
```

## Outputs

- `results/GA04/preprocessed/*.npz` — cached preprocessed profiles
- `results/GA04/pair_results.csv` — one row per consecutive profile pair
- `figures/GA04/preprocess/*.png` — per-profile amplitude/range diagnostics
- `figures/GA04/pairs/*.png` — per-pair displacement/fit/coherence diagnostics
- `figures/GA04/timeseries.png` — strain + melt-rate time series

## Configuration

Edit the constants at the top of:

- `scripts/mcmurdo_apres_meltrates.py` (station name, caching)
- `scripts/apres/config.py` (processing thresholds + window sizes)

Important parameters for McMurdo deployments:

- `max_range_m`
- `bed_search_range_m`
- `firn_depth_m`
- correlation/coherence thresholds (`min_*` values)

