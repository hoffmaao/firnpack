# Firn aquifer persistence (1D, confined)

Do perennial firn aquifers survive the winter, and what decides it?

Southeast Greenland holds liquid water in firn pore space year-round (Forster
et al. 2014; Koenig et al. 2014; Miège et al. 2016). Kuipers Munneke et al.
(2014) explained it as a race: accumulation buries summer meltwater below the
reach of the winter cold wave faster than that cold wave can refreeze it. This
tutorial tests that directly, first with idealised forcing where one control
changes at a time, then at the real aquifer belt under ERA5.

## Scope

One dimension, no lateral transport, aquifer treated as confined. Water leaves
the column by refreezing or by being carried out of the base with the firn
itself (the advective flux `theta * w`, which is not a parameterisation: it is
the vertical strain rate the column already imposes below the aquifer). There
is no runoff or seepage term, because that would be lateral transport the
column does not resolve. This is a deliberate choice, not an unfinished one.
Lateral drainage sets the *equilibrium water-table depth*; it does not decide
*survival*, and survival is what these runs measure. All three competing rates
- recharge, burial, refreezing - are vertical.

## Physics

`firnpack.models.firn_richards` - mixed-form (Celia et al. 1990) Richards with
van Genuchten/Mualem curves whose porosity and conductivity are fields from the
prognostic density and grain radius, coupled to `firnpack.solvers.firn_solver`
for densification. Refreezing is a local-equilibrium phase change applied as an
implicit sink inside the Richards residual, so the water budget closes to
round-off and no cell can be overdrawn; its latent heat goes to the matrix
enthalpy and its mass to the density equation (Meyer & Hewitt's ice equation),
where it fills pores in place. Melt comes from `firnpack.surface_energy`, a
skin-temperature energy balance. The derivations, the discretisation and the
four failure modes found on the way are in `doc/firn_hydrology.md`.

## Layout

    run.py                 four synthetic contrast cases (firnpack.aquifer.AQUIFER_SITES)
    run_era5.py            the real belt under ERA5, experiments from config.ERA5_EXPERIMENTS
    config.py              column settings + the ERA5 experiment table; no Firedrake
    plot.py                synthetic cases: theta(z,t), storage minima, fate of melt
    plot_fac_saturation.py ERA5 runs: firn air content and saturation S(z,t) vs observations
    plot_history.py        the 1940-on melt record over the belt
    fetch_era5_aoi.py      stages 3-hourly ERA5 over the AOI into data/ (hours, resumable)
    fetch_era5.py          the earlier monthly point extraction (kept: see data/README.md)
    diagnostics/           seb_forcing.py (energy-balance audit + annual table),
                           map_sites.py, map_aoi_optical.py
    data/                  curated forcing CSVs + provenance README          [tracked]
    output/                one JSON per run                                  [ignored]
    figures/               rebuilt from output/ by the plot scripts          [ignored]

## Experiment 1: which control? (synthetic forcing)

Four columns. The base case is SE Greenland; each contrast changes exactly one
control, so an outcome can be attributed:

| case | changed | question |
|---|---|---|
| `se_greenland` | - | does an aquifer form at all? |
| `low_accum` | accumulation 1.5 -> 0.25 m i.e./yr | is burial the control? |
| `cold` | mean T -7 -> -17 C | is cold content the control? |
| `low_melt` | melt 0.80 -> 0.15 m w.e./yr | is recharge the control? |

Melt is prescribed as a raised cosine over a fixed melt season, **not** as a
positive-degree-day share of the temperature. Tying the two together confounds
the experiment: lowering the mean temperature to test cold content would also
collapse the melt season, and below about -11 C the air never reaches freezing,
so the cold case would get no melt and answer a different question.

## Experiment 2: the real belt under ERA5

Forcing is 3-hourly ERA5 (1940 on) averaged over the ice-only cells of the
1200-1800 m band around the Helheim aquifer sites, run through the firnpack
surface energy balance. Why not ERA5's own melt: the monthly product gives
0.03-0.08 m w.e./yr at the core sites where an aquifer needs about 0.5, and
even the 3-hourly `snowmelt` runs at a third of an energy-balance estimate.
Why an area: the aquifer is a belt, and the two published core sites sit in
ERA5 cells whose elevations bracket the true ones from opposite sides. Cells
containing bare rock are excluded by late-summer albedo; with them in, the
melt trend was +102% instead of +23% (both at the 0.72 aged-firn albedo
floor that `data/aquifer_annual_forcing.csv` is built at, which is the
record `plot_history.py` reads). `data/README.md` has the details.

Each experiment cycles one block of years for 80 years so the column
equilibrates to that climate (a column started from a dry spinup carries a
cold reservoir that decades of melt must first overcome). The aged-firn albedo
floor is the largest free knob in the balance and is swept, not calibrated.
Observations to compare against: water table 10-22.5 m, aquifer base 27.7 m,
recharge 9-30 cm/yr (Montgomery et al. 2017; Miller et al. 2017).

## Two things this column does not yet do

* **The surface energy balance is only half coupled.** It supplies the melt
  rate; its skin temperature and conductive flux `Q_C` are computed and
  discarded, and the enthalpy boundary condition still prescribes the clamped
  2 m air temperature. Closing that loop with a Neumann `Q_C` condition is the
  next step, and is deliberately separate because it changes the thermal
  structure that sets how much of the melt refreezes.
* **The surface cannot lower.** The matrix influx is the net surface mass
  balance, snowfall minus melt, taken on a trailing year. If that ever goes
  negative the site is ablating, which a fixed-mesh Eulerian column cannot
  represent; runs report `ablation_steps` so it is visible rather than silent.

## Running

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python tutorials/aquifer/run.py                 # synthetic, all four
PYTHONPATH=src OMP_NUM_THREADS=1 python tutorials/aquifer/run_era5.py            # every ERA5 experiment
PYTHONPATH=src OMP_NUM_THREADS=1 python tutorials/aquifer/run_era5.py recent_a72 # one of them
python tutorials/aquifer/plot.py                          # figures from output/
python tutorials/aquifer/plot_fac_saturation.py
python tutorials/aquifer/plot_history.py
```

Settings come from `config.py` and are env-overridable (`AQ_YEARS`, `AQ_NZ`,
`AQ_ERA5_YEARS`, ...). An 80-year ERA5 experiment takes roughly 45 minutes
on one core (measured: 2760-2790 s for the completed runs); the experiments
are independent and can run in parallel.

## What the runs show

> **The numbers in this section are stale and are being regenerated.** They
> were produced before two physics fixes. First, the surface mass balance
> double-counted melt: the surface velocity BC was driven with gross snowfall
> while the melt derived from it was also injected as water, putting 30-50%
> more mass into the ERA5-forced column than the climate delivers. This is
> the larger effect on the ERA5 table. Second, the interior gravity flux took
> its conductivity from the receiving cell instead of the donor cell, which
> throttles a wetting front descending into dry firn and so biases the split
> between refreezing in the cold-wave zone and recharge reaching depth. The
> synthetic table is affected only by the flux fix, since `AquiferSite` was
> unchanged. Both tables are kept for comparison until the runs are repeated;
> treat every rate, depth and refrozen fraction below as provisional.

**Synthetic contrasts** (12 years after a 50-year dry spinup):

| case | refrozen | perennial | note |
|---|---|---|---|
| `se_greenland` | 47% | yes | fills from the base up, table at 27-33 m by year 12 |
| `low_accum` | 60% | yes | the one case with a floor: old dense firn from the slow spinup at ~52 m, water perched above it at 16-22 m; a superimposed-ice cap forms at the surface |
| `cold` | 77% | yes | forms later (year 6) and stays deep (41-49 m) |
| `low_melt` | 100% | no | refreezes away every winter |

At 0.8 m w.e./yr of melt, neither halving the burial rate six-fold nor
cooling the column by 10 C stops a perennial aquifer in this column; only
removing the melt does. Persistence here is set by recharge exceeding what the
retained water in the cold-wave zone can refreeze, which is a statement about
the retention curve as much as about climate (see Caveats).

**The real belt under ERA5**, 80-year equilibrium to each block of years:

| experiment | melt | refrozen | out of base | net recharge | water table (last 20 yr) | perennial |
|---|---|---|---|---|---|---|
| `recent_a76` | 463 | 78% | 83 | 19 | 42-47 m | yes |
| `recent_a74` | 541 | 77% | 94 | 31 | 36-41 m | yes |
| `recent_a72` | 625 | 76% | 104 | 48 | 30-34 m | yes |
| `recent_a70` | 712 | 74% | 113 | 70 | 24-28 m | yes |
| `recent_a72_deep` | 625 | 76% | 104 | 48 | 30-34 m | yes |
| `midcentury_a72` | 443 | 75% | 94 | 15 | 44-60 m | yes, marginal |

(kg m^-2 yr^-1; 2006-2019 for `recent`, 1940-1959 for `midcentury`.)

* **A perennial aquifer forms under every albedo floor tried**, and under
  the mid-century climate, where it is thin and breathes with the 20-year
  forcing cycle. The refrozen fraction is nearly constant, 74-78%, so net
  recharge scales with melt, and the water table rises with it: from 45 m at
  0.46 m w.e./yr of melt to 26 m at 0.71.
* `recent_a72_deep` repeats `recent_a72` with the conductivity reduced
  ten-fold in the density range where it was measured. Under the downwinded
  gravity flux the two agreed to three digits, but that conclusion is
  withdrawn pending the re-run: a flux throttled by the dry receiving cell is
  insensitive to the donor cell's conductivity by construction, so the
  apparent insensitivity may be the bug rather than the column.
* **The modelled aquifer sits too deep, and that is the densification.**
  Observed: table 10-22.5 m, base 27.7 m. Modelled: table 24-47 m, base at
  the bottom of the domain, because bubble close-off is reached at 50-52 m
  rather than ~28 m. Wet firn compacts faster than the dry densification law
  used here allows (and the refreezing that is now in the density equation
  is not enough on its own), so there is no impermeable floor at the observed
  depth and the water fills down to 60 m. The recharge that reaches depth,
  19-70 kg m^-2 yr^-1, is at or below the observed 90-300; with the observed
  base the same water would stand in a thinner, shallower layer.
* `recent_a72` (0.62 m w.e./yr of melt) is the reference: inside the RACMO
  melt range for these sites and the middle of the sweep.

## Caveats

* The van Genuchten parameters are numerical defaults, not fitted to firn
  water-retention measurements. `n = 2` in particular leaves a long retention
  tail, and the water that tail holds in the top ~10 m at the end of summer is
  what the winter cold wave refreezes - so the refreezing fraction, and with it
  the recharge, rests on the retention curve more than on anything else.
* Each run reports `budget_residual_kg_m2`: melt in, minus refrozen, drained
  out of the base, and the change in stored and elastically stored water. It
  is of order 1e-4 of the melt total and is a check, not an assumption.
* Permeability is deliberately left non-zero below pore close-off (see
  `FirnRichardsModel.pore_connectivity`): close-off is gradual and wet firn
  keeps some deep mobility.
* With `theta_r = 0` the retention curve cannot dry below a trace at `h_min`,
  so every column carries a few tens of kg/m2 of water it never received.
  Persistence is therefore measured *above that trace*, not in absolute terms.
* The ERA5 record still lacks 1966 and 1986; the plots draw gaps as gaps.
