# South Pole Firn Inversion - Working Notes

> **⚠️ 2026-06 UPDATE - much of the history below is SUPERSEDED.**
> The inversion now WORKS end-to-end. **Read `doc/southpole_inversion_handoff.md` first.**
> The old "H&L can't fit / max density 646 / degenerate MAP" conclusions were setup
> bugs (under-spinup, −50 °C forcing, pyadjoint tape-replay), NOT physics. With those
> fixed, a 12-control joint inversion fits density+age+velocity+temperature and
> recovers densification + firn conductivity + a surface-T history. See
> `doc/southpole_inversion_handoff.md`. The notes below are kept for historical
> context only.

## Overview

MAP inversion for firn densification parameters at South Pole using SP19/SPICE core data.
Scripts in `scripts/herron_langway/` (H&L model) and `scripts/{arthern,kingslake}/`.

## Observation Data (`processed/`)

| File | Contents | Notes |
|------|----------|-------|
| `sp19_density.csv` | depth_m, rho_kgm3 | rho in g/cm3, multiply by 1000 |
| `sp19_depth_age.csv` | depth_m, year_CE | age = 2015 - year_CE |
| `spicecore_borehole_T.csv` | depth_m, T_C | borehole temperature |
| `era5_monthly_point.csv` | t2m_K, net_accum_m_iceeq_month | 1940-2015 |
| `buizert2021_spice_temp.csv` | year_CE, temp (C) | 740-1950 CE reconstruction |
| `buizert2021_spice_accum.csv` | year_CE, accum (m ice/yr) | 740-1950 CE |

## Temperature Forcing

> **⚠ 2026-07-05 DATUM CORRECTION.** The claim below that "borehole deep
> temperature is ~−45.5 C" is WRONG - `processed/spicecore_borehole_T.csv`
> carries the +5.8 offset applied to the wrong series at staging. Raw USP50
> thermistors (data/usap_dc/601525, uncorrected) measure 20–40 m firn at
> −50.8..−51.3 C, matching (staged − 5.8) exactly, and the canonical SP 10-m
> firn temperature is ≈ −51 C. The staged csv is kept as-is for historical
> reproducibility; r5b+ inversions apply `FIRN_T_SHIFT=-5.8` coherently
> (obs, prior ramp, knot bounds). All pre-r5b absolute temperatures (and the
> "H&L rates at −45 C" statements) are in the WRONG datum; anomalies/shapes
> are unaffected. See `doc/southpole_inversion_handoff.md`.

**Buizert cloud-to-surface offset: +5.8 C.**
Buizert reconstruction is cloud temperature (~-51 C mean). Borehole deep temperature
is ~-45.5 C. The +5.8 C offset corrects for the difference between precipitating cloud
temperature and surface temperature. ERA5 temperatures are already surface (2m) values
(~-45.9 C mean) and do NOT need the offset.

## Inversion Scripts - Version History

### v11 (working, 2026-04-16)
- H&L, 6 controls, on-tape Buizert+ERA5 spinup
- J: 1238 → 997, 4/6 params data-constrained
- **No Buizert T offset** - used raw Buizert temps
- MAP: k0=11.4, k1=562, Ea1=8598, Ea2=28963 J/mol

### v12 (working)
- Off-tape ALE spinup + on-tape ERA5 (DG density)
- 6 controls including rho_surf, Q_base
- MAP: k0=11.2, k1=558, Ea1=9110, Ea2=30596

### v14 (working, 2026-04-19)
- Simple CG fixed mesh, on-tape spinup, 4 controls (hl_k0/k1/Ea1/Ea2)
- Q_base fixed at 0.05 W/m2
- **No Buizert T offset**
- J: 1232 → 1036
- MAP: k0=5.90, k1=539, Ea1=16405, Ea2=21526

### v19 (Stokes, 2026-04-24)
- Stokes compressible densification (n=1), 9 controls
- **Buizert T offset +5.8 C applied**
- Uses CG1 intermediates for control-dependent forcing (wb_cg1 pattern)
- J: 771 → 746 (modest reduction, trust region collapsed)

### v20 (H&L + corrected T, 2026-04-27)
- H&L densification with +5.8 C Buizert offset
- Multiple attempts - see "v20 Attempts" below

## v20 Attempts - What We Tried

### Attempt 1: Time-varying forcing (Buizert+ERA5), no tape replay fix
- **Result**: J_init=59,871 vs J_replay=37,059 - tape replay bug!
- Optimizer fitted the corrupted replay (constant forcing at last step's T)
- MAP: Ea1=31,573 J/mol (3x initial) - kills densification entirely

### Attempt 2: Constant forcing at Buizert mean (-50 C), sig_T=0.5 C
- Tape replay correct (J_init = J_replay = 112,078)
- J converged to 37,732
- **MAP: Ea1=36,085, Ea2=31,765 - again kills densification (rho stays at 350)**
- Temperature data dominated objective despite icepack 1/N normalization
- Q_base=0.036 W/m2 drove 8 C temperature gradient to fit borehole

### Attempt 3: Same as #2 but with sig_T=5 C (loose temperature)
- Nearly identical result: J=37,720, same extreme Ea values
- Temperature weighting was NOT the issue

### Attempt 4: Time-varying forcing with pre-created R-space Constants + T_offset control
- Forcing values as R-space Functions (created off-tape, never reassigned)
- T_offset control (initial 0, prior sigma=1 C) makes interpolation control-dependent
- **Result**: J_init=59,871 vs J_replay=37,058 - tape replay STILL broken!
- Pre-created Constants don't fix the bug - the solver's internal operations
  also have R-space assigns that get skipped
- MAP: same bad result (Ea1=31,926, no densification)

## Key Findings

### Tape replay drift (not a simple "skip all assigns" bug)
R-space `.assign(float_value)` calls that change per time step are NOT faithfully
replayed by pyadjoint. The error is **not** "all steps use the last assigned
value" - it's a gradual drift that accumulates over steps.

**Quantified behavior** (2026-04-28 test):

| Steps | T range | J mismatch |
|-------|---------|------------|
| 5     | 1 K     | 0.016%     |
| 20    | 5 K     | 0.42%      |
| 50    | 10 K    | 2.1%       |
| 100   | 10 K    | 4.2%       |
| 270   | 10 K    | ~38% (v20) |

The drift occurs for ANY per-step R-space assign (temperature, accumulation,
velocity), not just temperature. Pre-creating R-space Functions off-tape and
using control-dependent CG1 interpolation did NOT fix it.

**Constant forcing replays perfectly** (0.000% mismatch).

**v14 also had the bug**: J_init=1,232 but J_hist[0]=37,797. The optimizer
minimized the corrupted replay J and happened to find parameters that also
looked reasonable when run with actual forcing. The bug was present but
unnoticed because results were plausible.

**Previous synthetic-data tests worked** because they used constant forcing
(steady-state synthetic problems don't vary T or accumulation per step).

**Root cause unclear**: The exact pyadjoint mechanism causing the drift needs
investigation. It's not a simple skip - the behavior suggests partial replay
or incorrect checkpointing of R-space state.

### H&L at -50 C can't fit SP19 density
With constant forcing at the Buizert mean (-50 C), the H&L model CANNOT produce
the observed SP19 density profile regardless of parameter values. The optimizer
finds a degenerate solution that kills densification (extreme Ea values).

This may be because:
1. -50 C is too cold for near-surface densification (the actual surface T is -45.5 C)
2. The H&L Arrhenius sensitivity is too strong - small T changes cause large rate changes
3. The constant-forcing steady state doesn't match the transient observed profile

### Model/framework limitations
The deeper issue is the interaction between:
1. The tape replay bug (forcing frozen at last value)
2. H&L's exponential sensitivity to temperature
3. The mismatch between Buizert cloud T and surface T
4. Fixed-mesh limitations for density advection

These need to be addressed at the model/framework level, not by tuning
uncertainties or forcing conventions.

## Next Steps to Consider

1. **Fix the tape replay bug**: Either fix pyadjoint or use tape-rebuild approach
2. **Off-tape spinup**: Run Buizert spinup OFF tape (establishes IC), put only
   ERA5 on tape. Shorter tape = fewer blocks that can fail.
3. **Use ERA5 mean as constant forcing**: -45.9 C is the actual surface T.
   With Q_base for geothermal, this should produce reasonable profiles.
4. **Revisit v14 with the offset**: v14 worked - understand WHY (was it the
   tape bug accidentally helping?)
5. **Try the firnstokes model**: Full Stokes PDE has different coupling
   (velocity from momentum, not continuity) which may behave differently.

## File Locations

- Inversion scripts: `scripts/herron_langway/southpole_inversion_v{11..20}_*.py`
- Plotting: `scripts/herron_langway/plot_v{14,20}_results.py`
- Results: `results/southpole_map_v{11..20}*.json`
- Firnstokes (standalone): a separate repository, not part of firnpack
