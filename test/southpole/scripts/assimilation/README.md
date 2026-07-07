# South Pole firn data assimilation (clean rebuild, 2026-06)

> **➡ Start with `HANDOFF.md`** — the current state-of-the-project entry point
> (all three inversions, the capstone joint result, how-to-run, adjoint rules,
> next steps). This README covers the density-focused `sp_assimilate.py` details.

Goal: derive firn **densification physics** by assimilating real South Pole data
(density, depth-age, borehole temperature, ApRES velocity) into the Evan Cummings
1-D firn column model.

## TL;DR — the forward model was sound all along

A clean constant-forcing steady-state run of the Cummings `(H, ρ, w, age)` model
with Herron–Langway densification reproduces the SP data **without any tuning**:

| quantity | model | SP data |
|----------|-------|---------|
| age @ base (130 m) | 1200 yr | 1197 yr |
| surface settling w | −0.223 m/yr | −0.223 (=ρᵢ·ḃ/ρₛ) |
| ApRES (w−wₛ)/n_ice | matches | binned multi-site |
| density (whole column) | **+35..65 high** | — |

The years of "the South Pole model doesn't work / H&L caps at 646 kg/m³" were **not**
a physics failure.  They were three setup problems, now fixed:

1. **Spinup too short** — 1200 yr ≈ one transit time, so the deep column never
   reached steady state.  Need ≥ ~1500 yr (use ~2500).
2. **Wrong temperature** — forcing used −50 °C (Buizert *cloud* temp).  The firn is
   near-isothermal at **−45.5 °C** (borehole).  −50 °C over-suppresses densification.
3. **pyadjoint tape-replay bug** — silently corrupted the optimizer with time-varying
   forcing.  Fixed by the **tape-rebuild** approach (clear + re-annotate per eval).

The residual density high-bias is the well-posed inversion signal: H&L overshoots,
Arthern/Ligtenberg undershoots — the data sits **between** the two literature laws.

## Files

- `../diagnostics/forward_basics_diagnostic.py` — constant-forcing forward run vs data.
  `python forward_basics_diagnostic.py {hl|arthern} {cg|dg} <spin_years> [dt_years]`
- `sp_assimilate.py` — the assimilation (tape-rebuild adjoint, L-BFGS-B).

## sp_assimilate.py design notes (lessons baked in)

- **Objective**: Gaussian-kernel predictions `pred_i = assemble(field·φ_i·dx)` as
  scalars, χ² accumulated as scalar `AdjFloat` arithmetic.  VOM point-sampling and
  direct `r*r` assembly **both break the adjoint** (order-1 Taylor) — do not use them.
- **Density barrier kink**: `density_form` adds `max_value(ρ−ρᵢ,0)`.  If the IC or a
  transient reaches ρᵢ=917, the kink makes J non-differentiable and the Taylor test
  fails (erratic/negative order).  Cap the **initial** density well below ρᵢ
  (`RHO_IC_CAP=820`); steady-state max is ~880, so the barrier stays inactive.
- **Velocity operator**: ApRES `v(z) = (w(z) − w_surf)/n_ice`, `n_ice=√3.18=1.783`.
- **Step cost**: `dt=5 yr` gives the same steady state as `dt=1 yr` (validated) →
  ~500 on-tape steps for a 2500-yr spinup.

## Run

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python sp_assimilate.py            # rho+age+T
FIRN_W_V=1 PYTHONPATH=src ... python sp_assimilate.py               # + velocity
FIRN_VERIFY_ONLY=1 ...                                              # repro + Taylor only
FIRN_FD_CHECK=1 ...                                                 # FD gradient check
```
