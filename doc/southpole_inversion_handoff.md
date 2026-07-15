# South Pole firn assimilation — session handoff / state of the project

**Status (2026-07-03): WORKING end-to-end; 46-ctrl MAP (physics+T(t)+b(t)+s2);
per-point UQ round in flight.** Time-dependent data assimilation on real South
Pole data derives the firn model physics (densification + thermal) from density
+ age + ApRES velocity + borehole temperature. Adjoint FD-validated across all
controls. Densification params unchanged through FOUR structural revisions
(ezz, b(t), s2, knot densification) — that's the robust product. The
ATTRIBUTION headline is NOT robust: h'(2015) swung −28 → −21 → −3.7 cm as the
T-history reshaped under denser knots (exactly the prior-bound soft modes the
Laplace UQ flagged; honest error bars pending the per-point round). Modern-era
rate additionally hostage to the Buizert↔ERA5 accum datum seam (see 2026-07-03).

## Environment (essential for resuming)
```bash
# venv + editable firn package on PYTHONPATH; ALWAYS set OMP_NUM_THREADS=1
PYTHONPATH=src OMP_NUM_THREADS=1 /home/andrew/venv-firedrake-2026/bin/python <script>
```
Firedrake's PETSc eats `python -c`; run script files. Runs are slow (~1 min/eval
at 2500-yr spinup); launch with `nohup ... &` and poll the log. Repo is NOT git.

## The three inversions (in `test/southpole/scripts/assimilation/`)
| script | controls | data | result |
|--------|----------|------|--------|
| `sp_assimilate.py` | H&L k0,k1,Ea1,Ea2 + Q_base | rho+age+T (+vel via `FIRN_W_V=1`) | 4-data MAP, J 1.7→0.69 |
| `sp_paleo_assimilate.py` | conductivity k_factor + 6 surface-T knots (densif fixed) | borehole T | thermal-only, T RMS 0.30→0.10C |
| **`sp_joint_assimilate.py`** | **all 12: densif + k_factor + 6 T-knots + Q_base** | **rho+age+vel+T** | **CAPSTONE, J=0.82** |

Plot scripts: `finalize_sp_assim.py <json>`, `plot_paleo_result.py`, `plot_joint_result.py`.
Diagnostics: `../diagnostics/forward_basics_diagnostic.py` (forward vs data),
`forward_paleo_test.py` (conductivity sensitivity).

Env flags (all scripts): `FIRN_VERIFY_ONLY=1` (repro+Taylor+FD, no optimize),
`FIRN_FD_CHECK=1` (finite-diff gradient check), `FIRN_MAX_ITER`, `FIRN_SPIN_YEARS`,
`FIRN_W_RHO/W_AGE/W_T/W_V` (data weights).

## Capstone result (`results/sp_joint.json`, plot `sp_joint_result.png`)
Fits ALL data simultaneously: density 0.66σ, age 0.71σ, velocity 0.63σ, T 0.45σ (~0.09C).
- Densification ≈ literature H&L: k0=10.8, k1=569, Ea1=10466, Ea2=21877 (robust).
- Firn conductivity k_factor=0.42 (~1.5-2× below literature — see caveat).
- Q_base=-0.0025. Surface-T history: warm peak -44.6C @1900CE, cold before, cooling since.

## What was actually wrong before (the diagnosis)
The forward model was SOUND. Prior failures were setup, not physics:
1. Spinup too short (1200yr ≈ 1 transit time; need ≥1500, use 2500).
2. Wrong T: −50C Buizert *cloud* temp; firn is ~−45.5C (borehole). bulk ~−45.05C.
3. pyadjoint tape-replay bug corrupting time-varying forcing.
With these fixed, H&L overshoots density +35..65, Arthern undershoots — data is BETWEEN
the two literature laws (well-posed). dt=5yr == dt=1yr steady state (validated).

## Hard-won adjoint rules (also in memory)
- **Objective**: Gaussian-kernel SCALAR misfits `pred=assemble(field*phi*dx)`, χ² as
  scalar arithmetic. NOT VertexOnlyMesh, NOT direct `r*r` assembly (both → order-1).
- **Density barrier kink**: `density_form` has `max_value(rho-rho_i,0)`; if IC/transient
  hits rho_i=917 the kink breaks the adjoint. Gentle IC (cap 820) keeps it inactive.
- **Control-dependent surface T** → PENALTY enthalpy BC, never Dirichlet:
  `F_H += Constant(beta)*(H_trial - c_i*(Ts_eff - T_ref))*psi*ds(SID)`, beta~5e2.
- **Time-varying control forcing**: build each step's expr with FRESH literal Constants
  on the bracketing knots (`Constant(1-f)*Tk[j] + Constant(f)*Tk[j+1]`). NEVER reassign
  shared Constants (breaks adjoint — past knots get 0 gradient). See
  memory `pyadjoint_timevarying_control_forcing.md`.
- **Tape-rebuild per eval** (clear+re-annotate); trust the **FD check** over Taylor
  (Taylor is unreliable here due to fixed-quadrature of the tanh switch).

## Reanalysis product (`../reanalysis/`, 2026-07)
`sp_reanalysis.py` — forward-only (NO adjoint, ~7 s/scenario!) runs of the MAP
model under CTRL (constant T = recovered 1000 CE value = spinup climate) vs FULL
(recovered T history). Attribution is exact within the 130 m column:
`dh'/dt = |w_bot|_FULL − |w_bot|_CTRL` (surface-following frame ⇒ datum-relative
surface rate = V_deep + |w_bot|; scenario difference isolates the firn response).
Cross-check ΔFAC agrees within ~12% (air-export leak at ρ_bot≈846). Sanity: pre-1000
anomaly ≡ 0, CTRL drift 1.5 cm/kyr.
**Result** (`results/sp_reanalysis.{json,npz,png}`): h'(2015) = −27.6 cm (ΔFAC −24.2),
mean dh'/dt 1957–2015 = −0.47 mm/yr, −0.33 mm/yr at 2015 — the firn is STILL
thinning from the 1300→1900 warming (thermal+advective memory) despite post-1900
surface cooling. Density anomaly Hovmöller shows the warm-period +Δρ advecting down.
**Numerics-robust**: dt=1 (−27.7/−0.46) and NZ=200 (−27.6/−0.47) match dt=5/NZ=100
to <1% (`sp_reanalysis_{dt1,nz200}.json`; script has FIRN_TAG/FIRN_NZ/FIRN_DT_YEARS).

## Structural audit (2026-07-01, "is it the physics/implementation?")
Ruled OUT with evidence: dt/mesh/splitting (h' invariant, above); adjoint (FD
1.000–1.003); penalty enthalpy BC (surface T exact to <1e-4 °C at beta=5e2);
ApRES operator formula (matches corrected (w_s−w)/n_ice, `pyadjoint_apres_dRdt`);
seasonal contamination of borehole-T (only 1 assimilated point <15 m).
**Found — missing ice-dynamic vertical strain** in the compaction-only velocity
closure: deep (95–128 m) ApRES strain −1.5e-4/yr vs model apparent −1.1e-4/yr
(~45% excess); deep v_smooth 6.61 cm/yr vs model 7.23 — a coherent ~0.6 cm/yr
(~0.6σ) residual matching dynamic thinning ėzz ≈ −5e-5/yr integrating to ~6 mm/yr
at 120 m. Currently aliased into k1/Ea2.
**FIX IMPLEMENTED 2026-07-02** (`sp_joint_assimilate_ezz.py`): the term already
existed dormant in `FirnModel.velocity_delta(horizontal_divergence=...)` (UFL
pass-through, tape-safe; same convention as firnvertical: div_h = −ε_zz, 1/s).
13th control `ezz_yr` (prior N(0,1e-4)/yr, bounds [−3e-4,2e-4]), velocity solve
only — density/age/enthalpy untouched (dynamic strain thins WITHOUT densifying;
that signature separates it from compaction). Verified: ezz=0 warm start replays
sp_joint MAP exactly (same rms; ΔJ = 12→13 prior renorm); **adjoint/FD ratio for
ezz_yr = +1.0000** (h=2e-6); dJ/dezz(0) = +4.4e3 → data pull ezz NEGATIVE
(thinning), as the audit predicted.
**RESULT (`results/sp_joint_ezz.json`): J 0.814→0.5794 (−29%), ezz=−1.05e-4/yr.**
Age RMS 0.71→0.31σ, density 0.66→0.55, T/velocity unchanged. Densification +
T-history ROBUST (k0 −0.8%, k1 −0.3%, Ea1 +1.6%, Ea2 +1.0%, knots <0.02 °C,
warming amplitude 0.74 °C unchanged) — the strain fixed the deep age/velocity
structure without repartitioning physics or climate. CAVEATS: |ezz| is 2–3×
the kinematic ballpark (ḃ/H≈3e-5/yr) → possibly absorbing bdot error (ezz↔bdot
degeneracy: velocity+age constrain ~the sum); check published SP GPS/coffee-can
submergence. Optimizer note: warm starts need PRIOR-SIGMA scaling, not 1/|g|
(flat gradients → scale blowup → first step slams bounds; fixed in the script).
NOTE for reanalysis: once the ezz MAP exists, pass the same div_h term into
`../reanalysis/sp_reanalysis.py`'s velocity solve before rerunning attribution
(constant ezz mostly cancels in CTRL−FULL differencing, but keep w consistent).
Remaining physics suspects (untested): k(ρ)=k_ice(ρ/ρ_i)² SHAPE (k_factor=0.42
low = red flag; Calonne 2019 swap is the test); H&L stage-1 adequacy at
low-accum SP (T-gradient metamorphism; weak-constraint η localizes it); n(ρ)
firn refraction in ApRES range registration (audit); seasonal Arrhenius
rectification absorbed in prefactors (+3% stage-1, +15% stage-2 — matters for
transferring the law to other climates, not for the SP fit).

**UQ (Laplace) — DONE 2026-07-01, verdict important**: `sp_uq_hessian.py`
(assimilation/, 25 taped gradient evals → 12×12 FD-of-gradient Hessian at MAP,
asym 6e-4, all eigs>0 → `results/sp_uq_hessian.json`), `../reanalysis/
sp_uq_sensitivity.py` (forward dq/dx), `sp_uq_combine.py` + `sp_uq_breakdown.py`
(numpy → `results/sp_uq_result.json`). **Findings:**
- Under the J actually minimized (per-dataset χ²/N_d ⇒ σ_d inflated √N_d, priors
  /N_CTRL ⇒ inflated √12): h'(2015) = −28 ± 217 cm — NOT significant; warming
  amplitude Tk4−Tk0 = +0.74 ± 2.83 °C.
- **89% of var(h') = three eigenmodes with λ ≈ 0.232 = EXACTLY the inflated
  T-prior curvature (1/(12·0.6²))** — i.e. differential shapes of the pre-1750
  knots (Tk0–Tk3) carry ~zero data information; deep-time T-history shape is
  pure prior. Physically correct: ≤130 m borehole T + density in 2015 have
  almost no memory of 1000–1550 CE.
- Conductivity soft mode (λ=0.069, k_factor ×/35) adds 8% — the known degeneracy.
- What IS constrained: H&L stage-1/stage-2 RATES at −45.2 °C to ×/1.50 each
  (the (k,Ea) decomposition is degenerate at a near-isothermal site, the rate is
  not); Ea2 ×/1.37; Q_base ±0.011.
- Error-model caveat is LOAD-BEARING: per-point independent errors would shrink
  σ by ~√N_d (~×6) → amplitude ±~0.5 °C, h' ±~35 cm — still marginal. Truth
  (correlated R, N_eff≈5–15/dataset) is in between.

## Mid-record misfit diagnosis (2026-07-02, `../diagnostics/sp_midrecord_diagnostic.py`)
Remaining sp_joint_ezz residuals organized by model-ρ (law view) vs deposition
year via model age(z) (forcing view) → `results/sp_midrecord_diagnostic.png`:
- **FORCING signal dominates the mid-record**: 25-yr-binned density residuals
  correlate with the INDEPENDENT Buizert-2021 accum anomaly at **r=+0.69**
  (Buizert 0.024–0.100 vs our constant 0.085; post-1800 is +10–17%). Age
  residuals sweep coherently (−1σ @ ~1500 CE → +0.5σ @ ~1900) — the signature
  of integrating layer thickness with wrong bdot(t). T-knots cannot mimic this.
- Law view: residual median ~0 for ρ<550, ±0.4σ wiggles 550–750, then
  **+1–1.5σ for ρ>800** — but deep ≈ pre-1200 deposition (low Buizert accum),
  so law-shape vs old-forcing is CONFOUNDED there; only a bdot(t) inversion
  separates them.
- NOTE: buizert2021_spice_accum.csv extends to ~40 kyr BP (not just 740–1950
  as older notes said).
**⇒ IMPLEMENTED 2026-07-02: `sp_joint_assimilate_bdot.py`** (27 controls =
ezz-13 + 14 log-b knots 1000–2015; priors: Buizert ≤1950 / ERA5 >1950,
σ_log=0.15 env FIRN_SIG_LOGB). Machinery (ALL verified, ratios +1.0000 for
ezz_yr/Tk4/b1910/b2015; replay deterministic):
- bdot per-step via CG1 `b_eff.interpolate(exp(fresh-Constant log-knot bracket))`
  (geometric interpolation) — Ts_eff pattern.
- w surface BC = PENALTY `beta_w*(wt − ws_expr)*psi*ds(SID)`, beta_w=1e7 →
  |w_s−ws| rel err 1.3e-10 (control-dep Dirichlet forbidden).
- ApRES antenna reference = MODEL surface velocity `assemble(w_f*ds(SID))`
  on tape (with b(t), w_s(2015) ≠ const ws0 — old scripts' ws0 was only valid
  for constant bdot).
- Note: physics @ ezz-MAP + b @ Buizert priors WITHOUT re-opt: J=0.688
  (age degrades 0.31→0.71σ, velocity improves 0.63→0.54) — joint re-equilibration
  needed, as expected.
**RESULT (`results/sp_joint_bdot.json`): J 0.688→0.4149** (vs 0.579 ezz-only,
0.814 original). Fits: density 0.51σ, **age 0.13σ**, T 0.40σ, velocity 0.57σ.
- **ezz = −4.7e-5/yr — relaxed INTO the kinematic range** (was −1.05e-4 with
  const bdot; the ezz↔bdot alias resolved itself). Soft direction though:
  traversed [−1.05e-4…−4.6e-5] at ΔJ≈0.008 → UQ needed for its error bar.
- **b(t) tracks Buizert within ~2.5%** (prior σ 15%): data CONFIRM the
  reconstruction shape; mid-1700s–1950 pulled slightly high (+2%), b(1950)=0.102.
- Densification UNCHANGED a third time (k0=10.76,k1=568,Ea1=10511,Ea2=22163).
- k_factor drifted 0.412→0.356 (the known soft conductivity mode).
- T-history mildly reshaped: amplitude Tk4−Tk0=0.67 °C, post-1900 cooling −0.54.
- Stopped at maxiter 80 (J flat since ~eval 65 — converged in practice).

**Diagnostic rerun (step-2 gate)**: forcing-view corr ρ-res↔Buizert +0.69→+0.31,
age-residual sweep GONE. **Law-shape signal SURVIVES: +1–1.5σ obs-denser for
ρ>780** → H&L stage-2 (ρ_i−ρ) driving term dies too fast approaching close-off.
⇒ step 2 = free a stage-2 SHAPE exponent (rate2 ∝ (ρ_i−ρ)·[(ρ_i−ρ)/(ρ_i−ρ_m)]^(s−1),
s=1 ≡ H&L, s<1 sustains deep densification).

**Decomposition (`results/sp_reanalysis_decomp.{json,png}`), rel. 1000 CE baseline:**
- TONLY: h'(2015)=−21 cm (air −18) — warming-driven firn thinning, −0.26 mm/yr
  in 1957–2015 (previous T-only product −27 cm at the older T history).
- BONLY: +6.2 m, but **+4.84 m is ice-equivalent MASS** (cumulative ∫Δb dt vs
  the b(1000)=0.0768 baseline — interpretation depends on which rate the ice
  dynamics actually balance; our 1D column can't arbitrate that) + air +0.69 m.
- FULL: air (ΔFAC) = **+0.49 m** — the altimetry-relevant firn-air change since
  1000 CE (accum-driven +0.69 minus warming-driven −0.18, near-additive: 1.5 cm
  nonlinearity in 6 m). Instrument era 1957–2015: total +2.3 mm/yr (mass-dom.),
  T-part −0.26 mm/yr. Closure gap (basal export) 0.67 m of 6 m (11%).

## Step 2+3 combined run (2026-07-02 evening, IN FLIGHT)
`sp_joint_assimilate_shape.py` — **46 controls** = 8 scalars (+ NEW s2_shape,
stage-2 exponent, prior logN(1,0.3), pivot at ρ_m so no k1 trade; library edit
in `firn/physics/densification.py` guarded so s2=1 leaves other scripts'
kernels unchanged; field `hl_stage2_shape` added to FirnParameters) + **15
T-knots** (decadal in instrumental era) + **23 b-knots** (~6–10 yr modern),
per user request to densify both. Priors = same curves sampled denser; warm
start = bdot MAP interpolated onto new knots (replay reproduces its data rms
exactly; J offset is pure prior re-evaluation). VERIFIED: FD ratios +1.0000
(s2_shape, Tk12, b1875, b2008); dJ/ds2(1) > 0 ⇒ data want s2 < 1 (sustained
deep densification), as the ρ>780 residual predicted.
**RESULT (landed 2026-07-03 03:08, `results/sp_joint_shape.json`): J 0.4756
(warm-start prior re-eval) → 0.4048.** s2_shape = **0.880** (data pulled the
stage-2 exponent below 1, as the ρ>780 residual predicted). Densification core
UNCHANGED a 4th time (k0=10.69, k1=566, Ea1=10502, Ea2=22258; prior z ≤0.03!).
ezz=−5.96e-5/yr (near-kinematic). k_factor 0.356→**0.157** (soft mode still
sliding; watch the 0.05 bound — Calonne swap is the real fix). T-history
reshaped by the denser knots: warm peak moved 1900→**~1750** (−44.73), T(1000)
warmed to −45.12, modern warming +0.26 °C from 1965→1995 (Clem-consistent
direction). Fit stats (via NEW `../diagnostics/sp_fit_stats.py`, replay
validates J to 4 decimals): density 0.49σ, age 0.16σ, T 0.34σ, v 0.56σ
(vs bdot: 0.51/0.13/0.40/0.57). Per-point J at this MAP = 15.64 (13.11 data
+ 2.53 prior).

**Step-3 gate (2026-07-03 diagnostic rerun): PASSED — the ρ>780 obs-denser
signal is GONE** (+1–1.5σ → ≤+0.3σ; `sp_midrecord_diagnostic_shape.png`).
Remaining structure: known near-surface +1σ (ρ<500, top 15 m), small modern
wiggles; ρ-res↔Buizert corr +0.35. **NEW finding — Buizert↔ERA5 datum seam:**
b-priors step −33% across 1950 (Buizert 0.0998 → ERA5-era 0.067–0.078) and the
MAP follows (b 0.104@1950 → 0.069@1962). Age residuals push back (coherent
−0.5σ dip at deposition ~1920–1950) but σ_age can't overrule σ_logB=0.15.
This seam now CONTROLS the modern attribution rate (below). Round-5 fix
options: widen post-1950 σ_logB (let data arbitrate), or one shared log-offset
control on the ERA5-era prior centers. Stake-farm climatology (~0.085–0.092)
sits BETWEEN the two datums. (Quantity nuance: Buizert accum is layer-derived
ice-eq, ERA5 is P−E — a constant multiplicative offset control is still a
defensible transfer.)
**TEMPERATURE ANALOG — DO NOT SPLICE (user, 2026-07-04):** Buizert T is
δ18O-derived CONDENSATION/CLOUD temperature; ERA5 is 2 m SURFACE temperature —
different physical quantities. The project's +5.8 °C maps the LEVEL only; the
variability transfer is state-dependent (inversion strength covaries with
climate; precip-intermittency weighting), so δ18O anomalies are damped/
distorted vs surface-T anomalies. Never build a T prior by splicing
Buizert(+5.8)≤1950 with ERA5>1950 — that would inject a T seam worse than the
b one. Current hand-ramp T prior is already insulated (Buizert-T is NOT in the
assimilation loop). If a reconstruction-informed T prior is ever wanted:
anomaly-space with free scale (damping) + offset nuisance controls. The
datum-clean T constraints remain: borehole T (absolute level — note ERA5 t2m
level sits ~0.5–1 °C below our borehole-anchored curve) and, for shape,
deep SPICEcore T. Panel (b) of `sp_pp_uq_verdict.png` now overlays both
series as labeled REFERENCES (not priors): ERA5's decadal shape independently
matches the recovered modern warming (+0.5 °C ERA5 vs +0.27±0.82 recovered).

**Decomposition @ shape MAP (2026-07-03; bdot version backed up as
`sp_reanalysis_decomp_bdot.{json,png}`): the attribution REVISED HARD.**
TONLY h'(2015) −21.2 → **−3.7 cm** (warm peak at 1750 ⇒ the warming-thinning
transient mostly equilibrated by 2015; rate −0.08 mm/yr). FULL air ΔFAC +0.49
→ +0.65 m. Modern rate 1957–2015 FLIPPED +2.3 → −3.5 mm/yr — that is the
datum seam (modern b < b(1000)=0.0771 baseline), NOT physics. Additivity
0.11 cm, closure 71 cm/6.4 m (11%, as before). Lesson: the attribution rides
the T-history/b-datum soft modes; only the per-point UQ + seam fix + deep-T
make it publishable.

## Per-point round VERDICT — 2026-07-04 00:00, COMPLETE
Chain done (opt 4 h → Hessian 6.3 h → sensitivity 19 min → combine). Quality:
asym 5.9e-7, |g·σ|=0.68 ≪ min eig 0.645 (σ-coords; NO clipped modes — prior
regularizes all 46), J replay exact. One caught solver-fail at opt eval 2
(wild first probe; recovered). → `results/sp_uq_result_pp.json`.
**WHAT IS CONSTRAINED (the publishable physics):** Ea1 ×/1.18, Ea2 ×/1.10
(was ×/1.37 inflated); stage rates via (k,Ea) still trade (k0 ×/2.5, k1 ×/2.9
individually); **s2 = 0.81 ×/1.27 — 68% CI [0.64, 1.03], i.e. −1.3σ from
H&L s2=1** (marginally suggestive; the ρ>780 residual-pattern collapse is the
stronger evidence); k_factor 0.145 ×/1.79 (~2σ below 0.5 — Calonne suspicion
stands); ezz ±1e-4 = PRIOR (b↔ezz alias; GPS submergence is the fix).
**WHAT IS NOT (even per-point):** T-knots marginals ±0.53–0.59 ≈ prior 0.6 °C,
and CONTRASTS too: warm-peak amplitude T(1750)−T(1150) = +0.65±0.82 (0.8σ),
modern warming T(2015)−T(1965) = +0.27±0.83 (0.3σ). ≤130 m data have no
T-memory — deep SPICEcore T (open step 1) is the ONLY fix. b-knots ×/1.09–1.16
vs prior ×/1.162 (data add little). **Attribution: rate 1957–2015 = −2.2±12.3
mm/yr; ΔFAC(2015) = +0.66±0.94 m; h'=+6.3±11.3 m — all consistent with 0;
var(h') is 98% b-knot PRIOR modes** (top-5 modes = 75%, all b-dominated; the
∫Δb mass integral inherits ±13%/knot). FIGURES: `plot_pp_uq.py` →
`results/sp_pp_uq_verdict.png` (4-panel: scalars vs prior, T/b histories with
bands, attribution significance + var split) and `sp_pp_law_envelope.png`
(dρ/dt(ρ) posterior envelope from joint draws vs literature H&L; s2 effect
+31% at ρ=830).
**Seam is now a 2σ posterior artifact: b(1950)/b(1962) = 1.52× (+2.0σ)** — the
posterior asserts a 52% accum drop in 12 yr; stake farms say no. Round-5 fixes,
in leverage order: (1) deep borehole T; (2) seam fix + assimilate HIGH-RES
d(age)/dz (SP19 is annual — we subsample 42 pts; obs-error says N_eff≈N, so
full-res age-gradient could pin b(t) ≪ ±15% and kill both b-prior dominance
and the seam); (3) Calonne k(ρ). Contrast tool: `sp_uq_result_pp.json` Sigma +
x_star → any linear contrast σ via S[i,i]+S[j,j]−2S[i,j].

## Per-point (honest error) round — 2026-07-03 design notes
**pp MAP LANDED 17:18 (`results/sp_joint_pp.json`): J 15.6444→15.2736**, 59
evals (maxiter stop, flat since ~55; |g|~0.3–0.7 scaled — fine for Laplace).
Fit ρ 0.47σ, age 0.17σ, T 0.33σ, v 0.59σ. Drift from shape MAP as predicted
(small): k0 +2.9%/k1 −3.0% (the k0↔k1 trade; RATE unchanged a 5th time),
Ea's <0.3%, **s2 0.880→0.812** (full per-point density weight strengthens the
deep-law signal), **ezz −5.96e-5→−4.04e-5/yr** (further into kinematic),
k_factor 0.157→0.145 (soft slide, did NOT hit 0.05 bound), T-knots ≤0.06 °C,
b-knots ≤1.9% (1950 seam persists by construction). Evals ~240 s ⇒ Hessian
step ≈6 h (landing ~23:30), then sensitivity (~40 min) + combine.
Per the obs-error analysis (N_eff≈N), the per-dataset 1/N_d and prior 1/N_CTRL
normalizations are DROPPED: J = Σ_d 0.5Σr² + 0.5Σz². Data:prior balance shifts
only ~N_CTRL/N_d ≈ 1.1 (×3 for velocity, N_v=15) ⇒ MAP drifts little; the point
is the ~×44 J rescale ⇒ Laplace σ shrink ~√44 ≈ ×6.6 (kills the "inflated
T-prior curvature" artifact that was 89% of var(h')).
- `sp_joint_assimilate_pp.py` — 46 ctrls, warm start = shape MAP verbatim.
  VERIFIED: J replay 15.64435 (matches sp_fit_stats), deterministic; FD
  ratios +1.0000 at per-point J (s2_shape, Tk12, b1980).
- `sp_uq_hessian_pp.py` — 46×46 FD-of-adjoint-gradient at the pp MAP,
  PRIOR-SIGMA-scaled probe steps (flat 1e-2 would be ~100σ for ezz_yr).
- `../reanalysis/sp_uq_sensitivity_pp.py` — dq/dx of [h'(2015), rate 57–15,
  ΔFAC] with b(t)+ezz+s2 forward, CTRL/FULL pairs, σ-scaled steps.
- `../reanalysis/sp_uq_combine_pp.py` — Σ=H⁻¹ (σ-coord conditioning), marginals
  (b-knots log-space), headline ±1σ, var(h') eigenmode breakdown.
- Orchestrated by `run_pp_chain.sh` → logs in `test/southpole/logs/` (repo-side
  now — the old scratchpad logs died with the session). ~6.5 h total: opt
  ~2.5 h (MAX_ITER 50) → Hessian ~3.6 h → sensitivity ~25 min → combine.
- Watch: k_factor vs its 0.05 bound (bound-active ⇒ that marginal is bogus);
  |g*σ| at MAP in pp_hessian.log (Laplace validity).
- NOTE: `../reanalysis/sp_reanalysis.py` (Hovmöller/npz product) is still
  CONSTANT-b — upgrade to b(t) before refreshing it at the shape/pp MAP.
  `sp_reanalysis_decomp.py` is the b(t)-aware attribution tool.

## ROUND 5 build (2026-07-04, user-approved: deep-T + seam/d(age)dz + Calonne)
`sp_joint_assimilate_r5.py` — per-point J, 48 ctrls (10 scalars + 15 T + 23 b),
warm start from pp MAP. Four structural changes:
1. **Calonne-2019 conductivity** (`firn.py`: `conductivity_law="calonne2019"`,
   guarded, legacy default untouched; smoke-tested: k(350)=0.36, k(550)=0.99,
   k(910)=2.65 W/m/K at −45 °C). TWO scales: k_snow_scale (inherits pp
   k_factor=0.145 at warm start, loose prior) + k_firn_scale (prior logN(1,
   0.15) TIGHT — if data drag it down against that prior, the low-k anomaly
   is real and deep, not a shape artifact. Verify constants vs Calonne 2019
   before publication (Yen T-scaling; air-term neglected).
2. **Seam offset** b_off ~ N(0, 0.25) added to post-1950 b-knot PRIOR CENTERS
   (hierarchical ERA5-datum control): ERA5 keeps shape, level floats,
   d(age)/dz arbitrates.
3. **d(age)/dz** from FULL-res SP19 (0.1 m native): ±0.6 m window slopes
   every 1 m (FIRN_DAGE_SPACING), 6–129 m, ~120 obs; σ = max(fit-s.e., 4%);
   pred = −assemble(age.dx(0)·φ)/YEAR_S (x is height ⇒ sign flip).
4. **Deep thermal domain** GATED on FIRN_DEEP_T_FILE (csv depth_m,T_C):
   H_col→FIRN_H_DEEP (500), NZ 160, H&L stage-2 rate cut off at ρ=900
   (`hl_deep_cutoff_rho`, new guarded param — keeps parcels off the ρ_i
   barrier kink), enthalpy IC = ON-TAPE steady advection–diffusion solve at
   the spinup climate (Tk0, b(1000), Q_base, k-scales enter ⇒ control-
   consistent deep memory; K linearized at uniform Tk0), deep-T kernels
   wmax=8 m, σ=0.1 °C. **DATA STILL MISSING**: processed borehole csv stops
   at 130 m; USAP-DC here has only USP50 (≤106 m); Kahle 601396 =
   reconstructions; Buizert NOAA archive has borehole-temp for DF/EDC only,
   NOT SPICE. Ask Andrew for the SPC14 deep log source.
Chain: verify (logs/r5_verify.log) → `run_r5_chain.sh` (opt MAX_ITER 60).

**r5 LEG-1 RESULT (2026-07-04 18:56, maxiter stop, |g|=30 — NOT converged;
backup `sp_joint_r5_it60.json`): J 399.5→322.7.** Fits ρ 0.49 / age 0.23 /
dage 2.19 / T 0.52 / v 0.60.
- **CONDUCTIVITY VERDICT (headline): k_firn_scale=0.988** — firn/ice branch
  at the PUBLISHED Calonne value; k_snow_scale=0.53 (×2 reduction confined
  to the snow branch — physically interpretable). **The old k_factor=0.145
  (×7) was a quadratic-shape ARTIFACT.** Borehole T fits at 0.52σ with
  literature deep k.
- Seam: b_off=+0.074 (ERA5 era +7.7%); 20th-century b LIFTED broadly
  (b(1950)=0.125 = +1.5σ ABOVE Buizert); the 1950/1962 step persists (1.48×)
  despite knot freedom ⇒ possibly REAL mid-century dip (cf. Mosley-Thompson
  1960s minimum) on top of the datum offset.
- s2=0.822, ezz=−4.4e-5 robust. k0 +8%/k1 −11%, Ea's <1% (net stage-2 rate
  ~−4%): expected re-equilibration under the corrected thermal model —
  ends the "unchanged" streak DEFENSIBLY (T(z) changed, so Arrhenius
  prefactors moved).
- T-history reshaped AGAIN (warm peak now ~1995; 1965→2015 ≈ −0.1) —
  T(t) is conductivity-model-dependent ⇒ deep borehole T is now the
  single most important missing constraint.
- **WARNING — dage dominance**: 124 gradient obs at 2.19σ = 92% of J;
  density over-pulled in 1850–1960 deposits (Buizert corr → −0.47,
  panel c seesaw). Desroziers fix ready (`FIRN_SIG_DAGE_SCALE=2.2`,
  `FIRN_WARM_FILE`, `FIRN_TAG` envs added) — **Andrew chose to CONVERGE
  the current objective first**; continuation launched 19:07 from the
  it-60 MAP (same tag; landing ~01:30). Revisit σ inflation after.
- Robustness fixes (in script): solvers REBUILT per eval + non-finite
  guard before gradient — after a NaN probe, reused
  LinearVariationalSolver objects kept sticky PETSc/adjoint state and
  every later gradient failed ("0 nonlinear iterations"); leg-0 died
  that way at J=1e8 (evals 2–5).

**r5 LEG-2 (continuation, landed 2026-07-05 00:13): J 322.67→318.32, 72
evals, maxiter AGAIN — but the remaining flat direction is now identified:
k_snow_scale ↔ recent-T-history DEGENERACY.** ks traversed 0.53→1.58 for
ΔJ≈4 while the T-history reshaped a third time (peak now ~1980,
post-1980 “cooling” −0.6 °C) — ≤130 m borehole T with one point <15 m
cannot pin near-surface k against the recent surface-T signal. Don't
iterate further; break it with DATA:
- **USP50 instrumented-hole T time series is ALREADY STAGED**
  (`data/usap_dc/601525/USP50_firn_temperatures_170109-181224.csv`,
  13–113 m, 2017–18): seasonal amplitude-decay vs depth is a direct
  near-surface conductivity measurement → fit k_snow_scale offline (or
  assimilate amplitude/phase) and FIX/prior-tighten it.
- Deep SPC14 T log (still awaited) pins the long-memory side.
Final fits: ρ 0.47, age 0.23, dage 2.18, **T 0.41σ**, v 0.59; corr
−0.47→−0.31 (density over-pull eased). k_firn_scale=0.987 (ROCK SOLID —
the conductivity headline stands). s2=0.80, ezz −4.7e-5, b_off +6.7%,
b(1950)=0.128 (+1.6σ above Buizert), 1950/62 step 1.52× (persistent ⇒
likely real mid-century dip + datum offset).
**LESSON — report the law as RATES, not (k,Ea)**: (k1,Ea2) slid −21%/−2%
across legs but the stage rates at −45.2 °C, b=0.08 are INVARIANT
(c0: 3.52→3.66e-3 /yr, +3.9%; c1: 1.215→1.194e-3 /yr, −1.7% vs pp).
The (k,Ea) decomposition is degenerate at an isothermal site; the
sensitivity-weighted rate is the robust deliverable.

## USP50 seasonal-damping conductivity fit (2026-07-05) + SUSPECTED T-DATUM BUG
`diagnostics/usp50_k_snow_fit.py` (numpy-only; USP50 601525 thermistors
0–40 m, 2017–18, + USP50 density cores): complex harmonic BVP through
Calonne k(ρ(z)) (k_firn=0.987 fixed), pinned at 1.2 m, fit 12 sensors to
14.2 m, joint lnA+phase rms 0.080. **k_snow_scale = 1.35 ± 0.02 (LOO)**
[1.25 without burial corr] → BREAKS the r5 ks↔recent-T degeneracy;
inside the r5 flat valley [0.53, 1.58]. Round-5b: fix/tight-prior
k_snow_scale = 1.35 (σ_log ~0.08). → `results/usp50_k_snow_fit.{json,png}`.

**⚠ SUSPECTED DATUM BUG in `processed/spicecore_borehole_T.csv`** (found
via the USP50 sensor means): staged file 13–130 m = −45.58..−44.87 °C;
staged MINUS 5.8 = −51.38..−50.67; USP50 MEASURED 20–40 m firn means =
−50.79..−51.25 (and canonical SP 10-m firn T ≈ −51). The staged borehole
file appears to be (true borehole + 5.8 °C) — the Buizert cloud→surface
offset applied to the WRONG series at staging. If confirmed: all SP
rounds fit an absolute-T anchor ~5.8 K too warm. Shapes/anomalies
(T-wiggles, b(t), seam, s2, ezz, k_firn) ~robust; the Arrhenius ANCHOR
is not — prefactors would recalibrate ×~1.3–1.5 and "rates at −45.2 °C"
becomes "rates at ~−51 °C" (transferability!). Fix = restage (−5.8),
shift T-knot bounds [−48,−43]→[−54,−49] and the _T6 prior ramp by −5.8,
rerun r5 warm-started. ALSO: ERA5-vs-recovered offset then reads as the
documented ERA5 warm bias (+5) instead of the puzzling −0.7. AWAITING
ANDREW's confirmation of file provenance.
**CONFIRMED from raw data 2026-07-05 (Andrew: "read from the csv — this
wouldn't have been corrected; where things agree push on")**: USP50 raw
thermistors are the uncorrected measurement (−50.8..−51.3 at 20–40 m);
601551 insitu compilation has no SP site; staged csv = raw + 5.8 exactly.
⇒ ROUND 5b launched: `FIRN_T_SHIFT=-5.8 FIRN_KSNOW_FIX=1.29
FIRN_WARM_FILE=sp_joint_r5.json FIRN_TAG=sp_joint_r5b` (k_snow refit at
the corrected site T: **1.29 ± 0.02**, F(T) at 222.25 K). Verify: replay
deterministic (J=486.84); warm-start rms = the predicted datum signature
(T carries over at 0.45σ; ρ 1.10/age 1.01 = rate deficit at colder anchor
→ prefactors re-anchor, expect k0 +~15%, k1 +~35%). The datum correction is
documented in `tutorials/southpole/data/README.md` (caveat 1); staged csv
kept for reproducibility (scripts shift via FIRN_T_SHIFT). fit_stats +
reanalysis_decomp now law-aware (+FIRN_T_SHIFT in fit_stats).
**GOAL (Andrew): comprehensive firn reanalysis** — after r5b: dage-σ
decision → UQ at final MAP (adapt hessian/sensitivity to 48-ctrl r5
forward) → upgrade `sp_reanalysis.py` to b(t)+Calonne → attribution +
ΔFAC with honest bars + figure suite.
DONE 2026-07-05 evening: `sp_reanalysis.py` upgraded (b(t) + law-aware +
mass/air split; smoke-tested at r5 MAP: sanity 0 exactly, closure ~7%).

**r5b RESULT (2026-07-05 23:21, corrected datum, `sp_joint_r5b.json`):
J 487.3→318.14 — statistically identical fits to r5 (ρ 0.47/age 0.23/
dage 2.18/T 0.43/v 0.59). THE DATUM TEST PASSED: rates at the site T are
INVARIANT across the 5.8 K shift (c0=3.69e-3, c1=1.197e-3 /yr @−50.8 °C
vs 3.66e-3/1.19e-3 @−45.2) — quote the law as RATES AT −50.8 °C.**
Re-anchoring went through the Ea's (Ea1 10424→10150, Ea2 21870→21510;
k0 flat, k1 +11%) — same degeneracy direction. k_snow sat ON the USP50
pin (1.293, no tension); k_firn 0.983; s2 0.801; ezz −4.9e-5; b_off
+8.1%; seam 1.51× persistent; corr −0.36. T-history (corrected datum):
T(1000)=−51.20, warm peak −50.27 @1980, post-1980 −0.57. Maxiter stop,
|g|=8.5 (fine as r6 warm start). Diagnostic png:
`sp_midrecord_diagnostic_r5b.png`. Warm-start double-shift bug FIXED
(auto-detect warm-file datum, ±3 °C rule).

**ROUND 6 queued (Andrew: denser T + b nodes).** `FIRN_KNOTS_FILE=knots_r6.json`
support added to the r5 script (layouts fully data-driven; priors/bounds/
brackets/warm-interp/b_off-mask all derive). Layout: **22 T knots**
(unchanged pre-1750 — prior-bound zone; 10-yr 1900–1990, 5-yr →2015; the
USP50 k_snow pin makes recent T identifiable) + **52 b knots** (50-yr
→1400, 25-yr →1750, 15-yr →1900, 10-yr →1945, 5-yr →2015; d(age)/dz
resolves 4–8 yr; **dt=5 is the aliasing floor** — hybrid dt is the step
after) = **84 controls**. Rationale: dage 2.18σ is largely REPRESENTATION
misfit — extract signal with knots BEFORE deciding σ-inflation (the two
compose; knots-first is more honest). Plan: when r5b lands → verify
(FD on b1985/Tk18/b_off, warm from sp_joint_r5b.json) → launch r6
(~100 evals ≈ 8 h). Later Hessian at 84 ctrls ≈ 12 h.
**r6 RESULT (2026-07-06 05:38, `sp_joint_r6.json`, 94 evals, maxiter):
J=322.15 (48c r5b was 318.14 — NOT comparable, +36 prior terms). Fits
ρ 0.47 / age 0.21 / **dage 2.18 UNCHANGED** / T 0.44 / v 0.60.**
- **Knot-starvation hypothesis REFUTED: dense b(t) did NOT touch the
  2.18σ layer-gradient floor** ⇒ the floor is the dt=5 model class
  (sub-5-yr accum variability + cell-scale smoothness). The honest next
  move is Desroziers σ_dage ×2.2 (or hybrid dt=1 post-1700 — bigger).
- **Mid-century b(t) RESOLVED into a peak-and-crash**: 0.082 (1900) →
  0.100 (1940) → 0.111 (1945) → **0.121 (1950)** → 0.096 (1955) → 0.088
  (1960) → 0.080–0.082 plateau (1965–75). The "seam step" is now a
  ~15-yr 1940s–50s accumulation high (+40–50%!) then crash. b_off →
  **+0.123 (ERA5 ×1.13)** ⇒ posterior modern b ≈ 0.079–0.088 = lands in
  the stake-farm band. VALIDATE the 1940s–50s spike against the
  PUBLISHED SP19 accumulation curve (Winski/Lilien layer thickness ×
  thinning) before believing it — could be an age-scale artifact
  (deposition 1940–55 ≈ 14–18 m).
- Rates INVARIANT again (c0=3.665e-3, c1=1.195e-3 @−50.8 °C; 7th
  revision). s2=0.7995, k_snow on pin (1.297), k_firn 0.975, ezz −4.3e-5.
- Decadal instrumental T (corrected datum): warm 1970–90 (−50.3..−50.4),
  cool 2000s (−50.9 @2005), mild 2010s recovery. COMPARE with the
  Amundsen–Scott station record for validation (note Clem-2020 post-1990
  warming is anti-phased with our 2000s dip — knot σ ~±0.5 °C though).
- Diagnostic: corr −0.39; density seesaw (−0.5σ 1850–1950 / +1σ post-
  1960) persists ⇒ σ-weighting matter, not resolution.
**DECISION POINT (Andrew): σ_dage inflation ×2.2 for r6b (recommended,
~4 h warm-started) vs hybrid dt=1 post-1700 (honest but ~1.5× cost,
bigger surgery) — then the 84-ctrl UQ chain and the reanalysis product.**

**⚠⚠ 2026-07-06: r6 ACCUM SPIKE IS AN ARTIFACT — the b-prior was the
problem all along** (`diagnostics/sp_accum_validation.py` →
`results/sp_accum_validation.{png,json}`). MODEL-FREE test: raw SP19 annual
layer thickness → apparent ice-eq accum b_app = λ·ρ/ρ_i crosses 1950
CONTINUOUSLY (no seam; the seam is only in the PRIORS). Findings:
- **The +49% MAP spike at 1950 is NOT in the raw layers** — raw b_app is
  SMOOTH ~0.093 across 1950 (z=−1.3σ, a slight LOCAL LOW, opposite of the
  MAP peak). Spike = Buizert/ERA5 prior-seam + b_off artifact.
- **ERA5 net-accum prior is 27% LOW** (0.0727 vs Buizert-1900-1950 clim
  0.096, raw layers 0.093, stakes 0.085-0.093 — 3 independent witnesses
  agree ERA5 is the outlier). ERA5 P−E has the known plateau sublimation/
  dry bias.
- **ROOT CAUSE of the 2.18σ dage floor: the biased prior at σ_logB=0.15
  OVERRODE the layer data** — the MAP b(t) tracks the PRIORS (Buizert wave
  + low ERA5), not the raw layers. It was NEVER dt=5 representation error
  (my earlier hypothesis was WRONG). ⇒ the σ_dage-inflation decision above
  is moot; the fix is the prior, not the obs weight.
**★ r8 = FINAL / FROZEN SP MAP (2026-07-07 03:58, `sp_joint_r8.json`).**
Coarse knots (15 b multidecadal + 13 T) + σ_dage×2.2 + de-biased prior +
corrected datum + k_snow pin. **J=81.3, ALL misfits ≤1σ: ρ 0.47, age 0.19,
dage 1.01 (was 2.16!), T 0.40, v 0.60** — cleanest fit of the campaign.
**Spike GONE** (model-free check: b(1950) +6% vs the +53% artifact; b(t)
smooth 0.075–0.099, mean 0.087 = stake-farm band). Rates INVARIANT 8th time
(c0=3.61e-3, c1=1.22e-3 @−50.8 °C). k_snow 1.297 (pin), k_firn 0.978,
s2 0.805, ezz −3.3e-5 (kinematic), b_off −0.099. T(1000)=−51.24, peak
−50.24@1975. corr(ρ-res,Buizert) −0.26. **THIS is the SP result for the
paper.** Next SP: reanalysis + UQ at r8 (unblocked), then distill to
`tutorials/southpole/`.

**⚠ r7 VERDICT (2026-07-06 20:36): DE-BIAS REFUTED the artifact hypothesis —
the spike is a d(age)/dz NULL-SPACE mode, not the prior.** `sp_joint_r7.json`:
b(1950)=0.123 (+53%, UNCHANGED from r6; b_off went −0.14). Rates invariant
(c0 3.66e-3, c1 1.21e-3). `diagnostics/sp_layer_fit_diagnostic.py`
(`sp_layer_fit_diagnostic.png`) is DEFINITIVE: control b(1950)=0.123 but the
model's OWN realized layer→b is only 0.098 (~4× damping) — the forward model
SMEARS b(t) spikes, so the dense 5-yr b-knots sit in the observation
operator's NULL SPACE and collect warm-start-inherited noise. **More b-nodes
(r6/r7) was COUNTERPRODUCTIVE.** The 2.16σ dage floor = unfittable interannual
layer scatter (obs 3–10 yr/m; model smooth 5–7) + a 0.7 m mean age-registration
offset at 13 m (model layers ~5% too thick; RHO_SURF=350 fixed). Trustworthy
accum = the multidecadal ~0.09 where model-layer-b, raw obs b_app AND stakes
agree. **⇒ accum approach: COARSER b-knots + σ_dage inflation (σ-inflation IS
right — fine structure genuinely unfittable); densification+T-history UNAFFECTED.**
See memory `sp_accum_prior_bias_finding.md` (corrected).

**⇒ ROUND 7 (de-biased prior) — superseded by the verdict above.** `FIRN_B_CLIM=0.096` (post-1950
centers = independent Buizert climatology, no ERA5, no seam) +
`FIRN_SIG_LOGB=0.20` (let layers express decadal structure). Warm from r6,
r6 knots, corrected datum, k_snow pin. Verify-gated → `logs/r7_chain.log`,
opt `logs/sp_joint_r7_opt.log`. b_off kept free (should → ~0). TEST: if
dage drops well below 2.18σ AND the spike dies AND b(t) tracks raw b_app,
the prior WAS the limiter (confirms finding). Kahle 2020 recon
(usap-dc 601396) would be a 4th witness — needs manual download to
`processed/kahle2020_spice_accum.csv` (script auto-detects; USAP-DC
file API needs auth).

**r6 LAUNCHED 2026-07-06 00:17** (verify: replay exact J=325.6447;
b_off/Tk18 FD +1.0000; b1985 +0.9999 = FD roundoff on a small 5-yr-knot
component (adj/FD agree to 3e-4; identical code path verified 4 rounds) —
NOT an adjoint issue. Gate false-alarmed on a stale name: b1875 is not in
the dense grid (1870/1885 are) + strict '+1.0000' string match; launched
manually. NOTE for future gates: match ratio within ±5e-4 and use names
that exist in the active layout.)

## Open next steps (reordered 2026-07-01 after UQ verdict + fit-quality discussion)
The joint fit's residual floor is the FORCING representation (6 T-knots, const
bdot): model profiles are smooth, data have coherent structure (density strata,
borehole-T decadal wiggles). UQ says pre-1750 T-shape is prior-bound at ≤130 m.
1. **Deep borehole T** — SPICEcore T extends far below our 130 m cut; the
   200–800 m range is the classic paleothermometry window that would pin
   Tk0–Tk3 (kills the 89% variance term). Extend thermal domain (firn column +
   simple advecting ice column below) or a dedicated deep-thermal assimilation.
2. **Time-varying accumulation** — bdot(t) knots as controls (priors from
   ERA5/Buizert/SP19 layer thickness); assimilate density in AGE space (ρ(age)
   maps wiggles to years). Forward-only experiment first (reassign ac/ws per
   step off-tape, trivial). Needs dt≈1 yr for the recent centuries (hybrid dt).
3. **Honest observation-error model** — EMPIRICALLY SIZED 2026-07-02
   (`../diagnostics/sp_obs_error_analysis.py` → results/sp_obs_error_analysis.*):
   noise L_corr: density 0.71 m (sd 5.5 kg/m³), T 2.0 m (sd 0.039 C),
   age-gradient 0.2 m ⇒ at ~3 m assimilation spacing N_eff ≈ N (45/45, 31/40,
   36/40). ⇒ next inversion round: per-POINT χ² weights (drop the 1/N_d),
   σ kept generous for representation error; consider assimilating d(age)/dz.
   The per-dataset 1/N was ~6.7× over-conservative — Laplace bars shrink
   accordingly when re-run.
4. **Weak-constraint 4D-Var** — model-error control η(z,t) after (1)-(3);
   recovered η maps WHERE/WHEN the physics is wrong (the structural-error figure).
5. **Break conductivity degeneracy** — Calonne-2019 law + small correction
   (softest mode k_factor:−1.00 quantifies it; 8% of var(h')).
6. **ES-MDA / MCMC cross-check** — forward is 7 s ⇒ 100-member ensembles are
   minutes; derivative-free posterior to validate Laplace.
7. Multi-site (multiple ApRES, USP50 second density core) for spatial variability.

## Memory (auto-loaded index: `MEMORY.md`)
Key notes: `sp_joint_inversion_complete.md`, `sp_forward_model_is_sound.md`,
`sp_adjoint_is_correct.md`, `sp_assimilation_working.md`, `sp_paleothermometry.md`,
`pyadjoint_timevarying_control_forcing.md`.
