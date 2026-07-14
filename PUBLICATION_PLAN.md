# FirnGrain → publication plan

Goal: a clean, public (GitHub) repository backing a methods paper on 1D firn
densification + thermal + accumulation inverse modeling (Firedrake + pyadjoint),
demonstrated on **three test cases**:

1. **Synthetic OSSE** — recover known truth; validate the adjoint & identifiability.
2. **South Pole** — cold, low-accumulation real site (flagship; already built).
3. **Summit, Greenland** — warm, high-accumulation real site (transferability).

Status of this doc: updated 2026-07-09. Grounded in a full repo map + data
inventory (not a sketch).

## PROGRESS (2026-07-09)
- **Phase 0 (freeze SP science): DONE** — r8 is the frozen SP MAP
  (`test/southpole/results/sp_joint_r8.json`, J=81.3, all misfits ≤1σ).
- **Phase 1 (skeleton + deletions): DONE** — git init (local, not pushed),
  README/LICENSE-TBD/gitignore/requirements/pyproject, 7 dead versions deleted,
  tutorials/{synthetic,southpole,summit}/ with data + provenance READMEs.
- **Phase 2 (shared inverse engine): DONE + VALIDATED** — `src/firnpack/inverse/`
  (config.py + engine.py). `tutorials/southpole/run.py` reproduces r8 EXACTLY
  (forward J 81.306). The engine is the through-line for all three cases.
- **Phase 5 (Summit): inversion DONE** — `tutorials/summit/run.py` recovers
  Summit's law; matches SP (stage-2 rate ratio 1.00, params 1–9%). Data staged
  (FirnCover+Fourteau density, GISP2 age, FirnCover firn-T).
  `results/law_comparison.png` = the transferability figure.
- **Phase 3 (synthetic OSSE): IN PROGRESS** — `tutorials/synthetic/run.py`
  built + verified (FD clean); full recovery run + `plot_recovery.py` figure.
- **REMAINING**: SP reanalysis+UQ at r8 (Phase 0 tail); distill SP to
  tutorials/southpole (Phase 4); LICENSE choice; CI + provenance polish
  (Phase 6). Summit refinements: σ_dage inflation, endpoint b-knot.

Nothing destructive done; git is local-only (not pushed).

---

## The linchpin: extract a shared inverse engine

The South Pole work reimplements its adjoint stepper + Gaussian-kernel misfits +
priors + optimizer loop **inline in each of ~15 scripts** (the package
`src/firnpack/solvers/FirnColumnSolver` is NOT used by the assimilation — the lean
adjoint stepper lives in the scripts). To make three *tests of one framework*
rather than three divergent scripts, pull that machinery into
`src/firnpack/inverse/` behind a config-driven `assimilate(SiteConfig)`. Then
synthetic / South Pole / Summit are thin configs (data paths, knot layouts,
priors, which observables are present) over one validated engine.

Target layout:
```
firnpack/
  README.md  LICENSE  .gitignore  pyproject.toml  requirements.txt (pinned, incl firedrake)
  src/firnpack/
    constants.py  mesh.py
    physics/densification.py          # arthern_ligtenberg, herron_langway, kingslake, stokes
    models/firn.py                    # FirnParameters, FirnModel, Calonne conductivity
    solvers/firn_solver.py            # forward-only FirnColumnSolver (kept; used by unit tests)
    inverse/                          # NEW — the shared assimilation engine
      stepper.py    # lean adjoint (H,rho,w,age) theta-scheme stepper
      observables.py# Gaussian-kernel scalar misfits; d(age)/dz; ApRES velocity operator
      priors.py     # log/linear controls, knot brackets, time-varying forcing (fresh-Constant)
      driver.py     # assimilate(SiteConfig) -> MAP; hessian(); sensitivities()
  tutorials/
    synthetic/    # OSSE: truth -> synthetic obs -> recover (test #1)
    southpole/    # config + data + results + figures (test #2)
    summit/       # config + data + results + figures (test #3)
  tests/          # real pytest: adjoint FD/Taylor, conservation, MMS, enthalpy round-trip
  data/           # staged obs per site, with provenance READMEs (DOIs)
  paper/          # figure scripts + a reproduce-all driver
```

---

## Phase 0 — Freeze the South Pole science (BEFORE any cleanup)

Cleaning while the science moves is how things break. Open SP items:

- **Accumulation approach (from the r7 layer-fit diagnostic, 2026-07-06).** The
  d(age)/dz observable has a broad NULL SPACE at sub-decadal scales — the
  forward model smears b(t) spikes (control b(1950)=0.123 → model realized
  layer-b 0.098). The dense 5-yr b-knots collect noise. FIX: coarser
  multidecadal b-knots + σ_dage inflation; the trustworthy accumulation is the
  smooth ~0.09 (raw b_app / model-layer-b / stakes agree). One more inversion
  round (r8: coarse b-knots, σ_dage×~2, corrected datum, k_snow pin) → lock the
  final SP MAP.
- **UQ** at the final MAP: adapt `sp_uq_hessian_pp.py` / sensitivities to the
  final control set; honest Laplace bars.
- **Reanalysis product**: `sp_reanalysis.py` (already upgraded to b(t)+Calonne)
  at the final MAP → attribution + ΔFAC with bars + figure suite.
- Densification law (rates, invariant 7×) + T-history are ALREADY robust and
  datum-corrected; they are not blocked by the accumulation item.

Deliverable: one locked `tutorials/southpole/results/sp_final.json` + figures.

## Phase 1 — Repo skeleton + safe deletions (non-destructive to science)

- `git init` (repo is currently NOT under git — clean slate, no history to scrub).
- Add README, LICENSE (pick one), .gitignore (outputs, *.h5, *.npz, __pycache__),
  pinned requirements.
- DELETE (zero references, agent-verified): `src/firnpack/models/firn_v2..v4.py`,
  `src/firnpack/solvers/firn_solver_v2..v5.py` (7 files), `src/firnpack/statistics/`
  (empty stub).
- Move `test/archive/` (88 dev scripts) OUT of the release tree (keep as a
  git tag / separate `history/` branch — it's the dev record, not deleted).
- Collapse case-duplicate scraper dirs: `test/SouthPole`, `test/WAIS` (metadata
  only) → fold into the lowercase curated dirs; delete the stray tmp* scrape
  files. Move root-level stray PNG/NPZ (synthetic_gadopt_*, w_*.png, etc.) into
  experiment results dirs.
- Trim `src/firnpack/__init__.py` eager imports if dropping solvers/mesh/data (it
  currently imports them transitively — see agent note).

## Phase 2 — Extract the shared inverse engine (highest value / highest risk)

- Refactor `sp_joint_assimilate_r5.py` (the most complete driver) into
  `src/firnpack/inverse/`. Validate: engine reproduces the locked SP MAP bit-for-bit.
- Config object carries: mesh/spinup, knot layouts, priors, which observables
  are present (density/age/dagedz/T/velocity/deep-T), datum shift, conductivity
  law. South Pole becomes the first `SiteConfig`.

## Phase 3 — Synthetic OSSE (test #1)

- Rebuild on the CURRENT H&L + thermal + accum framework (the archived
  `test_gadopt_age_layers_*v36` OSSE is grain-based/old physics — reuse its
  structure, not its model). Truth {densification params, T(t), b(t)} →
  synthetic density+age+T(+velocity) with realistic noise → recover within
  posterior bars. Fold in the adjoint Taylor/FD verification
  (`clean_stepper_taylor.py` is the best-matching seed).
- This is the method-credibility anchor + demonstrates identifiability (incl.
  the null-space / resolution limits found empirically at SP).

## Phase 4 — Distill South Pole (test #2)

- Collapse ~15 exploratory scripts → one `tutorials/southpole/run.py` (config
  + engine) + results + figures. The hard-won lessons become methods/discussion:
  borehole-T datum correction, ERA5 accum bias, Calonne vs quadratic
  conductivity, the d(age)/dz resolution limit, the isothermal-site (k,Ea)
  degeneracy (report rates, not k/Ea separately).

## Phase 5 — Build Summit (test #3) — the DATA lift

Summit currently has ONLY accumulation + climate forcing. MUST ACQUIRE &amp; stage
(to `data/summit/`, matching SP CSV formats):
- **Density profile** — FirnCover Summit site, or GISP2/Hawley Summit density.
- **Depth–age scale** — GISP2 or GRIP layer-counted timescale (NOAA Paleo).
- **Borehole temperature** — GISP2/GRIP borehole-T (e.g. Cuffey &amp; Clow).
- ApRES velocity almost certainly unavailable → drop that observable (engine
  already treats observables as optional).
Then Summit = a `SiteConfig` (warm ~−30 °C, high accum ~0.23 m/yr, possible
seasonal melt) over the Phase-2 engine. Data acquisition dominates the effort.

## Phase 6 — Package for publication

- pytest CI: the real unit tests (`test_firnmice_checkpoint.py` + new
  adjoint-FD / conservation / MMS / enthalpy round-trip) + a fast synthetic OSSE
  smoke test. Add pytest config.
- README (what/install/reproduce), per-dataset provenance (DOIs), pinned env,
  `paper/reproduce_all.py`, LICENSE.

---

## Rough effort (dominated by Summit data + the engine refactor)
Phase 0 ~days · P1 ~1 day · P2 ~2–3 days · P3 ~2 days · P4 ~2–3 days ·
P5 ~1 week · P6 ~2 days.

## Decisions for Andrew
1. Second site = Summit (WAIS is no better off — also missing density/age/T).
   OK to spend the acquisition effort on FirnCover-Summit density + GISP2
   age + GISP2 borehole-T?
2. LICENSE choice (MIT / BSD-3 / Apache-2).
3. Keep `test/archive/` as a git tag vs drop entirely.
4. Phase 0 accum fix: coarse b-knots + σ-inflation (recommended) — confirm
   before r8.
