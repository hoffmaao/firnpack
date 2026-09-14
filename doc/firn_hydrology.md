# Firn hydrology: percolation, refreezing and the aquifer column

Methods note for `firnpack.models.firn_richards`, `firnpack.solvers.firn_richards_solver`,
`firnpack.surface_energy` and the coupled driver `firnpack.aquifer`. The case
study that exercises them is `tutorials/aquifer/`.

## 1. The question

Southeast Greenland holds liquid water in firn pore space through the winter
(Forster et al. 2014; Koenig et al. 2014; Miège et al. 2016). Kuipers Munneke
et al. (2014) explain persistence as a race between three vertical rates:
recharge by summer melt, burial by accumulation, and refreezing by the winter
cold wave. The column here resolves all three and nothing lateral: it is one
dimensional, the aquifer is treated as confined, and water leaves only by
refreezing or by riding out of the base with the compacting firn. That is a
deliberate scope. Lateral drainage sets the equilibrium water-table depth; it
does not decide whether water survives the winter, which is what the column
measures.

## 2. Water: mixed-form Richards on a moving, changing medium

Pressure head `h` [m] is the unknown. With `theta(h)` the volumetric water
content, `K(h)` the hydraulic conductivity, `w` the (downward, negative) firn
velocity, `e_z` the upward unit vector and `m` the refreezing rate
[kg m^-3 s^-1]:

    d(theta)/dt + S_s Se(h) dh/dt
      = div( K(h) grad(h + z) ) + div( theta w e_z ) - m / rho_w

The storage term is Celia, Bouloutas & Zarba's (1990) mixed form: the time
derivative is written as `(theta(h) - theta_old)/dt`, never as `C(h) dh/dt`,
so mass is conserved to round-off regardless of the nonlinearity in `theta(h)`.
A small specific storage `S_s = 1e-4 m^-1` keeps the saturated rows of the
Jacobian non-singular. `theta_old` is the water content **measured with the
curves in force at the start of the step**, not `theta(h_old)` on the refreshed
curves: the medium compacts under the water, so `theta_s` shrinks between
steps, and re-evaluating both ends of the difference on the new curve would
make water disappear from the budget with nothing having moved.

The second divergence is matrix advection: the water is carried with the
firn. It is the burial term of Kuipers Munneke et al., and it also gives the
natural outflow at the base of the domain, `theta * w`, which is not a drainage
law and has nothing to tune - the firn below the aquifer keeps moving at the
velocity the densification model already imposes, and the water in its pores
moves with it.

### Constitutive relations

van Genuchten (1980) retention and Mualem (1976) conductivity:

    Se(h) = (1 + |alpha h|^n)^(-m),  m = 1 - 1/n
    theta = theta_r + (theta_s - theta_r) Se
    k_r(Se) = Se^(1/2) [1 - (1 - Se^(1/m))^m]^2

with `theta_r = 0`, `alpha = 4 m^-1`, `n = 2`. Firn is coarse and weakly
capillary, which argues for a large `alpha`; `n = 2` rather than the 3-4
reported for snow is a numerical concession (with `n = 3` the conductivity
climbs four orders of magnitude over 0.8 m of head and Newton fails at any
step size). Its consequence is a long retention tail, and that tail matters:
see section 6.

`theta_s = 1 - rho / rho_i` and `K_s = rho_w g k / mu_w` are fields, refreshed
every step from the prognostic density and grain radius. Permeability is
Calonne et al. (2012), `k = 3 r^2 exp(-0.013 rho)`, multiplied by a pore-
connectivity factor that falls to zero at solid ice but is deliberately left
non-zero below bubble close-off: close-off is gradual and wet firn keeps some
deep mobility. An optional density-dependent correction (`perm_scale_deep`)
applies a measured conductivity only in the density range where it was
measured: Miller et al. (2017) measured 2.7e-4 m/s inside the Helheim aquifer
at 550-650 kg m^-3, where Calonne's snow-based fit runs about 10x high, so the
correction ramps in over exactly that band (`perm_rho_lo` to `perm_rho_hi`)
and holds at its full value in anything denser. In firn lighter than
550 kg m^-3 - the near-surface firn of the cold-wave zone, where Calonne rests
on direct snow measurements - it does nothing.

### Discretisation

DG1 in the head, so the water table is an interior free surface that the
solution finds. Symmetric interior penalty for the diffusive flux, donor-cell
upwinding for the gravity flux and for matrix advection: the wetting front is
monotone by construction. Four numerical devices, each earned by a failure:

* an absolute floor `K_min = 1e-7 m/s` on the conductivity (film and vapour
  transport), so cells ahead of a drying front stay coupled instead of going
  singular. Absolute, because firn's `K_s` is 1e-2 to 1e-1 m/s and the usual
  relative floor of 1e-6 would be 0.3 to 5 m/yr of gravity drainage from
  every dry cell (section 5, item 3). 1e-7 is what that relative floor
  amounted to in near-surface firn; smaller values (1e-8 to 1e-10) leave the
  cells ahead of a refreezing front decoupled and Newton fails.
  The same `K` multiplies the diffusive and the gravity flux: floor only one
  of them and hydrostatic equilibrium stops being a solution;
* a linear continuation of the retention curve below `h_min` (C1 join), so
  the driest cells keep a constant capacity and a bounded head (section 5,
  item 3);
* a C1 Hermite blend of `k_r` on `[-blend, 0]`, because Mualem's curve has a
  derivative jump at `h = 0` and Newton stalls on every cell that straddles
  the water table;
* Newton with the L2 line search. Backtracking enforces sufficient decrease of
  a merit function and stalls once the column spans -40 to +8 m of head with
  conductivities varying over many orders of magnitude between neighbours;
  L2 minimises the residual norm along the direction and gets through. Steps
  that Newton refuses whole are halved down to `dt/256`.

## 3. Phase change

Per step, the cold content of a cell sets a freezing demand
`f = min(-rho H / L, freeze_safety (theta_entry - theta_min) rho_w)`, where `H`
is the matrix enthalpy (negative below the melting point) and `theta_min` is
the water content at the retention floor `h_min = -100 m`. The demand enters
the Richards residual as a volumetric sink, **implicit in the head**:

    sink = (f / (rho_w dt)) * min(1, (theta(h) - theta_min) / delta)

Treated implicitly, the ramp is a linear decay toward `theta_min`,
`theta - theta_min = (theta_old - theta_min) / (1 + dt f / delta)`, so no cell
can end a step below the driest state the curve expresses. The freezing that
actually occurred is recovered by projecting the sink with the residual's own
quadrature, so the reported mass is the mass the solve removed and the water
budget closes to round-off. Its latent heat is added to the matrix enthalpy,
which can therefore never overshoot the melting point.

The refrozen mass is an ice source. Meyer & Hewitt's (2017) ice equation is

    D(rho)/Dt + rho dw/dz = m

and refrozen water fills pores in place: it raises density and leaves the
column thickness alone. So `m` is a source in the density equation, and the
base-to-surface continuity integration that gives `w` sees it only through the
total density tendency, reducing to `rho dw/dz = -compaction`, exactly as in
the dry model.

## 4. Energy

The firn column solves an enthalpy equation. Only the firn solver transports
enthalpy; the Richards half touches `H` solely through the latent heat of
refreezing. The surface forcing is a skin-temperature energy balance
(`firnpack.surface_energy`): shortwave with an ageing albedo, longwave, bulk
turbulent fluxes with Richardson-number stability, conduction into the firn,
and melt as the surplus once the skin reaches the melting point. It replaced a
degree-day rule whose factor spanned the answer (0.26 to 0.53 m w.e./yr over
the same forcing for 3 vs 6 mm/C/day). The aged-firn albedo floor is the
largest remaining free parameter and is swept, not calibrated.

### Surface mass balance

Snowfall is the surface mass balance. Melt adds no mass and removes none: it
converts snow that is already at the surface into water, which Richards then
carries down. In the discrete column that is a split of the top-boundary flux
rather than an extra inflow: the matrix influx is `snowfall - melt` and the
water influx is `melt`, so the net crossing the surface is the snowfall
itself. Driving the surface velocity with gross snowfall *and* injecting the
melt counts the melted snow twice, once as the snow that fell and again as the
water it became, which put 30-50% more mass into the ERA5-forced column than
the climate delivers. Regression test:
`test_melt_does_not_add_mass_to_the_column`.

Applying that debit as a volumetric ice sink inside the column instead does
not work in a fixed-mesh Eulerian model, and the reason is physical rather
than numerical: over a melt season the melt exceeds the entire ice content of
the top metre, so no cell can supply it, and the attempt drives the near
surface density and enthalpy to nonsense. In reality the surface lowers. The
debit therefore belongs at the surface, where the mass is arriving, and it is
taken on the same trailing year as the accumulation so the annual mass balance
is exact; the sub-annual phasing is approximate, which does not affect annual
burial. If the trailing year ever goes net negative the site is ablating,
which this column cannot represent, so runs report `ablation_steps`.

The split is of the *kinematics* only. The densification sees the **gross**
snowfall, because `bdot` in the overburden stress is a loading rate and this
column has no runoff: every kilogram that falls stays in it, as matrix ice, as
refrozen melt or as pore water, and all of it weighs on the firn below. So the
driver carries two surface rates - net into the surface velocity boundary
condition, gross into `prognostic_solve(accumulation=...)`. Feeding the net to
both would understate the loading by the melt fraction, which slows
densification and pushes bubble close-off deeper, and close-off depth is
already this model's largest disagreement with the observed aquifer base
(section 6).

## 5. Four ways the aquifer failed to appear, and what each was

These are recorded because each looked like physics before it was found.

1. **The column stretched instead of densifying.** The refreezing source was
   applied in the velocity integration with the compaction rate standing in
   for `D(rho)/Dt`, which solves `rho dw/dz = m - compaction`: every refrozen
   kilogram thickened the column and none raised the density. Under ERA5
   forcing close-off sat at 55 m against an observed aquifer base of 27.7 m,
   so there was no floor for water to pond on. Regression test:
   `test_refreezing_fills_pores_instead_of_thickening_the_column`.
2. **Freezing overdrew cells a wetting front passed through.** With the
   density fix the column grows real ice lenses each spring. The freezing
   sink was sized from the water present when the step was staged; the water
   drained on within the step, the sink kept taking it, `theta` fell below
   `theta_min` and the head ran to -966 m beside a saturated node at +23 m.
   Fixed by making the sink implicit (section 3). A `min(demand, available)`
   limiter gives the same bound but places the Jacobian's switch exactly
   where every water-limited cell sits; Newton chattered on it, smoothed or
   not. Regression test:
   `test_freezing_cannot_overdraw_a_cell_the_water_leaves_within_the_step`.
3. **Dry cells drained themselves without bound.** Beside the lenses, whole
   zones reached heads of -200 to -2400 m with no freezing demand at all.
   With `theta_r = 0` the van Genuchten tail has a capacity that vanishes as
   `1/h^2`, so a cell at the retention floor that loses a trace of water -
   through the conductivity floor, or by riding out with the firn -
   sees a divergent head change, and Newton has no leverage on it. The
   curve now continues linearly below `h_min` with the slope it has there:
   a trace loss costs a bounded head change and the row keeps a constant
   capacity. Two other cures were tried first and rejected. Tapering the
   floor with head cured the runaway but stalled Newton on the extra
   nonlinearity in otherwise benign states; flooring the diffusive flux
   only removed the leak but broke hydrostatic equilibrium (the test caught
   it). Regression test: `test_dry_cells_keep_a_bounded_head`.
4. **The interior gravity flux was downwinded.** `0.5 (K e_z . n + |K e_z . n|)`
   takes the positive part along `+e_z`, but the gravity flux vector is
   `-K e_z`, so the numerical flux picked the receiving cell, not the donor
   cell. Every existing test ran with `K` continuous across facets, where the
   two choices coincide, so nothing caught it. A wetting front descending into
   dry firn then moved at the dry cell's conductivity - the stall the absolute
   `K_min` floor was masking - which biases the split between refreezing in
   the cold-wave zone and recharge reaching depth. Regression test:
   `test_the_gravity_facet_flux_takes_the_donor_cells_conductivity`.

   The permeability ablation that was reported here (deep-only versus uniform
   scaling of Calonne changing the refrozen fraction not at all) was run under
   the downwinded flux and is withdrawn pending a re-run: with the flux
   throttled by the dry receiving cell, that insensitivity may be an artefact
   of this bug rather than a property of the column.

## 6. What the ERA5-forced column says

> **Every number in this section predates the corrections made on this branch
> and is being regenerated.** All ten runs - the four synthetic contrasts and
> the six ERA5 experiments - are pending re-run against the current physics
> and configuration, so the figures here are kept for comparison only and
> every rate and depth in this section is provisional. The conclusion that
> recharge is insensitive to permeability is withdrawn.

Full tables in `tutorials/aquifer/README.md`. In brief, and subject to the
notice above: a perennial aquifer forms under
every aged-firn albedo
floor from 0.76 to 0.70 (melt 0.46 to 0.71 m w.e./yr over 2006-2019), and
under the 1940-1959 climate, where it is marginal. The refrozen fraction is
74-78% throughout, so net recharge (19 to 70 kg m^-2 yr^-1) scales with melt
and the water table rises with it (45 to 26 m). The aquifer sits deeper than
observed (table 10-20 m and base 27.7 m, Montgomery et al. 2017; the Miller
et al. 2017 boreholes span 10.0-22.5 m) because the column
reaches bubble close-off at 50-52 m rather than ~28 m: the remaining
discrepancy is in the densification of wet firn, not in the hydrology. Each
80-year experiment takes roughly 45 minutes on one core (measured:
2760-2790 s for the completed runs).

## References

Calonne et al. 2012, GRL. Celia, Bouloutas & Zarba 1990, WRR. Forster et al.
2014, Nat. Geosci. Koenig et al. 2014, GRL. Kuipers Munneke et al. 2014, GRL.
Meyer & Hewitt 2017, TC. Miège et al. 2016, JGR. Miller et al. 2017, Front.
Earth Sci. Montgomery et al. 2017, Front. Earth Sci. Mualem 1976, WRR.
van Genuchten 1980, SSSAJ.
