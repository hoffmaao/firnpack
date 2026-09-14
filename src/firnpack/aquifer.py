"""Perennial firn aquifer column experiments (1D, confined).

A firn aquifer is liquid water stored year-round in the pore space of firn.
They were mapped in southeast Greenland (Forster et al. 2014; Koenig et al.
2014), where heavy snowfall and strong summer melt coincide. Kuipers Munneke
et al. (2014) framed the explanation this experiment tests: an aquifer persists
when accumulation buries summer meltwater below the reach of the winter cold
wave faster than that cold wave can refreeze it.

Scope
-----
One dimension, no lateral transport, and the aquifer is treated as confined:
water leaves the column by refreezing or by riding out of the base with the
compacting firn (the advective flux ``theta * w``; no drainage law). That
is deliberate, not a shortcut waiting to be lifted. Lateral drainage sets the
*equilibrium water-table depth*; it does not decide *survival*. The persistence
question - does liquid water still exist at the end of winter - is a
competition between three rates that are all vertical:

* **recharge**    summer melt percolating down,
* **burial**      accumulation carrying it below the seasonal thermal wave,
* **refreezing**  the winter cold wave consuming it.

Because there is no outlet, the column accumulates water over a long enough
run. That is physical for the years-long integrations here (the pore space is
far from full) but means these experiments answer persistence, not steady-state
water-table depth.

Coupling
--------
Two solvers are stepped in sequence each step:

1. :class:`~firnpack.solvers.firn_solver.FirnColumnSolver` advances
   ``(H, rho, w, sigma, r2, age)``, taking the previous step's refreezing rate
   as the ice-mass source in the density equation (Meyer & Hewitt's
   ``D(rho)/Dt + rho dw/dz = m``). Refrozen water fills pores in place, so
   it raises density and leaves the base-to-surface continuity integration
   with ``rho dw/dz = -compaction``, exactly as in the dry model.
2. :class:`~firnpack.solvers.firn_richards_solver.FirnRichardsSolver` advances
   the pressure head and applies phase change.

The surface energy balance is only half coupled, deliberately and for now.
:class:`~firnpack.surface_energy.SurfaceEnergyBalance` returns a skin
temperature and a conductive flux ``Q_C`` as well as a melt rate, but only the
melt rate is used: the column's enthalpy boundary condition still prescribes
the clamped 2 m air temperature. That is the over-determination
``surface_energy`` was written to remove - prescribing the surface temperature
fixes the conductive flux to whatever the interior gradient implies instead of
letting the atmosphere set it - so closing the loop with a Neumann ``Q_C``
condition is the next step. It is left out here because it changes the thermal
structure of the column, and the thermal structure is what sets how much of
the melt refreezes, so it needs its own validation rather than riding along.

Only the firn solver transports enthalpy; the Richards half touches ``H`` solely
through the latent heat of refreezing. Running both would advect and diffuse
``H`` twice per step.

The firn column is CG1 and the Richards head is DG1 - the discontinuous space
is what makes the upwinded gravity flux and SIPG terms meaningful. CG1 is a
subspace of DG1, so density and enthalpy transfer upward exactly; the enthalpy
*increment* from refreezing is projected back, which preserves its integral
because CG1 contains the constants. The residual energy error that leaves is
reported in the run output rather than assumed small.

Surface forcing
---------------
Air temperature is a seasonal cosine. Melt for the climatological sites is a
raised cosine over a fixed melt season, normalised to a prescribed annual
total and deliberately independent of that temperature, so the cold contrast
case changes cold content alone; :class:`ReanalysisSite` replaces it with melt
from the surface energy balance. The enthalpy boundary condition uses the air
temperature *clamped at the melting point*: energy above 0 C becomes melt, not
superheated firn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

try:
    import firedrake as fd
except Exception:  # pragma: no cover
    fd = None

from firnpack.constants import year as YEAR_S, water_density
from firnpack.models.firn import FirnModel, FirnParameters
from firnpack.models.firn_richards import (
    FirnRichardsModel, FirnRichardsParameters,
)
from firnpack.solvers.firn_solver import FirnColumnSolver
from firnpack.solvers.firn_richards_solver import FirnRichardsSolver
from firnpack.firnmice import (
    FirnState,
    build_stretched_depth_mesh,
    depth_from_mesh,
    make_real,
    make_surface_bcs,
    set_surface_temperature,
    update_surface_velocity_bc,
)

SURFACE_ID, BASE_ID = 2, 1


# ----------------------------------------------------------------------
# Sites
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AquiferSite:
    """One column forcing configuration.

    ``melt_m_we_yr`` is the annual melt total; its seasonal distribution is a
    raised cosine of fixed length, independent of the temperature cycle, so
    lowering ``T_mean_C`` deepens the cold content and changes nothing else
    (see :meth:`melt_flux_m_s`).

    ``accum_m_ie_yr`` is the **net** surface mass balance of the ice matrix,
    i.e. what is left of the snowfall after the melt has been converted to
    water. Melting adds no mass to the column and removes none: it is a phase
    change in the surface skin, so the mass that arrives as snow and leaves
    the matrix as melt is one and the same. The column's total mass input is
    therefore ``accum + melt``, the gross snowfall, and a site quoting
    ``accum_m_ie_yr=1.5`` with ``melt_m_we_yr=0.80`` is a place where about
    2.4 m i.e./yr of snow falls and a third of it melts. Reading ``accum`` as
    gross snowfall instead would count the melted snow twice, once as the snow
    that fell and again as the water it became.
    """

    name: str
    T_mean_C: float
    T_amp_C: float
    accum_m_ie_yr: float
    melt_m_we_yr: float
    melt_season_yr: float = 0.17      # half-width; ~4 months of melt season
    rho_surf_kg_m3: float = 350.0
    label: str = ""

    def air_T_C(self, t_yr: float) -> float:
        """Seasonal air temperature; warmest mid-year, coldest at the turn.

        Takes absolute time and folds it to a phase internally, so a
        climatological site and a reanalysis-driven one present the same
        interface to the driver.
        """
        phase = t_yr % 1.0
        return self.T_mean_C + self.T_amp_C * np.cos(2.0 * np.pi * (phase - 0.5))

    def accum_m_ie_yr_at(self, t_yr: float) -> float:
        """Accumulation rate [m ice-equivalent/yr]; constant for a climatology."""
        return self.accum_m_ie_yr

    def melt_flux_m_s(self, t_yr: float, dt_yr: float | None = None) -> float:
        """Surface meltwater flux [m s^-1], positive into the firn.

        A raised cosine over a melt season of half-width ``melt_season_yr``,
        normalised so the annual integral is ``melt_m_we_yr``.

        Melt is deliberately *not* tied to positive degree days. Doing so ties
        two controls together and breaks the experiment: with a
        degree-day share, lowering ``T_mean_C`` to test cold content also
        collapses the melt season, and at -17 C the air never reaches freezing
        so the cold case gets no melt at all - it would answer "what if there
        were no melt", not "what if the firn were colder". It also concentrated
        the whole annual melt into a few days at 22 mm/day, which is a forcing
        artifact rather than a climate.
        """
        if self.melt_m_we_yr <= 0.0:
            return 0.0
        u = ((t_yr % 1.0) - 0.5) / self.melt_season_yr
        if abs(u) >= 1.0:
            return 0.0
        shape = 0.5 * (1.0 + np.cos(np.pi * u))       # integrates to season_yr
        return self.melt_m_we_yr * shape / (self.melt_season_yr * YEAR_S)


# Southeast Greenland, the Helheim/Kangerlussuaq percolation zone where the
# aquifer was first mapped. The contrast cases each change one control.
AQUIFER_SITES: List[AquiferSite] = [
    AquiferSite("se_greenland", T_mean_C=-7.0, T_amp_C=11.0,
                accum_m_ie_yr=1.5, melt_m_we_yr=0.80,
                label="SE Greenland: high accumulation, high melt"),
    AquiferSite("low_accum", T_mean_C=-7.0, T_amp_C=11.0,
                accum_m_ie_yr=0.25, melt_m_we_yr=0.80,
                label="Low accumulation: burial too slow to outrun the cold wave"),
    AquiferSite("cold", T_mean_C=-17.0, T_amp_C=11.0,
                accum_m_ie_yr=1.5, melt_m_we_yr=0.80,
                label="Cold: same melt, far more cold content to consume it"),
    AquiferSite("low_melt", T_mean_C=-7.0, T_amp_C=11.0,
                accum_m_ie_yr=1.5, melt_m_we_yr=0.15,
                label="Low melt: too little recharge to outlast the winter"),
]


class ReanalysisSite:
    """A column forced by the ERA5 record and the firnpack energy balance.

    Presents the same interface as :class:`AquiferSite` - absolute time in,
    temperature, melt flux and accumulation out - so the driver does not know
    which it is running.

    Melt comes from :mod:`firnpack.surface_energy` rather than from ERA5's own
    ``snowmelt``, which runs about three times low over this belt, and rather
    than from a degree-day factor, which can be tuned to any answer. Melt and
    snowfall are *interval totals*, so the driver's step is integrated over the
    3-hourly record instead of sampling it - sampling a melt series at 3-day
    intervals would miss most of the melt entirely.
    """

    # Southeast Greenland albedo calibration. The ageing-albedo floor is the
    # one free parameter in the melt, and it spans the aquifer outcome: at the
    # library default of 0.60 the column floods, at ERA5's 0.85 it stays dry.
    # The measured recharge into the Helheim aquifer is 9-30 cm/yr (J. Glaciol.
    # hydrologic-modelling study, field data), which at the 56-70% refreezing
    # the column shows implies gross melt of 0.3-0.7 m w.e./yr, centre ~0.45.
    # Mapping the floor through the energy balance on 2006-2019 forcing:
    #   0.60 -> 1.21   0.70 -> 0.72   0.74 -> 0.54   0.76 -> 0.46   0.80 -> 0.32
    # so 0.76 reproduces the observed recharge. A site calibration, kept here
    # with the application rather than in SurfaceEnergyParameters.
    ALBEDO_FIRN_SE_GREENLAND = 0.76

    def __init__(self, csv_path, year0, year1, *, albedo="model",
                 seb_params=None, name=None, label=None, rho_surf_kg_m3=350.0):
        import pandas as pd
        from firnpack.surface_energy import (
            SurfaceEnergyBalance, SurfaceEnergyParameters,
            specific_humidity_from_dewpoint,
        )
        if seb_params is None:
            seb_params = SurfaceEnergyParameters(
                albedo_firn=self.ALBEDO_FIRN_SE_GREENLAND)

        df = pd.read_csv(csv_path, parse_dates=["time"])
        df = df[(df["time"].dt.year >= year0) & (df["time"].dt.year <= year1)]
        df = df.dropna(subset=["t2m", "ssrd", "strd", "sp", "d2m", "u10", "v10"])
        if df.empty:
            raise SystemExit(f"no ERA5 rows for {year0}-{year1} in {csv_path}")
        df = df.sort_values("time").reset_index(drop=True)

        block_s = float(df["sample_hours"].iloc[0]) * 3600.0
        seb = SurfaceEnergyBalance(seb_params)
        if albedo == "era5" and "fal" in df:
            alb = df["fal"].values
        elif albedo == "model":
            # Our own ageing albedo, reset by snowfall. Kept as an option
            # because ERA5's is bright (0.845 mean) for firn that melts every
            # summer, and albedo is the largest single lever on the melt.
            step_days = float(df["sample_hours"].iloc[0]) / 24.0
            since, days = 30.0, np.empty(len(df))
            for i, snow in enumerate(df["sf"].values):
                since = (0.0 if snow >= seb.params.fresh_snow_m_we
                         else since + step_days)
                days[i] = since
            alb = np.asarray(seb.albedo(days), dtype=float)
        else:
            alb = np.full(len(df), float(albedo))
        _, melt, _, _ = seb.solve(
            sw_in=df["ssrd"].values / block_s, lw_in=df["strd"].values / block_s,
            T_air=df["t2m"].values, wind=np.hypot(df["u10"].values, df["v10"].values),
            pressure=df["sp"].values,
            q_air=specific_humidity_from_dewpoint(df["d2m"].values, df["sp"].values),
            T_firn=np.full(len(df), 270.0), k_eff=np.full(len(df), 0.5),
            albedo=alb)

        t0 = df["time"].iloc[0]
        self.t = ((df["time"] - t0).dt.total_seconds() / YEAR_S).values
        self.T_air_C = df["t2m"].values - 273.15
        # cumulative totals let any interval be integrated exactly
        self._melt_cum = np.concatenate([[0.0], np.cumsum(melt * block_s)])
        self._snow_cum = np.concatenate(
            [[0.0], np.cumsum(df["sf"].values)])          # m w.e.
        self._edges = np.concatenate([self.t, [self.t[-1] + block_s / YEAR_S]])
        self.span_years = float(self._edges[-1])

        # The driver seeds the column from a mean and reports an amplitude, so
        # expose both from the record rather than requiring a climatology.
        self.T_mean_C = float(np.mean(self.T_air_C))
        self.T_amp_C = float(0.5 * (np.percentile(self.T_air_C, 99)
                                    - np.percentile(self.T_air_C, 1)))
        self.name = name or f"era5_{year0}_{year1}"
        self.label = label or f"ERA5 + firnpack SEB, {year0}-{year1}"
        self.rho_surf_kg_m3 = rho_surf_kg_m3
        self.year0, self.year1 = year0, year1
        # Net ice-equivalent accumulation for the surface velocity BC. ERA5
        # `sf` is GROSS snowfall, and the melt derived from it is injected
        # into the column as water, so the matrix influx has to be the
        # snowfall that stays snow: the melted fraction leaves the matrix in
        # the surface skin at the moment it becomes water. Net top-boundary
        # mass flux is then the snowfall itself, which is the surface mass
        # balance. Driving the velocity BC with gross snowfall while also
        # injecting the melt counted that mass twice and put 30-50% more into
        # the column than the climate delivers.
        self._ie = water_density / 917.0
        self._net_cum = self._snow_cum - self._melt_cum
        self.accum_m_ie_yr = float(
            self._net_cum[-1] / self.span_years * self._ie)
        self.snowfall_m_ie_yr = float(
            self._snow_cum[-1] / self.span_years * self._ie)
        self.melt_m_we_yr = float(self._melt_cum[-1] / self.span_years)

    def _wrap(self, t_yr):
        """Runs longer than the record repeat it rather than run dry."""
        return float(t_yr) % self.span_years

    def air_T_C(self, t_yr):
        return float(np.interp(self._wrap(t_yr), self.t, self.T_air_C))

    def _interval(self, cum, t_yr, dt_yr):
        """Record total over the ``dt_yr`` window ending at ``t_yr``.

        The window wraps backwards through the record rather than being
        truncated at its start. Truncating returns the right total over the
        wrong window, and the callers divide by the window they asked for: a
        trailing-year accumulation sampled a month past the wrap point would
        report a twelfth of the true burial rate, every lap of the record.
        """
        dt = max(float(dt_yr or 0.0), 0.0)
        if dt <= 0.0:
            return 0.0
        span, total = self.span_years, float(cum[-1])
        laps, rem = divmod(dt, span)
        out = laps * total
        hi = self._wrap(t_yr)
        lo = hi - rem
        out += float(np.interp(hi, self._edges, cum))
        if lo >= 0.0:
            out -= float(np.interp(lo, self._edges, cum))
        else:
            out += total - float(np.interp(span + lo, self._edges, cum))
        return out

    def melt_flux_m_s(self, t_yr, dt_yr=None):
        if not dt_yr:
            return 0.0
        total = self._interval(self._melt_cum, t_yr, dt_yr)
        return total / (dt_yr * YEAR_S)

    def accum_m_ie_yr_at(self, t_yr, window_yr=1.0):
        """Trailing-year NET accumulation rate, so burial follows the record.

        Net of melt (see ``__init__``). The debit is taken on the same
        trailing window as the snowfall rather than instantaneously, which
        makes the mass balance exact over a year while spreading the matrix
        loss evenly through it instead of confining it to the melt season.
        That approximation is deliberate: the instantaneous melt rate peaks
        several times above the snowfall rate, so an instantaneous debit
        would drive the net surface mass balance sharply negative for months,
        and a negative matrix influx is surface lowering, which a fixed-mesh
        Eulerian column cannot represent (its surface density boundary
        condition is an inflow condition). Annual burial, which is what sets
        whether meltwater outruns the cold wave, is unaffected.
        """
        total = self._interval(self._net_cum, t_yr, window_yr)
        return total / window_yr * self._ie


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
def wet_layer(depth_sorted: np.ndarray, theta_sorted: np.ndarray,
              theta_threshold: float) -> tuple[float, float]:
    """Top and bottom depth of the thickest contiguous wet zone, or (nan, nan).

    Contiguity matters: during a melt season a column with a deep aquifer
    carries two disjoint wet zones, a near-surface wet layer and the aquifer
    itself. Reporting the first and last wet node spans both and the dry firn
    between them, which is neither layer.
    """
    wet = np.asarray(theta_sorted) >= theta_threshold
    if not wet.any():
        return float("nan"), float("nan")
    edges = np.flatnonzero(np.diff(np.concatenate([[0], wet.view(np.int8), [0]])))
    starts, ends = edges[::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    return float(depth_sorted[starts[k]]), float(depth_sorted[ends[k] - 1])


def persists(res: Dict[str, Any], last_years: float = 3.0,
             floor_kg_m2: float = 1.0) -> bool:
    """Does liquid water survive every winter of the last ``last_years``?

    The test is not "is there water in summer" - that is just melt. It is
    whether water is still there at the seasonal minimum, which is what
    perennial means.

    Storage is measured *above the retention trace*, not in absolute terms.
    With ``theta_r = 0`` the curve cannot dry below its floor at ``h_min``, so
    even a column that has never seen melt carries tens of kg/m^2 - enough to
    clear any absolute threshold and report a bone-dry column as a perennial
    aquifer. Subtracting the column's own starting trace removes that.
    """
    t = np.asarray(res["time_years"])
    s = np.asarray(res["storage_kg_m2"])
    if t.size == 0:
        return False
    baseline = float(res.get("storage0_kg_m2", 0.0))
    sel = t >= (t[-1] - last_years)
    return bool((s[sel] - baseline).min() > floor_kg_m2)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def run_aquifer_column(
    *,
    site: AquiferSite,
    H0: float = 60.0,
    NZ: int = 160,
    # Near-uniform on purpose. The firn-only drivers stretch hard toward the
    # surface to resolve the thermal wave, but that is fatal here: at NZ=140 on
    # a 60 m column, stretch_p=2 makes the top cell 3.1 mm, which a single
    # 3-day step of peak melt overfills about twenty times over, and the
    # Richards solve fails at any sub-step. p=1.0 and 1.5 both run; 1.0 is the
    # default because percolation cares about the whole column, not just its
    # top.
    stretch_p: float = 1.0,
    spinup_years: float = 60.0,
    spinup_dt_years: float = 0.1,
    run_years: float = 10.0,
    dt_days: float = 2.0,
    save_every_days: float = 10.0,
    theta_threshold: float = 0.02,
    richards_params: "FirnRichardsParameters | None" = None,
    basal_drainage: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Spin a dry column to steady state, then force it with melt.

    The spinup is dry and coarse on purpose: it only has to set the density and
    temperature profile, and with no water the hydrology is a fixed point, so
    running it would cost a solve per step and change nothing.
    """
    assert fd is not None, "firedrake is required"

    mesh, V = build_stretched_depth_mesh(H0, NZ, stretch_p)
    Vd = fd.FunctionSpace(mesh, "DG", 1)
    R = fd.FunctionSpace(mesh, "R", 0)

    params = FirnParameters()
    model = FirnModel(params)
    firn_solver = FirnColumnSolver(model, horizontal_divergence=0.0)

    if richards_params is None:
        # Southeast Greenland calibration. Slug tests and aquifer recovery in
        # the Helheim aquifer give a hydraulic conductivity of 2.7e-4 m/s
        # (geometric mean; range 2.5e-5 to 1.1e-3; Miller et al. 2017,
        # Front. Earth Sci.). Calonne's fit at aquifer depths (rho ~ 550)
        # returns ~3e-3 m/s, about 12x too high, so it is scaled by 0.1 here.
        # This is a site calibration and lives with the case study, not in
        # the library default, which stays the published fit.
        richards_params = FirnRichardsParameters(perm_scale=0.1)
    rmodel = FirnRichardsModel(richards_params)
    rsolver = FirnRichardsSolver(rmodel)

    # --- forcing scalars ---
    Ts = make_real(R, site.T_mean_C + 273.15, "Ts")
    accum = make_real(R, site.accum_m_ie_yr, "accum")
    rho_surf = make_real(R, site.rho_surf_kg_m3, "rho_surf")
    dt = make_real(R, spinup_dt_years * YEAR_S, "dt")
    Hs = make_real(R, params.c_i * (site.T_mean_C + 273.15 - params.T_ref), "Hs")

    # --- CG state (firn) ---
    H = fd.Function(V, name="enthalpy")
    rho = fd.Function(V, name="density")
    w = fd.Function(V, name="firn_velocity")
    sigma = fd.Function(V, name="stress")
    r2 = fd.Function(V, name="grain_radius2")
    age = fd.Function(V, name="age")

    set_surface_temperature(params, Ts, Hs, site.T_mean_C + 273.15)
    H.assign(float(Hs.dat.data_ro[0]))
    x = fd.SpatialCoordinate(mesh)[0]
    depth_expr = H0 - x
    rho.interpolate(
        site.rho_surf_kg_m3
        + (params.rho_i - site.rho_surf_kg_m3) * (1.0 - fd.exp(-depth_expr / 20.0))
    )
    w.assign(-site.accum_m_ie_yr * params.rho_i / site.rho_surf_kg_m3 / YEAR_S)
    sigma.assign(0.0)
    r2.assign(float(getattr(params, "r2_surf", 2.5e-7)))
    age.assign(0.0)
    state = FirnState(H=H, rho=rho, w=w, sigma=sigma, r2=r2, age=age)

    # --- BCs for the firn half ---
    sbc = make_surface_bcs(V, params, accum=accum, rho_surf=rho_surf,
                           Hs_bc=Hs, surface_id=SURFACE_ID)
    bc_sigma = fd.DirichletBC(V, make_real(R, 0.0, "sig_s"), SURFACE_ID)
    bc_r2 = fd.DirichletBC(
        V, make_real(R, float(getattr(params, "r2_surf", 2.5e-7)), "r2_s"), SURFACE_ID)
    bc_age = fd.DirichletBC(V, make_real(R, 0.0, "age_s"), SURFACE_ID)
    bcs_list = [sbc.bc_H, sbc.bc_rho, sbc.bc_w]

    rhoCoef = fd.Function(V, name="rhoCoef")
    kc0 = float(getattr(params, "kc0", 9.2e-9))
    kc1 = float(getattr(params, "kc1", 3.7e-9))
    rho_m = float(getattr(params, "rho_m", 550.0))

    def update_rhoCoef() -> None:
        s = 0.5 * (1.0 + fd.tanh((state.rho - rho_m) / 20.0))
        rhoCoef.interpolate((1.0 - s) * kc0 + s * kc1)

    def firn_step(refreezing=None) -> None:
        update_rhoCoef()
        firn_solver.prognostic_solve(
            enthalpy=state.H, density=state.rho, firn_velocity=state.w, dt=dt,
            accumulation=accum, surface_density=rho_surf,
            boundary_conditions=bcs_list,
            surface_temperature=Ts, enthalpy_bc_constant=Hs,
            stress=state.sigma, grain_radius2=state.r2,
            stress_boundary_condition=bc_sigma,
            grain_radius2_boundary_condition=bc_r2,
            rhoCoef=rhoCoef, age=state.age, age_boundary_condition=bc_age,
            refreezing=refreezing,
        )

    # ------------------------------------------------------------------
    # Dry spinup
    # ------------------------------------------------------------------
    n_spin = int(round(spinup_years / spinup_dt_years))
    for _ in range(n_spin):
        firn_step()
    if verbose:
        d = depth_from_mesh(mesh, H0)
        j = int(np.argmin(np.abs(d - 10.0)))
        print(f"  [{site.name}] spinup {n_spin} steps: "
              f"rho(10 m) = {state.rho.dat.data_ro[j]:.1f} kg/m3")

    # ------------------------------------------------------------------
    # Wet transient
    # ------------------------------------------------------------------
    dt_s = dt_days * 86400.0
    dt.assign(dt_s)
    n_steps = int(round(run_years * YEAR_S / dt_s))
    save_every = max(1, int(round(save_every_days / dt_days)))

    # DG mirrors of the firn state; CG1 is a subspace of DG1 so this is exact
    rho_d = fd.Function(Vd, name="rho_dg")
    H_d = fd.Function(Vd, name="H_dg")
    w_d = fd.Function(Vd, name="w_dg")
    r2_d = fd.Function(Vd, name="r2_dg")
    dH_d = fd.Function(Vd, name="dH_dg")
    dH_c = fd.Function(V, name="dH_cg")
    m_c = fd.Function(V, name="m_cg")

    rho_d.interpolate(state.rho)
    curves = rmodel.curves(Vd, rho_d)
    head = fd.Function(Vd, name="head").interpolate(
        fd.Constant(rmodel.params.h_min))

    depth_nodes = depth_from_mesh(mesh, H0)
    order = np.argsort(depth_nodes)
    depth_sorted = depth_nodes[order]

    # The Richards form is built once and holds these objects, so the melt
    # flux has to be a Constant assigned into each step - a float would be
    # frozen at its initial (midwinter, zero) value and no melt would ever
    # enter the column.
    q_surf = fd.Constant(0.0)
    # The base is either sealed (no outlet: a closed column that must fill and
    # pressurise under any sustained net recharge) or open to the natural
    # advective outflow: water in the pore space leaves with the compacting
    # firn at the velocity the continuity integration already imposes, theta*w,
    # with no separate drainage law. See the 'advect' kind in firn_richards.
    # The aquifer experiments all run with the advective outlet; the sealed
    # base is an alternative outlet, retained and tested, that none of them
    # select.
    base_bc = {"advect": None} if basal_drainage else {"flux": fd.Constant(0.0)}
    # There is no runoff term: water leaving the column across its surface is
    # lateral transport, and a one-dimensional column resolves no lateral
    # exchange. Everything that melts either infiltrates or stays at the
    # surface.
    surf_bc = {"flux": q_surf}
    water_bcs = {BASE_ID: base_bc, SURFACE_ID: surf_bc}

    md = {"quadrature_degree": 3}
    dxq = fd.dx(domain=mesh, metadata=md)
    theta_fn = fd.Function(Vd, name="theta")

    out: Dict[str, List[float]] = {
        "time_years": [], "air_T_C": [], "melt_m_yr": [],
        "storage_kg_m2": [], "wet_top_m": [], "wet_bottom_m": [],
        "refreeze_cum_kg_m2": [], "melt_cum_kg_m2": [], "max_theta": [],
        # Firn air content: the depth-integrated porosity, in metres of air.
        # This is the pore volume available to store meltwater, and the
        # quantity an aquifer draws down as it fills.
        "fac_m": [], "fac_above_wt_m": [], "drain_cum_kg_m2": [],
    }
    # Depth-time snapshots. The whole point of a perennial aquifer is that the
    # wet layer survives the winter, and that is a statement about theta(z, t)
    # - a scalar storage series cannot show whether water persisted at depth or
    # merely returned each summer.
    theta_profiles: List[np.ndarray] = []
    T_profiles: List[np.ndarray] = []
    # Saturation needs the porosity at the same instant, and porosity changes
    # as the column densifies, so S cannot be reconstructed afterwards from
    # theta and a final density profile.
    S_profiles: List[np.ndarray] = []
    rho_profiles: List[np.ndarray] = []
    ablation_steps = 0
    m_prev = None
    refreeze_cum = 0.0
    melt_cum = 0.0
    drain_cum = 0.0
    energy_slip = 0.0
    # The column starts with the trace water the retention curve cannot dry
    # below (theta at h_min); the budget has to start from it, not from zero.
    theta_fn.interpolate(curves.moisture_content(head))
    storage0 = float(fd.assemble(theta_fn * dxq)) * water_density

    for i in range(n_steps):
        t_yr = (i + 1) * dt_s / YEAR_S

        T_air = site.air_T_C(t_yr)
        set_surface_temperature(params, Ts, Hs, min(T_air, 0.0) + 273.15)
        q_melt = site.melt_flux_m_s(t_yr, dt_s / YEAR_S)

        # Accumulation varies year to year under reanalysis forcing, and the
        # surface velocity boundary condition is built from it, so both have to
        # be refreshed - otherwise the column is buried at a constant rate no
        # matter what the record says.
        a_now = site.accum_m_ie_yr_at(t_yr)
        if a_now < 0.0:
            # Net surface mass balance negative: the melt outruns the snowfall
            # over the trailing window, which is surface lowering. A fixed-mesh
            # Eulerian column cannot lower its surface, and its surface density
            # condition is an inflow condition, so the result is not meaningful
            # there. Counted rather than clamped: clamping would quietly put
            # the mass back, which is the double count this formulation exists
            # to remove.
            ablation_steps += 1
        if abs(a_now - float(accum.dat.data_ro[0])) > 1e-12:
            accum.dat.data[:] = a_now
            update_surface_velocity_bc(sbc, params, accum, rho_surf)

        firn_step(refreezing=m_prev)

        # push the updated medium and enthalpy up to DG (exact: CG1 < DG1)
        rho_d.interpolate(state.rho)
        H_d.interpolate(state.H)
        w_d.interpolate(state.w)
        r2_d.interpolate(state.r2)
        H_before = H_d.copy(deepcopy=True)

        q_surf.assign(q_melt)
        try:
            head, m_d = rsolver.step(
                head=head, enthalpy=H_d, density=rho_d, curves=curves, dt=dt_s,
                bcs=water_bcs,
                firn_velocity=w_d, grain_radius2=r2_d, phase_change=True,
            )
        except fd.exceptions.ConvergenceError as err:
            # A failed step is a result too. Attach the state the solve was
            # handed and the record so far, so the caller can see where and
            # in what column the Richards solve broke instead of only that
            # it did.
            theta_fn.interpolate(curves.moisture_content(head))
            err.partial = out
            err.state = {
                "t_yr": t_yr, "melt_m_yr": q_melt * YEAR_S, "air_T_C": T_air,
                "depth_m": depth_sorted.tolist(),
                "head_m": fd.Function(V).project(head).dat.data_ro[order].tolist(),
                "theta": fd.Function(V).project(theta_fn).dat.data_ro[order].tolist(),
                "rho": state.rho.dat.data_ro[order].tolist(),
                "T_C": (state.H.dat.data_ro[order] / params.c_i).tolist(),
                "H": state.H.dat.data_ro[order].tolist(),
                "w_m_s": state.w.dat.data_ro[order].tolist(),
                # the DG fields themselves: a CG projection smears the jumps
                # that are usually the story
                "x_dg": fd.Function(head.function_space()).interpolate(
                    fd.SpatialCoordinate(mesh)[0]).dat.data_ro.tolist(),
                "head_dg": head.dat.data_ro.tolist(),
                "theta_dg": fd.Function(head.function_space()).interpolate(
                    curves.moisture_content(head)).dat.data_ro.tolist(),
                "theta_s_dg": fd.Function(head.function_space()).interpolate(
                    curves.theta_s).dat.data_ro.tolist(),
                "T_dg": fd.Function(head.function_space()).interpolate(
                    H_d / params.c_i).dat.data_ro.tolist(),
                # dof-ordered copies of everything a restart needs
                "restart": {
                    "H0": H0, "NZ": NZ, "stretch_p": stretch_p, "dt_s": dt_s,
                    "x_cg": fd.Function(V).interpolate(
                        fd.SpatialCoordinate(mesh)[0]).dat.data_ro.tolist(),
                    "rho_cg": state.rho.dat.data_ro.tolist(),
                    "r2_cg": state.r2.dat.data_ro.tolist(),
                    "H_cg": state.H.dat.data_ro.tolist(),
                    "w_cg": state.w.dat.data_ro.tolist(),
                    "head_dg": head.dat.data_ro.tolist(),
                    "q_melt_m_s": q_melt,
                    "basal_drainage": basal_drainage,
                    "richards_params": {k: getattr(richards_params, k)
                                        for k in richards_params.__dataclass_fields__},
                },
            }
            raise

        # return only the refreezing increment to CG; L2 projection preserves
        # its integral because CG1 contains the constants
        dH_d.assign(H_d - H_before)
        dH_c.project(dH_d)
        energy_slip += abs(float(fd.assemble((dH_d - dH_c) * rho_d * dxq)))
        state.H.assign(state.H + dH_c)
        m_c.project(m_d)
        m_prev = m_c

        refreeze_cum += float(fd.assemble(m_d * dxq)) * dt_s
        melt_cum += q_melt * dt_s * water_density
        if basal_drainage:
            # the outflow the solve actually applied this step [kg/m^2]
            drain_cum += rsolver.last_outflow_m * water_density

        if (i + 1) % save_every == 0 or i == n_steps - 1:
            theta_fn.interpolate(curves.moisture_content(head))
            theta_nodes = fd.Function(V).project(theta_fn).dat.data_ro[order]
            top, bot = wet_layer(depth_sorted, theta_nodes, theta_threshold)
            out["time_years"].append(t_yr)
            out["air_T_C"].append(T_air)
            out["melt_m_yr"].append(q_melt * YEAR_S)
            out["storage_kg_m2"].append(
                float(fd.assemble(theta_fn * dxq)) * water_density)
            out["wet_top_m"].append(top)
            out["wet_bottom_m"].append(bot)
            out["refreeze_cum_kg_m2"].append(refreeze_cum)
            out["drain_cum_kg_m2"].append(drain_cum)
            out["melt_cum_kg_m2"].append(melt_cum)
            out["max_theta"].append(float(theta_nodes.max()))
            theta_profiles.append(theta_nodes.copy())
            T_profiles.append(
                (state.H.dat.data_ro[order] / params.c_i).copy())
            rho_nodes = state.rho.dat.data_ro[order].copy()
            rho_profiles.append(rho_nodes)
            por = np.maximum(1.0 - rho_nodes / params.rho_i, 1e-6)
            S_profiles.append(np.clip(theta_nodes / por, 0.0, 1.5))
            out["fac_m"].append(float(fd.assemble(
                model.porosity(state.rho) * dxq)))
            # pore space still above the water table, i.e. not yet flooded
            wet = theta_nodes / por >= 0.5
            # depth_sorted ascends, so integrate in that order: reversing
            # both arrays integrates from the base up and flips the sign
            out["fac_above_wt_m"].append(float(np.trapezoid(
                np.where(wet, 0.0, por), depth_sorted)))

    theta_fn.interpolate(curves.moisture_content(head))
    result: Dict[str, Any] = {k: np.asarray(v) for k, v in out.items()}
    result["site"] = site.name
    result["label"] = site.label
    result["depth_m"] = depth_sorted
    result["rho_final"] = state.rho.dat.data_ro[order].copy()
    result["T_final_C"] = (state.H.dat.data_ro[order] / params.c_i).copy()
    result["theta_final"] = fd.Function(V).project(theta_fn).dat.data_ro[order].copy()
    result["head_final"] = fd.Function(V).project(head).dat.data_ro[order].copy()
    result["theta_profiles"] = np.asarray(theta_profiles)
    result["T_profiles"] = np.asarray(T_profiles)
    result["S_profiles"] = np.asarray(S_profiles)
    result["rho_profiles"] = np.asarray(rho_profiles)
    result["storage0_kg_m2"] = storage0
    result["energy_projection_slip_J_m2"] = energy_slip
    # Elastic (specific-storage) water is real storage and has to be counted,
    # or the budget reports a leak that is not there.
    elastic = rsolver.elastic_storage_m * water_density
    result["elastic_storage_kg_m2"] = elastic
    # in = frozen + drained + (stored now - at the start) + elastic
    result["drained_kg_m2"] = drain_cum
    # steps whose trailing-window surface mass balance was negative; nonzero
    # means the column spent time outside the percolation zone this model
    # represents (see the ablation note in the step loop)
    result["ablation_steps"] = ablation_steps
    result["budget_residual_kg_m2"] = (
        melt_cum - refreeze_cum - drain_cum
        - (float(fd.assemble(theta_fn * dxq)) * water_density - storage0)
        - elastic)
    result["persists"] = persists(result)
    return result
