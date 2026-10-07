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

import warnings

import numpy as np

try:
    import firedrake as fd
except Exception:  # pragma: no cover
    fd = None

from firnpack.constants import year as YEAR_S, ice_density, water_density
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
        """NET accumulation rate [m i.e./yr]; constant for a climatology."""
        return self.accum_m_ie_yr

    @property
    def snowfall_m_ie_yr(self) -> float:
        """GROSS snowfall rate [m i.e./yr], a diagnostic.

        ``accum_m_ie_yr`` is net of melt, so the gross total is that plus the
        melt in ice equivalent. The two are not interchangeable: the net is
        the matrix influx that buries and loads the firn, and the column is
        driven by it alone; the gross is reported for the mass budget.
        """
        return (self.accum_m_ie_yr
                + self.melt_m_we_yr * water_density / ice_density)

    def snowfall_m_ie_yr_at(self, t_yr: float) -> float:
        """GROSS snowfall rate [m i.e./yr]; constant for a climatology."""
        return self.snowfall_m_ie_yr

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

    # Southeast Greenland ageing-albedo floor. It is the one free parameter in
    # the melt and it spans the aquifer outcome - at the library default of
    # 0.60 the column floods, at ERA5's 0.85 it stays dry - so it is swept
    # across the experiments in tutorials/aquifer/config.py rather than fitted:
    # calibrating the largest free knob against the runs it controls would be
    # circular. Reference table for choosing sweep points, gross melt in
    # m w.e./yr over the 14 complete years 2006-2019 of the 83-year record -
    # the authoritative copy, which the tutorial config and data notes cite:
    #   0.60 -> 1.20   0.70 -> 0.71   0.72 -> 0.62   0.74 -> 0.54   0.76 -> 0.46
    # This constant is one of the swept floors, and the one a caller who
    # passes no seb_params gets; it is not a fitted value.
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
        if df.empty:
            raise ValueError(f"no ERA5 rows for {year0}-{year1} in {csv_path}")
        df = df.dropna(
            subset=["t2m", "ssrd", "strd", "sp", "d2m", "u10", "v10", "sf"])
        if df.empty:
            raise ValueError(
                f"no complete ERA5 rows for {year0}-{year1} in {csv_path}: "
                f"every row in the window is missing at least one of "
                f"t2m, ssrd, strd, sp, d2m, u10, v10, sf")
        df = df.sort_values("time").reset_index(drop=True)

        block_s = float(df["sample_hours"].iloc[0]) * 3600.0
        seb = SurfaceEnergyBalance(seb_params)
        if albedo not in ("model", "era5"):
            raise ValueError(
                f"albedo must be 'model' or 'era5', not {albedo!r}; the floor "
                f"is swept through seb_params.albedo_firn")
        if albedo == "era5":
            if "fal" not in df:
                raise ValueError(f"albedo='era5' needs a 'fal' column in {csv_path}")
            alb = df["fal"].values
        else:
            # Our own ageing albedo, reset by snowfall. Kept as an option
            # because ERA5's is bright (0.845 mean) for firn that melts every
            # summer, and albedo is the largest single lever on the melt.
            # Deliberately duplicated as `model_albedo` in
            # tutorials/aquifer/diagnostics/seb_forcing.py, which is a
            # diagnostic that must keep running without the column. The two
            # must agree or the audit table and the runs disagree about the
            # forcing while both still look right: change one, change both.
            step_days = float(df["sample_hours"].iloc[0]) / 24.0
            since, days = 30.0, np.empty(len(df))
            for i, snow in enumerate(df["sf"].values):
                since = (0.0 if snow >= seb.params.fresh_snow_m_we
                         else since + step_days)
                days[i] = since
            alb = np.asarray(seb.albedo(days), dtype=float)
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
        # The record has download gaps of two shapes: a missing year has no
        # rows at all, and a present year can carry a few blank cells where
        # one stream was short. The dropna above turns the second shape into
        # the first, so either way the retained blocks can cover less than
        # the wall-clock span. Rates are per unit time actually observed:
        # dividing a partial total by the full span reports every rate low by
        # the missing fraction, silently. `span_years` stays the wall-clock
        # span because it is the period the time axis wraps on.
        self.covered_years = float(len(df) * block_s / YEAR_S)
        self.coverage = self.covered_years / self.span_years
        if self.coverage < 1.0 - 1e-9:
            warnings.warn(
                f"ERA5 window {year0}-{year1} of {csv_path} covers "
                f"{100 * self.coverage:.1f}% of its {self.span_years:.1f}-year "
                f"span; rates are taken over the {self.covered_years:.1f} "
                f"years present, but the gaps still distort the trailing "
                f"windows that cross them",
                stacklevel=2)

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
        # ice_density must be the same rho_i the surface velocity BC uses
        # (FirnParameters.rho_i): the mass-neutrality invariant is stated in
        # terms of it, as accum_m_ie_yr * rho_i + melt_m_we_yr * rho_w == snow.
        self._ie = water_density / ice_density
        self._net_cum = self._snow_cum - self._melt_cum
        self.accum_m_ie_yr = float(
            self._net_cum[-1] / self.covered_years * self._ie)
        self.snowfall_m_ie_yr = float(
            self._snow_cum[-1] / self.covered_years * self._ie)
        self.melt_m_we_yr = float(self._melt_cum[-1] / self.covered_years)

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

    def snowfall_m_ie_yr_at(self, t_yr, window_yr=1.0):
        """Trailing-window GROSS snowfall rate [m i.e./yr], a diagnostic.

        The gross counterpart of :meth:`accum_m_ie_yr_at`, on the same
        trailing window. Not interchangeable with it: the net is the matrix
        influx that buries and loads the firn; the gross is not used to drive
        the column.
        """
        total = self._interval(self._snow_cum, t_yr, window_yr)
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

    This is the rule for reporting the *extent* of a wet zone. The water table
    itself is :func:`water_table_depth`, which takes the run that reaches the
    base instead.
    """
    starts, ends = _wet_runs(theta_sorted, theta_threshold)
    if starts is None:
        return float("nan"), float("nan")
    k = int(np.argmax(ends - starts))
    return float(depth_sorted[starts[k]]), float(depth_sorted[ends[k] - 1])


def _wet_runs(values_sorted: np.ndarray, threshold: float):
    wet = np.asarray(values_sorted) >= threshold
    if not wet.any():
        return None, None
    edges = np.flatnonzero(np.diff(np.concatenate([[0], wet.view(np.int8), [0]])))
    return edges[::2], edges[1::2]


def water_table_depth(depth_sorted: np.ndarray, theta_sorted: np.ndarray,
                      theta_threshold: float) -> float:
    """Top of the saturated run that reaches the base, or nan if none does.

    The aquifer here is by construction the saturated zone standing on the
    confining base: the column is confined and fills from the bottom upward,
    so water that has reached the water table cannot leave except by refreezing
    or by riding out with the firn. That makes "contains the deepest cell" the
    definition rather than a heuristic, and it decides both cases a rule has to
    get right. A perched melt lens with dry firn beneath it does not touch the
    base, so it is not an aquifer and there is no water table: the function
    returns nan and :func:`run_aquifer_column` integrates the whole profile as
    unflooded, which is the right answer when nothing is. A lens above a real
    aquifer is ignored, whether or not it is the thicker of the two.

    Selecting the thickest run instead - the right rule for :func:`wet_layer`,
    where the question is the extent of a zone rather than where its top is -
    reports the lens as the water table whenever it is the thicker. Selecting
    merely the deepest run shares that fix but not the first one: with a single
    perched run, deepest and thickest are the same run, so it still reports a
    lens as a water table.

    The assumption is the confining base. A column allowed to drain freely at
    the base can hold a water table that does not reach the bottom of the
    domain, and would need a different rule.

    Deliberately duplicated as ``_aquifer_zone`` in
    ``tutorials/aquifer/plot_fac_saturation.py``, which applies the same rule
    and also returns the bottom. They cannot share code: that script is
    specified to run without Firedrake, and importing this module pulls in
    ``firnpack.models.firn``, which imports Firedrake unconditionally. Change
    one and change the other.
    """
    wet = np.asarray(theta_sorted) >= theta_threshold
    if not wet.size or not wet[-1]:
        return float("nan")
    starts, _ = _wet_runs(theta_sorted, theta_threshold)
    return float(depth_sorted[starts[-1]])


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
        # Unscaled Calonne, the library default, so every experiment run by
        # this driver uses the same conductivity. This was a whole-column
        # perm_scale=0.1, from slug tests and aquifer recovery in the Helheim
        # aquifer giving 2.7e-4 m/s (geometric mean; range 2.5e-5 to 1.1e-3;
        # Miller et al. 2017, Front. Earth Sci.) against Calonne's ~3e-3 m/s.
        # But that measurement was made *inside* the aquifer, at 550-650
        # kg/m3, and applying it uniformly extends it far outside the density
        # range it was measured in. perm_scale_deep applies it only where it
        # belongs, and the `deep0.1` experiment is the variant that does.
        richards_params = FirnRichardsParameters()
    rmodel = FirnRichardsModel(richards_params)
    rsolver = FirnRichardsSolver(rmodel)

    # --- forcing scalars ---
    Ts = make_real(R, site.T_mean_C + 273.15, "Ts")
    # One surface mass flux, `accum_net`: the snowfall that stays matrix,
    # gross minus melt. It drives the surface velocity boundary condition,
    # and it is also what `bdot` in the overburden stress must be.
    #
    # Gross was tried for the loading, on the argument that with no runoff
    # every kilogram that falls stays in the column and loads the firn below.
    # That argument is right about the mass and wrong about this stress model.
    # Mode B integrates dsigma/dt + w dsigma/dz = bdot g with sigma = 0 at the
    # surface, so along a particle path sigma = bdot g age, which equals the
    # true overburden integral(rho g dz) only when bdot is the matrix influx
    # that produced that age. Using gross asserts that all the melt has
    # already refrozen ABOVE every parcel - true only below the whole
    # refreezing zone, and badly wrong near the surface, which is exactly
    # where densification is fastest. Measured: the low-accumulation case
    # reached bubble close-off at 1.7 m and 77% of the column saturated
    # before failing at year 10.5, and the ERA5 columns pressurised the same
    # way. Net is exact above the refreezing zone and understates below it by
    # the refrozen mass there, which is the error this keeps.
    #
    # The exact overburden is the density integral, and rho already carries
    # the refrozen mass (the refreezing source puts it there). Switching Mode
    # B's stress from the bdot proxy to `FirnModel.overburden_stress`, which
    # Mode A already uses, would remove the approximation entirely; it is left
    # alone here because it changes every densification case in the repo.
    # Seeded from the record mean, not a trailing window: the spinup is a
    # climatology, so its forcing has to be one too.
    accum_net = make_real(R, site.accum_m_ie_yr, "accum_net")
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
    # seeded from the same net matrix influx the boundary condition imposes
    w.assign(-site.accum_m_ie_yr * params.rho_i / site.rho_surf_kg_m3
             / YEAR_S)
    sigma.assign(0.0)
    r2.assign(float(getattr(params, "r2_surf", 2.5e-7)))
    age.assign(0.0)
    state = FirnState(H=H, rho=rho, w=w, sigma=sigma, r2=r2, age=age)

    # --- BCs for the firn half ---
    # One accumulation throughout, the net matrix influx, for the velocity
    # boundary condition and for the loading alike. The dry spinup is then
    # the dry analogue of the wet column's matrix dynamics, and the column is
    # fed the same thing before and after the transient begins.
    sbc = make_surface_bcs(V, params, accum=accum_net, rho_surf=rho_surf,
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
            # net: `accumulation` forms bdot, which must match the matrix
            # influx driving the velocity (see the note where accum_net is
            # built); gross overstates the load near the surface
            accumulation=accum_net, surface_density=rho_surf,
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
    # Melt water now arrives separately, through the Richards surface flux;
    # the matrix influx was already the net through the spinup, so this only
    # re-applies the velocity BC from accum_net before the transient starts.
    update_surface_velocity_bc(sbc, params, accum_net, rho_surf)

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
        if abs(a_now - float(accum_net.dat.data_ro[0])) > 1e-12:
            accum_net.dat.data[:] = a_now
            update_surface_velocity_bc(sbc, params, accum_net, rho_surf)

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
            sat = np.clip(theta_nodes / por, 0.0, 1.5)
            S_profiles.append(sat)
            out["fac_m"].append(float(fd.assemble(
                model.porosity(state.rho) * dxq)))
            # Pore space above the water table. Masking every unsaturated
            # node instead would also subtract a near-surface melt-season wet
            # layer sitting well above the aquifer, which is the drawdown this
            # series exists to isolate. Until saturation reaches the confining
            # base there is no aquifer and no water table, so the whole profile
            # is above it.
            wt = water_table_depth(depth_sorted, sat, 0.5)
            above = (np.ones_like(por, dtype=bool) if np.isnan(wt)
                     else depth_sorted < wt)
            # depth_sorted ascends, so integrate in that order: reversing
            # both arrays integrates from the base up and flips the sign
            out["fac_above_wt_m"].append(float(np.trapezoid(
                np.where(above, por, 0.0), depth_sorted)))

    theta_fn.interpolate(curves.moisture_content(head))
    result: Dict[str, Any] = {k: np.asarray(v) for k, v in out.items()}
    result["site"] = site.name
    result["label"] = site.label
    result["depth_m"] = depth_sorted
    result["rho_final"] = state.rho.dat.data_ro[order].copy()
    # The prognostic overburden. Reported because it is the quantity that
    # silently stopped matching the density profile when the loading rate and
    # the velocity field were fed different accumulations: sigma = bdot g age
    # is only the true integral(rho g dz) when the two agree.
    result["sigma_final_Pa"] = state.sigma.dat.data_ro[order].copy()
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
