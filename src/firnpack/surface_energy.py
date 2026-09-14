"""Surface energy balance: the melt flux and the surface heat flux.

Why this exists
---------------
The melt forcing was a positive-degree-day rule, and the degree-day factor
spanned the answer: over the aquifer AOI the same hourly ERA5 gives 0.26
m w.e./yr at 3 mm/C/day and 0.53 at 6, which is the difference between a column
that refreezes everything and one that carries a perennial aquifer. A
degree-day factor is a fitted stand-in for an energy balance, so with the
forcing in hand it is better to close the balance directly and let the melt
follow from it.

Doing so also removes an inconsistency. The firn column solves an *enthalpy*
equation whose surface condition was a prescribed temperature. Prescribing the
surface temperature over-determines the surface: it fixes the conductive flux
into the firn to whatever the interior gradient happens to imply, rather than
letting the atmosphere set it. The balance below returns that conductive flux,
so the enthalpy equation can take a Neumann condition instead and the two
halves agree on how much energy crossed the surface.

That coupling is not wired yet at the driver level: :mod:`firnpack.aquifer`
consumes only the melt rate this returns and still prescribes the surface
temperature. ``Q_C`` and ``T_s`` are returned, and correct, and unused.

The balance
-----------
Positive downward, all in W m^-2::

    Q_net(T_s) = SW_in (1 - alpha) + eps (LW_in - sigma T_s^4)
                 + Q_H(T_s) + Q_L(T_s) + Q_R

    Q_net(T_s) = Q_C(T_s) + Q_M

``Q_C`` is conduction into the firn and ``Q_M`` the energy consumed by melt.
The surface cannot exceed the melting point, which gives the two cases:

* **Cold surface.** ``Q_M = 0`` and the skin temperature is the root of
  ``Q_net(T_s) - Q_C(T_s) = 0`` with ``T_s < T_m``.
* **Melting surface.** If that root would exceed ``T_m``, the surface is pinned
  at ``T_m`` and the surplus becomes melt,
  ``Q_M = Q_net(T_m) - Q_C(T_m) >= 0``.

This is the standard skin-temperature formulation. It is deliberately not a
sub-surface-absorption scheme: shortwave is absorbed entirely at the surface,
which over-predicts melt slightly and under-predicts sub-surface warming. That
matters for the very top of the column and much less for water arriving at an
aquifer 10-30 m down.

Everything here is plain NumPy on scalars or arrays. It is forcing-side code
that runs once per step to produce two numbers - a melt flux and a heat flux -
and it does not belong on the Firedrake tape.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from firnpack.constants import (
    latent_heat, water_density, ice_density, heat_capacity,
    melting_temperature, gravity,
)

STEFAN_BOLTZMANN = 5.670374419e-8      # W m^-2 K^-4
LATENT_SUBLIMATION = 2.834e6           # J kg^-1
LATENT_VAPORISATION = 2.501e6          # J kg^-1
R_DRY = 287.058                        # J kg^-1 K^-1
KARMAN = 0.4


@dataclass
class SurfaceEnergyParameters:
    """Surface properties and turbulent-exchange settings."""

    # --- radiation ---------------------------------------------------------
    albedo_fresh: float = 0.85     # fresh dry snow
    albedo_firn: float = 0.60      # aged/wet firn
    albedo_ice: float = 0.45
    albedo_decay_days: float = 15.0   # e-folding of the ageing after snowfall
    fresh_snow_m_we: float = 0.005    # snowfall that resets the albedo
    emissivity: float = 0.98

    # --- turbulent exchange ------------------------------------------------
    # Neutral bulk aerodynamic transfer with a stability correction. z0 for
    # snow and firn is 1e-4 to 1e-3 m; the larger end suits the wind-worked
    # surfaces of southeast Greenland.
    z0_m: float = 1.0e-3           # roughness length for momentum [m]
    z_obs_m: float = 2.0           # temperature/humidity reference height [m]
    z_wind_m: float = 10.0         # wind reference height [m]
    stability_max: float = 5.0     # cap on the Richardson-number correction
    wind_min_m_s: float = 1.0      # floor: never a perfectly calm surface

    # --- thermodynamics ----------------------------------------------------
    c_air: float = 1005.0          # heat capacity of air [J kg^-1 K^-1]
    T_m: float = melting_temperature
    L_f: float = latent_heat
    rho_w: float = water_density
    rho_i: float = ice_density
    c_i: float = heat_capacity

    # --- conduction --------------------------------------------------------
    # Thickness over which the surface-to-firn temperature gradient is taken.
    # This should be the top cell of the column when coupled.
    dz_conduct_m: float = 0.5


class SurfaceEnergyBalance:
    """Closes the surface energy balance and returns melt and heat fluxes."""

    def __init__(self, params: SurfaceEnergyParameters | None = None):
        self.params = params or SurfaceEnergyParameters()

    # ------------------------------------------------------------------
    # Radiation
    # ------------------------------------------------------------------
    def albedo(self, days_since_snowfall, wet=False):
        """Exponential ageing from fresh snow toward firn.

        Wet firn is darker, which is a positive feedback on an aquifer: melt
        lowers the albedo, which raises the melt.
        """
        p = self.params
        floor = p.albedo_firn if not wet else 0.5 * (p.albedo_firn + p.albedo_ice)
        decay = np.exp(-np.maximum(days_since_snowfall, 0.0) / p.albedo_decay_days)
        return floor + (p.albedo_fresh - floor) * decay

    def net_shortwave(self, sw_in, albedo):
        return sw_in * (1.0 - albedo)

    def net_longwave(self, lw_in, T_s):
        p = self.params
        return p.emissivity * (lw_in - STEFAN_BOLTZMANN * T_s ** 4)

    # ------------------------------------------------------------------
    # Turbulent fluxes
    # ------------------------------------------------------------------
    def _exchange(self, wind, T_air, T_s, pressure):
        """Bulk transfer coefficient with a bulk-Richardson stability factor.

        A stable surface layer - which is the normal state over melting snow,
        where the air is warmer than the surface - suppresses turbulent
        exchange. Ignoring that overestimates the turbulent heat supply and
        hence the melt.
        """
        p = self.params
        u = np.maximum(wind, p.wind_min_m_s)
        C_n = KARMAN ** 2 / (np.log(p.z_obs_m / p.z0_m)
                             * np.log(p.z_wind_m / p.z0_m))
        Ri = (gravity * (T_air - T_s) * p.z_obs_m) / (T_air * u ** 2 + 1e-12)
        Ri = np.clip(Ri, -p.stability_max, 0.20)
        # np.where evaluates BOTH branches, so the unstable expression is
        # computed at stable points too, where 1 - 16 Ri goes negative and the
        # square root returns NaN. The result is discarded, but a NaN that only
        # survives because of branch selection is one refactor away from
        # escaping - so clamp the argument. Where the branch is actually taken
        # (Ri < 0) the clamp is inactive, since 1 - 16 Ri > 1 there.
        phi_stable = np.maximum(1.0 - 5.0 * Ri, 0.05) ** 2
        phi_unstable = np.sqrt(np.maximum(1.0 - 16.0 * Ri, 1.0))
        phi = np.where(Ri >= 0.0, phi_stable, phi_unstable)
        rho_air = pressure / (R_DRY * T_air)
        return C_n * phi * u * rho_air

    def sensible_heat(self, wind, T_air, T_s, pressure):
        p = self.params
        return self._exchange(wind, T_air, T_s, pressure) * p.c_air * (T_air - T_s)

    def latent_heat_flux(self, wind, T_air, T_s, pressure, q_air):
        """Sublimation/condensation. Uses the sublimation latent heat, since a
        snow or firn surface is ice even when it is at the melting point."""
        p = self.params
        q_s = specific_humidity_saturation(T_s, pressure)
        L = LATENT_SUBLIMATION
        return self._exchange(wind, T_air, T_s, pressure) * L * (q_air - q_s)

    # ------------------------------------------------------------------
    def conductive_flux(self, T_s, T_firn, k_eff):
        """Conduction from the surface into the firn, positive downward."""
        return k_eff * (T_s - T_firn) / self.params.dz_conduct_m

    # ------------------------------------------------------------------
    def net_flux(self, T_s, *, sw_in, lw_in, T_air, wind, pressure, q_air,
                 albedo, rain_heat=0.0):
        """Atmospheric energy delivered to the surface at skin temperature T_s."""
        return (self.net_shortwave(sw_in, albedo)
                + self.net_longwave(lw_in, T_s)
                + self.sensible_heat(wind, T_air, T_s, pressure)
                + self.latent_heat_flux(wind, T_air, T_s, pressure, q_air)
                + rain_heat)

    def solve(self, *, sw_in, lw_in, T_air, wind, pressure, q_air, T_firn,
              k_eff, albedo, rain_heat=0.0, tol=1e-6, max_iter=60):
        """Close the balance.

        Returns ``(T_s, melt_flux_m_we_s, Q_C, fluxes)`` where ``Q_C`` is the
        conductive flux into the firn (positive downward, i.e. warming it) and
        ``fluxes`` is a dict of the individual terms at the solution.

        The residual ``Q_net(T_s) - Q_C(T_s)`` is monotonically decreasing in
        ``T_s`` - raising the surface increases its outgoing longwave and its
        conduction into the firn while reducing the sensible gain - so a
        bisection is unconditionally convergent and needs no derivative.
        """
        p = self.params

        def residual(T):
            return (self.net_flux(T, sw_in=sw_in, lw_in=lw_in, T_air=T_air,
                                  wind=wind, pressure=pressure, q_air=q_air,
                                  albedo=albedo, rain_heat=rain_heat)
                    - self.conductive_flux(T, T_firn, k_eff))

        lo = np.full_like(np.asarray(T_air, dtype=float), 180.0)
        hi = np.full_like(lo, p.T_m)
        r_hi = residual(hi)

        # Surplus at the melting point => the surface is melting.
        melting = r_hi > 0.0

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            r = residual(mid)
            lo = np.where(r > 0.0, mid, lo)
            hi = np.where(r > 0.0, hi, mid)
            if np.all(hi - lo < tol):
                break
        T_s = np.where(melting, p.T_m, 0.5 * (lo + hi))

        Q_C = self.conductive_flux(T_s, T_firn, k_eff)
        Q_M = np.where(melting, np.maximum(r_hi, 0.0), 0.0)
        melt = Q_M / (p.rho_w * p.L_f)          # m w.e. per second

        fluxes = dict(
            SW_net=self.net_shortwave(sw_in, albedo),
            LW_net=self.net_longwave(lw_in, T_s),
            Q_H=self.sensible_heat(wind, T_air, T_s, pressure),
            Q_L=self.latent_heat_flux(wind, T_air, T_s, pressure, q_air),
            Q_C=Q_C, Q_M=Q_M, T_s=T_s,
        )
        return T_s, melt, Q_C, fluxes


# ----------------------------------------------------------------------
# Humidity helpers
# ----------------------------------------------------------------------
def saturation_vapour_pressure(T):
    """Over ice below the melting point, over water above (Pa).

    The distinction matters: over ice the saturation pressure is lower, so a
    given air humidity is closer to saturation and sublimation is weaker.
    """
    T = np.asarray(T, dtype=float)
    over_ice = 611.21 * np.exp(22.587 * (T - 273.15) / (T + 0.71))
    over_water = 611.21 * np.exp(17.502 * (T - 273.15) / (T - 32.19))
    return np.where(T < melting_temperature, over_ice, over_water)


def specific_humidity_saturation(T, pressure):
    e = saturation_vapour_pressure(T)
    return 0.622 * e / np.maximum(pressure - 0.378 * e, 1.0)


def specific_humidity_from_dewpoint(T_dew, pressure):
    e = saturation_vapour_pressure(T_dew)
    return 0.622 * e / np.maximum(pressure - 0.378 * e, 1.0)
