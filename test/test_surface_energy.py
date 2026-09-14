"""Surface energy balance closure, limits and melt magnitude."""
from __future__ import annotations

import numpy as np
import pytest

from firnpack.surface_energy import (
    STEFAN_BOLTZMANN,
    SurfaceEnergyBalance,
    SurfaceEnergyParameters,
    saturation_vapour_pressure,
    specific_humidity_from_dewpoint,
    specific_humidity_saturation,
)

T_M = 273.15


def _forcing(**over):
    """A summer noon over the aquifer AOI unless overridden."""
    f = dict(sw_in=350.0, lw_in=290.0, T_air=274.5, wind=5.0,
             pressure=84000.0, q_air=0.004, T_firn=272.0, k_eff=0.5,
             albedo=0.65)
    f.update(over)
    return f


def test_balance_closes_at_the_solution():
    """Q_net(T_s) must equal Q_C + Q_M - that is what "closes" means."""
    seb = SurfaceEnergyBalance()
    f = _forcing()
    T_s, melt, Q_C, fx = seb.solve(**f)
    Q_net = seb.net_flux(T_s, sw_in=f["sw_in"], lw_in=f["lw_in"],
                         T_air=f["T_air"], wind=f["wind"],
                         pressure=f["pressure"], q_air=f["q_air"],
                         albedo=f["albedo"])
    assert float(Q_net) == pytest.approx(float(Q_C + fx["Q_M"]), rel=1e-6)


def test_cold_night_does_not_melt():
    """No sun, cold air: the surface cools below freezing and melt is zero."""
    seb = SurfaceEnergyBalance()
    T_s, melt, Q_C, fx = seb.solve(**_forcing(sw_in=0.0, lw_in=200.0,
                                              T_air=255.0, T_firn=258.0))
    assert float(T_s) < T_M
    assert float(melt) == 0.0
    assert float(fx["Q_M"]) == 0.0


def test_melting_surface_is_pinned_at_the_melting_point():
    """A surplus cannot raise the surface above 0 C; it becomes melt."""
    seb = SurfaceEnergyBalance()
    T_s, melt, Q_C, fx = seb.solve(**_forcing(sw_in=600.0, lw_in=310.0,
                                              T_air=278.0))
    assert float(T_s) == pytest.approx(T_M)
    assert float(melt) > 0.0
    assert float(fx["Q_M"]) > 0.0


def test_skin_temperature_never_exceeds_the_melting_point():
    """Across a wide forcing sweep, including absurd energy inputs."""
    seb = SurfaceEnergyBalance()
    rng = np.random.default_rng(0)
    n = 400
    T_s, melt, _, _ = seb.solve(
        sw_in=rng.uniform(0, 900, n), lw_in=rng.uniform(150, 350, n),
        T_air=rng.uniform(240, 285, n), wind=rng.uniform(0, 20, n),
        pressure=np.full(n, 84000.0), q_air=rng.uniform(0, 0.006, n),
        T_firn=rng.uniform(240, 273, n), k_eff=np.full(n, 0.5),
        albedo=rng.uniform(0.4, 0.85, n))
    assert np.all(T_s <= T_M + 1e-6)
    assert np.all(melt >= 0.0)


def test_melt_scales_with_absorbed_shortwave():
    """More sun, more melt - and albedo is the lever that controls it."""
    seb = SurfaceEnergyBalance()
    _, melt_bright, _, _ = seb.solve(**_forcing(albedo=0.85))
    _, melt_dark, _, _ = seb.solve(**_forcing(albedo=0.45))
    assert float(melt_dark) > float(melt_bright)


def test_albedo_ages_from_fresh_snow_toward_firn():
    seb = SurfaceEnergyBalance()
    p = seb.params
    assert float(seb.albedo(0.0)) == pytest.approx(p.albedo_fresh)
    assert float(seb.albedo(1e6)) == pytest.approx(p.albedo_firn)
    assert float(seb.albedo(30.0)) < float(seb.albedo(5.0))
    # wet firn is darker, which feeds back onto melt
    assert float(seb.albedo(1e6, wet=True)) < float(seb.albedo(1e6))


def test_stable_stratification_suppresses_turbulent_exchange():
    """Warm air over a melting surface is stable; ignoring that inflates melt.

    This is the standard failure of a neutral bulk scheme over melting snow,
    and it biases melt high in exactly the conditions an aquifer depends on.
    """
    seb = SurfaceEnergyBalance()
    stable = seb._exchange(wind=3.0, T_air=278.0, T_s=T_M, pressure=84000.0)
    neutral = seb._exchange(wind=3.0, T_air=T_M, T_s=T_M, pressure=84000.0)
    unstable = seb._exchange(wind=3.0, T_air=268.0, T_s=T_M, pressure=84000.0)
    assert float(stable) < float(neutral) < float(unstable)


def test_longwave_loss_grows_with_surface_temperature():
    seb = SurfaceEnergyBalance()
    cold = seb.net_longwave(280.0, 250.0)
    warm = seb.net_longwave(280.0, T_M)
    assert float(cold) > float(warm)
    # and matches Stefan-Boltzmann explicitly
    expect = seb.params.emissivity * (280.0 - STEFAN_BOLTZMANN * T_M ** 4)
    assert float(warm) == pytest.approx(expect)


def test_saturation_uses_ice_below_freezing():
    """Over ice the saturation pressure is lower than over water."""
    e_ice = float(saturation_vapour_pressure(263.0))
    e_water_formula = 611.21 * np.exp(17.502 * (263.0 - 273.15) / (263.0 - 32.19))
    assert e_ice < e_water_formula
    assert float(saturation_vapour_pressure(280.0)) > float(
        saturation_vapour_pressure(270.0))


def test_dewpoint_humidity_is_consistent_with_saturation():
    """q from a dewpoint equals saturation q at that temperature."""
    p = 84000.0
    for T_d in (250.0, 265.0, 272.0):
        assert float(specific_humidity_from_dewpoint(T_d, p)) == pytest.approx(
            float(specific_humidity_saturation(T_d, p)), rel=1e-12)


def test_annual_melt_is_physically_plausible_for_the_aoi():
    """A crude seasonal cycle must give melt of the right order.

    This is the check the degree-day scheme could not provide: the answer
    follows from the energy balance rather than from a factor chosen between
    3 and 6 mm/C/day, which spanned 0.26 to 0.53 m w.e./yr and with it the
    entire aquifer outcome.
    """
    seb = SurfaceEnergyBalance()
    hours = np.arange(365 * 24)
    doy = hours / 24.0
    # seasonal + diurnal temperature, AOI-like: mean about -10 C
    T_air = (273.15 - 10.0 + 11.0 * np.cos(2 * np.pi * (doy - 195) / 365.0)
             + 2.0 * np.cos(2 * np.pi * (hours % 24 - 14) / 24.0))
    sun = np.maximum(np.cos(2 * np.pi * (hours % 24 - 13) / 24.0), 0.0)
    season = np.maximum(np.cos(2 * np.pi * (doy - 172) / 365.0), 0.0)
    sw_in = 750.0 * sun * season
    lw_in = 250.0 + 1.2 * (T_air - 263.15)

    _, melt, _, _ = seb.solve(
        sw_in=sw_in, lw_in=lw_in, T_air=T_air, wind=np.full_like(T_air, 4.0),
        pressure=np.full_like(T_air, 84000.0),
        q_air=specific_humidity_saturation(T_air - 3.0, 84000.0),
        T_firn=np.full_like(T_air, 270.0), k_eff=np.full_like(T_air, 0.5),
        albedo=np.full_like(T_air, 0.7))
    annual = float(melt.sum() * 3600.0)     # m w.e. per year
    assert 0.05 < annual < 3.0, f"{annual:.3f} m w.e./yr is not plausible"
