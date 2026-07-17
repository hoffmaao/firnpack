"""Unit tests for firnpack.inverse.diagnostics.

These are pure numpy -- no Firedrake, no solve -- so they run in the default
tier in milliseconds and pin the residual-analysis formulas the tutorial
diagnostics used to reimplement by hand.
"""
from __future__ import annotations

import numpy as np
import pytest

from firnpack.inverse import diagnostics as diag


def _result(blocks):
    """Wrap block arrays into a forward-result-shaped dict."""
    return {"J": None, "obs": blocks}


def test_block_residuals_are_standardized():
    blocks = [dict(label="rho", depths=[10, 20], obs=[600.0, 700.0],
                   sig=[10.0, 20.0], pred=[610.0, 660.0])]
    res = diag.block_residuals(_result(blocks))
    # (610-600)/10 = 1.0 ; (660-700)/20 = -2.0
    assert np.allclose(res["rho"], [1.0, -2.0])


def test_block_residuals_accepts_bare_obs_list():
    blocks = [dict(label="v", depths=[5], obs=[0.0], sig=[2.0], pred=[3.0])]
    assert np.allclose(diag.block_residuals(blocks)["v"], [1.5])


def test_j_budget_chi2_and_shares():
    # rho: residuals [1, -1] -> chi2 = 0.5*(1+1) = 1.0
    # dage: residuals [2]     -> chi2 = 0.5*4     = 2.0
    blocks = [
        dict(label="rho", depths=[1, 2], obs=[0.0, 0.0], sig=[1.0, 1.0], pred=[1.0, -1.0]),
        dict(label="dage", depths=[3], obs=[0.0], sig=[1.0], pred=[2.0]),
    ]
    r = dict(J=3.0, obs=blocks)  # J = J_obs here (no priors)
    b = diag.j_budget(r)

    assert b["J_obs"] == pytest.approx(3.0)
    assert b["J_prior"] == pytest.approx(0.0)
    # sorted by descending chi2: dage (2.0) before rho (1.0)
    assert [d["label"] for d in b["blocks"]] == ["dage", "rho"]
    dage, rho = b["blocks"]
    assert dage["chi2"] == pytest.approx(2.0)
    assert rho["chi2"] == pytest.approx(1.0)
    assert dage["frac_J"] == pytest.approx(2.0 / 3.0)
    assert rho["n"] == 2 and rho["rms"] == pytest.approx(1.0)


def test_j_budget_separates_prior_from_obs():
    blocks = [dict(label="rho", depths=[1], obs=[0.0], sig=[1.0], pred=[2.0])]  # chi2 = 2
    b = diag.j_budget(dict(J=5.0, obs=blocks))
    assert b["J_obs"] == pytest.approx(2.0)
    assert b["J_prior"] == pytest.approx(3.0)  # 5 total - 2 obs


def test_implied_sigma_repr_zero_when_measurement_explains_scatter():
    # residual scatter exactly matches the measurement sigma -> no extra needed
    res = np.array([2.0, -2.0, 2.0, -2.0])
    assert diag.implied_sigma_repr(res, sigma_meas=np.full(4, 2.0)) == pytest.approx(0.0)


def test_implied_sigma_repr_quadrature():
    # <res^2> = 25, <sig_meas^2> = 9 -> sqrt(16) = 4
    res = np.full(10, 5.0)
    assert diag.implied_sigma_repr(res, sigma_meas=np.full(10, 3.0)) == pytest.approx(4.0)


def test_implied_sigma_repr_floors_at_zero():
    # measurement sigma larger than the scatter -> clipped to 0, never NaN
    res = np.full(5, 1.0)
    assert diag.implied_sigma_repr(res, sigma_meas=np.full(5, 3.0)) == 0.0


def test_residual_acf_white_noise_decorrelates():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(4000)
    acf = diag.residual_acf(r, max_lag=4)
    assert acf[0] == pytest.approx(1.0)
    assert np.all(np.abs(acf[1:]) < 0.1)  # white -> small at all nonzero lags


def test_residual_acf_detects_correlation():
    # a smooth ramp is strongly autocorrelated at lag 1
    r = np.linspace(-1, 1, 200)
    acf = diag.residual_acf(r, max_lag=1)
    assert acf[1] > 0.9


def test_common_mode_decomposition_pure_common():
    # every site shows the same residual -> all variance is common-mode
    common = np.array([0.5, -0.5, 1.0])
    Rm = np.vstack([common, common, common])
    d = diag.common_mode_decomposition(Rm)
    assert np.allclose(d["common"], common)
    assert d["var_site"] == pytest.approx(0.0)
    assert d["frac_common"] == pytest.approx(1.0)


def test_common_mode_decomposition_pure_site():
    # zero-mean-across-sites at each depth -> no common mode, all site-specific
    Rm = np.array([[1.0, -1.0], [-1.0, 1.0]])
    d = diag.common_mode_decomposition(Rm)
    assert np.allclose(d["common"], [0.0, 0.0])
    assert d["var_common"] == pytest.approx(0.0)
    assert d["frac_common"] == pytest.approx(0.0)
