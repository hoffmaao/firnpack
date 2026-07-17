"""firnpack.inverse.diagnostics — analysis of an assimilation result.

Pure-numpy functions that operate on a forward-evaluation result -- the dict
``assimilate(cfg, mode="forward")`` returns, whose ``obs`` field is a list of
per-block dicts ``{kind, label, depths, obs, sig, pred}``. They compute the
residual structure the tutorial diagnostics reimplemented by hand: the chi^2
budget, per-site common-mode decomposition, representativeness sigma, residual
whiteness. None import Firedrake, so they run on any result JSON and are unit
tested directly.

The recurring kernel across all of them is the standardized residual
``r = (pred - obs) / sigma``; :func:`block_residuals` computes it once.
"""
from __future__ import annotations

import numpy as np


def block_residuals(result):
    """Standardized residuals ``(pred - obs) / sigma`` per observation block.

    Parameters
    ----------
    result
        A forward result dict with an ``obs`` list of block dicts, or the
        ``obs`` list itself.

    Returns
    -------
    dict
        ``{label: residual_array}`` in sigma units.
    """
    blocks = result["obs"] if isinstance(result, dict) else result
    out = {}
    for b in blocks:
        o, p, s = (np.asarray(b[k], float) for k in ("obs", "pred", "sig"))
        out[b["label"]] = (p - o) / s
    return out


def j_budget(result):
    """chi^2 budget: each block's share of the objective.

    A block's pull on the controls is its chi^2 share ``0.5 * sum(r^2)``, not
    its rms -- many points at a slightly-too-small sigma can dominate J while
    looking innocuous per-panel.

    Returns
    -------
    dict
        ``J`` (the total objective, if present in ``result``), ``J_obs`` (the
        summed block chi^2), ``J_prior`` (``J - J_obs``), and ``blocks``: a list
        of ``{label, n, rms, chi2, frac_J}`` sorted by descending chi^2.
        ``frac_J`` is relative to ``J`` when available, else to ``J_obs``.
    """
    rows = []
    for label, r in block_residuals(result).items():
        n = int(r.size)
        rows.append(dict(label=label, n=n,
                         rms=float(np.sqrt(np.mean(r**2))) if n else 0.0,
                         chi2=0.5 * float(np.sum(r**2))))
    J_obs = sum(d["chi2"] for d in rows)
    J = float(result["J"]) if isinstance(result, dict) and result.get("J") is not None else J_obs
    denom = J if J else 1.0
    for d in rows:
        d["frac_J"] = d["chi2"] / denom
    rows.sort(key=lambda d: -d["chi2"])
    return dict(J=J, J_obs=J_obs, J_prior=J - J_obs, blocks=rows)


def implied_sigma_repr(residual_physical, sigma_meas):
    """Representativeness sigma that would make a block self-consistent.

    Given the raw (physical-unit) residual ``pred - obs`` and the measurement
    sigma, returns ``sqrt(max(<res^2> - <sigma_meas^2>, 0))`` -- the extra
    (representation) error that, added in quadrature, brings the block to
    chi^2/N = 1. Zero means the measurement error already explains the scatter.
    """
    res = np.asarray(residual_physical, float)
    sm = np.asarray(sigma_meas, float)
    return float(np.sqrt(max(np.mean(res**2) - np.mean(sm**2), 0.0)))


def residual_acf(residual, max_lag=5):
    """Autocorrelation of a residual sequence at lags 0..max_lag.

    A well-specified error model leaves white residuals (acf ~ 0 beyond lag 0);
    structure at low lags means the sigma is fighting real, correlated signal.
    """
    r = np.asarray(residual, float)
    r = r - r.mean()
    denom = float(np.dot(r, r))
    if denom == 0.0:
        return np.zeros(max_lag + 1)
    return np.array([float(np.dot(r[: r.size - k], r[k:])) / denom
                     for k in range(max_lag + 1)])


def common_mode_decomposition(residual_matrix):
    """Split a per-site residual matrix into common-mode and site-specific.

    ``residual_matrix`` is ``(n_site, n_depth)`` of standardized residuals for
    the same observable measured at several sites (e.g. per-station ApRES
    velocity). The common mode is the across-site mean at each depth; the
    site-specific part is the deviation from it. If the variance is mostly
    common the residual is the model/operator's; if mostly site-specific it is
    real spatial variation the pooled sigma cannot represent.

    Returns
    -------
    dict
        ``common`` (n_depth), ``sitedev`` (n_site, n_depth), ``var_common``,
        ``var_site``, ``frac_common``.
    """
    Rm = np.asarray(residual_matrix, float)
    common = Rm.mean(0)
    sitedev = Rm - common
    var_common = float(np.mean(common**2))
    var_site = float(np.mean(sitedev**2))
    total = var_common + var_site
    return dict(common=common, sitedev=sitedev,
                var_common=var_common, var_site=var_site,
                frac_common=(var_common / total if total else 0.0))
