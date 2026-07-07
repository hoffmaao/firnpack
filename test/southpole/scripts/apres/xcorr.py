"""Cross-correlation utilities (translation of MATLAB fmcw_xcorr.m)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class XCorrResult:
    """Result of correlating a profile segment across a set of lags."""

    iw: np.ndarray  # effective range index (float, in original bin units)
    ic: np.ndarray  # incoherent (amplitude-only) correlation (0..1)
    cc: np.ndarray  # coherent (complex) correlation
    lags: np.ndarray  # lags searched (bin units)
    pe: np.ndarray  # phase error (radians)
    pse: np.ndarray  # phase standard error (radians)


def _lags_from_maxlag(maxlag: Sequence[int] | int) -> np.ndarray:
    if isinstance(maxlag, (int, np.integer)):
        n = int(abs(maxlag))
        return np.arange(-n, n + 1, dtype=int)
    if len(maxlag) == 2:
        return np.arange(int(maxlag[0]), int(maxlag[1]) + 1, dtype=int)
    raise ValueError("maxlag should be an int or a 2-element sequence")


def fmcw_xcorr(
    f: np.ndarray,
    g: np.ndarray,
    fi: np.ndarray,
    maxlag: Sequence[int] | int,
    fe: Optional[np.ndarray] = None,
    ge: Optional[np.ndarray] = None,
    pad_factor: Optional[int] = None,
) -> XCorrResult:
    """Cross-correlate portions of two complex vectors.

    This is a Python translation of the MATLAB function `fmcw_xcorr`.

    Parameters
    ----------
    f, g
        Complex spectra (1D arrays) from profile 1 and profile 2.
    fi
        Indices of the segment in *f* (and *g*) to correlate.
    maxlag
        Either an integer meaning +/- maxlag, or a 2-element [min,max].
    fe, ge
        Optional phase standard error estimates for f and g (same length as f).
        If provided, error metrics pe and pse are computed.
    pad_factor
        Pad factor used in FFT (only needed for pse calculation).

    Returns
    -------
    XCorrResult
    """
    f = np.asarray(f).reshape(-1)
    g = np.asarray(g).reshape(-1)
    fi = np.asarray(fi, dtype=int).reshape(-1)

    lags = _lags_from_maxlag(maxlag)

    # Pad to cope with end effects, as in MATLAB.
    n = int(np.max(np.abs(lags))) if lags.size else 0
    if n > 0:
        zp = np.zeros(n, dtype=f.dtype)
        f_pad = np.concatenate([zp, f, zp])
        g_pad = np.concatenate([zp, g, zp])
        fi_pad = fi + n
        if fe is not None and ge is not None:
            fe_pad = np.concatenate([np.zeros(n), np.asarray(fe).reshape(-1), np.zeros(n)])
            ge_pad = np.concatenate([np.zeros(n), np.asarray(ge).reshape(-1), np.zeros(n)])
        else:
            fe_pad = ge_pad = None
    else:
        f_pad = f
        g_pad = g
        fi_pad = fi
        fe_pad = ge_pad = None

    fc = f_pad[fi_pad]
    cff0 = np.sum(np.abs(fc) ** 2)

    iw = np.zeros_like(lags, dtype=float)
    ic = np.zeros_like(lags, dtype=float)
    cc = np.zeros_like(lags, dtype=complex)
    pe = np.full_like(lags, np.nan, dtype=float)
    pse = np.full_like(lags, np.nan, dtype=float)

    do_error = fe_pad is not None and ge_pad is not None and pad_factor is not None
    if do_error:
        fec = fe_pad[fi_pad]
        p = float(pad_factor)

    for i, lag in enumerate(lags):
        gc = g_pad[fi_pad + lag]
        fg = fc * np.conj(gc)
        cgg0 = np.sum(np.abs(gc) ** 2)
        scale = np.sqrt(cff0 * cgg0) if (cff0 > 0 and cgg0 > 0) else np.nan
        if not np.isfinite(scale) or scale == 0:
            cc[i] = 0.0 + 0.0j
            ic[i] = 0.0
            iw[i] = float(np.mean(fi))
            continue

        cc[i] = np.sum(fg) / scale
        ic[i] = np.sum(np.abs(fg)) / scale

        # Effective bin of correlation (amplitude centre of mass)
        w = np.abs(fg)
        if np.sum(w) > 0:
            iw[i] = float(np.sum(w * fi_pad) / np.sum(w) - n)
        else:
            iw[i] = float(np.mean(fi))

        if do_error:
            gec = ge_pad[fi_pad + lag]
            pder = np.sqrt(fec**2 + gec**2)  # phase diff std dev (fractional error of product)
            ab = np.abs(fc) * np.abs(gc)
            sum_er = np.sqrt(np.sum((pder * ab) ** 2))
            sum_mag = np.abs(np.sum(fg))
            pe_i = sum_er / sum_mag if sum_mag > 0 else np.inf
            pe[i] = pe_i

            # effective number of samples with weighting
            if np.max(w) > 0:
                ne = np.sum(w) / np.max(w)
            else:
                ne = 0.0
            ne = ne / p
            pse[i] = pe_i / np.sqrt(ne) if ne > 0 else np.inf

    return XCorrResult(iw=iw, ic=ic, cc=cc, lags=lags, pe=pe, pse=pse)
