"""Engine coverage for the observation machinery added with the recalibration.

Three engine features arrived with the data-derived error models and have no
other automated cover, because the tutorials that use them (South Pole, Summit,
the synthetic OSSE) are multi-minute runs over curated data:

* the ``compaction`` observation kind (FirnCover-style coil: the shortening
  rate of a MATERIAL interval, predicted as ``(w@ztop - w@zbot)*year``),
* epoch-tagged blocks (``ObsBlock.year``), which must be scored against the
  state at that step rather than the final state, and must refuse an epoch
  older than the simulated span rather than silently snapping it to step 0,
* per-knot prior sigma (``KnotCtrl.sigma`` as an array).

Everything here runs on a deliberately tiny column (40 m, 30 cells, 20 steps)
so the whole module is a few seconds: these are contract tests on the misfit
and prior operators, not physics benchmarks.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import firedrake as fd
except Exception:  # pragma: no cover - exercised only without firedrake
    fd = None

if fd is not None:
    from firnpack.inverse import KnotCtrl, ObsBlock, ScalarCtrl, SiteConfig, assimilate


PRESENT = 2015.0
SPIN = 200.0
DT = 10.0
H_COL = 40.0
# the b history is a step in time, so the firn state at 1915 is genuinely
# different from the state at 2015 -- otherwise "scored at its epoch" and
# "scored at the end" would be indistinguishable.
B_YEARS = np.array([1815.0, 1915.0, 2015.0])
B_VALUES = np.array([0.10, 0.10, 0.22])


def _cfg(out_dir, obs, b_sigma=0.25, invert_b=True):
    return SiteConfig(
        name="tiny", out_dir=str(out_dir), tag="tiny",
        H_col=H_COL, NZ=30, spin_years=SPIN, dt_years=DT, present_year=PRESENT,
        rho_surf=350.0, rho_ic_deep=820.0, rho_ic_scale=25.0,
        conductivity_law="calonne2019",
        scalars=[ScalarCtrl("hl_k0", 11.0, 11.0, 1.0, 0.5, 500.0, log=True)],
        T_knots=KnotCtrl(np.array([1815.0, 2015.0]), np.array([-30.0, -30.0]),
                         np.array([-30.0, -30.0]), 0.6, -40.0, -20.0,
                         log=False, invert=False, name="Tk"),
        b_knots=KnotCtrl(B_YEARS, B_VALUES, B_VALUES, b_sigma, 0.03, 0.60,
                         log=True, invert=invert_b, name="b"),
        obs=obs, max_iter=1,
    )


def _rho_block(**kw):
    """A density block in the mid-column, where the profile is smooth."""
    return ObsBlock("rho", np.array([12.0, 20.0]), np.array([600.0, 700.0]),
                    np.array([10.0, 10.0]), label="rho", **kw)


def _comp_block():
    """Two coils spanning [1, 5] m and [1, 16] m, as at Summit."""
    return ObsBlock("compaction", np.array([5.0, 16.0]),
                    np.array([-0.05, -0.12]), np.array([0.02, 0.02]),
                    ztop=np.array([1.0, 1.0]), label="comp")


def _block(result, label):
    return [b for b in result["obs"] if b["label"] == label][0]


# -----------------------------------------------------------------------------
# compaction observation kind
# -----------------------------------------------------------------------------

def test_compaction_prediction_is_the_interval_velocity_difference(tmp_path, adjoint_tape):
    """pred == w(ztop) - w(zbot) in m/yr, negative for a shortening interval."""
    r = assimilate(_cfg(tmp_path, [_rho_block(), _comp_block()]),
                   mode="forward", verbose=False)
    prof = r["profiles"]
    order = np.argsort(np.asarray(prof["depth"]))
    depth = np.asarray(prof["depth"])[order]
    w = np.asarray(prof["w_m_yr"])[order]        # m/yr, negative downwards

    blk = _block(r, "comp")
    pred = np.asarray(blk["pred"])
    independent = np.array([np.interp(1.0, depth, w) - np.interp(zb, depth, w)
                            for zb in blk["depths"]])

    # the engine kernel-averages where this reads point values, so they agree to
    # the kernel width, not to machine precision
    assert pred == pytest.approx(independent, abs=5.0e-3)
    # shortening: the interval gets shorter, and the longer interval more so
    assert np.all(pred < 0.0)
    assert pred[1] < pred[0]


def test_compaction_gradient_matches_finite_difference(tmp_path, adjoint_tape):
    """The adjoint gradient through the new operator is the real gradient."""
    r = assimilate(_cfg(tmp_path, [_comp_block()]), mode="verify",
                   fd_names=["hl_k0", "b2015"], fd_h=1e-4, verbose=False)
    assert r["match"], "replay is not deterministic"
    for name, ratio in r["fd_ratios"].items():
        assert ratio == pytest.approx(1.0, abs=2.0e-3), f"{name} adjoint != FD"


# -----------------------------------------------------------------------------
# epoch-tagged observations
# -----------------------------------------------------------------------------

def test_tagged_block_is_scored_at_its_own_epoch(tmp_path, adjoint_tape):
    """A block tagged to 1915 sees the 1915 column, not the 2015 one.

    Checked against the engine's own recorded snapshots: the tagged prediction
    must track the density profile at the tagged step and NOT the final one.
    """
    epoch = 1915.0
    r = assimilate(_cfg(tmp_path, [_rho_block(year=epoch)]), mode="forward",
                   verbose=False, record_series=True, snap_every=1)
    pred = np.asarray(_block(r, "rho")["pred"])
    depths = np.asarray(_block(r, "rho")["depths"])

    years = np.asarray(r["snaps"]["year"])
    k_epoch = int(np.argmin(np.abs(years - epoch)))
    zs = np.asarray(r["snap_depth"])
    at_epoch = np.array([np.interp(z, zs, np.asarray(r["snaps"]["rho"])[k_epoch]) for z in depths])
    at_end = np.array([np.interp(z, zs, np.asarray(r["snaps"]["rho"])[-1]) for z in depths])

    assert abs(years[k_epoch] - epoch) <= DT / 2.0
    # the two states must be distinguishable, or the test proves nothing
    assert np.max(np.abs(at_epoch - at_end)) > 5.0
    assert np.max(np.abs(pred - at_epoch)) < np.max(np.abs(pred - at_end))

    untagged = assimilate(_cfg(tmp_path, [_rho_block()]), mode="forward", verbose=False)
    assert np.max(np.abs(np.asarray(_block(untagged, "rho")["pred"]) - at_end)) < 5.0


def test_epoch_older_than_the_span_is_rejected(tmp_path, adjoint_tape):
    """Silently snapping a pre-span epoch to step 0 would score it wrongly."""
    with pytest.raises(ValueError, match="older than the simulated span"):
        assimilate(_cfg(tmp_path, [_rho_block(year=1500.0)]), mode="forward",
                   verbose=False)


def test_epoch_after_the_span_keeps_the_final_state(tmp_path, adjoint_tape):
    """A measurement made after present_year is the closing state (Summit 2017)."""
    late = assimilate(_cfg(tmp_path, [_rho_block(year=PRESENT + 2.0)]),
                      mode="forward", verbose=False)
    plain = assimilate(_cfg(tmp_path, [_rho_block()]), mode="forward", verbose=False)
    assert _block(late, "rho")["pred"] == pytest.approx(_block(plain, "rho")["pred"])


# -----------------------------------------------------------------------------
# per-knot prior sigma
# -----------------------------------------------------------------------------

def test_per_knot_prior_sigma(tmp_path, adjoint_tape):
    """KnotCtrl.sigma accepts an array, and each knot pays its own prior cost."""
    # warm-start the b-knots away from their prior centres so the prior term is
    # nonzero and its sigma weighting is visible in J
    warm = dict(m_map={}, b_knots=(B_VALUES * 1.25).tolist(),
                b_knot_years=B_YEARS.tolist())

    def J_of(sig):
        return assimilate(_cfg(tmp_path, [_rho_block()], b_sigma=sig),
                          mode="forward", warm=warm, verbose=False)["J"]

    j_scalar = J_of(0.25)
    j_uniform_array = J_of(np.full(len(B_YEARS), 0.25))
    j_mixed = J_of(np.array([0.25, 0.25, 0.05]))

    # an array of identical values must reproduce the scalar exactly
    assert j_uniform_array == pytest.approx(j_scalar, rel=1e-12)
    # tightening ONE knot must raise J by that knot's prior term alone
    d = np.log(1.25)
    expected = 0.5 * d**2 * (1.0 / 0.05**2 - 1.0 / 0.25**2)
    assert j_mixed - j_scalar == pytest.approx(expected, rel=1e-6)
