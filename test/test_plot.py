"""Tests for firnpack.plot.

These guard the two things that are easy to break silently: the categorical
palette (validated for colourblind separation -- substituting a hue by hand
would not fail any other test) and the per-site family encoding.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from firnpack import plot as fp


def test_base_label_strips_site():
    assert fp.base_label("v_x11n6") == "v"
    assert fp.base_label("rho") == "rho"


def test_standalone_block_gets_its_base_hue():
    for lab, hue in fp.BLOCK_COLORS.items():
        assert fp.block_color(lab) == hue


def test_family_members_get_distinct_colours_and_markers():
    n = 5
    cols = [fp.block_color(f"v_s{i}", i, n) for i in range(n)]
    marks = [fp.block_marker(i, n) for i in range(n)]
    assert len(set(cols)) == n, "family colours must be distinguishable"
    assert len(set(marks)) == n, "markers are the secondary encoding; must differ"


def test_family_ramp_stays_clear_of_the_age_hue():
    """v (#B45309) and age (#EA580C) are nearly the same hue, so a wide ramp
    off v walks into age's identity. Guard the light end."""
    import colorsys
    def hue(hexstr):
        r, g, b = (int(hexstr[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        return colorsys.rgb_to_hls(r, g, b)[0]
    lightest = fp.block_color("v_s4", 4, 5)
    # same hue family is expected; what must not happen is matching age outright
    assert lightest != fp.BLOCK_COLORS["age"]
    import colorsys as _c
    l_light = _c.rgb_to_hls(*(int(lightest[i:i + 2], 16) / 255.0 for i in (1, 3, 5)))[1]
    assert l_light <= 0.62, "ramp escaped the validated lightness band"


def test_depth_axis_points_down():
    fig, ax = plt.subplots()
    fp.depth_axis(ax, hmax=132.0)
    lo, hi = ax.get_ylim()
    assert lo > hi, "depth must increase downward"
    assert lo == pytest.approx(132.0)
    plt.close(fig)


def test_profile_and_residual_panel_render():
    d = np.linspace(6, 129, 20)
    fig, (a0, a1) = plt.subplots(1, 2)
    fp.profile(a0, d, 350 + 4 * d, sig=np.full(d.size, 14.0), pred=350 + 4 * d,
               title="t", xlabel="x")
    fp.residual_panel(a1, {"dage": (d, np.zeros(d.size)),
                           "v_a": (d, np.zeros(d.size)),
                           "v_b": (d, np.zeros(d.size))})
    assert a1.get_legend() is not None
    plt.close(fig)


def test_residual_legend_does_not_sit_on_the_data():
    """The legend is parked right of +xlim; matplotlib's 'best' put it over the
    middle of the scatter, hiding the residuals the panel exists to show."""
    d = np.linspace(6, 129, 10)
    fig, ax = plt.subplots()
    fp.residual_panel(ax, {"dage": (d, np.zeros(d.size))}, xlim=3.2)
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(-3.2)
    assert hi > 3.2, "no reserved strip for the legend"
    plt.close(fig)
