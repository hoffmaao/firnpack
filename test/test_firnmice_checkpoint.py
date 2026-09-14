"""test_firnmice_checkpoint.py - FirnMICE suite as an integration test.

The experiment table, column helpers, and single-experiment driver live in
`firnpack.firnmice`; this file is only the test wrapper around them. The same
machinery is driven for the case study by tutorials/firnmice/run.py, which is
where you want to be if you are producing the summary figure rather than
checking that the suite still runs.

Self-contained: writes into pytest's tmp_path and reads nothing from the
tutorials. Marked slow - it spins up and integrates six columns.

Environment variables:
  FIRNMICE_RUN_FULL=1          full 2000-yr experiments (default: 200-yr short)
  FIRNMICE_SPINUP_YEARS        spinup duration (default 5000 full / 1000 short)
  FIRNMICE_DT_YEARS            timestep in years (default 1)
  FIRNMICE_NZ                  elements (default 320 full / 220 short)
  FIRNMICE_OUTPUT_DIR          write outside tmp_path (default: tmp_path)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

try:
    import firedrake as fd
except Exception:  # pragma: no cover
    fd = None

from firnpack.firnmice import (
    FIRNMICE_EXPERIMENTS,
    FirnMICEExperiment,
    run_firnmice_experiment,
)


def _float_env(name: str, default: float) -> float:
    v = os.getenv(name, "")
    return float(v) if v.strip() else float(default)


def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    return default if not v else v in ("1", "true", "yes", "on")


@pytest.mark.slow
def test_firnmice_suite_checkpoint(tmp_path: Path) -> None:
    """Run the FirnMICE experiments (optionally shortened) and write checkpoints."""
    if fd is None:
        pytest.skip("firedrake not available")

    run_full = _bool_env("FIRNMICE_RUN_FULL", False)
    H0 = _float_env("FIRNMICE_H0", 1000.0)
    NZ = int(_float_env("FIRNMICE_NZ", 320 if run_full else 220))
    dt_years = _float_env("FIRNMICE_DT_YEARS", 1.0)
    spinup_years = _float_env("FIRNMICE_SPINUP_YEARS", 5000.0 if run_full else 1000.0)
    save_every_years = _float_env("FIRNMICE_SAVE_EVERY_YEARS", 1.0 if run_full else 10.0)

    out_dir = Path(os.getenv("FIRNMICE_OUTPUT_DIR", str(tmp_path / "firnmice")))
    out_dir.mkdir(parents=True, exist_ok=True)

    # short mode keeps the step at 100 yr but stops at 200 yr
    t_end_short = 200.0

    for exp0 in FIRNMICE_EXPERIMENTS:
        exp = exp0 if run_full else FirnMICEExperiment(
            name=exp0.name, T0_C=exp0.T0_C, a0_mieq_yr=exp0.a0_mieq_yr,
            dT_C=exp0.dT_C, da_mieq_yr=exp0.da_mieq_yr,
            t_step_yr=exp0.t_step_yr, t_end_yr=t_end_short,
        )

        res = run_firnmice_experiment(
            exp=exp, out_dir=out_dir, H0=H0, NZ=NZ,
            STRETCH_P=_float_env("FIRNMICE_STRETCH_P", 3.0),
            dt_years=dt_years, spinup_years=spinup_years,
            save_every_years=save_every_years,
            make_plots=_bool_env("FIRNMICE_PLOTS", False),
            use_seasonal=_bool_env("FIRNMICE_SEASONAL", False),
            rho_surf_kg_m3=360.0,
        )

        # Minimal, robust assertions (avoid brittle numeric targets)
        time = res["time_years"]
        DIP = res["DIP_m"]
        bco_z = res["BCO_depth_m"]
        bco_age = res["BCO_age_yr"]

        assert np.isfinite(time).all()
        assert np.isfinite(DIP).all()

        # We expect to reach BCO in a 1000 m domain for these forcings.
        assert np.isfinite(bco_z).any(), f"BCO depth never found in {exp.name}"
        assert np.isfinite(bco_age).any(), f"BCO age never found in {exp.name}"

        # Sign sanity: warming reduces DIP; more accumulation increases it.
        #
        # Measure the response from the step, not from t=0. DIP[0] is the state
        # just after spinup, and spinup is not always complete: at the short
        # tier's 1000 yr, ex1 (-50 C, the slowest-densifying column) is still
        # drifting by +1.7 m over the pre-step century, which is three times the
        # -0.6 m the warming itself produces. Comparing endpoints therefore
        # measured residual spinup drift rather than the step response and
        # failed on ex1, even though the response is correctly signed at every
        # spinup length tried (-0.59 m at 1000 yr, -2.00 at 3000, -2.02 at
        # 6000). Anchoring at the step removes the drift common to both sides.
        i_step = int(np.searchsorted(time, exp.t_step_yr))
        if i_step < len(DIP) - 1:
            if exp.dT_C != 0.0:
                assert DIP[-1] <= DIP[i_step] + 1e-8
            if exp.da_mieq_yr != 0.0:
                assert DIP[-1] >= DIP[i_step] - 1e-8

        assert (out_dir / f"firnmice_{exp.name}.h5").exists()
