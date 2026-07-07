"""Pluggable densification rate laws.

Each law is a callable that returns a UFL expression for dρ/dt [kg m⁻³ s⁻¹].
Callables receive the current state fields and a FirnParameters object via
keyword arguments so users can freely mix and match.

Standard signature:
    rate(rho, T, *, params, bdot=None, sigma=None, grain_radius2=None,
         rhoCoef=None, **kwargs) -> UFL expression in kg m⁻³ s⁻¹

This mirrors icepack's approach where viscosity/friction laws are pluggable.

Available laws:
    arthern_ligtenberg  — Arthern (2010) + Ligtenberg (2011) corrections
    herron_langway      — Herron & Langway (1980) empirical two-stage
    kingslake           — Kingslake (2022) stress + grain-size dependent
    stokes_compressible — Compressible Stokes (Gagliardini & Meyssonnier 1997)
"""
import firedrake as fd


# ---------------------------------------------------------------------------
# Helper: smooth switch between two expressions at a threshold
# ---------------------------------------------------------------------------
def _rho_switch(rho, rho_m, smooth_width, low_expr, high_expr):
    """Return ``low_expr`` when rho <= rho_m, else ``high_expr``,
    with a smooth tanh transition of width ``smooth_width``."""
    if smooth_width > 0:
        s = 0.5 * (1.0 + fd.tanh((rho - rho_m) / smooth_width))
        return (1.0 - s) * low_expr + s * high_expr
    return fd.conditional(fd.le(rho, rho_m), low_expr, high_expr)


# ---------------------------------------------------------------------------
# Arthern / Ligtenberg 2011 (default firngrain law)
# ---------------------------------------------------------------------------
def arthern_ligtenberg(rho, T, *, params, bdot, smooth=False, **_kwargs):
    """dρ/dt = c * (ρ_i - ρ) where c depends on accumulation and T.

    Ligtenberg (2011) corrections M0, M1 depend on accumulation rate.
    Switches between low-density kc0 and high-density kc1 at ρ_m.
    """
    p = params
    M0 = 1.435 - 0.151 * fd.ln(bdot * 1.0e3)
    M1 = 2.366 - 0.293 * fd.ln(bdot * 1.0e3)

    exp_factor = fd.exp(-p.Ec / (p.R * T) + p.Eg / (p.R * p.Tavg))
    c0 = M0 * bdot * p.g * p.kc0 / p.kg * exp_factor
    c1 = M1 * bdot * p.g * p.kc1 / p.kg * exp_factor

    smooth_width = 20.0 if smooth else 0.0
    c = _rho_switch(rho, p.rho_m, smooth_width, c0, c1)
    return c * (p.rho_i - rho)


# ---------------------------------------------------------------------------
# Herron & Langway 1980 (simpler empirical law)
# ---------------------------------------------------------------------------
def herron_langway(rho, T, *, params, bdot, smooth=False, **_kwargs):
    """Herron & Langway (1980) empirical densification.

    Stage 1 (ρ ≤ 550 kg/m³):  dρ/dt = K0 * bdot_yr * (ρ_i - ρ)
        where K0 = k0_prefactor * exp(-Ea_stage1/(R T)).
    Stage 2 (ρ > 550 kg/m³):  dρ/dt = K1 * sqrt(bdot_yr) * (ρ_i - ρ)
        where K1 = k1_prefactor * exp(-Ea_stage2/(R T)).

    Activation energies are ~42.5 kJ/mol (stage 1) and ~89.5 kJ/mol (stage 2),
    calibrated empirically from Antarctic and Greenlandic ice cores.

    `bdot` input is mass accumulation in kg m⁻² s⁻¹. Internally converts to
    m ice-equivalent / yr for the H&L formula, then returns dρ/dt in kg m⁻³ s⁻¹.
    """
    p = params

    # Convert bdot [kg m⁻² s⁻¹] → [m ice eq / yr]
    bdot_m_ice_yr = bdot / p.rho_i * p.spy

    # Rate constants (Herron & Langway 1980, Eq. 6a, 6b)
    # NOTE: Modern implementations (CFM, Kingslake 2022) use H&L's quoted
    # activation-energy NUMBERS directly in J/mol with SI R=8.314, rather
    # than the cal/mol values with R=1.987 that H&L 1980 originally used.
    # This gives an "effective" H&L that matches published firn profiles.
    k0_pref = getattr(p, "hl_k0_prefactor", 11.0)    # 1/yr
    k1_pref = getattr(p, "hl_k1_prefactor", 575.0)   # (m ice eq)^(-1/2) / yr
    Ea1     = getattr(p, "hl_Ea_stage1", 10160.0)    # J/mol (used with SI R)
    Ea2     = getattr(p, "hl_Ea_stage2", 21400.0)    # J/mol (used with SI R)

    K0 = k0_pref * fd.exp(-Ea1 / (p.R * T))
    K1 = k1_pref * fd.exp(-Ea2 / (p.R * T))

    # sqrt(bdot) needs positivity guard
    bdot_safe = fd.max_value(bdot_m_ice_yr, 1.0e-8)

    c0 = K0 * bdot_m_ice_yr          # per year
    c1 = K1 * fd.sqrt(bdot_safe)     # per year

    # Optional stage-2 shape exponent: rate2 *= ((rho_i-rho)/(rho_i-rho_m))^(s-1).
    # The factor is exactly 1 at rho = rho_m (pivot at the transition, so it does
    # not trade against k1), and s < 1 sustains densification approaching
    # close-off. s = 1 reproduces H&L exactly and skips the factor entirely so
    # existing forms/kernels are unchanged.
    s2 = getattr(p, "hl_stage2_shape", 1.0)
    if not (isinstance(s2, (int, float)) and float(s2) == 1.0):
        phi_n = fd.max_value((p.rho_i - rho) / (p.rho_i - p.rho_m), 1.0e-4)
        c1 = c1 * phi_n ** (s2 - 1.0)

    # Optional deep cutoff (deep-column configs): smoothly kill the stage-2
    # rate approaching close-off so parcels never reach the rho_i barrier
    # kink in density_form. None = off (legacy kernels unchanged).
    cutoff = getattr(p, "hl_deep_cutoff_rho", None)
    if cutoff is not None:
        c1 = c1 * 0.5 * (1.0 - fd.tanh((rho - cutoff) / 8.0))

    smooth_width = 20.0 if smooth else 0.0
    c = _rho_switch(rho, p.rho_m, smooth_width, c0, c1)

    # Convert 1/yr → 1/s
    return (c / p.spy) * (p.rho_i - rho)


# ---------------------------------------------------------------------------
# Kingslake 2022 (stress- and grain-size-dependent)
# ---------------------------------------------------------------------------
def kingslake(rho, T, *, params, sigma, grain_radius2, rhoCoef, **_kwargs):
    """Kingslake-style densification:
        dρ/dt = rhoCoef * exp(-Ec/RT) * (ρ_i - ρ) * σ / (r² + ε)
    """
    p = params
    r_eff = grain_radius2 + p.eps_r2
    return rhoCoef * fd.exp(-p.Ec / (p.R * T)) * (p.rho_i - rho) * sigma / r_eff


# ---------------------------------------------------------------------------
# Compressible Stokes densification (Gagliardini & Meyssonnier 1997)
# ---------------------------------------------------------------------------
# Default parameters for the compressible rheology.
# Stage 1 empirical coefficients from Rathmann (firndens-1d).
_STOKES_DEFAULTS = dict(
    n_glen=1.0,              # Glen exponent (1 = linear, 3 = standard)
    A_glen=6.8e-26,          # Glen rate factor [Pa^-n s^-1] (~ -25 °C)
    Q_glen=60.0e3,           # activation energy for A(T) [J/mol]
    A_ref_T=248.15,          # reference temperature for A_glen [K]
    rhohat_thres=0.81,       # stage 1/2 threshold in relative density
    # Stage 1: empirical a,b (Rathmann / Zwinger et al. 2007)
    c1_a=13.22240,
    c2_a=15.78652,
    c1_b=15.09371,
    c2_b=20.46489,
    E_lin=5.0e9,             # enhancement factor for n=1 (Rathmann)
)


def calibrate_A_glen(n, sigma_ref=200.0e3, rho_hat_ref=0.5,
                     T_ref_calib=243.15):
    """Calibrate A_glen for a given Glen exponent n.

    Finds A_glen such that the volumetric strain rate at reference
    conditions (sigma_ref, rho_hat_ref, T_ref_calib) matches the n=1
    reference solution (with E_lin=5e9).

    Parameters
    ----------
    n : float
        Glen exponent.
    sigma_ref : float
        Reference overburden stress [Pa]. Default 200 kPa.
    rho_hat_ref : float
        Reference relative density. Default 0.5.
    T_ref_calib : float
        Reference temperature [K]. Default 243.15 K (-30°C).

    Returns
    -------
    A_glen : float
        Calibrated rate factor [Pa^-n s^-1].
    """
    import math
    d = _STOKES_DEFAULTS
    R = 8.3144621
    A_T_ref = d["A_glen"] * math.exp(
        -d["Q_glen"] / R * (1.0 / T_ref_calib - 1.0 / d["A_ref_T"])
    )
    # Stage 1 compressibility at rho_hat_ref
    a1 = math.exp(d["c1_a"] - d["c2_a"] * rho_hat_ref)
    b1 = math.exp(d["c1_b"] - d["c2_b"] * rho_hat_ref)
    C = 4.0 / (3.0 * a1) + 1.0 / b1
    sig_C = sigma_ref / C
    # n=1 reference rate (with E_lin)
    eps_ref = 2.0 * A_T_ref * d["E_lin"] * sig_C
    # For target n: eps = 2 * A_eff * sig_C^n
    A_eff = eps_ref / (2.0 * sig_C ** n)
    # Undo Arrhenius to get A_glen
    boltz = math.exp(-d["Q_glen"] / R * (1.0 / T_ref_calib - 1.0 / d["A_ref_T"]))
    return A_eff / boltz


def _compressibility_ab(rho_hat, n, thres, c1_a, c2_a, c1_b, c2_b,
                         smooth_width=0.0):
    """Compressibility functions a(ρ̂) and b(ρ̂).

    Stage 1 (ρ̂ ≤ threshold): empirical exponential fits.
    Stage 2 (ρ̂ > threshold): self-consistent micromechanical model
        (Duva & Crow 1994, Gagliardini & Meyssonnier 1997).

    Parameters
    ----------
    rho_hat : UFL expression
        Relative density ρ/ρ_i.
    n : float
        Glen exponent.
    thres : float
        Transition relative density (default 0.81).
    c1_a, c2_a, c1_b, c2_b : float
        Stage 1 empirical coefficients.
    smooth_width : float
        Tanh smoothing width in relative density units (0 = hard switch).

    Returns
    -------
    a, b : UFL expressions
        Deviatoric and volumetric compressibility functions.
    """
    # Stage 1: empirical (Rathmann / Zwinger et al. 2007)
    a1 = fd.exp(c1_a - c2_a * rho_hat)
    b1 = fd.exp(c1_b - c2_b * rho_hat)

    # Stage 2: micromechanical (Duva & Crow 1994 / GM97)
    porosity = fd.max_value(1.0 - rho_hat, 1.0e-10)
    exp_2n = 2.0 * n / (n + 1.0)

    a2 = (1.0 + 2.0 / 3.0 * porosity) * rho_hat ** (-exp_2n)

    # b2 = (3/4) * ((porosity^(1/n)) / (n * (1 - porosity^(1/n))))^(2n/(n+1))
    por_1n = porosity ** (1.0 / n)
    denom = fd.max_value(n * (1.0 - por_1n), 1.0e-30)
    b2 = 0.75 * (por_1n / denom) ** exp_2n

    a = _rho_switch(rho_hat, thres, smooth_width, a1, a2)
    b = _rho_switch(rho_hat, thres, smooth_width, b1, b2)
    return a, b


def _glen_rate_factor(T, params, A_ref, Q, T_ref):
    """Temperature-dependent Glen rate factor A(T).

    A(T) = A_ref * exp(-Q/R * (1/T - 1/T_ref))
    """
    R = params.R
    return A_ref * fd.exp(-Q / R * (1.0 / T - 1.0 / T_ref))


def stokes_compressible(rho, T, *, params, sigma, smooth=True, **_kwargs):
    """Compressible Stokes densification (Gagliardini & Meyssonnier 1997).

    Densification emerges from the balance between overburden stress and a
    compressible viscous rheology.  In 1D quasi-static equilibrium the
    volumetric strain rate is determined by inverting the constitutive law:

        ε̇_v = -(|σ| / η_eff)^n · sign(σ)

    where ``η_eff`` incorporates the density-dependent compressibility
    functions ``a(ρ̂)`` and ``b(ρ̂)`` and Glen's rate factor ``A(T)``.
    The densification rate is then:

        dρ/dt = -ρ · ε̇_v

    Parameters
    ----------
    rho : UFL / Function
        Density [kg m⁻³].
    T : UFL / Function
        Temperature [K].
    params : FirnParameters
        Must have ``rho_i``, ``R``.  Stokes-specific parameters are read
        from ``params.stokes_*`` attributes with fallback to defaults.
    sigma : UFL / Function
        Overburden stress [Pa] (positive in compression).
    smooth : bool
        Use tanh smoothing for stage 1/2 transition (default True).

    Returns
    -------
    UFL expression for dρ/dt in [kg m⁻³ s⁻¹].
    """
    p = params

    # Read Stokes-specific parameters from params or use defaults.
    def _get(name):
        return getattr(p, f"stokes_{name}", _STOKES_DEFAULTS[name])

    n = _get("n_glen")
    A_ref = _get("A_glen")
    Q = _get("Q_glen")
    T_ref = _get("A_ref_T")
    thres = _get("rhohat_thres")
    c1_a, c2_a = _get("c1_a"), _get("c2_a")
    c1_b, c2_b = _get("c1_b"), _get("c2_b")
    E_lin = _get("E_lin")

    rho_hat = rho / p.rho_i
    smooth_w = 0.10 if smooth else 0.0  # in ρ̂ units (~92 kg/m³)

    a, b = _compressibility_ab(rho_hat, n, thres,
                                c1_a, c2_a, c1_b, c2_b,
                                smooth_width=smooth_w)

    # Temperature-dependent rate factor
    A_T = _glen_rate_factor(T, p, A_ref, Q, T_ref)

    # Enhancement for n=1 (Rathmann convention).
    # E_lin may be a UFL expression (control on tape) or a plain number.
    if abs(float(n) - 1.0) < 0.01:
        A_eff = A_T * (E_lin if hasattr(E_lin, 'ufl_shape') else fd.Constant(E_lin))
    else:
        A_eff = A_T

    # Effective 1D viscosity coefficient.
    # In a 1D column (vertical velocity only), the stress–strain-rate
    # relationship is:
    #   σ_zz = B^{-1/n} · C(ρ̂) · |ε̇_v|^{1/n-1} · ε̇_v
    # where C = 4/(3a) + 1/b  (combining deviatoric and volumetric parts).
    #
    # Inverting for |ε̇_v|:
    #   |ε̇_v| = (2A) · (|σ| / C)^n

    C = 4.0 / (3.0 * a) + 1.0 / b

    # Guard against zero stress
    sigma_safe = fd.max_value(sigma, 1.0e-10)

    # Volumetric strain rate magnitude
    two_A = 2.0 * A_eff
    n_val = float(n)
    if abs(n_val - 1.0) < 0.01:
        # n=1: linear, eps_v = 2A * sigma / C
        eps_v_mag = two_A * sigma_safe / C
    else:
        # General n: eps_v = 2A * (sigma/C)^n
        eps_v_mag = two_A * (sigma_safe / C) ** fd.Constant(n_val)

    # dρ/dt = ρ · |ε̇_v|  (compaction increases density)
    return rho * eps_v_mag
