"""tutorials/southpole/config.py — build the South Pole SiteConfig.

Extracted verbatim from run.py so both run.py and the diagnostics under
output/ construct the SAME config by calling build_cfg(), instead of the old
exec(open("run.py")...) hack. build_cfg() reads os.environ at call time, so a
script that sets FIRN_* before calling it gets the matching config.

Returns a SimpleNamespace of every builder local: .cfg is the SiteConfig;
.VEL_SITE/.VEL_SRC/._zeising_site_points and the rest are the handles the
diagnostics and run.py dispatch need.
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
from firnpack.inverse import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl
from firnpack.constants import year as YEAR_S


def build_cfg():
    """Construct the South Pole SiteConfig from env vars; return the namespace."""
    HERE = Path(__file__).parent
    DATA = HERE / "data"
    SP_RESULTS = HERE / "output"                                # frozen r8 lives here
    OUT = str(HERE / "output")
    MODE = os.environ.get("FIRN_MODE", "validate")
    # Resolution knobs (discretization-sensitivity study): defaults = r8 numerics.
    NZ = int(os.environ.get("FIRN_NZ", "100"))
    DT = float(os.environ.get("FIRN_DT", "5.0"))
    TAG = os.environ.get("FIRN_TAG", "sp_engine")
    T_SHIFT = -5.8
    H0 = 130.0   # DATA cut: all obs stay <= 130 m regardless of domain depth
    # Domain-truncation study (basal BC): H_COL deepens the column so the
    # advected/diffused T-history part of the basal gradient is SIMULATED and the
    # imposed G is only the quasi-steady deep part; obs cuts stay at H0.
    H_COL = float(os.environ.get("FIRN_HCOL", str(H0)))
    SPIN = float(os.environ.get("FIRN_SPIN", "2500.0"))

    # ---- knot layouts (r8 coarse; override with FIRN_KNOTS=<json> for the
    # temporal-sampling probe — file holds {"knot_years": [...], "b_knot_years": [...]}) ----
    KNOT_YEARS = np.array([1000.,1300.,1550.,1750.,1850.,1900.,1930.,1955.,1975.,1990.,2000.,2008.,2015.])
    KNOT_B_YEARS = np.array([1000.,1100.,1200.,1300.,1400.,1500.,1600.,1700.,1800.,1850.,1900.,1930.,1960.,1990.,2015.])
    _kf = os.environ.get("FIRN_KNOTS", "")
    if _kf:
        _kj = json.load(open(_kf if os.path.isabs(_kf) else HERE/_kf))
        KNOT_YEARS = np.array(_kj["knot_years"], dtype=float)
        KNOT_B_YEARS = np.array(_kj["b_knot_years"], dtype=float)
        print(f"knot layouts from {_kf}: {len(KNOT_YEARS)} T + {len(KNOT_B_YEARS)} b")

    # ---- T prior ramp (hand-drawn 6-knot), datum-shifted ----
    _T6Y = np.array([1000.,1300.,1550.,1750.,1900.,2015.]); _T6V = np.array([-45.50,-45.45,-45.35,-45.20,-45.10,-45.00])+T_SHIFT
    T_CTR = np.interp(KNOT_YEARS, _T6Y, _T6V)

    # ---- b prior centers: Buizert <=1950, de-biased climatology 0.096 >1950 ----
    buiz = pd.read_csv(DATA/"buizert2021_spice_accum.csv").query("year_CE>=900")
    b_yr, b_ac = buiz.year_CE.values[::-1], buiz.accum.values[::-1]
    B_CLIM = 0.096
    B_CENTER = np.array([np.interp(y,b_yr,b_ac) if y<=1950 else B_CLIM for y in KNOT_B_YEARS])

    # ---- observations (verbatim SP data prep) ----
    def _sub(df,n): return df.iloc[::max(1,len(df)//n)]
    dens=_sub(pd.read_csv(DATA/"sp19_density.csv").query("depth_m<=@H0"),45)
    age=pd.read_csv(DATA/"sp19_depth_age.csv"); age["age_yr"]=2015.0-age.year_CE
    agedf=age.query("depth_m<=@H0 and age_yr>=0"); ages=_sub(agedf,40)
    bT=_sub(pd.read_csv(DATA/"spicecore_borehole_T.csv").query("depth_m<=@H0"),40)
    # ---- ApRES velocity: FIRN_VEL_SITE selects the observable ----
    #   "pooled" (default): legacy all-site bin-median absolute velocity (chimera —
    #     kept only for r8/r9 reproduction).
    #   "<site>" (e.g. x11n6): that site's DIFFERENCED profile v(z)-v(z_ref)
    #     (antenna-offset immune) with n(rho) depth REGISTRATION (both pipelines
    #     assumed eps_ice=3.18; true depth > reported range in firn).
    #   "none": no velocity block (control for the site-selection screen).
    # Default: x11n6, the same column the ezz pin is measured from — one site's
    # strain and one site's velocity, never the pooled chimera.
    VEL_SITE = os.environ.get("FIRN_VEL_SITE", "x11n6")
    ap=pd.read_csv(DATA/"apres_vertical_velocity_processed.csv")
    ap=ap[(ap.range_m<=H0)&ap.v_smooth_m_yr.notna()&(ap.coherence>0.5)]
    _vb=np.arange(8.0,H0,8.0); _vm=0.5*(_vb[:-1]+_vb[1:]); ap["_b"]=np.digitize(ap.range_m.values,_vb)
    _vr=[(_vm[i],ap.v_smooth_m_yr[ap._b==i+1].median(),ap.v_smooth_m_yr[ap._b==i+1].std(),int((ap._b==i+1).sum())) for i in range(len(_vm))]
    _vr=[r for r in _vr if r[3]>=3]
    # n(rho) registration map (Kovacs) from the observed SP19 density profile
    _zz=np.linspace(0,200,900)
    _rho_z=np.interp(_zz,pd.read_csv(DATA/"sp19_density.csv").depth_m,
                     pd.read_csv(DATA/"sp19_density.csv").rho_kgm3*1000.0)
    _n_z=1.0+0.845e-3*np.where(_zz<130,_rho_z,917.0)
    _NICE=math.sqrt(3.18)
    _r_of_z=np.cumsum(_n_z/_NICE)*(_zz[1]-_zz[0])
    def _z_true(rng): return float(np.interp(rng,_r_of_z,_zz))
    def _n_at(z): return float(np.interp(z,_zz,_n_z))
    VEL_SRC = os.environ.get("FIRN_VEL_SRC", "zeising")  # zeising (raw-burst reprocessing; default) | authors (pipeline-A 25-m smoothed)
    def _zeising_resid_sigma(rr, y, ee, deg=2):
        """Per-site sigma by ZEISING'S OWN METHOD (apres/strain.py menke_fit).

        His method: the phase errors WEIGHT the fit, but the uncertainty comes from
        the RESIDUAL SCATTER about the fitted model --
            M = solve(G'WG, G'Wy);  resid = y - G@M;  var = np.var(resid, ddof=1)
        -- NOT from the phase errors, which are ~0.003 mm/yr and meaningless as an
        observation sigma. Including his (N-1)/(N-P) degrees-of-freedom correction
        for the P = deg+1 parameters the fit spends.

        His fit_ice sets fit_start = firn_depth_m: it EXCLUDES the firn and fits
        LINEAR ice. Our observable is entirely in the firn, where the profile is
        curved, so we keep the METHOD and use the lowest-order smooth curve the
        firn model can produce (deg 2; deg 3 moves the answer <2%, deg 1 inflates
        it ~12% by mistaking real curvature for scatter).

        This uses ONE site's data. It replaces the old sig_shape=3.5 mm/yr, which
        was the CROSS-SITE shape scatter -- i.e. pooling across columns that are
        genuinely different (real strain variation: reduced chi2 733 about the
        5-site mean, vs stated errors).
        """
        G = np.column_stack([(rr - rr.mean())**k for k in range(deg + 1)])
        w = np.where(np.isfinite(ee) & (ee > 0), 1.0/ee**2, 1.0)
        M = np.linalg.solve(G.T @ np.diag(w) @ G, G.T @ np.diag(w) @ y)
        resid = y - G @ M
        if resid.size <= 1: return 0.0
        N = float(resid.size); P = float(deg + 1)
        return float(np.sqrt(np.var(resid, ddof=1) * ((N - 1.0)/max(1.0, N - P))))

    # ---- ezz measured BELOW the firn -------------------------------------------
    # ezz is the vertical strain due to horizontal extension (less often
    # compression). It acts on FIRN AND ICE ALIKE — the model imposes it on the
    # whole column — but it is UNDIAGNOSABLE within the firn, where compaction
    # contributes to the same apparent vertical strain rate. The two enter the firn
    # velocity gradient as a SUM, so no amount of ApRES precision separates them;
    # that confound (not any operator bug) is why the inverted ezz flipped twice.
    #
    # Below close-off compaction ceases and n(z)=n_ice, so the reported range rate
    # dR/dt IS w(z) and ezz = d(dR/dt)/dz — a straight slope, with no densification
    # model, no refractive-index operator and no null space. Measure it there,
    # impose it on the whole column, and the firn dR/dt profile is freed to TEST
    # densification instead of fighting for ezz.
    #
    # Uncertainty by Zeising's own method (phase errors weight the fit, residual
    # scatter sets the covariance). Andrew 2026-07-14: pin it hard with the fit
    # error — i.e. adopt the selected column's measured strain as the core's. The
    # pin is hard, so the value and the error both have to survive scrutiny; three
    # corrections to the first cut (2026-07-15), each of which loosened it:
    #
    #  1. DEPTH WINDOW. dR/dt is not linear over the whole column: the 127-300 m
    #     slope is 13 sigma from the 127-864 m one. So the window is a real choice,
    #     and it is FIRN_EZZ_ZMIN -> FIRN_EZZ_ZMAX. See the window note below.
    #  2. SANDWICH COVARIANCE. The estimator is WEIGHTED least squares, so the
    #     covariance must be too. Zeising's menke_fit uses inv(G'G) — the OLS normal
    #     matrix — and we deliberately depart from it here: his weights would have to
    #     be near-uniform for that to hold, and ours span ~45x in this window.
    #  3. AUTOCORRELATION. The range bins are 6 m windows stepped 2 m, i.e. ~3x
    #     oversampled, so the raw residuals are correlated (lag-1 0.59 here) and
    #     an i.i.d. variance would understate the slope error. We THIN [::3] to
    #     independent samples — the same choice the firn velocity block already
    #     makes — rather than inflating by a rho-dependent factor, because thinning
    #     is checkable: the printed lag-1 of the thinned residuals shows whether it
    #     worked (it lands at -0.22). The cost is N, which the sigma then reflects.
    #
    # NOTE: Zeising's published vsr_per_year fits from cfg.firn_depth_m = 100 m
    # (apres/config.py:158), which is ABOVE SP's close-off (~127 m), so it still
    # carries ~27 m of firn compaction — that shifts x11n0 by 38%. Hence our own
    # refit over a deeper window.
    #
    # ---- THE WINDOW: 250-500 m (Andrew 2026-07-15). Measured, not assumed. ------
    # Why not start at close-off (~127 m)? Because 127 m is OUR OWN MODELLED
    # close-off — an output of the very density solution ezz feeds into — so a hard
    # pin anchored there rests on a number we chose. It is also a real lever, not a
    # nominal one: over a 127-300 m window, moving zmin 127 -> 150 moves x11n6 by
    # 18% (1.8 sigma of that pin). Starting at 250 m sits clear of the close-off
    # transition entirely and removes that circularity.
    #
    # What 250-500 m costs, stated plainly: it is FAR from the firn the pin is
    # applied to, and ezz demonstrably has depth structure, so we are buying
    # independence from our own close-off estimate at the price of extrapolating a
    # deeper strain rate up into the firn. The size of that price is measurable:
    # the 250-500 m slope (-1.157e-4) and the near-firn 127-300 m slope (-1.321e-4)
    # differ by 1.06 combined sigma, i.e. they are consistent but not identical. A
    # quadratic over 250-500 m buys nothing (weighted rss 9.78e6 -> 9.60e6), so a
    # straight line is adequate THERE; that is not evidence it extrapolates.
    #
    # Sensitivity of the new window, measured for x11n6 (sigma of the pin = 7.9e-6):
    #   zmax 400/450/500/550/600 -> -1.230e-4 .. -1.137e-4, all within 1 sigma.
    #     The zmax half IS insensitive.
    #   zmin 200/225/250/275/300 -> -1.086e-4 .. -8.66e-5; zmin=300 sits +3.7 sigma
    #     from the default. The zmin half is NOT insensitive, and this comment does
    #     not claim it is. The sub-windows stay mutually consistent given their own
    #     (larger) errors — zmin=300 is -1.4 sigma from the default on its own
    #     sigma of 1.9e-5 — but the pin's sigma does NOT span the zmin choice.
    #     250 m is chosen for independence from the modelled close-off, NOT because
    #     the answer is insensitive to it. It is not.
    EZZ_SITE = os.environ.get("FIRN_EZZ_SITE", "x11n6")   # nearest clean site, 9.05 km
    EZZ_ZMIN = float(os.environ.get("FIRN_EZZ_ZMIN", "250.0"))   # clear of the close-off transition
    EZZ_ZMAX = float(os.environ.get("FIRN_EZZ_ZMAX", "500.0"))   # ezz is not constant with depth
    def ezz_below_firn(site, zmin=EZZ_ZMIN, zmax=EZZ_ZMAX):
        zei = pd.read_csv(DATA/"apres_zeising_processed.csv")
        d = zei[(zei.site==site)&(zei.range_m>=zmin)&(zei.range_m<=zmax)].sort_values("range_m")
        d = d.iloc[::3]                     # 6-m windows stepped 2 m -> independent samples
        if len(d) < 10: raise ValueError(f"site {site}: only {len(d)} points in {zmin}-{zmax} m")
        x = d.range_m.values; y = d.dRdt_myr.values; e = d.dRdt_err_myr.values
        G = np.column_stack([np.ones_like(x), x - x.mean()])
        w = np.where(np.isfinite(e) & (e > 0), 1.0/e**2, 1.0)
        W = np.diag(w); A = G.T @ W @ G
        M = np.linalg.solve(A, G.T @ W @ y)
        resid = y - G @ M
        var = float(np.var(resid, ddof=1))
        N = float(len(y))
        # WLS sandwich, inv(A) G'W diag(r^2) W G inv(A), with the (N)/(N-P) dof
        # correction. NOT var*inv(G'G): see note 2 above.
        Ainv = np.linalg.inv(A)
        cov = (N/max(1.0, N - 2.0))*(Ainv @ (G.T @ W @ np.diag(resid**2) @ W @ G) @ Ainv)
        rc = resid - resid.mean()
        acf1 = float(np.sum(rc[:-1]*rc[1:])/np.sum(rc**2)) if len(rc) > 2 else float("nan")
        return float(M[1]), float(np.sqrt(cov[1, 1])), float(np.sqrt(var)), len(d), acf1

    def _zeising_site_points(site, zref_range=30.0):
        """One site's Zeising points and its error terms — the SINGLE definition.

        The velocity builder and error_model_audit.py both call this, so the audit
        cannot drift from the error model actually in use (it once subtracted a
        sig_shape=3.5 mm/yr that the builder had already stopped using).

        Zeising raw-burst product (proper phase errors; no 25-m smoothing — the
        pipeline-A smoothing biased firn gradients by up to 4x): 6-m fine windows
        stepped 2 m -> thin [::3] for independent samples.
          sigma_meas = the stated phase errors (~0.003 mm/yr — negligible, but free)
          sigma_repr = residual scatter about a smooth curve, per Zeising's own
                       menke_fit. Per-site: 3.06 (x11n6) .. 5.42 (x17s2) mm/yr —
                       a 1.8x spread the old single 3.5 could not express. The
                       broken sites convict themselves here: x5n2 71.2, x8n0 92.6.
        """
        zei = pd.read_csv(DATA/"apres_zeising_processed.csv")
        d = zei[(zei.site==site)&(zei.range_m>=12.0)&(zei.range_m<=112.0)].sort_values("range_m")
        rr = d.range_m.values[::3]; vv = d.dRdt_myr.values[::3]; ee = d.dRdt_err_myr.values[::3]
        if len(rr) < 6: raise ValueError(f"site {site}: only {len(rr)} zeising points")
        vref = float(np.interp(zref_range, rr, vv)); eref = float(np.interp(zref_range, rr, ee))
        sig_repr = _zeising_resid_sigma(rr, vv, ee, deg=int(os.environ.get("FIRN_VEL_FITDEG","2")))
        keep = np.abs(rr - zref_range) > 3.0
        return dict(rr=rr, vv=vv, ee=ee, keep=keep, vref=vref, eref=eref,
                    sig_repr=sig_repr, sig_meas=np.sqrt(ee[keep]**2 + eref**2))

    def site_vel_block_zeising(site, zref_range=30.0, label="v"):
        # sigma^2 = sigma_meas^2 + 2*sigma_repr^2, every term from THIS site's data.
        d = _zeising_site_points(site, zref_range)
        rr, vv, keep, sig_repr = d["rr"], d["vv"], d["keep"], d["sig_repr"]
        zt = np.array([_z_true(r) for r in rr[keep]]); zr = _z_true(zref_range)
        # LOCAL-index kinematics (Case & Kingslake 2022): dR/dt = n(z) w(z) / n_ice
        nfac = np.array([_n_at(z)/_NICE for z in zt])
        # The 2x on sigma_repr is the REFERENCE point's share: vref is interpolated
        # from this same scattered profile, so it carries a representation error of
        # the same size as every other point's, and subtracting it puts that one
        # error on the whole block at once. That is a fully-correlated offset, which
        # a DIAGONAL sigma cannot express; carrying it in quadrature is the
        # conservative diagonal approximation, so the block's effective information
        # is somewhat LESS than these independent-looking sigmas imply. (The eref
        # phase term below is the reference's measurement error — ~0.003 mm/yr, a
        # negligible stand-in for this, which is why it alone was not enough.)
        sig = np.sqrt(d["sig_meas"]**2 + 2.0*sig_repr**2)
        print(f"  [{site}] sigma_repr = {sig_repr*1000:.2f} mm/yr (own scatter, Zeising method); "
              f"sigma = {np.median(sig)*1000:.2f} mm/yr median (incl. the reference's share)")
        return ObsBlock("dRdt_diff", zt, vv[keep]-d["vref"], sig, label=label, ref_depth=zr,
                        nfac=nfac, nfac_ref=_n_at(zr)/_NICE)
    def site_vel_block(site, zref_range=30.0, sig=0.012, label="v"):
        # per-site data are ~4.2 m spaced and already 25-m smoothed: no rebinning,
        # just thin the native samples to ~8 m
        d=ap[(ap.site==site)&(ap.range_m>=12.0)&(ap.range_m<=112.0)].sort_values("range_m")
        rr=d.range_m.values[::2]; vv=d.v_smooth_m_yr.values[::2]
        if len(rr)<6: raise ValueError(f"site {site}: only {len(rr)} usable points")
        vref=float(np.interp(zref_range,rr,vv))
        keep=np.abs(rr-zref_range)>4.0
        zt=np.array([_z_true(r) for r in rr[keep]]); zr=_z_true(zref_range)
        nfac=np.array([0.5*(_n_at(z)+_n_at(zr))/_NICE for z in zt])
        return ObsBlock("dRdt_diff", zt, vv[keep]-vref, np.full(keep.sum(),sig),
                        label=label, ref_depth=zr, nfac=nfac)
    # ---- d(age)/dz slopes at full resolution, with a DATA-DERIVED sigma --------
    # sigma^2 = sigma_meas^2 + sigma_repr^2, both from our own data, no external
    # uncertainty column and no hand-picked number:
    #
    #   sigma_meas : FORMAL error propagation of the age -> slope fit (the lstsq
    #     standard error). Honest, and free -- but only ~6% of the variance.
    #   sigma_repr : measured FROM THE LAYERS -- the local scatter of d(age)/dz
    #     about the smoothest curve the model can represent. The b-knots resolve
    #     ~68 yr ~ 7 m of core, so everything finer is unfittable at ANY control
    #     setting (the sub-decadal d(age)/dz null space) and must live in sigma.
    #     For WEIGHTING the split between real layering and picking noise is
    #     irrelevant: what matters is the total variance the model cannot explain.
    #
    # This REPLACES `max(se, 4% of value)`. That floor overrode the formal
    # propagation at 90% of points and was ~2.4x too small, which handed dage 78%
    # of the objective (gradients go as 1/sigma^2) -- an inversion driven by layer
    # noise. The result is depth-dependent (3.9x across the column; ratio-to-old
    # 1.98 shallow, 2.82 at 60-80 m), so no scalar can express it.
    # FIRN_SIG_DAGE_LEGACY=1 restores the old floor form to reproduce archived
    # r8/r9/r10 MAPs (with FIRN_SIG_DAGE_SCALE=2.2 for r8's J=81.31 exactly).
    _ad,_aa=agedf.depth_m.values,agedf.age_yr.values
    SIG_DAGE_SCALE=float(os.environ.get("FIRN_SIG_DAGE_SCALE","1.0"))
    _LEGACY_DAGE=os.environ.get("FIRN_SIG_DAGE_LEGACY","0")=="1"
    dc,do,dse=[],[],[]
    for c in np.arange(6.0,H0-1.0+1e-9,1.0):
        s=np.abs(_ad-c)<=0.6
        if s.sum()>=4:
            A=np.vstack([_ad[s]-c,np.ones(s.sum())]).T; coef,res,*_=np.linalg.lstsq(A,_aa[s],rcond=None); nn=s.sum()
            se=math.sqrt(max(float(res[0]) if len(res) else 0.0,1e-12)/(nn-2)/max(np.sum((_ad[s]-c)**2),1e-12))
            dc.append(c); do.append(coef[0]); dse.append(se)
    dc=np.array(dc); do=np.array(do); dse=np.array(dse)
    if _LEGACY_DAGE:
        dsg=np.maximum(dse,0.04*np.abs(do))*SIG_DAGE_SCALE
    else:
        _SM_M=float(os.environ.get("FIRN_DAGE_SMOOTH_M","7.0"))   # b-knot scale, in m of core
        _w=max(3,int(round(_SM_M/1.0))|1)
        _smooth=np.convolve(np.pad(do,_w//2,mode="edge"),np.ones(_w)/_w,mode="valid")[:len(do)]
        _LW=21                                                    # window for the LOCAL sd
        _repr=np.array([np.std((do-_smooth)[max(0,i-_LW//2):min(len(do),i+_LW//2+1)])
                        for i in range(len(do))])
        dsg=np.sqrt(dse**2+_repr**2)*SIG_DAGE_SCALE
        print(f"dage sigma (data-derived): median {np.median(dsg):.3f} yr/m "
              f"= {100*np.median(dsg/np.abs(do)):.1f}% of value "
              f"(meas {100*np.median(dse**2/dsg**2):.0f}% of variance; "
              f"depth spread {dsg.max()/dsg.min():.1f}x)")
    P0T_ref=273.15; c_i=2009.0; T_ref=273.15   # match FirnParameters (c_i, T_ref)
    from firnpack.models.firn import FirnParameters as _FP
    _p=_FP(); c_i=float(_p.c_i); T_ref=float(_p.T_ref)
    obs=[
        ObsBlock("rho", dens.depth_m.values, dens.rho_kgm3.values*1000.0, 15.0+0.03*dens.rho_kgm3.values*1000.0, label="rho"),
        ObsBlock("dagedz", dc, do, dsg, label="dage"),
        ObsBlock("enthalpy", bT.depth_m.values, c_i*(bT.T_C.values+T_SHIFT+273.15-T_ref), c_i*np.full(len(bT),0.2), label="T"),
    ]
    # ---- the absolute-age block is DELETED (2026-07-14) ------------------------
    # It was the SAME MEASUREMENTS as dage: `ages = _sub(agedf,40)` subsampled the
    # very frame the dage windows are built from -- 41 of its 42 points fall INSIDE
    # a dage window, and dage touches 97% of agedf against the age block's 4%.
    # It also did nothing: its invented sigma (10 yr + 3% -> 46 yr at 130 m) gave
    # it 0.1% of the objective, and even at a STRICT measurement error the model
    # fits it at 0.63 sigma, because dage already pins the layers (derived
    # representation error = 0).
    # And its covariance was wrong in principle: SP19 is LAYER-COUNTED (rows are
    # exactly 1.000 yr apart), so absolute age is a CUMULATIVE count whose errors
    # accumulate down the core -- not 42 independent draws. Layer thickness (dage)
    # is the primitive with ~independent errors; differencing is the whitening
    # transform for that cumulative structure. So dage is the right single
    # representation, and the age block was a redundant, mis-specified copy.
    # FIRN_AGE_BLOCK=1 restores it to reproduce archived r8/r9/r10 MAPs.
    if os.environ.get("FIRN_AGE_BLOCK","0")=="1":
        obs.insert(1, ObsBlock("age", ages.depth_m.values, ages.age_yr.values*YEAR_S,
                               (10.0+0.03*ages.age_yr.values)*YEAR_S, label="age"))
        print("age block RESTORED (legacy reproduction mode)")
    if VEL_SITE == "pooled":
        obs.append(ObsBlock("velocity", np.array([r[0] for r in _vr]), np.array([r[1] for r in _vr]),
                   np.maximum(np.array([r[2]/max(r[3],1)**0.5 for r in _vr]),0.01), label="v"))
    elif VEL_SITE != "none":
        # "+"-separated multi-site: one differenced block per site (distinct labels;
        # sites are genuinely different columns — never pool across sites)
        _sl = VEL_SITE.split("+")
        _build = site_vel_block_zeising if VEL_SRC == "zeising" else site_vel_block
        for _s in _sl:
            obs.append(_build(_s, label="v" if len(_sl) == 1 else f"v_{_s}"))
            print(f"velocity block[{VEL_SRC}]: site {_s}, differenced + n(rho)-registered "
                  f"({obs[-1].n} pts, ref z={obs[-1].ref_depth:.1f} m)")
    if os.environ.get("FIRN_SEAS", "0") == "1":
        # USP50 seasonal-amplitude damping IN the inversion (WKB ln-ratio operator
        # through the on-tape k law; replaces the offline k_snow pin — the OSSE
        # matrix shows mean profiles carry no k_snow info, the amplitude does).
        # obs pre-corrected for the WKB-vs-exact operator bias at s*=1.29.
        sa = pd.read_csv(DATA/"usp50_seasonal_lnratio.csv")
        ud = pd.read_csv(DATA/"usp50_density_nicl.csv")
        obs.append(ObsBlock("seas_lnamp", sa.zeff.values, sa.lnr.values, sa.sig.values,
                            label="seas", ref_depth=float(sa.zref.iloc[0]),
                            aux_z=ud.depth.values, aux_val=ud.Density.values))
        print(f"seasonal block: USP50 ln-amplitude ratios ({obs[-1].n} pts, "
              f"ref z={obs[-1].ref_depth:.2f} m), k_snow prior WIDENED")

    # ---- controls ----
    if EZZ_SITE != "none":
        _ez, _ez_e, _ez_sc, _ez_n, _ez_acf = ezz_below_firn(EZZ_SITE)
        print(f"ezz PINNED from {EZZ_SITE}, {EZZ_ZMIN:.0f}-{EZZ_ZMAX:.0f} m: "
              f"{_ez:.4e} +- {_ez_e:.2e} /yr (WLS sandwich s.e.; effective N={_ez_n} "
              f"after [::3] thinning, resid lag-1 {_ez_acf:+.2f}, deep scatter "
              f"{_ez_sc*1000:.2f} mm/yr)")
        _EZZ_CTRL = ScalarCtrl("ezz_yr", _ez, _ez, _ez_e, -3.0e-4, 2.0e-4)
    else:
        # legacy: free ezz, diagnosed (badly) from the firn profile
        _EZZ_CTRL = ScalarCtrl("ezz_yr", 0.0, 0.0, 1.0e-4, -3.0e-4, 2.0e-4)
        print("ezz FREE (legacy: diagnosed from the firn, where it is confounded with compaction)")
    scalars=[
        ScalarCtrl("hl_k0",10.79,10.79,1.0,0.5,500,log=True),
        ScalarCtrl("hl_k1",570.7,570.7,1.0,10,50000,log=True),
        ScalarCtrl("hl_Ea1",10432.,10432.,0.5,3000,40000,log=True),
        ScalarCtrl("hl_Ea2",21875.,21875.,0.5,5000,80000,log=True),
        (ScalarCtrl("k_snow_scale",1.29,1.0,0.5,0.02,5.0,log=True)     # SEAS: weak prior, damping data ON TAPE
         if os.environ.get("FIRN_SEAS","0")=="1" else
         ScalarCtrl("k_snow_scale",1.29,1.29,0.08,0.02,5.0,log=True)), # USP50 pin (offline)
        ScalarCtrl("k_firn_scale",1.0,1.0,0.15,0.5,2.0,log=True),
        # Basal thermal BC control (Andrew, 2026-07-11): parameterize by the basal
        # TEMPERATURE GRADIENT G_base (K/m, warming-downward positive) — the
        # quantity the borehole data actually constrain at a truncation boundary —
        # rather than the flux Q (which entangles the BC with the conductivity
        # scales; flux is now the DERIVED product q = k*G). FIRN_BASAL=Q restores
        # the r8-era flux control (needed to reproduce archived r8 exactly).
        (ScalarCtrl("Q_base",0.0,0.0,0.05,-0.2,0.2)
         if os.environ.get("FIRN_BASAL","G")=="Q" else
         ScalarCtrl("G_base",-0.005,0.0,0.025,-0.1,0.1)),
        # ezz: PINNED to the below-firn strain of one column, at its own fit error
        # (Andrew 2026-07-14). The old free control (center 0, sigma 1e-4) asked the
        # firn to diagnose a quantity it cannot separate from compaction; with this
        # pin, ezz is effectively known and the firn dR/dt profile becomes a clean
        # test of densification. FIRN_EZZ_SITE=none restores the free control.
        _EZZ_CTRL,
        ScalarCtrl("s2_shape",1.0,1.0,0.3,0.3,2.0,log=True),
        ScalarCtrl("b_off",0.0,0.0,0.25,-0.8,0.8),
    ]
    cfg=SiteConfig(name="SouthPole", out_dir=OUT, tag=TAG,
        H_col=H_COL, NZ=NZ, spin_years=SPIN, dt_years=DT, rho_surf=350.0,
        rho_ic_deep=820.0, rho_ic_scale=25.0, conductivity_law="calonne2019",
        scalars=scalars,
        T_knots=KnotCtrl(KNOT_YEARS, T_CTR, T_CTR, 0.6, -48.0+T_SHIFT, -43.0+T_SHIFT, log=False, invert=True, name="Tk"),
        b_knots=KnotCtrl(KNOT_B_YEARS, B_CENTER, B_CENTER, 0.20, 0.03, 0.20, log=True, invert=True, name="b"),
        b_off_era_year=1950.0, obs=obs, max_iter=int(os.environ.get("FIRN_MAX_ITER","80")))

    return SimpleNamespace(**{k: v for k, v in locals().items()
                              if not k.startswith("__")})

