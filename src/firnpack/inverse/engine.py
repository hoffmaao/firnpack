"""firnpack.inverse.engine — the shared adjoint assimilation engine.

Faithful refactor of sp_joint_assimilate_r5.py: same lean (H, rho, w, age)
theta-scheme stepper, Gaussian-kernel scalar misfits, per-point chi^2,
per-eval solver rebuild (NaN-robust), penalty surface BCs, and L-BFGS-B in
prior-sigma-scaled coordinates. Generalized so each observable and each scalar
control is independently on/off, and temperature/accumulation knots are either
inverted or prescribed as forcing. The South Pole config reproduces r8.

assimilate(cfg, mode) with mode in {"verify","optimize","forward"}:
  "verify"   -> deterministic replay (+ optional FD check via fd_names); no opt
  "optimize" -> full L-BFGS-B; writes cfg.out_dir/cfg.tag.json
  "forward"  -> single forward at the warm-start x0; returns diagnostics + J
Returns a dict of results.
"""
from __future__ import annotations
import functools, json, math, time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import firedrake as fd
from firedrake.adjoint import (Control, continue_annotation, pause_annotation,
                               stop_annotating, get_working_tape)
from pyadjoint import compute_gradient
from firnpack.models.firn import FirnParameters, FirnModel
from firnpack.physics.densification import herron_langway as _hl
from firnpack.constants import year as YEAR_S
from firnpack.inverse.statistics import StatisticsProblem, MaximumProbabilityEstimator

herron_langway = functools.partial(_hl, smooth=True)
K_ICE_BASE = 2.1

# names that map to a FirnParameters constructor kwarg (value or exp(log))
_PARAM_KW = {
    "hl_k0": "hl_k0_prefactor", "hl_k1": "hl_k1_prefactor",
    "hl_Ea1": "hl_Ea_stage1", "hl_Ea2": "hl_Ea_stage2",
    "s2_shape": "hl_stage2_shape", "Q_base": "basal_heat_flux_W_m2",
    "k_snow_scale": "k_snow_scale", "k_firn_scale": "k_firn_scale",
    "k_factor": "k_factor",   # handled specially (k_ice = base*k_factor)
}


def assimilate(cfg, mode="optimize", warm=None, fd_names=None, fd_h=1e-3,
               verbose=True):
    P0 = FirnParameters()
    c_i, T_ref, rho_i = float(P0.c_i), float(P0.T_ref), float(P0.rho_i)
    SID = cfg.surface_id
    log = print if verbose else (lambda *a, **k: None)

    # ---- mesh ----
    mesh = fd.IntervalMesh(cfg.NZ, 0.0, 1.0)
    xphys = cfg.H_col * (1.0 - (1.0 - fd.SpatialCoordinate(mesh)[0]) ** cfg.stretch_p)
    mesh.coordinates.assign(fd.Function(mesh.coordinates.function_space())
                            .interpolate(fd.as_vector([xphys])))
    V = fd.FunctionSpace(mesh, "CG", 1); R = fd.FunctionSpace(mesh, "R", 0)
    dx = fd.dx(domain=mesh); ds = fd.ds(domain=mesh)
    xc = fd.SpatialCoordinate(mesh)[0]; depth = cfg.H_col - xc; psi = fd.TestFunction(V)
    def mk(v, nm): f = fd.Function(R, name=nm); f.assign(float(v)); return f
    with stop_annotating():
        dlen = float(fd.assemble(fd.Constant(1.0) * dx)) + 1e-30

    # ---- kernels per obs block ----
    def make_kernels(depths, wmax):
        ks = []
        with stop_annotating():
            for d in depths:
                s = float(np.clip(0.04 * d + 0.5, 0.5, wmax))
                phi = fd.Function(V).interpolate(fd.exp(-0.5 * ((xc - (cfg.H_col - d)) / s) ** 2))
                phi.dat.data[:] /= float(fd.assemble(phi * dx)) + 1e-30
                ks.append(phi)
        return ks
    for ob in cfg.obs:
        ob._ker = make_kernels(ob.depths, ob.wmax)
        ob._ker_ref = make_kernels([ob.ref_depth], ob.wmax)[0] if ob.ref_depth is not None else None
        ob._q = None; ob._win = None; ob._aux_fn = None
        if ob.kind == "seas_lnamp":
            # erf-edged windows over depth [ref, z_j] for the WKB damping
            # integral pred_j = -<q * W_j>; edge width fixed for all sensors
            sw, rt2 = 0.6, math.sqrt(2.0)
            with stop_annotating():
                x2 = cfg.H_col - float(ob.ref_depth)
                ob._win = [fd.Function(V).interpolate(
                    0.5*(fd.erf((xc-(cfg.H_col-float(dj)))/(rt2*sw))
                         - fd.erf((xc-x2)/(rt2*sw)))) for dj in ob.depths]
                if ob.aux_z is not None:
                    f = fd.Function(V)
                    nd = cfg.H_col - mesh.coordinates.dat.data_ro.reshape(-1)
                    f.dat.data[:] = np.interp(nd, np.asarray(ob.aux_z, float),
                                              np.asarray(ob.aux_val, float))
                    ob._aux_fn = f
    _prior = fd.Function(V)

    # ---- timeline / brackets ----
    n_steps = int(cfg.spin_years / cfg.dt_years)
    step_years = cfg.present_year - (n_steps - 1 - np.arange(n_steps)) * cfg.dt_years

    def make_bracket(knot_years):
        br = []
        for yr in step_years:
            if yr <= knot_years[0]: br.append((0, None))
            elif yr >= knot_years[-1]: br.append((len(knot_years) - 1, None))
            else:
                j = int(np.searchsorted(knot_years, yr) - 1)
                br.append((j, float((yr - knot_years[j]) / (knot_years[j + 1] - knot_years[j]))))
        return br

    # ---- controls ----
    # scalar controls: active -> R-space fn + in ctrl list; inactive -> fixed fn
    ctrl_fns, ctrl_meta = [], []      # meta: (name, log, prior_center_internal, prior_sigma)
    scal_fn = {}                      # name -> fn (active or fixed)
    scal_by_name = {s.name: s for s in cfg.scalars}
    # scalar warm values live under warm["m_map"] (results JSON) or flat in warm
    warm_scal = (warm.get("m_map", warm) if warm else None)
    for s in cfg.scalars:
        # only warm-start ACTIVE controls; fixed params stay at their configured init
        has_w = warm_scal is not None and s.name in warm_scal and s.active
        wv = warm_scal[s.name] if has_w else s.init
        v0 = math.log(wv) if s.log else wv
        f = mk(v0, s.name); scal_fn[s.name] = f
        if s.active:
            ctrl_fns.append(f)
            ctr = math.log(s.center) if s.log else s.center
            ctrl_meta.append((s.name, s.log, ctr, s.sigma))

    # knot controls
    def knot_fns(kc, val_key, yr_key):
        if kc is None: return [], []
        vals = kc.init.copy()
        if warm and val_key in warm and yr_key in warm:
            wy = np.asarray(warm[yr_key]); wv = np.asarray(warm[val_key])
            if kc.log: vals = np.exp(np.interp(kc.years, wy, np.log(wv)))
            else: vals = np.interp(kc.years, wy, wv)
        fns = [mk(math.log(v) if kc.log else v, f"{kc.name}{i}") for i, v in enumerate(vals)]
        return fns, vals
    T_fns, T_vals = knot_fns(cfg.T_knots, "T_knots", "knot_years")
    b_fns, b_vals = knot_fns(cfg.b_knots, "b_knots", "b_knot_years")
    if cfg.T_knots is not None and cfg.T_knots.invert:
        for i, f in enumerate(T_fns):
            ctrl_fns.append(f); ctrl_meta.append((f"Tk{i}", False, float(cfg.T_knots.center[i]), cfg.T_knots.sigma))
    if cfg.b_knots is not None and cfg.b_knots.invert:
        for i, f in enumerate(b_fns):
            ctrl_fns.append(f); ctrl_meta.append((f"b{int(cfg.b_knots.years[i])}", True,
                                                  math.log(float(cfg.b_knots.center[i])), cfg.b_knots.sigma))
    bracket_T = make_bracket(cfg.T_knots.years) if cfg.T_knots is not None else None
    bracket_B = make_bracket(cfg.b_knots.years) if cfg.b_knots is not None else None
    N_CTRL = len(ctrl_fns)
    x0 = np.array([float(c.dat.data_ro[0]) for c in ctrl_fns])
    lb = np.array([(math.log(scal_by_name[m[0]].lo) if m[1] else scal_by_name[m[0]].lo)
                   if m[0] in scal_by_name else
                   (math.log(cfg.b_knots.lo) if m[1] else cfg.T_knots.lo) for m in ctrl_meta])
    ub = np.array([(math.log(scal_by_name[m[0]].hi) if m[1] else scal_by_name[m[0]].hi)
                   if m[0] in scal_by_name else
                   (math.log(cfg.b_knots.hi) if m[1] else cfg.T_knots.hi) for m in ctrl_meta])
    scale = np.array([m[3] for m in ctrl_meta])

    # ---- params + model ----
    def val(nm):
        f = scal_fn.get(nm)
        if f is None: return None
        s = scal_by_name[nm]
        return fd.exp(f) if s.log else f
    pk = {}
    for nm in ["hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "s2_shape", "Q_base"]:
        if nm in scal_fn: pk[_PARAM_KW[nm]] = val(nm)
    if cfg.conductivity_law == "calonne2019":
        pk["conductivity_law"] = "calonne2019"
        if "k_snow_scale" in scal_fn: pk["k_snow_scale"] = val("k_snow_scale")
        if "k_firn_scale" in scal_fn: pk["k_firn_scale"] = val("k_firn_scale")
    elif "k_factor" in scal_fn:
        pk["k_ice"] = K_ICE_BASE * val("k_factor")
    if cfg.deep_cutoff_rho is not None:
        pk["hl_deep_cutoff_rho"] = cfg.deep_cutoff_rho
    params = FirnParameters(**pk)
    model = FirnModel(params, densification_rate_fn=herron_langway)

    # ---- fields + stepper ----
    bdot0 = float(cfg.b_knots.init[0]) if cfg.b_knots is not None else 0.085
    ws0 = -bdot0 * rho_i / cfg.rho_surf / YEAR_S
    H_f=fd.Function(V); rho_f=fd.Function(V); w_f=fd.Function(V); age_f=fd.Function(V)
    H_o=fd.Function(V); rho_o=fd.Function(V); w_o=fd.Function(V); age_o=fd.Function(V)
    Ts_eff=fd.Function(V); b_eff=fd.Function(V)
    rs=mk(cfg.rho_surf,"rs"); dt_r=mk(cfg.dt_years*YEAR_S,"dt")
    bc_rho=fd.DirichletBC(V,rs,SID); bc_age=fd.DirichletBC(V,mk(0.0,"a0"),SID)
    bdot_mass=b_eff*rho_i/P0.spy
    ws_expr=b_eff*fd.Constant(-rho_i/(cfg.rho_surf*YEAR_S))
    Tk_C = fd.Constant(273.15)
    ezz_fn = scal_fn.get("ezz_yr")            # None -> no dynamic strain
    def Ts_expr_for(k):
        j, f = bracket_T[k]
        base = (lambda i: T_fns[i] + Tk_C) if cfg.T_knots.invert else \
               (lambda i: fd.Constant(float(cfg.T_knots.init[i])) + Tk_C)
        if not cfg.T_knots.invert:  # prescribed floats interpolated per step
            if f is None: return fd.Constant(float(cfg.T_knots.init[j])) + Tk_C
            v = (1.0-f)*cfg.T_knots.init[j] + f*cfg.T_knots.init[j+1]
            return fd.Constant(float(v)) + Tk_C
        if f is None: return T_fns[j] + Tk_C
        return fd.Constant(1.0-f)*T_fns[j] + fd.Constant(f)*T_fns[j+1] + Tk_C
    def b_expr_for(k):
        j, f = bracket_B[k]
        if cfg.b_knots.invert:
            if f is None: return fd.exp(b_fns[j])
            return fd.exp(fd.Constant(1.0-f)*b_fns[j] + fd.Constant(f)*b_fns[j+1])
        else:
            if f is None: return fd.Constant(float(cfg.b_knots.init[j]))
            return fd.Constant(float((1.0-f)*cfg.b_knots.init[j] + f*cfg.b_knots.init[j+1]))
    sp_lin={"ksp_type":"preonly","pc_type":"lu"}
    Ht=fd.TrialFunction(V); rt=fd.TrialFunction(V); wt=fd.TrialFunction(V); at=fd.TrialFunction(V)
    with stop_annotating():
        rho_ic = fd.Function(V).interpolate(
            cfg.rho_surf + (cfg.rho_ic_deep - cfg.rho_surf) * (1.0 - fd.exp(-depth / cfg.rho_ic_scale)))
    # Basal thermal BC: if a "G_base" control exists (basal temperature
    # gradient, K/m, positive = warming downward), the flux is DERIVED from
    # the model's own conductivity at the base: q_up = k(rho, T_old) * G.
    # This decouples the BC control from the conductivity scales (with Q_base
    # the same flux implies different gradients as k changes — a built-in
    # posterior correlation). T lagged via H_o (consistent with the K lag).
    if "G_base" in scal_fn:
        k_expr = model.thermal_diffusivity(H_o, rho_f) * rho_f * c_i
        params.basal_heat_flux_W_m2 = k_expr * scal_fn["G_base"]
    SLV = {}
    def rebuild():
        F_H=model.enthalpy_form(Ht,H_o,rho_f,w_f,psi,dt_r)+fd.Constant(cfg.beta_enth)*(Ht-c_i*(Ts_eff-fd.Constant(T_ref)))*psi*ds(SID)
        SLV["H"]=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(F_H),fd.rhs(F_H),H_f,constant_jacobian=False),solver_parameters=sp_lin)
        T_expr=model.temperature_from_enthalpy(H_f)
        F_rho,_=model.density_form(rt,rho_o,T_expr,w_f,None,bdot_mass,psi,dt_r)
        SLV["rho"]=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(F_rho),fd.rhs(F_rho),rho_f,bcs=[bc_rho],constant_jacobian=False),solver_parameters=sp_lin)
        drhodt=herron_langway(rho_f,T_expr,params=params,bdot=bdot_mass)
        div_h = fd.Constant(0.0) if ezz_fn is None else fd.Constant(-1.0/YEAR_S)*ezz_fn
        dW=model.velocity_delta(wt,w_o,rho_f,drhodt,psi,regularization=1e-3,horizontal_divergence=div_h) \
           + fd.Constant(cfg.beta_w)*(wt-ws_expr)*psi*ds(SID)
        SLV["w"]=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(dW),fd.rhs(dW),w_f,constant_jacobian=False),solver_parameters=sp_lin)
        F_age=model.age_form(at,age_o,w_f,w_o,psi,dt_r)
        SLV["age"]=fd.LinearVariationalSolver(fd.LinearVariationalProblem(fd.lhs(F_age),fd.rhs(F_age),age_f,bcs=[bc_age],constant_jacobian=False),solver_parameters=sp_lin)

    b_off_fn = scal_fn.get("b_off")
    T0_mean = float(np.mean(cfg.T_knots.center)) if cfg.T_knots is not None else -45.0

    # ---- the forward split into icepack-style simulation / loss / prior ----
    # simulation(controls) -> state (the settled column); loss_functional(state)
    # -> misfit; regularization(controls) -> prior. forward() composes them and
    # is bit-identical to the original monolithic objective: same operations in
    # the same order, so J = loss + reg reproduces the frozen South Pole MAP.
    # _bk carries the per-eval bookkeeping (preds, rms diag) that the misfit
    # produces and the result serialisation reads back.
    _bk = {}

    def simulation(controls=None):
        H_f.assign(c_i*(T0_mean+273.15-T_ref)); H_o.assign(H_f)
        rho_f.assign(rho_ic); rho_o.assign(rho_f)
        w_f.assign(ws0); w_o.assign(w_f); age_f.assign(0.0); age_o.assign(0.0)
        for k in range(n_steps):
            Ts_eff.interpolate(Ts_expr_for(k)); b_eff.interpolate(b_expr_for(k))
            SLV["H"].solve(); SLV["rho"].solve(); SLV["w"].solve(); SLV["age"].solve()
            H_o.assign(H_f); rho_o.assign(rho_f); w_o.assign(w_f); age_o.assign(age_f)
        with stop_annotating():
            if not all(np.isfinite(fld.dat.data_ro).all() for fld in (H_f,rho_f,w_f,age_f)):
                raise RuntimeError("field blowup: non-finite state")
        return SimpleNamespace(H=H_f, rho=rho_f, w=w_f, age=age_f)

    def loss_functional(state):
        w_srf = fd.assemble(w_f*ds(SID))
        for ob in cfg.obs:
            if ob.kind == "seas_lnamp":
                # WKB damping rate q(z) = sqrt(w rho c / 2k) through the ON-TAPE
                # conductivity law (k_snow/k_firn scales), damped in the
                # instrumented hole's density column; CG1-interpolated before
                # assembling (nonlinear-UFL rule)
                rr = ob._aux_fn if ob._aux_fn is not None else rho_f
                kk = model.thermal_diffusivity(H_f, rr)*rr*c_i
                ob._q = fd.Function(V).interpolate(
                    fd.sqrt(fd.Constant(2.0*math.pi/YEAR_S)*rr*c_i/(2.0*kk)))
        J = 0.0; diag = {}; preds = {}
        for ob in cfg.obs:
            sq = 0.0; Jo = 0.0; pl = []
            w_ref = fd.assemble(w_f*ob._ker_ref*dx) if ob._ker_ref is not None else None
            for j, (phi, o, s) in enumerate(zip(ob._ker, ob.obs, ob.sig)):
                if ob.kind == "rho":      pred = fd.assemble(rho_f*phi*dx)
                elif ob.kind == "age":    pred = fd.assemble(age_f*phi*dx)
                elif ob.kind == "enthalpy": pred = fd.assemble(H_f*phi*dx)
                elif ob.kind == "dagedz": pred = -fd.assemble(age_f.dx(0)*phi*dx)/YEAR_S
                elif ob.kind == "velocity": pred = (fd.assemble(w_f*phi*dx)-w_srf)/cfg.n_ice*YEAR_S
                elif ob.kind == "dRdt_diff":
                    nf = float(ob.nfac[j]) if ob.nfac is not None else 1.0
                    nfr = float(ob.nfac_ref) if ob.nfac_ref is not None else nf
                    pred = (nfr*w_ref - nf*fd.assemble(w_f*phi*dx))*YEAR_S
                elif ob.kind == "seas_lnamp":
                    pred = -fd.assemble(ob._q*ob._win[j]*dx)
                else: raise ValueError(f"unknown obs kind {ob.kind}")
                r = (pred-float(o))/float(s); Jo = Jo + r*r
                with stop_annotating(): sq += float(r)**2; pl.append(float(pred))
            J = J + ob.weight*0.5*Jo
            diag["rms_"+ob.label] = (sq/max(ob.n,1))**0.5
            preds[ob.label] = pl
        _bk["preds"] = preds; _bk["diag"] = diag
        return J

    def regularization(controls):
        Jp = 0.0
        for c,(nm,islog,ctr,sig) in zip(controls, ctrl_meta):
            if nm.startswith("b") and nm[1:].isdigit() and cfg.b_off_era_year is not None \
               and b_off_fn is not None and float(nm[1:]) > cfg.b_off_era_year:
                resid = c - fd.Constant(ctr) - b_off_fn
            else:
                resid = c - fd.Constant(ctr)
            _prior.interpolate(resid/fd.Constant(sig))
            Jp = Jp + 0.5*fd.assemble(_prior*_prior*dx)/dlen
        return Jp

    def forward():
        state = simulation(ctrl_fns)
        J = loss_functional(state) + regularization(ctrl_fns)
        forward._preds = _bk["preds"]
        diag = _bk["diag"]
        with stop_annotating():
            diag["rho_max"] = float(rho_f.dat.data_ro.max())
        forward._diag = diag
        return J

    tape = get_working_tape()
    def run_forward_only():
        tape.clear_tape(); continue_annotation(); rebuild()
        for c,v in zip(ctrl_fns,x0): c.assign(float(v))
        J = float(forward()); pause_annotation(); return J

    def snapshot():
        """Everything a figure needs from the state a forward() just left behind.

        The prognostic Functions and forward._preds hold that state, so this
        must run directly after a forward solve. Both "forward" mode and the
        final MAP evaluation in "optimize" go through here, so a results JSON
        carries the same block arrays however it was produced -- which is what
        lets the plot scripts read output/ instead of re-solving.
        """
        with stop_annotating():
            xs_ = mesh.coordinates.dat.data_ro.reshape(-1); dprof = cfg.H_col - xs_
            o = np.argsort(dprof)
            prof = dict(depth=dprof[o].tolist(),
                        rho=rho_f.dat.data_ro[o].tolist(),
                        age_yr=(age_f.dat.data_ro[o]/YEAR_S).tolist(),
                        T_C=(fd.Function(V).interpolate(model.temperature_from_enthalpy(H_f)).dat.data_ro[o]-273.15).tolist(),
                        w_m_yr=(w_f.dat.data_ro[o]*YEAR_S).tolist())
            pvals = {nm: (math.exp(float(scal_fn[nm].dat.data_ro[0])) if scal_by_name[nm].log
                          else float(scal_fn[nm].dat.data_ro[0])) for nm in scal_fn}
        obs_out = [dict(kind=ob.kind, label=ob.label, depths=ob.depths.tolist(),
                        obs=ob.obs.tolist(), sig=ob.sig.tolist(),
                        pred=forward._preds[ob.label]) for ob in cfg.obs]
        return prof, pvals, obs_out

    if mode == "forward":
        J = run_forward_only()
        prof, pvals, obs_out = snapshot()
        return dict(J=J, diag=forward._diag, n_ctrl=N_CTRL, profiles=prof,
                    params=pvals, n_steps=n_steps, ws0=ws0, T0_mean=T0_mean,
                    obs=obs_out)

    if mode == "verify":
        J1 = run_forward_only(); J2 = run_forward_only()
        log(f"  replay J1={J1:.8f} J2={J2:.8f} match={abs(J1-J2)<1e-10}")
        log("  rms: " + " ".join(f"{k[4:]}={v:.2f}" for k,v in forward._diag.items() if k.startswith("rms_")))
        out = dict(J=J1, match=abs(J1-J2)<1e-10, diag=forward._diag)
        if fd_names:
            names = [m[0] for m in ctrl_meta]
            tape.clear_tape(); continue_annotation(); rebuild()
            for c,v in zip(ctrl_fns,x0): c.assign(float(v))
            J0f=forward(); dJ=compute_gradient(J0f,[Control(c) for c in ctrl_fns]); pause_annotation()
            g=np.array([float(gi.dat.data_ro[0])*dlen for gi in dJ])
            def Jat(xv):
                tape.clear_tape(); continue_annotation(); rebuild()
                for c,v in zip(ctrl_fns,xv): c.assign(float(v))
                Jv=float(forward()); pause_annotation(); return Jv
            ratios = {}
            for i,nm in enumerate(names):
                if nm not in fd_names: continue
                h = 2e-6 if nm=="ezz_yr" else fd_h
                ep=x0.copy(); ep[i]+=h; em=x0.copy(); em[i]-=h
                fdg=(Jat(ep)-Jat(em))/(2*h); ratio=g[i]/fdg if abs(fdg)>1e-12 else float('nan')
                ratios[nm]=ratio; log(f"    {nm:12s} adj={g[i]:+.3e} FD={fdg:+.3e} ratio={ratio:+.4f}")
            out["fd_ratios"]=ratios
        return out

    # ---- optimize (via the icepack-style estimator) ----
    # The forward split above is composed into a StatisticsProblem and minimised
    # by MaximumProbabilityEstimator (scipy L-BFGS-B backend). reset reproduces
    # the per-eval tape-clear + solver rebuild; grad_scale=dlen recovers the true
    # R-space gradient. This is the same objective, scaling, bounds and optimiser
    # the historical inline loop used, so the trajectory is bit-identical.
    _eval_t0 = [0.0]
    def _reset():
        tape.clear_tape(); continue_annotation(); rebuild()
        _eval_t0[0] = time.perf_counter()
    def _log_eval(n, Jv, g, controls):
        d = _bk.get("diag", {})
        rms = " ".join(f"{k[4:]}={v:.2f}" for k, v in d.items() if k.startswith("rms_"))
        log(f"  [{n:03d}] J={Jv:.4f} ({rms}) |g|={np.linalg.norm(g):.1e} "
            f"({time.perf_counter()-_eval_t0[0]:.0f}s)")
    problem = StatisticsProblem(
        simulation, loss_functional, regularization, ctrl_fns,
        bounds=list(zip(lb, ub)), scale=scale, copy_controls=False,
        reset=_reset, grad_scale=dlen)
    est = MaximumProbabilityEstimator(
        problem, method="L-BFGS-B", max_iterations=cfg.max_iter,
        ftol=cfg.ftol, gtol=cfg.gtol, maxls=cfg.maxls,
        verbose=verbose, callback=_log_eval)
    log(f"\nOptimize {cfg.name} (L-BFGS-B, {N_CTRL} ctrls, max {cfg.max_iter}):")
    est.solve()
    res = est.result
    J_hist = est.J_hist
    x_map = np.array([float(c.dat.data_ro[0]) for c in ctrl_fns])
    names=[m[0] for m in ctrl_meta]; islogs=[m[1] for m in ctrl_meta]
    m_map={}; out_extra={}
    for i,(nm,il) in enumerate(zip(names,islogs)):
        if nm in scal_by_name: m_map[nm]=math.exp(x_map[i]) if il else x_map[i]
    result=dict(name=cfg.name, J=float(res.fun), message=str(res.message),
                m_map=m_map, n_ctrl=N_CTRL, J_hist=J_hist)
    if cfg.T_knots is not None and cfg.T_knots.invert:
        i0=[m[0] for m in ctrl_meta].index("Tk0")
        result["T_knots"]=x_map[i0:i0+len(cfg.T_knots.years)].tolist()
        result["knot_years"]=cfg.T_knots.years.tolist()
    if cfg.b_knots is not None and cfg.b_knots.invert:
        bnames=[f"b{int(y)}" for y in cfg.b_knots.years]
        i0=[m[0] for m in ctrl_meta].index(bnames[0])
        result["b_knots"]=np.exp(x_map[i0:i0+len(bnames)]).tolist()
        result["b_knot_years"]=cfg.b_knots.years.tolist()
        result["b_prior_centers"]=cfg.b_knots.center.tolist()
    # One more forward, at the MAP, purely so the results JSON carries what a
    # figure needs: per-block obs/sig/pred and the model profiles. L-BFGS-B
    # leaves the tape at its last trial point, not necessarily x_map, so this
    # cannot reuse the final optimiser evaluation. It costs one solve against
    # the dozens the optimisation already spent, and it is what lets plot.py be
    # a pure reader -- without it every figure script has to re-solve to draw
    # anything, which is how they ended up importing the engine.
    tape.clear_tape(); continue_annotation(); rebuild()
    for c,v in zip(ctrl_fns,x_map): c.assign(float(v))
    J_map=float(forward()); pause_annotation()
    prof, pvals, obs_out = snapshot()
    result["J_map"]=J_map
    result["diag"]=forward._diag
    result["profiles"]=prof
    result["params"]=pvals
    result["obs"]=obs_out
    result["n_steps"]=n_steps
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(result, open(Path(cfg.out_dir)/f"{cfg.tag}.json","w"), indent=2)
    log(f"\n{res.message}  J={res.fun:.4f}  -> {Path(cfg.out_dir)/(cfg.tag+'.json')}")
    for nm in m_map: log(f"  {nm:13s} = {m_map[nm]:.4e}")
    return result
