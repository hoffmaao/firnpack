"""Coupled firn + hydrology column driver.

PARKED, not yet a test. This is still the original script: it runs a 7-year
seasonal column and prints diagnostics, and it asserts nothing.

It is not converted because its intended physics is unresolved. ``W_surf`` is
pinned at 0.0, so no water is ever injected and the hydrology solve runs on an
all-zero field -- while ``surface_energy_step`` below, which computes the melt
flux that would drive it, is defined and never called. Any assertion written
against the current behaviour (``W >= 0``, ``S in [0, 1]``) would pass on a
completely broken hydrology solver, so writing one now would be worse than
writing none.

What it needed immediately was the guard: every line below used to run at
import, which under pytest meant collection alone executed the whole
simulation. Resolve the melt path, then convert.

Run directly:
    PYTHONPATH=src OMP_NUM_THREADS=1 python test/test_hydrology.py
"""

import firedrake as fd
import numpy as np
from math import sin, pi

from firnpack.constants import year
from firnpack.models.firn import FirnModel, FirnParameters
from firnpack.solvers.firn_solver import FirnColumnSolver
from firnpack.models.hydrology import HydrologyModel, HydrologyParameters
from firnpack.solvers.hydrology_solver import HydrologySolver


def main():

    # --------------------------------------------------------
    #  Setup mesh and function spaces
    # --------------------------------------------------------

    H0 = 40.0          # firn column thickness [m]
    nz = 80

    mesh = fd.IntervalMesh(nz, 0.0, 1)
    p = 3.0  # >1 -> stronger clustering near the surface
    xi = fd.SpatialCoordinate(mesh)[0]
    z_expr = H0 * (1.0 - (1.0 - xi)**p)

    # # Example B (alternative): exponential grading
    # a = 4.0
    # f_xi = (fd.exp(a * xi) - 1.0) / (fd.exp(a) - 1.0)
    # z_expr = H0 * f_xi

    # 3) update mesh.coordinates so all integrals use the new spacing
    coord_fs = mesh.coordinates.function_space()
    new_coords = fd.Function(coord_fs)
    new_coords.interpolate(fd.as_vector([z_expr]))   # 1D -> length-1 vector
    mesh.coordinates.assign(new_coords)

    V = fd.FunctionSpace(mesh, "CG", 1)
    H = fd.Function(V, name="enthalpy")
    rho = fd.Function(V, name="density")
    w = fd.Function(V, name="firn_velocity")
    W = fd.Function(V, name="water")

    firn_params = FirnParameters()
    firn_model = FirnModel(firn_params)
    firn_solver = FirnColumnSolver(firn_model)

    hydrology_params = HydrologyParameters()
    hydrology_model = HydrologyModel(hydrology_params)
    hydrology_solver = HydrologySolver(hydrology_model,firn_model)
    # --------------------------------------------------------
    #  Surface temperature forcing: seasonal cycle
    # --------------------------------------------------------

    # Mean annual surface temperature and amplitude [K]
    T_mean = 268.0      # average ~ -5 °C
    T_amp  = 7.0        # gives range ~ 261–275 K, so summer exceeds Tm

    def surface_T(t_seconds):
        """
        Simple sinusoidal annual cycle:
           T(t) = T_mean + T_amp * sin(2π t / year)
        """
        phase = 2.0 * pi * (t_seconds / year)
        return T_mean + T_amp * sin(phase)

    def surface_energy_step(Ts_old, t_years, dt, params):
        """
        Advance surface temperature by one time-step using a simple
        energy-balance, and compute the associated meltwater flux.

        Parameters
        ----------
        Ts_old : float
            Previous surface temperature [K].
        t_years : float
            Current time in years.
        dt : float
            Time-step [s].
        params : FirnParameters
            Parameter object (from constants.py).

        Returns
        -------
        Ts_new : float
            Updated surface temperature [K].
        R_melt : float
            Surface meltwater mass flux [kg m^-2 s^-1].
        Q_t : float
            Net surface energy flux used this step [W m^-2].
        """
        import numpy as np

        p = params

        # --- seasonal net surface flux Q(t), Meyer & Hewitt-style ---
        t_dim = t_years * p.sec_per_year
        Q_t = p.Q_mean - p.Q_amp * np.cos(2.0 * np.pi * t_dim / p.sec_per_year)

        # Heat capacity of the "skin" layer
        rho_eff = p.rho_i
        c_eff = p.c_i
        H_skin = p.H_skin
        Tm = p.T_m

        # Energy per unit area required (over this dt) to heat the skin from Ts_old to Tm
        if Ts_old < Tm:
            Q_heat = rho_eff * c_eff * H_skin * (Tm - Ts_old) / dt  # [W m^-2]
        else:
            Q_heat = 0.0

        if Q_t <= Q_heat or Ts_old < Tm:
            # Not enough energy to start/continue melting.
            # Use all of Q_t to change Ts.
            Ts_new = Ts_old + dt * Q_t / (rho_eff * c_eff * H_skin)
            Ts_new = min(Ts_new, Tm)
            R_melt = 0.0
        else:
            # Heat surface to melting, then use remaining energy for melt.
            Ts_new = Tm
            Q_excess = Q_t - Q_heat

            # Meltwater mass flux [kg m^-2 s^-1]:
            #   Q_excess [J s^-1 m^-2] / L [J kg^-1]
            R_melt = max(Q_excess / p.L, 0.0)

        return Ts_new, R_melt, Q_t

    # Initial surface temperature = value at t=0
    Ts = fd.Constant(surface_T(0.0))

    # Initial enthalpy & density
    H_init = firn_params.c_i * (float(Ts) - firn_params.T_ref)
    rho_init = fd.Constant(300.0)

    H.project(fd.Constant(H_init))
    rho.project(fd.Constant(rho_init))
    w.project(fd.Constant(0.0))
    W.project(fd.Constant(0.0))
    # Accumulation (m ice eq / yr) and surface density
    accum = fd.Constant(0.3)
    rho_s = fd.Constant(300.0)

    Hs_bc = fd.Constant(H_init)

    # Surface vertical velocity from kinematic boundary condition
    w_surf = -accum * firn_params.rho_i / rho_s / year

    # On IntervalMesh(0.0, H0): 1 = left (x=0, base), 2 = right (x=H0, surface)
    surface_id = 2

    bc_H   = fd.DirichletBC(V, Hs_bc, surface_id)
    bc_rho = fd.DirichletBC(V, rho_s, surface_id)
    bc_w   = fd.DirichletBC(V, w_surf, surface_id)


    # maybe a Dirichlet BC giving melt at the surface:
    W_surf = fd.Constant(0.0)  # or a time-dependent melt/rain rate
    bc_W = fd.DirichletBC(V, W_surf, surface_id)

    bcs = [bc_H, bc_rho, bc_w]






    # --------------------------------------------------------
    #  Time stepping setup
    # --------------------------------------------------------

    dt = fd.Constant(5.0 * 86400.0)   # 5 days
    t = 0.0
    t_end = 7.0 * float(year)        # 50 years – adjust as you like

    outfile = fd.VTKFile("firn_column_seasonal_output.pvd")
    outfile.write(H, rho, w, time=t)


    # --------------------------------------------------------
    #  Diagnostics arrays for plotting after the run
    # --------------------------------------------------------

    # Depth coordinates as distance below surface: depth = H0 - x
    x = fd.SpatialCoordinate(mesh)[0]
    z_fun = fd.Function(V, name="depth")
    z_fun.interpolate(H0 - x)           # 0 at surface, H0 at base
    z = z_fun.dat.data_ro.copy()

    density_history = []
    time_history = []
    Ts_history = []

    # store initial state
    density_history.append(rho.dat.data_ro.copy())
    time_history.append(t / year)
    Ts_history.append(float(Ts))


    # --------------------------------------------------------
    #  Time integration loop with seasonal Ts(t)
    # --------------------------------------------------------

    step = 0

    while t < t_end:
        step += 1
        print(f"Step {step}, t = {t/float(year):.3f} years")


        # --- basic water diagnostics BEFORE any updates this step ---
        W_arr = W.dat.data_ro
        print("  [water pre-step]")
        print(f"    W: min={W_arr.min():.3e}, max={W_arr.max():.3e}, total={W_arr.sum():.3e}")



        # Update surface temperature for this time
        Ts.assign(surface_T(t))

        # 1. dry thermo-mechanics: H, rho, w
        H, rho, w = firn_solver.prognostic_solve(
            enthalpy=H,
            density=rho,
            firn_velocity=w,
            dt=dt,
            accumulation=accum,
            surface_density=rho_s,
            boundary_conditions=bcs,
            surface_temperature=Ts,
            enthalpy_bc_constant=Hs_bc,
        )

        # temperature diagnostics
        T_fun = fd.assemble(firn_model.temperature_from_enthalpy(H))
        T_arr = T_fun.dat.data_ro
        print("  [firn state]")
        print(f"    T:    min={T_arr.min():.2f} K, max={T_arr.max():.2f} K")
        rho_arr = rho.dat.data_ro
        print(f"    rho:  min={rho_arr.min():.3e}, max={rho_arr.max():.3e}")
        w_arr = w.dat.data_ro
        print(f"    w:    min={w_arr.min():.3e}, max={w_arr.max():.3e}")

        # --- 2. hydrology diagnostics BEFORE solving water equation ---
        # saturation & Darcy flux at current W, rho
        S_expr = hydrology_model.saturation(W, rho)
        q_expr = hydrology_model.darcy_flux(W, rho)

        S_fun = fd.project(S_expr, V)
        q_fun = fd.project(q_expr, V)

        S_arr = S_fun.dat.data_ro
        q_arr = q_fun.dat.data_ro

        print("  [hydrology pre-solve]")
        print(f"    S (sat): min={S_arr.min():.3e}, max={S_arr.max():.3e}")
        print(f"    q (flux): min={q_arr.min():.3e}, max={q_arr.max():.3e}")


        # 2. hydrology step: update H and W with Darcy + latent
        H, W = hydrology_solver.prognostic_solve(
            enthalpy=H,
            water=W,
            density=rho,
            firn_velocity=w,
            dt=dt,
            water_bcs=[bc_W],
            update_enthalpy=True,   # or False if you want dry enthalpy only
        )


        # --- water diagnostics AFTER hydrology solve ---
        W_arr = W.dat.data_ro
        print("  [water post-solve]")
        print(f"    W: min={W_arr.min():.3e}, max={W_arr.max():.3e}, total={W_arr.sum():.3e}")

        t += float(dt)
        density_history.append(rho.dat.data_ro.copy())
        time_history.append(t / year)
        Ts_history.append(float(Ts))

        if step % 50 == 0:
            outfile.write(H, rho, w, time=t)



if __name__ == "__main__":
    main()
