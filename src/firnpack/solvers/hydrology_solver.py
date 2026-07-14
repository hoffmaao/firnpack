# firn/solvers/hydro_solver.py
import firedrake as fd

class HydrologySolver:
    def __init__(self, hydrology_model, firn_model, solver_parameters=None):
        self.hydrology_model = hydrology_model
        self.firn_model = firn_model
        self.solver_parameters = solver_parameters or {
            "snes_type": "newtonls",
            "snes_rtol": 1e-5,
            "snes_atol": 1e-7,
            "snes_max_it": 30,
            "ksp_type": "gmres",
            "pc_type": "lu",
        }
        self._fields = {}

    @property
    def fields(self):
        return self._fields

    def _setup(self, *, enthalpy, water, density, firn_velocity, dt, water_bcs):
        self._fields["H"] = enthalpy
        self._fields["W"] = water
        self._fields["rho"] = density
        self._fields["w"] = firn_velocity

        V = enthalpy.function_space()
        self._fields["V"] = V
        self._fields["H_old"] = enthalpy.copy(deepcopy=True)
        self._fields["W_old"] = water.copy(deepcopy=True)
        self._fields["dt"] = dt
        self._fields["bcs_W"] = water_bcs
        self._fields["test"] = fd.TestFunction(V)

    def prognostic_solve(
        self,
        *,
        enthalpy,
        water,
        density,
        firn_velocity,
        dt,
        water_bcs=None,
        update_enthalpy=True,
    ):
        if water_bcs is None:
            water_bcs = []

        if "H_old" not in self._fields:
            self._setup(
                enthalpy=enthalpy,
                water=water,
                density=density,
                firn_velocity=firn_velocity,
                dt=dt,
                water_bcs=water_bcs,
            )
        else:
            self._fields["H_old"].assign(enthalpy)
            self._fields["W_old"].assign(water)
            self._fields["dt"].assign(dt)

        V   = self._fields["V"]
        psi = self._fields["test"]
        H   = enthalpy
        W   = water
        rho = density
        w   = firn_velocity
        H_old = self._fields["H_old"]
        W_old = self._fields["W_old"]
        dt_val = self._fields["dt"]

        hydro = self.hydrology_model
        firn  = self.firn_model

        # 1) enthalpy step with latent + Darcy, if requested
        if update_enthalpy:
            F_H = hydro.enthalpy_form(
                H, H_old, W, rho, w,
                T_func=firn.temperature_from_enthalpy,
                K_func=firn.thermal_diffusivity,
                test=psi, dt=dt_val,
            )
            dH  = fd.TrialFunction(V)
            J_H = fd.derivative(F_H, H, dH)
            problem_H = fd.NonlinearVariationalProblem(F_H, H, J=J_H)
            solver_H  = fd.NonlinearVariationalSolver(
                problem_H, solver_parameters=self.solver_parameters
            )
            solver_H.solve()

        # 2) water step
        F_W = hydro.water_form(W, W_old, rho, w, psi, dt_val)
        dW  = fd.TrialFunction(V)
        J_W = fd.derivative(F_W, W, dW)
        problem_W = fd.NonlinearVariationalProblem(
            F_W, W, bcs=water_bcs, J=J_W
        )
        solver_W = fd.NonlinearVariationalSolver(
            problem_W, solver_parameters=self.solver_parameters
        )
        solver_W.solve()

        H_old.assign(H)
        W_old.assign(W)

        return H, W
