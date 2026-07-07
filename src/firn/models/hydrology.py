from dataclasses import dataclass
import firedrake as fd
from firn.constants import (
    ice_density, water_density, latent_heat,
    water_viscosity, gravity,
)
from firn.models.firn import FirnParameters  # reuse or subclass

@dataclass
class HydrologyParameters:
    rho_i: float = ice_density
    rho_w: float = water_density
    L: float = latent_heat
    g: float = gravity

    # Richards / Darcy parameters
    phi_min: float = 1e-4
    S_min: float = 1e-6
    S_res: float = 0.05
    K0: float = 1e-7      # m^2/s
    m_phi: float = 3.0
    m_S: float = 2.0
    psi0: float = 0.5     # m
    theta_W: float = 0.5
    theta_H: float = 0.5  # if enthalpy is handled here

class HydrologyModel:
    def __init__(self, params: HydrologyParameters | None = None):
        self.params = params or HydrologyParameters()

    # -- porosity & saturation --
    def porosity(self, rho):
        p = self.params
        phi_raw = 1.0 - rho / p.rho_i
        return fd.max_value(phi_raw, p.phi_min)

    def saturation(self, W, rho):
        p = self.params
        phi = self.porosity(rho)
        S_raw = W / (p.rho_w * phi)
        S = fd.max_value(fd.min_value(S_raw, 1.0), p.S_min)
        return S

    # -- hydraulic relations --
    def K_sat(self, rho):
        p = self.params
        phi = self.porosity(rho)
        return fd.Constant(p.K0) * phi**p.m_phi

    def k_rel(self, S):
        p = self.params
        Sr = fd.Constant(p.S_res)
        S_eff = fd.max_value(S - Sr, 0.0)
        return S_eff**p.m_S

    def capillary_potential_prime(self, S):
        p = self.params
        # Psi = psi0 * (1 - S) => Psi' = -psi0
        return -fd.Constant(p.psi0)

    def darcy_flux(self, W, rho):
        p = self.params
        S = self.saturation(W, rho)
        Ksat = self.K_sat(rho)
        krel = self.k_rel(S)

        Psi_prime = self.capillary_potential_prime(S)
        dSdx = S.dx(0)
        dPsidx = Psi_prime * dSdx

        # depth coordinate is downward, so +1 is gravity
        q = -p.rho_w * Ksat * krel * (dPsidx + 1.0)
        return q

    # -- weak form for water --
    def water_form(self, W, W_old, rho, w, test, dt):
        """
        (W - W_old)/dt + d/dx(w W + q) = 0
        """
        p = self.params
        psi = test
        dx = fd.dx

        thetaW = fd.Constant(p.theta_W)
        W_mid = thetaW*W + (1 - thetaW)*W_old
        w_mid = w  # or a mid-point if you have w_old

        q_mid = self.darcy_flux(W_mid, rho)

        F_W = (
            (W - W_old)/dt * psi * dx
            - (w_mid*W_mid + q_mid) * psi.dx(0) * dx
        )
        return F_W

    # -- optional: enthalpy form including latent transport --
    def enthalpy_form(self, H, H_old, W, rho, w, T_func, K_func, test, dt):
        """
        dH/dt + d/dx( w H - L q - K T_x ) = 0
        T_func(H) and K_func(H, rho) are callables from FirnModel.
        """
        p   = self.params
        psi = test
        dx  = fd.dx

        thetaH = fd.Constant(p.theta_H)
        H_mid  = thetaH*H + (1 - thetaH)*H_old
        w_mid  = w  # or midpoint with old if you store w_old

        # --- use the firn model relations correctly ---
        T_mid = T_func(H_mid)          # only H, matches FirnModel.temperature_from_enthalpy
        K_mid = K_func(H_mid, rho)     # H and rho, matches FirnModel.thermal_diffusivity

        # time-centred water content
        thetaW = fd.Constant(p.theta_W)
        W_mid  = thetaW*W + (1 - thetaW)*W  # you can later use W_old if you like

        q_mid = self.darcy_flux(W_mid, rho)

        flux = w_mid*H_mid - p.L * q_mid - K_mid * T_mid.dx(0)

        F_H = (
            (H - H_old)/dt * psi * dx
            - flux * psi.dx(0) * dx
        )
        return F_H