"""
Physical constants for the firn model.

These values come **directly from Evan Cummings' physical_constants.py**
and are kept in SI units exactly as he uses them.
"""

# --- Time ---
year = 365.0 * 24 * 60 * 60          # seconds per year

# --- Densities ---
ice_density   = 917.0                # kg m^-3
water_density = 1000.0               # kg m^-3
transition_density = 550.0           # rhom, kg m^-3
closeoff_density   = 815.0           # rhoc, kg m^-3

# --- Thermodynamic constants ---
heat_capacity = 2009.0               # J kg^-1 K^-1 (cpi)
melting_temperature = 273.15         # K (Tw)
reference_temperature = 0.0          # K (T0)
latent_heat = 3.34e5                 # J kg^-1 (Lf)

# --- Thermal conductivity of ice ---
thermal_conductivity = 2.1           # W m^-1 K^-1 (ki)

# Thermal diffusivity κ = k / (ρ c)
thermal_diffusivity = thermal_conductivity / (ice_density * heat_capacity)

# --- Gravity ---
gravity = 9.81                       # m s^-2

# --- Ideal gas constant ---
gas_constant = 8.3144621             # J mol^-1 K^-1 (R)

# --- Activation energies ---
activation_energy_creep = 60e3       # J mol^-1 (Ec)
activation_energy_grain  = 42.4e3    # J mol^-1 (Eg)

# --- Clausius–Clapeyron slope ---
clausius_clapeyron = 7.9e-8          # K Pa^-1 (beta)

# --- Water viscosity ---
water_viscosity = 1.787e-3           # Pa s (etaw)

# --- Permeability coefficient (for water flow) ---
grain_growth_coefficient = 1.3e-7    # m^2 s^-1 (kg)
kcHh = 9.2e-9                       # (m^3
kcLw = 3.7e-9
Ec = 60e3
Eg = 42.4e3
kg = 1.3e-7