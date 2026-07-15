import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
from math import sin, pi

from firnpack.constants import year
from firnpack.models.firn import FirnModel, FirnParameters
from firnpack.solvers.firn_solver import FirnColumnSolver


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

params = FirnParameters()
model = FirnModel(params)
solver = FirnColumnSolver(model)

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

# Initial surface temperature = value at t=0
Ts = fd.Constant(surface_T(0.0))

# Initial enthalpy & density
H_init = params.c_i * (float(Ts) - params.T_ref)
rho_init = fd.Constant(300.0)

H.project(fd.Constant(H_init))
rho.project(fd.Constant(rho_init))
w.project(fd.Constant(0.0))

# Accumulation (m ice eq / yr) and surface density
accum = fd.Constant(0.3)
rho_s = fd.Constant(300.0)


# --------------------------------------------------------
#  Boundary conditions
# --------------------------------------------------------

Hs_bc = fd.Constant(H_init)

# Surface vertical velocity from kinematic boundary condition
w_surf = -accum * params.rho_i / rho_s / year

# On IntervalMesh(0.0, H0): 1 = left (x=0, base), 2 = right (x=H0, surface)
surface_id = 2

bc_H   = fd.DirichletBC(V, Hs_bc, surface_id)
bc_rho = fd.DirichletBC(V, rho_s, surface_id)
bc_w   = fd.DirichletBC(V, w_surf, surface_id)

bcs = [bc_H, bc_rho, bc_w]


# --------------------------------------------------------
#  Time stepping setup
# --------------------------------------------------------

dt = fd.Constant(5.0 * 86400.0)   # 5 days
t = 0.0
t_end = 50.0 * float(year)        # 50 years – adjust as you like

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

    # Update surface temperature for this time
    Ts.assign(surface_T(t))

    H, rho, w = solver.prognostic_solve(
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

    t += float(dt)

    density_history.append(rho.dat.data_ro.copy())
    time_history.append(t / year)
    Ts_history.append(float(Ts))

    if step % 50 == 0:
        outfile.write(H, rho, w, time=t)


# --------------------------------------------------------
#  Convert lists to arrays and save for external use
# --------------------------------------------------------

density_history = np.array(density_history)   # shape (nt, nz)
time_history = np.array(time_history)         # years
Ts_history = np.array(Ts_history)

np.savez(
    "firn_density_seasonal_surface.npz",
    depth=z,
    time=time_history,
    density=density_history,
    Ts=Ts_history,
)


# --------------------------------------------------------
#  Plot density evolution (profiles + contour)
# --------------------------------------------------------

# --- Multi-profile over the entire run ---
plt.figure(figsize=(6, 8))

n_profiles = 10
indices = np.linspace(0, len(time_history)-1, n_profiles).astype(int)

for idx in indices:
    plt.plot(density_history[idx], z, label=f"t={time_history[idx]:.1f} yr")

plt.gca().invert_yaxis()
plt.xlabel("Density (kg m⁻³)")
plt.ylabel("Depth below surface (m)")
plt.title("Firn Density Evolution with Seasonal Surface Forcing (Profiles)")
plt.legend()
plt.tight_layout()
plt.savefig("firn_density_profiles_seasonal.png", dpi=200)
plt.show()

# --- Time–depth contour plot ---
plt.figure(figsize=(8, 6))

plt.pcolormesh(time_history, z, density_history.T,
               shading='auto', cmap="viridis")
plt.gca().invert_yaxis()
plt.colorbar(label="Density (kg m⁻³)")
plt.xlabel("Time (years)")
plt.ylabel("Depth below surface (m)")
plt.title("Firn Density Evolution with Seasonal Surface Forcing (Contour)")
plt.tight_layout()
plt.savefig("firn_density_contour_seasonal.png", dpi=200)
plt.show()
