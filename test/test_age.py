import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
from math import sin, pi

from firn.constants import year
from firn.models.firn import FirnModel, FirnParameters
from firn.solvers.firn_solver import FirnColumnSolver


OMP_NUM_THREADS=1
# --------------------------------------------------------
#  Mesh (same stretched-mesh pattern you already used)
# --------------------------------------------------------
H0 = 40.0
nz = 80

mesh = fd.IntervalMesh(nz, 0.0, 1.0)
xi = fd.SpatialCoordinate(mesh)[0]
p = 3.0
z_expr = H0 * (1.0 - (1.0 - xi)**p)

coord_fs = mesh.coordinates.function_space()
new_coords = fd.Function(coord_fs).interpolate(fd.as_vector([z_expr]))
mesh.coordinates.assign(new_coords)

V = fd.FunctionSpace(mesh, "CG", 1)

# --------------------------------------------------------
#  Fields
# --------------------------------------------------------
H   = fd.Function(V, name="enthalpy")
rho = fd.Function(V, name="density")
w   = fd.Function(V, name="firn_velocity")
age = fd.Function(V, name="age")          # seconds

params = FirnParameters()
model  = FirnModel(params)
solver = FirnColumnSolver(model)

# --------------------------------------------------------
#  Forcing: seasonal surface temperature (dry still)
# --------------------------------------------------------
T_mean = 268.0
T_amp  = 7.0

def surface_T(t_seconds):
    return T_mean + T_amp * sin(2.0 * pi * (t_seconds / float(year)))

Ts = fd.Constant(surface_T(0.0))

H_init = params.c_i * (float(Ts) - params.T_ref)
H.project(fd.Constant(H_init))
rho.project(fd.Constant(300.0))
w.project(fd.Constant(0.0))
age.project(fd.Constant(0.0))

accum = fd.Constant(0.3)
rho_s = fd.Constant(300.0)

# --------------------------------------------------------
#  Boundary conditions
# --------------------------------------------------------
surface_id = 2

Hs_bc = fd.Constant(H_init)
w_surf = -accum * params.rho_i / rho_s / year

bc_H   = fd.DirichletBC(V, Hs_bc, surface_id)
bc_rho = fd.DirichletBC(V, rho_s, surface_id)
bc_w   = fd.DirichletBC(V, w_surf, surface_id)

# NEW: age BC at surface
bc_age = fd.DirichletBC(V, fd.Constant(0.0), surface_id)

bcs = [bc_H, bc_rho, bc_w]

# --------------------------------------------------------
#  Time stepping
# --------------------------------------------------------
dt = fd.Constant(60.0 * 86400.0)   # 5 days
t = 0.0
t_end = 200.0 * float(year)

out = fd.File("firn_column_with_age.pvd")
out.write(H, rho, w, age, time=t)

# depth below surface for plotting
x = fd.SpatialCoordinate(mesh)[0]
depth = fd.Function(V).interpolate(H0 - x).dat.data_ro.copy()

age_hist = []
rho_hist = []
time_hist = []

age_hist.append(age.dat.data_ro.copy())
rho_hist.append(rho.dat.data_ro.copy())
time_hist.append(t / float(year))

step = 0
while t < t_end:
    step += 1
    #print(f"Step {step}, t = {t/float(year):.3f} years")

    Ts.assign(surface_T(t))
    Ts_val = float(Ts)
    Hs_bc.assign(params.c_i * (Ts_val - params.T_ref))

    H, rho, w, age = solver.prognostic_solve(
        enthalpy=H,
        density=rho,
        firn_velocity=w,
        dt=dt,
        accumulation=accum,
        surface_density=rho_s,
        boundary_conditions=bcs,
        age=age,
        age_boundary_condition=bc_age,
    )

    t += float(dt)

    age_hist.append(age.dat.data_ro.copy())
    rho_hist.append(rho.dat.data_ro.copy())
    time_hist.append(t / float(year))

    #if step % 50 == 0:
    #    out.write(H, rho, w, age, time=t)

# Convert to arrays
age_hist = np.array(age_hist) / float(year)  # convert to years
rho_hist = np.array(rho_hist)
time_hist = np.array(time_hist)

# Plot age profiles at a few times
plt.figure(figsize=(6, 8))
for idx in np.linspace(0, len(time_hist)-1, 8).astype(int):
    plt.plot(age_hist[idx], depth, label=f"t={time_hist[idx]:.1f} yr")
plt.gca().invert_yaxis()
plt.xlabel("Age (years)")
plt.ylabel("Depth below surface (m)")
plt.title("Firn age evolution (profiles)")
plt.legend()
plt.tight_layout()
plt.savefig("firn_age_profiles.png", dpi=200)
plt.show()
