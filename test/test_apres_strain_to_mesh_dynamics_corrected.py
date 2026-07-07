"""
test_apres_strain_to_mesh_dynamics_corrected.py

Standalone script (no argparse, no main) to:
  1) load an ApRES MATLAB v7.3 (HDF5) strain file,
  2) compute vertical velocity w(z) [m/yr],
  3) estimate an ice-dynamics background vertical strain rate from a linear
     fit to deep (incompressible-ice) velocities,
  4) remove that background to isolate firn-compaction velocity/strain rate,
  5) map the corrected profiles onto a 1D stretched Firedrake mesh (if available),
  6) plot raw vs corrected velocity and strain rate.

This follows the logic used in Case (2022, J. Glaciol.):
- assume the observed vertical velocity is the sum of an ice-dynamic component
  (from horizontal divergence/convergence) and a firn-compaction component,
- estimate the ice-dynamic component via a linear fit to the deep part of the
  velocity profile (where firn compaction is negligible),
- subtract it to isolate firn compaction.

------------------------------------------------------------
USER SETTINGS
------------------------------------------------------------
"""

import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt


# --------------------------
# USER SETTINGS
# --------------------------

# Path to your ApRES strain .mat file (MATLAB v7.3 / HDF5)
MAT_FILE = "G4-08-05_2023_2024_strain.mat"

# Optional: if MAT_FILE does not exist, set a URL to download it
DOWNLOAD_URL = None  # e.g. "https://.../G4-08-05_2023_2024_strain.mat"

# Use only data in the firn region (range <= firn_depth from file) for plotting/mapping
USE_FIRN_ONLY = True

# Include a "surface" velocity point (depth=0). WARNING: many MATLAB pipelines store a
# fitted/extrapolated surface velocity rather than a direct near-surface ApRES return.
# Leave False unless you *know* dhsurfaceRate is an independent observation you want to use.
INCLUDE_SURFACE_POINT = False

# Apply the ice-dynamics correction (linear-fit background strain rate)
APPLY_ICE_DYNAMICS_CORRECTION = True

# How to choose the fit window for the background ice-dynamics strain rate:
#   - If the .mat file has /vdat_strain/fit_ice/vertical_vel_range and vertical_vel,
#     we will use those by default (these often already encode a "good" fit window).
#   - Otherwise, we'll fit w(z) over [FIT_DEPTH_MIN, FIT_DEPTH_MAX] using dh/dt.
USE_FILE_FIT_ARRAYS_IF_AVAILABLE = True

# Manual fit window (only used if file-fit arrays are not available)
FIT_DEPTH_MIN = None  # default: firn_depth_m
FIT_DEPTH_MAX = None  # default: 2/3 of ice thickness H if available, else 0.7*max(range)

# Anchor depth for the fit (line forced through (z0, 0)). If your w(z) is referenced to a
# "zero-shift depth" in the file, this is the correct anchor.
ANCHOR_AT_Z0 = True  # if False, we fit w = a + b z with free intercept

# Mesh settings (for mapping onto your firn column mesh)
# If H0_MESH is None, we set it to firn_depth from the file.
H0_MESH = 100.0  # e.g. 40.0 to map into your 40 m firn inversion mesh; or None to use firn_depth from file
NZ = 200
STRETCH_P = 3.0

# Strain-rate smoothing window (meters): larger -> smoother strain estimates
WINDOW_M = 10.0

# Plot settings
PLOT_MAX_DEPTH = 100.0  # e.g. 40.0 to zoom to upper 40 m, or None for full mesh depth
SAVE_PNG = True
OUTDIR = "."  # where to write PNGs / optional VTK
WRITE_VTK = False  # requires Firedrake


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def ensure_file(path: str, url: str | None) -> None:
    if os.path.exists(path):
        return
    if url is None:
        raise FileNotFoundError(f"Could not find {path!r} and DOWNLOAD_URL is None.")
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def read_apres_v73(mat_file: str, group: str = "vdat_strain") -> dict:
    """
    Read ApRES v7.3 (HDF5) .mat file using h5py.

    Expected minimal structure:
      /vdat_strain/range      (Nx1)  [m]
      /vdat_strain/dh         (Nx1)  [m]
      /vdat_strain/dt         (1x1)  [days]
      /vdat_strain/firn_depth (1x1)  [m]

    Optional fields we use if available:
      /vdat_strain/H                    (1x1)  [m]    ice thickness
      /vdat_strain/dhsurfaceRate        (1x1)  [m/yr] surface rate (interpretation depends on pipeline!)
      /vdat_strain/fit_ice/zeroShiftDepth      (1x1) [m]
      /vdat_strain/fit_ice/start, end         (1x1) [m] fit bounds
      /vdat_strain/fit_ice/vertical_vel_range (Kx1) [m]
      /vdat_strain/fit_ice/vertical_vel       (Kx1) [m/yr]
    """
    try:
        import h5py
    except Exception as e:
        raise RuntimeError(
            "h5py is required to read MATLAB v7.3 files. "
            "Install h5py or provide a v7.2 MAT file readable by scipy.io.loadmat."
        ) from e

    with h5py.File(mat_file, "r") as f:
        if group not in f:
            raise KeyError(f"Group {group!r} not found in {mat_file}. Keys: {list(f.keys())}")

        g = f[group]

        def get_array_from(obj, name: str) -> np.ndarray | None:
            if name not in obj:
                return None
            return np.array(obj[name]).squeeze()

        def get_scalar_from(obj, name: str, default=None):
            if name not in obj:
                return default
            a = np.array(obj[name]).squeeze()
            if np.size(a) != 1:
                return default
            return float(a)

        out = {}
        out["range_m"] = get_array_from(g, "range").astype(float)
        out["dh_m"] = get_array_from(g, "dh").astype(float)
        out["dt_days"] = float(np.array(g["dt"]).squeeze())
        out["firn_depth_m"] = float(np.array(g["firn_depth"]).squeeze())

        out["H_m"] = get_scalar_from(g, "H", default=None)
        out["dhsurfaceRate_myr"] = get_scalar_from(g, "dhsurfaceRate", default=None)

        # fit_ice metadata
        out["zeroShiftDepth_m"] = None
        out["fit_start_m"] = None
        out["fit_end_m"] = None
        out["fit_z_m"] = None
        out["fit_w_myr"] = None

        if "fit_ice" in g:
            fit = g["fit_ice"]
            out["zeroShiftDepth_m"] = get_scalar_from(fit, "zeroShiftDepth", default=None)
            out["fit_start_m"] = get_scalar_from(fit, "start", default=None)
            out["fit_end_m"] = get_scalar_from(fit, "end", default=None)
            out["fit_z_m"] = get_array_from(fit, "vertical_vel_range")
            out["fit_w_myr"] = get_array_from(fit, "vertical_vel")

        return out


def moving_slope(x: np.ndarray, y: np.ndarray, window: float) -> np.ndarray:
    """
    Moving-window linear regression slope dy/dx.
    window: half-width in the x variable (same units as x).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    slope = np.full(n, np.nan, dtype=float)

    for i in range(n):
        m = np.abs(x - x[i]) <= window
        if np.count_nonzero(m) < 3:
            continue
        xm = x[m]
        ym = y[m]
        xm0 = xm - xm.mean()
        denom = np.sum(xm0 * xm0)
        if denom == 0.0:
            continue
        slope[i] = np.sum(xm0 * (ym - ym.mean())) / denom

    return slope


def build_stretched_depth_mesh(H0: float, nz: int, p: float):
    """
    Build a 1D stretched mesh using Firedrake if available.
    Returns (mesh, V, depth_dofs) where depth_dofs is depth below surface at V dofs.

    NOTE: Some Firedrake builds no longer expose FunctionSpace.tabulate_dof_coordinates().
    We therefore compute depth at DOFs by interpolating the coordinate expression into V.
    """
    try:
        import firedrake as fd
    except Exception:
        return None, None, None

    mesh = fd.IntervalMesh(nz, 0.0, 1.0)
    xi = fd.SpatialCoordinate(mesh)[0]
    z_expr = H0 * (1.0 - (1.0 - xi) ** p)

    coord_fs = mesh.coordinates.function_space()
    new_coords = fd.Function(coord_fs).interpolate(fd.as_vector([z_expr]))
    mesh.coordinates.assign(new_coords)

    V = fd.FunctionSpace(mesh, "CG", 1)

    # Robust: compute depth at the DOF points by interpolation.
    x = fd.SpatialCoordinate(mesh)[0]  # physical coordinate after stretching
    depth_fun = fd.Function(V).interpolate(H0 - x)
    depth = depth_fun.dat.data_ro.copy()

    return mesh, V, depth


def apply_depth_limits():
    if PLOT_MAX_DEPTH is not None:
        plt.ylim(PLOT_MAX_DEPTH, 0.0)
    else:
        plt.gca().invert_yaxis()


def slope_through_point(z: np.ndarray, w: np.ndarray, z0: float) -> float:
    """
    Least-squares slope b for w ≈ b*(z - z0), i.e. line forced through (z0, 0).
    """
    dz = z - z0
    denom = float(np.dot(dz, dz))
    if denom == 0.0:
        raise ValueError("Degenerate fit: all z equal to z0.")
    return float(np.dot(dz, w) / denom)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

ensure_file(MAT_FILE, DOWNLOAD_URL)
data = read_apres_v73(MAT_FILE)

range_m = data["range_m"]
dh_m = data["dh_m"]
dt_days = data["dt_days"]
firn_depth_m = data["firn_depth_m"]
H_m = data["H_m"]
dhsurfaceRate_myr = data["dhsurfaceRate_myr"]
zeroShiftDepth_m = data["zeroShiftDepth_m"]
fit_start_m = data["fit_start_m"]
fit_end_m = data["fit_end_m"]
fit_z_m = data["fit_z_m"]
fit_w_myr = data["fit_w_myr"]

dt_years = dt_days / 365.25
w_total = dh_m / dt_years  # NOTE: typically referenced/shifted by the MATLAB pipeline

print("\nLoaded ApRES data:")
print(f"  file: {MAT_FILE}")
print(f"  dt:   {dt_days:.3f} days = {dt_years:.4f} years")
print(f"  firn_depth in file: {firn_depth_m:.2f} m")
if H_m is not None:
    print(f"  ice thickness H in file: {H_m:.2f} m")
print(f"  range min/max: {np.nanmin(range_m):.2f} / {np.nanmax(range_m):.2f} m")
if zeroShiftDepth_m is not None:
    print(f"  zeroShiftDepth (fit_ice): {zeroShiftDepth_m:.2f} m")
if dhsurfaceRate_myr is not None:
    print(f"  dhsurfaceRate: {dhsurfaceRate_myr:.3f} m/yr (interpretation depends on pipeline)")
if fit_start_m is not None and fit_end_m is not None:
    print(f"  fit_ice window in file: start={fit_start_m:.2f} m, end={fit_end_m:.2f} m")
if fit_z_m is not None and fit_w_myr is not None:
    print(f"  fit_ice arrays: {fit_z_m.size} points")

# Near-surface availability warning
shallowest = float(np.nanmin(range_m))
if shallowest > 5.0:
    print(f"WARNING: shallowest return is {shallowest:.2f} m; no <5 m near-surface ApRES returns in this file.")
else:
    print("Near-surface returns appear to be present (<5 m).")

# ------------------------------------------------------------
# Estimate ice-dynamics background strain rate and correct
# ------------------------------------------------------------

eps_dyn = 0.0
w_dyn = np.zeros_like(w_total)

if APPLY_ICE_DYNAMICS_CORRECTION:
    # Choose anchor depth
    z0 = zeroShiftDepth_m if zeroShiftDepth_m is not None else firn_depth_m

    # Choose fit dataset and fit window
    if USE_FILE_FIT_ARRAYS_IF_AVAILABLE and (fit_z_m is not None) and (fit_w_myr is not None):
        z_fit = np.asarray(fit_z_m, dtype=float)
        w_fit = np.asarray(fit_w_myr, dtype=float)
        mfit = np.isfinite(z_fit) & np.isfinite(w_fit)
        z_fit = z_fit[mfit]
        w_fit = w_fit[mfit]
        fit_desc = "fit_ice/vertical_vel(_range) arrays from file"
    else:
        z_fit = np.asarray(range_m, dtype=float)
        w_fit = np.asarray(w_total, dtype=float)

        zmin = firn_depth_m if FIT_DEPTH_MIN is None else float(FIT_DEPTH_MIN)
        if FIT_DEPTH_MAX is None:
            if H_m is not None:
                zmax = 2.0 * float(H_m) / 3.0
            else:
                zmax = 0.7 * float(np.nanmax(z_fit))
        else:
            zmax = float(FIT_DEPTH_MAX)

        mfit = np.isfinite(z_fit) & np.isfinite(w_fit) & (z_fit >= zmin) & (z_fit <= zmax)
        z_fit = z_fit[mfit]
        w_fit = w_fit[mfit]
        fit_desc = f"dh/dt over [{zmin:.1f}, {zmax:.1f}] m"

    if z_fit.size < 3:
        raise RuntimeError("Not enough points in fit window to estimate background strain rate.")

    if ANCHOR_AT_Z0:
        eps_dyn = slope_through_point(z_fit, w_fit, z0)
        # dynamic component is linear and passes through (z0, 0)
        w_dyn = eps_dyn * (range_m - z0)
    else:
        # free-intercept fit: w ≈ a + b z
        b, a = np.polyfit(z_fit, w_fit, 1)
        eps_dyn = float(b)
        w_dyn = a + eps_dyn * range_m

    print("\nIce-dynamics correction:")
    print(f"  anchor depth z0 = {z0:.2f} m")
    print(f"  fit used: {fit_desc}")
    print(f"  estimated background vertical strain rate eps_dyn = dw/dz = {eps_dyn:.6e} 1/yr")
    print("  (For incompressible ice in 2-D: horizontal strain rate eps_x ≈ -eps_dyn)")

    # Build corrected velocity everywhere
    w_comp = w_total - w_dyn
else:
    w_comp = w_total

# ------------------------------------------------------------
# Build observation arrays for plotting/mapping (firn-only optional)
# ------------------------------------------------------------

mask = np.isfinite(range_m) & np.isfinite(w_total)
if USE_FIRN_ONLY:
    mask &= (range_m <= firn_depth_m)

z_obs = range_m[mask].astype(float)

w_obs_raw = w_total[mask].astype(float)
w_obs_comp = w_comp[mask].astype(float)

# Sort by depth increasing
order = np.argsort(z_obs)
z_obs = z_obs[order]
w_obs_raw = w_obs_raw[order]
w_obs_comp = w_obs_comp[order]

# Optional surface point handling
if INCLUDE_SURFACE_POINT and (dhsurfaceRate_myr is not None):
    z_obs = np.concatenate(([0.0], z_obs))
    w_obs_raw = np.concatenate(([float(dhsurfaceRate_myr)], w_obs_raw))

    if APPLY_ICE_DYNAMICS_CORRECTION:
        # treat dhsurfaceRate as a "total" surface velocity and correct it consistently
        z0 = zeroShiftDepth_m if zeroShiftDepth_m is not None else firn_depth_m
        w0_dyn = eps_dyn * (0.0 - z0) if ANCHOR_AT_Z0 else (w_dyn[np.nanargmin(np.abs(range_m - 0.0))])
        w0_comp = float(dhsurfaceRate_myr) - float(w0_dyn)
        w_obs_comp = np.concatenate(([w0_comp], w_obs_comp))
    else:
        w_obs_comp = np.concatenate(([float(dhsurfaceRate_myr)], w_obs_comp))

# Compute strain rates from raw and corrected velocities
eps_obs_raw = moving_slope(z_obs, w_obs_raw, WINDOW_M)
eps_obs_comp = moving_slope(z_obs, w_obs_comp, WINDOW_M)

# ------------------------------------------------------------
# Map onto mesh (if Firedrake is available)
# ------------------------------------------------------------

H0 = firn_depth_m if H0_MESH is None else float(H0_MESH)
mesh, V, depth_dofs = build_stretched_depth_mesh(H0, NZ, STRETCH_P)

have_fd = mesh is not None
if have_fd:
    # Interpolate observations onto mesh dofs (constant end extrapolation)
    w_raw_on_mesh = np.interp(depth_dofs, z_obs, w_obs_raw, left=w_obs_raw[0], right=w_obs_raw[-1])
    w_comp_on_mesh = np.interp(depth_dofs, z_obs, w_obs_comp, left=w_obs_comp[0], right=w_obs_comp[-1])

    eps_raw_on_mesh = np.interp(depth_dofs, z_obs, eps_obs_raw, left=eps_obs_raw[0], right=eps_obs_raw[-1])
    eps_comp_on_mesh = np.interp(depth_dofs, z_obs, eps_obs_comp, left=eps_obs_comp[0], right=eps_obs_comp[-1])

    import firedrake as fd
    w_raw_fd = fd.Function(V, name="apres_w_raw_m_per_yr")
    w_comp_fd = fd.Function(V, name="apres_w_comp_m_per_yr")
    eps_raw_fd = fd.Function(V, name="apres_eps_raw_per_yr")
    eps_comp_fd = fd.Function(V, name="apres_eps_comp_per_yr")

    w_raw_fd.dat.data[:] = w_raw_on_mesh
    w_comp_fd.dat.data[:] = w_comp_on_mesh
    eps_raw_fd.dat.data[:] = eps_raw_on_mesh
    eps_comp_fd.dat.data[:] = eps_comp_on_mesh

    if WRITE_VTK:
        out = fd.File(os.path.join(OUTDIR, "apres_profiles_corrected.pvd"))
        out.write(w_raw_fd, w_comp_fd, eps_raw_fd, eps_comp_fd)

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

# 1) Velocity plot: raw vs corrected
plt.figure()
plt.plot(w_obs_raw, z_obs, linestyle="None", marker="o", label="ApRES velocity (raw)")
plt.plot(w_obs_comp, z_obs, linestyle="None", marker="x", label="ApRES velocity (compaction-corrected)")

if APPLY_ICE_DYNAMICS_CORRECTION and not USE_FIRN_ONLY:
    # show fitted dynamic line in the full-depth plot
    z0 = zeroShiftDepth_m if zeroShiftDepth_m is not None else firn_depth_m
    z_line = np.linspace(np.nanmin(range_m), np.nanmax(range_m), 200)
    w_line = eps_dyn * (z_line - z0) if ANCHOR_AT_Z0 else (np.polyfit(z_fit, w_fit, 1)[1] + eps_dyn * z_line)
    plt.plot(w_line, z_line, linewidth=1.5, label="Estimated ice-dynamic component (fit)")

if have_fd:
    idx = np.argsort(depth_dofs)
    plt.plot(w_raw_on_mesh[idx], depth_dofs[idx], label="Raw mapped to mesh")
    plt.plot(w_comp_on_mesh[idx], depth_dofs[idx], label="Corrected mapped to mesh")

# depth markers
plt.axhline(y=float(firn_depth_m), linewidth=0.8, linestyle="--", label="firn_depth (file)")
if zeroShiftDepth_m is not None:
    plt.axhline(y=float(zeroShiftDepth_m), linewidth=0.8, linestyle=":", label="zeroShiftDepth")

plt.gca().invert_yaxis()
plt.xlabel("Vertical velocity w (m/yr)")
plt.ylabel("Depth below surface (m)")
plt.title("ApRES vertical velocity: raw vs compaction-corrected")
plt.legend()
apply_depth_limits()
plt.tight_layout()

vel_png = os.path.join(OUTDIR, "apres_velocity_raw_vs_corrected.png")
if SAVE_PNG:
    plt.savefig(vel_png, dpi=200)
    print(f"Wrote {vel_png}")

# 2) Strain-rate plot: raw vs corrected
plt.figure()
plt.plot(eps_obs_raw, z_obs, linestyle="None", marker="o", label="εzz from raw w(z)")
plt.plot(eps_obs_comp, z_obs, linestyle="None", marker="x", label="εzz after ice-dynamics correction")

if APPLY_ICE_DYNAMICS_CORRECTION:
    # show constant background strain rate
    plt.axvline(x=float(eps_dyn), linewidth=0.8, linestyle="--", label="ε_dyn (background)")

if have_fd:
    idx = np.argsort(depth_dofs)
    plt.plot(eps_raw_on_mesh[idx], depth_dofs[idx], label="Raw mapped to mesh")
    plt.plot(eps_comp_on_mesh[idx], depth_dofs[idx], label="Corrected mapped to mesh")

plt.axhline(y=float(firn_depth_m), linewidth=0.8, linestyle="--", label="firn_depth (file)")
if zeroShiftDepth_m is not None:
    plt.axhline(y=float(zeroShiftDepth_m), linewidth=0.8, linestyle=":", label="zeroShiftDepth")

plt.gca().invert_yaxis()
plt.xlabel("Vertical strain rate εzz = dw/dz (1/yr)")
plt.ylabel("Depth below surface (m)")
plt.title(f"ApRES vertical strain rate (window={WINDOW_M:g} m): raw vs corrected")
plt.legend()
apply_depth_limits()
plt.tight_layout()

eps_png = os.path.join(OUTDIR, "apres_strainrate_raw_vs_corrected.png")
if SAVE_PNG:
    plt.savefig(eps_png, dpi=200)
    print(f"Wrote {eps_png}")

plt.show()
