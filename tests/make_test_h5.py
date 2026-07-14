import h5py
import numpy as np

# ============================================================
# DOMAIN SETUP
# ============================================================
Lx, Ly = 15000.0, 5000.0
dx, dy = 5.0, 5.0

x = np.arange(0, Lx, dx)
y = np.arange(0, Ly, dy)
X, Y = np.meshgrid(x, y)

# ============================================================
# TERRAIN GENERATION (BATHYMETRY)
# ============================================================
amplitude, wavelength = 550.0, 3000.0
centerline = Ly / 2.0 + amplitude * np.sin(2 * np.pi * X / wavelength)
distance = Y - centerline

channel_depth, channel_sigma = 6.0, 80.0
channel = channel_depth * np.exp(-(distance**2) / (2 * channel_sigma**2))

eta = 2.0 * (Y - Ly / 2.0) / Ly
valley = 8.0 * (eta**8)

texture = 0.5 * np.sin(2 * np.pi * X / 250.0) * np.cos(2 * np.pi * Y / 250.0)
longitudinal = 1.0e-4 * (Lx - X)

# Pure physical topography matrix
Z_bathymetry = valley + texture + longitudinal - channel

# ============================================================
# INITIAL FLUID CONDITIONAL SYNTHESIS
# ============================================================
# Let's say initial condition is a steady water surface elevation or a dam breach
water_surface_height = 12.0 
h_fluid = np.maximum(0.0, water_surface_height - Z_bathymetry)

# Initialize momentum components (e.g. constant initial discharges downstream)
hu_fluid = h_fluid * 0.5  # 0.5 m/s initial velocity in X
hv_fluid = np.zeros_like(h_fluid)

# ============================================================
# APPLY IRREGULAR NODATA BOUNDARY PRUNING
# ============================================================
nodata_value = np.nan
mask_width = 700.0 
prune_mask = np.abs(distance) > mask_width

# Prune Bathymetry
Z_bathymetry[prune_mask] = nodata_value

# Prune Fluid arrays (Dry/Inactive states inside masked wall domains)
h_fluid[prune_mask] = nodata_value
hu_fluid[prune_mask] = nodata_value
hv_fluid[prune_mask] = nodata_value

# Stack fluid states into a 3D matrix. Layout: [Component][Y][X]
# This cleanly translates to multi-component reading layouts
fluid_stacked = np.stack([h_fluid, hu_fluid, hv_fluid], axis=0)

# ============================================================
# WRITE HDF5 INPUT
# ============================================================
filename = "simulation_input.h5"

with h5py.File(filename, "w") as h5f:
    # 1. Write multi-component 3D Fluid Dataset [Comp][Y][X]
    dset_fluid = h5f.create_dataset("fluid", data=fluid_stacked, dtype="float64", compression="gzip")
    dset_fluid.attrs["dx"] = dx
    dset_fluid.attrs["dy"] = dy
    dset_fluid.attrs["x_ll"] = 0.0
    dset_fluid.attrs["y_ll"] = 0.0
    dset_fluid.attrs["nodata"] = nodata_value
    
    # 2. Write 2D Bathymetry Dataset [Y][X]
    dset_bath = h5f.create_dataset("bathymetry", data=Z_bathymetry, dtype="float64", compression="gzip")
    dset_bath.attrs["dx"] = dx
    dset_bath.attrs["dy"] = dy
    dset_bath.attrs["x_ll"] = 0.0
    dset_bath.attrs["y_ll"] = 0.0
    dset_bath.attrs["nodata"] = nodata_value

print(f"Generated {filename}")
print(f"  -> 'fluid' dataset shape: {fluid_stacked.shape} (Components x Rows x Cols)")
print(f"  -> 'bathymetry' dataset shape: {Z_bathymetry.shape} (Rows x Cols)")