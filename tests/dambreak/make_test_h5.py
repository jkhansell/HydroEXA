#!/usr/bin/env python3
"""
Generate HDF5 input file for a 2D rectangular dam break test case.

This is the classic St. Venant dam break problem with a planar discontinuity
along the x-axis. It is ideal for debugging fluxes because:
  - The exact solution is 1D (varies only in x), so any y-asymmetry is numerical.
  - The left-going rarefaction and right-going shock have known analytical forms.
  - Mass conservation can be checked to machine precision on a periodic domain.

Initial conditions (Sod-like):
  Left state (x < x_dam):  h = h_L, hu = 0, hv = 0
  Right state (x > x_dam): h = h_R, hu = 0, hv = 0

Bathymetry: flat (z = 0) everywhere.
"""

import h5py
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
filename = "RectangularDambreak.h5"

# Grid Dimensions
nx, ny = 256, 16  # High resolution in x, coarse in y (1D problem)
dx, dy = 1.0, 1.0  # 1-meter spatial resolution
x_ll, y_ll = 0.0, 0.0

# Physics Parameters
z_flat = 0.0          # Flat bathymetry (m)
h_L = 0.5             # High water depth left of dam (m)
h_R = 0.1             # Low water depth right of dam (m)
x_dam = (nx * dx) / 2.0  # Dam location at domain center

# ============================================================
# ALLOCATE ARRAYS
# ============================================================
# Fluid: [h, hu, hv]
fluid_stacked = np.zeros((3, ny, nx), dtype=np.float64)

# Terrain: [Z, Manning_n]
terrain_stacked = np.zeros((2, ny, nx), dtype=np.float64)

# Set uniform bathymetry and zero friction
terrain_stacked[0, :, :] = z_flat
terrain_stacked[1, :, :] = 0.0  # Manning n = 0 (frictionless)

# ============================================================
# DEFINE RECTANGULAR DAM BREAK
# ============================================================
# Generate cell-center coordinates
y_coords = (np.arange(ny) + 0.5) * dy
x_coords = (np.arange(nx) + 0.5) * dx
xx, yy = np.meshgrid(x_coords, y_coords)

# Set depth profile: step function at x_dam
left_of_dam = xx < x_dam
fluid_stacked[0, left_of_dam] = h_L
fluid_stacked[0, ~left_of_dam] = h_R

# Guaranteed rest state
fluid_stacked[1, :, :] = 0.0  # hu = 0.0
fluid_stacked[2, :, :] = 0.0  # hv = 0.0

# ============================================================
# DIAGNOSTICS
# ============================================================
total_mass = np.sum(fluid_stacked[0, :, :]) * dx * dy
print("=" * 60)
print(f"Dataset File      : {filename}")
print(f"Domain Extent     : {nx * dx:.1f} m x {ny * dy:.1f} m ({nx}x{ny})")
print(f"Dam Location      : x = {x_dam:.1f} m (center)")
print(f"Bathymetry Z      : Constant {z_flat:.2f} m")
print(f"Water Depth (h)   : Left={h_L:.2f} m | Right={h_R:.2f} m")
print(f"Momenta (hu, hv)  : Strictly 0.0")
print(f"Total Mass        : {total_mass:.6f} m^2")
print("=" * 60)

# ============================================================
# WRITE HDF5
# ============================================================
with h5py.File(filename, "w") as h5f:

    dset_fluid = h5f.create_dataset(
        "fluid",
        data=fluid_stacked,
        compression="gzip",
        dtype=np.float64
    )
    dset_fluid.attrs["dx"] = dx
    dset_fluid.attrs["dy"] = dy
    dset_fluid.attrs["x_ll"] = x_ll
    dset_fluid.attrs["y_ll"] = y_ll
    dset_fluid.attrs["nodata"] = 0.0

    dset_terrain = h5f.create_dataset(
        "terrain",
        data=terrain_stacked,
        compression="gzip",
        dtype=np.float64
    )
    dset_terrain.attrs["dx"] = dx
    dset_terrain.attrs["dy"] = dy
    dset_terrain.attrs["x_ll"] = x_ll
    dset_terrain.attrs["y_ll"] = y_ll
    dset_terrain.attrs["nodata"] = -9999.0

print(f"\nSuccessfully generated '{filename}'.")
