#!/usr/bin/env python3

import argparse
import h5py
import numpy as np


parser = argparse.ArgumentParser(
    description="Generate a HydroEXA lake-at-rest HDF5 input."
)

parser.add_argument(
    "--H",
    type=float,
    default=0.010,
    help="Constant free-surface elevation",
)

parser.add_argument(
    "--amplitude",
    type=float,
    default=0.002,
    help="Gaussian bathymetry amplitude",
)

parser.add_argument(
    "--x0",
    type=float,
    default=5.0,
    help="Bathymetry bump center in x",
)

parser.add_argument(
    "--y0",
    type=float,
    default=None,
    help="Bathymetry bump center in y; defaults to domain center",
)

parser.add_argument(
    "--sigma",
    type=float,
    default=1.0,
    help="Gaussian bathymetry standard deviation",
)

parser.add_argument(
    "--L",
    type=float,
    default=10.0,
    help="Domain extent in x",
)

parser.add_argument(
    "--width",
    type=float,
    default=0.5,
    help="Requested domain extent in y",
)

parser.add_argument(
    "--nx",
    type=int,
    default=256,
    help="Requested Level-0 cells in x",
)

parser.add_argument(
    "--ny",
    type=int,
    default=16,
    help="Requested Level-0 cells in y",
)

parser.add_argument(
    "--blocking-factor",
    type=int,
    default=16,
)

parser.add_argument(
    "--terrain-ref-lev",
    type=int,
    default=2,
)

parser.add_argument(
    "-o",
    "--output",
    default="LakeAtRest.h5",
)

args = parser.parse_args()

assert args.L > 0.0
assert args.width > 0.0
assert args.H > 0.0
assert args.amplitude >= 0.0
assert 0.0 < args.x0 < args.L
assert args.nx > 0 and args.ny > 0
assert args.blocking_factor > 0
assert args.terrain_ref_lev >= 0

scale = 2 ** args.terrain_ref_lev

# ----------------------------------------------------------------------
# Adjust Level-0 dimensions to satisfy the AMReX blocking factor.
# ----------------------------------------------------------------------

adj_nx = int(
    np.ceil(args.nx / args.blocking_factor)
    * args.blocking_factor
)

# Preserve the requested physical aspect ratio while enforcing
# the blocking factor.
aspect_ratio = args.L / args.width

adj_ny = int(
    np.round(
        adj_nx
        / aspect_ratio
        / args.blocking_factor
    )
    * args.blocking_factor
)

if adj_ny < args.blocking_factor:
    adj_ny = args.blocking_factor

# ----------------------------------------------------------------------
# HDF5 resolution.
#
# HydroEXA uses terrain_ref_lev to relate the HDF5 resolution to
# Level-0 AMReX cells.
# ----------------------------------------------------------------------

hdf5_nx = adj_nx * scale
hdf5_ny = adj_ny * scale

# Square HDF5 cells.
cell_size = args.L / hdf5_nx

# Actual physical domain implied by the adjusted grid.
x_ll = 0.0
y_ll = 0.0

x_hi = x_ll + hdf5_nx * cell_size
y_hi = y_ll + hdf5_ny * cell_size

actual_L = x_hi - x_ll
actual_width = y_hi - y_ll

level0_cell_size = cell_size * scale

# ----------------------------------------------------------------------
# Coordinates at cell centers.
# ----------------------------------------------------------------------

x_coords = x_ll + (np.arange(hdf5_nx) + 0.5) * cell_size
y_coords = y_ll + (np.arange(hdf5_ny) + 0.5) * cell_size

xx, yy = np.meshgrid(x_coords, y_coords)

if args.y0 is None:
    y0 = 0.5 * actual_width
else:
    y0 = args.y0

assert 0.0 < y0 < actual_width

# ----------------------------------------------------------------------
# Gaussian bathymetry.
#
#       z(x,y) = A exp(-r^2 / 2 sigma^2)
#
# Lake-at-rest condition:
#
#       eta = h + z = H
#
# therefore:
#
#       h = H - z
#       hu = 0
#       hv = 0
# ----------------------------------------------------------------------

r2 = (xx - args.x0) ** 2 + (yy - y0) ** 2

z = args.amplitude * np.exp(
    -r2 / (2.0 * args.sigma ** 2)
)

h = args.H - z

if np.min(h) <= 0.0:
    raise ValueError(
        f"Dry cells detected: minimum h = {np.min(h):.8e}. "
        f"Increase H or reduce --amplitude."
    )

# ----------------------------------------------------------------------
# HydroEXA state arrays.
#
# fluid:
#   [0] = h
#   [1] = hu
#   [2] = hv
#
# terrain:
#   [0] = z
#   [1] = roughness
# ----------------------------------------------------------------------

fluid_stacked = np.zeros(
    (3, hdf5_ny, hdf5_nx),
    dtype=np.float64,
)

terrain_stacked = np.zeros(
    (2, hdf5_ny, hdf5_nx),
    dtype=np.float64,
)

fluid_stacked[0, :, :] = h
fluid_stacked[1, :, :] = 0.0
fluid_stacked[2, :, :] = 0.0

terrain_stacked[0, :, :] = z
terrain_stacked[1, :, :] = 0.0

# ----------------------------------------------------------------------
# Diagnostics.
# ----------------------------------------------------------------------

eta = fluid_stacked[0] + terrain_stacked[0]

eta_error = np.max(
    np.abs(eta - args.H)
)

total_volume = (
    np.sum(fluid_stacked[0])
    * cell_size
    * cell_size
)

print("=" * 70)
print(f"Dataset File       : {args.output}")
print()
print(f"Requested Level 0  : {args.nx} x {args.ny}")
print(f"Adjusted Level 0   : {adj_nx} x {adj_ny}")
print(f"HDF5 Resolution    : {hdf5_nx} x {hdf5_ny}")
print(f"Refinement Scale   : {scale}x")
print()
print(f"Level 0 Cell Size  : {level0_cell_size:.10f} m")
print(f"HDF5 Cell Size     : {cell_size:.10f} m")
print(f"dx == dy           : {np.isclose(cell_size, cell_size)}")
print()
print(f"Domain Extent      : {actual_L:.10f} m x {actual_width:.10f} m")
print(f"Requested Width    : {args.width:.10f} m")
print()
print(f"Free Surface H     : {args.H:.10f} m")
print(f"Bathymetry Max     : {np.max(z):.10f} m")
print(f"Water Depth Min    : {np.min(h):.10f} m")
print(f"Water Depth Max    : {np.max(h):.10f} m")
print()
print(f"Bathymetry Center  : ({args.x0:.6f}, {y0:.6f})")
print(f"Bathymetry Sigma   : {args.sigma:.6f} m")
print(f"Bathymetry Amp.    : {args.amplitude:.6f} m")
print()
print(f"Max |eta - H|      : {eta_error:.6e} m")
print(f"Total Volume       : {total_volume:.10f} m^3")
print()
print(f"Blocking Factor    : {args.blocking_factor}")
print(f"Terrain Ref Level  : {args.terrain_ref_lev}")
print("Geometry Status    : VALID")
print("=" * 70)

# ----------------------------------------------------------------------
# Write HDF5.
#
# IMPORTANT:
# HydroEXA expects "fluid" and "terrain" to be datasets directly.
# ----------------------------------------------------------------------

with h5py.File(args.output, "w") as h5f:

    dset_fluid = h5f.create_dataset(
        "fluid",
        data=fluid_stacked,
        compression="gzip",
        dtype=np.float64,
    )

    dset_fluid.attrs["dx"] = cell_size
    dset_fluid.attrs["dy"] = cell_size
    dset_fluid.attrs["x_ll"] = x_ll
    dset_fluid.attrs["y_ll"] = y_ll
    dset_fluid.attrs["nodata"] = 0.0

    dset_terrain = h5f.create_dataset(
        "terrain",
        data=terrain_stacked,
        compression="gzip",
        dtype=np.float64,
    )

    dset_terrain.attrs["dx"] = cell_size
    dset_terrain.attrs["dy"] = cell_size
    dset_terrain.attrs["x_ll"] = x_ll
    dset_terrain.attrs["y_ll"] = y_ll
    dset_terrain.attrs["nodata"] = -9999.0

print(f"\nSuccessfully generated '{args.output}'.")