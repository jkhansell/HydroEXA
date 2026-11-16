#!/usr/bin/env python3

import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(
    description="Generate a HydroEXA rectangular dam-break HDF5 input."
)
parser.add_argument("--hl", type=float, default=0.005, help="Upstream water depth")
parser.add_argument("--hr", type=float, default=0.001, help="Downstream water depth")
parser.add_argument("--x0", type=float, default=5.0, help="Initial dam location")
parser.add_argument("--L", type=float, default=10.0, help="Domain extent in x")
parser.add_argument("--width", type=float, default=0.5, help="Domain extent in y")
parser.add_argument("--nx", type=int, default=256, help="Requested Level-0 cells in x")
parser.add_argument("--ny", type=int, default=16, help="Requested Level-0 cells in y")
parser.add_argument("--blocking-factor", type=int, default=16)
parser.add_argument("--terrain-ref-lev", type=int, default=2)
parser.add_argument("-o", "--output", default="RectangularDambreak.h5")
args = parser.parse_args()

assert args.L > 0.0
assert args.width > 0.0
assert 0.0 < args.x0 < args.L
assert args.nx > 0 and args.ny > 0
assert args.blocking_factor > 0
assert args.terrain_ref_lev >= 0

scale = 2 ** args.terrain_ref_lev

# Adjust Level-0 upward to satisfy blocking factor, preserve square cells.
adj_nx = int(np.ceil(args.nx / args.blocking_factor) * args.blocking_factor)
adj_ny = int(np.ceil(args.ny / args.blocking_factor) * args.blocking_factor)

# Enforce square cells: adjust ny so aspect ratio matches physical domain.
aspect_ratio = args.L / args.width
adj_ny = int(np.round(adj_nx / aspect_ratio / args.blocking_factor) * args.blocking_factor)
if adj_ny < args.blocking_factor:
    adj_ny = args.blocking_factor

# HDF5 dimensions derived from adjusted Level-0.
hdf5_nx = adj_nx * scale
hdf5_ny = adj_ny * scale

cell_size = args.L / hdf5_nx

x_ll, y_ll = 0.0, 0.0
x_hi = x_ll + hdf5_nx * cell_size
y_hi = y_ll + hdf5_ny * cell_size

level0_cell_size = cell_size * scale

z_flat = 0.0

fluid_stacked = np.zeros((3, hdf5_ny, hdf5_nx), dtype=np.float64)
terrain_stacked = np.zeros((2, hdf5_ny, hdf5_nx), dtype=np.float64)

terrain_stacked[0, :, :] = z_flat
terrain_stacked[1, :, :] = 0.0

x_coords = x_ll + (np.arange(hdf5_nx) + 0.5) * cell_size
y_coords = y_ll + (np.arange(hdf5_ny) + 0.5) * cell_size
xx, yy = np.meshgrid(x_coords, y_coords)

left_of_dam = xx < args.x0

fluid_stacked[0, left_of_dam] = args.hl
fluid_stacked[0, ~left_of_dam] = args.hr

total_volume = np.sum(fluid_stacked[0]) * cell_size * cell_size

print("=" * 60)
print(f"Dataset File       : {args.output}")
print(f"Requested Level 0  : {args.nx} x {args.ny}")
print(f"Adjusted Level 0   : {adj_nx} x {adj_ny}")
print(f"HDF5 Resolution    : {hdf5_nx} x {hdf5_ny}")
print(f"Refinement Scale   : {scale}x")
print(f"Level 0 Cell Size  : {level0_cell_size:.8f} m")
print(f"HDF5 Cell Size     : {cell_size:.8f} m")
print(f"Domain Extent      : {args.L:.6f} m x {args.width:.6f} m")
print(f"Dam Location       : x = {args.x0:.6f} m")
print(f"Water Depth        : Left={args.hl:.6f} m | Right={args.hr:.6f} m")
print(f"Total Volume       : {total_volume:.6f} m^3")
print(f"Blocking Factor    : {args.blocking_factor}")
print("Geometry Status    : VALID")
print("=" * 60)

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
