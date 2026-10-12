#!/usr/bin/env python3
import argparse
import h5py
import numpy as np

def generate_bathymetry(case_type, xx, yy, L, width, x0, y0, amplitude, sigma):
    """
    Generate bathymetry z(x,y) for different well-balanced benchmark cases.
    """
    if case_type == "gaussian":
        # Standard smooth Gaussian bump
        r2 = (xx - x0)**2 + (yy - y0)**2
        z = amplitude * np.exp(-r2 / (2.0 * sigma**2))

    elif case_type == "step":
        # Discontinuous step across x0 (Tests slope limiters & flux reconstruction)
        z = np.where(xx <= x0, amplitude, 0.0)

    elif case_type == "cone":
        # Non-smooth C1 cone (Tests face reconstruction over corners)
        r = np.sqrt((xx - x0)**2 + (yy - y0)**2)
        z = np.maximum(0.0, amplitude * (1.0 - r / sigma))

    elif case_type == "island":
        # Island with partially emerged dry region (Tests wet/dry C-property)
        r2 = (xx - x0)**2 + (yy - y0)**2
        z = amplitude * np.exp(-r2 / (2.0 * sigma**2))

    elif case_type == "twin_bump":
        # Asymmetric multi-bump (Tests AMR refinement boundaries cutting peaks)
        r1_sq = (xx - (x0 - 1.5*sigma))**2 + (yy - y0)**2
        r2_sq = (xx - (x0 + 1.5*sigma))**2 + (yy - y0)**2
        z = amplitude * (0.8 * np.exp(-r1_sq / (2.0 * sigma**2)) + 
                         1.1 * np.exp(-r2_sq / (2.0 * (0.7*sigma)**2)))

    else:
        raise ValueError(f"Unknown case type: {case_type}")

    return z

parser = argparse.ArgumentParser(
    description="Generate enhanced HydroEXA Lake-at-Rest benchmark datasets."
)

parser.add_argument("--case", type=str, default="gaussian",
                    choices=["gaussian", "step", "cone", "island", "twin_bump"],
                    help="Benchmark bathymetry case type.")
parser.add_argument("--H", type=float, default=0.010, help="Free-surface elevation")
parser.add_argument("--amplitude", type=float, default=0.002, help="Bathymetry peak/height")
parser.add_argument("--x0", type=float, default=5.0, help="Bathymetry center x")
parser.add_argument("--y0", type=float, default=None, help="Bathymetry center y")
parser.add_argument("--sigma", type=float, default=1.0, help="Bathymetry characteristic width/radius")
parser.add_argument("--L", type=float, default=10.0, help="Domain extent in x")
parser.add_argument("--width", type=float, default=0.5, help="Domain extent in y")
parser.add_argument("--nx", type=int, default=256, help="Level-0 cells in x")
parser.add_argument("--ny", type=int, default=16, help="Level-0 cells in y")
parser.add_argument("--blocking-factor", type=int, default=16)
parser.add_argument("--terrain-ref-lev", type=int, default=2)
parser.add_argument("-o", "--output", default="LakeAtRest.h5")

args = parser.parse_args()

assert args.L > 0.0 and args.width > 0.0
assert args.blocking_factor > 0 and args.terrain_ref_lev >= 0

scale = 2 ** args.terrain_ref_lev

# Grid dimension alignment for AMReX
adj_nx = int(np.ceil(args.nx / args.blocking_factor) * args.blocking_factor)
aspect_ratio = args.L / args.width
adj_ny = max(args.blocking_factor, int(np.round(adj_nx / aspect_ratio / args.blocking_factor) * args.blocking_factor))

hdf5_nx = adj_nx * scale
hdf5_ny = adj_ny * scale
cell_size = args.L / hdf5_nx

x_ll, y_ll = 0.0, 0.0
actual_L, actual_width = hdf5_nx * cell_size, hdf5_ny * cell_size
level0_cell_size = cell_size * scale

x_coords = x_ll + (np.arange(hdf5_nx) + 0.5) * cell_size
y_coords = y_ll + (np.arange(hdf5_ny) + 0.5) * cell_size
xx, yy = np.meshgrid(x_coords, y_coords)

y0 = 0.5 * actual_width if args.y0 is None else args.y0

# Bathymetry generation
z = generate_bathymetry(args.case, xx, yy, actual_L, actual_width, args.x0, y0, args.amplitude, args.sigma)

# Lake-at-rest free surface assignment: h = max(0, H - z)
h = np.maximum(0.0, args.H - z)

# Case check for dry fronts
if args.case != "island" and np.min(h) <= 0.0:
    raise ValueError(f"Dry cells detected in non-island test: min h = {np.min(h):.8e}. Increase --H or decrease --amplitude.")

# Assemble datasets
fluid_stacked = np.zeros((3, hdf5_ny, hdf5_nx), dtype=np.float64)
terrain_stacked = np.zeros((2, hdf5_ny, hdf5_nx), dtype=np.float64)

fluid_stacked[0, :, :] = h
fluid_stacked[1, :, :] = 0.0  # hu = 0
fluid_stacked[2, :, :] = 0.0  # hv = 0

terrain_stacked[0, :, :] = z
terrain_stacked[1, :, :] = 0.0  # roughness

# Diagnostics
eta = fluid_stacked[0] + terrain_stacked[0]
wet_mask = fluid_stacked[0] > 1e-12
eta_error = np.max(np.abs(eta[wet_mask] - args.H)) if np.any(wet_mask) else 0.0
total_volume = np.sum(fluid_stacked[0]) * (cell_size**2)

print("=" * 70)
print(f"Benchmark Case     : {args.case.upper()}")
print(f"Dataset File       : {args.output}")
print(f"HDF5 Resolution    : {hdf5_nx} x {hdf5_ny} (Cell Size: {cell_size:.6e} m)")
print(f"Max |eta - H| (wet): {eta_error:.6e} m")
print(f"Dry Cells Count    : {np.sum(~wet_mask)} / {hdf5_nx * hdf5_ny}")
print(f"Total Water Vol    : {total_volume:.10f} m^3")
print("=" * 70)

# Write HDF5 file
with h5py.File(args.output, "w") as h5f:
    dset_fluid = h5f.create_dataset("fluid", data=fluid_stacked, compression="gzip", dtype=np.float64)
    dset_fluid.attrs["dx"] = cell_size
    dset_fluid.attrs["dy"] = cell_size
    dset_fluid.attrs["x_ll"] = x_ll
    dset_fluid.attrs["y_ll"] = y_ll
    dset_fluid.attrs["nodata"] = 0.0

    dset_terrain = h5f.create_dataset("terrain", data=terrain_stacked, compression="gzip", dtype=np.float64)
    dset_terrain.attrs["dx"] = cell_size
    dset_terrain.attrs["dy"] = cell_size
    dset_terrain.attrs["x_ll"] = x_ll
    dset_terrain.attrs["y_ll"] = y_ll
    dset_terrain.attrs["nodata"] = -9999.0

print(f"Generated '{args.output}' successfully.")