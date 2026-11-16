import h5py
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
filename = "CircularDambreak.h5"

# Grid Dimensions
nx, ny = 256, 256
dx, dy = 1.0, 1.0  # 1-meter spatial resolution (200m x 200m domain)
x_ll, y_ll = 0.0, 0.0

# Physics Parameters
z_flat = 0.1          # Flat bathymetry across entire domain (m)
h_inner = 0.6         # High water depth inside circle (m)
h_outer = 0.1         # Ambient water depth outside circle (m)
dam_radius = 25.0     # Cylinder radius (m)

# Center of the domain
center_x = (nx * dx) / 2.0
center_y = (ny * dy) / 2.0

# ============================================================
# ALLOCATE ARRAYS
# ============================================================
# Fluid: [h, hu, hv]
fluid_stacked = 0.1*np.ones((3, ny, nx), dtype=np.float64)

# Terrain: [Z, Manning_n]
terrain_stacked = np.zeros((2, ny, nx), dtype=np.float64)

# Set uniform bathymetry and zero friction
terrain_stacked[0, :, :] = z_flat
terrain_stacked[1, :, :] = 0.0  # Manning n = 0 (frictionless ideal benchmark)

# ============================================================
# DEFINE CIRCULAR WATER COLUMN
# ============================================================
# Generate cell-center coordinates
y_coords = (np.arange(ny) + 0.5) * dy
x_coords = (np.arange(nx) + 0.5) * dx
xx, yy = np.meshgrid(x_coords, y_coords)

# Distance from domain center
r_dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

# Set depth profile
inside_dam = r_dist <= dam_radius
fluid_stacked[0, inside_dam] = h_inner
fluid_stacked[0, ~inside_dam] = h_outer

# Guaranteed rest state
fluid_stacked[1, :, :] = 0.0  # hu = 0.0
fluid_stacked[2, :, :] = 0.0  # hv = 0.0

# ============================================================
# DIAGNOSTICS
# ============================================================
print("=" * 60)
print(f"Dataset File      : {filename}")
print(f"Domain Extent     : {nx * dx:.1f} m x {ny * dy:.1f} m ({nx}x{ny})")
print(f"Dam Radius        : {dam_radius:.1f} m (Center at {center_x:.1f}, {center_y:.1f})")
print(f"Bathymetry Z      : Constant {z_flat:.2f} m")
print(f"Water Depth (h)   : Inner={h_inner:.2f} m | Outer={h_outer:.2f} m")
print(f"Momenta (hu, hv)  : Strictly 0.0")
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