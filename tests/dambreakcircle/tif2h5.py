import h5py
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, distance_transform_edt

# ============================================================
# INPUT / OUTPUT CONFIG
# ============================================================
tiff_filename = "/data/jovillalobos/CFD/data/rasterization_CR/geooutputs/Puntos_Alajuela_Alajuela.tif"
filename = "Alajuela.h5"

default_water_depth = 2.0  # meters in detected rivers/valleys
fluid_nodata = 0.0          # Fluid initialized to 0.0 for safe C++ state handling
terrain_nodata = -9999.0    # Bathymetry NODATA flag

# ============================================================
# READ TERRAIN
# ============================================================
with rasterio.open(tiff_filename) as src:
    Z_bathymetry = src.read(1).astype(np.float64)

    # Flip Y to match AMReX Cartesian coordinate convention (Y=0 at bottom)
    Z_bathymetry = Z_bathymetry[::-1, :]

    dx = float(src.transform.a)
    dy = float(abs(src.transform.e))

    x_ll = float(src.transform.c)
    y_ul = float(src.transform.f)
    y_ll = y_ul - Z_bathymetry.shape[0] * dy

    tiff_nodata = src.nodata
    if tiff_nodata is None:
        tiff_nodata = -9999.0

# ============================================================
# VALID TERRAIN MASK
# ============================================================
if np.isnan(tiff_nodata):
    valid_mask = ~np.isnan(Z_bathymetry)
else:
    valid_mask = (Z_bathymetry != tiff_nodata) & np.isfinite(Z_bathymetry)

ny, nx = Z_bathymetry.shape

# ============================================================
# BOUNDARY-SAFE GRADIENT & TPI ANALYSIS
# ============================================================
# 1. Fill invalid cells using distance transform to prevent boundary gradient artifacts
indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
Z_filled = Z_bathymetry[tuple(indices)]

# 2. Compute spatial elevation gradients
dZ_dy, dZ_dx = np.gradient(Z_filled, dy, dx)
grad_mag = np.sqrt(dZ_dx**2 + dZ_dy**2)
grad_mag[~valid_mask] = 0.0

# 3. Compute Normalized Local Topographic Position Index (TPI)
neighborhood_size = 31

valid_weights = valid_mask.astype(np.float64)
Z_weighted = np.where(valid_mask, Z_bathymetry, 0.0)

sum_Z = uniform_filter(Z_weighted, size=neighborhood_size, mode='constant', cval=0.0)
sum_w = uniform_filter(valid_weights, size=neighborhood_size, mode='constant', cval=0.0)

# SAFE DIVISION (fixes division-by-zero warnings)
mean_neighborhood = np.copy(Z_bathymetry)
np.divide(sum_Z, sum_w, out=mean_neighborhood, where=(sum_w > 0))

tpi = np.zeros_like(Z_bathymetry)
tpi[valid_mask] = Z_bathymetry[valid_mask] - mean_neighborhood[valid_mask]

# ============================================================
# SELECT RIVER VALLEYS
# ============================================================
tpi_threshold = np.percentile(tpi[valid_mask], 15)
water_mask = (tpi < tpi_threshold) & valid_mask

if water_mask.sum() == 0:
    water_mask = (tpi < np.percentile(tpi[valid_mask], 10)) & valid_mask

# ============================================================
# ALLOCATE & INITIALIZE ARRAYS
# ============================================================
fluid_stacked = np.full((3, ny, nx), fluid_nodata, dtype=np.float64)
fluid_stacked[0, water_mask] = default_water_depth
fluid_stacked[1, valid_mask] = 0.0  # hu = 0.0
fluid_stacked[2, valid_mask] = 0.0  # hv = 0.0

terrain_stacked = np.full((2, ny, nx), terrain_nodata, dtype=np.float64)
terrain_stacked[0, valid_mask] = Z_bathymetry[valid_mask]
terrain_stacked[1, valid_mask] = 0.0  # Friction / Manning's n

# ============================================================
# DIAGNOSTICS
# ============================================================
print("=" * 60)
print(f"Grid size         : {nx} x {ny}")
print(f"dx, dy            : {dx:.3f}, {dy:.3f} m")
print(f"Terrain min/max   : {Z_bathymetry[valid_mask].min():.2f} m / {Z_bathymetry[valid_mask].max():.2f} m")
print(f"Valid cells       : {valid_mask.sum()}")
print(f"Wet River cells   : {water_mask.sum()}")
print(f"Max Gradient mag  : {grad_mag[valid_mask].max():.4f} m/m")
print(f"Fluid NODATA val  : {fluid_nodata}")
print(f"Terrain NODATA val: {terrain_nodata}")
print("=" * 60)

# ============================================================
# WRITE HDF5
# ============================================================
with h5py.File(filename, "w") as h5f:

    dset_fluid = h5f.create_dataset("fluid", data=fluid_stacked, compression="gzip", dtype=np.float64)
    dset_fluid.attrs["dx"] = dx
    dset_fluid.attrs["dy"] = dy
    dset_fluid.attrs["x_ll"] = x_ll
    dset_fluid.attrs["y_ll"] = y_ll
    dset_fluid.attrs["nodata"] = fluid_nodata

    dset_terrain = h5f.create_dataset("terrain", data=terrain_stacked, compression="gzip", dtype=np.float64)
    dset_terrain.attrs["dx"] = dx
    dset_terrain.attrs["dy"] = dy
    dset_terrain.attrs["x_ll"] = x_ll
    dset_terrain.attrs["y_ll"] = y_ll
    dset_terrain.attrs["nodata"] = terrain_nodata

print(f"\nSuccessfully generated '{filename}' with zero division warnings.")