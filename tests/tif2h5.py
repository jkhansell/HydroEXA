import h5py
import numpy as np
import rasterio

# ============================================================
# READ INPUT TIFF (BATHYMETRY / TOPO)
# ============================================================
tiff_filename = "/lustre/orion/geo161/scratch/jkhansell/rasterization_CR/geooutputs/Puntos_Alajuela/Puntos_Alajuela_Alajuela.tif"  # Change to your actual TIFF path
with rasterio.open(tiff_filename) as src:
    # Read the first band as the bathymetry array [Y][X]
    Z_bathymetry = src.read(1).astype("float64")

    # === ADD THIS LINE TO FLIP THE Y-AXIS ===
    Z_bathymetry = Z_bathymetry[::-1, :] 

    # Extract spatial resolution
    dx = float(src.transform[0])
    dy = float(abs(src.transform[4]))

    # === EXTRACT TRUE GEOSPATIAL COORDINATES ===
    # src.transform[2] is the exact Western (Left) edge: X_ll
    x_ll = float(src.transform[2])
    
    # src.transform[5] is the Northern (Top) edge. Since we flipped the array 
    # to orient from the bottom up, the new lower-left Y is the original top 
    # minus the total physical height of the image.
    y_ul = float(src.transform[5])
    total_height_m = Z_bathymetry.shape[0] * dy
    y_ll = float(y_ul - total_height_m)

    # Handle nodata values dynamically from the TIFF
    tiff_nodata = -9999

# ============================================================
# INITIAL FLUID CONDITIONAL SYNTHESIS
# ============================================================
# Create a mask for valid vs invalid terrain data
valid_mask = ~np.isnan(Z_bathymetry) if np.isnan(tiff_nodata) else (Z_bathymetry != tiff_nodata)

# === DYNAMIC WATER SURFACE HEIGHT FIX ===
# San José elevations are >1000m. We grab the lowest elevation point in your valid data
# and set the water level to be 5.0 meters above that baseline. 
# (Adjust the + 5.0 offset or replace with a flat contour like 1200.0 depending on your test case needs)
lowest_point = np.nanmin(Z_bathymetry[valid_mask])
water_surface_height = lowest_point + 5.0

# Initialize fluid states with safe fallbacks
h_fluid = np.full_like(Z_bathymetry, np.nan)
hu_fluid = np.full_like(Z_bathymetry, np.nan)
hv_fluid = np.full_like(Z_bathymetry, np.nan)

# Compute fluid parameters ONLY where terrain data is valid
h_fluid[valid_mask] = np.maximum(0.0, water_surface_height - Z_bathymetry[valid_mask])
hu_fluid[valid_mask] = h_fluid[valid_mask] * 0.0 # 0.5 m/s downstream velocity
hv_fluid[valid_mask] = 0.0

# Stack fluid states into a 3D matrix. Layout: [Component][Y][X]
fluid_stacked = np.stack([h_fluid, hu_fluid, hv_fluid], axis=0)

# Target HDF5 nodata value (using standard np.nan as per your original format)
hdf5_nodata = np.nan

# ============================================================
# WRITE HDF5 INPUT
# ============================================================
filename = "Alajuela.h5"

with h5py.File(filename, "w") as h5f:
    # 1. Write multi-component 3D Fluid Dataset [Comp][Y][X]
    dset_fluid = h5f.create_dataset(
        "fluid", data=fluid_stacked, dtype="float64", compression="gzip"
    )
    dset_fluid.attrs["dx"] = dx
    dset_fluid.attrs["dy"] = dy
    dset_fluid.attrs["x_ll"] = x_ll
    dset_fluid.attrs["y_ll"] = y_ll
    dset_fluid.attrs["nodata"] = hdf5_nodata

    # 2. Write 2D Bathymetry Dataset [Y][X]
    dset_bath = h5f.create_dataset(
        "bathymetry", data=Z_bathymetry, dtype="float64", compression="gzip"
    )
    dset_bath.attrs["dx"] = dx
    dset_bath.attrs["dy"] = dy
    dset_bath.attrs["x_ll"] = x_ll
    dset_bath.attrs["y_ll"] = y_ll
    dset_bath.attrs["nodata"] = hdf5_nodata

print(f"Generated {filename} successfully from {tiff_filename}")
print(f"  -> Dynamic Water Level: {water_surface_height:.2f} m (Lowest Terrain Point: {lowest_point:.2f} m)")
print(f"  -> Extracted metadata: dx={dx}, dy={dy}, x_ll={x_ll}, y_ll={y_ll}")
print(f"  -> 'fluid' dataset shape: {fluid_stacked.shape} (Components x Rows x Cols)")
print(f"  -> 'bathymetry' dataset shape: {Z_bathymetry.shape} (Rows x Cols)")