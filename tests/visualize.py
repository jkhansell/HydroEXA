import h5py
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# READ GENERATED DATA FROM FILE
# ============================================================
filename = "Alajuela.h5"

with h5py.File(filename, "r") as h5f:
    Z_bathymetry = h5f["bathymetry"][:]
    dx = h5f["bathymetry"].attrs["dx"]
    dy = h5f["bathymetry"].attrs["dy"]
    nodata_value = h5f["bathymetry"].attrs["nodata"]

# Reconstruct physical axes lengths based on dataset shape
ny, nx = Z_bathymetry.shape
Lx = nx * dx
Ly = ny * dy

x = np.arange(0, Lx, dx)
y = np.arange(0, Ly, dy)

# ============================================================
# MASK NODATA FOR CLEAN COLOR RANGE REPRESENTATION
# ============================================================
# Masking -9999.0 stops it from crushing your real topography color scaling
Z_active = np.ma.masked_equal(Z_bathymetry, nodata_value)

# ============================================================
# TOP-DOWN 2D VISUALIZATION TIMELINE
# ============================================================
plt.figure(figsize=(14, 14), dpi=250)

# Set up a distinct facecolor for the plot background to visualize pruned spaces clearly
ax = plt.gca()
ax.set_facecolor('#222222') # Dark gray for pruned NODATA regions

# Render the color mesh
# 'terrain' or 'viridis' or 'gist_earth' work wonderfully for topography
mesh = plt.pcolormesh(x, y, Z_active, cmap='terrain', shading='auto')

# Visual configurations
cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
cbar.set_label('Topography Elevation ($Z_{bathymetry}$) [m]', fontsize=11)

plt.title('Top-Down 2D Bathymetry Field ($Z$) with Irregular NODATA Pruning Mask', 
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Physical X Dimension [m]', fontsize=11)
plt.ylabel('Physical Y Dimension [m]', fontsize=11)

# Ensure physically accurate aspect ratio mapping (1:1 spatial scaling)
plt.gca().set_aspect('equal', adjustable='box')

plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.grid(True, linestyle=':', alpha=0.3, color='white')

plt.tight_layout()
plt.savefig("viz.png")
plt.close()


plt.imshow(Z_active)
plt.savefig("z.png")
plt.close()