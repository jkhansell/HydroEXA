import h5py
import numpy as np
import os
import shutil
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.interpolate import NearestNDInterpolator

def patch_internal_voids(input_file, output_file, dataset_name="bathymetry", nodata_value=None):
    if not os.path.exists(input_file):
        print(f"[-] Error: Source file not found at {input_file}")
        return

    print(f"[+] Reading source file: {input_file}")
    
    with h5py.File(input_file, 'r') as h5f:
        data = np.squeeze(h5f[dataset_name][...])
        attrs = dict(h5f[dataset_name].attrs)

    ny, nx = data.shape
    print(f"[+] Dataset dimension: {nx} x {ny}")

    # 1. Build invalid and domain masks
    invalid_mask = np.isnan(data) | np.isinf(data)
    if nodata_value is not None:
        invalid_mask |= (data == nodata_value)
        
    valid_domain_mask = ~invalid_mask
    solid_footprint = binary_fill_holes(valid_domain_mask)
    
    # Internal holes inside the active terrain footprint
    internal_void_mask = solid_footprint & invalid_mask
    
    num_holes = np.count_nonzero(internal_void_mask)
    if num_holes == 0:
        print("[+] Domain is already holeless! Copying file directly.")
        shutil.copyfile(input_file, output_file)
        return

    print(f"[+] Isolated {num_holes} internal hole cells. Commencing spatial interpolation...")

    # 2. Fast & Smooth Interpolation for Medium/Large Holes
    # Extract coordinates of valid boundary cells and hole cells
    valid_coords = np.argwhere(valid_domain_mask)
    valid_values = data[valid_domain_mask]
    void_coords = np.argwhere(internal_void_mask)

    # Nearest / Distance-weighted spatial interpolator
    # For very large grids, we fit nearest neighbor on the valid perimeter
    interpolator = NearestNDInterpolator(valid_coords, valid_values)
    patched_values = interpolator(void_coords)

    patched_data = np.copy(data)
    patched_data[internal_void_mask] = patched_values

    # 3. Save patched dataset
    print(f"[+] Writing patched holeless dataset to: {output_file}")
    if os.path.exists(output_file):
        os.remove(output_file)
    shutil.copyfile(input_file, output_file)

    with h5py.File(output_file, 'r+') as h5f:
        del h5f[dataset_name]
        dset = h5f.create_dataset(
            dataset_name, 
            data=patched_data, 
            compression="gzip", 
            chunks=True
        )
        for k, v in attrs.items():
            dset.attrs[k] = v

    print("[+] Operation successful! DEM holes filled cleanly.")

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

def visualize_dem_patching(
    original_h5="SJ.h5", 
    patched_h5="SJ_Patched.h5", 
    dataset_name="bathymetry", 
    nodata_value=-9999,
    output_file="dem_interpolation_check.png"
):
    print(f"[+] Loading original: {original_h5}")
    with h5py.File(original_h5, "r") as f:
        orig_data = np.squeeze(f[dataset_name][...])

    print(f"[+] Loading patched:  {patched_h5}")
    with h5py.File(patched_h5, "r") as f:
        patched_data = np.squeeze(f[dataset_name][...])

    # Convert invalid/nodata sentinel values to NaN for plotting
    orig_viz = np.copy(orig_data)
    patched_viz = np.copy(patched_data)

    invalid_mask = np.isnan(orig_data) | np.isinf(orig_data)
    if nodata_value is not None:
        invalid_mask |= (orig_data == nodata_value)

    orig_viz[invalid_mask] = np.nan
    
    if nodata_value is not None:
        patched_viz[patched_data == nodata_value] = np.nan

    # Identify where internal holes were located
    valid_domain_mask = ~invalid_mask
    solid_footprint = binary_fill_holes(valid_domain_mask)
    hole_mask = solid_footprint & invalid_mask

    hole_coords = np.argwhere(hole_mask)
    num_holes = len(hole_coords)
    print(f"[+] Identified {num_holes} patched hole cells.")

    # Calculate global color bounds for consistent scale
    vmin = np.nanmin(orig_viz)
    vmax = np.nanmax(orig_viz)

    fig = plt.figure(figsize=(18, 10), dpi=300)
    fig.patch.set_facecolor('#1e1e1e')

    # Subplot 1: Original Raw DEM
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#121212')
    im1 = ax1.imshow(orig_viz, cmap="terrain", vmin=vmin, vmax=vmax, origin="lower")
    ax1.set_title("1. Original Raw DEM (Unpatched Holes)", color="white", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=ax1, label="Elevation [m]", pad=0.02)
    ax1.tick_params(colors="white")

    # Subplot 2: Patched DEM
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor('#121212')
    im2 = ax2.imshow(patched_viz, cmap="terrain", vmin=vmin, vmax=vmax, origin="lower")
    ax2.set_title("2. Patched DEM (Holes Filled)", color="white", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=ax2, label="Elevation [m]", pad=0.02)
    ax2.tick_params(colors="white")

    # Subplot 3: Absolute Elevation Difference (|Patched - Original|)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor('#121212')
    diff = np.abs(patched_data - orig_data)
    diff[~hole_mask] = np.nan  # Mask non-hole cells
    im3 = ax3.imshow(diff, cmap="magma", origin="lower")
    ax3.set_title("3. Spatial Hole Distribution (Delta Magnitude)", color="white", fontsize=12, fontweight="bold")
    plt.colorbar(im3, ax=ax3, label="Interpolated Fill [m]", pad=0.02)
    ax3.tick_params(colors="white")

    # Subplot 4: Zoomed-In Patch Focus (Centered on the largest hole cluster)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor('#121212')

    if num_holes > 0:
        center_y, center_x = hole_coords[0]
        radius = 25  # 50x50 cell window around the hole
        y0, y1 = max(0, center_y - radius), min(orig_data.shape[0], center_y + radius)
        x0, x1 = max(0, center_x - radius), min(orig_data.shape[1], center_x + radius)

        crop_patched = patched_viz[y0:y1, x0:x1]
        im4 = ax4.imshow(crop_patched, cmap="terrain", origin="lower", extent=[x0, x1, y0, y1])
        
        # Highlight hole coordinates with red circles
        for hy, hx in hole_coords:
            if y0 <= hy < y1 and x0 <= hx < x1:
                ax4.plot(hx + 0.5, hy + 0.5, "ro", markersize=6, fillstyle="none", markeredgewidth=1.5)

        ax4.set_title(f"4. Zoom View around Hole Cluster at [{center_y}, {center_x}]", color="white", fontsize=12, fontweight="bold")
        plt.colorbar(im4, ax=ax4, label="Elevation [m]", pad=0.02)
    else:
        ax4.text(0.5, 0.5, "No internal holes detected", color="white", ha="center", va="center")
        ax4.set_title("4. Zoom View (No Holes)", color="white", fontsize=12, fontweight="bold")
    
    ax4.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"[+] Diagnostic image saved to: {output_file}")
    

if __name__ == "__main__":
    patch_internal_voids(
        input_file="SJ.h5",
        output_file="SJ_Patched.h5",
        dataset_name="bathymetry",
        nodata_value=-9999
    )

    visualize_dem_patching(
        original_h5="SJ.h5",
        patched_h5="SJ_Patched.h5",
        dataset_name="bathymetry",
        nodata_value=-9999,
        output_file="dem_interpolation_check.png"
    )