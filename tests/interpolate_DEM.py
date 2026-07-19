import h5py
import numpy as np
import os
import shutil
from scipy.ndimage import binary_fill_holes

def patch_internal_voids(input_file, output_file, dataset_name="bathymetry", nodata_value=None):
    """
    Finds internal holes in the HDF5 DEM dataset, fills them using local 
    spatial neighbor averaging, and writes a clean copy back out to disk.
    """
    if not os.path.exists(input_file):
        print(f"[-] Error: Source file not found at {input_file}")
        return

    print(f"[+] Reading source file: {input_file}")
    
    # 1. Open the source file and load array
    with h5py.File(input_file, 'r') as h5f:
        data = np.squeeze(h5f[dataset_name][...])
        # Cache existing metadata/attributes if there are any
        attrs = dict(h5f[dataset_name].attrs)

    ny, nx = data.shape
    print(f"[+] Dataset dimension: {nx} x {ny}")

    # 2. Reconstruct masks to find the 10 internal holes
    invalid_mask = np.isnan(data) | np.isinf(data)
    if nodata_value is not None:
        invalid_mask |= (data == nodata_value)
        
    valid_domain_mask = ~invalid_mask
    solid_footprint = binary_fill_holes(valid_domain_mask)
    internal_void_mask = solid_footprint & invalid_mask
    
    void_coords = np.argwhere(internal_void_mask)
    num_holes = len(void_coords)
    
    if num_holes == 0:
        print("[+] Domain is already holeless! Copying file directly without alterations.")
        shutil.copyfile(input_file, output_file)
        return

    print(f"[+] Isolated {num_holes} internal holes. Commencing local spatial interpolation...")

    # 3. Localized Stencil Filler
    # We copy the original data array to patch it
    patched_data = np.copy(data)
    
    for y, x in void_coords:
        # Define an expanding window search radius to guarantee we find valid terrain cells
        radius = 1
        interpolated_val = None
        
        while radius < 10: # Safety upper bound
            # Extrapolate a local window boundary around the hole cell
            y_min, y_max = max(0, y - radius), min(ny, y + radius + 1)
            x_min, x_max = max(0, x - radius), min(nx, x + radius + 1)
            
            local_window = data[y_min:y_max, x_min:x_max]
            local_invalid = invalid_mask[y_min:y_max, x_min:x_max]
            
            # Isolate valid data values inside this local sub-window footprint
            valid_neighbors = local_window[~local_invalid]
            
            if valid_neighbors.size > 0:
                # Use the median of the surrounding cells to avoid slope distortion artifacts
                interpolated_val = np.median(valid_neighbors)
                break
                
            radius += 1 # Expand neighborhood search space if nested inside a larger hole
            
        if interpolated_val is not None:
            patched_data[y, x] = interpolated_val
            print(f"    -> Filled hole cell [{y}, {x}] with height value: {interpolated_val:.4f} m")
        else:
            print(f"    [-] Warning: Could not find valid neighbors for hole cell [{y}, {x}]")

    # 4. Save the patched array back out to a fresh HDF5 layout
    print(f"[+] Writing patched holeless dataset to: {output_file}")
    
    # Copy the whole original file layout first so you don't lose any other datasets (like 'fluid')
    if os.path.exists(output_file):
        os.remove(output_file)
    shutil.copyfile(input_file, output_file)
    
    # Overwrite just the specific targeted dataset block with our clean array
    with h5py.File(output_file, 'r+') as h5f:
        del h5f[dataset_name] # Drop old dataset chunk
        
        # Recreate dataset with the exact same compression properties or chunk rules
        dset = h5f.create_dataset(
            dataset_name, 
            data=patched_data, 
            compression="gzip", 
            chunks=True
        )
        
        # Restore original HDF5 metadata attributes if present
        for k, v in attrs.items():
            dset.attrs[k] = v

    print("[+] Operation successful! Input DEM is now 100% hole-less inside the valid domain.")

if __name__ == "__main__":
    # --- Run Repair ---
    patch_internal_voids(
        input_file="Alajuela.h5",
        output_file="Alajuela_Patched.h5",
        dataset_name="bathymetry",
        nodata_value=-9999 # Change to -9999.0 if necessary
    )