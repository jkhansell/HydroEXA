import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import binary_fill_holes

def plot_dem_holes(file_path, dataset_name="bathymetry", nodata_value=None, output_file="dem_internal_holes.png", padding=50):
    """
    Locates internal holes in the DEM, zooms in on the affected region, 
    and saves a visualization marking their exact locations.
    """
    if not os.path.exists(file_path):
        print(f"[-] Error: File not found at {file_path}")
        return

    with h5py.File(file_path, 'r') as h5f:
        data = np.squeeze(h5f[dataset_name][...])
        
        # 1. Recreate the void masks
        invalid_mask = np.isnan(data) | np.isinf(data)
        if nodata_value is not None:
            invalid_mask |= (data == nodata_value)
            
        valid_domain_mask = ~invalid_mask
        solid_footprint = binary_fill_holes(valid_domain_mask)
        internal_void_mask = solid_footprint & invalid_mask
        
        void_coords = np.argwhere(internal_void_mask)
        num_holes = len(void_coords)
        
        if num_holes == 0:
            print("[+] No internal holes found to plot!")
            return

        print(f"[+] Found {num_holes} holes. Generating focused zoom plot...")

        # 2. Calculate the bounding box around all holes to determine the zoom window
        # void_coords format is [Row(Y), Col(X)]
        min_y, min_x = np.min(void_coords, axis=0)
        max_y, max_x = np.max(void_coords, axis=0)

        # Apply spatial padding around the holes so we can see the surrounding terrain context
        y_start = max(0, min_y - padding)
        y_end = min(data.shape[0], max_y + padding + 1)
        x_start = max(0, min_x - padding)
        x_end = min(data.shape[1], max_x + padding + 1)

        # 3. Slice out the sub-regions for plotting
        cropped_data = data[y_start:y_end, x_start:x_end]
        
        # 4. Initialize Plotting Window
        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
        ax.set_facecolor("#1a1a1a") # Dark background for background voids

        # Mask out NaNs locally just for clean color plotting
        plot_data = np.copy(cropped_data)
        plot_data[np.isnan(plot_data)] = np.nanmin(plot_data)

        # Plot the local terrain background
        im = ax.imshow(
            plot_data, 
            origin='lower', 
            cmap='terrain',
            extent=[x_start, x_end, y_start, y_end]
        )
        plt.colorbar(im, ax=ax, label="Elevation / Depth [m]", shrink=0.8, pad=0.02)

        # 5. Overlay clear, high-contrast indicators on top of every single hole coordinate
        for y, x in void_coords:
            # Draw a bright red box outline around the single pixel hole
            rect = patches.Rectangle(
                (x - 0.5, y - 0.5), 1, 1, 
                linewidth=1.5, 
                edgecolor='red', 
                facecolor='none', 
                zorder=5
            )
            ax.add_patch(rect)
            
            # Add an outer glowing marker circle to guide the eye in case it's small
            ax.plot(x, y, marker='o', markersize=8, markeredgecolor='magenta', markerfacecolor='none', alpha=0.7, zorder=4)

        # Formatting
        ax.set_title(f"Critical Internal Voids Zoom View ({num_holes} Holes Detected)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Matrix Column Index (X)")
        ax.set_ylabel("Matrix Row Index (Y)")
        ax.grid(True, color='white', alpha=0.15, linestyle='--')

        # Strictly lock plot boundaries to our padded crop window
        ax.set_xlim(x_start, x_end)
        ax.set_ylim(y_start, y_end)

        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"[+] Hole map successfully saved to: {output_file}")

if __name__ == "__main__":
    # --- Execute Visualizer ---
    plot_dem_holes(
        file_path="Alajuela.h5", 
        dataset_name="bathymetry", 
        nodata_value=None, 
        output_file="alajuela_holes_location.png",
        padding=25 # Adjust padding if you want to zoom further in or out
    )