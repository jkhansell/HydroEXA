import glob
import os
import re
import sys
import imageio.v2 as imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import yt

def natural_sort_key(s):
    """Sorts plotfiles in numerical order (e.g. plt00010 before plt00100)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def process_single_frame(
    plotfile_path,
    variable_name="h_fluid",
    output_file="frame.png",
    cmap="jet",
    figsize=(12, 10),
    dpi=150,
    show_grids=True,
    grid_colors=None,
    background_color="#1a1a1a",
    cbar_label="Value",
    nodata_val=-9999,
    logplot=False
):
    if grid_colors is None:
        grid_colors = ["white", "yellow", "cyan", "magenta", "red", "lime"]

    ds = yt.load(plotfile_path)

    # Extract simulation time if available
    sim_time = float(ds.current_time) if hasattr(ds, "current_time") else 0.0

    left_edge = ds.domain_left_edge.to_ndarray()
    right_edge = ds.domain_right_edge.to_ndarray()
    dims = ds.domain_dimensions

    # Uniform covering grid for background visualization
    covering_grid = ds.covering_grid(
        level=0,
        left_edge=left_edge,
        dims=dims,
    )

    data = covering_grid[("boxlib", variable_name)].to_ndarray()
    data = np.squeeze(data).T

    x = np.linspace(left_edge[0], right_edge[0], dims[0])
    y = np.linspace(left_edge[1], right_edge[1], dims[1])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background_color)

    mesh = ax.pcolormesh(
        x,
        y,
        data,
        cmap=cmap,
        shading="auto",
        norm="log" if logplot else None
    )

    if show_grids:
        for grid in ds.index.grids:
            level = grid.Level
            left = grid.LeftEdge
            right = grid.RightEdge

            rect = patches.Rectangle(
                (left[0], left[1]),
                right[0] - left[0],
                right[1] - left[1],
                linewidth=0.8,
                edgecolor=grid_colors[level % len(grid_colors)],
                facecolor="none",
                alpha=0.8,
            )
            ax.add_patch(rect)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.7)
    cbar.set_label(cbar_label)

    # Clean plotfile folder name for title
    filename_clean = os.path.basename(os.path.normpath(plotfile_path))
    full_title = f"{filename_clean} | Variable: {variable_name} | Time: {sim_time:.4f} s"

    ax.set_title(full_title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Physical X [m]")
    ax.set_ylabel("Physical Y [m]")

    ax.set_aspect("equal")
    ax.set_xlim(left_edge[0], right_edge[0])
    ax.set_ylim(left_edge[1], right_edge[1])

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)

from PIL import Image

def generate_time_series_gif(
    plotfiles,
    variable_name="h_fluid",
    cbar_label="Water Depth",
    cmap="jet",
    output_gif="evolution.gif",
    duration=0.25,
    logplot=False
):
    print(f"\n[INFO] Processing {len(plotfiles)} plotfiles for '{variable_name}'...")
    
    frame_files = []
    temp_dir = f"_temp_frames_{variable_name}"
    os.makedirs(temp_dir, exist_ok=True)

    for i, pltfile in enumerate(plotfiles):
        frame_name = os.path.join(temp_dir, f"frame_{i:04d}.png")
        print(f"  --> [{i+1}/{len(plotfiles)}] Rendering: {pltfile}")
        
        try:
            process_single_frame(
                plotfile_path=pltfile,
                variable_name=variable_name,
                output_file=frame_name,
                cbar_label=cbar_label,
                cmap=cmap,
                logplot=logplot
            )
            frame_files.append(frame_name)
        except Exception as e:
            print(f"  [ERROR] Failed to process {pltfile}: {e}")

    if not frame_files:
        print(f"[ERROR] No valid frames created for {variable_name}.")
        return

    print(f"[INFO] Compiling GIF: {output_gif}")
    
    # --- FIX: Load frames and force all to match the exact dimensions of Frame 0 ---
    pil_images = [Image.open(f) for f in frame_files]
    target_size = pil_images[0].size  # (width, height)

    resized_np_images = [
        np.array(img.resize(target_size, Image.Resampling.LANCZOS)) if img.size != target_size else np.array(img)
        for img in pil_images
    ]

    # Save animation with uniform shapes
    imageio.mimsave(output_gif, resized_np_images, duration=duration, loop=0)
    print(f"[SUCCESS] Saved GIF: {output_gif}")

    # Cleanup temporary PNG frames and close PIL pointers
    for img in pil_images:
        img.close()
        
    for f in frame_files:
        os.remove(f)
    os.rmdir(temp_dir)


if __name__ == "__main__":
    # Accept a directory path or pattern as CLI argument, default to current dir searching for "plt*"
    pattern = sys.argv[1] if len(sys.argv) > 1 else "plt*"

    if os.path.isdir(pattern) and not pattern.startswith("plt"):
        search_pattern = os.path.join(pattern, "plt*")
    else:
        search_pattern = pattern

    # Find all plotfiles and sort numerically
    plotfiles = [p for p in glob.glob(search_pattern) if os.path.isdir(p) or "Header" in os.listdir(p)]
    plotfiles = sorted(plotfiles, key=natural_sort_key)

    if not plotfiles:
        print(f"[ERROR] No plotfiles matching '{search_pattern}' were found.")
        sys.exit(1)

    print(f"[INFO] Found {len(plotfiles)} plotfile directories.")

    # 1. Generate GIF for Water Depth (h_fluid)
    generate_time_series_gif(
        plotfiles=plotfiles,
        variable_name="h_fluid",
        cbar_label="Water Depth [m]",
        cmap="jet",
        output_gif="h_fluid_evolution.gif",
        duration=0.25,
        logplot=False
    )

    # 2. Generate GIF for Bed Elevation (z_bathymetry)
    # generate_time_series_gif(
    #     plotfiles=plotfiles,
    #     variable_name="z_bathymetry",
    #     cbar_label="Bed Elevation [m]",
    #     cmap="terrain",
    #     output_gif="z_bathymetry_evolution.gif",
    #     duration=0.25,
    #     logplot=False
    # )