import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import yt

def count_nans_per_level(plotfile_path, variable_name="h_fluid", nodata_val=-9999):
    """
    Counts NaNs and NoData sentinel values for each AMR level in an AMReX plotfile.
    """
    print(f"[YT] Inspecting plotfile for NaNs: {plotfile_path}")
    ds = yt.load(plotfile_path)

    # Dictionary to aggregate stats per level
    level_stats = {}

    for grid in ds.index.grids:
        level = grid.Level
        if level not in level_stats:
            level_stats[level] = {
                "total_cells": 0,
                "nan_count": 0,
                "nodata_count": 0,
                "valid_count": 0
            }

        # Extract raw native grid data for this patch
        raw_data = grid[("boxlib", variable_name)].to_ndarray()

        # Total cells in this grid patch
        patch_total = raw_data.size

        # Detect IEEE NaNs
        patch_nans = np.count_nonzero(np.isnan(raw_data))

        # Detect Sentinel NoData values (e.g. -9999 or values < -9000)
        patch_nodata = np.count_nonzero(raw_data == nodata_val)

        # Combine NaNs and NoData
        patch_invalid = np.count_nonzero(np.isnan(raw_data) | (raw_data == nodata_val))
        patch_valid = patch_total - patch_invalid

        # Accumulate stats for this level
        level_stats[level]["total_cells"] += patch_total
        level_stats[level]["nan_count"] += patch_nans
        level_stats[level]["nodata_count"] += patch_nodata
        level_stats[level]["valid_count"] += patch_valid

    # Summary Report
    print("=" * 65)
    print(f"{'Level':<7} | {'Total Cells':<12} | {'NaN Count':<10} | {'-9999 Count':<12} | {'Valid Cells':<12}")
    print("-" * 65)

    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        print(
            f"Level {level:<1} | "
            f"{stats['total_cells']:<12} | "
            f"{stats['nan_count']:<10} | "
            f"{stats['nodata_count']:<12} | "
            f"{stats['valid_count']:<12}"
        )
    print("=" * 65)

    return level_stats

# Usage:
# stats = count_nans_per_level("plt00000", variable_name="h_fluid")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import yt

def plot_amrex_grid(
    plotfile_path,
    variable_name="h_fluid",
    output_file=None,
    cmap="terrain",
    figsize=(15, 6),
    dpi=300,
    show_grids=True,
    grid_colors=None,
    background_color="#1a1a1a",
    cbar_label="Value",
    title=None,
    nodata_val=-9999,
):
    if grid_colors is None:
        grid_colors = ["white", "yellow", "cyan", "magenta", "red", "lime"]

    if output_file is None:
        output_file = f"grid_viz_{variable_name}.png"

    if title is None:
        title = f"AMR Grid Topology ({variable_name})"

    print(f"[YT] Loading plotfile: {plotfile_path}")
    ds = yt.load(plotfile_path)

    # --- NEW: Count NaNs / NoData per Level ---
    level_stats = {}
    for grid in ds.index.grids:
        lev = grid.Level
        if lev not in level_stats:
            level_stats[lev] = {"total": 0, "nans": 0, "nodata": 0}

        arr = grid[("boxlib", variable_name)].to_ndarray()
        level_stats[lev]["total"] += arr.size
        level_stats[lev]["nans"] += np.count_nonzero(np.isnan(arr))
        level_stats[lev]["nodata"] += np.count_nonzero(arr == nodata_val)

    print("\n[DIAGNOSTIC] --- NaN / NoData Count Per Level ---")
    for lev in sorted(level_stats.keys()):
        tot = level_stats[lev]["total"]
        nans = level_stats[lev]["nans"]
        nodata = level_stats[lev]["nodata"]
        print(f"  Level {lev}: Total={tot}, NaNs={nans}, NoData({nodata_val})={nodata}, Valid={tot - nans - nodata}")
    print("--------------------------------------------------\n")

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
    data[data < 0.1] = np.nan

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

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Physical X [m]")
    ax.set_ylabel("Physical Y [m]")

    ax.set_aspect("equal")
    ax.set_xlim(left_edge[0], right_edge[0])
    ax.set_ylim(left_edge[1], right_edge[1])

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)

    print(f"[YT] Saved: {output_file}")


#plot_amrex_grid(
#    "plt00000",
#    variable_name="h_fluid",
#    cbar_label="Water Depth",
#    title="Water Depth with AMR Levels",
#    output_file="h_fluid.png",
#    figsize=(16, 15),
#    cmap="jet"
#)

plot_amrex_grid(
    "plt00000",
    variable_name="z_bathymetry",
    cbar_label="Bed Elevation",
    title="Bed Elevation with AMR Levels",
    output_file="z_bath.png",
    figsize=(16, 15),
    cmap="terrain"
)

