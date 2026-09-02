import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_amrex_levels_horizontal(
    plotfile_path,
    variable_name="h_fluid",
    output_file=None,
    cmap="jet",
    figsize_per_level=(6, 5),  # Width, Height per panel
    dpi=150,
    show_grids=True,
    grid_colors=None,
    background_color="#1a1a1a",
    cbar_label="Value",
    title_prefix="AMR Level Breakdown",
):
    """
    Load an AMReX plotfile and plot each refinement level side-by-side (horizontally).
    Aligns spatial coordinates and uses a unified colorbar scale for perfect physical comparison.
    """
    if grid_colors is None:
        grid_colors = ["white", "yellow", "cyan", "magenta", "red", "lime", "orange"]

    if output_file is None:
        output_file = f"level_breakdown_horizontal_{variable_name}.png"

    print(f"[YT] Loading plotfile: {plotfile_path}")
    ds = yt.load(plotfile_path)
    
    # 1. Determine active levels
    max_level = ds.index.max_level
    num_levels = max_level + 1
    print(f"[YT] Found {num_levels} active AMR levels (Level 0 to {max_level})")

    left_edge = ds.domain_left_edge.to_ndarray()
    right_edge = ds.domain_right_edge.to_ndarray()

    # 2. Extract and organize data by level to find global vmin/vmax
    level_patches = {lvl: [] for lvl in range(num_levels)}
    vmin, vmax = float("inf"), float("-inf")

    for grid in ds.index.grids:
        lvl = grid.Level
        raw_data = grid[("boxlib", variable_name)].to_ndarray()

        # Handle 2D data embedded in a 3D YTArray
        if raw_data.ndim == 3:
            if raw_data.shape[2] == 1:
                raw_data = raw_data[:, :, 0]
            elif raw_data.shape[0] == 1:
                raw_data = raw_data[0, :, :]

        data = np.squeeze(raw_data)
        if data.ndim != 2:
            continue

        # Align with Matplotlib's (Y, X) coordinate mapping
        data = data.T
        
        # Calculate local limits ignoring NaNs
        if data.size > 0:
            vmin = min(vmin, np.nanmin(data))
            vmax = max(vmax, np.nanmax(data))

        g_left = grid.LeftEdge.to_ndarray()
        g_right = grid.RightEdge.to_ndarray()
        level_patches[lvl].append((g_left, g_right, data))

    # 3. Initialize Figure and Subplots (Side-by-Side Horizontally)
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=num_levels, 
        figsize=(figsize_per_level[0] * num_levels, figsize_per_level[1]), 
        dpi=dpi,
        squeeze=False # Ensures axes is always a 2D array even if num_levels == 1
    )
    
    mesh = None

    # 4. Populate each subplot
    for lvl in range(num_levels):
        ax = axes[0, lvl]
        ax.set_facecolor(background_color)
        
        # Draw all patches belonging to this specific level
        for g_left, g_right, data in level_patches[lvl]:
            ny, nx = data.shape
            x_edges = np.linspace(g_left[0], g_right[0], nx + 1)
            y_edges = np.linspace(g_left[1], g_right[1], ny + 1)

            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                data,
                cmap=cmap,
                shading="flat",
                vmin=vmin,
                vmax=vmax,
            )

            # Draw patch outline for this level
            if show_grids:
                rect = patches.Rectangle(
                    (g_left[0], g_left[1]),
                    g_right[0] - g_left[0],
                    g_right[1] - g_left[1],
                    edgecolor=grid_colors[lvl % len(grid_colors)],
                    facecolor="none",
                    linewidth=0.02,
                    alpha=0.9,
                )
                ax.add_patch(rect)

        # Formatting each panel
        ax.set_xlim(left_edge[0], right_edge[0])
        ax.set_ylim(left_edge[1], right_edge[1])
        ax.set_aspect("equal")
        
        ax.set_xlabel("Physical X [m]", fontsize=10)
        ax.set_title(f"Level {lvl} Refinement", fontsize=12, fontweight="bold", loc="center")
        
        # Only put the Y label on the leftmost subplot to keep the horizontal layout clean
        if lvl == 0:
            ax.set_ylabel("Physical Y [m]", fontsize=10)
        else:
            ax.set_yticklabels([])

    # 5. Add a unified horizontal colorbar at the bottom
    if mesh is not None:
        # Leaves some room at the bottom for the horizontal colorbar
        fig.subplots_adjust(bottom=0.25)
        # Coordinate layout: [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.04])  
        cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(cbar_label, fontsize=11, fontweight="bold")

    fig.suptitle(f"{title_prefix} - {variable_name}", fontsize=14, fontweight="bold", y=0.95)
    
    # Save Layout
    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    print(f"[YT] Completed! Horizontal multi-level plot saved to: {output_file}")


plot_amrex_levels_horizontal(
    plotfile_path="plt00000",
    variable_name="z_bathymetry",
    cbar_label="Water Depth / Elevation [m]",
    output_file="z_levels.png",
    cmap="terrain",
    figsize_per_level=(5, 10),
)