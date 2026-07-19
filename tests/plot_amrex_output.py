import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
):
    """
    Visualize an AMReX plotfile field together with its AMR grid hierarchy.

    Parameters
    ----------
    plotfile_path : str
        Path to the AMReX plotfile.
    variable_name : str
        Variable to visualize.
    output_file : str, optional
        Output image filename. Defaults to
        'grid_viz_<variable_name>.png'.
    cmap : str
        Matplotlib colormap.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.
    show_grids : bool
        Whether to overlay AMR grid boundaries.
    grid_colors : list, optional
        Colors used for each AMR level.
    background_color : str
        Axes background color.
    cbar_label : str
        Colorbar label.
    title : str, optional
        Plot title.
    """

    if grid_colors is None:
        grid_colors = ["white", "yellow", "cyan", "magenta", "red", "lime"]

    if output_file is None:
        output_file = f"grid_viz_{variable_name}.png"

    if title is None:
        title = f"AMR Grid Topology ({variable_name})"

    print(f"[YT] Loading plotfile: {plotfile_path}")

    ds = yt.load(plotfile_path)

    left_edge = ds.domain_left_edge.to_ndarray()
    right_edge = ds.domain_right_edge.to_ndarray()
    dims = ds.domain_dimensions

    # Uniform covering grid
    covering_grid = ds.covering_grid(
        level=0,
        left_edge=left_edge,
        dims=dims,
    )

    data = covering_grid[("boxlib", variable_name)].to_ndarray()
    data = np.squeeze(data).T
    data[np.isnan(data)] = np.nan

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

    # Draw AMR grid boundaries
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



def plot_amrex_gradient(plotfile_path, variable_name="z_bathymetry", output_file="gradient_check_manual.png"):
    ds = yt.load(plotfile_path)
    
    # 1. Get the data as a 2D array
    # Using a covering grid for a consistent array
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    arr = cg[("boxlib", variable_name)].to_ndarray().squeeze()
    
    # 2. Compute finite difference gradient
    # dy is rows, dx is columns
    gy, gx = np.gradient(arr)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # 3. Plot manually
    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(grad_mag.T, origin='lower', cmap='magma', extent=[ds.domain_left_edge[0], ds.domain_right_edge[0], ds.domain_left_edge[1], ds.domain_right_edge[1]])
    plt.colorbar(im, label="Gradient Magnitude",pad=0.01, shrink=0.7)
    
    # Add grids
    #for grid in ds.index.grids:
    #    left = grid.LeftEdge
    #    right = grid.RightEdge
    #    rect = patches.Rectangle((left[0], left[1]), right[0]-left[0], right[1]-left[1], 
    #                             linewidth=0.5, edgecolor='white', facecolor='none')
    #    ax.add_patch(rect)
        
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"[MANUAL] Saved: {output_file}")


def plot_amr_resolution_hierarchy(
    plotfile_path,
    variable_name="h_fluid",
    output_file=None,
    cmap="viridis",
    figsize=(15, 6),
    dpi=150,
    alpha_base=0.40,
    alpha_step=0.15,
    show_grids=True,
    grid_colors=None,
    background_color="#1a1a1a",
    cbar_label=None,
    title=None,
):
    """Overlay every native AMReX AMR grid patch at its exact resolution and location."""
    if grid_colors is None:
        grid_colors = [
            "white",
            "yellow",
            "cyan",
            "magenta",
            "red",
            "lime",
            "orange",
        ]

    ds = yt.load(plotfile_path)
    print(ds.field_list)


    if output_file is None:
        output_file = f"amr_levels_{variable_name}.png"

    if title is None:
        title = f"AMR Resolution Hierarchy ({variable_name})"

    if cbar_label is None:
        cbar_label = variable_name

    left = ds.domain_left_edge.to_ndarray()
    right = ds.domain_right_edge.to_ndarray()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background_color)

    # Track min/max globally across patches to normalize the colorbar properly
    vmin, vmax = float("inf"), float("-inf")
    grid_data_list = []

    # --- Phase 1: Read data from native individual grids ---
    print(f"Processing {len(ds.index.grids)} total grids...")
    for grid in ds.index.grids:
        level = grid.Level

        # Extract data for this individual patch
        raw_data = grid[("boxlib", variable_name)].to_ndarray()

        # Handle 2D data embedded in a 3D YTArray (Z-axis dimension is 1)
        if raw_data.ndim == 3:
            if raw_data.shape[2] == 1:
                raw_data = raw_data[:, :, 0]
            elif raw_data.shape[0] == 1:
                raw_data = raw_data[0, :, :]

        # Ensure we have a clean 2D array
        data = np.squeeze(raw_data)
        if data.ndim != 2:
            continue

        # Transpose to align with matplotlib's (Y, X) pcolormesh mapping
        data = data.T
        vmin = min(vmin, np.nanmin(data))
        vmax = max(vmax, np.nanmax(data))

        # Store boundaries and data for plotting
        g_left = grid.LeftEdge.to_ndarray()
        g_right = grid.RightEdge.to_ndarray()
        grid_data_list.append((level, g_left, g_right, data))

    # --- Phase 2: Plot patches sequentially by level ---
    # Sorting ensures finer levels are painted over coarser levels cleanly
    grid_data_list.sort(key=lambda x: x[0])

    mesh = None
    for level, g_left, g_right, data in grid_data_list:
        ny, nx = data.shape

        # Generate cell edge coordinates specific to this patch
        x_edges = np.linspace(g_left[0], g_right[0], nx + 1)
        y_edges = np.linspace(g_left[1], g_right[1], ny + 1)

        # Calculate alpha step per level to visualize depth/hierarchy
        level_alpha = min(alpha_base + alpha_step * level, 1.0)

        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            data,
            cmap=cmap,
            shading="flat",
            alpha=level_alpha,
            vmin=vmin,
            vmax=vmax,
        )

        # --- Phase 3: Draw patch outlines if requested ---
        if show_grids:
            rect = patches.Rectangle(
                (g_left[0], g_left[1]),
                g_right[0] - g_left[0],
                g_right[1] - g_left[1],
                edgecolor=grid_colors[level % len(grid_colors)],
                facecolor="none",
                linewidth=0.8,
                alpha=0.8,
            )
            ax.add_patch(rect)

    # --- Formatting & Legend ---
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.7)
        cbar.set_label(cbar_label)

    ax.set_xlim(left[0], right[0])
    ax.set_ylim(left[1], right[1])
    ax.set_aspect("equal")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)

    # Create a clean, unique legend matching your resolution levels
    unique_levels = sorted(list(set([g[0] for g in grid_data_list])))
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=grid_colors[lvl % len(grid_colors)],
            lw=2,
            label=f"Level {lvl}",
        )
        for lvl in unique_levels
    ]
    ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)
    plt.close()

    print(f"Successfully saved hierarchy visualization to: {output_file}")


plot_amrex_grid(
    "plt00000",
    variable_name="h_fluid",
    cbar_label="Bed Elevation [m]",
    title="Bed Elevation with AMR Levels",
    output_file="h_fluid.png",
    figsize=(16, 15),
    cmap="jet"
)

