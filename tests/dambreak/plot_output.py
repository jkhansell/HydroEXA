#!/usr/bin/env python3
"""
Visualize HydroEXA rectangular dam break plotfiles.

Produces:
  1. 2D colormap of water depth h (planform view) with MeshBlock outlines
  2. 1D line cut along y=center showing h, hu, hv profiles
  3. Time-series GIF of h evolution

The 1D line cut is the key diagnostic: for a perfect 1D problem,
all y-cuts should overlap. Any spread = numerical error / flux bug.

Usage:
  uv run python plot_output.py              # process latest plotfiles in current dir
  uv run python plot_output.py plotfiles/   # process a specific directory
  uv run python plot_output.py plt00010     # process a single plotfile
"""

import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

# ------------------------------------------------------------------
try:
    import yt
    HAS_YT = True
except ImportError:
    HAS_YT = False

# ------------------------------------------------------------------
# Variable naming: HydroEXA writes these via AMReX Plotfile
# Common conventions — try multiple names
VARIABLE_ALIASES = {
    "h":      ["h_fluid", "h", "water_depth", "Depth"],
    "hu":     ["hu", "x_momentum", "MomentumX"],
    "hv":     ["hv", "y_momentum", "MomentumY"],
    "z":      ["z_bathymetry", "z", "bathymetry", "BedElevation", "elev"],
}


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def find_variable(ds, var_key):
    """Find the actual dataset variable name for a logical key."""
    for alias in VARIABLE_ALIASES[var_key]:
        full = ("boxlib", alias)
        if full in ds.field_list:
            return full
        # Also check without boxlib prefix
        if alias in ds.field_list:
            return ("boxlib", alias)
    # Fallback: search all fields
    for f in ds.field_list():
        fname = f[1] if len(f) > 1 else str(f)
        if any(a in fname for a in VARIABLE_ALIASES[var_key]):
            return f
    return None


def get_meshblock_rects(ds):
    """
    Return a list of dicts with rectangle info for each MeshBlock,
    including its AMR refinement level.
    """
    lefts = ds.index.grid_left_edge   # (n_grids, 3) world coords (unyt_array)
    dims  = ds.index.grid_dimensions  # (n_grids, 3) cell counts
    levels = ds.index.grid_levels.flatten()  # (n_grids,) refinement level per grid
    # Get cell sizes per level
    dx_table = ds.index.level_dds  # (max_level+1, 3)

    rects = []
    for i in range(len(lefts)):
        x0 = float(lefts[i, 0])
        y0 = float(lefts[i, 1])
        lvl = int(levels[i])
        dx = dx_table[lvl]
        w  = float(dims[i, 0] * dx[0])
        h  = float(dims[i, 1] * dx[1])
        rects.append({"x0": x0, "y0": y0, "w": w, "h": h, "level": lvl})
    return rects


def extract_line_cut(ds, var_key, y_center=None, ny_lines=7):
    """
    Extract water depth profiles at multiple y-positions.
    For a perfect 1D problem, all profiles should overlap.
    """
    if var_key is None:
        return None, None

    data_key = find_variable(ds, var_key)
    if data_key is None:
        return None, None

    left_edge = np.asarray(ds.domain_left_edge.to_ndarray(), dtype=np.float64)
    right_edge = np.asarray(ds.domain_right_edge.to_ndarray(), dtype=np.float64)
    dims = ds.domain_dimensions

    if y_center is None:
        y_center = (left_edge[1] + right_edge[1]) / 2.0

    covering = ds.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data = np.asarray(covering[data_key].to_ndarray(), dtype=np.float64)
    data = np.squeeze(data).T  # (ny, nx)

    x = np.linspace(left_edge[0], right_edge[0], dims[0])
    y = np.linspace(left_edge[1], right_edge[1], dims[1])

    # Extract profiles at ny_lines evenly-spaced y positions
    profiles = []
    y_positions = np.linspace(left_edge[1], right_edge[1], ny_lines)

    for y_pos in y_positions:
        # Find nearest row index in the full y grid
        row_idx = int((y_pos - y[0]) / (y[-1] - y[0]) * (len(y) - 1))
        row_idx = np.clip(row_idx, 0, len(y) - 1)
        profiles.append((y_pos, data[row_idx, :]))

    return x, profiles


def plot_single_frame(plotfile_path, output_prefix=""):
    """Plot a single time step: 2D field (top) + 1D line cuts (bottom)."""
    if not HAS_YT:
        print(f"  [WARN] yt not available for {plotfile_path}; skipping 2D plot.")
        return

    ds = yt.load(plotfile_path)
    sim_time = float(ds.current_time)

    left_edge = np.asarray(ds.domain_left_edge.to_ndarray(), dtype=np.float64)
    right_edge = np.asarray(ds.domain_right_edge.to_ndarray(), dtype=np.float64)

    # --- 2D field ---
    h_key = find_variable(ds, "h")
    if h_key is None:
        print(f"  [WARN] No water depth variable found in {plotfile_path}")
        return

    covering = ds.covering_grid(level=0, left_edge=left_edge,
                                dims=ds.domain_dimensions)
    h_data = np.asarray(covering[h_key].to_ndarray(), dtype=np.float64)
    h_data = np.squeeze(h_data).T

    # MeshBlock rectangles with level info
    rects = get_meshblock_rects(ds)
    max_level = int(ds.index.max_level)

    # Color map for levels: coarser = darker, finer = brighter
    level_colors = plt.cm.Reds_r(np.linspace(0.3, 1.0, max_level + 1))
    level_linewidths = {lvl: 1.0 + 1.5 * (max_level - lvl) for lvl in range(max_level + 1)}

    # Layout: two rows (2D on top, line cuts below)
    fig = plt.figure(figsize=(14, 10), dpi=150)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2], hspace=0.35)

    # ---- Top: 2D colormap with MeshBlock/level outlines ----
    ax1 = fig.add_subplot(gs[0])
    im = ax1.pcolormesh(
        np.linspace(left_edge[0], right_edge[0], h_data.shape[1]),
        np.linspace(left_edge[1], right_edge[1], h_data.shape[0]),
        h_data,
        cmap="viridis", shading="auto"
    )
    ax1.set_aspect("equal")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_title(f"h (water depth) | t = {sim_time:.4f} s")

    # Draw MeshBlock boundaries colored by refinement level
    for i, r in enumerate(rects):
        lvl = r["level"]
        rect = mpatches.Rectangle(
            (r["x0"], r["y0"]), r["w"], r["h"],
            fill=False, edgecolor=level_colors[lvl],
            linewidth=level_linewidths[lvl],
            linestyle="-", alpha=0.8
        )
        ax1.add_patch(rect)

    # Level legend
    legend_handles = []
    for lvl in range(max_level + 1):
        legend_handles.append(
            mpatches.Patch(color=level_colors[lvl], label=f"Level {lvl}",
                           linewidth=level_linewidths[lvl], fill=False)
        )
    ax1.legend(handles=legend_handles, loc="upper right", fontsize=8,
               framealpha=0.85, title="Refinement Level")

    # Horizontal colorbar
    cbar = plt.colorbar(im, ax=ax1, orientation="horizontal",
                        fraction=0.03, pad=0.08, label="h [m]")

    # ---- Bottom: 1D line cuts with legend ----
    ax2 = fig.add_subplot(gs[1])
    x, profiles = extract_line_cut(ds, "h", ny_lines=7)
    if profiles is not None:
        domain_y_center = (left_edge[1] + right_edge[1]) / 2.0
        domain_y_span   = (right_edge[1] - left_edge[1]) / 2.0

        for y_pos, profile in profiles:
            # Alpha fades away from center
            alpha = 0.3 + 0.7 * abs(y_pos - domain_y_center) / domain_y_span
            label = f"y = {y_pos:.1f} m"
            ax2.plot(x, profile, alpha=alpha, linewidth=1.0, label=label)

        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("h [m]")
        ax2.set_title("Line cuts at multiple y-positions — should all overlap")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", fontsize=7, framealpha=0.85)

    # Use manual adjustment instead of tight_layout (incompatible with horizontal colorbar)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.15)
    outname = f"{output_prefix}frame_{sim_time:.4f}.png" if output_prefix else f"frame_{sim_time:.4f}.png"
    plt.savefig(outname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outname}")
    return outname


def generate_gif(plotfiles, variable_name="h", output_gif="h_evolution.gif",
                 duration=0.2, output_prefix=""):
    """Generate a time-series GIF from plotfiles."""
    if not HAS_YT:
        print("[ERROR] yt is required for GIF generation. Install with: pip install yt")
        return

    import imageio.v2 as imageio

    print(f"\n[INFO] Generating GIF from {len(plotfiles)} plotfiles...")
    frame_files = []

    for i, pltfile in enumerate(plotfiles):
        print(f"  [{i+1}/{len(plotfiles)}] {os.path.basename(pltfile)}")
        frame_name = f"{output_prefix}frame_{i:04d}.png"
        saved = plot_single_frame(pltfile, output_prefix=frame_name)
        if saved:
            frame_files.append(saved)

    images = [imageio.imread(f) for f in frame_files if os.path.exists(f)]
    if images:
        imageio.mimsave(output_gif, images, duration=duration, loop=0)
        print(f"[SUCCESS] Saved: {output_gif}")
        # Cleanup
        for f in frame_files:
            if os.path.exists(f):
                os.remove(f)
    else:
        print("[ERROR] No frames generated.")


# ------------------------------------------------------------------
if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "plt*"

    if os.path.isdir(pattern) and not pattern.startswith("plt"):
        search_pattern = os.path.join(pattern, "plt*")
    else:
        search_pattern = pattern

    plotfiles = [p for p in glob.glob(search_pattern)
                 if os.path.isdir(p) or "Header" in os.listdir(p)]
    plotfiles = sorted(plotfiles, key=natural_sort_key)

    if not plotfiles:
        print(f"[ERROR] No plotfiles matching '{search_pattern}'.")
        sys.exit(1)

    print(f"[INFO] Found {len(plotfiles)} plotfiles.\n")

    # Plot each frame individually, collect saved filenames for GIF
    frame_files = []
    for pf in plotfiles:
        saved = plot_single_frame(pf)
        if saved:
            frame_files.append(saved)

    # Generate GIF from the already-saved frames
    if frame_files and HAS_YT:
        import imageio.v2 as imageio
        images = [imageio.imread(f) for f in frame_files if os.path.exists(f)]
        if images:
            imageio.mimsave("h_evolution.gif", images, duration=0.2, loop=0)
            print(f"[SUCCESS] Saved: h_evolution.gif")
            for f in frame_files:
                if os.path.exists(f):
                    os.remove(f)

    print("\n[DONE] All frames and GIF generated.")
    print("Tip: For the rectangular dam break, check the line-cut plot.")
    print("     If all y-cuts overlap perfectly, your fluxes are 1D-correct.")
    print("     Spread in the line cuts = numerical diffusion / flux asymmetry.")
