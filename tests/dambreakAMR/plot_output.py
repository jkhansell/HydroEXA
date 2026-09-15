#!/usr/bin/env python3

"""
Visualize HydroEXA rectangular dam-break plotfiles with analytical solution.

Produces:
  1. 2D water-depth colormap with MeshBlock outlines
  2. 1D h line cuts at multiple y positions + analytical solution
  3. Time-series GIF of h evolution

For a perfect 1D problem, all y-cuts should overlap.
Spread between y-cuts indicates numerical error or flux asymmetry.

Usage:
  uv run python plot_output.py
  uv run python plot_output.py plotfiles/
  uv run python plot_output.py plt00010
  uv run python plot_output.py --hl 0.5 --hr 0.1 --x0 25.6 --L 51.2
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Patch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytical import dambreak_on_wet_no_friction_analytical

try:
    import yt
    HAS_YT = True
except ImportError:
    HAS_YT = False

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


OUTPUT_DIR = "outputs"
FIGSIZE = (14, 12)
DPI = 150
NY_LINES = 7

VARIABLE_ALIASES = {
    "h": ["h_fluid", "h", "water_depth", "Depth"],
    "hu": ["hu", "hu_momentum", "MomentumX"],
    "hv": ["hv", "hu_momentum", "MomentumY"],
    "z": ["z_bathymetry", "z", "bathymetry", "BedElevation", "elev"],
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def find_variable(ds, var_key):
    for alias in VARIABLE_ALIASES[var_key]:
        field = ("boxlib", alias)
        if field in ds.field_list:
            return field
        if alias in ds.field_list:
            return ("boxlib", alias)

    for field in ds.field_list:
        name = field[1] if len(field) > 1 else str(field)
        if any(alias in name for alias in VARIABLE_ALIASES[var_key]):
            return field

    return None


def get_domain_bounds(ds):
    left_edge = np.asarray(ds.domain_left_edge.to_ndarray(), dtype=np.float64)
    right_edge = np.asarray(ds.domain_right_edge.to_ndarray(), dtype=np.float64)
    return left_edge, right_edge


def get_meshblock_rects(ds):
    lefts = ds.index.grid_left_edge
    dims = ds.index.grid_dimensions
    levels = ds.index.grid_levels.flatten()
    dx_table = ds.index.level_dds

    rects = []
    for i in range(len(lefts)):
        level = int(levels[i])
        dx = dx_table[level]
        rects.append({
            "x0": float(lefts[i, 0]),
            "y0": float(lefts[i, 1]),
            "w": float(dims[i, 0] * dx[0]),
            "h": float(dims[i, 1] * dx[1]),
            "level": level,
        })

    return rects


def extract_line_cut(ds, var_key, y_center=None, ny_lines=NY_LINES):
    field = find_variable(ds, var_key)
    if field is None:
        return None, None

    left_edge, right_edge = get_domain_bounds(ds)
    dims = np.asarray(ds.domain_dimensions, dtype=int)

    covering = ds.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data = np.asarray(covering[field].to_ndarray(), dtype=np.float64).squeeze().T

    if data.ndim != 2:
        raise RuntimeError(f"Expected 2D data for '{var_key}', got shape {data.shape}")

    ny, nx = data.shape
    dx = (right_edge[0] - left_edge[0]) / nx
    dy = (right_edge[1] - left_edge[1]) / ny

    x = left_edge[0] + (np.arange(nx) + 0.5) * dx
    y = left_edge[1] + (np.arange(ny) + 0.5) * dy

    y_positions = np.linspace(left_edge[1], right_edge[1], ny_lines)
    profiles = []

    for y_pos in y_positions:
        row_idx = int(np.argmin(np.abs(y - y_pos)))
        profiles.append((float(y[row_idx]), data[row_idx, :]))

    return x, profiles


def plot_single_frame(plotfile_path, output_prefix="", hl=0.005, hr=0.001, x0=5.0, L=10.0):
    if not HAS_YT:
        raise RuntimeError("yt is required to generate plots.")

    if hl <= hr:
        raise ValueError(f"Expected hl > hr, but received hl={hl}, hr={hr}")

    print(f"Processing: {plotfile_path}")

    ds = yt.load(plotfile_path)
    sim_time = float(ds.current_time)
    left_edge, right_edge = get_domain_bounds(ds)
    dims = np.asarray(ds.domain_dimensions, dtype=int)

    h_field = find_variable(ds, "h")
    if h_field is None:
        raise RuntimeError("Could not find water-depth variable.")

    covering = ds.covering_grid(level=0, left_edge=left_edge, dims=dims)
    h = np.asarray(covering[h_field].to_ndarray(), dtype=np.float64).squeeze().T

    if h.ndim != 2:
        raise RuntimeError(f"Expected 2D water-depth data, got shape {h.shape}")

    norm = Normalize(vmin=hr, vmax=hl, clip=True)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 4.0, 4.0], hspace=0.40)

    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(
        h,
        origin="lower",
        extent=[left_edge[0], right_edge[0], left_edge[1], right_edge[1]],
        cmap="viridis",
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    ax1.set_xlim(left_edge[0], right_edge[0])
    ax1.set_ylim(left_edge[1], right_edge[1])
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title(f"Water depth at t = {sim_time:.4f} s")

    rects = get_meshblock_rects(ds)
    levels = sorted(set(rect["level"] for rect in rects))
    level_colors = plt.cm.tab20(np.linspace(0, 1, max(len(levels), 1)))
    level_color_map = dict(zip(levels, level_colors))

    for rect in rects:
        ax1.add_patch(Rectangle(
            (rect["x0"], rect["y0"]),
            rect["w"],
            rect["h"],
            fill=False,
            edgecolor=level_color_map[rect["level"]],
            linewidth=1.2,
            alpha=0.9,
            zorder=10,
        ))

    level_handles = [
        Patch(
            facecolor="none",
            edgecolor=level_color_map[level],
            linewidth=2.0,
            label=f"Level {level}",
        )
        for level in levels
    ]

    ax1.legend(
        handles=level_handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        title="AMR Level",
        fontsize=9,
        title_fontsize=10,
        ncol=len(level_handles),
    )

    cbar = fig.colorbar(
        im,
        ax=ax1,
        orientation="horizontal",
        fraction=0.045,
        pad=0.25,
    )
    cbar.set_label("h [m]")
    cbar.set_ticks(np.linspace(hr, hl, 5))

    ax2 = fig.add_subplot(gs[1])

    x_ana = np.linspace(left_edge[0], right_edge[0], 1000)
    
    h_ana, u_ana = dambreak_on_wet_no_friction_analytical(
        sim_time,
        x_ana,
        L=L,
        hl=hl,
        hr=hr,
        x0=x0,
    )

    ax2.plot(
        x_ana,
        h_ana,
        linewidth=2.5,
        linestyle="--",
        label="Analytical",
    )

    x_cut, profiles = extract_line_cut(
        ds,
        "h",
        y_center=0.5 * (left_edge[1] + right_edge[1]),
        ny_lines=NY_LINES,
    )

    if x_cut is not None:
        for y_pos, values in profiles:
            ax2.plot(
                x_cut,
                values,
                linewidth=1.5,
                alpha=0.75,
                label=f"y = {y_pos:.2f} m",
            )

    ax2.set_xlim(left_edge[0], right_edge[0])
    ax2.set_ylim(hr-0.001, hl+0.001)
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("h [m]")
    ax2.set_title("Water-depth line cuts")
    ax2.legend(loc="best", ncol=2, fontsize=9)
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[2])

    ax3.plot(
        x_ana,
        h_ana*u_ana,
        linewidth=2.5,
        linestyle="--",
        label="Analytical Mom-X",
    )

    x_cut, profiles = extract_line_cut(
        ds,
        "hu",
        y_center=0.5 * (left_edge[1] + right_edge[1]),
        ny_lines=NY_LINES,
    )

    if x_cut is not None:
        for y_pos, values in profiles:
            ax3.plot(
                x_cut,
                values,
                linewidth=1.5,
                alpha=0.75,
                label=f"y = {y_pos:.2f} m",
            )

    ax3.set_xlim(left_edge[0], right_edge[0])
    ax3.set_ylim(-0.05, 0.6)
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("h [m]")
    ax3.set_title("Water-depth line cuts")
    ax3.legend(loc="best", ncol=2, fontsize=9)
    ax3.grid(True, alpha=0.25)

    fig.subplots_adjust(
        left=0.08,
        right=0.96,
        top=0.94,
        bottom=0.09,
        hspace=0.40,
    )

    outname = os.path.join(OUTPUT_DIR, f"frame_{sim_time:.4f}.png")
    fig.savefig(outname, dpi=DPI, facecolor="white")
    plt.close(fig)

    print(f"Saved: {outname} ({int(FIGSIZE[0] * DPI)}x{int(FIGSIZE[1] * DPI)} px)")
    return outname


def generate_gif(frame_files, output_gif=None, duration=0.2):
    if imageio is None:
        print("[ERROR] imageio is required for GIF generation.")
        return

    if output_gif is None:
        output_gif = os.path.join(OUTPUT_DIR, "h_evolution.gif")

    images = [imageio.imread(f) for f in frame_files if os.path.exists(f)]

    if not images:
        print("[ERROR] No frames available for GIF.")
        return

    imageio.mimsave(output_gif, images, duration=duration, loop=0)
    print(f"[SUCCESS] Saved: {output_gif}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot HydroEXA dam-break plotfiles with analytical solution."
    )
    parser.add_argument("pattern", nargs="?", default="plt*", help="Glob pattern or directory containing plotfiles")
    parser.add_argument("--hl", type=float, default=0.005, help="Upstream water depth")
    parser.add_argument("--hr", type=float, default=0.001, help="Downstream water depth")
    parser.add_argument("--x0", type=float, default=5.0, help="Initial dam location")
    parser.add_argument("--L", type=float, default=10.0, help="Domain extent")
    args = parser.parse_args()

    search_pattern = os.path.join(args.pattern, "plt*") if os.path.isdir(args.pattern) and not args.pattern.startswith("plt") else args.pattern

    plotfiles = [
        p for p in glob.glob(search_pattern)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "Header"))
    ]
    plotfiles.sort(key=natural_sort_key)

    if not plotfiles:
        print(f"[ERROR] No plotfiles matching '{search_pattern}'.")
        sys.exit(1)

    print(f"[INFO] Found {len(plotfiles)} plotfiles.\n")

    frame_files = []
    for plotfile in plotfiles:
        frame_files.append(
            plot_single_frame(
                plotfile,
                hl=args.hl,
                hr=args.hr,
                x0=args.x0,
                L=args.L,
            )
        )

    if frame_files:
        generate_gif(frame_files)

    print("\n[DONE] All frames and GIF generated.")
    print("Tip: For the rectangular dam break, check the line-cut plot.")
    print("     If all y-cuts overlap, the solution is effectively 1D.")
    print("     Spread in the line cuts indicates numerical error or flux asymmetry.")


if __name__ == "__main__":
    main()