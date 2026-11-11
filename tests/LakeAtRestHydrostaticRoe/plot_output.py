#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np

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
FIGSIZE = (14, 16)
DPI = 200

VARIABLE_ALIASES = {
    "h": ["h_fluid", "h", "water_depth", "Depth"],
    "hu": ["hu", "hu_momentum", "MomentumX"],
    "hv": ["hv", "hv_momentum", "MomentumY"],
    "z": ["z_bathymetry", "z", "bathymetry", "BedElevation", "elev"],
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def natural_sort_key(s):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def find_variable(ds, var_key):
    for alias in VARIABLE_ALIASES[var_key]:
        if ("boxlib", alias) in ds.field_list:
            return ("boxlib", alias)
        if alias in ds.field_list:
            return ("boxlib", alias)
    for field in ds.field_list:
        name = field[1] if len(field) > 1 else str(field)
        if any(alias in name for alias in VARIABLE_ALIASES[var_key]):
            return field
    return None


def get_domain_bounds(ds):
    lo = np.asarray(ds.domain_left_edge.to_ndarray(), dtype=float)
    hi = np.asarray(ds.domain_right_edge.to_ndarray(), dtype=float)
    return lo, hi


def build_amr_composite(ds, field):
    lo, hi = get_domain_bounds(ds)
    levels = np.asarray(ds.index.grid_levels, dtype=int).flatten()
    max_level = int(levels.max())
    fine_dx = np.asarray(ds.index.level_dds[max_level], dtype=float)
    size = hi - lo
    nx = int(round(size[0] / fine_dx[0]))
    ny = int(round(size[1] / fine_dx[1]))

    composite = np.full((nx, ny), np.nan, dtype=float)
    level_map = np.full((nx, ny), -1, dtype=np.int16)
    grids = ds.index.grids
    order = np.argsort(levels)

    for grid_index in order:
        grid = grids[grid_index]
        level = int(levels[grid_index])
        left = np.asarray(grid.LeftEdge.to_ndarray(), dtype=float)
        data = np.asarray(grid[field].to_ndarray(), dtype=float).squeeze()
        if data.ndim != 2:
            continue

        gx = int(round((left[0] - lo[0]) / fine_dx[0]))
        gy = int(round((left[1] - lo[1]) / fine_dx[1]))
        stride = max(1, int(round(float(grid.dds[0]) / fine_dx[0])))

        if gx < 0 or gy < 0 or gx >= nx or gy >= ny:
            continue

        expanded = np.repeat(np.repeat(data, stride, axis=0), stride, axis=1)
        ex = min(nx - gx, expanded.shape[0])
        ey = min(ny - gy, expanded.shape[1])

        composite[gx:gx + ex, gy:gy + ey] = expanded[:ex, :ey]
        level_map[gx:gx + ex, gy:gy + ey] = level

    return composite, level_map, nx, ny, fine_dx[0], fine_dx[1]


def composite_profile(ds, field, y_position):
    levels = np.asarray(ds.index.grid_levels, dtype=int).flatten()
    grids = ds.index.grids
    segments = []

    for grid, level in zip(grids, levels):
        left = np.asarray(grid.LeftEdge.to_ndarray(), dtype=float)
        right = np.asarray(grid.RightEdge.to_ndarray(), dtype=float)
        if y_position < left[1] or y_position > right[1]:
            continue

        data = np.asarray(grid[field].to_ndarray(), dtype=float).squeeze()
        if data.ndim != 2:
            continue

        dx = float(grid.dds[0])
        dy = float(grid.dds[1])
        iy = int(np.floor((y_position - left[1]) / dy))
        iy = np.clip(iy, 0, data.shape[1] - 1)
        x = left[0] + (np.arange(data.shape[0]) + 0.5) * dx
        segments.append({
            "level": int(level), "x": x, "values": data[:, iy],
            "x0": left[0], "x1": right[0],
        })

    if not segments:
        return np.array([]), np.array([]), np.array([])

    boundaries = np.sort(np.unique(np.concatenate(
        [np.array([s["x0"], s["x1"]]) for s in segments])))
    result_x, result_values, result_levels = [], [], []

    for xa, xb in zip(boundaries[:-1], boundaries[1:]):
        if xb <= xa:
            continue
        candidates = [s for s in segments
                      if s["x0"] <= xa + 1e-12 and s["x1"] >= xb - 1e-12]
        if not candidates:
            continue
        selected = max(candidates, key=lambda s: s["level"])
        mask = (selected["x"] >= xa - 1e-12) & (selected["x"] < xb + 1e-12)
        result_x.append(selected["x"][mask])
        result_values.append(selected["values"][mask])
        result_levels.append(np.full(np.count_nonzero(mask), selected["level"], dtype=int))

    if not result_x:
        return np.array([]), np.array([]), np.array([])

    x = np.concatenate(result_x)
    values = np.concatenate(result_values)
    level_map = np.concatenate(result_levels)
    order = np.argsort(x)
    x, values, level_map = x[order], values[order], level_map[order]
    keep = np.ones(len(x), dtype=bool)
    keep[1:] = np.diff(x) > 1e-12
    return x[keep], values[keep], level_map[keep]


def draw_amr_boxes(ax, ds):
    """Overlay AMR grid box boundaries on 2D surface plot."""
    levels = np.asarray(ds.index.grid_levels, dtype=int).flatten()
    grids = ds.index.grids
    colors = ["white", "cyan", "red", "yellow"]

    for grid, level in zip(grids, levels):
        if level == 0:
            continue  # Don't clutter level 0
        left = grid.LeftEdge.to_ndarray()
        right = grid.RightEdge.to_ndarray()
        width = right[0] - left[0]
        height = right[1] - left[1]

        c = colors[level % len(colors)]
        rect = patches.Rectangle((left[0], left[1]), width, height,
                                 linewidth=1.2, edgecolor=c, facecolor="none", linestyle="--")
        ax.add_patch(rect)


def plot_single_frame(plotfile_path, H=0.010):
    if not HAS_YT:
        raise RuntimeError("yt is required.")

    ds = yt.load(plotfile_path)
    sim_time = float(ds.current_time)
    lo, hi = get_domain_bounds(ds)

    h_field = find_variable(ds, "h")
    hu_field = find_variable(ds, "hu")
    hv_field = find_variable(ds, "hv")
    z_field = find_variable(ds, "z")

    if h_field is None or z_field is None:
        raise RuntimeError("Missing h or z variables in plotfile.")

    # Build composites
    h, level_map, nx, ny, dx, dy = build_amr_composite(ds, h_field)
    z, _, _, _, _, _ = build_amr_composite(ds, z_field)
    hu = build_amr_composite(ds, hu_field)[0] if hu_field is not None else np.zeros_like(h)
    hv = build_amr_composite(ds, hv_field)[0] if hv_field is not None else np.zeros_like(h)

    # Free-surface calculation
    eta = h + z
    eta_error = eta - H  # CORRECTED: Surface elevation error eta - H

    # Velocity calculation (prevent zero division)
    u_vel = np.where(h > 1e-12, hu / h, 0.0)
    v_vel = np.where(h > 1e-12, hv / h, 0.0)
    vel_mag = np.sqrt(u_vel**2 + v_vel**2)

    # Cross-section profiles
    y_center = 0.5 * (lo[1] + hi[1])
    x_h, h_profile, levels = composite_profile(ds, h_field, y_center)
    x_z, z_profile, _ = composite_profile(ds, z_field, y_center)
    _, hu_profile, _ = composite_profile(ds, hu_field, y_center) if hu_field else (None, np.zeros_like(h_profile), None)

    eta_profile = h_profile + z_profile
    eta_err_profile = eta_profile - H
    u_profile = np.where(h_profile > 1e-12, hu_profile / h_profile, 0.0)

    # Diagnostics
    max_eta_error = np.nanmax(np.abs(eta_error))
    max_u_vel = np.nanmax(np.abs(u_vel))
    max_v_vel = np.nanmax(np.abs(v_vel))

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = gridspec.GridSpec(5, 1, height_ratios=[3.0, 2.5, 2.0, 2.0, 0.4], hspace=0.35)

    # 1. 2D Free-Surface Error Map with AMR Boxes
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(eta_error.T, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]],
                     cmap="coolwarm", interpolation="nearest", aspect="equal")
    draw_amr_boxes(ax1, ds)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title(rf"2D Surface Elevation Error $\eta - H$ (Overlaid with AMR Level Boxes)")
    cbar1 = fig.colorbar(im1, ax=ax1, orientation="vertical", pad=0.02)
    cbar1.set_label(r"$\eta - H$ [m]")

    # 2. Cross-Section: Water Depth, Free Surface & Bathymetry
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x_h, eta_profile, "b-", linewidth=2.0, label=r"Free Surface $\eta = h + z$")
    ax2.plot(x_z, z_profile, "k-", linewidth=1.5, label=r"Bathymetry $z$")
    ax2.axhline(H, color="r", linestyle="--", linewidth=1.0, label=r"Target Surface $H$")
    ax2.fill_between(x_z, z_profile, eta_profile, color="cyan", alpha=0.3, label="Water Body ($h$)")
    ax2.set_xlim(lo[0], hi[0])
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("Elevation [m]")
    ax2.set_title("Centerline Cross-Section Profile")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Surface Error Cross-Section (C-Property Check)
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(x_h, eta_err_profile, "m-", linewidth=1.2, label=r"$\eta - H$")
    ax3.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax3.set_xlim(lo[0], hi[0])
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel(r"$\eta - H$ [m]")
    ax3.set_title(rf"Free Surface Discrepancy (Max $|\eta - H| = {max_eta_error:.3e}$ m)")
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax3.grid(True, alpha=0.3)

    # 4. Velocity Spikes (u)
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(x_h, u_profile, "g-", linewidth=1.2, label="u velocity")
    ax4.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax4.set_xlim(lo[0], hi[0])
    ax4.set_xlabel("x [m]")
    ax4.set_ylabel(r"Velocity $u$ [m/s]")
    ax4.set_title(rf"Spurious Velocity Spikes (Max |u| = {max_u_vel:.3e} m/s)")
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax4.grid(True, alpha=0.3)

    # 5. AMR Level Map Strip
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(x_h, levels, "k-", drawstyle="steps-mid")
    ax5.set_xlim(lo[0], hi[0])
    ax5.set_xlabel("x [m]")
    ax5.set_ylabel("AMR Level")
    ax5.set_yticks(np.unique(levels))
    ax5.grid(True, alpha=0.3)

    fig.suptitle(f"HydroEXA Lake-at-Rest C-Property Diagnostics\nTime = {sim_time:.6e} s", fontsize=14)
    outname = os.path.join(OUTPUT_DIR, f"frame_{sim_time:.6f}.png")
    fig.savefig(outname, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {outname} | Max |eta-H| = {max_eta_error:.4e} m | Max |u| = {max_u_vel:.4e} m/s")
    return outname

def main():
    parser = argparse.ArgumentParser(description="Plot C-property diagnostics for HydroEXA AMR.")
    parser.add_argument("pattern", nargs="?", default="plt*", help="Plotfile directory/pattern")
    parser.add_argument("--H", type=float, default=0.010, help="Target constant surface elevation")
    args = parser.parse_args()

    plotfiles = sorted(glob.glob(args.pattern), key=natural_sort_key)
    plotfiles = [p for p in plotfiles if os.path.isdir(p) and os.path.exists(os.path.join(p, "Header"))]

    if not plotfiles:
        print("[ERROR] No plotfiles found.")
        sys.exit(1)

    for p in plotfiles:
        plot_single_frame(p, H=args.H)

if __name__ == "__main__":
    main()