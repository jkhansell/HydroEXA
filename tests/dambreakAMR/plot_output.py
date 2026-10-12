#!/usr/bin/env python3

import argparse, glob, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
FIGSIZE = (15, 13)
DPI = 250

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
    return (
        np.asarray(ds.domain_left_edge.to_ndarray(), dtype=float),
        np.asarray(ds.domain_right_edge.to_ndarray(), dtype=float),
    )


def get_level_info(ds):
    lo, hi = get_domain_bounds(ds)
    size = hi - lo
    levels = np.asarray(ds.index.grid_levels, dtype=int).flatten()
    dds = ds.index.level_dds
    info = []
    for level in sorted(set(levels)):
        dx = np.asarray(dds[level], dtype=float)
        nx = int(round(size[0] / dx[0]))
        ny = int(round(size[1] / dx[1]))
        info.append({
            "level": level,
            "nx": nx, "ny": ny,
            "dx": dx[0], "dy": dx[1],
            "nblocks": int(np.count_nonzero(levels == level)),
        })
    return info


def build_amr_composite(ds, field):
    """Build composite from native AMR grids: level 0 first, finer levels overwrite."""
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

    # Process from coarsest to finest so finer data overwrites
    order = np.argsort(levels)
    for i in order:
        grid = grids[i]
        level = int(levels[i])
        left = np.asarray(grid.LeftEdge.to_ndarray(), dtype=float)
        dims = np.asarray(grid.ActiveDimensions, dtype=int)
        data = np.asarray(grid[field].to_ndarray(), dtype=float).squeeze()
        if data.ndim != 2:
            continue

        gx = int(round((left[0] - lo[0]) / fine_dx[0]))
        gy = int(round((left[1] - lo[1]) / fine_dx[1]))
        stride = max(1, int(round(float(grid.dds[0]) / fine_dx[0])))

        x1 = min(nx, gx + data.shape[0] * stride)
        y1 = min(ny, gy + data.shape[1] * stride)
        if gx < 0 or gy < 0 or gx >= nx or gy >= ny:
            continue

        data = data[:max(0, (x1 - gx + stride - 1) // stride),
                     :max(0, (y1 - gy + stride - 1) // stride)]
        expanded = np.repeat(np.repeat(data, stride, axis=0), stride, axis=1)
        ex = min(nx - gx, expanded.shape[0])
        ey = min(ny - gy, expanded.shape[1])

        composite[gx:gx + ex, gy:gy + ey] = expanded[:ex, :ey]
        level_map[gx:gx + ex, gy:gy + ey] = level

    return composite, level_map, nx, ny, fine_dx[0], fine_dx[1]


def composite_profile(ds, field, y_position):
    """Extract centerline profile using finest available level in each x-region."""
    lo, hi = get_domain_bounds(ds)
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
        values = data[:, iy]
        segments.append({
            "level": int(level), "x": x, "values": values,
            "dx": dx, "x0": left[0], "x1": right[0],
        })

    if not segments:
        return np.array([]), np.array([]), np.array([])

    # Build boundaries from all segment edges
    boundaries = np.sort(np.unique(np.concatenate(
        [np.array([s["x0"], s["x1"]]) for s in segments])))

    result_x, result_values, result_levels = [], [], []
    for xa, xb in zip(boundaries[:-1], boundaries[1:]):
        if xb <= xa:
            continue
        candidates = [s for s in segments if s["x0"] <= xa + 1e-12 and s["x1"] >= xb - 1e-12]
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


def plot_level_strip(ax, x, levels, level_info):
    """Draw a thin resolution strip showing active AMR level along x."""
    if len(x) == 0:
        return

    changes = np.flatnonzero(np.diff(levels) != 0) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(x)]))

    unique_levels = sorted(set(levels))
    cmap = plt.cm.viridis
    colors = {level: cmap(i / max(len(unique_levels) - 1, 1)) for i, level in enumerate(unique_levels)}

    for start, end in zip(starts, ends):
        if end <= start:
            continue
        xa = x[start]
        xb = x[end - 1]
        if end < len(x):
            xb = x[end]
        level = int(levels[start])
        ax.axvspan(xa, xb, color=colors[level], alpha=0.75, linewidth=0)

    for level in unique_levels:
        info = next(i for i in level_info if i["level"] == level)
        ax.plot([], [], color=colors[level], linewidth=8,
                label=f"L{level}: Δx={info['dx']:.5f} m")

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("AMR")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False)
    ax.grid(False)


def plot_single_frame(plotfile_path, hl=0.005, hr=0.001, x0=5.0, L=10.0):
    if not HAS_YT:
        raise RuntimeError("yt is required to generate plots.")

    ds = yt.load(plotfile_path)
    sim_time = float(ds.current_time)
    lo, hi = get_domain_bounds(ds)

    h_field = find_variable(ds, "h")
    hu_field = find_variable(ds, "hu")
    if h_field is None:
        raise RuntimeError("Could not find water-depth variable.")

    level_info = get_level_info(ds)

    # Composite 2D fields
    h_composite, h_levels, nx, ny, dx, dy = build_amr_composite(ds, h_field)
    hu_composite = None
    if hu_field is not None:
        hu_composite, _, _, _, _, _ = build_amr_composite(ds, hu_field)

    h_image = h_composite.T
    level_image = h_levels.T

    # Analytical solution
    x = np.linspace(lo[0], hi[0], 1000)
    h_ana, u_ana = dambreak_on_wet_no_friction_analytical(
        sim_time, x, L=L, hl=hl, hr=hr, x0=x0)

    # Centerline profiles (native AMR, finest-level selection)
    y_center = 0.5 * (lo[1] + hi[1])
    x_h, h_profile, h_profile_levels = composite_profile(ds, h_field, y_center)
    x_hu, hu_profile, hu_profile_levels = (np.array([]), np.array([]), np.array([]))
    if hu_field is not None:
        x_hu, hu_profile, hu_profile_levels = composite_profile(ds, hu_field, y_center)

    # Build figure
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = gridspec.GridSpec(4, 1, height_ratios=[4.2, 0.45, 3.0, 2.8], hspace=0.32)

    # Panel 1: composite 2D water depth
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(
        h_image, origin="lower",
        extent=[lo[0], hi[0], lo[1], hi[1]],
        cmap="viridis", vmin=hr, vmax=hl,
        interpolation="nearest", aspect="equal",
    )
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title(f"Composite AMR water depth at t = {sim_time:.4f} s")
    cbar = fig.colorbar(im, ax=ax1, orientation="horizontal", fraction=0.07, pad=0.18)
    cbar.set_label("h [m]")

    # Panel 2: AMR resolution strip
    ax2 = fig.add_subplot(gs[1])
    if len(x_h) > 0:
        plot_level_strip(ax2, x_h, h_profile_levels, level_info)
    ax2.set_xlim(lo[0], hi[0])
    ax2.set_xlabel("x [m]")
    ax2.set_title("Native AMR resolution along centerline", fontsize=10, pad=3)

    # Panel 3: h profile
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(x, h_ana, "k--", linewidth=2.5, label="Analytical", zorder=5)
    if len(x_h) > 0:
        ax3.plot(x_h, h_profile, linewidth=1, markersize=2, label="Composite AMR", zorder=10)
    ax3.set_xlim(lo[0], hi[0])
    ax3.set_ylim(hr - 0.001, hl + 0.001)
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("h [m]")
    ax3.set_title("Water-depth profile: composite AMR")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.25)

    # Panel 4: hu profile
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(x, h_ana * u_ana, "k--", linewidth=2.5, label="Analytical", zorder=5)
    if len(x_hu) > 0:
        ax4.plot(x_hu, hu_profile, linewidth=1, markersize=2, label="Composite AMR", zorder=10)
    ax4.set_xlim(lo[0], hi[0])
    ax4.set_xlabel("x [m]")
    ax4.set_ylabel("hu [m²/s]")
    ax4.set_title("x-momentum profile: composite AMR")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.25)

    fig.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.06)

    outname = os.path.join(OUTPUT_DIR, f"frame_{sim_time:.4f}.png")
    fig.savefig(outname, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outname}")
    print("AMR levels:", ", ".join(f"L{i['level']}={i['dx']:.5f} m" for i in level_info))
    return outname


def generate_gif(frame_files, output_gif=None, duration=0.2):
    if imageio is None:
        print("[ERROR] imageio is required for GIF generation.")
        return
    output_gif = output_gif or os.path.join(OUTPUT_DIR, "h_evolution.gif")
    images = [imageio.imread(f) for f in frame_files if os.path.exists(f)]
    if not images:
        print("[ERROR] No frames available for GIF.")
        return
    imageio.mimsave(output_gif, images, duration=duration, loop=0)
    print(f"[SUCCESS] Saved: {output_gif}")


def main():
    parser = argparse.ArgumentParser(description="Plot HydroEXA composite AMR solutions.")
    parser.add_argument("pattern", nargs="?", default="plt*",
                        help="Glob pattern or directory containing plotfiles")
    parser.add_argument("--hl", type=float, default=0.005, help="Upstream water depth")
    parser.add_argument("--hr", type=float, default=0.001, help="Downstream water depth")
    parser.add_argument("--x0", type=float, default=5.0, help="Initial dam location")
    parser.add_argument("--L", type=float, default=10.0, help="Domain extent")
    args = parser.parse_args()

    search_pattern = (
        os.path.join(args.pattern, "plt*")
        if os.path.isdir(args.pattern) and not args.pattern.startswith("plt")
        else args.pattern
    )

    plotfiles = [p for p in glob.glob(search_pattern)
                 if os.path.isdir(p) and os.path.exists(os.path.join(p, "Header"))]
    plotfiles.sort(key=natural_sort_key)

    if not plotfiles:
        print(f"[ERROR] No plotfiles matching '{search_pattern}'.")
        sys.exit(1)

    print(f"[INFO] Found {len(plotfiles)} plotfiles.\n")

    frame_files = [
        plot_single_frame(p, hl=args.hl, hr=args.hr, x0=args.x0, L=args.L)
        for p in plotfiles
    ]

    if frame_files:
        generate_gif(frame_files)

    print("\n[DONE] All frames and GIF generated.")


if __name__ == "__main__":
    main()
