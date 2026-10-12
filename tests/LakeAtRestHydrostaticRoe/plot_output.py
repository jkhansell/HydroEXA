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
FIGSIZE = (15, 15)
DPI = 250

VARIABLE_ALIASES = {
    "h": ["h_fluid", "h", "water_depth", "Depth"],
    "hu": ["hu", "hu_momentum", "MomentumX"],
    "hv": ["hv", "hv_momentum", "MomentumY"],
    "z": ["z_bathymetry", "z", "bathymetry", "BedElevation", "elev"],
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def natural_sort_key(s):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", s)
    ]


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

    levels = np.asarray(
        ds.index.grid_levels,
        dtype=int,
    ).flatten()

    dds = ds.index.level_dds

    info = []

    for level in sorted(set(levels)):
        dx = np.asarray(dds[level], dtype=float)

        nx = int(round(size[0] / dx[0]))
        ny = int(round(size[1] / dx[1]))

        info.append({
            "level": level,
            "nx": nx,
            "ny": ny,
            "dx": dx[0],
            "dy": dx[1],
            "nblocks": int(np.count_nonzero(levels == level)),
        })

    return info


def build_amr_composite(ds, field):
    """
    Build a composite at the finest AMR resolution.
    Coarser data are expanded and overwritten by finer data.
    """

    lo, hi = get_domain_bounds(ds)

    levels = np.asarray(
        ds.index.grid_levels,
        dtype=int,
    ).flatten()

    max_level = int(levels.max())

    fine_dx = np.asarray(
        ds.index.level_dds[max_level],
        dtype=float,
    )

    size = hi - lo

    nx = int(round(size[0] / fine_dx[0]))
    ny = int(round(size[1] / fine_dx[1]))

    composite = np.full(
        (nx, ny),
        np.nan,
        dtype=float,
    )

    level_map = np.full(
        (nx, ny),
        -1,
        dtype=np.int16,
    )

    grids = ds.index.grids

    order = np.argsort(levels)

    for grid_index in order:

        grid = grids[grid_index]
        level = int(levels[grid_index])

        left = np.asarray(
            grid.LeftEdge.to_ndarray(),
            dtype=float,
        )

        data = np.asarray(
            grid[field].to_ndarray(),
            dtype=float,
        ).squeeze()

        if data.ndim != 2:
            continue

        gx = int(
            round(
                (left[0] - lo[0]) / fine_dx[0]
            )
        )

        gy = int(
            round(
                (left[1] - lo[1]) / fine_dx[1]
            )
        )

        stride = max(
            1,
            int(
                round(
                    float(grid.dds[0]) / fine_dx[0]
                )
            ),
        )

        if gx < 0 or gy < 0 or gx >= nx or gy >= ny:
            continue

        expanded = np.repeat(
            np.repeat(
                data,
                stride,
                axis=0,
            ),
            stride,
            axis=1,
        )

        ex = min(
            nx - gx,
            expanded.shape[0],
        )

        ey = min(
            ny - gy,
            expanded.shape[1],
        )

        composite[
            gx:gx + ex,
            gy:gy + ey,
        ] = expanded[:ex, :ey]

        level_map[
            gx:gx + ex,
            gy:gy + ey,
        ] = level

    return (
        composite,
        level_map,
        nx,
        ny,
        fine_dx[0],
        fine_dx[1],
    )


def composite_profile(ds, field, y_position):
    """
    Extract the finest available AMR centerline profile.
    """

    levels = np.asarray(
        ds.index.grid_levels,
        dtype=int,
    ).flatten()

    grids = ds.index.grids

    segments = []

    for grid, level in zip(grids, levels):

        left = np.asarray(
            grid.LeftEdge.to_ndarray(),
            dtype=float,
        )

        right = np.asarray(
            grid.RightEdge.to_ndarray(),
            dtype=float,
        )

        if (
            y_position < left[1]
            or y_position > right[1]
        ):
            continue

        data = np.asarray(
            grid[field].to_ndarray(),
            dtype=float,
        ).squeeze()

        if data.ndim != 2:
            continue

        dx = float(grid.dds[0])
        dy = float(grid.dds[1])

        iy = int(
            np.floor(
                (y_position - left[1]) / dy
            )
        )

        iy = np.clip(
            iy,
            0,
            data.shape[1] - 1,
        )

        x = (
            left[0]
            + (np.arange(data.shape[0]) + 0.5) * dx
        )

        values = data[:, iy]

        segments.append({
            "level": int(level),
            "x": x,
            "values": values,
            "x0": left[0],
            "x1": right[0],
        })

    if not segments:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )

    boundaries = np.sort(
        np.unique(
            np.concatenate(
                [
                    np.array([s["x0"], s["x1"]])
                    for s in segments
                ]
            )
        )
    )

    result_x = []
    result_values = []
    result_levels = []

    for xa, xb in zip(
        boundaries[:-1],
        boundaries[1:],
    ):

        if xb <= xa:
            continue

        candidates = [
            s
            for s in segments
            if (
                s["x0"] <= xa + 1e-12
                and s["x1"] >= xb - 1e-12
            )
        ]

        if not candidates:
            continue

        selected = max(
            candidates,
            key=lambda s: s["level"],
        )

        mask = (
            (selected["x"] >= xa - 1e-12)
            & (selected["x"] < xb + 1e-12)
        )

        result_x.append(
            selected["x"][mask]
        )

        result_values.append(
            selected["values"][mask]
        )

        result_levels.append(
            np.full(
                np.count_nonzero(mask),
                selected["level"],
                dtype=int,
            )
        )

    if not result_x:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )

    x = np.concatenate(result_x)
    values = np.concatenate(result_values)
    level_map = np.concatenate(result_levels)

    order = np.argsort(x)

    x = x[order]
    values = values[order]
    level_map = level_map[order]

    keep = np.ones(
        len(x),
        dtype=bool,
    )

    keep[1:] = (
        np.diff(x) > 1e-12
    )

    return (
        x[keep],
        values[keep],
        level_map[keep],
    )


def plot_level_strip(
    ax,
    x,
    levels,
    level_info,
):

    if len(x) == 0:
        return

    changes = (
        np.flatnonzero(
            np.diff(levels) != 0
        )
        + 1
    )

    starts = np.concatenate(
        ([0], changes)
    )

    ends = np.concatenate(
        (changes, [len(x)])
    )

    unique_levels = sorted(
        set(levels)
    )

    cmap = plt.cm.viridis

    colors = {
        level: cmap(
            i / max(
                len(unique_levels) - 1,
                1,
            )
        )
        for i, level in enumerate(
            unique_levels
        )
    }

    for start, end in zip(
        starts,
        ends,
    ):

        if end <= start:
            continue

        xa = x[start]
        xb = x[end - 1]

        if end < len(x):
            xb = x[end]

        level = int(levels[start])

        ax.axvspan(
            xa,
            xb,
            color=colors[level],
            alpha=0.75,
            linewidth=0,
        )

    for level in unique_levels:

        info = next(
            i
            for i in level_info
            if i["level"] == level
        )

        ax.plot(
            [],
            [],
            color=colors[level],
            linewidth=8,
            label=(
                f"L{level}: "
                f"Δx={info['dx']:.5f} m"
            ),
        )

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("AMR")

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=False,
    )

    ax.grid(False)


def plot_single_frame(
    plotfile_path,
    H=0.010,
):

    if not HAS_YT:
        raise RuntimeError(
            "yt is required."
        )

    ds = yt.load(plotfile_path)

    sim_time = float(
        ds.current_time
    )

    lo, hi = get_domain_bounds(ds)

    h_field = find_variable(ds, "h")
    hu_field = find_variable(ds, "hu")
    hv_field = find_variable(ds, "hv")
    z_field = find_variable(ds, "z")

    if h_field is None:
        raise RuntimeError(
            "Could not find water-depth variable."
        )

    if z_field is None:
        raise RuntimeError(
            "Could not find bathymetry variable."
        )

    level_info = get_level_info(ds)

    # ------------------------------------------------------------------
    # Composite fields
    # ------------------------------------------------------------------

    h, h_levels, nx, ny, dx, dy = (
        build_amr_composite(
            ds,
            h_field,
        )
    )

    z, _, _, _, _, _ = (
        build_amr_composite(
            ds,
            z_field,
        )
    )

    hu = None
    hv = None

    if hu_field is not None:
        hu, _, _, _, _, _ = (
            build_amr_composite(
                ds,
                hu_field,
            )
        )

    if hv_field is not None:
        hv, _, _, _, _, _ = (
            build_amr_composite(
                ds,
                hv_field,
            )
        )

    eta = h + z

    eta_error = eta - H

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    valid = np.isfinite(h)

    max_eta_error = np.nanmax(
        np.abs(eta_error[valid])
    )

    max_hu = (
        np.nanmax(np.abs(hu[valid]))
        if hu is not None
        else np.nan
    )

    max_hv = (
        np.nanmax(np.abs(hv[valid]))
        if hv is not None
        else np.nan
    )

    max_h = np.nanmax(h[valid])
    min_h = np.nanmin(h[valid])

    # ------------------------------------------------------------------
    # Centerline
    # ------------------------------------------------------------------

    y_center = (
        0.5 * (lo[1] + hi[1])
    )

    x_h, h_profile, levels = (
        composite_profile(
            ds,
            h_field,
            y_center,
        )
    )

    x_z, z_profile, _ = (
        composite_profile(
            ds,
            z_field,
            y_center,
        )
    )

    x_hu = hu_profile = None
    x_hv = hv_profile = None

    if hu_field is not None:
        x_hu, hu_profile, _ = (
            composite_profile(
                ds,
                hu_field,
                y_center,
            )
        )

    if hv_field is not None:
        x_hv, hv_profile, _ = (
            composite_profile(
                ds,
                hv_field,
                y_center,
            )
        )

    if len(x_h) > 0 and len(x_z) > 0:

        eta_profile = (
            h_profile + z_profile
        )

        eta_error_profile = (
            eta_profile - H
        )

    else:

        eta_profile = np.array([])
        eta_error_profile = np.array([])

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig = plt.figure(
        figsize=FIGSIZE,
        dpi=DPI,
    )

    gs = gridspec.GridSpec(
        5,
        1,
        height_ratios=[
            3.5,
            0.4,
            2.5,
            2.5,
            2.5,
        ],
        hspace=0.35,
    )

    # ------------------------------------------------------------------
    # 1. Water depth
    # ------------------------------------------------------------------

    ax1 = fig.add_subplot(gs[0])

    im = ax1.imshow(
        h.T,
        origin="lower",
        extent=[
            lo[0],
            hi[0],
            lo[1],
            hi[1],
        ],
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )

    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")

    ax1.set_title(
        f"Lake-at-rest water depth, "
        f"t = {sim_time:.6e} s"
    )

    cbar = fig.colorbar(
        im,
        ax=ax1,
        orientation="horizontal",
        fraction=0.07,
        pad=0.18,
    )

    cbar.set_label("h [m]")

    # ------------------------------------------------------------------
    # 2. AMR strip
    # ------------------------------------------------------------------

    ax2 = fig.add_subplot(gs[1])

    if len(x_h) > 0:
        plot_level_strip(
            ax2,
            x_h,
            levels,
            level_info,
        )

    ax2.set_xlim(
        lo[0],
        hi[0],
    )

    ax2.set_xlabel("x [m]")
    ax2.set_title(
        "AMR resolution",
        fontsize=10,
        pad=3,
    )

    # ------------------------------------------------------------------
    # 3. h + z
    # ------------------------------------------------------------------

    ax3 = fig.add_subplot(gs[2])

    if len(x_h) > 0:

        ax3.plot(
            x_h,
            eta_profile,
            linewidth=1.5,
            label=r"$\eta=h+z$",
        )

        ax3.axhline(
            H,
            linestyle="--",
            linewidth=2,
            label=r"target $H$",
        )

    ax3.set_xlim(
        lo[0],
        hi[0],
    )

    ax3.set_xlabel("x [m]")
    ax3.set_ylabel(r"$\eta=h+z$ [m]")

    ax3.set_title(
        r"Free-surface elevation: "
        r"$\eta=h+z$"
    )

    ax3.legend(
        loc="best",
        fontsize=8,
    )

    ax3.grid(
        True,
        alpha=0.25,
    )

    # ------------------------------------------------------------------
    # 4. Free-surface error
    # ------------------------------------------------------------------

    ax4 = fig.add_subplot(gs[3])

    if len(x_h) > 0:

        ax4.plot(
            x_h,
            eta_error_profile,
            linewidth=1.0,
        )

    ax4.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax4.set_xlim(
        lo[0],
        hi[0],
    )

    ax4.set_xlabel("x [m]")
    ax4.set_ylabel(r"$\eta-H$ [m]")

    ax4.set_title(
        rf"Lake-at-rest free-surface error "
        rf"(max = {max_eta_error:.3e} m)"
    )

    ax4.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(-2, 2),
    )

    ax4.grid(
        True,
        alpha=0.25,
    )

    # ------------------------------------------------------------------
    # 5. Momentum
    # ------------------------------------------------------------------

    ax5 = fig.add_subplot(gs[4])

    if x_hu is not None:
        ax5.plot(
            x_hu,
            hu_profile,
            linewidth=1.0,
            label="hu",
        )

    if x_hv is not None:
        ax5.plot(
            x_hv,
            hv_profile,
            linewidth=1.0,
            label="hv",
        )

    ax5.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax5.set_xlim(
        lo[0],
        hi[0],
    )

    ax5.set_xlabel("x [m]")
    ax5.set_ylabel(
        r"momentum [m$^2$/s]"
    )

    ax5.set_title(
        rf"Momentum "
        rf"(max |hu| = {max_hu:.3e}, "
        rf"max |hv| = {max_hv:.3e})"
    )

    ax5.legend(
        loc="best",
        fontsize=8,
    )

    ax5.grid(
        True,
        alpha=0.25,
    )

    # ------------------------------------------------------------------
    # Overall title / diagnostics
    # ------------------------------------------------------------------

    fig.suptitle(
        (
            f"Lake-at-Rest Well-Balanced Test\n"
            f"t={sim_time:.6e} s | "
            f"max|η-H|={max_eta_error:.3e} m | "
            f"max|hu|={max_hu:.3e} | "
            f"max|hv|={max_hv:.3e}"
        ),
        fontsize=13,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        top=0.93,
        bottom=0.06,
    )

    outname = os.path.join(
        OUTPUT_DIR,
        f"frame_{sim_time}.png",
    )

    fig.savefig(
        outname,
        dpi=DPI,
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {outname}")
    print(
        f"  t              = {sim_time:.6e}"
    )
    print(
        f"  max |eta-H|    = {max_eta_error:.6e}"
    )
    print(
        f"  max |hu|       = {max_hu:.6e}"
    )
    print(
        f"  max |hv|       = {max_hv:.6e}"
    )
    print(
        f"  h range        = "
        f"[{min_h:.6e}, {max_h:.6e}]"
    )

    return outname


def generate_gif(
    frame_files,
    output_gif=None,
    duration=0.2,
):

    if imageio is None:
        print(
            "[ERROR] imageio is required."
        )
        return

    output_gif = (
        output_gif
        or os.path.join(
            OUTPUT_DIR,
            "lake_at_rest.gif",
        )
    )

    images = [
        imageio.imread(f)
        for f in frame_files
        if os.path.exists(f)
    ]

    if not images:
        print(
            "[ERROR] No frames available."
        )
        return

    imageio.mimsave(
        output_gif,
        images,
        duration=duration,
        loop=0,
    )

    print(
        f"[SUCCESS] Saved: {output_gif}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot HydroEXA lake-at-rest "
            "AMR solutions."
        )
    )

    parser.add_argument(
        "pattern",
        nargs="?",
        default="plt*",
        help=(
            "Glob pattern or directory "
            "containing plotfiles"
        ),
    )

    parser.add_argument(
        "--H",
        type=float,
        default=0.010,
        help="Expected constant free-surface elevation",
    )

    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Do not generate GIF",
    )

    args = parser.parse_args()

    search_pattern = (
        os.path.join(
            args.pattern,
            "plt*",
        )
        if os.path.isdir(args.pattern)
        and not args.pattern.startswith("plt")
        else args.pattern
    )

    plotfiles = [
        p
        for p in glob.glob(search_pattern)
        if (
            os.path.isdir(p)
            and os.path.exists(
                os.path.join(
                    p,
                    "Header",
                )
            )
        )
    ]

    plotfiles.sort(
        key=natural_sort_key
    )

    if not plotfiles:
        print(
            f"[ERROR] No plotfiles "
            f"matching '{search_pattern}'."
        )
        sys.exit(1)

    print(
        f"[INFO] Found "
        f"{len(plotfiles)} plotfiles.\n"
    )

    frame_files = [
        plot_single_frame(
            p,
            H=args.H,
        )
        for p in plotfiles
    ]

    if frame_files and not args.no_gif:
        generate_gif(frame_files)

    print(
        "\n[DONE] All frames generated."
    )


if __name__ == "__main__":
    main()