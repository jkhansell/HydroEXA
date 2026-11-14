#!/usr/bin/env python3
"""
plot.py — rendering for the dambreak battery (plot_output.py style).

One figure per plotfile: composite 2D h, AMR resolution strip, centerline
h and hu profiles vs the SWASHES exact solution, plus the CI metrics
(computed in analyze.py) in a text box. Reuses the AMR-composite readers
and the analytical helper — one rendering source of truth.

analysis.py imports the renderer from here; the CLI renders one plotfile.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import (find_variable, get_domain_bounds, get_level_info,
                     build_amr_composite, composite_profile, plot_level_strip,
                     all_metrics, metrics_summary, exact_profile)

FIGSIZE = (15, 14)
DPI = 200


def render_single_frame(ds, outname, metrics_text=None,
                        hl=0.005, hr=0.001, x0=5.0, L=10.0):
    """Render the 4-panel composite figure for a loaded yt dataset."""
    sim_time = float(ds.current_time)
    lo, hi = get_domain_bounds(ds)

    h_field = find_variable(ds, "h")
    hu_field = find_variable(ds, "hu")
    if h_field is None:
        raise RuntimeError("Could not find water-depth variable.")

    level_info = get_level_info(ds)

    h_composite, h_levels, nx, ny, dx, dy = build_amr_composite(ds, h_field)
    hu_composite = None
    if hu_field is not None:
        hu_composite, _, _, _, _, _ = build_amr_composite(ds, hu_field)

    h_image = h_composite.T
    level_image = h_levels.T

    xg, h_ana, u_ana = exact_profile(sim_time, L, hl, hr, x0,
                                     n=2000, x_lo=lo[0])

    y_center = 0.5 * (lo[1] + hi[1])
    x_h, h_profile, h_profile_levels = composite_profile(ds, h_field, y_center)
    x_hu, hu_profile, hu_profile_levels = (np.array([]), np.array([]), np.array([]))
    if hu_field is not None:
        x_hu, hu_profile, hu_profile_levels = composite_profile(ds, hu_field, y_center)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = gridspec.GridSpec(5, 1, height_ratios=[4.2, 0.45, 3.0, 2.8, 0.9], hspace=0.45)

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
    ax3.plot(xg, h_ana, "k--", linewidth=2.5, label="Analytical", zorder=5)
    if len(x_h) > 0:
        ax3.plot(x_h, h_profile, linewidth=1, marker=".", markersize=2,
                 label="Composite AMR", zorder=10)
    ax3.set_xlim(lo[0], hi[0])
    ax3.set_ylim(hr - 0.001, hl + 0.001)
    ax3.set_ylabel("h [m]")
    ax3.set_title("Water-depth profile: composite AMR")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.25)

    # Panel 4: hu profile
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(xg, h_ana * u_ana, "k--", linewidth=2.5, label="Analytical", zorder=5)
    if len(x_hu) > 0:
        ax4.plot(x_hu, hu_profile, linewidth=1, marker=".", markersize=2,
                 label="Composite AMR", zorder=10)
    ax4.set_xlim(lo[0], hi[0])
    ax4.set_xlabel("x [m]")
    ax4.set_ylabel("hu [m²/s]")
    ax4.set_title("x-momentum profile: composite AMR")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.25)

    # Panel 5: metrics text
    ax5 = fig.add_subplot(gs[4])
    ax5.axis("off")
    if metrics_text:
        ax5.text(0.0, 0.6, metrics_text, fontsize=11, family="monospace",
                 va="center", ha="left",
                 bbox=dict(boxstyle="round", facecolor="#f2f7ff", edgecolor="#88aadd"))

    fig.subplots_adjust(left=0.08, right=0.88, top=0.96, bottom=0.05)
    fig.savefig(outname, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {outname}")
    return outname


def main():
    parser = argparse.ArgumentParser(description="Render one dambreak plotfile (plot_output style)")
    parser.add_argument("plotfile", help="Path to AMReX native plotfile directory")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--metrics-text", type=str, default=None, help="Metrics block to embed")
    parser.add_argument("--hl", type=float, default=0.005)
    parser.add_argument("--hr", type=float, default=0.001)
    parser.add_argument("--x0", type=float, default=5.0)
    parser.add_argument("--L", type=float, default=10.0)
    args = parser.parse_args()

    import yt
    ds = yt.load(args.plotfile)
    out = args.output or os.path.join(
        os.path.dirname(args.plotfile),
        f"frame_t{float(ds.current_time):.4f}.png")
    render_single_frame(ds, out, metrics_text=args.metrics_text,
                        hl=args.hl, hr=args.hr, x0=args.x0, L=args.L)


if __name__ == "__main__":
    main()
