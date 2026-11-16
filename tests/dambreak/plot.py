#!/usr/bin/env python3
"""
plot.py — rendering for the dambreak battery.

One figure per plotfile:
  1. composite 2D water depth with the native AMR patches
  2. native AMR resolution strip along the centerline
  3. water-depth profiles: AMR vs the two single-level solutions vs analytical
  4. errors across the solution: h - h_exact and h - h_single_level for each
     single-level solution (full domain on a symlog scale, plus a zoom of the
     flat shelf that runs to the analytical shock elbow)
  5. x-momentum profiles: AMR vs single levels vs analytical
  6. the CI metrics (computed in analyze.py)

The two single-level curves are:
  stable — uniform run at the AMR level-1 cell size (dx0/2). This is what the
           AMR run should look like if its refinement were doing its job.
  base   — uniform run at the level-0 cell size, i.e. the unrefined solution
           the AMR run has to beat.

Every profile is drawn as a continuous line — no markers, so the eye reads
level rather than point spacing. The single-level and analytical solutions
are drawn with reduced opacity behind the AMR result.

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
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import (find_variable, get_domain_bounds, get_level_info,
                     build_amr_composite, composite_profile, plot_level_strip,
                     all_metrics, metrics_summary, exact_profile, plateau_band,
                     norms, default_references, load_reference, grid_boxes)

FIGSIZE = (15, 17)
# High by default: the plateau error is ~1% of the depth range, so the
# morphological detail only survives at large pixel counts. Override with
# HYDROEXA_PLOT_DPI or --dpi (e.g. 250 for quick looks / CI artifacts).
DPI = int(os.environ.get("HYDROEXA_PLOT_DPI", 400))

C_AMR = "C0"
C_ANA = "black"
# One style per yardstick slot (see analyze.REFERENCE_RUNS).
REF_STYLE = {
    "stable": dict(color="C4", lw=1.5, alpha=0.65),   # dx0/2 uniform
    "base": dict(color="C2", lw=1.4, alpha=0.55),     # dx0 uniform
}
REF_STYLE_FALLBACK = dict(color="C4", lw=1.4, alpha=0.60)


def render_single_frame(ds, outname, metrics_text=None,
                        hl=0.005, hr=0.001, x0=5.0, L=10.0, dpi=None,
                        references=None):
    """Render the composite figure for a loaded yt dataset.

    `references` is an optional list of dicts with keys x, h, hu, label, t and
    optional slot — the single-level solutions at (approximately) the same
    time. When absent the figure falls back to AMR-vs-analytical only.
    """
    dpi = dpi or DPI
    sim_time = float(ds.current_time)
    lo, hi = get_domain_bounds(ds)

    h_field = find_variable(ds, "h")
    hu_field = find_variable(ds, "hu")
    if h_field is None:
        raise RuntimeError("Could not find water-depth variable.")

    level_info = get_level_info(ds)

    h_composite, h_levels, nx, ny, dx, dy = build_amr_composite(ds, h_field)
    h_image = h_composite.T
    level_image = h_levels.T

    xg, h_ana, u_ana = exact_profile(sim_time, L, hl, hr, x0,
                                     n=2000, x_lo=lo[0])

    y_center = 0.5 * (lo[1] + hi[1])
    x_h, h_profile, h_profile_levels = composite_profile(ds, h_field, y_center)
    x_hu, hu_profile = np.array([]), np.array([])
    if hu_field is not None:
        x_hu, hu_profile, _ = composite_profile(ds, hu_field, y_center)

    # Keep only references that actually carry a usable h profile, and give
    # each its drawing style.
    refs = []
    for r in (references or []):
        if r.get("x") is None or len(np.atleast_1d(r["x"])) < 2:
            continue
        r = dict(r)
        r["style"] = REF_STYLE.get(r.get("slot", ""), REF_STYLE_FALLBACK)
        refs.append(r)
    has_ref = bool(refs)

    dh = hl - hr
    band = plateau_band(sim_time, L, hl, hr, x0)

    fig = plt.figure(figsize=FIGSIZE, dpi=dpi)
    gs = gridspec.GridSpec(6, 1,
                           height_ratios=[4.0, 0.45, 3.0, 2.4, 2.6, 1.15],
                           hspace=0.45)

    # --- Panel 1: composite 2D water depth ---------------------------------
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(
        h_image, origin="lower",
        extent=[lo[0], hi[0], lo[1], hi[1]],
        cmap="viridis", vmin=hr, vmax=hl,
        interpolation="nearest", aspect="equal",
    )
    # AMR box limits: a wiggle that lines up with these is a patch-boundary
    # artefact (ghost fill, reflux); one that does not is generated inside a
    # grid, where neither mechanism applies.
    boxes = [b for b in grid_boxes(ds) if b[0] > 0]
    for _lev, bx0, by0, bx1, by1 in boxes:
        ax1.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                fill=False, edgecolor="white", lw=0.7,
                                alpha=0.85, zorder=6))
    if boxes:
        ax1.plot([], [], color="white", lw=0.9,
                 label=f"level boxes ({len(boxes)})")
        ax1.legend(loc="upper right", fontsize=7, framealpha=0.55,
                   handlelength=1.4)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title(f"Composite AMR water depth at t = {sim_time:.4f} s")
    cbar = fig.colorbar(im, ax=ax1, orientation="horizontal", fraction=0.07,
                        pad=0.18)
    cbar.set_label("h [m]")

    # --- Panel 2: AMR resolution strip -------------------------------------
    ax2 = fig.add_subplot(gs[1])
    if len(x_h) > 0:
        plot_level_strip(ax2, x_h, h_profile_levels, level_info)
    ax2.set_xlim(lo[0], hi[0])
    ax2.set_xlabel("x [m]")
    ax2.set_title("Native AMR resolution along centerline", fontsize=10, pad=3)

    # --- Panel 3: h profiles, analytical + single levels + AMR --------------
    ax3 = fig.add_subplot(gs[2])
    if band is not None:
        ax3.axvspan(band[0], band[1], color="0.5", alpha=0.10, zorder=0)
    ax3.plot(xg, h_ana, color=C_ANA, ls="--", lw=2.2, alpha=0.70,
             label="Analytical", zorder=5)
    for r in refs:
        ax3.plot(r["x"], r["h"], zorder=8,
                 label=f"{r.get('label', 'single level')} (t={r.get('t', np.nan):.3f})",
                 **r["style"])
    if len(x_h) > 0:
        ax3.plot(x_h, h_profile, color=C_AMR, lw=1.0, alpha=0.95,
                 label="AMR", zorder=10)
    ax3.set_xlim(lo[0], hi[0])
    ax3.set_ylim(hr - 0.02 * dh, hl + 0.02 * dh)
    ax3.set_ylabel("h [m]")
    ax3.set_title("Water-depth profile: AMR vs single level(s) vs analytical")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.25)

    # --- Panel 4: errors across the solution -------------------------------
    sub = gs[3].subgridspec(1, 2, wspace=0.16)
    ax4a = fig.add_subplot(sub[0])
    ax4b = fig.add_subplot(sub[1])

    e_ana = h_profile - np.interp(x_h, xg, h_ana) if len(x_h) else np.array([])
    # AMR minus each single-level solution, on the overlapping x range.
    e_refs = []
    for r in refs:
        if not len(x_h):
            continue
        lo_ov = max(x_h.min(), np.min(r["x"]))
        hi_ov = min(x_h.max(), np.max(r["x"]))
        over = (x_h >= lo_ov) & (x_h <= hi_ov)
        e = np.full_like(x_h, np.nan)
        e[over] = h_profile[over] - np.interp(x_h[over], r["x"], r["h"])
        if e.size == e_ana.size and np.any(np.isfinite(e)):
            e_refs.append((r, e))

    # full domain, symlog: the shock-position spike is O(1e-2) while the
    # plateau structure is O(1e-5), so a linear axis would hide the latter.
    ax4a.axhline(0.0, color="0.45", lw=0.8, ls=":")
    if len(x_h):
        ax4a.plot(x_h, e_ana, color=C_AMR, lw=0.9, alpha=0.9,
                  label="AMR $-$ exact")
        for r, e in e_refs:
            ax4a.plot(x_h, e, lw=0.9, alpha=0.65, color=r["style"]["color"],
                      label=f"AMR $-$ {r.get('label', 'single level')}")
    if band is not None:
        ax4a.axvspan(band[0], band[1], color="0.5", alpha=0.10, zorder=0)
    ax4a.set_yscale("symlog", linthresh=1e-3 * dh)
    ax4a.set_xlim(lo[0], hi[0])
    ax4a.set_xlabel("x [m]")
    ax4a.set_ylabel(r"$\Delta h$ [m]")
    ax4a.set_title("Error across the domain (symlog)", fontsize=10)
    ax4a.legend(loc="lower right", fontsize=7)
    ax4a.grid(True, alpha=0.25, which="both")

    # plateau band, linear: this is the resolution the battery cares about.
    if band is not None and len(x_h):
        sel = (x_h > band[0]) & (x_h < band[1])
        if np.count_nonzero(sel) >= 3:
            ax4b.axhline(0.0, color="0.45", lw=0.8, ls=":")
            ax4b.plot(x_h[sel], e_ana[sel], color=C_AMR, lw=1.2, alpha=0.95,
                      label="AMR $-$ exact")
            for r, e in e_refs:
                ax4b.plot(x_h[sel], e[sel], lw=1.2, alpha=0.65,
                          color=r["style"]["color"],
                          label=f"AMR $-$ {r.get('label', 'single level')}")
            pn = norms(x_h[sel], e_ana[sel])
            ax4b.text(0.02, 0.95,
                      "L1 %.2e   L2 %.2e   Linf %.2e"
                      % (pn["L1"], pn["L2"], pn["Linf"]),
                      transform=ax4b.transAxes, fontsize=8, va="top",
                      bbox=dict(boxstyle="round", facecolor="white",
                                edgecolor="0.7", alpha=0.85))
            ax4b.set_xlim(band[0], band[1])
            # The band now runs all the way to the shock elbow, so the O(1e-3)
            # shock spike shares the axes with the O(1e-6) shelf structure:
            # a linear y-axis would flatten exactly the detail we zoomed in for.
            ax4b.set_yscale("symlog", linthresh=1e-4 * dh)
            ax4b.set_title("Shelf → shock elbow (symlog)", fontsize=10)
            ax4b.legend(loc="lower right", fontsize=7)
            ax4b.grid(True, alpha=0.25, which="both")
        else:
            ax4b.text(0.5, 0.5, "band too short", transform=ax4b.transAxes,
                      ha="center", va="center", fontsize=9, color="0.45")
    else:
        ax4b.text(0.5, 0.5, "no intermediate plateau at this time",
                  transform=ax4b.transAxes, ha="center", va="center",
                  fontsize=9, color="0.45")
    ax4b.set_xlabel("x [m]")
    ax4b.set_ylabel(r"$\Delta h$ [m]")
    ax4b.grid(True, alpha=0.25)

    # --- Panel 5: hu profiles ----------------------------------------------
    ax5p = fig.add_subplot(gs[4])
    ax5p.plot(xg, h_ana * u_ana, color=C_ANA, ls="--", lw=2.2, alpha=0.70,
              label="Analytical", zorder=5)
    for r in refs:
        if r.get("hu") is None or len(np.atleast_1d(r["hu"])) < 2:
            continue
        ax5p.plot(r["x"], r["hu"], label=r.get("label", "single level"),
                  zorder=8, **r["style"])
    if len(x_hu) > 0:
        ax5p.plot(x_hu, hu_profile, color=C_AMR, lw=1.0, alpha=0.95,
                  label="AMR", zorder=10)
    ax5p.set_xlim(lo[0], hi[0])
    ax5p.set_xlabel("x [m]")
    ax5p.set_ylabel("hu [m²/s]")
    ax5p.set_title("x-momentum profile: AMR vs single level(s) vs analytical")
    ax5p.legend(loc="upper right", fontsize=8)
    ax5p.grid(True, alpha=0.25)

    # --- Panel 6: metrics ---------------------------------------------------
    ax6 = fig.add_subplot(gs[5])
    ax6.axis("off")
    if metrics_text:
        ax6.text(0.0, 0.55, metrics_text, fontsize=10.5, family="monospace",
                 va="center", ha="left",
                 bbox=dict(boxstyle="round", facecolor="#f2f7ff",
                           edgecolor="#88aadd"))

    fig.subplots_adjust(left=0.08, right=0.90, top=0.965, bottom=0.04)
    fig.savefig(outname, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {outname}")
    return outname


def main():
    parser = argparse.ArgumentParser(
        description="Render one dambreak plotfile (AMR vs single level vs analytical)")
    parser.add_argument("plotfile", help="Path to AMReX native plotfile directory")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--metrics-text", type=str, default=None,
                        help="Metrics block to embed")
    parser.add_argument("--reference", action="append", default=None, metavar="PATH",
                        help="Single-level run or plotfile to compare against; may be "
                             "repeated. Omit to auto-resolve the sibling "
                             "single_level*__<tag> runs; pass 'none' to skip.")
    parser.add_argument("--hl", type=float, default=0.005)
    parser.add_argument("--hr", type=float, default=0.001)
    parser.add_argument("--x0", type=float, default=5.0)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=None,
                        help="Figure DPI (default: HYDROEXA_PLOT_DPI or 400)")
    args = parser.parse_args()

    import yt

    ds = yt.load(args.plotfile)
    t = float(ds.current_time)
    if args.reference:
        if [str(r).lower() for r in args.reference] == ["none"]:
            spec = []
        else:
            spec = [(("stable", "base")[i] if i < 2 else f"ref{i}", p)
                    for i, p in enumerate(args.reference)]
    else:
        spec = [(slot, path) for slot, _, path in default_references(args.plotfile)]

    refs = []
    for slot, path in spec:
        ref = load_reference(path, t, args.hl, args.hr, args.x0, args.L)
        if ref is None:
            continue
        ref["slot"] = slot
        refs.append(ref)
        print(f"reference: {ref['label']} -> {ref['path']}  (t={ref['t']:.6f})")

    out = args.output or os.path.join(
        os.path.dirname(args.plotfile),
        f"frame_t{t:.4f}.png")
    render_single_frame(ds, out, metrics_text=args.metrics_text,
                        hl=args.hl, hr=args.hr, x0=args.x0, L=args.L,
                        dpi=args.dpi, references=refs)


if __name__ == "__main__":
    main()
