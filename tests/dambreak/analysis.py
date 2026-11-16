#!/usr/bin/env python3
"""
analysis.py — ONE entry point for the dambreak battery:
computes the CI metrics (analyze.py) AND renders the plot_output-style
figure (plot.py), with the metrics embedded in the figure.

Two single-level yardsticks are resolved automatically from the sibling run
directories (stable first):

  single_level_fine__<tag>  uniform run at the AMR level-1 cell size (dx0/2)
                            -> the STABLE solution AMR should reproduce
  single_level__<tag>       unrefined level-0 run -> what AMR must beat

Output naming uses the ACTUAL simulation time read from the plotfile
(plotfile sequence numbers do NOT map linearly to time — io.plot_freq is a
time interval and .old rotation files can shift numbering).

Usage:
  uv run python analysis.py outputs/roe/amr__n1/plt00006 \
      [--output analysis.png] [--hl 0.005 --hr 0.001 --x0 5.0 --L 10.0]

If --output is omitted the PNG is written next to the plotfile as
analysis_t<t>.png with the ACTUAL time.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import (composite_profile, all_metrics, metrics_summary,
                     find_variable, default_references, load_reference)
from plot import render_single_frame

import yt


def main():
    parser = argparse.ArgumentParser(description="Dambreak analysis: metrics + figure in one shot")
    parser.add_argument("plotfile", help="Path to AMReX native plotfile directory")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--hl", type=float, default=0.005)
    parser.add_argument("--hr", type=float, default=0.001)
    parser.add_argument("--x0", type=float, default=5.0)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--reference", action="append", default=None, metavar="PATH",
                        help="Single-level run or plotfile to compare against; may be "
                             "repeated, first one is the stable yardstick. Omit to "
                             "auto-resolve the sibling single_level*__<tag> runs; "
                             "pass 'none' to skip.")
    parser.add_argument("--dpi", type=int, default=None,
                        help="Figure DPI (default: HYDROEXA_PLOT_DPI or 400)")
    args = parser.parse_args()

    ds = yt.load(args.plotfile)
    t = float(ds.current_time)
    lo = tuple(float(v) for v in ds.domain_left_edge.to_ndarray())

    h_field = find_variable(ds, "h")
    if h_field is None:
        raise RuntimeError("Could not find water-depth variable.")

    y_center = 0.5 * (lo[1] + float(ds.domain_right_edge.to_ndarray()[1]))
    x, h, _ = composite_profile(ds, h_field, y_center)
    if len(x) == 0:
        raise RuntimeError("Empty centerline profile.")

    # Resolve the yardsticks: explicit --reference (in given order) wins,
    # otherwise the sibling single_level_fine / single_level runs.
    if args.reference:
        if [str(r).lower() for r in args.reference] == ["none"]:
            refs_spec = []
        else:
            labels = {"stable": "single level dx0/2", "base": "single level dx0"}
            refs_spec = [(("stable", "base")[i] if i < 2 else f"ref{i}",
                          labels.get(("stable", "base")[i], f"reference {i}") if i < 2
                          else f"reference {i}", p)
                         for i, p in enumerate(args.reference)]
    else:
        refs_spec = default_references(args.plotfile)

    refs, labels, render_refs = {}, {}, []
    for slot, label, path in refs_spec:
        ref = load_reference(path, t, args.hl, args.hr, args.x0, args.L, label=label)
        if ref is None:
            print(f"reference: {label} not found — skipped")
            continue
        refs[slot] = (ref["x"], ref["h"], ref["L1"], ref["L2"], ref["Linf"])
        labels[slot] = label
        ref["slot"] = slot
        render_refs.append(ref)
        print(f"reference: {label} -> {ref['path']}  (t={ref['t']:.6f})")

    m = all_metrics(x, h, t, args.L, args.hl, args.hr, args.x0, refs=refs)
    txt = metrics_summary(m, ref_labels=labels)
    print(f"time = {t:.6f}")
    print(txt)

    out = args.output or os.path.join(
        os.path.dirname(args.plotfile), f"analysis_t{t:.4f}.png")
    render_single_frame(ds, out, metrics_text=f"t = {t:.4f}\n\n{txt}",
                        hl=args.hl, hr=args.hr, x0=args.x0, L=args.L,
                        dpi=args.dpi, references=render_refs)


if __name__ == "__main__":
    main()
