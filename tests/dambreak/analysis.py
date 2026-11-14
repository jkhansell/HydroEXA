#!/usr/bin/env python3
"""
analysis.py — ONE entry point for the dambreak battery:
computes the CI metrics (analyze.py) AND renders the plot_output-style
figure (plot.py), with the metrics embedded in the figure.

Output naming uses the ACTUAL simulation time read from the plotfile
(plotfile sequence numbers do NOT map linearly to time — io.plot_freq is a
time interval and .old rotation files can shift numbering).

Usage:
  uv run python analysis.py outputs/roe/amr__r27/plt00006 \
      [--output analysis.png] [--hl 0.005 --hr 0.001 --x0 5.0 --L 10.0]

If --output is omitted the PNG is written next to the plotfile as
analysis_t<t>.png with the ACTUAL time.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import composite_profile, all_metrics, metrics_summary
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
    args = parser.parse_args()

    ds = yt.load(args.plotfile)
    t = float(ds.current_time)
    lo, _ = ds.domain_left_edge.to_ndarray(), ds.domain_right_edge.to_ndarray()
    lo = tuple(float(v) for v in lo)

    h_field = None
    from analyze import find_variable
    h_field = find_variable(ds, "h")
    if h_field is None:
        raise RuntimeError("Could not find water-depth variable.")

    y_center = 0.5 * (lo[1] + float(ds.domain_right_edge.to_ndarray()[1]))
    x, h, _ = composite_profile(ds, h_field, y_center)
    if len(x) == 0:
        raise RuntimeError("Empty centerline profile.")

    m = all_metrics(x, h, t, args.L, args.hl, args.hr, args.x0)
    txt = metrics_summary(m)
    print(f"time = {t:.6f}")
    print(txt)

    out = args.output or os.path.join(
        os.path.dirname(args.plotfile), f"analysis_t{t:.4f}.png")
    render_single_frame(ds, out, metrics_text=f"t = {t:.4f}\n\n{txt}",
                        hl=args.hl, hr=args.hr, x0=args.x0, L=args.L)


if __name__ == "__main__":
    main()
