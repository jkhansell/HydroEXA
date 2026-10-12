"""Small CLI for generic plotfile inspection and plotting.

Examples
--------
.. code-block:: bash

    pyhydroexa-inspect plt00010
    pyhydroexa-plot plt00010 h --output depth.png
    pyhydroexa-animate "plt*" h --output evolution.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yt
    HAS_YT = True
except ImportError:
    HAS_YT = False


def _inspect(args: argparse.Namespace) -> None:
    """Print metadata for a plotfile."""
    from .plotfile import Plotfile

    pf = Plotfile(args.path)
    print(f"Path       : {pf.path}")
    print(f"Time       : {pf.time:.6f} s")
    print(f"Dimension  : {pf.dim}")
    print(f"Domain     : {pf.domain_lo} -> {pf.domain_hi}")
    print(f"AMR levels : {pf.num_levels}")
    print(f"Fields     : {', '.join(pf.field_names)}")


def _plot(args: argparse.Namespace) -> None:
    """Plot a field from a single plotfile."""
    from .plotfile import Plotfile
    from .plots import plot_field_with_amr_overlay

    pf = Plotfile(args.path)
    kwargs = {}
    if args.amr:
        kwargs["show_amr"] = True
    if args.cmap:
        kwargs["cmap"] = args.cmap
    if args.label:
        kwargs["label"] = args.label
    if args.units:
        kwargs["units"] = args.units

    output = args.output or f"{args.field}_plot.png"
    plot_field_with_amr_overlay(pf, field=args.field, output=output, **kwargs)
    print(f"Saved: {output}")


def _animate(args: argparse.Namespace) -> None:
    """Generate an animation from a plotfile series."""
    from .plotfile import PlotfileSeries
    from .animation import generate_animation

    series = PlotfileSeries(args.pattern)
    kwargs = {}
    if args.cmap:
        kwargs["cmap"] = args.cmap
    if args.duration:
        kwargs["duration"] = args.duration

    output = args.output or "animation.gif"
    generate_animation(series, field=args.field, output=output, **kwargs)


def _list_fields(args: argparse.Namespace) -> None:
    """List available fields in a plotfile."""
    from .plotfile import Plotfile

    pf = Plotfile(args.path)
    for name in pf.field_names:
        print(name)


def _preprocess(args: argparse.Namespace) -> None:
    """Run the preprocessing pipeline."""
    fields: dict[str, str] = {}
    if args.field:
        for spec in args.field:
            if ":" not in spec:
                print(f"Error: field spec must be NAME:PATH, got '{spec}'", file=sys.stderr)
                sys.exit(1)
            name, path = spec.split(":", 1)
            fields[name] = path

    domain = None
    if args.domain:
        domain = tuple(args.domain)

    try:
        from .preprocessing import preprocess
        preprocess(
            input_file=args.input_file,
            output_file=args.output_file,
            fields=fields,
            crs=args.crs,
            domain=domain,
            cell_size=args.cell_size,
            verbose=args.verbose,
        )
    except ValueError as e:
        print(f"Preprocessing failed: {e}", file=sys.stderr)
        sys.exit(1)


def _input_info(args: argparse.Namespace) -> None:
    """Show parsed HydroEXA input file parameters."""
    from .preprocessing import parse_input_file

    parsed = parse_input_file(args.input_file)

    print(f"Input file: {args.input_file}")
    print()

    if parsed.hydroexa:
        print("HydroEXA.* parameters:")
        for k, v in parsed.hydroexa.items():
            print(f"  {k}: {v}")
        print()

    if parsed.amr:
        print("amr.* parameters:")
        for k, v in parsed.amr.items():
            print(f"  {k}: {v}")
        print()

    if parsed.io:
        print("io.* parameters:")
        for k, v in parsed.io.items():
            print(f"  {k}: {v}")
        print()

    if not parsed.hydroexa and not parsed.amr and not parsed.io:
        print("No recognised parameters found.")
        print(f"Raw keys: {list(parsed.raw.keys())}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pyhydroexa",
        description="HydroEXA postprocessing CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Show plotfile metadata")
    p_inspect.add_argument("path", help="Plotfile path")
    p_inspect.set_defaults(handler=_inspect)

    # plot
    p_plot = sub.add_parser("plot", help="Plot a field")
    p_plot.add_argument("path", help="Plotfile path")
    p_plot.add_argument("field", help="Field name")
    p_plot.add_argument("--output", "-o", help="Output file")
    p_plot.add_argument("--amr", action="store_true", help="Show AMR overlay")
    p_plot.add_argument("--cmap", help="Colormap name")
    p_plot.add_argument("--label", help="Colorbar label")
    p_plot.add_argument("--units", help="Unit string")
    p_plot.set_defaults(handler=_plot)

    # animate
    p_anim = sub.add_parser("animate", help="Generate animation from series")
    p_anim.add_argument("pattern", help="Glob pattern for plotfiles")
    p_anim.add_argument("field", help="Field name")
    p_anim.add_argument("--output", "-o", help="Output GIF")
    p_anim.add_argument("--duration", type=float, help="Frame duration (s)")
    p_anim.add_argument("--cmap", help="Colormap name")
    p_anim.set_defaults(handler=_animate)

    # list-fields
    p_list = sub.add_parser("list-fields", help="List available fields")
    p_list.add_argument("path", help="Plotfile path")
    p_list.set_defaults(handler=_list_fields)

    # preprocess
    p_prep = sub.add_parser(
        "preprocess",
        help="Convert inputs + physical fields to HDF5",
    )
    p_prep.add_argument("input_file", help="HydroEXA input file (.in)")
    p_prep.add_argument("output_file", help="Output HDF5 file")
    p_prep.add_argument(
        "--field",
        "-f",
        action="append",
        metavar="NAME:PATH",
        help="Physical field: --field z=bathymetry.tif --field h=depth.tif",
    )
    p_prep.add_argument("--crs", help="CRS string (e.g. EPSG:4326)")
    p_prep.add_argument(
        "--domain",
        nargs=4,
        type=float,
        metavar=("XLO", "YLO", "XHI", "YHI"),
        help="Physical domain override",
    )
    p_prep.add_argument(
        "--cell-size",
        type=float,
        help="Explicit cell size (meters)",
    )
    p_prep.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress messages",
    )
    p_prep.set_defaults(handler=_preprocess)

    # input-info
    p_info = sub.add_parser(
        "input-info",
        help="Show parsed HydroEXA input file parameters",
    )
    p_info.add_argument("input_file", help="HydroEXA input file (.in)")
    p_info.set_defaults(handler=_input_info)

    parsed = parser.parse_args(argv)
    if parsed.command is None:
        parser.print_help()
        sys.exit(0)

    parsed.handler(parsed)


if __name__ == "__main__":
    main()
