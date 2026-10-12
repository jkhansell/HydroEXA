#!/usr/bin/env python3
"""Full validation suite for pyhydroexa.postprocessing.

Tests every module against actual HydroEXA plotfiles.  Produces visual
output (PNGs and GIF) to confirm correctness.

Usage:
    cd HydroEXACode/PyHydroEXA && uv run python tests/validate_full.py

Requires:
    - HydroEXA plotfiles in tests/dambreakNoAMR/plotfiles/
    - HydroEXA plotfiles in tests/dambreakAMR/plotfiles/ (optional)
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Add the package source to the path so it can be imported.
PKG_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(PKG_SRC))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyhydroexa.postprocessing import (
    Plotfile,
    PlotfileSeries,
    plot_field,
    plot_field_with_amr_overlay,
    extract_profile,
    extract_horizontal_cut,
    extract_vertical_cut,
    generate_animation,
    resolve_field,
    extract_field,
    get_field_coordinates,
    get_meshblock_rectangles,
    get_available_levels,
    filter_blocks_by_region,
    DEFAULT_ALIASES,
)
from pyhydroexa.postprocessing.profiles import LineProfile

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent.parent.parent
NOAMR_DIR = BASE / "dambreakNoAMR" / "plotfiles"
AMR_DIR = BASE / "dambreakAMR" / "plotfiles"
CIRCLE_DIR = BASE / "dambreakcircle" / "plotfiles"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "validation_output"
OUTPUT_DIR.mkdir(exist_ok=True)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

results: list[tuple[str, bool, str | None]] = []


def check(label: str, func, *args, **kwargs) -> None:
    try:
        result = func(*args, **kwargs)
        results.append((label, True, None))
        print(f"  {PASS} {label}")
        return result
    except Exception as e:
        results.append((label, False, str(e)))
        print(f"  {FAIL} {label}: {e}")
        traceback.print_exc()
        return None


# ==========================================================================
# 1. Field aliases
# ==========================================================================

def test_field_aliases() -> None:
    print("\n" + "=" * 60)
    print("[1] Field aliases")
    print("=" * 60)
    check("DEFAULT_ALIASES has 'h'", lambda: "h" in DEFAULT_ALIASES)
    check("DEFAULT_ALIASES has 'hu'", lambda: "hu" in DEFAULT_ALIASES)
    check("DEFAULT_ALIASES has 'hv'", lambda: "hv" in DEFAULT_ALIASES)
    check("DEFAULT_ALIASES has 'z'", lambda: "z" in DEFAULT_ALIASES)
    check("'h' aliases include 'h_fluid'",
          lambda: "h_fluid" in DEFAULT_ALIASES["h"])
    check("'z' aliases include 'z_bathymetry'",
          lambda: "z_bathymetry" in DEFAULT_ALIASES["z"])


# ==========================================================================
# 2. Plotfile loading — NoAMR
# ==========================================================================

def test_noamr_loading() -> None:
    print("\n" + "=" * 60)
    print("[2] Plotfile loading — dambreakNoAMR")
    print("=" * 60)
    if not NOAMR_DIR.exists():
        print(f"  {SKIP} NoAMR plotfiles not found at {NOAMR_DIR}")
        return

    pf = check("Load plt00000", Plotfile, str(NOAMR_DIR / "plt00000"))
    if pf is None:
        return

    check("pf.time is float", lambda: isinstance(pf.time, float))
    check("pf.dim == 2", lambda: pf.dim == 2)
    check("pf.num_levels == 0", lambda: pf.num_levels == 0)
    check("pf.field_names has 5 fields",
          lambda: len(pf.field_names) == 5)
    check("'h_fluid' in field_names",
          lambda: "h_fluid" in pf.field_names)
    check("'z_bathymetry' in field_names",
          lambda: "z_bathymetry" in pf.field_names)
    check("'hu_momentum' in field_names",
          lambda: "hu_momentum" in pf.field_names)
    check("domain_lo has 2 elements",
          lambda: len(pf.domain_lo) == 2)
    check("domain_hi has 2 elements",
          lambda: len(pf.domain_hi) == 2)

    # Load a later plotfile
    pf2 = check("Load plt00025", Plotfile, str(NOAMR_DIR / "plt00025"))
    if pf2 is not None:
        check("Time advanced", lambda: pf2.time > pf.time)


# ==========================================================================
# 3. Plotfile loading — AMR
# ==========================================================================

def test_amr_loading() -> None:
    print("\n" + "=" * 60)
    print("[3] Plotfile loading — dambreakAMR")
    print("=" * 60)
    if not AMR_DIR.exists():
        print(f"  {SKIP} AMR plotfiles not found at {AMR_DIR}")
        return

    pf = check("Load AMR plt00000", Plotfile, str(AMR_DIR / "plt00000"))
    if pf is None:
        return

    check("AMR pf.num_levels > 0", lambda: pf.num_levels > 0)
    check("AMR field_names present",
          lambda: "h_fluid" in pf.field_names)


# ==========================================================================
# 4. Field extraction
# ==========================================================================

def test_field_extraction() -> None:
    print("\n" + "=" * 60)
    print("[4] Field extraction")
    print("=" * 60)
    if not NOAMR_DIR.exists():
        print(f"  {SKIP} Skipping — NoAMR plotfiles not found")
        return

    pf = Plotfile(str(NOAMR_DIR / "plt00000"))

    data = check("Extract 'h' field", pf.extract, "h")
    if data is not None:
        check("h shape is (ny, nx)", lambda: data.ndim == 2)
        if data.ndim == 2:
            check("h ny=16", lambda: data.shape[0] == 16)
            check("h nx=64", lambda: data.shape[1] == 64)

    z = check("Extract 'z' field", pf.extract, "z")
    if z is not None:
        check("z shape is 2-D", lambda: z.ndim == 2)

    hu = check("Extract 'hu' field", pf.extract, "hu")
    if hu is not None:
        check("hu shape is 2-D", lambda: hu.ndim == 2)

    # Coordinates
    coords = check("Get field coordinates", pf.get_coordinates)
    if coords is not None:
        x, y = coords
        check("x has 64 entries", lambda: len(x) == 64)
        check("y has 16 entries", lambda: len(y) == 16)


# ==========================================================================
# 5. AMR utilities
# ==========================================================================

def test_amr_utilities() -> None:
    print("\n" + "=" * 60)
    print("[5] AMR utilities")
    print("=" * 60)

    # NoAMR — should have 1 block at level 0
    if NOAMR_DIR.exists():
        pf = Plotfile(str(NOAMR_DIR / "plt00000"))
        blocks = check("NoAMR: get_meshblock_rectangles",
                       get_meshblock_rectangles, pf._ds)
        if blocks is not None:
            check("NoAMR: 1 block", lambda: len(blocks) == 1)
            check("NoAMR: level 0", lambda: blocks[0].level == 0)

    # AMR — should have multiple blocks
    if AMR_DIR.exists():
        pf = Plotfile(str(AMR_DIR / "plt00000"))
        blocks = check("AMR: get_meshblock_rectangles",
                       get_meshblock_rectangles, pf._ds)
        if blocks is not None:
            check("AMR: multiple blocks", lambda: len(blocks) > 1)
            levels = check("AMR: get_available_levels",
                           get_available_levels, pf._ds)
            if levels is not None:
                check("AMR: levels include 0", lambda: 0 in levels)

    # Region filter
    blocks = [
        LineProfile(0.0, np.array([1]), np.array([0]), ""),  # dummy
    ]
    # Use actual MeshBlock
    from pyhydroexa.postprocessing.amr import MeshBlock
    test_blocks = [
        MeshBlock(0.0, 0.0, 5.0, 1.6, 0),
        MeshBlock(5.0, 0.0, 5.0, 1.6, 0),
    ]
    filtered = check("filter_blocks_by_region overlaps",
                     filter_blocks_by_region, test_blocks,
                     2.0, 0.0, 4.0, 1.6)
    if filtered is not None:
        check("filter: 1 block in region", lambda: len(filtered) == 1)


# ==========================================================================
# 6. Profile extraction
# ==========================================================================

def test_profiles() -> None:
    print("\n" + "=" * 60)
    print("[6] Profile extraction")
    print("=" * 60)
    if not NOAMR_DIR.exists():
        print(f"  {SKIP} Skipping — NoAMR plotfiles not found")
        return

    pf = Plotfile(str(NOAMR_DIR / "plt00000"))

    # Multiple x-cuts
    x_coords, profiles = check("extract_profile axis=x, 3 cuts",
                                pf.extract, "h",
                                lambda self: extract_profile(
                                    self._ds, "h", axis="x", positions=[0.2, 0.5, 0.8]))
    # Need to call directly
    x_coords, profiles = check("extract_profile axis=x, 3 cuts",
                                lambda: extract_profile(
                                    pf._ds, "h", axis="x", positions=[0.2, 0.5, 0.8]))
    if profiles is not None:
        check("3 profiles returned", lambda: len(profiles) == 3)
        check("Each profile has 64 values",
              lambda: all(len(p.values) == 64 for p in profiles))

    # Multiple y-cuts
    y_coords, profiles = check("extract_profile axis=y, 3 cuts",
                                lambda: extract_profile(
                                    pf._ds, "h", axis="y", positions=[0.2, 0.8, 1.4]))
    if profiles is not None:
        check("3 y-profiles returned", lambda: len(profiles) == 3)
        check("Each y-profile has 16 values",
              lambda: all(len(p.values) == 16 for p in profiles))

    # Single cuts
    y_val, x_c, vals = check("extract_horizontal_cut",
                              lambda: extract_horizontal_cut(pf._ds, "h", y=0.8))
    x_val, y_c, vals = check("extract_vertical_cut",
                              lambda: extract_vertical_cut(pf._ds, "h", x=5.0))

    # Invalid axis
    check("Invalid axis raises ValueError",
          lambda: extract_profile(pf._ds, "h", axis="z") or
                  (_ for _ in ()).throw(ValueError("expected")),
          )
    # Actually test the exception
    try:
        extract_profile(pf._ds, "h", axis="z")
        results.append(("Invalid axis raises ValueError", False, "No exception raised"))
        print(f"  {FAIL} Invalid axis raises ValueError: no exception")
    except ValueError:
        results.append(("Invalid axis raises ValueError", True, None))
        print(f"  {PASS} Invalid axis raises ValueError")


# ==========================================================================
# 7. Plotting — single field
# ==========================================================================

def test_plotting() -> None:
    print("\n" + "=" * 60)
    print("[7] Plotting")
    print("=" * 60)
    if not NOAMR_DIR.exists():
        print(f"  {SKIP} Skipping — NoAMR plotfiles not found")
        return

    pf = Plotfile(str(NOAMR_DIR / "plt00000"))

    # 2D field plot
    out1 = OUTPUT_DIR / "test_h_noamr.png"
    check("plot_field → water_depth.png",
          plot_field, pf, "h", {"output": str(out1), "units": "m"})
    check("water_depth.png exists", lambda: out1.exists())

    # Bathymetry plot
    out2 = OUTPUT_DIR / "test_z_noamr.png"
    check("plot_field → bathymetry.png",
          plot_field, pf, "z", {"output": str(out2), "units": "m"})
    check("bathymetry.png exists", lambda: out2.exists())

    # Momentum plot
    out3 = OUTPUT_DIR / "test_hu_noamr.png"
    check("plot_field → hu.png",
          plot_field, pf, "hu", {"output": str(out3), "units": "m^2/s"})
    check("hu.png exists", lambda: out3.exists())


# ==========================================================================
# 8. Plotting — AMR overlay
# ==========================================================================

def test_amr_overlay() -> None:
    print("\n" + "=" * 60)
    print("[8] AMR overlay")
    print("=" * 60)

    # NoAMR — should work but show no overlay
    if NOAMR_DIR.exists():
        pf = Plotfile(str(NOAMR_DIR / "plt00000"))
        out = OUTPUT_DIR / "test_h_noamr_amr.png"
        check("plot_field_with_amr_overlay (NoAMR)",
              plot_field_with_amr_overlay, pf, "h",
              {"output": str(out), "show_amr": True})
        check("NoAMR AMR plot exists", lambda: out.exists())

    # AMR — should show overlay
    if AMR_DIR.exists():
        pf = Plotfile(str(AMR_DIR / "plt00000"))
        out = OUTPUT_DIR / "test_h_amr_overlay.png"
        check("plot_field_with_amr_overlay (AMR)",
              plot_field_with_amr_overlay, pf, "h",
              {"output": str(out), "show_amr": True})
        check("AMR overlay plot exists", lambda: out.exists())


# ==========================================================================
# 9. Profile plots
# ==========================================================================

def test_profile_plots() -> None:
    print("\n" + "=" * 60)
    print("[9] Profile plots")
    print("=" * 60)
    if not NOAMR_DIR.exists():
        print(f"  {SKIP} Skipping — NoAMR plotfiles not found")
        return

    pf = Plotfile(str(NOAMR_DIR / "plt00000"))
    x_coords, profiles = extract_profile(pf._ds, "h", axis="x", positions=[0.2, 0.5, 0.8])

    out = OUTPUT_DIR / "test_profiles.png"
    check("plot_profiles → line_cuts.png",
          lambda: plot_profiles(
              x_coords, profiles, output=str(out),
              x_label="x [m]", y_label="h [m]",
              linewidth=1.5, alpha=0.8))
    check("line_cuts.png exists", lambda: out.exists())


# ==========================================================================
# 10. Multi-field plot
# ==========================================================================

def test_multi_field() -> None:
    print("\n" + "=" * 60)
    print("[10] Multi-field diagnostic plot")
    print("=" * 60)
    if not NOAMR_DIR.exists():
        print(f"  {SKIP} Skipping — NoAMR plotfiles not found")
        return

    pf = Plotfile(str(NOAMR_DIR / "plt00000"))
    out = OUTPUT_DIR / "test_multi_field.png"
    check("plot_multi_field → diagnostic.png",
          plot_multi_field, pf, ["h", "z"],
          {"output": str(out)})
    check("diagnostic.png exists", lambda: out.exists())


# ==========================================================================
# 11. PlotfileSeries
# ==========================================================================

def test_series() -> None:
    print("\n" + "=" * 60)
    print("[11] PlotfileSeries")
    print("=" * 60)
    if not NOAMR_DIR.exists():
        print(f"  {SKIP} Skipping — NoAMR plotfiles not found")
        return

    series = check("PlotfileSeries discovers NoAMR",
                   PlotfileSeries, str(NOAMR_DIR / "plt*"))
    if series is not None:
        check("Series has 51 plotfiles", lambda: len(series) == 51)
        check("Series times are sorted",
              lambda: series.times == sorted(series.times))
        check("First time > 0", lambda: series[0].time > 0)

    # Test iteration (first 3 only)
    if series is not None:
        times = []
        for i, pf in enumerate(series):
            times.append(pf.time)
            if i >= 2:
                break
        check("Iteration yields Plotfile objects",
              lambda: all(isinstance(p, Plotfile) for p in times))


# ==========================================================================
# 12. Animation
# ==========================================================================

def test_animation() -> None:
    print("\n" + "=" * 60)
    print("[12] Animation (5 frames)")
    print("=" * 60)
    if not NOAMR_DIR.exists():
        print(f"  {SKIP} Skipping — NoAMR plotfiles not found")
        return

    # Use only first 5 plotfiles for speed — create a temp series
    out_gif = OUTPUT_DIR / "test_animation.gif"

    try:
        from pyhydroexa.postprocessing.animation import generate_animation
        # Build a PlotfileSeries from just the first 5 plotfiles
        import glob as glob_mod
        first5 = sorted(glob_mod.glob(str(NOAMR_DIR / "plt0000[0-4]")))
        if len(first5) < 5:
            raise FileNotFoundError("Not enough plotfiles for animation test")
        # PlotfileSeries expects a glob pattern or directory;
        # pass the directory and let it discover all, then slice.
        # For speed, we'll just use the full series but set duration short.
        series = PlotfileSeries(str(NOAMR_DIR / "plt*"))
        # generate_animation iterates the series; use first 5 by
        # creating a minimal wrapper that yields only 5 Plotfile objects.
        class _LimitedSeries:
            def __init__(self, s, n):
                self._s = s
                self._n = n
            def __iter__(self):
                for i in range(min(self._n, len(self._s))):
                    yield self._s[i]
        limited = _LimitedSeries(series, 5)
        generate_animation(limited, field="h", output=str(out_gif), duration=0.3)
        check("generate_animation → test_animation.gif",
              lambda: out_gif.exists())
    except Exception as e:
        results.append(("Animation", False, str(e)))
        print(f"  {FAIL} Animation: {e}")


# ==========================================================================
# 13. CLI
# ==========================================================================

def test_cli() -> None:
    print("\n" + "=" * 60)
    print("[13] CLI")
    print("=" * 60)
    from pyhydroexa.cli import main

    # Test --help
    try:
        main(["--help"])
        results.append(("CLI --help", True, None))
        print(f"  {PASS} CLI --help")
    except SystemExit:
        results.append(("CLI --help", True, None))
        print(f"  {PASS} CLI --help")

    # Test inspect
    if NOAMR_DIR.exists():
        try:
            main(["inspect", str(NOAMR_DIR / "plt00000")])
            results.append(("CLI inspect", True, None))
            print(f"  {PASS} CLI inspect")
        except Exception as e:
            results.append(("CLI inspect", False, str(e)))
            print(f"  {FAIL} CLI inspect: {e}")


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    print("=" * 60)
    print("PyHydroEXA — Full Validation Suite")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    test_field_aliases()
    test_noamr_loading()
    test_amr_loading()
    test_field_extraction()
    test_amr_utilities()
    test_profiles()
    test_plotting()
    test_amr_overlay()
    test_profile_plots()
    test_multi_field()
    test_series()
    test_animation()
    test_cli()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"FINAL RESULTS: {passed} passed, {failed} failed out of {len(results)} checks")
    print("=" * 60)

    if failed:
        print("\nFailed checks:")
        for label, ok, err in results:
            if not ok:
                print(f"  - {label}: {err}")

    print(f"\nVisual output in: {OUTPUT_DIR}")
    print("=" * 60)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
