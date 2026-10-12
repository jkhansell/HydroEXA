#!/usr/bin/env python3
"""Quick smoke test for pyhydroexa.postprocessing.

Run this first to verify the package imports and basic functionality
work before running the full validation suite.

Usage:
    cd HydroEXACode/PyHydroEXA && uv run python tests/validate_quick.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

results: list[tuple[str, bool, str | None]] = []


def check(label: str, func, *args, **kwargs) -> None:
    try:
        func(*args, **kwargs)
        results.append((label, True, None))
        print(f"  {PASS} {label}")
    except Exception as e:
        results.append((label, False, str(e)))
        print(f"  {FAIL} {label}: {e}")


def main() -> None:
    print("=" * 60)
    print("PyHydroEXA — Quick Smoke Test")
    print("=" * 60)

    # 1. Package import
    print("\n[1] Package imports")
    check("import pyhydroexa", __import__, "pyhydroexa")
    check("from pyhydroexa.postprocessing import Plotfile",
          __import__, "pyhydroexa.postprocessing", fromlist=["Plotfile"])
    check("from pyhydroexa.postprocessing import PlotfileSeries",
          __import__, "pyhydroexa.postprocessing", fromlist=["PlotfileSeries"])
    check("from pyhydroexa.postprocessing import plot_field",
          __import__, "pyhydroexa.postprocessing", fromlist=["plot_field"])
    check("from pyhydroexa.postprocessing import plot_field_with_amr_overlay",
          __import__, "pyhydroexa.postprocessing", fromlist=["plot_field_with_amr_overlay"])
    check("from pyhydroexa.postprocessing import extract_profile",
          __import__, "pyhydroexa.postprocessing", fromlist=["extract_profile"])
    check("from pyhydroexa.postprocessing import generate_animation",
          __import__, "pyhydroexa.postprocessing", fromlist=["generate_animation"])
    check("from pyhydroexa.cli import main",
          __import__, "pyhydroexa.cli", fromlist=["main"])

    # 2. Field aliases
    print("\n[2] Field aliases")
    from pyhydroexa.postprocessing import DEFAULT_ALIASES
    check("DEFAULT_ALIASES has 'h'", lambda: "h" in DEFAULT_ALIASES)
    check("DEFAULT_ALIASES has 'hu'", lambda: "hu" in DEFAULT_ALIASES)
    check("DEFAULT_ALIASES has 'hv'", lambda: "hv" in DEFAULT_ALIASES)
    check("DEFAULT_ALIASES has 'z'", lambda: "z" in DEFAULT_ALIASES)

    # 3. Plotfile discovery (use existing plotfiles)
    print("\n[3] Plotfile discovery")
    noamr_dir = Path(__file__).resolve().parent.parent.parent / "dambreakNoAMR" / "plotfiles"
    amr_dir = Path(__file__).resolve().parent.parent.parent / "dambreakAMR" / "plotfiles"
    circle_dir = Path(__file__).resolve().parent.parent.parent / "dambreakcircle" / "plotfiles"

    if noamr_dir.exists():
        first_noamr = str(noamr_dir / "plt00000")
        check("Plotfile loads NoAMR", Plotfile, first_noamr)
    else:
        print(f"  {SKIP} NoAMR plotfiles not found at {noamr_dir}")

    if amr_dir.exists():
        first_amr = str(amr_dir / "plt00000")
        check("Plotfile loads AMR", Plotfile, first_amr)
    else:
        print(f"  {SKIP} AMR plotfiles not found at {amr_dir}")

    if circle_dir.exists():
        first_circle = str(circle_dir / "plt00000")
        check("Plotfile loads circle", Plotfile, first_circle)
    else:
        print(f"  {SKIP} Circle plotfiles not found at {circle_dir}")

    # 4. PlotfileSeries
    print("\n[4] PlotfileSeries")
    if noamr_dir.exists():
        check("PlotfileSeries discovers NoAMR",
              PlotfileSeries, str(noamr_dir / "plt*"), {"directory": str(noamr_dir)})
    if amr_dir.exists():
        check("PlotfileSeries discovers AMR",
              PlotfileSeries, str(amr_dir / "plt*"), {"directory": str(amr_dir)})

    # 5. Metadata access
    print("\n[5] Metadata access")
    if noamr_dir.exists():
        pf = Plotfile(str(noamr_dir / "plt00000"))
        check("pf.time is float", lambda: isinstance(pf.time, float))
        check("pf.field_names is tuple", lambda: isinstance(pf.field_names, tuple))
        check("pf.num_levels is int", lambda: isinstance(pf.num_levels, int))
        check("pf.domain_lo is tuple", lambda: isinstance(pf.domain_lo, tuple))
        check("pf.domain_hi is tuple", lambda: isinstance(pf.domain_hi, tuple))

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")
    print("=" * 60)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
