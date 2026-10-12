"""PyHydroEXA — Python tools for HydroEXA simulation data."""

from .cli import main


def main() -> None:
    """Entry-point stub — delegates to :func:`cli.main`."""
    from .cli import main as _cli_main
    _cli_main()
