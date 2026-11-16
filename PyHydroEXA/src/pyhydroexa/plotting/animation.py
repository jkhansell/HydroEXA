"""Animation / GIF generation from a sequence of plotfiles.

Loads frames lazily to avoid loading all simulation data into memory
simultaneously.
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    imageio = None  # type: ignore
    HAS_IMAGEIO = False

from ..io.plotfile import Plotfile, PlotfileSeries
from .plots import plot_field_with_amr_overlay


def _render_frame(
    pf: Plotfile,
    field: str,
    dpi: int = 150,
    figsize: tuple = (14, 8),
    **kwargs: Any,
) -> bytes:
    """Render one plotfile to raw PNG bytes.

    Parameters
    ----------
    pf :
        A :class:`Plotfile` instance.
    field :
        Logical field name.
    dpi :
        Figures-per-inch.
    figsize :
        Figure size in inches.
    **kwargs :
        Passed to :func:`plots.plot_field_with_amr_overlay`.

    Returns
    -------
    bytes
        Raw PNG data.
    """
    buf = io.BytesIO()
    ax = plot_field_with_amr_overlay(
        pf, field, dpi=dpi, figsize=figsize, return_ax=True, **kwargs
    )
    if ax is not None:
        fig = ax.figure
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
    else:
        plt.close()
    buf.seek(0)
    return buf.read()


def generate_animation(
    series: PlotfileSeries,
    field: str,
    output: str = "animation.gif",
    duration: float = 0.2,
    dpi: int = 150,
    figsize: tuple = (14, 8),
    **plot_kwargs: Any,
) -> str:
    """Generate an animation GIF from a :class:`PlotfileSeries`.

    Frames are generated and collected sequentially.  For very long
    series consider writing frames to disk instead.

    Parameters
    ----------
    series :
        A :class:`PlotfileSeries` instance.
    field :
        Logical field name to animate.
    output :
        Output GIF file path.
    duration :
        Frame duration in seconds.
    dpi :
        Figures-per-inch for each frame.
    figsize :
        Figure size in inches ``(width, height)``.
    **plot_kwargs :
        Additional keyword arguments passed to
        :func:`plots.plot_field_with_amr_overlay`.

    Returns
    -------
    str
        Path to the generated GIF.

    Raises
    ------
    RuntimeError
        If ``imageio`` is not installed.
    """
    if not HAS_IMAGEIO:
        raise RuntimeError("imageio is required for animation generation.")

    frames: list[bytes] = []

    for pf in series:
        buf = _render_frame(
            pf,
            field=field,
            dpi=dpi,
            figsize=figsize,
            **plot_kwargs,
        )
        frames.append(buf)
        print(f"  Frame: {pf.path} (t = {pf.time:.6f})")

    output_path = str(output)
    imageio.mimsave(output_path, frames, duration=duration, loop=0)
    print(f"[SUCCESS] Saved animation: {output_path}")
    return output_path
