"""Reusable plotting functions for HydroEXA simulation data.

All functions work with arbitrary scalar fields — water depth, momentum,
bathymetry, velocity-derived quantities, or any future HydroEXA output.

Design principles
-----------------
* Plot physical coordinates, not array indices.
* Preserve domain aspect ratio correctly.
* Accept field names, labels, limits, colormaps, figure sizes, and
  output paths as arguments — no hard-coded values.
* Never embed validation-case analytical solutions.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


def _ensure_agg() -> None:
    """Switch to the Agg backend if no interactive backend is active."""
    if matplotlib.get_backend().lower() == "agg":
        return
    # Only set if the user hasn't already configured a backend.
    if hasattr(plt, "get_backend") and plt.get_backend().lower() != "agg":
        pass  # Let the user's backend ride.


_ensure_agg()


# ---------------------------------------------------------------------------
# 2-D scalar field plot
# ---------------------------------------------------------------------------

def plot_field(
    pf: "Plotfile",
    field: str,
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
    norm: Optional[Normalize] = None,
    label: Optional[str] = None,
    units: str = "",
    extent: Optional[Tuple[float, float, float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
    colorbar_orientation: str = "horizontal",
    dpi: int = 150,
    figsize: Tuple[float, float] = (10, 8),
    output: Optional[str] = None,
    aspect: str = "equal",
    return_ax: bool = False,
    **kwargs: Any,
) -> Optional[Axes]:
    """Plot a 2-D scalar field from a plotfile.

    Parameters
    ----------
    pf :
        A :class:`Plotfile` instance.
    field :
        Logical field name.
    ax :
        Existing Matplotlib axis to plot into.  If ``None``, a new
        figure is created.
    cmap :
        Matplotlib colormap name.
    norm :
        Normalization object.  If given, *vmin* and *vmax* are ignored.
    label :
        Colorbar label.  Defaults to ``"<field> [<units>]"``.
    units :
        Unit string for the colorbar label.
    extent :
        ``[x_lo, x_hi, y_lo, y_hi]`` for ``imshow``.  Defaults to the
        domain bounds.
    vmin, vmax :
        Color scale limits.
    colorbar :
        Whether to draw a colorbar.
    colorbar_orientation :
        ``"horizontal"`` or ``"vertical"``.
    dpi :
        Figures-per-inch (used when creating a new figure).
    figsize :
        Figure size in inches.
    output :
        If given, save the figure to this path.
    aspect :
        ``"equal"`` or ``"auto"``.
    return_ax :
        If ``True``, return the :class:`Axes` instead of ``None``.
    **kwargs :
        Passed to :func:`extract_field`.

    Returns
    -------
    Axes or None
        The axis used for plotting, or ``None`` (when *return_ax* is
        ``False``).
    """
    from .fields import extract_field, get_field_coordinates

    data = pf.extract(field, **kwargs)
    x, y, x_lo, x_hi, y_lo, y_hi = get_field_coordinates(pf._ds, level=kwargs.get("level", 0))

    if extent is None:
        extent = [x_lo, x_hi, y_lo, y_hi]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect=aspect,
    )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(f"x [{units}]")
    ax.set_ylabel(f"y [{units}]")

    if colorbar:
        cbar_label = label or f"{field}"
        if units:
            cbar_label += f" [{units}]"
        cbar = plt.colorbar(
            im, ax=ax, orientation=colorbar_orientation,
            fraction=0.046 if colorbar_orientation == "horizontal" else 0.15,
            pad=0.04,
        )
        cbar.set_label(cbar_label)

    if output:
        fig.savefig(output, dpi=dpi, facecolor="white")
        plt.close(fig)
    elif return_ax:
        return ax

    return None


# ---------------------------------------------------------------------------
# 2-D field with AMR MeshBlock overlay
# ---------------------------------------------------------------------------

def plot_field_with_amr_overlay(
    pf: "Plotfile",
    field: str,
    show_amr: bool = True,
    amr_level_colors: Optional[List[str]] = None,
    amr_linewidth: float = 1.0,
    amr_alpha: float = 0.6,
    cmap: str = "viridis",
    label: Optional[str] = None,
    units: str = "",
    colorbar: bool = True,
    colorbar_orientation: str = "horizontal",
    dpi: int = 150,
    figsize: Tuple[float, float] = (14, 8),
    output: Optional[str] = None,
    return_ax: bool = False,
    level: int = 0,
    **kwargs: Any,
) -> Optional[Axes]:
    """Plot a 2-D field with AMR MeshBlock boundaries overlaid.

    Parameters
    ----------
    pf :
        A :class:`Plotfile` instance.
    field :
        Logical field name.
    show_amr :
        Whether to draw MeshBlock outlines.
    amr_level_colors :
        Per-level colours (e.g. ``["red", "green", "blue"]``).
        Defaults to ``tab20``.
    amr_linewidth :
        Line width for MeshBlock outlines.
    amr_alpha :
        Alpha transparency for MeshBlock outlines.
    **kwargs :
        Passed to :func:`plot_field`.

    Returns
    -------
    Axes or None
    """
    from .amr import get_meshblock_rectangles

    ax = plot_field(
        pf,
        field,
        cmap=cmap,
        label=label,
        units=units,
        colorbar=colorbar,
        colorbar_orientation=colorbar_orientation,
        dpi=dpi,
        figsize=figsize,
        output=None,
        return_ax=True,
        level=level,
        **kwargs,
    )

    if ax is None:
        ax = plt.gca()

    if show_amr and pf.num_levels > 0:
        rects = get_meshblock_rectangles(pf._ds)
        if amr_level_colors is None:
            levels = sorted(set(r.level for r in rects)) if rects else [0]
            amr_level_colors = plt.cm.tab20(
                np.linspace(0, 1, max(len(levels), 1))
            ).tolist()
        level_colors = {}
        for r in rects:
            if r.level not in level_colors:
                level_colors[r.level] = amr_level_colors[
                    len(level_colors) % len(amr_level_colors)
                ]
        for rect in rects:
            ax.add_patch(Rectangle(
                (rect.x0, rect.y0),
                rect.width,
                rect.height,
                fill=False,
                edgecolor=level_colors[rect.level],
                linewidth=amr_linewidth,
                alpha=amr_alpha,
                zorder=10,
            ))

    if output:
        fig = ax.figure
        fig.savefig(output, dpi=dpi, facecolor="white")
        plt.close(fig)
    elif return_ax:
        return ax

    return None


# ---------------------------------------------------------------------------
# 1-D profile plot
# ---------------------------------------------------------------------------

def plot_profiles(
    axis_coords: np.ndarray,
    profiles: List[Any],
    ax: Optional[Axes] = None,
    label_template: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    colorbar: bool = False,
    dpi: int = 150,
    figsize: Tuple[float, float] = (12, 6),
    output: Optional[str] = None,
    return_ax: bool = False,
    **line_kwargs: Any,
) -> Optional[Axes]:
    """Plot 1-D profiles / line cuts.

    Parameters
    ----------
    axis_coords :
        Coordinates along the cut direction.
    profiles :
        List of :class:`LineProfile` objects or tuples of
        ``(coordinate, values, label)``.
    ax :
        Existing axis to plot into.
    label_template :
        f-string for auto-labels, e.g. ``"y = {coord:.3f}"``.
    x_label, y_label :
        Axis labels.
    xlim, ylim :
        Axis limits.
    dpi :
        Figures-per-inch.
    figsize :
        Figure size in inches.
    output :
        Save path.
    return_ax :
        Return the :class:`Axes`.
    **line_kwargs :
        Passed to ``ax.plot()``.

    Returns
    -------
    Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for prof in profiles:
        if hasattr(prof, "values"):
            # LineProfile dataclass.
            coord = prof.coordinate
            vals = prof.values
            coords = prof.axis_coords
            lbl = prof.label or (label_template.format(coord=coord) if label_template else f"y = {coord:.3g}")
        else:
            coord, vals, lbl = prof
            coords = axis_coords
        ax.plot(coords, vals, **line_kwargs, label=lbl)

    ax.set_xlabel(x_label or f"x [{axis_coords.unit if hasattr(axis_coords, 'unit') else ''}]")
    ax.set_ylabel(y_label or "value")

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if len(profiles) > 1:
        ax.legend(loc="best", ncol=min(3, len(profiles)), fontsize=9)
    ax.grid(True, alpha=0.25)

    if output:
        fig.savefig(output, dpi=dpi, facecolor="white")
        plt.close(fig)
    elif return_ax:
        return ax

    return None


# ---------------------------------------------------------------------------
# Multi-field diagnostic figure
# ---------------------------------------------------------------------------

def plot_multi_field(
    pf: "Plotfile",
    fields: List[str],
    layout: Optional[Tuple[int, int]] = None,
    share_axes: bool = True,
    dpi: int = 150,
    figsize: Tuple[float, float] = (16, 10),
    output: Optional[str] = None,
    **field_kwargs: Any,
) -> List[Axes]:
    """Create a multi-panel diagnostic figure, one panel per field.

    Parameters
    ----------
    pf :
        A :class:`Plotfile` instance.
    fields :
        List of logical field names.
    layout :
        ``(nrows, ncols)`` for the subplot grid.  Defaults to a single
        column.
    share_axes :
        Whether panels share x and y axes.
    dpi :
        Figures-per-inch.
    figsize :
        Total figure size.
    output :
        Save path.
    **field_kwargs :
        Passed to :func:`plot_field` for each panel.

    Returns
    -------
    list[Axes]
    """
    if layout is None:
        layout = (len(fields), 1)

    nrows, ncols = layout
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        dpi=dpi,
        sharex=share_axes,
        sharey=share_axes,
        squeeze=False,
    )
    axes_flat = axes.flatten() if nrows > 1 or ncols > 1 else [axes[0, 0]]

    for i, field in enumerate(fields):
        plot_field(pf, field, ax=axes_flat[i], return_ax=True, **field_kwargs)
        axes_flat[i].set_title(f"{field}")

    # Hide unused subplots.
    for i in range(len(fields), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(f"t = {pf.time:.6f} s", fontsize=12)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=dpi, facecolor="white")
        plt.close(fig)

    return axes_flat[:len(fields)]
