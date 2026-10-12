"""Reusable abstraction for loading HydroEXA / AMReX plotfiles.

Wraps ``yt`` to expose domain metadata, available fields, AMR levels,
and grid information without assuming any particular simulation.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Set

try:
    import yt
    HAS_YT = True
except ImportError:
    yt = None  # type: ignore
    HAS_YT = False


# ---------------------------------------------------------------------------
# Natural-sort helper
# ---------------------------------------------------------------------------

def _natural_sort_key(s: str) -> List[Any]:
    """Return a sort key that interleaves numeric and string parts."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


# ---------------------------------------------------------------------------
# Plotfile metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlotfileMetadata:
    """Immutable metadata extracted from a single plotfile.

    Attributes
    ----------
    path :
        Resolved path to the plotfile directory.
    time :
        Simulation time (seconds).
    dim :
        Spatial dimensionality (1, 2, or 3).
    domain_lo :
        Lower corner of the computational domain.
    domain_hi :
        Upper corner of the computational domain.
    field_names :
        Logical field names available in the plotfile.
    num_levels :
        Number of AMR levels (0 means single-level).
    """

    path: str
    time: float
    dim: int
    domain_lo: tuple
    domain_hi: tuple
    field_names: tuple
    num_levels: int


def _is_plotfile(path: str) -> bool:
    """Return ``True`` if *path* looks like an AMReX plotfile directory."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "Header"))


# ---------------------------------------------------------------------------
# Plotfile — single dataset
# ---------------------------------------------------------------------------

class Plotfile:
    """Lazy wrapper around a single ``yt`` AMReX plotfile.

    The ``yt`` dataset is not created until properties or methods are
    called, keeping initial overhead minimal.

    Examples
    --------
    >>> pf = Plotfile("plt00010")
    >>> print(pf.time)
    >>> print(pf.field_names)
    >>> data = pf.extract("h")
    """

    def __init__(self, path: str) -> None:
        """
        Parameters
        ----------
        path :
            Path to the plotfile directory (must contain a ``Header`` file).

        Raises
        ------
        FileNotFoundError
            If *path* does not point to a valid plotfile.
        RuntimeError
            If ``yt`` is not installed.
        """
        if not HAS_YT:
            raise RuntimeError("yt is required to load plotfiles.")

        path = os.path.abspath(path)
        if not _is_plotfile(path):
            raise FileNotFoundError(
                f"Not a valid plotfile: '{path}' "
                "(expected a directory containing a 'Header' file)."
            )

        self._path: str = path
        self._ds: Optional["yt.Dataset"] = None
        self._metadata: Optional[PlotfileMetadata] = None

    # -- public properties --------------------------------------------------

    @property
    def path(self) -> str:
        """Resolved path to the plotfile directory."""
        return self._path

    @property
    def time(self) -> float:
        """Simulation time in seconds."""
        return self._metadata.time

    @property
    def dim(self) -> int:
        """Spatial dimensionality."""
        return self._metadata.dim

    @property
    def domain_lo(self) -> tuple:
        """Lower corner of the computational domain."""
        return self._metadata.domain_lo

    @property
    def domain_hi(self) -> tuple:
        """Upper corner of the computational domain."""
        return self._metadata.domain_hi

    @property
    def field_names(self) -> tuple:
        """Logical field names available in the plotfile."""
        return self._metadata.field_names

    @property
    def num_levels(self) -> int:
        """Number of AMR levels (0 means single-level / root only)."""
        return self._metadata.num_levels

    @property
    def domain_dimensions(self) -> tuple:
        """Grid dimensions at each AMR level."""
        return tuple(self._ds.domain_dimensions)

    @property
    def _ds(self) -> "yt.Dataset":
        if self._ds is None:
            self._ds = yt.load(self._path)
            self._metadata = PlotfileMetadata(
                path=self._path,
                time=float(self._ds.current_time),
                dim=int(self._ds.dimensionality),
                domain_lo=tuple(
                    float(v) for v in self._ds.domain_left_edge
                ),
                domain_hi=tuple(
                    float(v) for v in self._ds.domain_right_edge
                ),
                field_names=tuple(
                    f[1] if len(f) > 1 else str(f)
                    for f in sorted(self._ds.field_list)
                ),
                num_levels=int(self._ds.num_levels),
            )
        return self._ds

    # -- methods ------------------------------------------------------------

    def extract(self, field_name: str, level: int = 0) -> "np.ndarray":
        """Extract a field as a NumPy array (delegates to :mod:`fields`)."""
        from .fields import extract_field
        return extract_field(self._ds, field_name, level=level)

    def get_coordinates(self, level: int = 0) -> "np.ndarray":
        """Return cell-centred coordinate arrays (delegates to :mod:`fields`)."""
        from .fields import get_field_coordinates
        x, y, _, _, _, _ = get_field_coordinates(self._ds, level=level)
        return x, y

    def __repr__(self) -> str:
        return (
            f"Plotfile(path='{self._path}', time={self.time:.6f}, "
            f"fields={self._metadata.field_names})"
        )


# ---------------------------------------------------------------------------
# PlotfileSeries — ordered sequence of plotfiles
# ---------------------------------------------------------------------------

def _discover_plotfiles(
    pattern: str,
    directory: Optional[str] = None,
) -> List[str]:
    """Discover and naturally-sort plotfile directories.

    Parameters
    ----------
    pattern :
        A glob pattern (e.g. ``"plt*"``, ``"plt?????"``) or an explicit
        path to a single plotfile directory.  If the pattern already
        points to a valid plotfile directory, it is returned as-is.
    directory :
        Optional base directory to search when *pattern* is a bare glob.

    Returns
    -------
    list[str]
        Sorted list of plotfile directory paths.
    """
    pattern = str(pattern)

    # If pattern is already a valid plotfile dir, return it.
    if _is_plotfile(pattern):
        return [os.path.abspath(pattern)]

    # Try globbing directly.
    candidates = fnmatch.filter(
        sorted(os.listdir(".")), pattern
    ) if directory is None else fnmatch.filter(
        sorted(os.listdir(directory)), pattern
    )

    dirs: List[str] = []
    base = directory or "."
    for name in candidates:
        full = os.path.join(base, name) if directory else name
        if _is_plotfile(full):
            dirs.append(os.path.abspath(full))

    dirs.sort(key=_natural_sort_key)
    return dirs


class PlotfileSeries:
    """Ordered sequence of plotfiles discovered from a glob pattern.

    Supports iteration, indexing, and animation generation.  Plotfiles
    are loaded lazily — only one is open at a time by default.

    Examples
    --------
    >>> series = PlotfileSeries("plt*")
    >>> for pf in series:
    ...     print(pf.time)
    >>> series.animate(field="h", output="evolution.gif")
    """

    def __init__(
        self,
        pattern: str,
        directory: Optional[str] = None,
        cache: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        pattern :
            Glob pattern or path to a single plotfile directory.
        directory :
            Base directory for glob discovery.
        cache :
            If ``True``, cache all loaded :class:`Plotfile` objects in
            memory (convenient but memory-intensive).
        """
        self._paths = _discover_plotfiles(pattern, directory=directory)
        if not self._paths:
            raise FileNotFoundError(
                f"No plotfiles found matching '{pattern}'"
                + (f" in '{directory}'" if directory else "")
            )
        self._cache = cache
        self._cache_dict: dict[int, Plotfile] = {}

    # -- sequence protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> Plotfile:
        if self._cache and index in self._cache_dict:
            return self._cache_dict[index]
        pf = Plotfile(self._paths[index])
        if self._cache:
            self._cache_dict[index] = pf
        return pf

    def __iter__(self) -> Iterator[Plotfile]:
        for i in range(len(self)):
            yield self[i]

    # -- utilities ----------------------------------------------------------

    @property
    def paths(self) -> List[str]:
        """Resolved plotfile paths in natural-sorted order."""
        return list(self._paths)

    @property
    def times(self) -> List[float]:
        """Simulation times for all plotfiles."""
        return [pf.time for pf in self]

    def animate(
        self,
        field: str,
        output: str = "animation.gif",
        duration: float = 0.2,
        **plot_kwargs: Any,
    ) -> str:
        """Generate an animation GIF from the sequence.

        Parameters
        ----------
        field :
            Logical field name to animate.
        output :
            Output file path.
        duration :
            Frame duration in seconds.
        **plot_kwargs :
            Additional keyword arguments passed to
            :func:`plot_field_with_amr_overlay`.

        Returns
        -------
        str
            Path to the generated GIF.
        """
        from .animation import generate_animation
        return generate_animation(
            self, field, output=output, duration=duration, **plot_kwargs
        )

    def __repr__(self) -> str:
        return (
            f"PlotfileSeries(paths={len(self._paths)}, "
            f"times=[{self._paths[0]} ... {self._paths[-1]}])"
        )
