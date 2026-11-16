"""Shared test doubles for the pyhydroexa suite.

The reader tests never touch a real plotfile: they hand the library a fake
``yt.Dataset`` whose only job is to answer the handful of attribute lookups
that :mod:`pyhydroexa.io.fields` and :mod:`pyhydroexa.io.amr` perform.

Two details about the real ``yt`` API are baked into the doubles on purpose,
because getting them wrong is exactly how the reader broke in the first place:

* a BoxLib covering grid returns ``(nx, ny, nz)``, not ``(ny, nx)`` — the
  transpose to image order happens in :func:`pyhydroexa.io.fields.extract_field`;
* domain edges are ``YTArray`` objects, so code must call ``to_ndarray()``
  rather than treat them as plain arrays.
"""

from __future__ import annotations

import numpy as np

from unittest.mock import MagicMock


class ArrayLike:
    """Stand-in for ``yt``'s ``YTArray``: only ``to_ndarray()`` is used."""

    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=float)

    def to_ndarray(self) -> np.ndarray:
        """Return the wrapped plain array."""
        return self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]


class _FieldLike:
    """What ``covering_grid[field_tuple]`` returns: something to_ndarray-able."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def to_ndarray(self) -> np.ndarray:
        """Return the field data."""
        return self._data


class CoveringGrid:
    """Fake ``ds.covering_grid(...)`` result: any field yields the same data."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def __getitem__(self, key) -> _FieldLike:
        return _FieldLike(self._data)


def make_mock_ds(
    field_names: list[str] | None = None,
    nx: int = 64,
    ny: int = 16,
) -> MagicMock:
    """Build a minimal mock ``yt.Dataset`` on a 10 x 1.6 domain.

    Parameters
    ----------
    field_names :
        Logical field names to advertise (as ``("boxlib", name)`` tuples).
    nx, ny :
        Root-grid dimensions.

    Returns
    -------
    MagicMock
        A dataset whose ``covering_grid`` yields an ``(nx, ny)`` array, i.e.
        the layout a real BoxLib covering grid produces.
    """
    if field_names is None:
        field_names = ["h_fluid", "hu_momentum", "hv_momentum", "z_bathymetry"]

    ds = MagicMock()
    ds.field_list = [("boxlib", name) for name in field_names]
    ds.domain_left_edge = ArrayLike([0.0, 0.0, 0.0])
    ds.domain_right_edge = ArrayLike([10.0, 1.6, 0.0])
    ds.domain_dimensions = [nx, ny, 1]
    ds.current_time = 0.15625
    ds.dimensionality = 2

    ds.index.grid_left_edge = np.array([[0.0, 0.0]])
    ds.index.grid_dimensions = np.array([[nx, ny]])
    ds.index.grid_levels = np.array([0])
    ds.index.level_dds = {0: np.array([10.0 / nx, 1.6 / ny, 0.0])}

    # Distinct values per cell, so a transposed read is detectable.
    data = np.arange(nx * ny, dtype=float).reshape(nx, ny)
    ds.covering_grid.return_value = CoveringGrid(data)
    return ds
