Python API --- ``pyhydroexa``
========================================

The whole package is discovered at build time by a recursive
`autosummary <https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html>`_:
one stub page per module is generated into ``docs/api/generated/`` (gitignored),
so there is **no module list to maintain here**.  Splitting
``postprocessing/`` into ``io/`` + ``plotting/``, or moving per-case analysis
code into the package, documents itself.

.. autosummary::
   :toctree: generated
   :recursive:

   pyhydroexa

.. note::

   Modules whose optional dependencies (``yt``, ``rasterio``, ``geopandas``,
   …) are absent from the documentation environment are mocked during the
   build, so the reference always renders.  Install the package for real
   (``uv sync``) to resolve the type links.
