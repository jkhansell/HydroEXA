# API reference

```{toctree}
:maxdepth: 2

python
cpp
```

Two generated references sit side by side in this one tree:

- **{doc}`python`** — `pyhydroexa` and everything under it, walked at build
  time. Move a module inside the package and it is documented; this page never
  changes.
- **{doc}`cpp`** — the solver, generated from the Doxygen XML of `include/`
  and `src/`. Same rule: the Doxyfile names directories, not files.

Where the two meet — plotfile layout, HDF5 input schema, field names, units —
is described in the {doc}`pipeline reference <../PIPELINE>`, the single source
of truth for the numerics.
