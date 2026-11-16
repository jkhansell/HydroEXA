# PyHydroEXA

Python companion package for the HydroEXA shallow-water solver.

## Quick start

```python
from pyhydroexa.io import Plotfile, PlotfileSeries
from pyhydroexa.plotting import plot_field_with_amr_overlay, generate_animation

# Single plotfile
pf = Plotfile("plt00010")
print(pf.field_names)
plot_field_with_amr_overlay(pf, field="h", output="water_depth.png")

# Series
series = PlotfileSeries("plt*")
for frame in series:
    print(frame.time)

generate_animation(series, field="h", output="evolution.gif")
```

## CLI

```bash
pyhydroexa-inspect plt00010
pyhydroexa-plot plt00010 h --amr --output depth.png
pyhydroexa-animate "plt*" h --output evolution.gif
pyhydroexa-audit --top 5          # measure the source tree itself
```

## Modules

The package is split by responsibility, so importing one layer does not
drag the others in. Reading a plotfile must work on a machine with no
plotting stack; drawing must not need to know how a plotfile is stored.

| Module | Description |
|--------|-------------|
| `io.plotfile` | `Plotfile` / `PlotfileSeries`, lazy `yt` load |
| `io.fields` | Field discovery, alias resolution, array extraction |
| `io.amr` | AMR MeshBlock geometry |
| `analysis.profiles` | Line / profile extraction (numbers, no figures) |
| `plotting.plots` | Field maps, AMR overlays, profile plots |
| `plotting.animation` | GIF generation from a plotfile series |
| `preprocessing` | Rasters / DEMs / fields -> one HydroEXA HDF5 file |
| `audit` | SLOC entropy and include/import coupling of the source tree |
| `cli` | Command-line interface |
