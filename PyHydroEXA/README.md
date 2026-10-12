# PyHydroEXA

Python companion package for the HydroEXA shallow-water solver.

## Quick start

```python
from pyhydroexa.postprocessing import Plotfile, PlotfileSeries, plot_field_with_amr_overlay

# Single plotfile
pf = Plotfile("plt00010")
print(pf.field_names)
plot_field_with_amr_overlay(pf, field="h", output="water_depth.png")

# Series
series = PlotfileSeries("plt*")
for pf in series:
    print(pf.time)

series.animate(field="h", output="evolution.gif")
```

## CLI

```bash
pyhydroexa-inspect plt00010
pyhydroexa-plot plt00010 h --amr --output depth.png
pyhydroexa-animate "plt*" h --output evolution.gif
```

## Modules

| Module | Description |
|--------|-------------|
| `postprocessing.plotfile` | Plotfile / PlotfileSeries abstractions |
| `postprocessing.fields` | Field discovery and alias resolution |
| `postprocessing.amr` | AMR MeshBlock utilities |
| `postprocessing.profiles` | Line / profile extraction |
| `postprocessing.plots` | Reusable plotting functions |
| `postprocessing.animation` | GIF generation from plotfile series |
| `cli` | Command-line interface |
