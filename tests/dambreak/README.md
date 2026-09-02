# Rectangular Dam Break (1D St. Venant)

## Purpose

This test case is designed specifically for **debugging Riemann solver fluxes**. Unlike the circular dam break (radial symmetry), the rectangular dam break has a **planar discontinuity** that propagates along one axis only. This makes it trivial to spot directional flux errors.

## Setup

```bash
cd HydroEXA/tests/dambreak

# Generate the HDF5 input file (requires h5py + numpy)
uv run python make_test_h5.py

# Run the simulation
cmake --build build --target HydroEXA
mpirun -np 4 ./build/HydroEXA inputs
```

## Test Case Parameters

| Parameter | Value |
|---|---|
| Domain | 256 m × 16 m (256 × 16 cells, 1 m resolution) |
| Dam location | x = 128 m (domain center) |
| Left water depth (h_L) | 0.5 m |
| Right water depth (h_R) | 0.1 m |
| Bathymetry | Flat (z = 0) |
| Boundary conditions | Outflow on x-ends, symmetry on y-ends |
| Simulation time | 0.3 s |
| CFL | 0.45 |

## Expected Physics

The exact solution consists of:
- **Left-going rarefaction wave** into the high-pressure side (h_L = 0.5)
- **Right-going shock wave** into the low-pressure side (h_R = 0.1)

Both waves originate at x = 128 m at t = 0.

## Diagnostics

### Key Check: Y-Symmetry

Since this is a 1D problem, **all y-cuts of the solution should be identical**. Any y-dependence is purely numerical error.

```bash
# After running, visualize:
uv run python plot_output.py
```

The plot will show:
1. **2D colormap** of water depth (planform view)
2. **7 overlaid line cuts** at different y-positions — if they don't overlap, your fluxes have directional bias
3. **Time-series GIF** of the evolution

### Mass Conservation

Initial total mass = 1228.8 m² (check this against the plotfile output).

### Analytical Solution Comparison

The exact solution at time t can be computed from the Riemann solver for the shallow water equations with:
- Left state: h_L = 0.5, u_L = 0
- Right state: h_R = 0.1, u_R = 0

The rarefaction fan speed: `u_L - c_L = -sqrt(g * h_L)`
The shock speed follows from the Rankine-Hugoniot conditions.

## Comparison with Circular Dam Break

| Aspect | Rectangular | Circular |
|---|---|---|
| Symmetry | Planar (1D) | Radial (2D) |
| Flux debugging | Easy (y-cuts should overlap) | Harder (x/y entangled) |
| Wave structure | Rarefaction + Shock | Radial rarefaction |
| Best for | Directional flux errors | Radial/Riemann correctness |

## Files

- `make_test_h5.py` — Generates `RectangularDambreak.h5` input file
- `inputs` — Simulation input parameters
- `plot_output.py` — Visualizes plotfiles (2D + 1D line cuts + GIF)
