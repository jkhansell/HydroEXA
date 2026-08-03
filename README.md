# HydroEXA

HydroEXA is a GPU-accelerated, performance-portable AMR solver for shallow water and morphodynamic systems built on [AMReX](https://amrex-codes.github.io). It targets large-scale simulations of coupled flow–sediment dynamics with exascale-ready design.

## Features

- **Adaptive Mesh Refinement (AMR)** — 2D spatial discretization with dynamic refinement based on gradient thresholds
- **GPU Acceleration** — CUDA and HIP backends via AMReX for portable GPU execution
- **Mesh Pruning** — Runtime removal of AMR Level 0 grid boxes in NODATA regions (bathymetry value ≤ −9999.0), reducing memory and computation without affecting physics in active regions
- **NODATA-Aware Terrain Handling** — Custom volume-weighted terrain down-sampling (`masked_average_down`) that skips NaN/NODATA cells during AMR level transfers, preventing NODATA contamination of valid terrain data
- **Roe Riemann Solver** — With Exner term support via ACM and FCM schemes for sediment dynamics
- **HDF5 I/O** — Parallel HDF5 plotfile writing and input file parsing

## Build Instructions

### Prerequisites

- CMake ≥ 3.20
- MPI (MPICH or Cray MPICH)
- HDF5 (parallel build)
- CUDA Toolkit (for GPU targets) or ROCm (for Frontier)
- AMReX source (checked out as `tpl/amrex` submodule)

### Quick Build

```bash
bash scripts/build_HydroEXA.sh <machine_target> <build_type>
```

**Machine targets:** `local.gpu`, `local.cpu`, `kabreL40S.gpu`, `kabreV100.gpu`, `frontier.gpu`
**Build types:** `Release`, `Debug`, `RelWithDebInfo`

Example:
```bash
bash scripts/build_HydroEXA.sh kabreL40S.gpu Release
```

The build script handles a two-stage cascade: AMReX first, then HydroEXA. The executable is installed to `install/bin/HydroEXA`.

### Manual CMake Build

```bash
# Configure (after AMReX is built)
cmake -S . -B build \
    -DHYDROEXA_GPU_BACKEND=<machine_target> \
    -DAMReX_ROOT=<amrex_install_path> \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++

# Build and install
cmake --build build -j 16
cmake --install build
```

### Machine Profiles

Machine-specific compiler and GPU settings are stored in `machines/`:

| Profile | Platform | GPU Backend | Architecture |
|---|---|---|---|
| `local.gpu` | Linux (gcc) | CUDA | native |
| `local.cpu` | Linux (gcc) | CPU-only | — |
| `kabreL40S.gpu` | Polaris (MPICH + CUDA 12.4) | CUDA | 8.9 (L40S Ada) |
| `kabreV100.gpu` | Polaris (MPICH + CUDA 12.4) | CUDA | 7.0 (V100 Volta) |
| `frontier.gpu` | OLCF Frontier (Cray) | HIP | gfx90a (MI250X) |

## Running Simulations

The executable reads simulation parameters from an inputs file and produces AMReX plotfile output:

```bash
mpirun -np <N> install/bin/HydroEXA inputs_file=<path>
```

## Terrain Representation: StaticTerrain

HydroEXA uses a **StaticTerrain** approach to maintain reliable, consistent bathymetry throughout the simulation. A single high-resolution `MultiFab` is loaded once from the input HDF5 file at startup and serves as the "ground truth" terrain database, split across MPI ranks uniformly at its own resolution.

During the simulation, `TerrainMapStaticToDynamic` maps terrain from StaticTerrain to each AMR level's `DynamicTerrain` MultiFab whenever refinement changes:

- **StaticTerrain finer than AMR level** → restrict via `masked_average_down` (NODATA-aware)
- **StaticTerrain coarser than AMR level** → prolongate via `InterpFromCoarseLevel` with proper GPU/CPU boundary handling

The terrain reference level (`amr.terrain_ref_lev`) controls the resolution of StaticTerrain relative to the AMR grid. Lower values place the terrain at coarser AMR levels (More cells, more memory). Higher values keep the terrain at finer levels (Refined cells, coarsened level 0) — this is the typical configuration.

This approach trades static GPU memory for reliability — terrain is never lost or corrupted during AMR refinement/coarsening cycles. On modern GPUs (L40S 48GB, V100 32GB, Frontier MI250X 16GB per chip) the overhead is manageable, especially since the pruned active domain reduces the effective memory pressure on DynamicTerrain.

## Mesh Pruning

Mesh pruning removes AMR Level 0 grid boxes where bathymetry/terrain data is marked as NODATA (−9999.0). At startup, `PostProcessBaseGrids` checks each candidate box against the StaticTerrain input:

1. Each box is scaled to the terrain grid resolution
2. A GPU-accelerated `ReduceOps` scans all cells for NODATA values
3. Boxes with no valid cells are excluded from the active box list
4. Retained boxes are classified as `pure_fluid` (no NODATA) or `boundary` (mixed)

If fewer than 50% of fine cells are valid during terrain down-sampling, the coarse cell is also marked NODATA. This reduces memory usage and computation for simulations with large offshore domains.

The pruning mask is defined in the input HDF5 file — set bathymetry values to −9999.0 in regions you want excluded.

## Code Architecture

```
HydroEXA/
├── src/
│   ├── HydroEXADriver.cpp      # Main entry point
│   ├── HydroEXA.cpp            # Core solver: mesh setup, time-stepping
│   ├── io/                     # I/O handlers (plotfiles, checkpoints)
│   ├── state/                  # AMR mesh state management
│   ├── solvers/                # Roe Riemann solver
│   └── utils/                  # Helpers (params, constants, logging)
├── include/
│   ├── HydroEXA.H              # Main class declaration
│   ├── solvers/                # Solver interface + GPU kernels
│   ├── io/                     # I/O and checkpoint headers
│   ├── boundaries/             # Boundary condition fillers
│   ├── amr/                    # AMR refinement criteria
│   └── utils/                  # Constants, params, debug macros
├── machines/                   # Machine-specific compiler profiles
├── scripts/                    # Build and visualization scripts
└── tests/                      # Test scripts and input files
```

**Key patterns:**

- **Solver variant** — `BaseSolver` is an abstract interface; `SolverVariant` dispatches the concrete solver based on input parameters
- **Two-phase init** — Constructor reads inputs, then `Initialize()` sets up mesh and constructs the solver (avoids virtual calls in constructors)
- **GPU kernels** — Flux computations use `AMREX_GPU_DEVICE` lambdas via `amrex::ParallelFor` and `amrex::ReduceOps`, portable across CUDA/HIP

## Python Post-Processing

Python dependencies are managed by `uv` (see `pyproject.toml`):

```bash
uv sync
```

Available tools: `geopandas`, `h5py`, `matplotlib`, `scipy`, `yt`, `rasterio`, `shapely`, `polars`.

Test utilities in `tests/`:
- `visualize.py` — Bathymetry and simulation visualization
- `plot_amrex_output.py` — AMReX plotfile plotting
- `tif2h5.py` / `make_test_h5.py` — Raster-to-HDF5 conversion for input files
