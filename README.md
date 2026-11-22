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

- CMake ≥ 3.25 (≥ 3.20 for the standalone solver project)
- MPI (MPICH or Cray MPICH)
- CUDA Toolkit (for GPU targets) or ROCm (for Frontier)
- AMReX and HDF5 sources (checked out as `tpl/amrex` and `tpl/hdf5` submodules)

### Quick Build

The repository root is a superbuild: HDF5 → AMReX → HydroEXA, driven by CMake
presets. The machine profile supplies the toolchain, the preset supplies the
build options.

```bash
./scripts/with-machine <machine> -- cmake --preset <card>-<precision>
./scripts/with-machine <machine> -- cmake --build build/<card>-<precision> -j 16
```

Example:

```bash
./scripts/with-machine kabreV100.gpu -- cmake --preset kabreV100-gpu-double
./scripts/with-machine kabreV100.gpu -- cmake --build build/kabreV100-gpu-double -j 16
```

The binary is installed to `install/<card>-<precision>/bin/HydroEXA`.
`cmake --list-presets` shows every card and precision; selecting a fragment
(`backend-cuda`, `precision-double`, …) stops with instructions instead of
building half a configuration.

Artifacts are keyed by toolchain and by option set, not by machine name:

```
deps/<toolchain-tag>/<machine>/hdf5/                      # parallel HDF5, precision-independent
deps/<toolchain-tag>/<machine>/amrex_<PRECISION>_<hash>/  # AMReX, per precision and option set
build/<preset>/                                           # superbuild tree
install/<preset>/bin/HydroEXA                             # the solver
```

HDF5 has no single/double variant, so it is shared per toolchain; AMReX and
HydroEXA encode `amrex::Real` at compile time and are keyed by precision, so
`DOUBLE` and `SINGLE` builds coexist.

The full reference — preset layers, compiler overrides, adding a machine,
troubleshooting — is [`docs/building.md`](docs/building.md).

### Legacy shell pipeline

The three-stage shell script is still supported and now shares the profiles with
the presets:

```bash
bash scripts/build_HydroEXA.sh <machine_target> <build_type> <precision>
bash scripts/build_HydroEXA.sh kabreL40S.gpu Release DOUBLE
```

**Machine targets:** `local.gpu`, `local.cpu`, `kabreL40S.gpu`, `kabreV100.gpu`,
`kabreKURA.cpu`, `frontier.gpu`, `frontier.cpu`
**Build types:** `Release`, `Debug`, `RelWithDebInfo`
**Precision:** `DOUBLE`, `SINGLE`

It installs to `install_<machine_target>/hydroexa_<PRECISION>/bin/HydroEXA`,
side by side with any preset build.

### Standalone solver build

`src/` is a complete CMake project and can be configured against any AMReX and
HDF5 installation:

```bash
cmake -S src -B build-hydroexa \
    -DAMReX_ROOT=install_<machine_target>/amrex_<PRECISION>/lib/cmake/AMReX \
    -DHDF5_ROOT=install_<machine_target>/hdf5 \
    -DHYDROEXA_GPU_BACKEND=CUDA \
    -DHYDROEXA_GPU_ARCH=70 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=mpicxx

cmake --build build-hydroexa -j 16
cmake --install build-hydroexa
```

### Machine Profiles

Machine profiles hold the toolchain only — modules plus `CC`/`CXX`/`FC` — and
defer to the caller's environment when a compiler is already set:

| Profile | Platform | GPU Backend | Architecture |
|---|---|---|---|
| `local.gpu` | Linux (gcc) | CUDA | native |
| `local.cpu` | Linux (gcc) | CPU-only | — |
| `kabreL40S.gpu` | Kabré (MPICH + CUDA 12.4) | CUDA | 8.9 (L40S Ada) |
| `kabreV100.gpu` | Kabré (MPICH + CUDA 12.4) | CUDA | 7.0 (V100 Volta) |
| `kabreKURA.cpu` | Kabré (MPICH, gcc 11.1) | CPU-only | — |
| `frontier.gpu` | OLCF Frontier (Cray) | HIP | gfx90a (MI250X) |
| `frontier.cpu` | OLCF Frontier (Cray) | CPU-only | — |

## Running Simulations

The executable reads simulation parameters from an inputs file and produces AMReX plotfile output:

```bash
mpirun -np <N> install_<machine_target>/hydroexa_<PRECISION>/bin/HydroEXA inputs_file=<path>
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
