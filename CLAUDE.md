# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HydroEXA is a GPU-accelerated, performance-portable AMR (Adaptive Mesh Refinement) solver for shallow water and morphodynamic systems, built on [AMReX](https://amrex-codes.github.io/). It targets large-scale simulations of coupled flow-sediment dynamics with embedded boundaries.

The solver uses a 2D spatial dimension (`AMReX_SPACEDIM=2`) with MPI parallelism and optional CUDA/HIP GPU acceleration.

## Build System

CMake-based build with a two-stage process: AMReX first, then HydroEXA.

```bash
# Build AMReX + HydroEXA for a target machine and build type
bash scripts/build_HydroEXA.sh <machine> <build_type>
```

**Machine targets:** `local.gpu`, `local.cpu`, `kabre.gpu`, `frontier.gpu`, `juwelsbooster.gpu` (and `.cpu` variants)
**Build types:** `Release`, `Debug`, `RelWithDebInfo`

The build script sources machine-specific compiler settings from `machines/<machine>` and passes GPU flags via `set_machine_env()`.

### Manual CMake configure (after AMReX is built)

```bash
cmake -S . -B build \
    -DHYDROEXA_GPU_BACKEND=<machine_target> \
    -DAMReX_ROOT=<amrex_install_path> \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++
cmake --build build -j 16
cmake --install build
```

### Python environment

Uses `uv` (via `pyproject.toml`). Install with:
```bash
uv sync
```

Python dependencies are for post-processing/visualization: `geopandas`, `h5py`, `matplotlib`, `scipy`, `yt`, `rasterio`, `shapely`, `polars`.

## Code Architecture

```
HydroEXA/
├── src/
│   ├── HydroEXADriver.cpp      # Main entry point (amrex::Initialize → HydroEXA::Initialize → Compute → Finalize)
│   ├── HydroEXA.cpp            # Core solver class: mesh setup, ICs, time-stepping loop
│   ├── io/                     # I/O handlers
│   │   ├── IOHandler.cpp       # Input file parsing, plotfile writing
│   │   └── Checkpointer.cpp    # State checkpointing/restoring
│   ├── state/
│   │   └── AmrMeshState.cpp    # AMR mesh state management
│   └── solvers/
│       └── Roe.cpp             # Roe Riemann solver implementation
├── include/
│   ├── HydroEXA.H              # Main HydroEXA class declaration
│   ├── HydroEXA_prob_parm.H    # Problem-specific parameters
│   ├── solvers/
│   │   ├── BaseSolver.H        # Abstract solver interface
│   │   ├── SolverVariant.H     # Solver factory/dispatch
│   │   ├── Roe.H               # Roe solver (C++ header)
│   │   ├── RoeExnerACM.H       # Roe solver for Exner term (ACM scheme)
│   │   ├── RoeExnerFCM.H       # Roe solver for Exner term (FCM scheme)
│   │   └── kernels/Roe.H       # GPU kernel implementations
│   ├── io/
│   │   ├── IOHandler.H
│   │   ├── Checkpointer.H
│   │   ├── PlotFileWriter.H    # Base plotfile writer
│   │   ├── HDF5_File.H         # HDF5 I/O utilities
│   │   ├── HDF5PlotFileWriter.H
│   │   └── NativePlotFileWriter.H
│   ├── boundaries/
│   │   ├── EmptyFill.H         # Empty/ghost cell boundary conditions
│   │   └── DynamicTerrain.H    # Moving terrain boundary condition
│   ├── amr/
│   │   └── Tagging.H           # AMR refinement criteria
│   └── utils/
│       ├── Params.H            # Parameter parsing utilities
│       ├── Constants.H         # Physical/mathematical constants
│       └── Debug.H             # Debug logging macros
├── machines/                   # Machine-specific compiler profiles
├── tests/                      # Test scripts and data
├── scripts/
│   ├── build_HydroEXA.sh       # Main build script (AMReX + HydroEXA)
│   ├── build_AMReX.sh          # AMReX-only build
│   ├── machine_selection.sh    # Compiler/GPU flag dispatch
│   └── visualize.py            # Python visualization helper
└── CMakeLists.txt
```

### Key design patterns

- **Solver variant pattern**: `BaseSolver` is an abstract interface; `SolverVariant` constructs the concrete solver (Roe, RoeExnerACM, RoeExnerFCM) based on input parameters.
- **Two-phase initialization**: `HydroEXA` constructor reads inputs, then `Initialize()` sets up mesh state and constructs the solver variant (avoids calling virtual functions in constructors).
- **GPU kernels**: Solver flux computations are split into host-side C++ (`Roe.cpp`) and device-side kernels (`kernels/Roe.H`) for CUDA/HIP portability.
- **I/O**: Supports AMReX native plotfiles, HDF5 plotfiles, and a generic `IOHandler` for input parsing.

## Running Simulations

The built executable `build/HydroEXA` (or `install/bin/HydroEXA`) reads simulation parameters from an inputs file and produces AMReX plotfile output. Run with MPI:

```bash
mpirun -np <N> build/HydroEXA inputs_file=<path>
```

## Python Post-Processing

The `tests/` directory contains utility scripts:
- `plot_amrex_output.py` — Plot AMReX simulation output
- `visualize.py` — General visualization
- `subplots.py` — Multi-panel plotting
- `interpolate_DEM.py` — Interpolate digital elevation models
- `tif2h5.py` / `make_test_h5.py` — Convert raster data to HDF5 for input

Python deps managed by `uv` (see `pyproject.toml`).
