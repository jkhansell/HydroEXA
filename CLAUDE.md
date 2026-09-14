# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HydroEXA is a GPU-accelerated, performance-portable AMR (Adaptive Mesh Refinement) solver for shallow water and morphodynamic systems, built on [AMReX](https://amrex-codes.github.io/). It targets large-scale simulations of coupled flow–sediment dynamics with embedded boundaries.

The solver uses 2D spatial discretization (`AMReX_SPACEDIM=2`) with MPI parallelism and optional CUDA/HIP GPU acceleration. Conserved variables: water depth `h`, x-momentum `hu`, y-momentum `hv`. Terrain: bathymetry `z` + mask `n` (1 = fluid, -9999 = NODATA).

## Build System

CMake-based build with a two-stage process: AMReX first, then HydroEXA.

```bash
# Build AMReX + HydroEXA for a target machine and build type
bash scripts/build_HydroEXA.sh <machine> <build_type>
```

**Machine targets:** `local.gpu`, `local.cpu`, `kabreL40S.gpu`, `kabreV100.gpu`, `frontier.gpu`
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

### Local build notes (RTX 4070 / Ubuntu 24.04)

Use `mamba run -n hydroexa_env` to run the build inside the conda environment — this ensures `mpicc`/`mpicxx` wrappers and HDF5 cmake config are found consistently.

* `machine_selection.sh` uses `mpicc`/`mpicxx` for `local.gpu`/`local.cpu` so MPI wrappers match conda HDF5's transitive MPI dependencies (avoids system/conda MPI mismatch).
* Do **NOT** force `LANGUAGE CUDA` on HydroEXA C++ sources — nvcc's C++ parser cannot handle C++20 features (`std::numbers::pi_v`, `std::ssize`) used in AMReX headers. Sources compile with `mpicxx`; AMReX handles its own CUDA compilation.
* **Debug builds recommended** for development (better backtraces). Release for production.
* Test input generation: `uv run python tests/make_test_h5.py` (creates `CircularDambreak.h5`).
* Run from `tests/` directory: `mamba run -n hydroexa_env ../build/HydroEXA inputs`

## Code Architecture

```
HydroEXA/
├── src/
│   ├── HydroEXADriver.cpp      # Main entry point (amrex::Initialize → HydroEXA::Initialize → Compute → Finalize)
│   ├── HydroEXA.cpp            # Core driver: ReadParameters, Initialize, Compute (time-stepping loop), Finalize
│   ├── io/
│   │   ├── IOHandler.cpp       # Input file parsing, plotfile writing, checkpoint I/O
│   │   └── Checkpointer.cpp    # State checkpointing/restoring (HDF5 or Native format)
│   ├── state/
│   │   ├── AmrMeshState_Init.cpp    # Constructor, Initialize(), InitializeTerrainFluid(), InitializeSolver()
│   │   ├── AmrMeshState_AMR.cpp     # AMR overrides: MakeNewLevelFromScratch/Coarse, RemakeLevel, ClearLevel, ErrorEst, PostProcessBaseGrids
│   │   ├── AmrMeshState_Physics.cpp # Time-stepping: ComputeDt, TimeStepWithSubcycling, AdvanceLevel
│   │   └── AmrMeshState_Utils.cpp   # Utils: ResizeLevels, WritePlotfile, WriteCheckpoint, StoreInitialMassMomentum, ComputeMassDiagnostics
│   ├── solvers/
│   │   └── Roe.cpp             # Roe Riemann solver: compute_fluxes, compute_dt, tag_cells
│   └── utils/
│       └── MaskedAverageDown.cpp  # GPU-aware volume-weighted terrain down-sampling with NODATA masking
├── include/
│   ├── HydroEXA.H              # Main HydroEXA class declaration
│   ├── solvers/
│   │   ├── BaseSolver.H        # CRTP abstract solver interface (compute_fluxes, compute_dt, tag_cells)
│   │   ├── SolverVariant.H     # std::variant wrapper for concrete solver dispatch
│   │   ├── SolverContext.H     # Context struct passed to solver methods (MultiFabs, BCs, FluxRegisters)
│   │   ├── Roe.H               # Roe solver (C++ header — host-side methods)
│   │   ├── RoeExnerACM.H       # Roe solver for Exner term (ACM scheme) — declared but not implemented
│   │   ├── RoeExnerFCM.H       # Roe solver for Exner term (FCM scheme) — declared but not implemented
│   │   └── kernels/Roe.H       # GPU kernel: roeSolver() — device-side Riemann solver
│   ├── io/
│   │   ├── IOHandler.H         # Input parsing, plotfile/checkpoint I/O
│   │   ├── Checkpointer.H      # Checkpoint read/write
│   │   ├── CheckpointerContext.H  # Context for checkpoint I/O
│   │   ├── PlotFileWriter.H    # Base plotfile writer
│   │   ├── HDF5_File.H         # HDF5 I/O utilities (HDF5Reader)
│   │   ├── HDF5PlotFileWriter.H
│   │   └── NativePlotFileWriter.H
│   ├── boundaries/
│   │   └── BCFill.H            # Physical BC type → mathematical BCRec mapping + HydroEXAFill functor
│   └── utils/
│       ├── Params.H            # Parameter structs: RuntimeParameters, IOParameters, AMRParameters, PhysicsParameters, BCParams
│       ├── Constants.H         # Physical constants (GRAV) and tolerances (DEPTH, FLUX, TOLDRY, TOL12)
│       ├── Logging.H           # Structured logging: LogLevel enum, Logger singleton, LOG/DIAG/DEBUG macros
│       └── MaskedAverageDown.H # GPU kernel + host function for NODATA-aware terrain averaging
├── machines/                   # Machine-specific compiler/GPU profiles (local.gpu, local.cpu, kabreL40S.gpu, kabreV100.gpu, frontier.gpu)
├── tests/                      # Test cases with inputs files, visualization, and analysis scripts
├── scripts/
│   ├── build_HydroEXA.sh       # Main build script (AMReX + HydroEXA)
│   ├── build_AMReX.sh          # AMReX-only build
│   ├── machine_selection.sh    # Compiler/GPU flag dispatch
│   └── print_banner.sh         # Build script banner
├── tpl/amrex/                  # AMReX submodule (git submodule)
├── CMakeLists.txt
└── docs/PIPELINE.md            # Detailed simulation pipeline documentation
```

### Key design patterns

- **CRTP solver interface**: `BaseSolver<Derived>` is a CRTP base; concrete solvers (currently only `Roe`) inherit and implement `compute_fluxes_Impl`, `compute_dt_Impl`, `tag_cells_Impl`.
- **Variant dispatch**: `SolverVariant` wraps `std::variant<std::monostate, Roe>` for runtime solver selection based on input parameters.
- **Two-phase initialization**: `HydroEXA` constructor reads inputs via `ReadParameters()`, then `Initialize()` sets up mesh state and constructs the solver variant (avoids virtual calls in constructors).
- **GPU kernels**: Solver flux computations are split into host-side C++ (`Roe.cpp`) and device-side kernels (`kernels/Roe.H`) using `AMREX_GPU_DEVICE` lambdas via `amrex::ParallelFor` and `amrex::ReduceOps`, portable across CUDA/HIP.
- **StaticTerrain pattern**: A single high-resolution `MultiFab` is loaded once from HDF5 at startup as the "ground truth" terrain database. `TerrainMapStaticToDynamic` maps terrain to each AMR level using NODATA-aware `masked_average_down` (finer→coarser) or `cell_cons_interp` (coarser→finer).
- **Mesh pruning**: `PostProcessBaseGrids` removes AMR Level 0 blocks that are 100% NODATA (bathymetry = -9999.0), reducing memory and computation.
- **Structured logging**: `Logging.H` provides a `Logger` singleton with `LogLevel` enum (INFO=0, DIAG=1, DEBUG=2, EXPL=3, WARN=4, ERROR=5). Configurable via `HydroEXA.log_verbosity`. IO-processor gated.
- **I/O**: Supports AMReX native plotfiles, HDF5 plotfiles, and HDF5/native checkpoints. `IOHandler` orchestrates all I/O.

## Running Simulations

The built executable reads simulation parameters from an inputs file and produces AMReX plotfile output. Run with MPI:

```bash
mpirun -np <N> build/HydroEXA inputs
```

Test cases in `tests/dambreak/` and `tests/dambreakcircle/` include `inputs` files and HDF5 terrain data.

## Simulation Pipeline

See `docs/PIPELINE.md` for the complete simulation pipeline documentation, including:
- Entry point & initialization flow
- Parameter reading (all input parameters with meanings)
- Geometry & mesh setup from HDF5 metadata
- AMR time-stepping with subcycling
- Roe Riemann solver details (dual effective flux formulation, source terms)
- AMR refinement (gradient-based tagging)
- Conservation diagnostics
- Output (plotfiles, checkpoints)
- What's implemented vs. what's missing

## Python Post-Processing

The `tests/` directory contains utility scripts:
- `tests/dambreak/plot_output.py`, `analyze.py` — Dam break test analysis
- `tests/dambreakcircle/visualize.py`, `plot_amrex_output.py`, `audit.py` — Circular dam break visualization and auditing
- `tests/dambreak/make_test_h5.py`, `tests/dambreakcircle/make_test_h5.py` — Generate test HDF5 input files
- `scripts/visualize.py` — General visualization helper
- `tests/dambreakcircle/interpolate_DEM.py`, `tif2h5.py` — Raster-to-HDF5 conversion

Python deps managed by `uv` (see `pyproject.toml`).

## Key Files for Common Tasks

| Task | Key Files |
|------|-----------|
| Add new solver | `include/solvers/BaseSolver.H`, `include/solvers/SolverVariant.H`, `src/solvers/Roe.cpp` |
| Modify time-stepping | `src/HydroEXA.cpp` (Compute), `src/state/AmrMeshState_Physics.cpp` |
| Modify AMR/refinement | `src/state/AmrMeshState_AMR.cpp`, `src/solvers/Roe.cpp` (tag_cells) |
| Modify I/O format | `src/io/IOHandler.cpp`, `src/io/Checkpointer.cpp`, `include/io/` |
| Modify boundary conditions | `include/boundaries/BCFill.H` |
| Add new physics constants | `include/utils/Constants.H` |
| Add logging | `include/utils/Logging.H` |
| Build for different machines | `machines/`, `scripts/build_HydroEXA.sh` |
| Terrain/NODATA handling | `src/state/AmrMeshState_Init.cpp`, `src/utils/MaskedAverageDown.cpp` |

## Dependencies

- **AMReX** (git submodule at `tpl/amrex`) — AMR framework with CUDA/HIP backends
- **MPI** (MPICH or Cray MPICH) — parallelism
- **HDF5** (parallel build) — I/O
- **CUDA Toolkit** or **ROCm** — GPU acceleration (machine-dependent)
- **CMake ≥ 3.20** — build system
- **Python 3.12+** with `uv` — post-processing (see `pyproject.toml`)
