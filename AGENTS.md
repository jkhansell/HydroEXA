# HydroEXA — Solver Codebase

Guidance for AI coding agents working in this repository. Local path in WSL:
`~/Documents/HydroEXA/HydroEXACode` (sshfs mount of `kabre:/data/jovillalobos/CFD/HydroEXA`).

## Project Overview

HydroEXA is a GPU-accelerated, performance-portable AMR solver for shallow water flow
coupled to sediment transport (Exner equation), built on
[AMReX](https://amrex-codes.github.io/).

- 2D spatial discretization (`AMReX_SPACEDIM=2`), MPI parallel with optional CUDA/HIP GPU offload.
- Conserved variables: water depth `h`, x-momentum `hu`, y-momentum `hv`.
- Terrain: bathymetry `z` plus mask `n` (1 = fluid, -9999.0 = NODATA).
- Two Riemann solvers are selectable at runtime: `Roe` (classic) and `HydrostaticRoe`
  (well-balanced / lake-at-rest preserving).
- Active development branch: `AmrCore_refactor`.

## Build System

CMake build driven by a three-stage shell pipeline, orchestrated by one entry point:

```bash
bash scripts/build_HydroEXA.sh <machine.cpu|machine.gpu> <build_type> <precision>
# e.g.
bash scripts/build_HydroEXA.sh local.gpu Release DOUBLE
bash scripts/build_HydroEXA.sh kabreL40S.gpu Debug DOUBLE
```

- **Stages:** HDF5 → AMReX → HydroEXA. The child scripts (`build_HDF5.sh`,
  `build_AMReX.sh`) take no arguments; the parent exports `HDF5_ROOT`, `AMREX_ROOT`,
  `HYDROEXA_BUILD`, etc.
- **Build types:** `Release`, `Debug`, `RelWithDebInfo`.
- **Precision:** `DOUBLE` or `SINGLE`.
- **Shared logic:** `scripts/lib.sh` (sourced, never executed) provides `print_banner()`,
  `set_machine_env()`, `resolve_paths()`, `preflight()`, and execution guards.
  It replaced the older `print_banner.sh` / `machine_selection.sh` pair.
- **Output:** `build/HydroEXA`, installed to `install/bin/HydroEXA`.
- Third-party installs land in `tpl/hdf5/install` and `tpl/amrex/install`.

### Manual CMake configure (after AMReX is built)

```bash
cmake -S . -B build \
    -DAMReX_ROOT=<amrex_install>/lib/cmake/AMReX \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build -j 16
cmake --install build
```

### Local development notes (RTX 4070 / Ubuntu 24.04)

- Run builds inside the conda env: `mamba run -n hydroexa_env bash scripts/build_HydroEXA.sh ...`
  so `mpicc`/`mpicxx` and the conda HDF5 CMake config are found consistently.
- `local.cpu` / `local.gpu` use `mpicc`/`mpicxx` deliberately, to match the MPI variant
  that conda HDF5 links against (avoids `ompi_mpi_errors_throw_exceptions` symbol errors).
- Do **NOT** set `LANGUAGE CUDA` on HydroEXA C++ sources: nvcc's C++ parser rejects C++20
  features used by AMReX headers (`std::numbers::pi_v`, `std::ssize`). Sources compile with
  `mpicxx`; AMReX compiles its own CUDA.
- `Debug` builds are recommended for development (usable backtraces); `Release` for production.

## Machine Targets

| Profile | Platform | Toolchain / GPU |
|---------|----------|-----------------|
| `local.cpu` | WSL2 Ubuntu, RTX 4070 host | gcc via `mpicxx`, CPU only |
| `local.gpu` | WSL2 Ubuntu, RTX 4070 | CUDA `native` arch |
| `kabreL40S.gpu` | KISTI Kabré | CUDA 12.4, gcc 13.4, MPICH 4.1.1 |
| `kabreV100.gpu` | KISTI Kabré | CUDA 12.4, arch 7.0 (Volta) |
| `kabreKURA.cpu` | KISTI Kabré | gcc 11.1, MPICH 4.1.1, parallel HDF5 |
| `frontier.gpu` | OLCF Frontier | Cray PrgEnv, ROCm 7.0, `gfx90a` (MI250X) |
| `frontier.cpu` | OLCF Frontier | Cray PrgEnv, no GPU arch |

Each profile is a shell file sourcing its own `module load` set and exporting `GPU_FLAGS`,
`CC`/`CXX`/`FC`, and `HDF5_ROOT` where needed.

## Repository Layout

```
HydroEXACode/
├── src/
│   ├── HydroEXADriver.cpp        # main(): amrex::Initialize → HydroEXA → Compute → Finalize
│   ├── HydroEXA.cpp              # ReadParameteres, Initialize, Compute (subcycled time loop), Finalize
│   ├── io/
│   │   ├── IOHandler.cpp         # input parsing + plotfile/checkpoint orchestration
│   │   ├── Checkpointer.cpp      # HDF5 and native checkpoint read/write
│   │   ├── HDF5PlotFile.cpp      # multi-level MultiFab → HDF5 plotfile
│   │   └── NativePlotFile.cpp    # multi-level MultiFab → AMReX native plotfile
│   ├── solvers/
│   │   ├── Roe.cpp               # classic Roe fluxes, dt, tagging
│   │   └── HydrostaticRoe.cpp    # hydrostatic-reconstruction (well-balanced) Roe
│   ├── state/
│   │   ├── AmrMeshState_Init.cpp     # ctor, Initialize(), terrain + solver setup
│   │   ├── AmrMeshState_AMR.cpp      # AmrCore overrides, ErrorEst, PostProcessBaseGrids
│   │   ├── AmrMeshState_Physics.cpp  # ComputeDt, TimeStepWithSubcycling, AdvanceLevel
│   │   └── AmrMeshState_Utils.cpp    # ResizeLevels, plotfile/checkpoint writers, diagnostics
│   └── utils/MaskedAverageDown.cpp   # NODATA-aware terrain averaging
├── include/
│   ├── HydroEXA.H
│   ├── boundaries/BCFill.H       # physical BC type → BCRec mapping, HydroEXAFill functor
│   ├── io/                       # Checkpointer.H, CheckpointerContext.H, HDF5_File.H (HDF5Reader),
│   │                             # HDF5PlotFile.H, NativePlotFile.H, IOHandler.H
│   ├── solvers/
│   │   ├── BaseSolver.H          # CRTP interface: compute_fluxes, compute_dt, tag_cells
│   │   ├── SolverVariant.H       # std::variant<monostate, Roe, HydrostaticRoe> dispatch
│   │   ├── SolverContext.H       # MultiFabs, BCs, flux registers passed to solvers
│   │   ├── Roe.H, HydrostaticRoe.H
│   │   ├── RoeExnerACM.H, RoeExnerFCM.H   # header-only, see Known Gaps
│   │   └── kernels/              # device-side: Roe.H, AlternateRoe.H, HydrostaticRoe.H
│   ├── state/AmrMeshState.H      # AmrCore subclass owning state, terrain, solver, BCs
│   └── utils/                    # Params.H, Constants.H, Logging.H,
│                                 # MaskedAverageDown.H, FillPatchUtils.H
├── machines/                     # one shell profile per target (7)
├── scripts/                      # build_HydroEXA.sh, build_AMReX.sh, build_HDF5.sh, lib.sh, visualize.py
├── tests/                        # dambreakAMR, dambreakNoAMR, dambreakcircle,
│                                 # LakeAtRestAugmentedRoe, LakeAtRestHydrostaticRoe
├── PyHydroEXA/                   # Python companion package (uv workspace member)
├── docs/PIPELINE.md              # full simulation pipeline reference
├── plans/                        # design notes (HydrostaticRoe, pyhydroexa plans)
├── tpl/                          # amrex + hdf5 third-party sources/installs
├── CMakeLists.txt, pyproject.toml, uv.lock
└── CHANGELOG.md, README.md, LICENSE
```

## Key Design Patterns

- **CRTP solver interface** — `BaseSolver<Derived>`; concrete solvers implement
  `compute_fluxes_Impl`, `compute_dt_Impl`, `compute_dt_level`, `tag_cells_Impl`.
- **Variant dispatch** — `SolverVariant` wraps `std::variant<std::monostate, Roe, HydrostaticRoe>`;
  the solver is chosen from input parameters, then `std::visit` routes every call.
- **Two-phase initialization** — constructor parses inputs (`ReadParameters`), `Initialize()`
  builds mesh state and the solver variant (avoids virtual calls in constructors).
- **AmrMeshState** — declares the full AMR state as an `amrex::AmrCore` subclass; owns
  `U_old`/`U_new`, `StaticTerrain` (ground truth), `DynamicTerrain`, flux registers, BCRecs,
  and the solver variant. Implementation split across four `.cpp` files by concern.
- **GPU kernels** — host-side solver code stays in `.cpp`; device code lives in
  `include/solvers/kernels/*.H` as `AMREX_GPU_DEVICE` lambdas driven by `amrex::ParallelFor`
  / `ReduceOps`, portable across CUDA and HIP.
- **StaticTerrain** — one high-resolution `MultiFab` loaded from HDF5 at startup acts as the
  terrain database; it is mapped to each AMR level with `masked_average_down` (fine→coarse)
  or `cell_cons_interp` (coarse→fine), so NODATA never bleeds into valid cells.
- **Mesh pruning** — `PostProcessBaseGrids` drops Level-0 blocks that are 100% NODATA.
- **Structured logging** — `Logging.H` `Logger` singleton, `LogLevel`
  (INFO=0, DIAG=1, DEBUG=2, EXPL=3, WARN=4, ERROR=5), set via `HydroEXA.log_verbosity`,
  IO-processor gated.
- **I/O split** — `IOHandler` orchestrates; plotfiles and checkpoints each have HDF5 and
  native backends; `HDF5Reader` parses input terrain/bathymetry.

## Running

```bash
mpirun -np <N> build/HydroEXA inputs
```

Run from the test directory (input paths are relative). Test cases:

| Directory | Purpose |
|-----------|---------|
| `tests/dambreakAMR` | rectangular dam break with AMR refinement |
| `tests/dambreakNoAMR` | same case, fixed base grid (AMR control) |
| `tests/dambreakcircle` | circular dam break, includes DEM/TIFF → HDF5 tooling |
| `tests/LakeAtRestAugmentedRoe` | lake-at-rest preservation, augmented Roe |
| `tests/LakeAtRestHydrostaticRoe` | lake-at-rest preservation, hydrostatic Roe |

Generate test input: `uv run python tests/make_test_h5.py` (per test directory).
Analysis/plot scripts sit beside each `inputs` file (`plot_output.py`, `analyze.py`,
`visualize.py`, `audit.py`, `analytical.py`).

## Python Companion Package (`PyHydroEXA`)

The root `pyproject.toml` is a **uv workspace** whose only member is `PyHydroEXA`
(distribution name `pyhydroexa`). Install everything with `uv sync` from the repo root.

- Preprocessing: raster/DEM → HydroEXA HDF5 (`preprocessing/`)
- Postprocessing: plotfile readers, AMR-aware plotting, animation (`postprocessing/`)
- CLI entry points: `pyhydroexa`, `pyhydroexa-inspect`, `pyhydroexa-plot`, `pyhydroexa-animate`
- Dependencies: numpy, scipy, h5py, matplotlib, imageio, yt, geopandas, shapely, rasterio,
  pyogrio, fiona, polars (plus optional `gdal` wheels on Linux)
- Tests: `pytest` (config in `PyHydroEXA/pyproject.toml`); sample outputs in `validation_output/`

## Key Files for Common Tasks

| Task | Files |
|------|-------|
| Add a solver | `include/solvers/BaseSolver.H`, `SolverVariant.H`, `src/solvers/*.cpp`, `kernels/` |
| Change time stepping | `src/HydroEXA.cpp` (`Compute`), `src/state/AmrMeshState_Physics.cpp` |
| Change AMR / tagging | `src/state/AmrMeshState_AMR.cpp`, solver `tag_cells` |
| Change I/O formats | `src/io/*.cpp`, `include/io/` |
| Boundary conditions | `include/boundaries/BCFill.H` |
| Physics constants / tolerances | `include/utils/Constants.H` |
| Input parameters | `include/utils/Params.H`, `src/io/IOHandler.cpp` |
| Terrain / NODATA handling | `src/state/AmrMeshState_Init.cpp`, `src/utils/MaskedAverageDown.cpp` |
| AMR level state | `include/state/AmrMeshState.H`, `src/state/AmrMeshState_*.cpp` |
| Machine profiles | `machines/`, `scripts/lib.sh` |

## Known Gaps & Gotchas

- **Exner solvers are not wired in.** `include/solvers/RoeExnerACM.H` and
  `RoeExnerFCM.H` are declared but absent from `SolverVariant` and have no `.cpp`;
  sediment/morphodynamic coupling is still in progress.
- **`kernels/AlternateRoe.H`** is an experimental kernel variant; confirm it is referenced
  before assuming it is active.
- **Artifacts in tree** (do not treat as source): `build/`, `install/`, `.venv/`,
  `tpl/*/install`, `logs/`, `tests/*/plotfiles/`, `tests/*/outputs/`, `Backtrace.*`, `err.log`.
- This checkout is a **git submodule parent** (`tpl/amrex`); after `git pull` verify
  `git submodule update --init --recursive`.
- The tree lives on an **sshfs mount**: metadata is slow and inconsistent (odd UIDs,
  unreliable `stat`), so avoid recursive `grep`/`find` over the whole repo and prefer
  explicit paths.

## Dependencies

- **AMReX** (`tpl/amrex`, git submodule) — AMR framework, CUDA/HIP backends
- **HDF5** (parallel build, `tpl/hdf5`) — terrain I/O, plotfiles, checkpoints
- **MPI** — MPICH 4.1.1 on Kabré, Cray MPICH on Frontier
- **CUDA 12.4** or **ROCm 7.0** — GPU acceleration
- **CMake ≥ 3.20**
- **Python ≥ 3.12** with `uv`
