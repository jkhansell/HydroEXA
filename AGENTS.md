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
- Active development branch: `develop` (see `docs/contributing.md` for the branch model).

## Build System

A CMake **superbuild** (HDF5 → AMReX → HydroEXA via `ExternalProject`) described by
CMake presets, plus a legacy shell pipeline that still works. The reference is
`docs/building.md`; this section is the short version an agent needs.

```bash
./scripts/with-machine <machine> -- cmake --preset <card>-<precision>
./scripts/with-machine <machine> -- cmake --build build/<card>-<precision> -j 16
# e.g.
./scripts/with-machine kabreV100.gpu -- cmake --preset kabreV100-gpu-double
```

- **Split of responsibility.** `machines/<machine>` is the *toolchain* — `module
  purge`/`module load`, then `CC`/`CXX`/`FC`. The *preset* is the build — backend,
  GPU architecture, precision, build type. Never put a `module load` in a preset,
  and never put a build option in a profile: the dependency prefix is keyed by the
  toolchain, so the two layers must stay separable.
- **Preset layers:** `backend-*` → `card-*` → `precision-*`/`buildtype-*` → leaf
  `<card>-<precision>`. The first three are fragments: listed by
  `cmake --list-presets`, and a `FATAL_ERROR` if selected. Only leaves configure.
- **Compiler overrides** go through `with-machine` (`--cc`, `--cxx`, `--fc`,
  `--no-modules`), never through a preset, for the reason above.
- **Root `CMakeLists.txt` is the superbuild; `src/CMakeLists.txt` is a standalone
  project.** Never `add_subdirectory(src)` — the solver must stay configurable
  against any AMReX + HDF5 installation.
- **No machine names in CMake.** Build variation travels as capability variables:
  `HYDROEXA_GPU_BACKEND` (`CUDA`/`HIP`/`NONE`), `HYDROEXA_GPU_ARCH`,
  `HYDROEXA_MPI_LINK_WORKAROUND`. A new machine is a profile plus a preset, not an
  edit to `CMakeLists.txt`.
- **Artifacts** are keyed by toolchain and option set, not by machine name:
  - `deps/<toolchain-tag>/<machine>/hdf5` — parallel HDF5, precision-independent
  - `deps/<toolchain-tag>/<machine>/amrex_<PRECISION>_<hash>` — AMReX, per precision
    and AMReX option set
  - `build/<preset>/` — superbuild tree; `install/<preset>/bin/HydroEXA` — the solver
  - The toolchain tag is a hash of the compiler's link line (`mpicxx -show`), so a
    changed MPI lands in a new prefix instead of reusing an incompatible one.
- **Build types:** `Release`, `Debug`, `RelWithDebInfo`. **Precision:** `DOUBLE` or
  `SINGLE` (a preset axis; `amrex::Real` is fixed at compile time).

### Rebuilding after an edit

```bash
# The edit -> binary loop (the machine environment must be in scope):
./scripts/with-machine <machine> -- cmake --build build/<preset> -j 16
```

The solver stage is `BUILD_ALWAYS`, so build and install run on every invocation
while the configure step stays stamp-driven — `cmake --build` re-runs configure by
itself when a `CMakeLists.txt` changes. There is no cache deletion and no reason to
`rm -rf build/<preset>` for an ordinary source edit.

A toolchain or dependency change does not need a manual `rm -rf` either: the prefix
is keyed, so a different compiler or AMReX option set lands in a different directory.
Delete a specific `deps/…` prefix only to force a rebuild of that dependency.

### Legacy shell pipeline

`scripts/build_HydroEXA.sh <machine> <buildtype> <precision>` still works and still
runs the three stages, into `build_<machine>/` and `install_<machine>/`. It shares
the profiles with the presets:

- profiles are **toolchain only**; the `GPU_FLAGS` string is derived by
  `scripts/lib.sh` from the card description (`HYDROEXA_GPU_BACKEND` /
  `HYDROEXA_GPU_ARCH`), so a profile cannot disagree with the presets;
- a **missing profile is a hard error** (it used to warn and continue with no
  compiler set);
- the solver stage configures `-S "${HYDROEXA_SRC}/src"`, because the repository
  root is a superbuild now, not the solver project.

### CMake version coupling

`AMReXConfig.cmake` embeds `cmake_minimum_required(VERSION <the cmake that built
AMReX>)`, so `find_package(AMReX)` fails when the running cmake is *older* than the
one that built the AMReX install — even though AMReX's own sources declare 3.18.

The superbuild removes the problem instead of working around it: it builds AMReX
with the cmake you invoke it with, so the requirement always matches. Kabré's
`/usr/bin/cmake` (3.26.5) is enough; the `cmake/4.x` modules under `/opt/tools` are
unusable there anyway (`libssl.so.10: cannot open shared object file`).

The legacy path still points at the pre-existing `install_kabreV100.gpu/amrex_DOUBLE`,
which was built with cmake 3.31.8, so that path keeps demanding cmake ≥ 3.31.8 and
remains blocked on Kabré for the old reason.

### Manual CMake configure (against an existing AMReX + HDF5)

`src/` is a complete project; this is the path for a machine whose dependencies
already exist, or for an out-of-tree solver build:

```bash
cmake -S src -B build-hydroexa \
    -DHDF5_ROOT=<repo>/install_<target>/hdf5 \
    -DAMReX_ROOT=<repo>/install_<target>/amrex_<PRECISION>/lib/cmake/AMReX \
    -DHYDROEXA_GPU_BACKEND=CUDA \
    -DHYDROEXA_GPU_ARCH=70 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build-hydroexa -j 16
cmake --install build-hydroexa
```

### Local development notes (RTX 4070 / Ubuntu 24.04)

- Run builds inside the conda env — `mamba run -n hydroexa_env ./scripts/with-machine
  local.gpu -- cmake --preset local-gpu-double` (or the legacy
  `bash scripts/build_HydroEXA.sh …`) — so `mpicc`/`mpicxx` and the conda HDF5 CMake
  config are found consistently.
- `local.cpu` / `local.gpu` use `mpicc`/`mpicxx` deliberately, to match the MPI variant
  that conda HDF5 links against (avoids `ompi_mpi_errors_throw_exceptions` symbol errors).
- **`LANGUAGE CUDA` on the solver's `.cpp` files is required, and
  `CUDA_SEPARABLE_COMPILATION` with it.** With `AMREX_USE_GPU`,
  `AMReX_GpuQualifiers.H` maps `AMREX_GPU_HOST`/`AMREX_GPU_DEVICE` to
  `__host__`/`__device__` and includes `<hip/hip_runtime.h>` for HIP but nothing for
  CUDA, so a host compiler cannot parse the installed AMReX headers: building these
  sources with `mpicxx` stops at `AMReX.H:138: error: '__host__' does not name a type`.
  And because AMReX itself is built with `AMReX_GPU_RDC=ON` (its default), dropping
  separable compilation fails to link on `__cudaRegisterLinkedBinary_*` from
  `libamrex_2d.a`. Both measured on `nukwa-03` with CUDA 12.4 (2026-11-17). The older
  claim that nvcc rejects C++20 features (`std::numbers::pi_v`, `std::ssize`) is **not**
  reproducible — nvcc 12.4 compiles them fine.
- Consequence: every solver translation unit carries a relocatable device image, and
  the binary is ~164 MB (142 MB of CUDA fatbins). `-Xfatbin=-compress-all` changes
  nothing; shrinking it means rebuilding AMReX with `-DAMReX_GPU_RDC=OFF` (AMReX also
  hard-codes `--generate-line-info` into its exported CUDA options). See
  `docs/building.md`.
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

Each profile owns the toolchain only: `module purge`/`module load`, then `CC`/`CXX`/`FC`
(deferring to an already-set compiler), plus the card description
`HYDROEXA_GPU_BACKEND` / `HYDROEXA_GPU_ARCH`. Profiles do **not** set `GPU_FLAGS` or any
build option — `scripts/lib.sh` derives the legacy flag string from the card description.

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
│   │   └── kernels/              # device-side: Roe.H, HydrostaticRoe.H
│   ├── state/AmrMeshState.H      # AmrCore subclass owning state, terrain, solver, BCs
│   └── utils/                    # Params.H, Constants.H, Logging.H,
│                                 # MaskedAverageDown.H, FillPatchUtils.H
├── CMakeLists.txt                # superbuild: HDF5 → AMReX → HydroEXA (ExternalProject)
├── CMakePresets.json             # preset index; includes cmake/presets/*.json
├── cmake/presets/                # backend/card/precision/buildtype fragments + leaves
├── machines/                     # toolchain-only profiles (7)
├── scripts/                      # with-machine, build_HydroEXA.sh (legacy), lib.sh, visualize.py
├── tests/                        # dambreak (E2E battery), dambreakNoAMR (legacy),
│                                 # dambreakcircle, LakeAtRestRoe,
│                                 # LakeAtRestHydrostaticRoe, LakeAtRestAugmentedRoe (stale)
├── PyHydroEXA/                   # Python companion package (uv workspace member)
├── docs/building.md              # build reference: presets, artifacts, troubleshooting
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
# preset build (run from a test directory: input paths are relative)
mpirun -np <N> <repo>/install/<preset>/bin/HydroEXA inputs

# legacy build
mpirun -np <N> ../../build_<target>/hydroexa_<PRECISION>/HydroEXA inputs
```

The binary must run inside the machine environment (`./scripts/with-machine <machine> --
mpirun …`), otherwise the system `libstdc++` is too old for the gcc that built it.

Run from the test directory (input paths are relative). Test cases:

| Directory | Purpose |
|-----------|---------|
| `tests/dambreak/` | **E2E dam-break battery (SWASHES benchmark)**: shared `RectangularDambreak.h5` + analysis scripts at the parent; `solvers/{roe,hydrostatic_roe}/` hold input variants (`amr`, `amr_noreflux`, `amr_nosubcycle`, `amr_fastregrid`, `amr_buf`, `single_level`, `single_level_fine`) + per-solver test docs; all artifacts in `outputs/<solver>/`. Runner: `bash run.sh <solver> <variant> <exe> [np] [tag]` (`HYDROEXA_SKIP_PLOTS=1` = solve only, for bisects). Intended for CI/CD. Accuracy is scored as length-weighted L1/L2/Linf against the exact solution and against both uniform yardsticks (`single_level_fine` = level-1 dx = stable, `single_level` = level-0 dx = base); `metrics_table.py` emits the CSV scoreboard. The `regrid_int=5` variant reaches parity with the uniform fine run, which pins the AMR accuracy deficit on regrid cadence rather than the flux or the coupling. |
| `tests/dambreakNoAMR` | legacy dam break on a **different domain** (51.2×1.6) — not a control for `tests/dambreak/`; the fixed-grid reference there is `solvers/roe/inputs_single_level` |
| `tests/dambreakcircle` | circular dam break, includes DEM/TIFF → HDF5 tooling |
| `tests/LakeAtRestAugmentedRoe` | stale dir — its `inputs` is a dam-break config (`model="Roe"`); pending rename/removal |
| `tests/LakeAtRestHydrostaticRoe` | lake-at-rest preservation, hydrostatic Roe |

Generate test input: `uv run python tests/make_test_h5.py` (per test directory).
Analysis/plot scripts sit beside each `inputs` file (`plot_output.py`, `analyze.py`,
`visualize.py`, `audit.py`, `analytical.py`).

## Python Companion Package (`PyHydroEXA`)

The root `pyproject.toml` is a **uv workspace** whose only member is `PyHydroEXA`
(distribution name `pyhydroexa`). Install everything with `uv sync` from the repo root.

- Preprocessing: raster/DEM → HydroEXA HDF5 (`preprocessing/`)
- io: plotfile readers, field extraction, AMR block geometry (`io/`)
- analysis: reusable reductions, e.g. line profiles (`analysis/`)
- plotting: field maps, AMR overlays, animations (`plotting/`)
- audit: SLOC entropy + include/import coupling of the source tree (`audit/`)
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
- **`kernels/AlternateRoe.H`** was removed (2026-11-13): it was orphaned —
  no references anywhere in `src/` or `include/`, and not in `SolverVariant`.

- **Artifacts in tree** (do not treat as source): `build_*/`, `install_*/`, `.venv/`,
  `logs/`, `tests/*/plotfiles/`, `tests/*/outputs/`, `Backtrace.*`, `err.log`.
  Third-party *sources* stay in `tpl/`; their builds/installs live under the
  machine-namespaced directories.
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
