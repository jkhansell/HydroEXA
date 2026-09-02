# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] — 2026-08-31

### Fixed

- **`AmrMeshState::GetData`** — Always returns both `U_old` and `U_new` states with their correct logical timestamps. Previously returned only one state based on time matching, which broke `FillPatch` / `FillCoarsePatch` after internal state swaps because the time-value no longer matched the correct MultiFab.
- **`AmrMeshState::FillCoarsePatch`** — Selects the state closest to the requested time instead of asserting `cmf.size() == 1`. `InterpFromCoarseLevel` does not support time interpolation, so a single state must be selected.
- **`Roe::compute_fluxes_Impl`** — Removed `std::swap(ctx.t_new[lev], ctx.t_old[lev])`. The MultiFab swap stays (needed for the explicit update scheme to read from the unmodified old state), but timestamps now always describe the logical time of the old/new states. This fixes `runtime_params.etime` (was stuck at `0.0` after the swap).

---

## [Unreleased] — 2026-08-29

### Added

- **`machines/local.gpu` / `machines/local.cpu`** — Local machine configuration files for RTX 4070 / Ubuntu 24.04 development. HDF5_ROOT points to conda env `hydroexa_env`. No `module purge` — local machines use direct paths.

### Changed

- **`scripts/machine_selection.sh`** — `local.gpu` and `local.cpu` compiler entries changed from `gcc`/`g++`/`gfortran` to `mpicc`/`mpicxx`/`mpif90` to ensure MPI wrappers match conda env's HDF5 transitive MPI dependencies.
- **`CMakeLists.txt`** — Removed `set_source_files_properties(... PROPERTIES LANGUAGE CUDA)` block that forced all HydroEXA sources through nvcc. nvcc's C++ parser cannot handle C++20 features (e.g., `std::numbers::pi_v`, `std::ssize`) used in AMReX headers. Sources now compile with mpicxx; AMReX handles its own CUDA compilation.

### Fixed

- **nvcc C++20 incompatibility** — HydroEXA sources compiled with mpicxx (g++), avoiding nvcc's limited C++20 support.
- **MPI library mismatch** — Runtime `undefined symbol: ompi_mpi_errors_throw_exceptions` resolved by using conda env's MPI wrappers via `mamba run -n hydroexa_env`, ensuring consistent MPI library linkage.

---

## [Unreleased] — 2026-07-27

### Added

- **`include/utils/MaskedAverageDown.H` / `src/utils/MaskedAverageDown.cpp`** — New GPU-aware volume-weighted terrain down-sampling with NODATA/NaN masking. Replaces `amrex::average_down` in `TerrainMapStaticToDynamic`.
  - `masked_avgdown_with_vol` kernel: computes volume-weighted average of fine cells, skipping NaN and NODATA (-9999.0) values. If fewer than 50% of fine cells are valid, the coarse cell is marked NODATA.
  - `masked_average_down` host function: follows AMReX's `average_down_w_geom` pattern — creates an intermediate `MultiFab` with coarsened fine BoxArray + matching ghost cells, fills via `amrex::ParallelFor`, then copies to destination via `ParallelCopy`.
  - Volume array computed via `fgeom.GetVolume(fvolume, fine_BA, fine_dm, 0)` for proper weighted averaging.

- **`include/utils/Logging.H`** — Structured logging system replacing ad-hoc `amrex::Print()` calls.
  - `LogLevel` enum: INFO (0), DIAG (1), DEBUG (2), EXPL (3), WARN (4), ERROR (5).
  - `LOG(level, msg)` — respects verbosity threshold (configurable via `HydroEXA.log_verbosity`).
  - `LOG_WARN(msg)` / `LOG_ERROR(msg)` — always print.
  - IO-processor gated: only the IO rank calls `amrex::Print()`.

### Changed

- **`src/state/AmrMeshState.cpp`**
  - `TerrainMapStaticToDynamic`: replaced `amrex::average_down()` with `amrex::masked_average_down()`, adding NaN/NODATA-aware terrain restriction.
  - Changed interpolator from `&amrex::pc_interp` to `&amrex::cell_cons_interp` for prolongation (conservative interpolation).
  - Added `StaticTerrain.FillBoundary()` and NaN check before `masked_average_down`.
  - Added `FillBoundary` calls for DynamicTerrain, U_new, U_old in `MakeNewLevelFromScratch`.
  - Removed debug diagnostic prints (geometry checks, NaN diagnostics, stub checkpoint messages).
  - Terrain BCs now use `int_dir` for periodic directions instead of always `foextrap`.
  - `static_ba.maxSize()` removed — defaults to AMReX box decomposition (was previously set to 512, causing BoxArray misalignment with AMR level 0).
  - `CheckNaNsAndValidInMultiFab` moved from private to protected section.
  - `ErrorEst`: uses `geom[lev].CellSizeArray()` instead of `Geom(lev).CellSize()` for GPU compatibility; NODATA comparisons use `== -9999.0` literal.

- **`src/HydroEXA.cpp`** — Replaced verbose `amrex::Print()` geometry diagnostic block with `LOG(INFO, ...)` call.

- **`src/HydroEXADriver.cpp`** — Replaced `amrex::Print()` total time output with `LOG(INFO, ...)`.

- **`src/io/Checkpointer.cpp`** — Replaced `amrex::Print()` checkpoint read/write messages with `LOG(INFO, ...)`.

- **`src/io/IOHandler.cpp`** — Replaced `amrex::Print()` plotfile messages with `LOG(INFO, ...)` and `LOG_WARN(...)`.

- **`include/io/Checkpointer.H`** — Added `#include <string>` for LOG macro usage.

- **`include/state/AmrMeshState.H`** — Moved `CheckNaNsAndValidInMultiFab` from private to protected; removed redundant declaration.

### Fixed

- **Non-deterministic terrain in black (NODATA) regions** — Root cause: `static_ba.maxSize(512)` created a box decomposition that didn't align with AMR level 0's BoxArray. When `masked_average_down` created `crse_S_fine_BA = fine_BA.coarsen(ratio)` and used `ParallelCopy` to transfer to `S_crse`, the spatial lookup placed averaged values at incorrect physical positions, producing streaks of terrain data appearing in NODATA regions. Fix: remove the hardcoded `maxSize(512)` so the BoxArray decomposition matches naturally.
- **Ghost cell mismatch in ParallelCopy** — Fixed by constructing `crse_S_fine` with `S_crse.nGrow()` ghost cells (matching the destination), ensuring ParallelCopy operates on correctly sized arrays.
