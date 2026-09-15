# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] — 2026-08-31

### Fixed

- **`AmrMeshState::GetData`** — Always returns both `U_old` and `U_new` states with their correct logical timestamps. Previously returned only one state based on time matching, which broke `FillPatch` / `FillCoarsePatch` after internal state swaps because the time-value no longer matched the correct MultiFab.
- **`AmrMeshState::FillCoarsePatch`** — Selects the state closest to the requested time instead of asserting `cmf.size() == 1`. `InterpFromCoarseLevel` does not support time interpolation, so a single state must be selected.
- **`Roe::compute_fluxes_Impl`** — Removed `std::swap(ctx.t_new[lev], ctx.t_old[lev])`. The MultiFab swap stays (needed for the explicit update scheme to read from the unmodified old state), but timestamps now always describe the logical time of the old/new states. This fixes `runtime_params.etime` (was stuck at `0.0` after the swap).

### Changed

- **`scripts/lib.sh`** — New shared library consolidating `print_banner.sh` and `machine_selection.sh`. Provides `print_banner()`, `set_machine_env()`, `resolve_paths()`, `preflight()`, and execution guards. All build scripts source this single file.
- **`scripts/build_HydroEXA.sh`** — Thin orchestrator: sources `lib.sh`, resolves paths, validates target, exports install paths, then runs the 3-step pipeline (HDF5 → AMReX → HydroEXA). Removed redundant imports, duplicate banners, and machine folder check.
- **`scripts/build_HDF5.sh`** / **`scripts/build_AMReX.sh`** — Added execution guards; no longer accept arguments (parent sets everything). Both export their install paths on skip for downstream scripts.

---

## [Unreleased] — 2026-08-29

### Added

- **`machines/local.gpu`** / **`machines/local.cpu`** — Local machine configuration files for RTX 4070 / Ubuntu 24.04 development. HDF5_ROOT points to conda env `hydroexa_env`. No `module purge` — local machines use direct paths.

### Changed

- **`scripts/lib.sh` (formerly machine_selection.sh)** — `local.gpu` and `local.cpu` compiler entries changed from `gcc`/`g++`/`gfortran` to `mpicc`/`mpicxx`/`mpif90` to ensure MPI wrappers match conda env's HDF5 transitive MPI dependencies.
- **`CMakeLists.txt`** — Removed `set_source_files_properties(... PROPERTIES LANGUAGE CUDA)` block that forced all HydroEXA sources through nvcc. nvcc's C++ parser cannot handle C++20 features (e.g., `std::numbers::pi_v`, `std::ssize`) used in AMReX headers. Sources now compile with mpicxx; AMReX handles its own CUDA compilation.

### Fixed

- **nvcc C++20 incompatibility** — HydroEXA sources compiled with mpicxx (g++), avoiding nvcc's limited C++20 support.
- **MPI library mismatch** — Runtime `undefined symbol: ompi_mpi_errors_throw_exceptions` resolved by using conda env's MPI wrappers via `mamba run -n hydroexa_env`, ensuring consistent MPI library linkage.
