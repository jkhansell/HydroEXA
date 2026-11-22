# Building and compiling

A HydroEXA build is three stages -- HDF5, AMReX, HydroEXA -- and one
description of *what* those stages should produce. This page is the procedure:
the fast path, the preset taxonomy behind it, where the artifacts land, and the
failures that are easy to misread.

There are two entry points and they share the same sources:

| Entry point | Use it when |
|---|---|
| **Superbuild presets** (`cmake --preset …`, this page) | you want the dependencies built from source, reproducibly, from one command |
| **Legacy script** (`scripts/build_HydroEXA.sh`, [below](#the-legacy-pipeline)) | you are reproducing an older build tree, or want the shell pipeline you already know |

## The shape of a build

| Stage | What it is | Where it lands | Shared by |
|---|---|---|---|
| HDF5 1.14.6 | parallel HDF5, static, HL library | `deps/<toolchain>/<machine>/hdf5` | every precision of that toolchain |
| AMReX `26.05` | AMR framework, CUDA or HIP or CPU | `deps/<toolchain>/<machine>/amrex_<PRECISION>_<hash>` | builds with the same option set |
| HydroEXA | the solver | `install/<preset>/bin/HydroEXA` | -- |

The repository root is a **superbuild**: it does not compile the solver itself,
it drives three `ExternalProject` stages in order and passes each one the prefix
the next one needs. `src/` is a standalone CMake project that can be configured
against any AMReX and HDF5 installation, which is why the superbuild never does
`add_subdirectory(src)` -- the solver project does not know or care that a
superbuild exists.

## Quick start

On Kabré, from a compute node with an allocation:

```bash
./scripts/with-machine kabreV100.gpu -- cmake --preset kabreV100-gpu-double
./scripts/with-machine kabreV100.gpu -- cmake --build build/kabreV100-gpu-double -j 24
```

The binary is `install/kabreV100-gpu-double/bin/HydroEXA`. Nothing else is
required: no modules to load by hand, no `HDF5_ROOT`, no `AMReX_ROOT`.

`scripts/with-machine` sources `machines/<machine>` (module purge + the profile's
`module load`s, then `CC`/`CXX`/`FC`) and then `exec`s the command you gave it.
That split is deliberate and is the most important idea on this page:

* **the machine profile owns the toolchain** -- modules, compiler wrappers,
  MPI implementation;
* **the preset owns the build** -- backend, GPU architecture, precision, build
  type, and therefore which dependency prefix is used.

A preset never loads a module and a profile never chooses a build option. The
reason is the MPI ABI: HDF5, AMReX and HydroEXA have to be built by the *same*
MPI, so the dependency prefix is keyed by the toolchain (see
[Where artifacts land](#where-artifacts-land)). Switching compilers is therefore
a toolchain change, not a build-option change.

## Choosing what to build

```bash
cmake --list-presets
```

```
  "backend-cuda"         - CUDA backend (fragment)
  "precision-double"     - amrex::Real = double (fragment)
  "buildtype-release"    - Release (fragment)
  "card-kabreV100.gpu"   - Kabre V100 - Volta sm_70 (fragment)
  "kabreV100-gpu-double" - Kabre V100 / DOUBLE
  "kabreV100-gpu-single" - Kabre V100 / SINGLE
  ...
```

Presets come in four layers. The first three are **fragments** -- they describe
one axis and are never a build on their own. The fourth is a **leaf**, and a leaf
is the only thing you are meant to select:

| Layer | Answers | Example |
|---|---|---|
| `backend-*` | which GPU backend | `backend-cuda`, `backend-hip`, `backend-cpu` |
| `card-*` | which card, i.e. GPU architecture + MPI quirks | `card-kabreV100.gpu` (CUDA, sm_70) |
| `precision-*`, `buildtype-*` | `amrex::Real` and optimisation level | `precision-double`, `buildtype-release` |
| `<card>-<precision>` | everything at once | `kabreV100-gpu-double` |

Fragments are **visible**, not hidden, and selecting one is a `FATAL_ERROR` that
tells you what to do instead:

```
Pick a leaf preset, for example:
    ./scripts/with-machine kabreV100.gpu -- cmake --preset kabreV100-gpu-double
```

Visibility is the point. A fragment that is listed but refuses to configure
documents the axis without inviting a half-specified build; a hidden one would
be invisible to the person trying to understand what their machine *can* be.

There are no `buildPresets`, so builds name the binary directory:
`cmake --build build/<leaf>`. The configure preset already fixed that path, and a
build preset would only add a second name for it.

### Overriding the compiler

```bash
./scripts/with-machine kabreV100.gpu --cc /opt/mpich-4.2/bin/mpicxx -- \
    cmake --preset kabreV100-gpu-double
```

The override is applied *before* the preset is read, so the toolchain tag (and
therefore the dependency prefix) is computed from the compiler you actually use.
Putting `CMAKE_CXX_COMPILER` in a preset instead would let a build reuse an HDF5
built by a different MPI -- the failure that shows up much later as unresolved
`MPI_*` symbols or a silent mismatch between the HDF5 and the solver.

Other options: `--fc`, `--no-modules` (keep your current environment),
`--machines-dir`, `--list`.

## Where artifacts land

```
deps/<toolchain-tag>/<machine>/
    hdf5/                        parallel HDF5, precision-independent
    amrex_DOUBLE_e2e395/         AMReX for this precision and option set
build/<leaf>/                    superbuild tree (make/ninja files)
install/<leaf>/bin/HydroEXA      the solver
```

**Toolchain tag.** A short hash of the compiler's link line (`mpicxx -show`), not
the compiler's name. Two MPI builds of the same gcc version get different tags;
renaming a wrapper does not change the tag, re-pointing it does. This is what
replaces the old "does `install_<target>/hdf5` exist?" guard, which happily
reused a prefix built by a different toolchain.

**AMReX prefix.** Keyed by precision *and* a hash of the AMReX option set
(`amrex_DOUBLE_e2e395`), so a build with different AMReX options cannot pick up
a stale library. HDF5 stays precision-independent because it is.

The toolchain tag on a known-good Kabré build is `4c2a84ba`:

```
--   preset         : kabreV100-gpu-double
--   machine        : kabreV100.gpu
--   backend / arch : CUDA / 70
--   precision      : DOUBLE
--   toolchain tag  : 4c2a84ba
--   deps prefix    : .../deps/4c2a84ba/kabreV100.gpu
--   AMReX prefix   : .../deps/4c2a84ba/kabreV100.gpu/amrex_DOUBLE_e2e395
```

## Adding a machine

Two files, neither of them CMake. First the toolchain, `machines/<name>`:

```bash
# Toolchain only. Build options come from the presets.
if [ -z "${HYDROEXA_NO_MODULES:-}" ]; then
    module purge
    module load mpich/4.1.1 cuda/12.4.0 gcc/13.4.0
fi
export CC="${CC:-mpicc}" CXX="${CXX:-mpicxx}" FC="${FC:-mpif90}"
export HYDROEXA_GPU_BACKEND="CUDA"   # what the card is
export HYDROEXA_GPU_ARCH="70"
```

Then the build variants, in `CMakeUserPresets.json` (gitignored -- your machine,
your file). It inherits the card layer, so backend and architecture come for
free, and only the axes you care about need naming:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "mymachine-gpu-double",
      "displayName": "My machine / DOUBLE",
      "inherits": ["card-kabreV100.gpu", "precision-double", "buildtype-release"],
      "binaryDir": "${sourceDir}/build/${presetName}",
      "installDir": "${sourceDir}/install/${presetName}",
      "cacheVariables": {
        "HYDROEXA_FRAGMENT_ONLY": "OFF",
        "HYDROEXA_MACHINE": "mymachine",
        "HYDROEXA_PRESET": "${presetName}"
      }
    }
  ]
}
```

`HYDROEXA_FRAGMENT_ONLY=OFF` is what turns the tripwire off; `HYDROEXA_MACHINE`
only feeds the provenance strings. If your card is not one of the shipped ones,
copy a `cmake/presets/card-*.json` and change the architecture -- and note that
each card file must `include` `00-backends.json`, `01-precision.json` and
`02-buildtype.json` itself, because **an included preset file cannot inherit from
a sibling include** (`Inherited preset … is unreachable from preset's file`).

## Building against an existing AMReX and HDF5

`src/` is a complete project. This is the path for a machine where the
dependencies already exist, or for building HydroEXA out of tree:

```bash
cmake -S src -B build-hydroexa \
    -DAMReX_ROOT=<prefix>/lib/cmake/AMReX \
    -DHDF5_ROOT=<prefix> \
    -DHYDROEXA_GPU_BACKEND=CUDA \
    -DHYDROEXA_GPU_ARCH=70 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build-hydroexa -j 16
```

`HYDROEXA_GPU_BACKEND` is a capability (`CUDA`, `HIP`, `NONE`), never a machine
name: no filename in this project is compared against a hostname any more. The
provenance macros `HYDROEXA_BUILD_PRESET` / `HYDROEXA_BUILD_MACHINE` record which
preset and profile produced the binary; a manual build reports `standalone`.

## The legacy pipeline

`scripts/build_HydroEXA.sh <machine> <buildtype> <precision>` still works and
still runs the three stages, but it now goes through the same single source of
truth as the presets:

* a **missing machine profile is a hard error** (it used to print a warning and
  continue into a build with no compiler set);
* the `GPU_FLAGS` string is **derived** by `scripts/lib.sh` from the card
  description (`HYDROEXA_GPU_BACKEND` / `HYDROEXA_GPU_ARCH`) instead of being
  hand-written in each profile, so a profile cannot disagree with the presets
  about what the card is;
* the solver stage configures `-S "${HYDROEXA_SRC}/src"`, because the repository
  root is a superbuild now, not the solver project.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Inherited preset is unreachable from preset's file` | a card file inherits a fragment it does not itself `include` | each card file includes `00-backends.json`, `01-precision.json`, `02-buildtype.json` |
| `FATAL_ERROR` telling you to pick a leaf | you selected a fragment (`backend-cuda`, `precision-double`, …) | select a leaf, e.g. `kabreV100-gpu-double` |
| `undefined reference to 'inflateInit_'`, `compress2` | HDF5 was built with the deflate filter, but `find_package(HDF5)` reports only the HDF5 libraries | handled in `src/CMakeLists.txt`: `H5_HAVE_FILTER_DEFLATE` in `H5pubconf.h` triggers a `ZLIB::ZLIB` link |
| `AMReX.H:138: error: '__host__' does not name a type` | the source is being compiled by the host compiler, but AMReX was built with `AMREX_USE_GPU` | keep `LANGUAGE CUDA` on that source; only `nvcc` can parse those headers |
| `undefined reference to '__cudaRegisterLinkedBinary_…'` | `CUDA_SEPARABLE_COMPILATION` was turned off, but AMReX is built with `AMReX_GPU_RDC=ON` | keep it on, or rebuild AMReX with `-DAMReX_GPU_RDC=OFF` |
| `version 'GLIBCXX_3.4.32' not found` when running the binary | you ran it outside the machine environment | run it through `./scripts/with-machine <machine> -- …` |
| `find_package(AMReX)` demands a newer CMake than you have | `AMReXConfig.cmake` embeds the `cmake_minimum_required` of the CMake that built AMReX | build AMReX with the CMake you intend to use; the superbuild does this for you |
| the binary is ~164 MB | CUDA device images: `__nv_relfatbin` + `.nv_fatbin` dominate | expected; see [Binary size](#binary-size) |
| `szip` missing from `h5cc -showconfig` | HDF5's CMake path needs a `libaec` CMake package; Kabré ships autotools `libsz` only | deliberate: nothing in HydroEXA uses a compression filter |

### Binary size

A V100 build is 164 MB, of which `__nv_relfatbin` is 90.6 MB and `.nv_fatbin`
51.4 MB; host `.text` is 9.5 MB and there is no debug info, so `strip` recovers
about 2 MB. The cause is `LANGUAGE CUDA` on the solver's `.cpp` files combined
with `CUDA_SEPARABLE_COMPILATION`, which keeps a relocatable device image per
translation unit *and* the device-linked copy. This is unchanged from the
pre-superbuild binary (164,168,816 B), so it is not a build-system regression.

Neither half can be dropped as things stand — both were measured on `nukwa-03`
(CUDA 12.4, 2026-11-17):

| Attempt | Result |
|---|---|
| compile the `.cpp` files with `mpicxx` (no `LANGUAGE CUDA`) | `AMReX.H:138: error: '__host__' does not name a type` — AMReX's headers map `AMREX_GPU_HOST` to `__host__` whenever `AMREX_USE_GPU`, and include no CUDA header for a host compiler |
| keep `LANGUAGE CUDA`, drop `CUDA_SEPARABLE_COMPILATION` | link fails on `__cudaRegisterLinkedBinary_*` from `libamrex_2d.a`, which is itself built with `AMReX_GPU_RDC=ON` (AMReX's default) |
| add `-Xfatbin=-compress-all` | byte-identical binary (164,164,464 B); AMReX hard-codes `--generate-line-info` into its exported CUDA options, and that is what fills the fatbins |

So the size is a property of the dependency build, not of this project's
`CMakeLists.txt`. Shrinking it means rebuilding AMReX with `-DAMReX_GPU_RDC=OFF`
first — untested here, and a dependency decision rather than a solver one.

## Profiling

AMReX's [TinyProfiler](https://amrex-codes.github.io/amrex/docs_html/Profiling.html)
is a compile-time feature of AMReX: with `AMReX_TINY_PROFILE=ON`,
`AMReX_Config.H` defines `AMREX_TINY_PROFILING`, which turns every `BL_PROFILE`
scope in AMReX *and* in the solver into a timer. It therefore has to be an AMReX
option, not a solver-side flag — `HYDROEXA_TINY_PROFILE` does both halves:

```bash
# -tp leaf presets: same card/precision, AMReX built with TinyProfiler
./scripts/with-machine kabreV100.gpu -- cmake --preset kabreV100-gpu-double-tp
./scripts/with-machine kabreV100.gpu -- cmake --build build/kabreV100-gpu-double-tp -j 16
cd tests/dambreak && HYDROEXA_SKIP_PLOTS=1 \
    bash run.sh roe amr_fastregrid ../../install/kabreV100-gpu-double-tp/bin/HydroEXA 1 tp1
```

Two consequences of it being an AMReX option:

* the flag is part of the AMReX option key, so a profiling build gets its own
  `tpl/install/<toolchain-tag>/<machine>/amrex_<PRECISION>_<hash>` — it never
  reuses the uninstrumented library;
* `src/CMakeLists.txt` refuses to configure `HYDROEXA_TINY_PROFILE=ON` against an
  AMReX built without it, rather than silently producing a binary with no timers.

The report is printed at `amrex::Finalize()`, i.e. at the end of `log.txt`:

```
TinyProfiler total time across processes [min...avg...max]: 6.24 ... 6.24 ... 6.24
--------------------------------------------------------------------------------
Name                                          NCalls  Excl. Min  Excl. Avg  Excl. Max   Max %
--------------------------------------------------------------------------------
VisMF::Write(FabArray)                            62     0.9705     0.9705     0.9705  15.55%
FabArray::setVal()                             44152     0.6858     0.6858     0.6858  10.99%
HydroPhysBC::()                                20679     0.4969     0.4969     0.4969   7.96%
```

Read `Excl. Max` (exclusive, i.e. self time) and `Max %`, and use `NCalls` to
tell a cheap-but-frequent region from an expensive-but-rare one. Regions come
from `BL_PROFILE` in `src/HydroEXA.cpp`, `src/solvers/{Roe,HydrostaticRoe}.cpp`
and `src/state/AmrMeshState_*.cpp`; AMReX and AMReX's own headers are already
instrumented, so `FabArray::setVal()` and friends appear without any changes
here. Add a `BL_PROFILE("Name")` at the top of a function to widen the picture —
it is compiled out entirely when the option is off.

Instrumentation is not free, so compare a profiling build only against another
profiling build. The single-level costs are small but the region count is large:
`VisMF::Write` in the table above is I/O-bound and varies by a factor of three
between runs on a shared filesystem, which is why the interesting rows are the
compute ones.

Worked example — the physical-boundary fill in `include/boundaries/BCFill.H`
(2D dam break, `tests/dambreak`, `amr_fastregrid`, 1 rank, V100, 1361 coarse
steps). `io.plot_freq = -1` disables plotfiles, which is how to time the solver
itself: the `VisMF::Write` row alone swings by ±0.6 s between runs on a shared
filesystem.

| Implementation | BC row | Max % | Total |
|---|---|---|---|
| `PhysBCFunct<GpuBndryFuncFab<FabFillNoOp>>` rebuilt per level, no plotfiles | 2.434 s | 35.34 % | 6.886 s |
| cached device `BCRec`, direct `FilccCell`, no plotfiles | 0.4937 s | 10.01 % | 4.933 s |

That is 4.9× on the fill and 1.95 s of the 6.9 s run — all of the difference
sits in the one row, with no I/O in either number. With plotfiles enabled
(`io.plot_freq = 1`, ~31 dumps) the same pair reads 2.494 s / 8.449 s versus
0.497 s / 6.240 s, i.e. the same 2.0 s with ~1.5 s of I/O noise on top.

The trajectory is identical in both (`max|vel| = 0.128707045`, 1361 steps), so
the 2.0 s is pure overhead: AMReX's functor allocates an `AsyncArray<BCRec>` and
copies the `BCRec` set host→device for every FAB of every fill. Caching it once
per level is worth ~28 % of this run's wall time.

## What "it builds" means here

The superbuild was validated on `nukwa-03` (Tesla V100-PCIE-32GB, sm_70) with
MPICH 4.1.1, gcc 13.4, CUDA 12.4 and `/usr/bin/cmake` 3.26.5:

* all three stages configured, built and installed from an empty `deps/`;
* `readelf -d` shows `libmpi.so.12` in `DT_NEEDED`, so the MPI link workaround
  still does what it did under the old machine-name match;
* `tests/LakeAtRestHydrostaticRoe` runs 1925 coarse steps to `t = 30.01` with
  `max|vel| = 3.1e-15` on levels 0--2, i.e. the well-balanced property of
  `HydrostaticRoe` survives the rebuild.

A link is not the goal; a lake that stays at rest is.
