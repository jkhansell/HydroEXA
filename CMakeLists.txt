# =============================================================================
# HydroEXA — superbuild
# =============================================================================
# Builds, in order:
#
#   1. HDF5     tpl/hdf5   (CMake)  -> deps/<toolchain>/<machine>/hdf5
#   2. AMReX    tpl/amrex  (CMake)  -> deps/<toolchain>/<machine>/amrex_<PREC>_<opthash>
#   3. HydroEXA src/       (CMake)  -> install/<preset>
#
# Everything is selected by a preset. No machine name appears in this file: the
# backend, GPU architecture, precision and AMReX option set are cache variables
# set by the preset (see cmake/presets/). The toolchain — modules, CC/CXX/FC —
# comes from machines/<machine>, sourced by scripts/with-machine before cmake
# runs.
#
#   ./scripts/with-machine kabreV100.gpu -- cmake --preset kabreV100-gpu-double
#   ./scripts/with-machine kabreV100.gpu -- cmake --build --preset kabreV100-gpu-double
#
# HydroEXA itself remains a standalone project: configure src/ directly against
# any AMReX + HDF5 installation (see src/CMakeLists.txt). This root project is
# a *different* project and must never add_subdirectory(src).
# =============================================================================

cmake_minimum_required(VERSION 3.25)

project(HydroEXA-Superbuild VERSION 0.1.0 LANGUAGES C CXX)

include(ExternalProject)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# 0. Fragment tripwire
# -----------------------------------------------------------------------------
# Backend / card / precision / buildtype presets are documentation, not builds.
# They are deliberately visible (so `cmake --list-presets` documents the whole
# matrix) and therefore selectable — so catch a partial configuration here
# instead of letting AMReX silently default to DOUBLE on no precision at all.
if(HYDROEXA_FRAGMENT_ONLY)
    if(HYDROEXA_PRESET)
        set(_frag_name "${HYDROEXA_PRESET}")
    else()
        set(_frag_name "this preset")
    endif()
    message(FATAL_ERROR
        "\n"
        "  '${_frag_name}' is a preset fragment (backend / card / precision /\n"
        "  build type), not a build.\n"
        "\n"
        "  Pick a leaf preset, for example:\n"
        "      ./scripts/with-machine kabreV100.gpu -- cmake --preset kabreV100-gpu-double\n"
        "\n"
        "  List every preset:\n"
        "      cmake --list-presets\n")
endif()

# -----------------------------------------------------------------------------
# 1. Required inputs
# -----------------------------------------------------------------------------
foreach(_var HYDROEXA_PRESET HYDROEXA_MACHINE AMReX_PRECISION HYDROEXA_GPU_BACKEND)
    if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
        message(FATAL_ERROR
            "Required cache variable ${_var} is not set.\n"
            "Configure through a preset: cmake --preset <leaf>  (see cmake --list-presets)")
    endif()
endforeach()

if(NOT AMReX_PRECISION MATCHES "^(SINGLE|DOUBLE)$")
    message(FATAL_ERROR
        "AMReX_PRECISION must be SINGLE or DOUBLE (got '${AMReX_PRECISION}')")
endif()

if(NOT HYDROEXA_GPU_BACKEND MATCHES "^(CUDA|HIP|NONE)$")
    message(FATAL_ERROR
        "HYDROEXA_GPU_BACKEND must be CUDA, HIP or NONE (got '${HYDROEXA_GPU_BACKEND}')")
endif()

option(HYDROEXA_MPI_LINK_WORKAROUND
    "Force-link MPI (needed by some MPICH + static builds)" OFF)

# -----------------------------------------------------------------------------
# 2. Toolchain consistency
# -----------------------------------------------------------------------------
# The profile and the preset both describe the card. They should agree; a
# mismatch means the preset was written for different hardware than the
# profile loaded, which is exactly the silent failure this redesign removes.
if(DEFINED ENV{HYDROEXA_GPU_BACKEND}
   AND NOT "$ENV{HYDROEXA_GPU_BACKEND}" STREQUAL "${HYDROEXA_GPU_BACKEND}")
    message(WARNING
        "Preset says HYDROEXA_GPU_BACKEND=${HYDROEXA_GPU_BACKEND} but the machine "
        "profile exports $ENV{HYDROEXA_GPU_BACKEND}. Using the preset value.")
endif()
if(DEFINED ENV{HYDROEXA_GPU_ARCH}
   AND HYDROEXA_GPU_ARCH
   AND NOT "$ENV{HYDROEXA_GPU_ARCH}" STREQUAL "${HYDROEXA_GPU_ARCH}")
    message(WARNING
        "Preset says HYDROEXA_GPU_ARCH=${HYDROEXA_GPU_ARCH} but the machine "
        "profile exports $ENV{HYDROEXA_GPU_ARCH}. Using the preset value.")
endif()

# -----------------------------------------------------------------------------
# 3. Dependency prefix
# -----------------------------------------------------------------------------
# Keyed by (toolchain, machine). The toolchain tag comes from the C compiler's
# link line, so overriding the MPI wrapper via `with-machine --cc` automatically
# invalidates HDF5 — which must be built against the same MPI ABI as HydroEXA.
# This is why toolchain overrides belong in the shell layer, not in a preset.
execute_process(
    COMMAND ${CMAKE_C_COMPILER} -show
    OUTPUT_VARIABLE _mpi_link_line
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _mpi_show_rc)

if(_mpi_show_rc EQUAL 0 AND NOT "${_mpi_link_line}" STREQUAL "")
    string(MD5 _toolchain_hash "${_mpi_link_line}")
else()
    string(MD5 _toolchain_hash "${CMAKE_C_COMPILER}")
endif()
string(SUBSTRING "${_toolchain_hash}" 0 8 _toolchain_tag)

set(HYDROEXA_DEPS_PREFIX "${CMAKE_SOURCE_DIR}/tpl/install/${_toolchain_tag}/${HYDROEXA_MACHINE}"
    CACHE PATH "Precision-agnostic dependency prefix (HDF5 lives here)")
set(HYDROEXA_HDF5_PREFIX "${HYDROEXA_DEPS_PREFIX}/hdf5")
set(HYDROEXA_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install/${HYDROEXA_PRESET}"
    CACHE PATH "HydroEXA install prefix")

# -----------------------------------------------------------------------------
# 4. AMReX option set
# -----------------------------------------------------------------------------
# The option set is part of AMReX's install identity: a different option set is
# a different AMReX and must not be reused from an existing tree. This replaces
# the old existence-based skip guard, which was blind to options.
option(AMReX_MPI            "Build AMReX with MPI"       ON)
option(AMReX_OMP            "Build AMReX with OpenMP"    ON)
option(AMReX_PARTICLES      "Build AMReX particles"      ON)
option(AMReX_LINEAR_SOLVERS "Build AMReX linear solvers" ON)
option(AMReX_HDF5           "Build AMReX HDF5 support"   ON)
set(AMReX_SPACEDIM 2 CACHE STRING "Spatial dimension")

# AMReX's TinyProfiler is a compile-time feature of AMReX itself: enabling it
# defines AMREX_TINY_PROFILING in AMReX_Config.H, which turns every BL_PROFILE
# in AMReX *and* in the solver into a scope timer. So it has to be an AMReX
# option here (and in the option key below), not a solver-side flag.
option(HYDROEXA_TINY_PROFILE
    "Build AMReX with TinyProfiler and instrument the solver with BL_PROFILE" OFF)

set(_amrex_gpu_args)
if(HYDROEXA_GPU_BACKEND STREQUAL "CUDA")
    if(NOT HYDROEXA_GPU_ARCH)
        message(FATAL_ERROR
            "HYDROEXA_GPU_BACKEND=CUDA requires HYDROEXA_GPU_ARCH (e.g. 70, 89, native)")
    endif()
    list(APPEND _amrex_gpu_args
        -DAMReX_GPU_BACKEND=CUDA
        -DCMAKE_CUDA_ARCHITECTURES=${HYDROEXA_GPU_ARCH})
elseif(HYDROEXA_GPU_BACKEND STREQUAL "HIP")
    if(NOT HYDROEXA_GPU_ARCH)
        message(FATAL_ERROR
            "HYDROEXA_GPU_BACKEND=HIP requires HYDROEXA_GPU_ARCH (e.g. gfx90a)")
    endif()
    # AMReX requires AMReX_AMD_ARCH when the HIP backend is selected
    # (Tools/CMake/AMReXOptions.cmake: FATAL_ERROR otherwise).
    list(APPEND _amrex_gpu_args
        -DAMReX_GPU_BACKEND=HIP
        -DAMReX_AMD_ARCH=${HYDROEXA_GPU_ARCH})
else()
    list(APPEND _amrex_gpu_args -DAMReX_GPU_BACKEND=NONE)
endif()

set(_amrex_option_key
    "mpi=${AMReX_MPI};omp=${AMReX_OMP};particles=${AMReX_PARTICLES}"
    ";linear=${AMReX_LINEAR_SOLVERS};hdf5=${AMReX_HDF5};dim=${AMReX_SPACEDIM}"
    ";prec=${AMReX_PRECISION};gpu=${HYDROEXA_GPU_BACKEND};arch=${HYDROEXA_GPU_ARCH}"
    ";tiny=${HYDROEXA_TINY_PROFILE}")
string(MD5 _amrex_hash "${_amrex_option_key}")
string(SUBSTRING "${_amrex_hash}" 0 6 _amrex_opt_tag)

set(HYDROEXA_AMREX_PREFIX
    "${HYDROEXA_DEPS_PREFIX}/amrex_${AMReX_PRECISION}_${_amrex_opt_tag}"
    CACHE PATH "AMReX install prefix")

# -----------------------------------------------------------------------------
# 5. Stage 1 — HDF5
# -----------------------------------------------------------------------------
# HDF5 1.14.6 supports a native CMake build (cmake_minimum_required 3.18), so
# the autotools stage is gone. CMAKE_C_COMPILER is passed explicitly because
# ExternalProject starts from a fresh cache: that explicit hand-off is what
# keeps HDF5 on the same MPI ABI as AMReX and HydroEXA. HDF5 sets
# CMAKE_POSITION_INDEPENDENT_CODE itself, so no -fPIC CFLAGS hack is needed.
ExternalProject_Add(hdf5
    SOURCE_DIR  ${CMAKE_SOURCE_DIR}/tpl/hdf5
    INSTALL_DIR ${HYDROEXA_HDF5_PREFIX}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_STATIC_LIBS=ON
        -DHDF5_ENABLE_PARALLEL=ON
        -DHDF5_BUILD_HL_LIB=ON
        -DHDF5_BUILD_TOOLS=OFF
        -DHDF5_BUILD_CPP_LIB=OFF
        -DHDF5_BUILD_FORTRAN=OFF
    BUILD_BYPRODUCTS ${HYDROEXA_HDF5_PREFIX}/lib/libhdf5.a
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
)

# -----------------------------------------------------------------------------
# 6. Stage 2 — AMReX
# -----------------------------------------------------------------------------
# Mirrors scripts/build_AMReX.sh option-for-option, plus the backend variables
# that used to live in the machine profiles' GPU_FLAGS string.
ExternalProject_Add(amrex
    SOURCE_DIR  ${CMAKE_SOURCE_DIR}/tpl/amrex
    INSTALL_DIR ${HYDROEXA_AMREX_PREFIX}
    DEPENDS     hdf5
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DHDF5_ROOT=${HYDROEXA_HDF5_PREFIX}
        -DAMReX_MPI=${AMReX_MPI}
        -DAMReX_OMP=${AMReX_OMP}
        -DAMReX_PARTICLES=${AMReX_PARTICLES}
        -DAMReX_SPACEDIM=${AMReX_SPACEDIM}
        -DAMReX_LINEAR_SOLVERS=${AMReX_LINEAR_SOLVERS}
        -DAMReX_PRECISION=${AMReX_PRECISION}
        -DAMReX_HDF5=${AMReX_HDF5}
        -DAMReX_TINY_PROFILE=${HYDROEXA_TINY_PROFILE}
        ${_amrex_gpu_args}
    BUILD_BYPRODUCTS ${HYDROEXA_AMREX_PREFIX}/lib/cmake/AMReX/AMReXConfig.cmake
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
)

# -----------------------------------------------------------------------------
# 7. Stage 3 — HydroEXA
# -----------------------------------------------------------------------------
# BUILD_ALWAYS keeps the edit -> build loop honest: the build+install steps run
# on every invocation (as the old script did), while the configure step stays
# stamp-driven — `cmake --build` re-runs configure by itself when a CMakeLists
# changes. No CMakeCache.txt deletion, no rm -rf.
ExternalProject_Add(hydroexa
    SOURCE_DIR  ${CMAKE_SOURCE_DIR}/src
    INSTALL_DIR ${HYDROEXA_INSTALL_PREFIX}
    DEPENDS     amrex
    BUILD_ALWAYS TRUE
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DAMReX_ROOT=${HYDROEXA_AMREX_PREFIX}/lib/cmake/AMReX
        -DHDF5_ROOT=${HYDROEXA_HDF5_PREFIX}
        -DHYDROEXA_PRESET=${HYDROEXA_PRESET}
        -DHYDROEXA_MACHINE=${HYDROEXA_MACHINE}
        -DHYDROEXA_GPU_BACKEND=${HYDROEXA_GPU_BACKEND}
        -DHYDROEXA_GPU_ARCH=${HYDROEXA_GPU_ARCH}
        -DHYDROEXA_MPI_LINK_WORKAROUND=${HYDROEXA_MPI_LINK_WORKAROUND}
        -DHYDROEXA_TINY_PROFILE=${HYDROEXA_TINY_PROFILE}
    USES_TERMINAL_BUILD   TRUE
    USES_TERMINAL_INSTALL TRUE
)

# -----------------------------------------------------------------------------
# 8. Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "HydroEXA superbuild")
message(STATUS "  preset         : ${HYDROEXA_PRESET}")
message(STATUS "  machine        : ${HYDROEXA_MACHINE}")
message(STATUS "  backend / arch : ${HYDROEXA_GPU_BACKEND} / ${HYDROEXA_GPU_ARCH}")
message(STATUS "  precision      : ${AMReX_PRECISION}")
message(STATUS "  tiny profiler  : ${HYDROEXA_TINY_PROFILE}")
message(STATUS "  build type     : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C compiler     : ${CMAKE_C_COMPILER}")
message(STATUS "  C++ compiler   : ${CMAKE_CXX_COMPILER}")
message(STATUS "  toolchain tag  : ${_toolchain_tag}")
message(STATUS "  deps prefix    : ${HYDROEXA_DEPS_PREFIX}")
message(STATUS "  AMReX prefix   : ${HYDROEXA_AMREX_PREFIX}")
message(STATUS "  install prefix : ${HYDROEXA_INSTALL_PREFIX}")
message(STATUS "")
