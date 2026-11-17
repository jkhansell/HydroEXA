cmake_minimum_required(VERSION 3.20)

# =============================================================================
# HydroEXA — standalone project
# =============================================================================
# Configure this directory directly against any AMReX + HDF5 installation:
#
#   cmake -S src -B build \
#       -DAMReX_ROOT=<prefix>/lib/cmake/AMReX \
#       -DHDF5_ROOT=<prefix> \
#       -DHYDROEXA_GPU_BACKEND=CUDA \
#       -DHYDROEXA_GPU_ARCH=70
#
# The superbuild in the repository root does this for you, in order, with the
# dependencies it builds itself. This file must not depend on anything from the
# parent directory except include/ and tpl/ being present.
# =============================================================================

project(HydroEXA VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# GPU backend capability
# -----------------------------------------------------------------------------
# One of CUDA, HIP, NONE. This is a *capability*, not a machine identifier:
# no machine name appears anywhere in this file or below.
set(HYDROEXA_GPU_BACKEND "NONE" CACHE STRING "GPU backend: CUDA, HIP or NONE")
set_property(CACHE HYDROEXA_GPU_BACKEND PROPERTY STRINGS CUDA HIP NONE)
set(HYDROEXA_GPU_ARCH "" CACHE STRING
    "GPU architecture (CUDA: 70/89/native, HIP: gfx90a)")

option(HYDROEXA_MPI_LINK_WORKAROUND
    "Force-link MPI (needed by some MPICH + static builds)" OFF)

# Provenance: which preset produced this binary. Defaults to a manual build.
if(NOT HYDROEXA_PRESET)
    set(HYDROEXA_PRESET "standalone" CACHE STRING "Preset that configured this build")
endif()
if(NOT HYDROEXA_MACHINE)
    set(HYDROEXA_MACHINE "unknown" CACHE STRING "Machine profile used for this build")
endif()

# -----------------------------------------------------------------------------
# ccache
# -----------------------------------------------------------------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache found and enabled: ${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found. Proceeding with normal compilation.")
endif()

# -----------------------------------------------------------------------------
# AMReX
# -----------------------------------------------------------------------------
set(AMReX_SPACEDIM 2 CACHE STRING "Spatial Dimension")

# The backend language must be enabled before find_package(AMReX) so that
# AMReX's device targets resolve cleanly.
if(HYDROEXA_GPU_BACKEND STREQUAL "CUDA")
    if(HYDROEXA_GPU_ARCH AND NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "${HYDROEXA_GPU_ARCH}")
    endif()
    message(STATUS "Pre-detecting CUDA environment...")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
elseif(HYDROEXA_GPU_BACKEND STREQUAL "HIP")
    if(HYDROEXA_GPU_ARCH AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "${HYDROEXA_GPU_ARCH}")
    endif()
    message(STATUS "Pre-detecting HIP environment...")
    enable_language(HIP)
    find_package(HIP REQUIRED)
    find_package(hiprand REQUIRED)
    find_package(rocsparse REQUIRED)
    find_package(rocprim REQUIRED)
endif()

if(NOT DEFINED AMReX_ROOT)
    message(FATAL_ERROR
        "AMReX_ROOT is not set. Point it at the directory containing "
        "AMReXConfig.cmake:\n"
        "    -DAMReX_ROOT=<prefix>/lib/cmake/AMReX\n"
        "Build AMReX first (see scripts/build_AMReX.sh), or use the superbuild "
        "presets from the repository root.")
endif()
message(STATUS "Using external AMReX from ${AMReX_ROOT}")
find_package(AMReX REQUIRED)

if(AMReX_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "MPI found: ${MPI_C_VERSION}")
endif()

# -----------------------------------------------------------------------------
# HDF5
# -----------------------------------------------------------------------------
# HDF5_ROOT is honoured by FindHDF5 when set; otherwise the system is searched.
if(DEFINED HDF5_ROOT)
    message(STATUS "Using external HDF5 from ${HDF5_ROOT}")
endif()
find_package(HDF5 REQUIRED COMPONENTS C)

# --- Filter libraries -------------------------------------------------------
# HDF5's deflate filter calls zlib, but find_package(HDF5) reports only the
# HDF5 libraries themselves: the autotools wrapper used to leak `-lz` through
# `h5cc -show`, and the CMake build's wrapper emits only -I/-L. So ask the HDF5
# build itself — the generated H5pubconf.h says which filters are compiled in —
# and link what the filter needs. Without this the link fails on inflateInit_,
# compress2, etc.
set(_hdf5_filter_libs "")
set(_hdf5_pubconf "")
foreach(_hdf5_inc IN LISTS HDF5_INCLUDE_DIRS HDF5_C_INCLUDE_DIR HDF5_INCLUDE_DIR)
    if(EXISTS "${_hdf5_inc}/H5pubconf.h")
        set(_hdf5_pubconf "${_hdf5_inc}/H5pubconf.h")
        break()
    endif()
endforeach()

if(_hdf5_pubconf)
    file(STRINGS "${_hdf5_pubconf}" _hdf5_deflate
        REGEX "^#define H5_HAVE_FILTER_DEFLATE 1")
    if(_hdf5_deflate)
        find_package(ZLIB REQUIRED)
        list(APPEND _hdf5_filter_libs ZLIB::ZLIB)
        message(STATUS "HDF5 deflate filter enabled: linking zlib")
    endif()
else()
    message(WARNING
        "H5pubconf.h not found under the HDF5 include directories; assuming "
        "HDF5 was built without external filters. If the link fails on "
        "inflateInit_/compress2, link zlib explicitly.")
endif()

# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------
add_subdirectory(utils)
add_subdirectory(io)
add_subdirectory(solvers)
add_subdirectory(state)

# Main executable — driver + all object file bundles
add_executable(HydroEXA
    HydroEXADriver.cpp
    HydroEXA.cpp
    $<TARGET_OBJECTS:utils>
    $<TARGET_OBJECTS:io>
    $<TARGET_OBJECTS:solvers>
    $<TARGET_OBJECTS:state>
)

install(TARGETS HydroEXA
    RUNTIME DESTINATION bin
)

target_include_directories(HydroEXA PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_link_libraries(HydroEXA PUBLIC
    AMReX::amrex
    ${HDF5_LIBRARIES}
    ${_hdf5_filter_libs}
    MPI::MPI_CXX
    MPI::MPI_C
)

# Machine-name matching is gone: this is a named capability that any machine
# may switch on, set from the card preset.
if(HYDROEXA_MPI_LINK_WORKAROUND)
    target_link_options(HydroEXA PRIVATE "-Wl,--no-as-needed" "-lmpi")
endif()

if(HDF5_IS_PARALLEL)
    target_compile_definitions(HydroEXA PRIVATE H5_HAVE_PARALLEL)
endif()

# Provenance, queryable at runtime.
target_compile_definitions(HydroEXA PRIVATE
    HYDROEXA_BUILD_PRESET="${HYDROEXA_PRESET}"
    HYDROEXA_BUILD_MACHINE="${HYDROEXA_MACHINE}"
)

# NOTE: device compilation of .cpp sources is preserved exactly as before.
# AGENTS.md claims nvcc cannot parse the C++20 features used by AMReX headers,
# yet every src/*/CMakeLists.txt does this and the builds work. Unresolved;
# deliberately not changed here.
if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    set_source_files_properties(HydroEXA.cpp HydroEXADriver.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(HydroEXA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    set_source_files_properties(HydroEXA.cpp HydroEXADriver.cpp PROPERTIES LANGUAGE HIP)
endif()
