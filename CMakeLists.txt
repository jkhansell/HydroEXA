cmake_minimum_required(VERSION 3.20)

project(HydroEXA CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure AMReX settings (for internal fallback build)
set(AMReX_SPACEDIM 2 CACHE STRING "Spatial Dimension")

# Only enable MPI if a compiler wrapper (mpicc) is available;
# otherwise fall back to a serial build for local development.
if(NOT DEFINED AMReX_MPI)
  find_program(_MPICC_FOUND mpicc)
  set(AMReX_MPI ${_MPICC_FOUND} CACHE BOOL "Enable MPI" FORCE)
  mark_as_advanced(_MPICC_FOUND)
  if(AMReX_MPI)
    message(STATUS "MPI found via mpicc — enabling AMReX_MPI")
  else()
    message(STATUS "mpicc not found — building without MPI (local/dev mode)")
  endif()
endif()


find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache found and enabled: ${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found. Proceeding with normal compilation.")
endif()


# Load MPI (only when AMReX_MPI is enabled)
if(AMReX_MPI)
  find_package(MPI REQUIRED)
  message(STATUS "MPI found: ${MPI_C_VERSION}")
else()
  message(STATUS "MPI disabled — proceeding without it")
endif()


# ============================================================
# Load AMReX (Now hip::device or CUDA targets will resolve cleanly)
# ============================================================

if (DEFINED AMReX_ROOT)
  message(STATUS "Using external AMReX from ${AMReX_ROOT}")

  # Determine Backend directly from explicit configuration flag passed by shell
  if(HYDROEXA_GPU_BACKEND MATCHES ".*\\.gpu")

    if(HYDROEXA_GPU_BACKEND MATCHES "frontier.*")
      message(STATUS "Pre-detecting HIP environment...")
      enable_language(HIP)
      find_package(HIP REQUIRED)
      find_package(hiprand REQUIRED)
      find_package(rocsparse REQUIRED)
      find_package(rocprim REQUIRED)

    elseif(HYDROEXA_GPU_BACKEND MATCHES "juwelsbooster.*" OR HYDROEXA_GPU_BACKEND MATCHES "local.*" OR HYDROEXA_GPU_BACKEND MATCHES "kabre.*")
      message(STATUS "Pre-detecting CUDA environment...")
      enable_language(CUDA)
      find_package(CUDAToolkit REQUIRED)
    endif()

  endif()

  find_package(AMReX REQUIRED)
else()
  add_subdirectory(tpl/amrex)
endif()


# ============================================================
# Load HDF5 (Required for IOHandler and input parsing)
# Prefer local TPL build, fall back to system install
# ============================================================
if(DEFINED HDF5_ROOT)
  message(STATUS "Using external HDF5 from ${HDF5_ROOT}")
else()
  set(HDF5_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tpl/hdf5/install" CACHE PATH "HDF5 install prefix")
  message(STATUS "HDF5 not specified, defaulting to TPL: ${HDF5_ROOT}")
endif()
find_package(HDF5 REQUIRED COMPONENTS C)

# ============================================================
# Enable GPU language now that AMReX has set AMReX_GPU_BACKEND
# ============================================================
if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    enable_language(HIP)
endif()

# ============================================================
# Build from src/ subdirectory structure
# ============================================================
add_subdirectory(src)