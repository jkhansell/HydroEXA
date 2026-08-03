cmake_minimum_required(VERSION 3.20)
project(HydroEXA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure AMReX settings (for internal fallback build)
set(AMReX_SPACEDIM 2 CACHE STRING "Spatial Dimension")
set(AMReX_MPI ON CACHE BOOL "Enable MPI")


find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache found and enabled: ${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found. Proceeding with normal compilation.")
endif()


# Load MPI
find_package(MPI REQUIRED)


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
# ============================================================
find_package(HDF5 REQUIRED COMPONENTS C)
# ============================================================
# Target Configuration
# ============================================================

set(HYDROEXA_SOURCES
    src/HydroEXADriver.cpp
    src/HydroEXA.cpp
    src/io/IOHandler.cpp
    src/io/Checkpointer.cpp
    src/state/AmrMeshState_Init.cpp
    src/state/AmrMeshState_AMR.cpp
    src/state/AmrMeshState_Physics.cpp
    src/state/AmrMeshState_Utils.cpp
    src/solvers/Roe.cpp
    src/utils/MaskedAverageDown.cpp
)

add_executable(HydroEXA ${HYDROEXA_SOURCES})

target_include_directories(HydroEXA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(HydroEXA PUBLIC AMReX::amrex ${HDF5_LIBRARIES} MPI::MPI_CXX MPI::MPI_C)

if(HYDROEXA_GPU_BACKEND MATCHES "kabre.*") 
  # Add this right below target_link_libraries:
  target_link_options(HydroEXA PRIVATE "-Wl,--no-as-needed" "-lmpi")
endif()

if (HDF5_IS_PARALLEL)
  target_compile_definitions(HydroEXA PRIVATE H5_HAVE_PARALLEL)
endif()

# ============================================================
# Enforce GPU language compilation rules on standard variables
# ============================================================

# Detect CUDA or HIP backend directly from AMReX configuration flags
if(AMReX_GPU_BACKEND STREQUAL "CUDA" OR HYDROEXA_GPU_BACKEND MATCHES "kabre.*|juwelsbooster.*|local.*")
    message(STATUS "Enforcing CUDA compilation language on all HydroEXA sources.")
    set_target_properties(HydroEXA PROPERTIES 
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
    )
    set_source_files_properties(${HYDROEXA_SOURCES} PROPERTIES LANGUAGE CUDA)

elseif(AMReX_GPU_BACKEND STREQUAL "HIP" OR HYDROEXA_GPU_BACKEND MATCHES "frontier.*")
    message(STATUS "Enforcing HIP compilation language on all HydroEXA sources.")
    set_target_properties(HydroEXA PROPERTIES HIP_STANDARD 17)
    set_source_files_properties(${HYDROEXA_SOURCES} PROPERTIES LANGUAGE HIP)
endif()