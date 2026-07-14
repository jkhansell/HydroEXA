cmake_minimum_required(VERSION 3.20)
project(HydroEXA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure AMReX settings (for internal fallback build)
set(AMReX_SPACEDIM 2 CACHE STRING "Spatial Dimension")
set(AMReX_MPI ON CACHE BOOL "Enable MPI")

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

    elseif(HYDROEXA_GPU_BACKEND MATCHES "juwelsbooster.*" OR HYDROEXA_GPU_BACKEND MATCHES "local.*")
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
    src/state/AmrMeshState.cpp
)

add_executable(HydroEXA ${HYDROEXA_SOURCES})

target_include_directories(HydroEXA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(HydroEXA PUBLIC AMReX::amrex ${HDF5_LIBRARIES})

if (HDF5_IS_PARALLEL)
  target_compile_definitions(HydroEXA PRIVATE H5_HAVE_PARALLEL)
endif()

# ============================================================
# Enforce GPU language compilation rules on standard variables
# ============================================================
# if(TARGET AMReX::hip)
#   set_target_properties(HydroEXA PROPERTIES HIP_STANDARD 17)
#   set_source_files_properties(${HYDROEXA_SOURCES} PROPERTIES LANGUAGE HIP)
#   message(STATUS "Forcing HIP language rules on compilation source targets.")
#   
# elseif(TARGET AMReX::cuda)
#   set_target_properties(HydroEXA PROPERTIES CUDA_STANDARD 17)
#   set_source_files_properties(${HYDROEXA_SOURCES} PROPERTIES LANGUAGE CUDA)
#   message(STATUS "Forcing CUDA language rules on compilation source targets.")
# endif()