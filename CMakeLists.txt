cmake_minimum_required(VERSION 3.20)
project(HydroEXA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure AMReX
set(AMReX_SPACEDIM 2 CACHE STRING "Spatial Dimension")
set(AMReX_MPI ON CACHE BOOL "Enable MPI")
set(AMReX_OMP OFF CACHE BOOL "Enable OpenMP")
set(AMReX_GPU_BACKEND NONE CACHE STRING "GPU Backend") # Can be CUDA, HIP, SYCL

add_subdirectory(tpl/amrex)

add_executable(HydroEXA 
    src/driver/main.cpp
    src/runtime/HydroLevel.cpp
)

target_include_directories(HydroEXA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(HydroEXA PRIVATE amrex)
