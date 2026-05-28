cmake_minimum_required(VERSION 3.20)
project(HydroEXA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure AMReX
set(AMReX_SPACEDIM 2 CACHE STRING "Spatial Dimension")
set(AMReX_MPI ON CACHE BOOL "Enable MPI")

if (DEFINED AMReX_ROOT)
  message(STATUS "Using external AMReX from ${AMReX_ROOT}")
  find_package(AMReX REQUIRED)
else()
  add_subdirectory(tpl/amrex)
endif()

add_executable(HydroEXA 
    src/main.cpp
    src/HydroEXA_advance.cpp
    src/HydroEXA_bcfill.cpp
    src/HydroEXA_io.cpp
    src/HydroEXA_setup.cpp
    src/HydroEXA_tagging.cpp
    src/HydroEXABld.cpp
    src/HydroEXA.cpp
)


target_include_directories(HydroEXA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(HydroEXA PUBLIC AMReX::amrex)