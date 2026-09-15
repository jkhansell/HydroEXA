# libsolvers — Riemann solvers
add_library(solvers OBJECT
    Roe.cpp
)

target_include_directories(solvers PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(solvers PUBLIC
    AMReX::amrex
)

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    set_source_files_properties(Roe.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(solvers PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    set_source_files_properties(Roe.cpp PROPERTIES LANGUAGE HIP)
endif()
