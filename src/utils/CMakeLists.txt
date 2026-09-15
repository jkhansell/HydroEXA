# libutils — utility functions with no local dependencies
add_library(utils OBJECT
    MaskedAverageDown.cpp
)

target_include_directories(utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(utils PUBLIC
    AMReX::amrex
)

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    set_source_files_properties(MaskedAverageDown.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(utils PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    set_source_files_properties(MaskedAverageDown.cpp PROPERTIES LANGUAGE HIP)
endif()
