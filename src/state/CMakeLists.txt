# libstate — AMR mesh state management (initialization, physics, refinement)
add_library(state OBJECT
    AmrMeshState_Init.cpp
    AmrMeshState_AMR.cpp
    AmrMeshState_Physics.cpp
    AmrMeshState_Utils.cpp
)

target_include_directories(state PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(state PUBLIC solvers utils io)

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    set_source_files_properties(AmrMeshState_Init.cpp AmrMeshState_AMR.cpp AmrMeshState_Physics.cpp AmrMeshState_Utils.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(state PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    set_source_files_properties(AmrMeshState_Init.cpp AmrMeshState_AMR.cpp AmrMeshState_Physics.cpp AmrMeshState_Utils.cpp PROPERTIES LANGUAGE HIP)
endif()
