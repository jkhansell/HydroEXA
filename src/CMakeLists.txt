# Build sub-libraries first (OBJECT targets must exist before executable)
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
    MPI::MPI_CXX
    MPI::MPI_C
)

if(HYDROEXA_GPU_BACKEND MATCHES "kabre.*")
    target_link_options(HydroEXA PRIVATE "-Wl,--no-as-needed" "-lmpi")
endif()

if(HDF5_IS_PARALLEL)
    target_compile_definitions(HydroEXA PRIVATE H5_HAVE_PARALLEL)
endif()


if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    set_source_files_properties(HydroEXA.cpp HydroEXADriver.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(HydroEXA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    set_source_files_properties(HydroEXA.cpp HydroEXADriver.cpp PROPERTIES LANGUAGE HIP)
endif()