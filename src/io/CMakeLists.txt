# libio — I/O handlers (native + HDF5 plotfile, checkpointing)
add_library(io OBJECT
    IOHandler.cpp
    Checkpointer.cpp
    NativePlotFile.cpp
    HDF5PlotFile.cpp
)

target_include_directories(io PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(io PUBLIC utils)

if(HDF5_IS_PARALLEL)
    target_compile_definitions(io PRIVATE H5_HAVE_PARALLEL)
endif()

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    set_source_files_properties(IOHandler.cpp Checkpointer.cpp NativePlotFile.cpp HDF5PlotFile.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(io PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    set_source_files_properties(IOHandler.cpp Checkpointer.cpp NativePlotFile.cpp HDF5PlotFile.cpp PROPERTIES LANGUAGE HIP)
endif()
