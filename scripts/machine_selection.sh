#!/bin/bash

set_machine_env() {
    local TARGET=$1
    
    # Reset GPU_FLAGS so successive runs in the same shell don't mix flags
    GPU_FLAGS=""

    case ${TARGET} in
        frontier.gpu)
            export CC=cc
            export CXX=CC
            export FC=ftn

            GPU_FLAGS="
                -DAMReX_GPU_BACKEND=HIP
                -DAMReX_AMD_ARCH=gfx90a
                -DAMReX_GPU_RDC=ON
            "
            ;;

        frontier.cpu)
            export CC=cc
            export CXX=CC
            export FC=ftn

            GPU_FLAGS="
                -DAMReX_GPU_BACKEND=NONE
            "
            ;;

        juwelsbooster.gpu)
            export CC=mpicc
            export CXX=mpicxx
            export FC=mpif90

            GPU_FLAGS="
                -DCMAKE_CUDA_ARCHITECTURES=80
                -DAMReX_GPU_BACKEND=CUDA
                -DAMReX_CUDA_ARCH=8.0
                -DAMReX_GPU_RDC=ON
            "
            ;;

        kabreV100.gpu)
            export CC=mpicc
            export CXX=mpicxx
            export FC=mpif90

            GPU_FLAGS="
                -DCMAKE_CUDA_ARCHITECTURES=70
                -DAMReX_GPU_BACKEND=CUDA
                -DAMReX_CUDA_ARCH=7.0
                -DAMReX_GPU_RDC=ON
            "
            ;;

        kabreL40S.gpu)
            export CC=mpicc
            export CXX=mpicxx
            export FC=mpif90

            GPU_FLAGS="
                -DCMAKE_CUDA_ARCHITECTURES=89
                -DAMReX_GPU_BACKEND=CUDA
                -DAMReX_CUDA_ARCH=8.9
                -DAMReX_GPU_RDC=ON
            "
            ;;


        juwelsbooster.cpu)
            export CC=mpicc
            export CXX=mpicxx
            export FC=mpif90

            GPU_FLAGS="
                -DAMReX_GPU_BACKEND=NONE
            "
            ;;

        local.gpu)
            export CC=mpicc
            export CXX=mpicxx
            export FC=mpif90

            GPU_FLAGS="
                -DCMAKE_CUDA_ARCHITECTURES=native
                -DAMReX_GPU_BACKEND=CUDA
            "
            ;;

        local.cpu)
            export CC=mpicc
            export CXX=mpicxx
            export FC=mpif90

            GPU_FLAGS="
                -DAMReX_GPU_BACKEND=NONE
            "
            ;;

        *)
            # Return 1 if the target isn't recognized
            return 1
            ;;
    esac

    export GPU_FLAGS
    return 0
}