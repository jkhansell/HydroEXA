#!/bin/bash

# ============================================================
# Colors
# ============================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

WIDTH=60
BORDER=$(printf '#%.0s' $(seq 1 $WIDTH))

print_banner() {
    local color=$1
    local msg="$2"

    local padding=$(( (WIDTH - 2 - ${#msg}) / 2 ))
    local extra=$(( (WIDTH - 2 - ${#msg}) % 2 ))

    printf "\n${color}${BOLD}%s\n" "$BORDER"
    printf "#%*s%s%*s#\n" \
        $padding "" \
        "$msg" \
        $((padding + extra)) ""
    printf "%s${NC}\n\n" "$BORDER"
}

# ============================================================
# Input
# ============================================================

if [ $# -ne 1 ]; then
    print_banner "${RED}" "Invalid target"
    echo "Usage:"
    echo "  $0 [frontier.gpu|frontier.cpu|juwelsbooster.gpu|juwelsbooster.cpu|local.gpu|local.cpu]"
    exit 1
fi

TARGET=$1

# ============================================================
# Directories
# ============================================================

export HYDROEXA_DIR=$(git rev-parse --show-toplevel)
export AMREX_DIR="${HYDROEXA_DIR}/tpl/amrex"

build_dir="${AMREX_DIR}/build/"
install_dir="${AMREX_DIR}/install/"

amrex_config="${install_dir}/lib/cmake/AMReX/AMReXConfig.cmake"

# ============================================================
# Machine-specific configuration
# ============================================================


case ${TARGET} in

    frontier.gpu)

        export CC=cc
        export CXX=CC
        export FC=ftn

        GPU_FLAGS="
            -DCMAKE_HIP_COMPILER=hipcc
            -DCMAKE_HIP_ARCHITECTURES=gfx90a
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

    juwelsbooster.cpu)

        export CC=mpicc
        export CXX=mpicxx
        export FC=mpif90

        GPU_FLAGS="
            -DAMReX_GPU_BACKEND=NONE
        "
        ;;

    local.gpu)

        export CC=gcc
        export CXX=g++
        export FC=gfortran

        GPU_FLAGS="
            -DCMAKE_CUDA_ARCHITECTURES=native
            -DAMReX_GPU_BACKEND=CUDA
        "
        ;;

    local.cpu)



        GPU_FLAGS="
            -DAMReX_GPU_BACKEND=NONE
        "
        ;;

    *)

        print_banner "${RED}" "Unknown target: ${TARGET}"
        exit 1
        ;;

esac

# ============================================================
# Skip rebuild if already installed
# ============================================================

if [ -f "${amrex_config}" ]; then
    print_banner "${GREEN}" "AMReX already installed for ${TARGET}"
    exit 0
fi

print_banner "${BLUE}" "Building AMReX for ${TARGET}"

# ============================================================
# Configure
# ============================================================

print_banner "${BLUE}" "Configuring"

cmake -S ${AMREX_DIR} -B ${build_dir}          \
    -DAMReX_MPI=ON                             \
    -DAMReX_OMP=OFF                            \
    -DAMReX_PARTICLES=ON                       \
    -DAMReX_SPACEDIM=2                         \
    -DAMReX_LINEAR_SOLVERS=ON                  \
    -DAMReX_PRECISION=DOUBLE                   \
    -DCMAKE_BUILD_TYPE=Debug                 \
    -DCMAKE_INSTALL_PREFIX=${install_dir}      \
    -DCMAKE_CXX_COMPILER=${CXX}                \
    ${GPU_FLAGS}

# ============================================================
# Build
# ============================================================

print_banner "${BLUE}" "Building"

cmake --build ${build_dir} -j 16

# ============================================================
# Install
# ============================================================

print_banner "${BLUE}" "Installing"

cmake --install ${build_dir}

print_banner "${GREEN}" "AMReX build completed"