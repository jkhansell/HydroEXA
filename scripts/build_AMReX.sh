#!/bin/bash

# ============================================================
# Import Machine Environment Function
# ============================================================

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

if [ -f "${SCRIPT_DIR}/print_banner.sh" ]; then
    source "${SCRIPT_DIR}/print_banner.sh"
fi

if [ -f "${SCRIPT_DIR}/machine_selection.sh" ]; then
    source "${SCRIPT_DIR}/machine_selection.sh"
else
    print_banner "${RED}" "Missing machine selection file!"
    echo "Could not find machine_selection.sh next to this script."
    exit 1
fi

# ============================================================
# Input
# ============================================================

if [ $# -ne 2 ]; then
    print_banner "${RED}" "Invalid target"
    echo "Usage:"
    echo "  $0 [frontier.gpu|frontier.cpu|juwelsbooster.gpu|juwelsbooster.cpu|local.gpu|local.cpu]"
    exit 1
fi

TARGET=$1
BUILD_TYPE=$2

# ============================================================
# Directories
# ============================================================

export HYDROEXA_DIR=$(git rev-parse --show-toplevel)
export AMREX_DIR="${HYDROEXA_DIR}/tpl/amrex"

build_dir="${AMREX_DIR}/build/"
install_dir="${AMREX_DIR}/install/"

amrex_config="${install_dir}/lib/cmake/AMReX/AMReXConfig.cmake"

# ============================================================
# Machine-specific configuration (Imported Function)
# ============================================================

if ! set_machine_env "${TARGET}"; then
    print_banner "${RED}" "Unknown target: ${TARGET}"
    exit 1
fi

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
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"         \
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