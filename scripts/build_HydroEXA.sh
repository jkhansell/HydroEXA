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

if [ $# -ne 1 ]; then
    echo "Usage: $0 [frontier.gpu|frontier.cpu|juwelsbooster.gpu|juwelsbooster.cpu|local.gpu|local.cpu]"
    exit 1
fi

TARGET=$1

# ============================================================
# Directories
# ============================================================

export HYDROEXA_DIR=$(git rev-parse --show-toplevel)
export AMREX_DIR="${HYDROEXA_DIR}/tpl/amrex"

build_dir="${HYDROEXA_DIR}/build"
install_dir="${HYDROEXA_DIR}/install"

# ============================================================
# Machine-specific configuration
# ============================================================

source ${HYDROEXA_DIR}/machines/${TARGET}

bash ${HYDROEXA_DIR}/scripts/build_AMReX.sh ${TARGET}

if [ -f "${install_dir}" ]; then
    print_banner "${GREEN}" "HydroEXA already installed for ${TARGET}"
    exit 0
fi

print_banner "${BLUE}" "Building HydroEXA for ${TARGET}"

# ============================================================
# Configure
# ============================================================

print_banner "${BLUE}" "Configuring"

cmake -S ${HYDROEXA_DIR} -B ${build_dir}                                \
    -DAMReX_ROOT=${AMREX_DIR}/install/lib/cmake/AMReX/AMReXConfig.cmake \
    -DCMAKE_BUILD_TYPE=Debug                                            \
    -DCMAKE_INSTALL_PREFIX=${install_dir}                               \
    -DCMAKE_CXX_FLAGS="-g -O0 -fno-omit-frame-pointer" \
    -DCMAKE_CXX_COMPILER=${CXX}                

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

print_banner "${GREEN}" "HydroEXA build completed"