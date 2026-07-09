#!/bin/bash

# ============================================================
# Error Handling (Crash on Fail)
# ============================================================
set -e          # Exit immediately if any command returns a non-zero status
set -o pipefail # Captures errors hidden inside piped commands

cleanup_on_fail() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo -e "\n\033[0;31m\033[1m########################################"
        echo "   CRITICAL ERROR: Build Failed! (Exit: $exit_code)"
        echo -e "########################################\033[0m\n"
    fi
}
trap cleanup_on_fail EXIT

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
    if type print_banner &>/dev/null; then
        print_banner "${RED}" "Missing machine selection file!"
    else
        echo "Missing machine selection file!"
    fi
    exit 1
fi

if [ $# -ne 2 ]; then
    echo "Usage: $0 [frontier.gpu|frontier.cpu|juwelsbooster.gpu|juwelsbooster.cpu|local.gpu|local.cpu]"
    exit 1
fi

TARGET=$1
BUILD_TYPE=$2 

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
if [ ! -f "${HYDROEXA_DIR}/machines/${TARGET}" ]; then
    print_banner "${RED}" "Machine target profile folder missing: ${TARGET}"
    exit 1
fi

source "${HYDROEXA_DIR}/machines/${TARGET}"

if ! set_machine_env "${TARGET}"; then
    print_banner "${RED}" "Unknown target: ${TARGET}"
    exit 1
fi

bash "${HYDROEXA_DIR}/scripts/build_AMReX.sh" "${TARGET}" "${BUILD_TYPE}"

if [ -d "${install_dir}" ] && [ -f "${install_dir}/bin/HydroEXA" ]; then
    print_banner "${GREEN}" "HydroEXA already installed for ${TARGET}"
    exit 0
fi

print_banner "${BLUE}" "Building HydroEXA for ${TARGET}"

# ============================================================
# Configure
# ============================================================
print_banner "${BLUE}" "Configuring"

rm -f "${build_dir}/CMakeCache.txt"

# Added -DHYDROEXA_GPU_BACKEND passing explicitly to CMake
cmake -S "${HYDROEXA_DIR}" -B "${build_dir}"                              \
    -DHYDROEXA_GPU_BACKEND="${TARGET}"                                    \
    -DAMReX_ROOT="${AMREX_DIR}/install/lib/cmake/AMReX"                   \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"                                    \
    -DCMAKE_INSTALL_PREFIX="${install_dir}"                               \
    -DCMAKE_CXX_COMPILER="${CXX}"                                         \
    ${GPU_FLAGS} 

# ============================================================
# Build & Install
# ============================================================
print_banner "${BLUE}" "Building"
cmake --build "${build_dir}" -j 16

print_banner "${BLUE}" "Installing"
cmake --install "${build_dir}"

print_banner "${GREEN}" "HydroEXA build completed"