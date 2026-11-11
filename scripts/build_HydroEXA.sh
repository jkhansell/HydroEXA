#!/bin/bash
# ============================================================
# build_HydroEXA.sh — Main build orchestrator
# ============================================================
# Usage: bash scripts/build_HydroEXA.sh machine.{cpu,gpu} [Release|Debug] [DOUBLE|SINGLE]
#
# Pipeline: HDF5 → AMReX → HydroEXA
# ============================================================

set -e
set -o pipefail

# --- Load shared library ---
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "${SCRIPT_DIR}/lib.sh"

# --- Parse arguments ---
if [ $# -ne 3 ]; then
    echo "Usage: $0 machine.{cpu,gpu} [Release|Debug|...] [DOUBLE|SINGLE]}"
    exit 1
fi

TARGET=$1
BUILD_TYPE=$2
PRECISION=$3

case "${PRECISION}" in
    SINGLE|DOUBLE) ;;
    *)
        echo "Error: precision must be SINGLE or DOUBLE (got '${PRECISION}')"
        exit 1
        ;;
esac
export PRECISION

# --- Resolve paths (namespaced by machine target) ---
resolve_paths "${TARGET}"

# --- Source machine-specific module loads ---
MACHINE_FILE="${HYDROEXA_DIR}/machines/${TARGET}"
if [ -f "${MACHINE_FILE}" ]; then
    print_banner "${BLUE}" "Sourcing machine config: ${MACHINE_FILE}"
    source "${MACHINE_FILE}"
else
    print_banner "${RED}" "No machine config found at ${MACHINE_FILE}"
fi

# --- Export install paths (inherited by all child scripts) ---
# Everything for one machine lives under install_${TARGET}/:
#   install_${TARGET}/hdf5                 — parallel HDF5 (precision-agnostic)
#   install_${TARGET}/amrex_${PRECISION}   — AMReX (amrex::Real is build-specific)
#   install_${TARGET}/hydroexa_${PRECISION} — HydroEXA binary, lib, headers
# Build trees mirror this under build_${TARGET}/.
#
# HDF5 has no single/double build variant, so it is shared per machine.
# AMReX and HydroEXA encode amrex::Real at compile time, so they must be
# keyed by precision to allow SINGLE and DOUBLE builds to coexist.
export HDF5_INSTALL="${INSTALL_ROOT}/hdf5"
export HDF5_BUILD="${BUILD_ROOT}/hdf5"
export HDF5_ROOT="${HDF5_INSTALL}"
export HDF5_HOME="${HDF5_ROOT}"

export AMREX_INSTALL="${INSTALL_ROOT}/amrex_${PRECISION}"
export AMREX_BUILD="${BUILD_ROOT}/amrex_${PRECISION}"
export AMREX_ROOT="${AMREX_INSTALL}/lib/cmake/AMReX"

export HYDROEXA_SRC=${HYDROEXA_DIR}
export HYDROEXA_BUILD=${BUILD_ROOT}/hydroexa_${PRECISION}
export HYDROEXA_INSTALL=${INSTALL_ROOT}/hydroexa_${PRECISION}


# --- Step 1: HDF5 ---
print_banner "${BLUE}" "Step 1/3: Building HDF5"
source "${HYDROEXA_DIR}/scripts/build_HDF5.sh"

# --- Step 2: AMReX ---
print_banner "${BLUE}" "Step 2/3: Building AMReX"
source "${HYDROEXA_DIR}/scripts/build_AMReX.sh"

# --- Step 3: HydroEXA ---
if [ -d "${HYDROEXA_INSTALL}" ] && [ -f "${HYDROEXA_INSTALL}/bin/HydroEXA" ]; then
    print_banner "${GREEN}" "HydroEXA already installed for ${TARGET}"
    exit 0
fi

print_banner "${BLUE}" "Step 3/3: Building HydroEXA for ${TARGET}"

# --- Configure ---
print_banner "${BLUE}" "Configuring"

rm -f "${HYDROEXA_BUILD}/CMakeCache.txt"

cmake -S "${HYDROEXA_SRC}" -B "${HYDROEXA_BUILD}"                         \
    -DHYDROEXA_GPU_BACKEND="${TARGET}"                                    \
    -DHDF5_ROOT="${HDF5_ROOT}"                                            \
    -DAMReX_ROOT="${AMREX_ROOT}"                                          \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"                                    \
    -DCMAKE_INSTALL_PREFIX="${HYDROEXA_INSTALL}"                          \
    -DCMAKE_CXX_COMPILER="${CXX}"                                         \
    ${GPU_FLAGS}

# --- Build & Install ---
print_banner "${BLUE}" "Building"
cmake --build "${HYDROEXA_BUILD}" -j 16

print_banner "${BLUE}" "Installing"
cmake --install "${HYDROEXA_BUILD}"

print_banner "${GREEN}" "HydroEXA build completed"
