#!/bin/bash
# ============================================================
# build_AMReX.sh — Build AMReX from source
# ============================================================
# Sourced by build_HydroEXA.sh. Requires lib.sh to be loaded first.
# ============================================================

# --- Guard: must be sourced, not executed ---
if [[ "${LIB_LOADED:-}" != "1" ]]; then
    echo "Error: build_AMReX.sh must be sourced via build_HydroEXA.sh" >&2
    exit 1
fi

build_dir="${AMREX_DIR}/build/"
install_dir="${AMREX_DIR}/install/"
amrex_config="${install_dir}/lib/cmake/AMReX/AMReXConfig.cmake"

# ---------------------------------------------------------------------------
# Skip if already installed
# ---------------------------------------------------------------------------
if [ -f "${amrex_config}" ]; then
    print_banner "${GREEN}" "AMReX already installed for ${TARGET}"
    export AMREX_ROOT="${install_dir}/lib/cmake/AMReX"
    return 0 2>/dev/null
fi

print_banner "${BLUE}" "Building AMReX for ${TARGET}"

# --- Configure ---
print_banner "${BLUE}" "Configuring"

# Only enable MPI if an MPI wrapper (mpicc) is available on PATH.
if command -v mpicc &>/dev/null; then
    print_banner "${BLUE}" "MPI wrapper found — enabling AMReX_MPI"
else
    print_banner "${RED}" "No MPI wrapper — building AMReX without MPI (local/dev mode)"
fi

cmake -S ${AMREX_DIR} -B ${build_dir}          \
    -DAMReX_MPI=ON                             \
    -DAMReX_OMP=ON                             \
    -DAMReX_PARTICLES=ON                       \
    -DAMReX_SPACEDIM=2                         \
    -DAMReX_LINEAR_SOLVERS=ON                  \
    -DAMReX_PRECISION=${PRECISION}             \
    -DAMReX_HDF5=ON                            \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"         \
    -DCMAKE_INSTALL_PREFIX=${install_dir}      \
    -DCMAKE_CXX_COMPILER=${CXX}                \
    -DHDF5_ROOT=${HDF5_ROOT}                   \
    ${GPU_FLAGS}

# --- Build ---
print_banner "${BLUE}" "Building"
cmake --build ${build_dir} -j 16

# --- Install ---
print_banner "${BLUE}" "Installing"
cmake --install ${build_dir}

export AMREX_ROOT="${install_dir}/lib/cmake/AMReX"

print_banner "${GREEN}" "AMReX build completed"
