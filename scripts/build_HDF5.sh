#!/bin/bash
# ============================================================
# build_HDF5.sh — Build HDF5 from source
# ============================================================
# Sourced by build_HydroEXA.sh. Requires lib.sh to be loaded first.
# ============================================================

# --- Guard: must be sourced, not executed ---
if [[ "${LIB_LOADED:-}" != "1" ]]; then
    echo "Error: build_HDF5.sh must be sourced via build_HydroEXA.sh" >&2
    exit 1
fi

HDF5_SRC="${HYDROEXA_DIR}/tpl/hdf5"

# ---------------------------------------------------------------------------
# Skip if already installed
# ---------------------------------------------------------------------------
if [ -f "${HDF5_INSTALL:-${HYDROEXA_DIR}/tpl/hdf5/install}/bin/h5pcc" ]; then
    print_banner "${GREEN}" "HDF5 already installed at ${HDF5_INSTALL:-${HYDROEXA_DIR}/tpl/hdf5/install}"
    export HDF5_ROOT="${HDF5_INSTALL:-${HYDROEXA_DIR}/tpl/hdf5/install}"
    export HDF5_HOME="${HDF5_ROOT}"
    return 0 2>/dev/null
fi

print_banner "${BLUE}" "Building Parallel HDF5 for ${TARGET}"

# --- Configure ---
print_banner "${BLUE}" "Configuring HDF5"

HDF5_BUILD="${HYDROEXA_DIR}/tpl/hdf5/build"
HDF5_INSTALL="${HYDROEXA_DIR}/tpl/hdf5/install"

rm -rf "${HDF5_BUILD}"
mkdir -p "${HDF5_BUILD}"
cd "${HDF5_BUILD}"

CC="${CC}" "${HDF5_SRC}/configure" \
    --prefix="${HDF5_INSTALL}" \
    --enable-parallel \
    --disable-cxx \
    --disable-fortran \
    --disable-shared \
    --enable-static \
    --enable-hl

# --- Build ---
print_banner "${BLUE}" "Building HDF5"
make -j10

# --- Install ---
print_banner "${BLUE}" "Installing HDF5 to ${HDF5_INSTALL}"
make install

ln -sf "${HDF5_INSTALL}/bin/h5pcc" "${HDF5_INSTALL}/bin/h5cc"

export HDF5_ROOT="${HDF5_INSTALL}"
export HDF5_HOME="${HDF5_ROOT}"

print_banner "${GREEN}" "HDF5 build completed"
