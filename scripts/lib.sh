#!/bin/bash
# ============================================================
# lib.sh — Shared library for HydroEXA build scripts
# ============================================================
# This file must be sourced by build scripts, never executed directly.
# It provides: banner printing, machine environment setup, path
# resolution, and execution guards.
# ============================================================

# --- Execution guard ---
# BASH_SOURCE[0] is the current file; $0 is the calling shell.
# When sourced: BASH_SOURCE[0] != $0  (safe)
# When executed: BASH_SOURCE[0] == $0  (abort)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: lib.sh must be sourced, not executed directly." >&2
    exit 1
fi
LIB_LOADED=1

# --- Colors ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# --- Banner ---
_WIDTH=60
_BORDER=$(printf '#%.0s' $(seq 1 $_WIDTH))

print_banner() {
    local color=$1
    local msg="$2"

    local padding=$(( (_WIDTH - 2 - ${#msg}) / 2 ))
    local extra=$(( (_WIDTH - 2 - ${#msg}) % 2 ))

    printf "\n${color}${BOLD}%s\n" "$_BORDER"
    printf "#%*s%s%*s#\n" \
        $padding "" \
        "$msg" \
        $((padding + extra)) ""
    printf "%s${NC}\n\n" "$_BORDER"
}

# --- Error handler ---
cleanup_on_fail() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo -e "\n\033[0;31m\033[1m########################################"
        echo "   CRITICAL ERROR: Build Failed! (Exit: $exit_code)"
        echo -e "########################################\033[0m\n"
    fi
}
trap cleanup_on_fail EXIT


# --- Machine environment ---
# ---------------------------------------------------------------------------
# _fallback_compiler — use system compilers if the requested MPI wrapper is
#                      not on PATH.  Prints the compiler path; caller assigns
#                      it to CC / CXX / FC.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# hydroexa_derive_gpu_flags — turn the card description exported by a machine
# profile into the -D argument string the legacy build scripts pass to cmake.
#
# Profiles declare *what the card is* (HYDROEXA_GPU_BACKEND / HYDROEXA_GPU_ARCH);
# this is the single place that turns that into CMake arguments, so profiles
# never hand-write cmake flags. Removed together with the legacy scripts.
# ---------------------------------------------------------------------------
hydroexa_derive_gpu_flags() {
    local backend="${HYDROEXA_GPU_BACKEND:-NONE}"
    local arch="${HYDROEXA_GPU_ARCH:-}"
    local flags=""

    case "${backend}" in
        CUDA)
            flags="-DAMReX_GPU_BACKEND=CUDA"
            [ -n "${arch}" ] && flags="${flags} -DCMAKE_CUDA_ARCHITECTURES=${arch}"
            ;;
        HIP)
            # AMReX requires AMReX_AMD_ARCH when the HIP backend is selected.
            flags="-DAMReX_GPU_BACKEND=HIP"
            [ -n "${arch}" ] && flags="${flags} -DAMReX_AMD_ARCH=${arch}"
            ;;
        NONE|"")
            flags="-DAMReX_GPU_BACKEND=NONE"
            ;;
        *)
            echo "error: unknown HYDROEXA_GPU_BACKEND '${backend}'" >&2
            return 1
            ;;
    esac

    export GPU_FLAGS="${flags}"
}

# --- Path resolution ---
# Call resolve_paths <machine> once at the top of any build script.
# Sets: HYDROEXA_DIR, AMREX_DIR, BUILD_ROOT, INSTALL_ROOT
# Output directories are namespaced by the machine target because HDF5,
# AMReX and HydroEXA binaries are toolchain- and architecture-specific
# (CUDA arch, MPI flavour, compiler version).  Sharing them across
# machines on a common tree silently links the wrong libraries.
resolve_paths() {
    local machine="${1:-}"
    export HYDROEXA_DIR="$(git rev-parse --show-toplevel)"
    export AMREX_DIR="${HYDROEXA_DIR}/tpl/amrex"

    if [ -n "${machine}" ]; then
        export BUILD_ROOT="${HYDROEXA_DIR}/build_${machine}"
        export INSTALL_ROOT="${HYDROEXA_DIR}/install_${machine}"
    else
        export BUILD_ROOT="${HYDROEXA_DIR}/build"
        export INSTALL_ROOT="${HYDROEXA_DIR}/install"
    fi

    # Legacy aliases kept for child scripts that still read them.
    build_dir="${BUILD_ROOT}"
    install_dir="${INSTALL_ROOT}"
}


