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

# --- Machine environment ---
# ---------------------------------------------------------------------------
# _fallback_compiler — use system compilers if the requested MPI wrapper is
#                      not on PATH.  Prints the compiler path; caller assigns
#                      it to CC / CXX / FC.
# ---------------------------------------------------------------------------
_fallback_compiler() {
    local mpi_wrapper=$1
    if command -v "${mpi_wrapper}" &>/dev/null; then
        echo "${mpi_wrapper}"
    else
        # Map MPI wrappers to system equivalents
        case "${mpi_wrapper}" in
            mpicc)   echo "gcc" ;;
            mpicxx)  echo "g++" ;;
            mpif90)  echo "gfortran" ;;
            *)       echo "gcc" ;;
        esac
    fi
}

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
            export CC=$(_fallback_compiler mpicc)
            export CXX=$(_fallback_compiler mpicxx)
            export FC=$(_fallback_compiler mpif90)
            GPU_FLAGS="
                -DCMAKE_CUDA_ARCHITECTURES=80
                -DAMReX_GPU_BACKEND=CUDA
                -DAMReX_CUDA_ARCH=8.0
            "
            ;;

        kabreV100.gpu)
            export CC=$(_fallback_compiler mpicc)
            export CXX=$(_fallback_compiler mpicxx)
            export FC=$(_fallback_compiler mpif90)
            GPU_FLAGS="
                -DCMAKE_CUDA_ARCHITECTURES=70
                -DAMReX_GPU_BACKEND=CUDA
            "
            ;;

        kabreL40S.gpu)
            export CC=$(_fallback_compiler mpicc)
            export CXX=$(_fallback_compiler mpicxx)
            export FC=$(_fallback_compiler mpif90)
            GPU_FLAGS="
                -DCMAKE_CUDA_ARCHITECTURES=89
                -DAMReX_GPU_BACKEND=CUDA
            "
            ;;

        juwelsbooster.cpu)
            export CC=$(_fallback_compiler mpicc)
            export CXX=$(_fallback_compiler mpicxx)
            export FC=$(_fallback_compiler mpif90)
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
            export CC=gcc
            export CXX=g++
            export FC=gfortran
            GPU_FLAGS="
                -DAMReX_GPU_BACKEND=NONE
            "
            ;;

        *)
            return 1
            ;;
    esac

    export GPU_FLAGS
    return 0
}

# --- Path resolution ---
# Call resolve_paths() once at the top of any build script.
# Sets: HYDROEXA_DIR, build_dir, install_dir, AMREX_DIR
# All variables are exported so child scripts inherit them.
resolve_paths() {
    export HYDROEXA_DIR="$(git rev-parse --show-toplevel)"
    export AMREX_DIR="${HYDROEXA_DIR}/tpl/amrex"
    build_dir="${HYDROEXA_DIR}/build"
    install_dir="${HYDROEXA_DIR}/install"
}

# --- Preflight check ---
# Call after resolve_paths + set_machine_env.
# Exits with a banner if TARGET is invalid.
# Warns if the target needs MPI but mpicc is not on PATH.
preflight() {
    local TARGET=$1
    if ! set_machine_env "${TARGET}"; then
        print_banner "${RED}" "Unknown target: ${TARGET}"
        exit 1
    fi

    # Warn if MPI is needed but mpicc is not available.
    case ${TARGET} in
        *.gpu|*.cpu)
            if ! command -v mpicc &>/dev/null; then
                print_banner "${RED}" "mpicc not found on PATH."
                print_banner "${RED}" "Load an MPI module first (e.g. module load mpi/openmpi-x86_64) or use a local target."
                exit 1
            fi
            ;;
    esac
}
