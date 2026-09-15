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


