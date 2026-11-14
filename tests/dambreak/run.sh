#!/bin/bash
# ============================================================
# run.sh — generic E2E runner for the dambreak battery
# ============================================================
# Usage:
#   bash run.sh <solver> <variant> <exe> [np] [tag] [machine]
#
#   solver   roe | hydrostatic_roe          (folder under solvers/)
#   variant  amr | amr_noreflux | ...       (inputs_<variant>)
#   exe      path to the HydroEXA binary
#   np       MPI ranks (default 1)
#   tag      optional run label; default = timestamp
#   machine  machine profile to source (default: machines/kabreV100.gpu
#            resolved relative to the repo root). Sources module env
#            (MPI, CUDA, HDF5) so mpirun is available in this shell.
#
# Every run gets its OWN folder under outputs/<solver>/ :
#   outputs/<solver>/<variant>__<tag>/
#     plt####/        plotfiles
#     log.txt         full stdout
#     *_t*.png        analysis plots (generated post-run)
# ============================================================
set -u

SOLVER=${1:?usage: run.sh <solver> <variant> <exe> [np] [tag] [machine]}
VARIANT=${2:?usage: run.sh <solver> <variant> <exe> [np] [tag] [machine]}
EXE=${3:?usage: run.sh <solver> <variant> <exe> [np] [tag] [machine]}
NP=${4:-1}
TAG=${5:-$(date +%Y%m%d_%H%M%S)}
MACHINE=${6:-}

cd "$(dirname "$0")" || exit 1

# Source the machine profile (module loads: MPI/compilers/CUDA/HDF5) so that
# mpirun and friends exist in this shell regardless of how run.sh was invoked.
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MACHINE=${MACHINE:-${REPO_ROOT}/machines/kabreV100.gpu}
if [ -n "${MACHINE}" ] && [ -f "${MACHINE}" ]; then
    # shellcheck disable=SC1090
    source "${MACHINE}"
fi

CFG="solvers/${SOLVER}/inputs_${VARIANT}"
RUNDIR="outputs/${SOLVER}/${VARIANT}__${TAG}"
mkdir -p "${RUNDIR}"

# Per-run plotfile prefix
RUNCFG=$(mktemp)
sed -e "s#^io.plot_file.*#io.plot_file = \"${RUNDIR}/plt\"#" \
    "${CFG}" > "${RUNCFG}"

echo "=== dambreak/${SOLVER}/${VARIANT} tag=${TAG} np=${NP} exe=${EXE}"
echo "=== artifacts: ${RUNDIR}/"
mpirun -np "${NP}" "${EXE}" "${RUNCFG}" 2>&1 | tee "${RUNDIR}/log.txt"
STATUS=${PIPESTATUS[0]}
rm -f "${RUNCFG}"
echo "${SOLVER}/${VARIANT} exit status: ${STATUS}"

# Analyze the LAST plotfile if any were written.
# NOTE on plotfile selection: io writes rotation backups as plt####.old.NNN
# and plotfile numbers do NOT map linearly to time (io.plot_freq is a TIME
# interval). Filter the .old files with grep -v 'old', and let analysis.py
# name outputs by the ACTUAL simulation time read from the plotfile.
LAST=$(ls -d "${RUNDIR}"/plt* 2>/dev/null | grep -v 'old' | sort | tail -1)
if [ -n "${LAST}" ]; then
    uv run python analysis.py "${LAST}" || true
fi

echo "${RUNDIR}" > /tmp/dambreak_last_rundir
exit "${STATUS}"
