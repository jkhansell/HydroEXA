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
# Accept absolute, repo-relative (machines/...) or bare profile name, and
# resolve BEFORE use: we already cd-ed into tests/dambreak, so the relative
# test below silently failed and the run died with "mpirun: command not found".
MACHINE=${MACHINE:-machines/kabreV100.gpu}
case "${MACHINE}" in
    /*)         ;;
    machines/*) MACHINE="${REPO_ROOT}/${MACHINE}" ;;
    *)          MACHINE="${REPO_ROOT}/machines/${MACHINE}" ;;
esac
if [ ! -f "${MACHINE}" ]; then
    echo "run.sh: machine profile not found: ${MACHINE}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${MACHINE}"
command -v mpirun >/dev/null 2>&1 || {
    echo "run.sh: mpirun missing after sourcing ${MACHINE}" >&2
    exit 1
}
[ -x "${EXE}" ] || { echo "run.sh: executable not found: ${EXE}" >&2; exit 1; }

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

# Bisect mode: solve only. metrics_table.py needs the plotfiles, not the
# figures, so a sweep of variants can be scored afterwards in one pass.
if [ "${HYDROEXA_SKIP_PLOTS:-0}" = "1" ]; then
    echo "${RUNDIR}" > /tmp/dambreak_last_rundir
    exit "${STATUS}"
fi

# Render analysis figures for ALL plotfiles (grep -v old filters the plt####.old.NNN
# rotation backups; plotfile numbers do NOT map linearly to time, and analysis.py
# names outputs by the ACTUAL simulation time read from each plotfile).
#
# The battery's metric is accuracy against BOTH the analytical solution and the
# two single-level solutions, so `analysis.py` auto-resolves the sibling
#   single_level_fine__<tag>   uniform dx0/2  -> stable yardstick
#   single_level__<tag>        uniform dx0    -> unrefined baseline
# runs of the same tag. Run those two first; if one is missing the figures
# degrade gracefully to whatever references exist.
if [ "${VARIANT}" != "single_level" ] && [ "${VARIANT}" != "single_level_fine" ]; then
    for REF in "single_level_fine__${TAG}" "single_level__${TAG}"; do
        [ -d "outputs/${SOLVER}/${REF}" ] && echo "=== reference: outputs/${SOLVER}/${REF}"
    done
fi

for PLT in $(ls -d "${RUNDIR}"/plt* 2>/dev/null | grep -v old | sort); do
    echo "--- analysis: ${PLT}"
    uv run python analysis.py "${PLT}" 2>&1 \
        | grep -E '^reference|^time|shock position|error vs exact|plateau band|left reservoir|^vs single|accuracy recovered|accuracy gain|wrote' || true
done

echo "${RUNDIR}" > /tmp/dambreak_last_rundir
exit "${STATUS}"
