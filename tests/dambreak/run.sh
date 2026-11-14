#!/bin/bash
# ============================================================
# run.sh — generic E2E runner for the dambreak battery
# ============================================================
# Usage:
#   bash run.sh <solver> <variant> <exe> [np]
#
#   solver   roe | hydrostatic_roe          (folder under solvers/)
#   variant  amr | amr_noreflux | ...       (inputs_<variant>)
#   exe      path to the HydroEXA binary
#   np       MPI ranks (default 1)
#
# All artifacts go to outputs/<solver>/ :
#   plt_<variant>####/   plotfiles
#   log_<variant>.txt    full stdout
#   <variant>_t*.png     profile vs analytical overlay (last plotfile)
#   <variant>_probe_t*.png  centerline probe
# ============================================================
set -u

SOLVER=${1:?usage: run.sh <solver> <variant> <exe> [np]}
VARIANT=${2:?usage: run.sh <solver> <variant> <exe> [np]}
EXE=${3:?usage: run.sh <solver> <variant> <exe> [np]}
NP=${4:-1}

cd "$(dirname "$0")" || exit 1
CFG="solvers/${SOLVER}/inputs_${VARIANT}"
OUT="outputs/${SOLVER}"
mkdir -p "${OUT}"

# Per-run plotfile prefix and plot cadence
RUNCFG=$(mktemp)
sed -e "s#^io.plot_file.*#io.plot_file = \"${OUT}/plt_${VARIANT}\"#" \
    "${CFG}" > "${RUNCFG}"

echo "=== dambreak/${SOLVER}/${VARIANT}: np=${NP} exe=${EXE}"
mpirun -np "${NP}" "${EXE}" "${RUNCFG}" 2>&1 | tee "${OUT}/log_${VARIANT}.txt"
STATUS=${PIPESTATUS[0]}
rm -f "${RUNCFG}"
echo "${SOLVER}/${VARIANT} exit status: ${STATUS}"

# Analyze the LAST plotfile if any were written
LAST=$(ls -d "${OUT}/plt_${VARIANT}"* 2>/dev/null | grep -v '.old' | sort | tail -1)
if [ -n "${LAST}" ]; then
    BASE=$(basename "${LAST}")
    T=$(python3 -c "print(int('${BASE}'.replace('plt_${VARIANT}','')))")
    uv run python analyze.py "${LAST}" --output "${OUT}/${VARIANT}_t${T}.png" || true
    uv run python profile_probe.py "${LAST}" "${OUT}/${VARIANT}_probe_t${T}.png" || true
fi

exit "${STATUS}"
