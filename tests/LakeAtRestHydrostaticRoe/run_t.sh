#!/bin/bash
# ============================================================
# run_t.sh — generic T-battery runner for LakeAtRestHydrostaticRoe
# ============================================================
# Usage:
#   bash run_t.sh <exe> <tag> [np] [reflux] [subcycle] [regrid]
#
#   exe       path to the HydroEXA binary to test
#   tag       test label (names inputs/log files), e.g. T1a
#   np        MPI ranks (default 4)
#   reflux    amr.do_reflux value   (default 0)
#   subcycle  amr.do_subcycle value (default 0)
#   regrid    amr.regrid_int value  (default 0)
#
# Examples:
#   bash run_t.sh ../../install_kabreV100.gpu/hydroexa_DOUBLE/bin/HydroEXA T1a 4 0 0 0
#   bash run_t.sh ../../install_kabreV100.gpu/hydroexa_DOUBLE/bin/HydroEXA T1b 4 1 0 0
#   bash run_t.sh ../../install_kabreV100.gpu/hydroexa_DOUBLE/bin/HydroEXA T2  4 1 1 0
#   bash run_t.sh ../../install_kabreV100.gpu/hydroexa_DOUBLE/bin/HydroEXA T3  4 1 1 10
#
# Acceptance: the [Conservation] lines at the end of battery_<tag>.log
# should show per-level momenta at roundoff for the whole run.
# ============================================================
set -u

EXE=${1:?usage: run_t.sh <exe> <tag> [np] [reflux] [subcycle] [regrid]}
TAG=${2:?usage: run_t.sh <exe> <tag> [np] [reflux] [subcycle] [regrid]}
NP=${3:-4}
REFLUX=${4:-0}
SUBCYCLE=${5:-0}
REGRID=${6:-0}

cd "$(dirname "$0")" || exit 1

CFG=inputs_${TAG}
LOG=battery_${TAG}.log

sed -e "s/^amr.regrid_int.*/amr.regrid_int = ${REGRID}/" \
    -e "s/^amr.do_subcycle.*/amr.do_subcycle = ${SUBCYCLE}/" \
    -e "s/^amr.do_reflux.*/amr.do_reflux = ${REFLUX}/" \
    -e "s/^HydroEXA.max_time.*/HydroEXA.max_time = 30/" \
    -e "s/^HydroEXA.mass_diag_freq.*/HydroEXA.mass_diag_freq = 0.5/" \
    -e "s#^io.plot_file.*#io.plot_file = /tmp/plotfiles/plt_${TAG}#" \
    inputs > "${CFG}"

echo "=== ${TAG}: np=${NP} reflux=${REFLUX} subcycle=${SUBCYCLE} regrid=${REGRID}"
echo "=== exe: ${EXE}"

mpirun -np "${NP}" "${EXE}" "${CFG}" 2>&1 | tee "${LOG}"
STATUS=${PIPESTATUS[0]}

echo "${TAG} exit status: ${STATUS}"
echo "--- conservation diagnostics --------------------------------------"
grep "\[Conservation" "${LOG}" | tail -8
echo "--- final line ----------------------------------------------------"
tail -2 "${LOG}"

exit "${STATUS}"
