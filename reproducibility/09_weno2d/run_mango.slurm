#!/usr/bin/env bash
#SBATCH --job-name=btdma-weno2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --comment=etc
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch run_mango.slurm <cfl1p0|cfl1p5|cfl2p0>" >&2
    exit 2
fi

CASE_ID="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CASE_FILE="${SCRIPT_DIR}/cases.tsv"
EXECUTABLE="${SCRIPT_DIR}/bin/weno2d_gpu"
MANGO_SRUN_MPI="${MANGO_SRUN_MPI:-pmix}"
export OMP_NUM_THREADS=1

case_line="$(awk -F '\t' -v case_id="${CASE_ID}" \
    'NR > 1 && $1 == case_id {print; found=1} END {if (!found) exit 1}' \
    "${CASE_FILE}")" || {
    echo "Unknown case '${CASE_ID}'. Use cfl1p0, cfl1p5, or cfl2p0." >&2
    exit 2
}

IFS=$'\t' read -r _ CFL_NOMINAL NSTEPS NX NY NPX NPY GPUS <<< "${case_line}"
if [[ "${NX} ${NY} ${NPX} ${NPY} ${GPUS}" != "516 516 2 2 4" ]]; then
    echo "The selected row in cases.tsv is not the fixed manuscript case." >&2
    exit 2
fi

source "${ROOT_DIR}/setup_mango.sh"

if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Missing ${EXECUTABLE}; run ./build_mango.sh first." >&2
    exit 1
fi

JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_DIR="${SCRIPT_DIR}/output/${JOB_TAG}/${CASE_ID}"
mkdir -p "${OUT_DIR}"

{
    printf 'experiment=09_weno2d\n'
    printf 'case_id=%s\n' "${CASE_ID}"
    printf 'cfl_nominal=%s\n' "${CFL_NOMINAL}"
    printf 'nsteps=%s\n' "${NSTEPS}"
    printf 'grid=%sx%s\n' "${NX}" "${NY}"
    printf 'decomposition=%sx%s\n' "${NPX}" "${NPY}"
    printf 'gpus=%s\n' "${GPUS}"
    date --iso-8601=seconds
    hostname
    module list 2>&1
    "${MPIFC}" --version
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
    git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true
    sha256sum \
        "${SCRIPT_DIR}/weno2d_gpu.f90" \
        "${SCRIPT_DIR}/cases.tsv" \
        "${SCRIPT_DIR}/build_mango.sh" \
        "${SCRIPT_DIR}/run_mango.slurm" \
        "${SCRIPT_DIR}/postprocess.py" \
        "${ROOT_DIR}"/lib_btdma/*.f90
} > "${OUT_DIR}/manifest.txt"

echo "[run] ${CASE_ID}: CFL=${CFL_NOMINAL}, steps=${NSTEPS}, grid=516x516, NP=2x2, GPUs=4"
srun --mpi="${MANGO_SRUN_MPI}" --nodes=1 --ntasks=4 \
    --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=closest --exact \
    "${EXECUTABLE}" "${CASE_ID}" "${OUT_DIR}" \
    | tee "${OUT_DIR}/run.log"

python3 "${SCRIPT_DIR}/postprocess.py" \
    "${OUT_DIR}/metrics_${CASE_ID}.csv" \
    --summary "${OUT_DIR}/summary.csv" \
    --contour-dir "${OUT_DIR}/contours"

echo "Results: ${OUT_DIR}"

