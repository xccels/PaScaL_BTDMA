#!/usr/bin/env bash
#SBATCH --job-name=heat3d-scale
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${CASE_DIR}/../.." && pwd)"
CASE_FILE="${CASE_DIR}/cases.tsv"
RUN_ID="${SLURM_JOB_ID:-manual}"
OUT_DIR="${CASE_DIR}/output/${RUN_ID}/scaling"
REPEATS="${REPEATS:-3}"
MANGO_SRUN_MPI="${MANGO_SRUN_MPI:-pmix}"

source "${ROOT_DIR}/setup_mango.sh"
export OMP_NUM_THREADS=1
export UCX_MEMTYPE_CACHE=n

mkdir -p "${OUT_DIR}/logs"

{
    echo "workflow=scaling"
    echo "timestamp=$(date -Iseconds)"
    echo "job_id=${RUN_ID}"
    echo "allocated_nodes=${SLURM_NNODES:-unknown}"
    echo "allocated_tasks=${SLURM_NTASKS:-unknown}"
    echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
    echo "repeats=${REPEATS}"
    echo "global_grid=1024x1024x1024"
    echo "nsteps=100"
    hostname
    module list 2>&1
    "${MPIFC}" --version
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
    git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true
    sha256sum \
        "${CASE_DIR}/heat3d_gpu.f90" \
        "${CASE_DIR}/cases.tsv" \
        "${CASE_DIR}/build_mango.sh" \
        "${CASE_DIR}/run_scaling_mango.slurm" \
        "${CASE_DIR}/postprocess.py" \
        "${ROOT_DIR}"/lib_btdma/*.f90
} > "${OUT_DIR}/manifest.txt"

while IFS=$'\t' read -r case_id workflow n np1 np2 np3 nsteps gpus mango_nodes executable; do
    [[ "${case_id}" == "case_id" ]] && continue
    [[ "${workflow}" != "scaling" ]] && continue

    exe="${CASE_DIR}/bin/${executable}"
    if [[ ! -x "${exe}" ]]; then
        echo "Missing ${exe}; run ./build_mango.sh scaling first." >&2
        exit 1
    fi
    if (( n != 1024 || nsteps != 100 || np1*np2*np3 != gpus || 8*mango_nodes != gpus )); then
        echo "Invalid scaling row: ${case_id}" >&2
        exit 2
    fi

    for ((repeat=1; repeat<=REPEATS; repeat++)); do
        log_file="${OUT_DIR}/logs/${case_id}_r$(printf '%02d' "${repeat}").log"
        echo "[run] ${case_id}, repeat=${repeat}, P=${gpus}, topology=${np1}x${np2}x${np3}"
        srun --mpi="${MANGO_SRUN_MPI}" --nodes="${mango_nodes}" --ntasks="${gpus}" \
            --ntasks-per-node=8 --gpus-per-task=1 --gpu-bind=closest --exact \
            "${exe}" | tee "${log_file}"
    done
done < "${CASE_FILE}"

python3 "${CASE_DIR}/postprocess.py" scaling \
    --log-dir "${OUT_DIR}/logs" \
    --raw "${OUT_DIR}/scaling_raw.csv" \
    --summary "${OUT_DIR}/scaling_summary.csv"

echo "Results: ${OUT_DIR}"
