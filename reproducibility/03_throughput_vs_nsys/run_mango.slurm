#!/usr/bin/env bash
#SBATCH --job-name=btdma-throughput
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODE="${MODE:-}"
WARMUPS="${WARMUPS:-2}"
REPEATS="${REPEATS:-5}"
MANGO_SRUN_MPI="${MANGO_SRUN_MPI:-pmix}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT_DIR="${SCRIPT_DIR}/output/${JOB_ID}/${MODE:-unset}"
NSYS_ROOTS=(16 32 64 128 256)

if [[ "${MODE}" != "cpu" && "${MODE}" != "gpu" ]]; then
    echo "Set MODE=cpu or MODE=gpu when submitting this script." >&2
    exit 2
fi

source "${ROOT_DIR}/setup_mango.sh"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
mkdir -p "${OUT_DIR}"
cd "${SCRIPT_DIR}"

if [[ ! -x "${SCRIPT_DIR}/bin/throughput_${MODE}" ]]; then
    echo "Missing bin/throughput_${MODE}; run ./build_mango.sh before sbatch." >&2
    exit 1
fi

{
    date --iso-8601=seconds
    hostname
    echo "mode=${MODE}"
    echo "warmups=${WARMUPS}"
    echo "repeats=${REPEATS}"
    echo "slurm_job_id=${JOB_ID}"
    echo "slurm_nodes=${SLURM_JOB_NUM_NODES:-unknown}"
    echo "slurm_tasks=${SLURM_NTASKS:-unknown}"
    module list 2>&1
    "${MPIFC}" --version
    mpirun --version
    git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true
    sha256sum \
        "${ROOT_DIR}/lib_btdma/mpiutil.f90" \
        "${ROOT_DIR}/lib_btdma/mod_btdma_cpu.f90" \
        "${ROOT_DIR}/lib_btdma/mod_cudatools.f90" \
        "${ROOT_DIR}/lib_btdma/mod_btdma_gpu_v2.f90" \
        "${SCRIPT_DIR}/throughput_common.f90" \
        "${SCRIPT_DIR}/throughput_cpu.f90" \
        "${SCRIPT_DIR}/throughput_gpu.f90"
    if [[ "${MODE}" == "cpu" ]]; then
        lscpu
    else
        nvidia-smi -L
        nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
    fi
} > "${OUT_DIR}/environment.txt"

if [[ "${MODE}" == "cpu" ]]; then
    if (( ${SLURM_JOB_NUM_NODES:-0} < 4 )); then
        echo "CPU mode requires an allocation of at least four two-socket nodes." >&2
        exit 2
    fi
    export OMPI_MCA_coll_ucc_enable=0
    for m in 2 5 8; do
        for root in "${NSYS_ROOTS[@]}"; do
            nsys=$((root * root))
            log_file="${OUT_DIR}/cpu_P8_m${m}_nsys${nsys}.log"
            echo "[run] CPU P=8 sockets, ranks=192, N=2048, m=${m}, nsys=${nsys}"
            srun --mpi="${MANGO_SRUN_MPI}" --nodes=4 --ntasks=192 \
                --ntasks-per-node=48 --cpu-bind=cores --exact \
                "${SCRIPT_DIR}/bin/throughput_cpu" \
                8 2048 "${nsys}" "${m}" "${WARMUPS}" "${REPEATS}" \
                | tee "${log_file}"
        done
    done
else
    for p in 2 4 8; do
        if (( p == 8 )); then
            block_sizes=(2 5 8)
        else
            block_sizes=(8)
        fi
        for m in "${block_sizes[@]}"; do
            for root in "${NSYS_ROOTS[@]}"; do
                nsys=$((root * root))
                log_file="${OUT_DIR}/gpu_P${p}_m${m}_nsys${nsys}.log"
                echo "[run] GPU P=${p}, ranks=${p}, N=2048, m=${m}, nsys=${nsys}"
                srun --mpi="${MANGO_SRUN_MPI}" --nodes=1 --ntasks="${p}" \
                    --gpus-per-task=1 --gpu-bind=closest --exact \
                    "${SCRIPT_DIR}/bin/throughput_gpu" \
                    "${p}" 2048 "${nsys}" "${m}" "${WARMUPS}" "${REPEATS}" \
                    | tee "${log_file}"
            done
        done
    done
fi

python3 "${SCRIPT_DIR}/collect_results.py" "${OUT_DIR}"/*.log \
    --output-dir "${OUT_DIR}/csv"
