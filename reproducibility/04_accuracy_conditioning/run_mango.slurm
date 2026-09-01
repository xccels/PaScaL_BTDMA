#!/usr/bin/env bash
#SBATCH --job-name=btdma_accuracy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=output/slurm-%j.out
#SBATCH --error=output/slurm-%j.err

set -euo pipefail

EXP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${EXP_DIR}/../.." && pwd)"
OUTPUT_DIR="${EXP_DIR}/output"
CASE="${CASE:-smoke}"
MPI_RANKS="${MPI_RANKS:-8}"
RUN_DIR="${OUTPUT_DIR}/${SLURM_JOB_ID}-${CASE}"
RAW_DIR="${RUN_DIR}/raw"

mkdir -p "${RAW_DIR}"
source "${ROOT_DIR}/setup_mango.sh"
"${EXP_DIR}/build_mango.sh"

case "${CASE}" in
    smoke)
        M_VALUES=(2)
        K_EXPONENTS=(2)
        N_SYSTEMS_PER_FAMILY=8
        CHUNK_SIZE=8
        ;;
    full)
        M_VALUES=(2 5 8)
        K_EXPONENTS=(2 6 10 14)
        N_SYSTEMS_PER_FAMILY=8192
        CHUNK_SIZE="${CHUNK_SIZE:-64}"
        ;;
    singular)
        M_VALUES=()
        K_EXPONENTS=()
        N_SYSTEMS_PER_FAMILY=0
        CHUNK_SIZE=0
        ;;
    *)
        echo "CASE must be smoke, full, or singular" >&2
        exit 2
        ;;
esac

{
    echo "timestamp=$(date -Iseconds)"
    echo "case=${CASE}"
    echo "mpi_ranks=${MPI_RANKS}"
    echo "n_systems_per_family=${N_SYSTEMS_PER_FAMILY}"
    echo "chunk_size=${CHUNK_SIZE}"
    git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true
    "${MPIFC}" --version | head -n 2
    mpirun --version | head -n 2
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    module list 2>&1
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${EXP_DIR}/accuracy_support.f90"
    else
        shasum -a 256 "${EXP_DIR}/accuracy_support.f90"
    fi
} > "${RUN_DIR}/environment.txt"

for m in "${M_VALUES[@]}"; do
    for logk in "${K_EXPONENTS[@]}"; do
        mpirun -n "${MPI_RANKS}" "${EXP_DIR}/build/accuracy_cpu.out" \
            "${m}" "${logk}" "${N_SYSTEMS_PER_FAMILY}" "${CHUNK_SIZE}" \
            "${RAW_DIR}/accuracy_cpu_m${m}_K1e${logk}.csv" \
            | tee "${RUN_DIR}/accuracy_cpu_m${m}_K1e${logk}.log"

        mpirun -n "${MPI_RANKS}" "${EXP_DIR}/build/accuracy_gpu.out" \
            "${m}" "${logk}" "${N_SYSTEMS_PER_FAMILY}" "${CHUNK_SIZE}" \
            "${RAW_DIR}/accuracy_gpu_m${m}_K1e${logk}.csv" \
            | tee "${RUN_DIR}/accuracy_gpu_m${m}_K1e${logk}.log"
    done
done

printf 'solver,exit_status,stdout_log\n' > "${RAW_DIR}/singular_process_status.csv"
set +e
mpirun -n "${MPI_RANKS}" "${EXP_DIR}/build/singular_probe_cpu.out" \
    "${RAW_DIR}/singular_cpu.csv" 2>&1 | tee "${RUN_DIR}/singular_cpu.log"
cpu_status=${PIPESTATUS[0]}
printf 'cpu-pascal-btdma,%s,%s\n' "${cpu_status}" "singular_cpu.log" \
    >> "${RAW_DIR}/singular_process_status.csv"

mpirun -n "${MPI_RANKS}" "${EXP_DIR}/build/singular_probe_gpu.out" \
    "${RAW_DIR}/singular_gpu.csv" 2>&1 | tee "${RUN_DIR}/singular_gpu.log"
gpu_status=${PIPESTATUS[0]}
printf 'gpu-pascal-btdma,%s,%s\n' "${gpu_status}" "singular_gpu.log" \
    >> "${RAW_DIR}/singular_process_status.csv"
set -e

if compgen -G "${RAW_DIR}/accuracy_*.csv" > /dev/null; then
    python3 "${EXP_DIR}/summarize_results.py" "${RAW_DIR}" "${RUN_DIR}/summary"
fi
