#!/usr/bin/env bash
#SBATCH --job-name=btdma-phase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/output/${SLURM_JOB_ID}"
WARMUPS="${WARMUPS:-1}"
REPEATS="${REPEATS:-5}"
MANGO_SRUN_MPI="${MANGO_SRUN_MPI:-pmix}"
export OMP_NUM_THREADS=1

mkdir -p "${OUT_DIR}"
source "${ROOT_DIR}/setup_mango.sh"
"${SCRIPT_DIR}/build_mango.sh"

{
    date --iso-8601=seconds
    hostname
    module list 2>&1
    "${MPIFC}" --version
    mpirun --version
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
    nvidia-smi topo -m
    git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true
    sha256sum "${ROOT_DIR}"/lib_btdma/*.f90 \
        "${SCRIPT_DIR}/phase_timing_gpu.f90" \
        "${SCRIPT_DIR}/build_mango.sh" \
        "${SCRIPT_DIR}/run_mango.slurm"
    printf 'WARMUPS=%s\nREPEATS=%s\n' "${WARMUPS}" "${REPEATS}"
} > "${OUT_DIR}/environment.txt"

for p in 2 4 8; do
    log_file="${OUT_DIR}/phase_P${p}.log"
    echo "[run] optimized GPU phase timing: P=${p}, m=8, N=2048, nsys=16384"
    srun --mpi="${MANGO_SRUN_MPI}" --nodes=1 --ntasks="${p}" \
        --gpus-per-task=1 --gpu-bind=closest --exact \
        "${SCRIPT_DIR}/build/phase_timing_gpu.out" "${WARMUPS}" "${REPEATS}" \
        | tee "${log_file}"
done

python3 "${SCRIPT_DIR}/collect_phase_timing.py" "${OUT_DIR}"/phase_P*.log \
    --raw "${OUT_DIR}/phase_raw.csv" \
    --summary "${OUT_DIR}/phase_summary.csv"

echo "Results written under ${OUT_DIR}"
