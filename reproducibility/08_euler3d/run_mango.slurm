#!/usr/bin/env bash
#SBATCH --job-name=euler3d_btdma
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:30:00
#SBATCH --output=euler3d-%j.out
#SBATCH --error=euler3d-%j.err

set -euo pipefail

CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${CASE_DIR}/../.." && pwd)"
RUN_ID="${SLURM_JOB_ID:-manual}"
RUN_DIR="${CASE_DIR}/output/${RUN_ID}"

mkdir -p "${RUN_DIR}"
source "${ROOT_DIR}/setup_mango.sh"

N1=256 N2=256 N3=256 NP1=2 NP2=2 NP3=2 NSTEPS=0 \
    "${CASE_DIR}/build_mango.sh"

{
    echo "timestamp=$(date -Iseconds)"
    echo "source_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "grid=256x256x256"
    echo "decomposition=2x2x2"
    echo "mpi_ranks=8"
    echo "steps=one_period"
    "${MPIFC}" --version | head -n 2
    mpirun --version | head -n 2
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "${RUN_DIR}/environment.txt"

cd "${RUN_DIR}"
mpirun -np 8 "${CASE_DIR}/build/euler3d_mpi_gpu.out" \
    | tee "run.log"

python3 "${CASE_DIR}/postprocess.py" --run-dir "${RUN_DIR}"

echo "Results: ${RUN_DIR}"

