#!/usr/bin/env bash
#SBATCH --job-name=pascal_example
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --output=output/slurm-%j.out
#SBATCH --error=output/slurm-%j.err

set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${EXAMPLE_DIR}/.." && pwd)"
OUTPUT_DIR="${EXAMPLE_DIR}/output"

mkdir -p "${OUTPUT_DIR}"
source "${ROOT_DIR}/setup_mango.sh"
"${EXAMPLE_DIR}/build_mango.sh"

{
    date -Iseconds
    git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true
    "${MPIFC}" --version | head -n 2
    mpirun --version | head -n 2
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "${OUTPUT_DIR}/environment-${SLURM_JOB_ID}.txt"

mpirun -n 2 "${EXAMPLE_DIR}/build/example_cpu.out" \
    | tee "${OUTPUT_DIR}/cpu-${SLURM_JOB_ID}.txt"

mpirun -n 2 "${EXAMPLE_DIR}/build/example_gpu.out" \
    | tee "${OUTPUT_DIR}/gpu-${SLURM_JOB_ID}.txt"

