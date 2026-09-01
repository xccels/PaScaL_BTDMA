#!/usr/bin/env bash
#SBATCH --job-name=btdma-s-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --comment=etc
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module purge
module load nvidia-hpc-sdk-hpcx/25.3 cuda/12.8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/output/${SLURM_JOB_ID}/strong_gpu"
NSYS="${NSYS:-16384}"
WARMUPS="${WARMUPS:-2}"
REPEATS="${REPEATS:-5}"
MANGO_SRUN_MPI="${MANGO_SRUN_MPI:-pmix}"
export OMP_NUM_THREADS=1

mkdir -p "${OUT_DIR}"
cd "${SCRIPT_DIR}"

gpu_variants=(alltoall pascal_unoptimized pascal_optimized)
for variant in "${gpu_variants[@]}"; do
    executable="bin/bench_gpu_${variant}"
    if [[ ! -x "${executable}" ]]; then
        echo "Missing executable: ${executable}. Run ./build_mango.sh first." >&2
        exit 1
    fi
done

{
    date --iso-8601=seconds
    hostname
    module list 2>&1
    mpifort --version
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
    git -C "${SCRIPT_DIR}" rev-parse HEAD 2>/dev/null || true
} > "${OUT_DIR}/environment.txt"

for logical_p in 2 4 8; do
    for m in 2 5 8; do
        for variant in "${gpu_variants[@]}"; do
            executable="${SCRIPT_DIR}/bin/bench_gpu_${variant}"
            log_file="${OUT_DIR}/strong_gpu_${variant}_P${logical_p}_m${m}.log"
            echo "[run] strong gpu ${variant}: P=${logical_p}, ranks=${logical_p}, N=2048, m=${m}"
            srun --mpi="${MANGO_SRUN_MPI}" --nodes=1 --ntasks="${logical_p}" \
                --gpus-per-task=1 --gpu-bind=closest --exact \
                "${executable}" strong "${logical_p}" 2048 "${NSYS}" "${m}" \
                "${WARMUPS}" "${REPEATS}" | tee "${log_file}"
        done
    done
done

python3 "${SCRIPT_DIR}/collect_results.py" "${OUT_DIR}"/*.log \
    --raw "${OUT_DIR}/results_raw.csv" --summary "${OUT_DIR}/results_summary.csv"

