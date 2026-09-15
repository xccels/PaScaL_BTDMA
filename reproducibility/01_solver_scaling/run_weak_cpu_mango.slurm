#!/usr/bin/env bash
#SBATCH --job-name=btdma-w-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=48
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --comment=etc
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module purge
module load nvidia-hpc-sdk-hpcx/25.3 cuda/12.8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/output/${SLURM_JOB_ID}/weak_cpu"
NSYS="${NSYS:-16384}"
WARMUPS="${WARMUPS:-2}"
REPEATS="${REPEATS:-5}"
MANGO_SRUN_MPI="${MANGO_SRUN_MPI:-pmix}"
export OMP_NUM_THREADS=1
export OMPI_MCA_coll_ucc_enable=0

mkdir -p "${OUT_DIR}"
cd "${SCRIPT_DIR}"

for executable in bin/bench_cpu_alltoall bin/bench_cpu_pascal; do
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
    lscpu
    git -C "${SCRIPT_DIR}" rev-parse HEAD 2>/dev/null || true
} > "${OUT_DIR}/environment.txt"

for logical_p in 2 4 8; do
    ranks=$((24 * logical_p))
    nodes=$((logical_p / 2))
    n_global=$((256 * logical_p))
    for m in 2 5 8; do
        for variant in alltoall pascal; do
            executable="${SCRIPT_DIR}/bin/bench_cpu_${variant}"
            log_file="${OUT_DIR}/weak_cpu_${variant}_P${logical_p}_m${m}.log"
            echo "[run] weak cpu ${variant}: P=${logical_p}, ranks=${ranks}, N=${n_global}, m=${m}"
            srun --mpi="${MANGO_SRUN_MPI}" --nodes="${nodes}" --ntasks="${ranks}" \
                --ntasks-per-node=48 --cpu-bind=cores --exact \
                "${executable}" weak "${logical_p}" "${n_global}" "${NSYS}" "${m}" \
                "${WARMUPS}" "${REPEATS}" | tee "${log_file}"
        done
    done
done

python3 "${SCRIPT_DIR}/collect_results.py" "${OUT_DIR}"/*.log \
    --raw "${OUT_DIR}/results_raw.csv" --summary "${OUT_DIR}/results_summary.csv"

