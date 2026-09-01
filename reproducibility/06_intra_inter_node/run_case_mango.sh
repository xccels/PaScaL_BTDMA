#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <intra-node|inter-node>" >&2
    exit 2
fi

REGIME="$1"
case "${REGIME}" in
    intra-node)
        NODES=1
        TASKS_PER_NODE=4
        ;;
    inter-node)
        NODES=4
        TASKS_PER_NODE=1
        ;;
    *)
        echo "Regime must be intra-node or inter-node." >&2
        exit 2
        ;;
esac

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "This script must run inside the corresponding Slurm allocation." >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/output/${SLURM_JOB_ID}/${REGIME}"
WARMUPS="${WARMUPS:-2}"
REPEATS="${REPEATS:-5}"
MANGO_SRUN_MPI="${MANGO_SRUN_MPI:-pmix}"
export OMP_NUM_THREADS=1

source "${ROOT_DIR}/setup_mango.sh"
mkdir -p "${OUT_DIR}"

for executable in bench_alltoall_gpu bench_pascal_gpu; do
    if [[ ! -x "${SCRIPT_DIR}/bin/${executable}" ]]; then
        echo "Missing ${SCRIPT_DIR}/bin/${executable}; run ./build_mango.sh before sbatch." >&2
        exit 1
    fi
done

{
    printf 'experiment=06_intra_inter_node\n'
    printf 'regime=%s\n' "${REGIME}"
    printf 'fixed_condition=m8_N2048_nsys16384_nproc4\n'
    printf 'nodes=%s\n' "${NODES}"
    printf 'tasks_per_node=%s\n' "${TASKS_PER_NODE}"
    printf 'warmups=%s\n' "${WARMUPS}"
    printf 'repeats=%s\n' "${REPEATS}"
    date --iso-8601=seconds
    module list 2>&1
    "${MPIFC}" --version
    mpirun --version
    scontrol show job "${SLURM_JOB_ID}"
    git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true
    sha256sum \
        "${ROOT_DIR}"/lib_btdma/*.f90 \
        "${SCRIPT_DIR}"/*.f90 \
        "${SCRIPT_DIR}"/*.sh \
        "${SCRIPT_DIR}"/*.slurm \
        "${SCRIPT_DIR}"/*.py \
        "${SCRIPT_DIR}/README.md" \
        "${SCRIPT_DIR}/legacy/SOURCE_PROVENANCE.md" \
        "${SCRIPT_DIR}/legacy/mod_btdma_gpu.f90"
} > "${OUT_DIR}/manifest.txt"

srun --mpi="${MANGO_SRUN_MPI}" --nodes="${NODES}" --ntasks=4 \
    --ntasks-per-node="${TASKS_PER_NODE}" --gpus-per-task=1 --gpu-bind=closest \
    --exact bash -lc \
    'printf "rank=%s local_rank=%s host=%s cuda_visible=%s\n" "${SLURM_PROCID}" "${SLURM_LOCALID}" "$(hostname)" "${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader' \
    | sort > "${OUT_DIR}/rank_gpu_map.txt"

for method in alltoall pascal; do
    executable="${SCRIPT_DIR}/bin/bench_${method}_gpu"
    log_file="${OUT_DIR}/${method}.log"
    echo "[run] ${REGIME} ${method}: m=8, N=2048, nsys=16384, nproc=4"
    srun --mpi="${MANGO_SRUN_MPI}" --nodes="${NODES}" --ntasks=4 \
        --ntasks-per-node="${TASKS_PER_NODE}" --gpus-per-task=1 --gpu-bind=closest \
        --exact "${executable}" "${REGIME}" "${WARMUPS}" "${REPEATS}" \
        | tee "${log_file}"
done

echo "Raw logs and provenance were written to ${OUT_DIR}"
