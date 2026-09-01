#!/usr/bin/env bash
#SBATCH --job-name=btdma_comm_N
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../.." && pwd)"
MODE="${MODE:-gpu}"
WARMUPS="${WARMUPS:-2}"
REPEATS="${REPEATS:-5}"
P_LABEL=8
NSYS=16384
M_VALUES=(2 5 8)
N_VALUES=(512 1024 2048 4096)

# The allocation is intentionally supplied at sbatch time because the Mango
# partition/account names are allocation-specific.
source "${REPO_ROOT}/setup_mango.sh"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMPI_MCA_coll_ucc_enable=0
export UCX_MEMTYPE_CACHE=n

case "${MODE}" in
    cpu)
        expected_tasks=192
        expected_nodes=4
        alltoall_exe="${HERE}/bin/communication_alltoall_cpu"
        pascal_exe="${HERE}/bin/communication_pascal_cpu"
        srun_args=(--nodes=4 --ntasks=192 --ntasks-per-node=48 --cpus-per-task=1 \
                   --distribution=block:cyclic --hint=nomultithread --cpu-bind=cores)
        ;;
    gpu)
        expected_tasks=8
        expected_nodes=1
        alltoall_exe="${HERE}/bin/communication_alltoall_gpu"
        pascal_exe="${HERE}/bin/communication_pascal_gpu"
        srun_args=(--nodes=1 --ntasks=8 --ntasks-per-node=8 --distribution=block:block \
                   --gpus-per-task=1 --gpu-bind=single:1)
        ;;
    *)
        echo "MODE must be cpu or gpu" >&2
        exit 2
        ;;
esac

if [[ "${SLURM_NTASKS:-0}" -ne "${expected_tasks}" ]]; then
    echo "MODE=${MODE} requires SLURM_NTASKS=${expected_tasks}" >&2
    exit 2
fi
if [[ "${SLURM_NNODES:-0}" -ne "${expected_nodes}" ]]; then
    echo "MODE=${MODE} requires SLURM_NNODES=${expected_nodes}" >&2
    exit 2
fi
if [[ ! -x "${alltoall_exe}" || ! -x "${pascal_exe}" ]]; then
    echo "Build the ${MODE} executables with ./build_mango.sh ${MODE} first" >&2
    exit 2
fi

echo "MANIFEST,date_utc,$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "MANIFEST,mode,${MODE}"
echo "MANIFEST,job_id,${SLURM_JOB_ID:-unknown}"
echo "MANIFEST,nodes,${SLURM_NNODES}"
echo "MANIFEST,tasks,${SLURM_NTASKS}"
echo "MANIFEST,node_list,${SLURM_JOB_NODELIST:-unknown}"
echo "MANIFEST,P_label,${P_LABEL}"
echo "MANIFEST,nsys,${NSYS}"
echo "MANIFEST,warmups,${WARMUPS}"
echo "MANIFEST,repeats,${REPEATS}"
echo "MANIFEST,mpifc,$(${MPIFC} --version | sed -n '1p')"
module list 2>&1 | sed 's/^/MANIFEST,module,/'

for m in "${M_VALUES[@]}"; do
    for n_global in "${N_VALUES[@]}"; do
        echo "CASE,${MODE},alltoall,P=${P_LABEL},nsys=${NSYS},m=${m},N=${n_global}"
        srun "${srun_args[@]}" "${alltoall_exe}" \
            "${n_global}" "${m}" "${WARMUPS}" "${REPEATS}" "${P_LABEL}"

        echo "CASE,${MODE},pascal,P=${P_LABEL},nsys=${NSYS},m=${m},N=${n_global}"
        srun "${srun_args[@]}" "${pascal_exe}" \
            "${n_global}" "${m}" "${WARMUPS}" "${REPEATS}" "${P_LABEL}"
    done
done
