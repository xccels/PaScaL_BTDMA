#!/usr/bin/env bash

# PaScaL_BTDMA build environment.
#
# Edit this file when the compiler, MPI installation, CUDA version, or GPU
# target changes.  The default values below are the Mango login-node
# interactive environment used for the verified build and run.

PASCAL_ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PASCAL_ROOT_DIR

if [[ "${PASCAL_ENVIRONMENT_READY:-0}" != "1" ]]; then
    if ! type module >/dev/null 2>&1; then
        echo "Environment Modules is unavailable. Edit 00_enviroment.sh for this system." >&2
        return 1 2>/dev/null || exit 1
    fi

    module purge
    module load nvidia-hpc-sdk-hpcx/25.3
    module load cuda/12.8
    export PASCAL_ENVIRONMENT_READY=1
fi

export MPIFC="${MPIFC:-${PASCAL_MPIFC:-mpifort}}"
export PASCAL_MPIFC="${MPIFC}"
export GPU_ARCH="${GPU_ARCH:-cc90,cuda12.8}"
export OPT="${OPT:--O3}"

# Mango launch defaults used by the existing run scripts.
export MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
export MANGO_SRUN_MPI="${MANGO_SRUN_MPI:-pmix}"
