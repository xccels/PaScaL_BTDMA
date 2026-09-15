#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${CASE_DIR}/../.." && pwd)"
BUILD_DIR="${CASE_DIR}/build"
MOD_DIR="${BUILD_DIR}/mod"
source "${ROOT_DIR}/00_enviroment.sh"

N1="${N1:-256}"
N2="${N2:-256}"
N3="${N3:-256}"
NP1="${NP1:-2}"
NP2="${NP2:-2}"
NP3="${NP3:-2}"
NSTEPS="${NSTEPS:-0}"

mkdir -p "${BUILD_DIR}" "${MOD_DIR}"

if [[ "${PASCAL_CORE_READY:-0}" != "1" ]]; then
    "${ROOT_DIR}/build_mango.sh" gpu
fi

flags=(
    -Mfree
    -cpp
    -cuda
    "-gpu=${GPU_ARCH}"
    "${OPT}"
    -DGPU_ENABLED
    "-DN1=${N1}"
    "-DN2=${N2}"
    "-DN3=${N3}"
    "-DNP1=${NP1}"
    "-DNP2=${NP2}"
    "-DNP3=${NP3}"
    "-DNSTEPS=${NSTEPS}"
)

"${MPIFC}" "${flags[@]}" \
    -module "${MOD_DIR}" \
    -I"${ROOT_DIR}/include/gpu" \
    "${CASE_DIR}/euler3d_mpi_gpu.f90" \
    "${ROOT_DIR}/lib/libpascal_btdma_gpu.a" \
    -o "${BUILD_DIR}/euler3d_mpi_gpu.out"

echo "Built ${BUILD_DIR}/euler3d_mpi_gpu.out"
echo "Grid=${N1}x${N2}x${N3}; decomposition=${NP1}x${NP2}x${NP3}; NSTEPS=${NSTEPS}"
