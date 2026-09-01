#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
source "${ROOT_DIR}/00_enviroment.sh"

if ! command -v "${MPIFC}" >/dev/null 2>&1; then
    echo "MPI Fortran wrapper not found: ${MPIFC}" >&2
    echo "Check ${ROOT_DIR}/00_enviroment.sh" >&2
    exit 1
fi

if [[ "${PASCAL_CORE_READY:-0}" != "1" ]]; then
    "${ROOT_DIR}/build_mango.sh" gpu
fi
mkdir -p "${BUILD_DIR}"

"${MPIFC}" -Mfree -Mextend -cpp -cuda "-gpu=${GPU_ARCH}" "${OPT}" \
    -I"${ROOT_DIR}/include/gpu" \
    "${SCRIPT_DIR}/phase_timing_gpu.f90" \
    "${ROOT_DIR}/lib/libpascal_btdma_gpu.a" \
    -o "${BUILD_DIR}/phase_timing_gpu.out"

echo "Built ${BUILD_DIR}/phase_timing_gpu.out"
