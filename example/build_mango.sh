#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${EXAMPLE_DIR}/.." && pwd)"
source "${ROOT_DIR}/00_enviroment.sh"

if [[ "${PASCAL_CORE_READY:-0}" != "1" ]]; then
    "${ROOT_DIR}/build_mango.sh" all
fi
mkdir -p "${EXAMPLE_DIR}/build"

"${MPIFC}" -Mfree -Mextend -cpp "${OPT}" \
    -I"${ROOT_DIR}/include/cpu" \
    "${EXAMPLE_DIR}/example_cpu.f90" \
    "${ROOT_DIR}/lib/libpascal_btdma_cpu.a" \
    -llapack -lblas \
    -o "${EXAMPLE_DIR}/build/example_cpu.out"

"${MPIFC}" -Mfree -Mextend -cpp -cuda "-gpu=${GPU_ARCH}" "${OPT}" \
    -I"${ROOT_DIR}/include/gpu" \
    "${EXAMPLE_DIR}/example_gpu.f90" \
    "${ROOT_DIR}/lib/libpascal_btdma_gpu.a" \
    -o "${EXAMPLE_DIR}/build/example_gpu.out"

echo "Built:"
echo "  ${EXAMPLE_DIR}/build/example_cpu.out"
echo "  ${EXAMPLE_DIR}/build/example_gpu.out"
