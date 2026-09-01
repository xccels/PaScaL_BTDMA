#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BIN_DIR="${SCRIPT_DIR}/bin"
source "${ROOT_DIR}/00_enviroment.sh"

mkdir -p "${BUILD_DIR}" "${BIN_DIR}"
rm -f "${BUILD_DIR}/weno2d_gpu.o" "${BIN_DIR}/weno2d_gpu"

if [[ "${PASCAL_CORE_READY:-0}" != "1" ]]; then
    "${ROOT_DIR}/build_mango.sh" gpu
fi

flags=(-Mfree -cpp -cuda "-gpu=${GPU_ARCH}" "${OPT}" -DGPU_ENABLED)
defs=(-DN1=516 -DN2=516 -DNP1=2 -DNP2=2 -DM=3)

"${MPIFC}" "${flags[@]}" "${defs[@]}" \
    -I"${ROOT_DIR}/include/gpu" \
    -module "${BUILD_DIR}" \
    -c "${SCRIPT_DIR}/weno2d_gpu.f90" \
    -o "${BUILD_DIR}/weno2d_gpu.o"

"${MPIFC}" -cuda "-gpu=${GPU_ARCH}" "${OPT}" \
    "${BUILD_DIR}/weno2d_gpu.o" \
    "${ROOT_DIR}/lib/libpascal_btdma_gpu.a" \
    -o "${BIN_DIR}/weno2d_gpu"

echo "Built ${BIN_DIR}/weno2d_gpu"
