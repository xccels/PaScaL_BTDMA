#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${EXP_DIR}/../.." && pwd)"
BUILD_DIR="${EXP_DIR}/build"
MOD_DIR="${BUILD_DIR}/modules"
source "${ROOT_DIR}/00_enviroment.sh"

if [[ "${PASCAL_CORE_READY:-0}" != "1" ]]; then
    "${ROOT_DIR}/build_mango.sh" all
fi
mkdir -p "${BUILD_DIR}" "${MOD_DIR}"

"${MPIFC}" -Mfree -cpp "${OPT}" -module "${MOD_DIR}" -I"${MOD_DIR}" \
    -c "${EXP_DIR}/accuracy_support.f90" -o "${BUILD_DIR}/accuracy_support.o"

"${MPIFC}" -Mfree -cpp "${OPT}" -I"${MOD_DIR}" -I"${ROOT_DIR}/include/cpu" \
    "${EXP_DIR}/accuracy_cpu.f90" "${BUILD_DIR}/accuracy_support.o" \
    "${ROOT_DIR}/lib/libpascal_btdma_cpu.a" -llapack -lblas \
    -o "${BUILD_DIR}/accuracy_cpu.out"

"${MPIFC}" -Mfree -cpp "${OPT}" -I"${MOD_DIR}" -I"${ROOT_DIR}/include/cpu" \
    "${EXP_DIR}/singular_probe_cpu.f90" "${BUILD_DIR}/accuracy_support.o" \
    "${ROOT_DIR}/lib/libpascal_btdma_cpu.a" -llapack -lblas \
    -o "${BUILD_DIR}/singular_probe_cpu.out"

GPU_FLAGS=(-Mfree -cpp -cuda "-gpu=${GPU_ARCH}" "${OPT}")
"${MPIFC}" "${GPU_FLAGS[@]}" -I"${MOD_DIR}" -I"${ROOT_DIR}/include/gpu" \
    "${EXP_DIR}/accuracy_gpu.f90" "${BUILD_DIR}/accuracy_support.o" \
    "${ROOT_DIR}/lib/libpascal_btdma_gpu.a" -llapack -lblas \
    -o "${BUILD_DIR}/accuracy_gpu.out"

"${MPIFC}" "${GPU_FLAGS[@]}" -I"${MOD_DIR}" -I"${ROOT_DIR}/include/gpu" \
    "${EXP_DIR}/singular_probe_gpu.f90" "${BUILD_DIR}/accuracy_support.o" \
    "${ROOT_DIR}/lib/libpascal_btdma_gpu.a" -llapack -lblas \
    -o "${BUILD_DIR}/singular_probe_gpu.out"

echo "Built accuracy and singular-probe executables in ${BUILD_DIR}"
