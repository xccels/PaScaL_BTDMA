#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BIN_DIR="${SCRIPT_DIR}/bin"
source "${ROOT_DIR}/00_enviroment.sh"

if ! command -v "${MPIFC}" >/dev/null 2>&1; then
    echo "MPI Fortran wrapper not found: ${MPIFC}" >&2
    exit 1
fi

if [[ "${PASCAL_CORE_READY:-0}" != "1" ]]; then
    "${ROOT_DIR}/build_mango.sh" all
fi

mkdir -p "${BUILD_DIR}/cpu" "${BUILD_DIR}/gpu" "${BIN_DIR}"
rm -f "${BUILD_DIR}/cpu"/* "${BUILD_DIR}/gpu"/*

cpu_flags=(-Mfree -Mextend -cpp "${OPT}")
gpu_flags=(-Mfree -Mextend -cpp "${OPT}" -cuda "-gpu=${GPU_ARCH}")

"${MPIFC}" "${cpu_flags[@]}" -c "${SCRIPT_DIR}/throughput_common.f90" \
    -module "${BUILD_DIR}/cpu" -o "${BUILD_DIR}/cpu/throughput_common.o"
"${MPIFC}" "${cpu_flags[@]}" -I"${ROOT_DIR}/include/cpu" \
    -I"${BUILD_DIR}/cpu" -c "${SCRIPT_DIR}/throughput_cpu.f90" \
    -module "${BUILD_DIR}/cpu" -o "${BUILD_DIR}/cpu/throughput_cpu.o"
"${MPIFC}" "${cpu_flags[@]}" \
    "${BUILD_DIR}/cpu/throughput_common.o" \
    "${BUILD_DIR}/cpu/throughput_cpu.o" \
    "${ROOT_DIR}/lib/libpascal_btdma_cpu.a" -llapack -lblas \
    -o "${BIN_DIR}/throughput_cpu"

"${MPIFC}" "${gpu_flags[@]}" -c "${SCRIPT_DIR}/throughput_common.f90" \
    -module "${BUILD_DIR}/gpu" -o "${BUILD_DIR}/gpu/throughput_common.o"
"${MPIFC}" "${gpu_flags[@]}" -I"${ROOT_DIR}/include/gpu" \
    -I"${BUILD_DIR}/gpu" -c "${SCRIPT_DIR}/throughput_gpu.f90" \
    -module "${BUILD_DIR}/gpu" -o "${BUILD_DIR}/gpu/throughput_gpu.o"
"${MPIFC}" "${gpu_flags[@]}" \
    "${BUILD_DIR}/gpu/throughput_common.o" \
    "${BUILD_DIR}/gpu/throughput_gpu.o" \
    "${ROOT_DIR}/lib/libpascal_btdma_gpu.a" \
    -o "${BIN_DIR}/throughput_gpu"

echo "Built ${BIN_DIR}/throughput_cpu"
echo "Built ${BIN_DIR}/throughput_gpu"
