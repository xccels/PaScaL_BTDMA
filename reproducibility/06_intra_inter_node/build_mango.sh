#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CORE_DIR="${ROOT_DIR}/lib_btdma"
BUILD_DIR="${SCRIPT_DIR}/build"
BIN_DIR="${SCRIPT_DIR}/bin"

source "${ROOT_DIR}/00_enviroment.sh"

required_sources=(
    "${CORE_DIR}/mpiutil.f90"
    "${CORE_DIR}/mod_cudatools.f90"
    "${CORE_DIR}/mod_btdma_gpu_v2.f90"
    "${SCRIPT_DIR}/legacy/mod_btdma_gpu.f90"
    "${SCRIPT_DIR}/node_compare_support.f90"
    "${SCRIPT_DIR}/bench_alltoall_gpu.f90"
    "${SCRIPT_DIR}/bench_pascal_gpu.f90"
)
for source_file in "${required_sources[@]}"; do
    if [[ ! -f "${source_file}" ]]; then
        echo "Missing source: ${source_file}" >&2
        exit 1
    fi
done

mkdir -p "${BUILD_DIR}" "${BIN_DIR}"

gpu_flags=(-Mfree -Mextend -cpp "${OPT}" -cuda "-gpu=${GPU_ARCH}")
if [[ -n "${PASCAL_EXTRA_GPU_FLAGS:-}" ]]; then
    read -r -a extra_gpu_flags <<< "${PASCAL_EXTRA_GPU_FLAGS}"
    gpu_flags+=("${extra_gpu_flags[@]}")
fi

echo "[build] ${MPIFC} ${gpu_flags[*]}"
"${MPIFC}" --version

"${MPIFC}" "${gpu_flags[@]}" -DGPU_ENABLED -c "${CORE_DIR}/mpiutil.f90" \
    -module "${BUILD_DIR}" -o "${BUILD_DIR}/mpiutil.o"
"${MPIFC}" "${gpu_flags[@]}" -c "${CORE_DIR}/mod_cudatools.f90" \
    -module "${BUILD_DIR}" -o "${BUILD_DIR}/mod_cudatools.o"
"${MPIFC}" "${gpu_flags[@]}" -c "${SCRIPT_DIR}/legacy/mod_btdma_gpu.f90" \
    -module "${BUILD_DIR}" -o "${BUILD_DIR}/mod_btdma_gpu.o"
"${MPIFC}" "${gpu_flags[@]}" -c "${CORE_DIR}/mod_btdma_gpu_v2.f90" \
    -module "${BUILD_DIR}" -o "${BUILD_DIR}/mod_btdma_gpu_v2.o"
"${MPIFC}" "${gpu_flags[@]}" -c "${SCRIPT_DIR}/node_compare_support.f90" \
    -module "${BUILD_DIR}" -o "${BUILD_DIR}/node_compare_support.o"

for driver in bench_alltoall_gpu bench_pascal_gpu; do
    "${MPIFC}" "${gpu_flags[@]}" -c "${SCRIPT_DIR}/${driver}.f90" \
        -module "${BUILD_DIR}" -o "${BUILD_DIR}/${driver}.o"
done

"${MPIFC}" "${gpu_flags[@]}" \
    "${BUILD_DIR}/mpiutil.o" \
    "${BUILD_DIR}/mod_cudatools.o" \
    "${BUILD_DIR}/mod_btdma_gpu.o" \
    "${BUILD_DIR}/node_compare_support.o" \
    "${BUILD_DIR}/bench_alltoall_gpu.o" \
    -o "${BIN_DIR}/bench_alltoall_gpu"

"${MPIFC}" "${gpu_flags[@]}" \
    "${BUILD_DIR}/mod_cudatools.o" \
    "${BUILD_DIR}/mod_btdma_gpu_v2.o" \
    "${BUILD_DIR}/node_compare_support.o" \
    "${BUILD_DIR}/bench_pascal_gpu.o" \
    -o "${BIN_DIR}/bench_pascal_gpu"

echo "[build] Created ${BIN_DIR}/bench_alltoall_gpu and bench_pascal_gpu"
