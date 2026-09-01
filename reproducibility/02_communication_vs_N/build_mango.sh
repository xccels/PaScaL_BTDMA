#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../.." && pwd)"
CORE_SRC="${REPO_ROOT}/lib_btdma"
TARGET="${1:-all}"
source "${REPO_ROOT}/00_enviroment.sh"

BUILD_DIR="${HERE}/build"
BIN_DIR="${HERE}/bin"

if [[ ! -f "${CORE_SRC}/mod_btdma_gpu_v2.f90" ]]; then
    echo "Expected public core at ${HERE}/../../lib_btdma" >&2
    exit 2
fi

build_cpu() {
    local object_dir="${BUILD_DIR}/cpu"
    local module_dir="${object_dir}/modules"
    local core_modules="${REPO_ROOT}/include/cpu"
    local flags=(-Mfree -cpp "${OPT}")

    if [[ "${PASCAL_CORE_READY:-0}" != "1" ]]; then
        "${REPO_ROOT}/build_mango.sh" cpu
    fi
    mkdir -p "${object_dir}" "${module_dir}" "${BIN_DIR}"
    rm -f "${object_dir}"/*.o "${module_dir}"/*.mod

    "${MPIFC}" "${flags[@]}" -module "${module_dir}" -I"${core_modules}" \
        -c "${HERE}/benchmark_common.f90" -o "${object_dir}/benchmark_common.o"
    "${MPIFC}" "${flags[@]}" -module "${module_dir}" -I"${module_dir}" -I"${core_modules}" \
        -c "${HERE}/benchmark_alltoall_cpu.f90" -o "${object_dir}/benchmark_alltoall_cpu.o"
    "${MPIFC}" "${flags[@]}" -module "${module_dir}" -I"${module_dir}" -I"${core_modules}" \
        -c "${HERE}/benchmark_pascal_cpu.f90" -o "${object_dir}/benchmark_pascal_cpu.o"

    "${MPIFC}" "${object_dir}/benchmark_common.o" "${object_dir}/benchmark_alltoall_cpu.o" \
        -L"${REPO_ROOT}/lib" -lpascal_btdma_cpu -llapack -lblas \
        -o "${BIN_DIR}/communication_alltoall_cpu"
    "${MPIFC}" "${object_dir}/benchmark_common.o" "${object_dir}/benchmark_pascal_cpu.o" \
        -L"${REPO_ROOT}/lib" -lpascal_btdma_cpu -llapack -lblas \
        -o "${BIN_DIR}/communication_pascal_cpu"
}

build_gpu() {
    local object_dir="${BUILD_DIR}/gpu"
    local module_dir="${object_dir}/modules"
    local core_modules="${REPO_ROOT}/include/gpu"
    local flags=(-Mfree -cpp -cuda "-gpu=${GPU_ARCH}" "${OPT}" -DGPU_ENABLED)

    if [[ "${PASCAL_CORE_READY:-0}" != "1" ]]; then
        "${REPO_ROOT}/build_mango.sh" gpu
    fi
    mkdir -p "${object_dir}" "${module_dir}" "${BIN_DIR}"
    rm -f "${object_dir}"/*.o "${module_dir}"/*.mod

    "${MPIFC}" "${flags[@]}" -module "${module_dir}" -I"${core_modules}" \
        -c "${HERE}/benchmark_common.f90" -o "${object_dir}/benchmark_common.o"
    "${MPIFC}" "${flags[@]}" -module "${module_dir}" -I"${module_dir}" -I"${core_modules}" \
        -c "${HERE}/baseline_btdma_gpu.f90" -o "${object_dir}/baseline_btdma_gpu.o"
    "${MPIFC}" "${flags[@]}" -module "${module_dir}" -I"${module_dir}" -I"${core_modules}" \
        -c "${HERE}/benchmark_alltoall_gpu.f90" -o "${object_dir}/benchmark_alltoall_gpu.o"
    "${MPIFC}" "${flags[@]}" -module "${module_dir}" -I"${module_dir}" -I"${core_modules}" \
        -c "${HERE}/benchmark_pascal_gpu.f90" -o "${object_dir}/benchmark_pascal_gpu.o"

    "${MPIFC}" -cuda "-gpu=${GPU_ARCH}" "${OPT}" \
        "${object_dir}/benchmark_common.o" "${object_dir}/baseline_btdma_gpu.o" \
        "${object_dir}/benchmark_alltoall_gpu.o" \
        -L"${REPO_ROOT}/lib" -lpascal_btdma_gpu \
        -o "${BIN_DIR}/communication_alltoall_gpu"
    "${MPIFC}" -cuda "-gpu=${GPU_ARCH}" "${OPT}" \
        "${object_dir}/benchmark_common.o" "${object_dir}/benchmark_pascal_gpu.o" \
        -L"${REPO_ROOT}/lib" -lpascal_btdma_gpu \
        -o "${BIN_DIR}/communication_pascal_gpu"
}

case "${TARGET}" in
    cpu) build_cpu ;;
    gpu) build_gpu ;;
    all) build_cpu; build_gpu ;;
    *)
        echo "Usage: $0 [cpu|gpu|all]" >&2
        exit 2
        ;;
esac
