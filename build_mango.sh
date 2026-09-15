#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-all}"
source "${ROOT_DIR}/00_enviroment.sh"

SRC_DIR="${ROOT_DIR}/lib_btdma"
BUILD_DIR="${ROOT_DIR}/build"
INCLUDE_DIR="${ROOT_DIR}/include"
LIB_DIR="${ROOT_DIR}/lib"

build_cpu() {
    local obj_dir="${BUILD_DIR}/cpu"
    local mod_dir="${INCLUDE_DIR}/cpu"

    mkdir -p "${obj_dir}" "${mod_dir}" "${LIB_DIR}"
    rm -f "${obj_dir}"/*.o "${mod_dir}"/*.mod

    "${MPIFC}" -Mfree -Mextend -cpp "${OPT}" \
        -module "${mod_dir}" -I"${mod_dir}" \
        -c "${SRC_DIR}/mpiutil.f90" \
        -o "${obj_dir}/mpiutil.o"

    "${MPIFC}" -Mfree -Mextend -cpp "${OPT}" \
        -module "${mod_dir}" -I"${mod_dir}" \
        -c "${SRC_DIR}/mod_btdma_cpu.f90" \
        -o "${obj_dir}/mod_btdma_cpu.o"

    ar rcs "${LIB_DIR}/libpascal_btdma_cpu.a" \
        "${obj_dir}/mpiutil.o" \
        "${obj_dir}/mod_btdma_cpu.o"

    echo "Built ${LIB_DIR}/libpascal_btdma_cpu.a"
}

build_gpu() {
    local obj_dir="${BUILD_DIR}/gpu"
    local mod_dir="${INCLUDE_DIR}/gpu"
    local flags=(-Mfree -Mextend -cpp -cuda "-gpu=${GPU_ARCH}" "${OPT}" -DGPU_ENABLED)

    mkdir -p "${obj_dir}" "${mod_dir}" "${LIB_DIR}"
    rm -f "${obj_dir}"/*.o "${mod_dir}"/*.mod

    "${MPIFC}" "${flags[@]}" \
        -module "${mod_dir}" -I"${mod_dir}" \
        -c "${SRC_DIR}/mpiutil.f90" \
        -o "${obj_dir}/mpiutil.o"

    "${MPIFC}" "${flags[@]}" \
        -module "${mod_dir}" -I"${mod_dir}" \
        -c "${SRC_DIR}/mod_cudatools.f90" \
        -o "${obj_dir}/mod_cudatools.o"

    "${MPIFC}" "${flags[@]}" \
        -module "${mod_dir}" -I"${mod_dir}" \
        -c "${SRC_DIR}/mod_btdma_gpu_v2.f90" \
        -o "${obj_dir}/mod_btdma_gpu_v2.o"

    ar rcs "${LIB_DIR}/libpascal_btdma_gpu.a" \
        "${obj_dir}/mpiutil.o" \
        "${obj_dir}/mod_cudatools.o" \
        "${obj_dir}/mod_btdma_gpu_v2.o"

    echo "Built ${LIB_DIR}/libpascal_btdma_gpu.a"
}

case "${TARGET}" in
    cpu)
        build_cpu
        ;;
    gpu)
        build_gpu
        ;;
    all)
        build_cpu
        build_gpu
        ;;
    *)
        echo "Usage: $0 [cpu|gpu|all]" >&2
        exit 2
        ;;
esac
