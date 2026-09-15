#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CORE_DIR="${SCRIPT_DIR}/../../lib_btdma"
BUILD_DIR="${SCRIPT_DIR}/build"
BIN_DIR="${SCRIPT_DIR}/bin"
CPU_OBJ="${BUILD_DIR}/cpu"
GPU_OBJ="${BUILD_DIR}/gpu"
source "${ROOT_DIR}/00_enviroment.sh"

if ! command -v "${PASCAL_MPIFC}" >/dev/null 2>&1; then
    echo "[build] MPI Fortran wrapper not found: ${PASCAL_MPIFC}" >&2
    exit 1
fi

required_sources=(
    "${CORE_DIR}/mpiutil.f90"
    "${CORE_DIR}/mod_cudatools.f90"
    "${CORE_DIR}/mod_btdma_cpu.f90"
    "${CORE_DIR}/mod_btdma_gpu_v2.f90"
    "${SCRIPT_DIR}/legacy/mod_btdma_gpu.f90"
)
for source_file in "${required_sources[@]}"; do
    if [[ ! -f "${source_file}" ]]; then
        echo "[build] Required source is missing: ${source_file}" >&2
        exit 1
    fi
done

mkdir -p "${CPU_OBJ}" "${GPU_OBJ}" "${BIN_DIR}"

cpu_flags=(-Mfree -Mextend -cpp "${OPT}")
gpu_flags=(-Mfree -Mextend -cpp "${OPT}" -cuda "-gpu=${GPU_ARCH}")
cpu_link_flags=("${OPT}" -llapack -lblas)
gpu_link_flags=("${OPT}" -cuda "-gpu=${GPU_ARCH}")

if [[ -n "${PASCAL_EXTRA_CPU_FLAGS:-}" ]]; then
    read -r -a extra_cpu_flags <<< "${PASCAL_EXTRA_CPU_FLAGS}"
    cpu_flags+=("${extra_cpu_flags[@]}")
fi
if [[ -n "${PASCAL_EXTRA_GPU_FLAGS:-}" ]]; then
    read -r -a extra_gpu_flags <<< "${PASCAL_EXTRA_GPU_FLAGS}"
    gpu_flags+=("${extra_gpu_flags[@]}")
    gpu_link_flags+=("${extra_gpu_flags[@]}")
fi

echo "[build] compiler: ${PASCAL_MPIFC}"
"${PASCAL_MPIFC}" --version

echo "[build] CPU objects"
"${PASCAL_MPIFC}" "${cpu_flags[@]}" -c "${CORE_DIR}/mpiutil.f90" \
    -module "${CPU_OBJ}" -o "${CPU_OBJ}/mpiutil.o"
"${PASCAL_MPIFC}" "${cpu_flags[@]}" -c "${CORE_DIR}/mod_btdma_cpu.f90" \
    -module "${CPU_OBJ}" -o "${CPU_OBJ}/mod_btdma_cpu.o"
"${PASCAL_MPIFC}" "${cpu_flags[@]}" -c "${SCRIPT_DIR}/benchmark_support.f90" \
    -module "${CPU_OBJ}" -o "${CPU_OBJ}/benchmark_support.o"
"${PASCAL_MPIFC}" "${cpu_flags[@]}" -c "${SCRIPT_DIR}/bench_cpu_alltoall.f90" \
    -module "${CPU_OBJ}" -o "${CPU_OBJ}/bench_cpu_alltoall.o"
"${PASCAL_MPIFC}" "${cpu_flags[@]}" -c "${SCRIPT_DIR}/bench_cpu_pascal.f90" \
    -module "${CPU_OBJ}" -o "${CPU_OBJ}/bench_cpu_pascal.o"

"${PASCAL_MPIFC}" "${CPU_OBJ}/mpiutil.o" "${CPU_OBJ}/mod_btdma_cpu.o" \
    "${CPU_OBJ}/benchmark_support.o" "${CPU_OBJ}/bench_cpu_alltoall.o" \
    "${cpu_link_flags[@]}" -o "${BIN_DIR}/bench_cpu_alltoall"
"${PASCAL_MPIFC}" "${CPU_OBJ}/mpiutil.o" "${CPU_OBJ}/mod_btdma_cpu.o" \
    "${CPU_OBJ}/benchmark_support.o" "${CPU_OBJ}/bench_cpu_pascal.o" \
    "${cpu_link_flags[@]}" -o "${BIN_DIR}/bench_cpu_pascal"

echo "[build] GPU objects"
"${PASCAL_MPIFC}" "${gpu_flags[@]}" -DGPU_ENABLED -c "${CORE_DIR}/mpiutil.f90" \
    -module "${GPU_OBJ}" -o "${GPU_OBJ}/mpiutil.o"
"${PASCAL_MPIFC}" "${gpu_flags[@]}" -c "${CORE_DIR}/mod_cudatools.f90" \
    -module "${GPU_OBJ}" -o "${GPU_OBJ}/mod_cudatools.o"
"${PASCAL_MPIFC}" "${gpu_flags[@]}" -c "${SCRIPT_DIR}/legacy/mod_btdma_gpu.f90" \
    -module "${GPU_OBJ}" -o "${GPU_OBJ}/mod_btdma_gpu.o"
"${PASCAL_MPIFC}" "${gpu_flags[@]}" -c "${CORE_DIR}/mod_btdma_gpu_v2.f90" \
    -module "${GPU_OBJ}" -o "${GPU_OBJ}/mod_btdma_gpu_v2.o"
"${PASCAL_MPIFC}" "${gpu_flags[@]}" -c "${SCRIPT_DIR}/benchmark_support.f90" \
    -module "${GPU_OBJ}" -o "${GPU_OBJ}/benchmark_support.o"
"${PASCAL_MPIFC}" "${gpu_flags[@]}" -c "${SCRIPT_DIR}/gpu_benchmark_support.f90" \
    -module "${GPU_OBJ}" -o "${GPU_OBJ}/gpu_benchmark_support.o"

gpu_drivers=(
    bench_gpu_alltoall
    bench_gpu_pascal_unoptimized
    bench_gpu_pascal_optimized
)
for driver in "${gpu_drivers[@]}"; do
    "${PASCAL_MPIFC}" "${gpu_flags[@]}" -c "${SCRIPT_DIR}/${driver}.f90" \
        -module "${GPU_OBJ}" -o "${GPU_OBJ}/${driver}.o"
done

gpu_common=(
    "${GPU_OBJ}/benchmark_support.o"
    "${GPU_OBJ}/gpu_benchmark_support.o"
)
"${PASCAL_MPIFC}" "${GPU_OBJ}/mpiutil.o" "${GPU_OBJ}/mod_cudatools.o" \
    "${GPU_OBJ}/mod_btdma_gpu.o" "${gpu_common[@]}" \
    "${GPU_OBJ}/bench_gpu_alltoall.o" "${gpu_link_flags[@]}" \
    -o "${BIN_DIR}/bench_gpu_alltoall"
"${PASCAL_MPIFC}" "${GPU_OBJ}/mpiutil.o" "${GPU_OBJ}/mod_cudatools.o" \
    "${GPU_OBJ}/mod_btdma_gpu.o" "${gpu_common[@]}" \
    "${GPU_OBJ}/bench_gpu_pascal_unoptimized.o" "${gpu_link_flags[@]}" \
    -o "${BIN_DIR}/bench_gpu_pascal_unoptimized"
"${PASCAL_MPIFC}" "${GPU_OBJ}/mod_cudatools.o" "${gpu_common[@]}" \
    "${GPU_OBJ}/mod_btdma_gpu_v2.o" "${GPU_OBJ}/bench_gpu_pascal_optimized.o" \
    "${gpu_link_flags[@]}" -o "${BIN_DIR}/bench_gpu_pascal_optimized"

echo "[build] Created five benchmark executables in ${BIN_DIR}"
