#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%dT%H%M%S)}}"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/output/${RUN_ID}}"
SOURCE_DIR="${ROOT_DIR}/lib_btdma"
BUILD_DIR="${SCRIPT_DIR}/build/${RUN_ID}"
source "${ROOT_DIR}/00_enviroment.sh"
THREADS_PER_BLOCK="${THREADS_PER_BLOCK:-64}"

VARIANT_DIR="${BUILD_DIR}/variants"
COMMON_DIR="${BUILD_DIR}/common"
RAW_DIR="${OUTPUT_DIR}/raw"
VARIANT_MANIFEST="${OUTPUT_DIR}/variant_manifest.tsv"
COMPILE_MANIFEST="${OUTPUT_DIR}/compile_manifest.tsv"

mkdir -p "${BUILD_DIR}" "${COMMON_DIR}" "${RAW_DIR}" "${OUTPUT_DIR}"

if ! command -v "${MPIFC}" >/dev/null 2>&1; then
    echo "Compiler wrapper not found: ${MPIFC}" >&2
    echo "Check ${ROOT_DIR}/00_enviroment.sh" >&2
    exit 127
fi
if ! command -v nvfortran >/dev/null 2>&1; then
    echo "nvfortran is not available in the current environment." >&2
    echo "Check ${ROOT_DIR}/00_enviroment.sh" >&2
    exit 127
fi

python3 "${SCRIPT_DIR}/generate_variants.py" \
    --source "${SOURCE_DIR}/mod_btdma_gpu_v2.f90" \
    --output-dir "${VARIANT_DIR}" \
    --manifest "${VARIANT_MANIFEST}"

shell_quote_command() {
    local output_file="$1"
    shift
    {
        printf '%q ' "$@"
        printf '\n'
    } > "${output_file}"
}

BASE_FLAGS=(-Mfree -Mextend -cpp -cuda "-gpu=${GPU_ARCH}" "${OPT}" -DGPU_ENABLED)
PTX_FLAGS=(-Mfree -Mextend -cpp -cuda "-gpu=${GPU_ARCH},ptxinfo" "${OPT}" -DGPU_ENABLED)

COMMON_MOD_DIR="${COMMON_DIR}/mod"
COMMON_OBJ_DIR="${COMMON_DIR}/obj"
mkdir -p "${COMMON_MOD_DIR}" "${COMMON_OBJ_DIR}"

dependency_command=(
    "${MPIFC}" "${BASE_FLAGS[@]}"
    -module "${COMMON_MOD_DIR}" -I"${COMMON_MOD_DIR}"
    -c "${SOURCE_DIR}/mod_cudatools.f90"
    -o "${COMMON_OBJ_DIR}/mod_cudatools.o"
)
shell_quote_command "${OUTPUT_DIR}/dependency_command.txt" "${dependency_command[@]}"

{
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date -Iseconds)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'repository_commit=%s\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || printf unavailable)"
    printf 'MPIFC=%s\nGPU_ARCH=%s\nOPT=%s\nTHREADS_PER_BLOCK=%s\n' \
        "${MPIFC}" "${GPU_ARCH}" "${OPT}" "${THREADS_PER_BLOCK}"
    printf '\n[command paths]\n'
    command -v "${MPIFC}"
    command -v nvfortran
    printf '\n[mpifort version]\n'
    "${MPIFC}" --version
    printf '\n[nvfortran version]\n'
    nvfortran --version
    printf '\n[module list]\n'
    module list 2>&1 || true
    printf '\n[source and script hashes]\n'
    sha256sum \
        "${SOURCE_DIR}/mod_btdma_gpu_v2.f90" \
        "${SOURCE_DIR}/mod_cudatools.f90" \
        "${SCRIPT_DIR}/generate_variants.py" \
        "${SCRIPT_DIR}/parse_ptxinfo.py" \
        "${SCRIPT_DIR}/build_mango.sh" \
        "${SCRIPT_DIR}/run_compile_mango.slurm"
} > "${OUTPUT_DIR}/environment.txt"

"${dependency_command[@]}" > "${OUTPUT_DIR}/dependency_compile.log" 2>&1

printf 'build_id\tm\tcompile_status\tlog_path\tcommand_path\tgenerated_source\tgenerated_source_sha256\n' \
    > "${COMPILE_MANIFEST}"

had_failure=0
for m in 8 9 10 12; do
    build_id="W${m}"
    case_dir="${BUILD_DIR}/${build_id}"
    case_mod_dir="${case_dir}/mod"
    case_obj_dir="${case_dir}/obj"
    raw_case_dir="${RAW_DIR}/${build_id}"
    source_file="${VARIANT_DIR}/${build_id}/mod_btdma_gpu_v2.f90"
    log_file="${raw_case_dir}/ptxinfo.log"
    command_file="${raw_case_dir}/command.txt"

    mkdir -p "${case_mod_dir}" "${case_obj_dir}" "${raw_case_dir}"
    compile_command=(
        "${MPIFC}" "${PTX_FLAGS[@]}"
        -module "${case_mod_dir}" -I"${case_mod_dir}" -I"${COMMON_MOD_DIR}"
        -c "${source_file}"
        -o "${case_obj_dir}/mod_btdma_gpu_v2.o"
    )
    shell_quote_command "${command_file}" "${compile_command[@]}"

    echo "[compile-only] ${build_id}: workspace extent ${m}"
    set +e
    "${compile_command[@]}" > "${log_file}" 2>&1
    status=$?
    set -e
    cat "${log_file}"

    if (( status != 0 )); then
        had_failure=1
    fi
    generated_hash="$(sha256sum "${source_file}" | awk '{print $1}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${build_id}" "${m}" "${status}" \
        "raw/${build_id}/ptxinfo.log" "raw/${build_id}/command.txt" \
        "${source_file}" "${generated_hash}" >> "${COMPILE_MANIFEST}"
done

python3 "${SCRIPT_DIR}/parse_ptxinfo.py" "${COMPILE_MANIFEST}" \
    --source "${SOURCE_DIR}/mod_btdma_gpu_v2.f90" \
    --threads-per-block "${THREADS_PER_BLOCK}" \
    --csv "${OUTPUT_DIR}/resource_table.csv" \
    --markdown "${OUTPUT_DIR}/resource_table.md"

echo "Compile-only audit written to ${OUTPUT_DIR}"
if (( had_failure != 0 )); then
    echo "One or more variants did not compile; status and raw logs were retained." >&2
    exit 1
fi
