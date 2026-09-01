#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${CASE_DIR}/../.." && pwd)"
CASE_FILE="${CASE_DIR}/cases.tsv"
TARGET="${1:-all}"
source "${ROOT_DIR}/00_enviroment.sh"
NREF=1024

mkdir -p "${CASE_DIR}/bin" "${CASE_DIR}/build"
if [[ "${PASCAL_CORE_READY:-0}" != "1" ]]; then
    "${ROOT_DIR}/build_mango.sh" gpu
fi

built_names=" "
matched=0

while IFS=$'\t' read -r case_id workflow n np1 np2 np3 nsteps gpus mango_nodes executable; do
    [[ "${case_id}" == "case_id" ]] && continue

    if [[ "${TARGET}" != "all" && "${TARGET}" != "${workflow}" && "${TARGET}" != "${case_id}" ]]; then
        continue
    fi
    matched=1

    if [[ "${built_names}" == *" ${executable} "* ]]; then
        continue
    fi
    built_names+="${executable} "

    if (( np1*np2*np3 != gpus )); then
        echo "Invalid row ${case_id}: topology does not equal GPU count." >&2
        exit 2
    fi
    if (( n % np1 != 0 || n % np2 != 0 || n % np3 != 0 )); then
        echo "Invalid row ${case_id}: N is not divisible by the Cartesian topology." >&2
        exit 2
    fi
    if (( (n/np1) % 2 != 0 || (n/np2) % 2 != 0 || (n/np3) % 2 != 0 )); then
        echo "Invalid row ${case_id}: local line length must be even for m=2." >&2
        exit 2
    fi

    obj_dir="${CASE_DIR}/build/${executable}"
    mkdir -p "${obj_dir}"
    rm -f "${obj_dir}"/*.o "${obj_dir}"/*.mod

    flags=(
        -Mfree
        -cpp
        -cuda
        "-gpu=${GPU_ARCH}"
        "${OPT}"
        -DGPU_ENABLED
        "-DNX_VAL=${n}"
        "-DNY_VAL=${n}"
        "-DNZ_VAL=${n}"
        "-DNP1=${np1}"
        "-DNP2=${np2}"
        "-DNP3=${np3}"
        "-DNRUN=${nsteps}"
        "-DNREF_VAL=${NREF}"
    )

    echo "[build] ${case_id}: N=${n}, topology=${np1}x${np2}x${np3}, steps=${nsteps}"
    "${MPIFC}" "${flags[@]}" \
        -module "${obj_dir}" \
        -I"${obj_dir}" \
        -I"${ROOT_DIR}/include/gpu" \
        -c "${CASE_DIR}/heat3d_gpu.f90" \
        -o "${obj_dir}/heat3d_gpu.o"

    "${MPIFC}" -cuda "-gpu=${GPU_ARCH}" "${OPT}" \
        "${obj_dir}/heat3d_gpu.o" \
        "${ROOT_DIR}/lib/libpascal_btdma_gpu.a" \
        -o "${CASE_DIR}/bin/${executable}"
done < "${CASE_FILE}"

if (( matched == 0 )); then
    echo "Usage: $0 [all|convergence|scaling|case_id]" >&2
    exit 2
fi

echo "Built Heat3D target: ${TARGET}"
