#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/00_enviroment.sh"

required_commands=("${MPIFC}" nvfortran ar python3)
for command_name in "${required_commands[@]}"; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command not found: ${command_name}" >&2
        exit 1
    fi
done

mkdir -p \
    "${ROOT_DIR}/build/cpu" \
    "${ROOT_DIR}/build/gpu" \
    "${ROOT_DIR}/include/cpu" \
    "${ROOT_DIR}/include/gpu" \
    "${ROOT_DIR}/lib" \
    "${ROOT_DIR}/example/build"

echo "PaScaL_BTDMA build environment"
echo "  MPIFC=${MPIFC}"
echo "  GPU_ARCH=${GPU_ARCH}"
echo "  OPT=${OPT}"
"${MPIFC}" --version | sed -n '1p'

if type module >/dev/null 2>&1; then
    module list
fi
