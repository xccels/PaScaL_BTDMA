#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-default}"

case "${MODE}" in
    default|all) ;;
    *)
        echo "Usage: $0 [all]" >&2
        exit 2
        ;;
esac

source "${ROOT_DIR}/00_enviroment.sh"
"${ROOT_DIR}/01_setup.sh"

echo "[1/2] Building PaScaL_BTDMA libraries"
"${ROOT_DIR}/build_mango.sh" all
export PASCAL_CORE_READY=1

echo "[2/2] Building minimal CPU/GPU examples"
"${ROOT_DIR}/example/build_mango.sh"

if [[ "${MODE}" == "all" ]]; then
    echo "Building all reproducibility programs"
    reproducibility_cases=(
        01_solver_scaling
        02_communication_vs_N
        03_throughput_vs_nsys
        04_accuracy_conditioning
        05_phase_timing
        06_intra_inter_node
        07_compiler_resources
        08_euler3d
        09_weno2d
        10_heat3d
    )

    for case_name in "${reproducibility_cases[@]}"; do
        echo "[reproducibility/${case_name}]"
        "${ROOT_DIR}/reproducibility/${case_name}/build_mango.sh"
    done
fi

echo "Build completed: ${MODE}"
