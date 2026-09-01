#!/usr/bin/env bash
#SBATCH --job-name=btdma-ptxinfo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/${SLURM_JOB_ID}"

source "${ROOT_DIR}/setup_mango.sh"
"${SCRIPT_DIR}/build_mango.sh" "${OUTPUT_DIR}"

