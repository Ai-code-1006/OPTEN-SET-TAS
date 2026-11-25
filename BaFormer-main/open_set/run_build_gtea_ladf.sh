#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/srv/nfs/home/njnu_zya/anaconda3/envs/RTD/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/srv/nfs/data/TAS_dataset/gtea}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/open_set_data/gtea_ladf_v2}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/open_set/build_open_set_dataset.py" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset gtea \
    --all-percentages \
    "$@"
