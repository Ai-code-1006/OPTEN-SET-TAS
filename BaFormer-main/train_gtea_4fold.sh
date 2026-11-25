#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/srv/nfs/home/njnu_zya/anaconda3/envs/RTD/bin/python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/../data}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/gtea_q90}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/gtea_q90}"

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

for split in 1 2 3 4; do
    gpu=$((split - 1))
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT_DIR}" \
        "${PYTHON_BIN}" -u "${ROOT_DIR}/main.py" \
        --config "${ROOT_DIR}/configs/framed_en_de.yaml" \
        dataset.name gtea \
        dataset.split "${split}" \
        dataset.dataset_dir "${DATA_DIR}" \
        dataset.num_query 90 \
        scheduler.epochs 300 \
        train.output_dir "${OUTPUT_DIR}" \
        > "${LOG_DIR}/split${split}.log" 2>&1 &
    echo "split ${split}: pid $! gpu ${gpu}"
done

wait
