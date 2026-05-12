#!/bin/bash
# RiskPilot — 一键运行完整 Pipeline
# Usage: bash scripts/run_pipeline.sh [dataset] [mode]

DATASET=${1:-tfinance}
MODE=${2:-full}

echo "============================================"
echo "  RiskPilot Pipeline"
echo "  Dataset: $DATASET"
echo "  Mode:    $MODE"
echo "============================================"

cd "$(dirname "$0")/.."

python -m src.pipeline \
    --config configs/default.yaml \
    --dataset $DATASET \
    --mode $MODE
