#!/usr/bin/env bash
# v5 baseline: RIICHI upweighting (6×) + discard head + example weighting.
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT="${1:-models/allyears_v5.pt}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

EXTRA=()
if [[ -n "${MAX_EXAMPLES}" ]]; then
  EXTRA+=(--max-examples "${MAX_EXAMPLES}")
fi

./venv/bin/mahjong-ai-train \
  --train-data data \
  --validation-data data/2025 \
  --output "${OUTPUT}" \
  --epochs 5 \
  --extended \
  --model-arch conv \
  --model-type policy-network \
  --action-type-weight-power 0 \
  --riichi-weight-multiplier 6.0 \
  --train-discard-head \
  --example-weighting \
  --skip-bad-replays \
  --max-examples 250000 \
  "${EXTRA[@]}"

echo "Evaluate offline (imitation + bad-discard metrics):"
echo "  ./venv/bin/mahjong-ai-evaluate offline --data data/2025 --model ${OUTPUT} --max-examples 100000 > results/allyears_v5_offline.json"
