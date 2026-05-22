#!/usr/bin/env bash
# Pure IL baseline: extended features, no discard down-weighting, optional discard head.
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT="${1:-models/allyears_v4_no_discard_downweight.pt}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

EXTRA=()
if [[ -n "${MAX_EXAMPLES}" ]]; then
  EXTRA+=(--max-examples "${MAX_EXAMPLES}")
fi

mahjong-ai-train \
  --train-data data \
  --validation-data data/2025 \
  --output "${OUTPUT}" \
  --epochs 5 \
  --extended \
  --model-arch conv \
  --model-type policy-network \
  --action-type-weight-power 0 \
  --skip-bad-replays \
  --max-examples 250000 \
  "${EXTRA[@]}"

echo "Evaluate offline (imitation + bad-discard metrics):"
echo "  mahjong-ai-evaluate offline --data data/2025 --model ${OUTPUT} --max-examples 100000 > results/allyears_v4_offline.json"
