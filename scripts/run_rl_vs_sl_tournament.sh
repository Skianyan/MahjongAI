#!/usr/bin/env bash
# Fine-tune with REINFORCE (short run by default) then compare RL vs supervised checkpoints in RiichiEnv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RL_OUT="${RL_OUT:-models/rl_policy.pt}"
WARMSTART="${WARMSTART:-models/policy.pt}"
BASELINE="${BASELINE:-models/policy.pt}"
GAMES="${GAMES:-50}"
RL_EPOCHS="${RL_EPOCHS:-2}"
EP_PER_UPDATE="${EP_PER_UPDATE:-2}"

python3 -m mahjong_ai.training.rl_train \
  --warmstart "$WARMSTART" \
  --output "$RL_OUT" \
  --rl-epochs "$RL_EPOCHS" \
  --episodes-per-update "$EP_PER_UPDATE" \
  --device cpu

python3 -m mahjong_ai.evaluation.evaluate tournament \
  --model "$RL_OUT" \
  --baseline-model "$BASELINE" \
  --games "$GAMES" \
  > results/rl_vs_sl_tournament.json

echo "Wrote results/rl_vs_sl_tournament.json"
