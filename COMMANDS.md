# Command Reference

This document describes every CLI entry point, helper script, and common workflow in MahjongAI. For project overview and setup, see [README.md](README.md).

All commands accept `--config PATH` to load a TOML file (default: `MAHJONG_AI_CONFIG` or `config/default.toml`).

Installed console scripts (from `pyproject.toml`):

| Command | Module |
|---------|--------|
| `mahjong-ai-train` | Supervised baseline training |
| `mahjong-ai-rl-train` | REINFORCE fine-tuning |
| `mahjong-ai-evaluate` | Offline, local, and tournament evaluation |
| `mahjong-ai-bot` | RiichiLabs websocket bot (single game session) |

---

## Configuration

### Environment variables

| Variable | Purpose |
|----------|---------|
| `RIICHI_BOT_TOKEN` | Bearer token for RiichiLabs websocket (required for bot and ranked runner) |
| `RIICHI_BOT_ENDPOINT` | Override websocket URL (default from `[bot].endpoint` in config) |
| `MAHJONG_AI_CONFIG` | Path to TOML config file |

### Config sections (`config/default.toml`)

| Section | Controls |
|---------|----------|
| `[bot]` | Websocket endpoint and token env var name |
| `[data]` | `raw_dir`, `processed_dir` |
| `[model]` | Default checkpoint path (`artifact_path`) |
| `[training]` | Batch size, epochs, LR, architecture, validation, early stopping |
| `[evaluation]` | Default game count and action timeout |
| `[rl_training]` | REINFORCE hyper-parameters, opponent type, Mortal/Akochan paths, shaping bonuses |

---

## `mahjong-ai-train`

Train a supervised baseline from MJAI replays.

### Model types

| `--model-type` | Output | Requires PyTorch |
|----------------|--------|------------------|
| `auto` (default) | `policy-network` if torch installed, else `action-prior` | — |
| `policy-network` | `.pt` checkpoint with neural weights | Yes |
| `action-prior` | `.json` frequency baseline | No |

### Architectures (`--model-arch`, policy-network only)

| Value | Description |
|-------|-------------|
| `mlp` | Multi-layer perceptron over flattened observation |
| `conv` | Convolutional network (used by allyears v4/v5 recipes) |

Use `--extended` to enable extended riichienv observation features. The flag is stored in the checkpoint `feature_schema.extended` and must match at evaluation/serve time.

### Discard head (`--train-discard-head`)

Trains two models and merges them into one checkpoint with `model_type: policy_with_discard_head`:

1. **Global policy** — standard action vocabulary policy network.
2. **Discard specialist** — 34-type tile classifier used only on discard-only turns (no calls, riichi, or win available).

At inference, `CompositePolicyAgent` routes discard-only turns to the specialist and other turns to the global policy.

### All flags

```
--config PATH                  TOML config file
--data PATH                    Replay root (default: [data].raw_dir)
--train-data PATH              Explicit training replay path (overrides --data)
--validation-data PATH         Explicit validation replay path (disables ratio split)
--validation-ratio FLOAT       Hold-out fraction when no --validation-data (default: 0.1)
--output PATH                  Checkpoint output path (default: [model].artifact_path)
--batch-size INT               Default: 512
--epochs INT                   Default: 5
--device STR                   auto | cpu | cuda | mps (default: auto)
--learning-rate FLOAT          Default: 0.001
--hidden-size INT              Default: 512
--seed INT                     Default: 7
--num-workers INT              DataLoader workers (default: 0)
--early-stopping-patience INT  Stop after N epochs without val improvement (default: 3)
--model-type {auto,policy-network,action-prior}
--model-arch {mlp,conv}        Default: mlp
--max-examples INT             Cap examples per epoch / vocab scan
--action-type NAME             Filter to action types (repeatable, e.g. DISCARD)
--extended                     Use extended observation encoding
--skip-bad-replays             Skip files riichienv cannot parse (sets strict=False)
--action-type-weight-power F   Reweight CE loss: (1/freq)**power per action type; 0 disables
--riichi-weight-multiplier F   Extra loss multiplier for RIICHI actions (default: 1.0)
--example-weighting            Up-weight examples from seats with higher final replay scores
--train-discard-head           Train global + discard specialist combined checkpoint
```

### Outputs

- **Checkpoint** at `--output` (`.pt` for policy-network, `.json` for action-prior).
- **Metrics sidecar** at `<output>.metrics.json` with per-epoch loss history.

Policy-network checkpoints include: `action_vocabulary`, `model_config`, `state_dict`, `feature_schema`, `training` metadata, and `selection` (best epoch/metric).

### Examples

Basic policy network:

```bash
mahjong-ai-train \
  --data data \
  --output models/policy.pt \
  --model-type policy-network \
  --epochs 10 \
  --batch-size 512
```

Explicit train/validation split by directory:

```bash
mahjong-ai-train \
  --train-data data \
  --validation-data data/2025 \
  --output models/allyears.pt \
  --extended \
  --model-arch conv \
  --skip-bad-replays \
  --max-examples 250000
```

Action-prior baseline (no PyTorch):

```bash
mahjong-ai-train --data data --output models/action_prior.json --model-type action-prior
```

Combined global + discard head with RIICHI emphasis (v5 recipe):

```bash
mahjong-ai-train \
  --train-data data \
  --validation-data data/2025 \
  --output models/allyears_v5.pt \
  --extended --model-arch conv \
  --action-type-weight-power 0 \
  --riichi-weight-multiplier 6.0 \
  --train-discard-head \
  --example-weighting \
  --skip-bad-replays \
  --max-examples 250000
```

Or use the bundled script:

```bash
bash scripts/train_allyears_v5.sh models/allyears_v5.pt
MAX_EXAMPLES=100000 bash scripts/train_allyears_v4.sh models/allyears_v4.pt
```

---

## `mahjong-ai-rl-train`

Fine-tune a supervised `policy_network` checkpoint with REINFORCE in `RiichiEnv`.

### Algorithm

- Warm-starts from a supervised PyTorch checkpoint (`policy_network` or legacy `mlp_policy`).
- Collects `episodes_per_update` full games per RL epoch.
- Terminal reward: **rank** (1st=+1, 2nd=+0.5, 3rd=-0.5, 4th=-1) or **score** (normalized final points).
- Optional **shaping rewards** on discard/riichi steps: shanten reduction, tenpai, riichi declaration.
- Moving-average baseline, entropy regularization, AdamW optimizer.
- Writes a new `policy_network` checkpoint (compatible with evaluation and bot serving).

### Opponents (`--opponent`)

| Value | Description |
|-------|-------------|
| `random` | riichienv `RandomAgent` for seats 1–3 |
| `fallback` | Project `FallbackAgent` for all opponent seats |
| `akochan` | Akochan MJAI subprocess per seat (default in config) |
| `mortal` | Mortal MJAI subprocess per seat (native binary or Docker) |

Akochan and Mortal require riichienv with incremental `Observation.new_events()`. Configure paths under `[rl_training]` or via CLI flags.

### All flags

```
--config PATH
--warmstart PATH               Supervised checkpoint (default: [rl_training].warmstart_model)
--output PATH                  RL checkpoint output (default: models/rl_policy.pt)
--device STR
--learning-rate FLOAT          Default: 0.0001
--episodes-per-update INT      Games collected per RL epoch (default: 16)
--rl-epochs INT                Default: 200
--gamma FLOAT                  Discount factor (default: 1.0)
--entropy-coef FLOAT           Entropy bonus (default: 0.01)
--baseline-momentum FLOAT      Reward baseline EMA (default: 0.99)
--reward-mode {rank,score}
--opponent {random,fallback,mortal,akochan}
--game-mode STR                RiichiEnv mode (default: 4p-red-single)
--max-actions-per-game INT     Truncate stuck games (default: 5000)
--controlled-seat INT          Seat trained by policy (default: 0)
--seed INT
--shanten-bonus FLOAT          Shaping per shanten step reduced (default: 0.05)
--tenpai-bonus FLOAT           Shaping for reaching tenpai (default: 0.10)
--riichi-bonus FLOAT           Shaping for declaring riichi (default: 0.10)

# Mortal opponent
--mortal-binary STR            Executable path or "docker"
--mortal-model-dir PATH        Host dir mounted as /mnt when using docker
--mortal-docker-image STR      Default: mortal:latest
--mortal-timeout FLOAT         Subprocess timeout seconds (default: 30)

# Akochan opponent
--akochan-dir STR              Directory with system.exe, libai.so, tactics.json
--akochan-tactics PATH         tactics.json path
--akochan-timeout FLOAT        Default: 30
```

### Examples

Basic CPU fine-tune against random opponents:

```bash
mahjong-ai-rl-train \
  --warmstart models/policy.pt \
  --output models/rl_policy.pt \
  --opponent random \
  --rl-epochs 50 \
  --device cpu
```

Against Akochan (default config):

```bash
mahjong-ai-rl-train \
  --warmstart models/allyears_v5.pt \
  --output models/rl_akochan.pt \
  --opponent akochan \
  --akochan-dir RLModels/akochan
```

Against Mortal via Docker:

```bash
mahjong-ai-rl-train \
  --warmstart models/policy.pt \
  --output models/rl_mortal.pt \
  --opponent mortal \
  --mortal-binary docker \
  --mortal-model-dir /path/to/mortal/model
```

Short comparison run (bundled script):

```bash
WARMSTART=models/policy.pt RL_EPOCHS=2 GAMES=50 bash scripts/run_rl_vs_sl_tournament.sh
```

Environment variables for the script: `RL_OUT`, `WARMSTART`, `BASELINE`, `GAMES`, `RL_EPOCHS`, `EP_PER_UPDATE`.

---

## `mahjong-ai-evaluate`

Three subcommands. All print JSON to stdout.

### `offline`

Measure imitation quality against MJAI replay expert actions.

**Metrics (overall and per action type):**

| Field | Meaning |
|-------|---------|
| `top1_accuracy` | Masked top-1 matches expert action |
| `topk_accuracy` | Expert action in masked top-k (default k=5) |
| `raw_top1_illegal_rate` | Unmasked argmax was illegal |
| `unknown_legal_action_rate` | Legal actions missing from checkpoint vocabulary |

**Discard metrics** (`discard_metrics`, when discard examples exist):

| Field | Meaning |
|-------|---------|
| `discard_match_rate` | Model discard matches expert |
| `model_red_five_discard_rate` | Model discards red fives |
| `model_pair_break_rate` | Model breaks pairs in hand |
| `model_shanten_regression_rate` | Model discard increases shanten |
| `expert_shanten_regression_rate` | Expert shanten regression rate (reference) |

```
mahjong-ai-evaluate offline [OPTIONS]

--data PATH
--model PATH
--device STR
--top-k INT                    Default: 5
--max-examples INT
--action-type NAME             Filter examples (repeatable)
--skip-bad-replays
```

```bash
mahjong-ai-evaluate offline \
  --data data/2025 \
  --model models/allyears_v5.pt \
  --max-examples 100000 \
  > results/allyears_v5_offline.json
```

### `local`

Play games in `RiichiEnv`. The controlled seat uses the loaded checkpoint wrapped in `SafeAgent` (falls back on inference errors). Opponent seats use random or fallback agents.

**Result fields include:** `mean_score`, `mean_rank`, `first_place_rate`, `tie_rate`, `unfinished_game_rate`, action timing, `seat_metrics`, `agent_stats`, and optionally `comparison` vs fallback.

```
mahjong-ai-evaluate local [OPTIONS]

--model PATH                   Checkpoint path
--fallback-only                Run deterministic fallback instead of a model
--games INT                    Default: 100
--timeout-seconds FLOAT        Flag slow decisions (default: 3.0)
--device STR
--game-mode STR                Default: 4p-red-single
--controlled-seat INT          Fixed seat (default: 0)
--all-seats                    Rotate controlled seat each game (0→1→2→3)
--opponent {random,fallback}   Default: random
--max-actions-per-game INT     Default: 5000
--compare-fallback             Also run fallback-only and include deltas
--fail-on-fallback             Exit with error if any decision used fallback
```

```bash
# Model vs random, seat 0
mahjong-ai-evaluate local --model models/policy.pt --games 200 > results/policy_local.json

# Validate checkpoint loads cleanly (no fallback decisions)
mahjong-ai-evaluate local --model models/policy.pt --games 50 --fail-on-fallback

# Compare model vs fallback baseline
mahjong-ai-evaluate local \
  --model models/policy.pt \
  --games 100 \
  --compare-fallback
```

**Agent stats** (`agent_stats` in output):

| Field | Meaning |
|-------|---------|
| `primary_decisions` | Actions chosen by the loaded policy |
| `fallback_decisions` | Heuristic / unknown-legal / exception fallbacks |
| `fallback_rate` | Fraction of decisions that fell back |
| `shanten_filter_redirects` | Discard redirected to avoid shanten regression |
| `red_five_filter_redirects` | Discard redirected away from red five |

### `tournament`

Run multiple local evaluation variants and return them in one JSON object:

| Variant key | Description |
|-------------|-------------|
| `candidate_vs_random` | `--model` vs random opponents, rotating seats |
| `fallback_vs_random` | Fallback-only vs random |
| `baseline_model_vs_random` | Optional `--baseline-model` vs random |

```
mahjong-ai-evaluate tournament [OPTIONS]

--model PATH
--baseline-model PATH          Optional second checkpoint
--games INT
--timeout-seconds FLOAT
--device STR
--game-mode STR
--max-actions-per-game INT
```

```bash
mahjong-ai-evaluate tournament \
  --model models/rl_policy.pt \
  --baseline-model models/policy.pt \
  --games 500 \
  > results/rl_vs_sl_tournament.json
```

---

## `mahjong-ai-bot`

Connect to RiichiLabs websocket, respond to `request_action` messages, exit on `end_game`.

The serving stack:

1. `build_serving_agent` loads the checkpoint (or fallback-only mode).
2. `SafeAgent` wraps the policy; inference exceptions trigger conservative fallback.
3. At act time, discard safety filters may redirect shanten-regressing or red-five discards.

```
mahjong-ai-bot [OPTIONS]

--config PATH
--endpoint URL                 Override config / RIICHI_BOT_ENDPOINT
--model PATH                   Checkpoint (default: [model].artifact_path)
--device STR                   Inference device (default: [training].device)
--fallback-only                Skip checkpoint; use FallbackAgent only
```

```bash
# Validation endpoint (default)
set -a && source .env && set +a
mahjong-ai-bot --model models/policy.pt

# Explicit endpoint override
mahjong-ai-bot \
  --endpoint wss://game.riichi.dev/ws/validate \
  --model models/allyears_v5.pt \
  --device cpu
```

Equivalent local entry point:

```bash
python bot.py --model models/policy.pt
```

---

## `scripts/run_ranked_games.py`

Play **N ranked games** on RiichiLabs and append one JSON object per game to a JSONL file. Uses the same agent logic as `mahjong-ai-bot`.

Each record includes: `game_index`, `duration_s`, `actions`, `end_game` payload, and per-game `agent_stats` (delta when reconnecting each game).

```
python scripts/run_ranked_games.py [OPTIONS]

--games INT                    Required: number of games
--output PATH                  JSONL output (default: results/ranked_runs.jsonl)
--endpoint URL                 Default: RIICHI_BOT_ENDPOINT or config
--config PATH
--model PATH
--device STR
--fallback-only
--reuse-connection             Keep one websocket open for N end_game events
--verbose                      Print progress to stderr
--log-all                      Log every WS message (requires --debug-output)
--debug-output PATH            JSONL debug log
```

```bash
set -a && source .env && set +a

# 10 ranked games with a checkpoint
python scripts/run_ranked_games.py \
  --games 10 \
  --endpoint wss://game.riichi.dev/ws/ranked \
  --model models/allyears_v5.pt \
  --device cpu \
  --output results/allyears_v5_ranked.jsonl \
  --verbose

# Fallback baseline on ranked
python scripts/run_ranked_games.py \
  --games 5 \
  --fallback-only \
  --output results/ranked_fb.jsonl
```

---

## `scripts/summarize_results.py`

Build a comparison table from paired result files in a directory.

**Expected naming:** `<experiment>_offline.json` + `<experiment>_local.json`, plus `fallback_local.json` as reference.

```
python scripts/summarize_results.py --results-dir results
```

Columns include offline top-1/top-k/illegal rates, local score/rank/first-place, and deltas vs fallback. Rows are sorted by best `d_rank`, then `d_score`, then `d_fp`.

---

## `scripts/run_rl_vs_sl_tournament.sh`

End-to-end RL smoke test:

1. Runs `mahjong-ai-rl-train` with short defaults (`RL_EPOCHS=2`, `EP_PER_UPDATE=2`).
2. Runs `mahjong-ai-evaluate tournament` comparing RL vs supervised baseline.
3. Writes `results/rl_vs_sl_tournament.json`.

Override via environment: `RL_OUT`, `WARMSTART`, `BASELINE`, `GAMES`, `RL_EPOCHS`, `EP_PER_UPDATE`.

---

## Recommended Workflows

### Train → offline eval → local eval

```bash
mahjong-ai-train \
  --train-data data \
  --validation-data data/2025 \
  --output models/experiment.pt \
  --extended --model-arch conv \
  --skip-bad-replays

mahjong-ai-evaluate offline \
  --data data/2025 \
  --model models/experiment.pt \
  --max-examples 100000 \
  > results/experiment_offline.json

mahjong-ai-evaluate local \
  --model models/experiment.pt \
  --games 200 \
  --compare-fallback \
  > results/experiment_local.json

python scripts/summarize_results.py --results-dir results
```

Ensure `results/fallback_local.json` exists (run `mahjong-ai-evaluate local --fallback-only --games 200` once).

### Checkpoint validation before ranked play

```bash
mahjong-ai-evaluate local \
  --model models/allyears_v5.pt \
  --games 50 \
  --fail-on-fallback

set -a && source .env && set +a
python scripts/run_ranked_games.py \
  --games 10 \
  --endpoint wss://game.riichi.dev/ws/ranked \
  --model models/allyears_v5.pt \
  --output results/allyears_v5_ranked.jsonl
```

### Supervised → RL → tournament

```bash
mahjong-ai-train --output models/sl.pt --extended --model-arch conv --train-data data
mahjong-ai-rl-train --warmstart models/sl.pt --output models/rl.pt --opponent akochan
mahjong-ai-evaluate tournament \
  --model models/rl.pt \
  --baseline-model models/sl.pt \
  --games 500
```

---

## Checkpoint Types

| `model_type` | Loaded by | Notes |
|--------------|-----------|-------|
| `action_prior` | `ActionPriorAgent` | JSON, no PyTorch |
| `policy_network` | `MLPPolicyAgent` | Standard neural policy |
| `policy_with_discard_head` | `CompositePolicyAgent` | Global + discard specialist |
| `discard_policy_network` | `DiscardPolicyAgent` | Standalone discard head only |
| `mlp_policy` | `MLPPolicyAgent` | Legacy alias |

All PyTorch checkpoints should use the same `--extended` setting at train and serve time. Vocabulary mismatches surface as elevated `fallback_rate` in `agent_stats`.

---

## Tests

```bash
pip install -e '.[train]'
python -m pytest tests/
```

Smoke tests cover fallback counting and basic import/runtime paths.
