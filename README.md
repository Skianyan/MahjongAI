# Mahjong AI

Riichi Mahjong AI training and bot built to work with `riichienv`, a local Riichi Mahjong simulator. The project can train supervised baseline policies from MJAI replays, evaluate checkpoints offline or in local games, fine-tune with REINFORCE, and play games with the RiichiLabs API.

For detailed CLI reference (every flag, bundled scripts, and workflow examples), see [COMMANDS.md](COMMANDS.md).

## Features

- Supervised policy training from MJAI replay data with streaming I/O (files, directories, ZIP archives).
- Two baseline model types:
  - `policy-network`: a PyTorch neural policy with legal-action masking.
  - `action-prior`: a lightweight frequency baseline that does not require PyTorch.
- Network architectures: `mlp` (default) or `conv` with optional **extended** riichienv observation features (`--extended`).
- **Discard specialist head** (`--train-discard-head`): trains a global policy plus a 34-type discard model, saved as `policy_with_discard_head` and routed at inference on discard-only turns.
- Training options: validation split, early stopping, action-type reweighting, RIICHI loss multiplier, and seat-score example weighting.
- Offline imitation metrics: top-1/top-k accuracy, illegal-action rate, per-action-type breakdown, and **discard diagnostics** (red-five rate, pair breaks, shanten regressions).
- Local `RiichiEnv` evaluation against random or fallback opponents, with per-seat rotation and optional fallback comparison.
- **Agent stats** tracking: primary vs fallback decisions, shanten/red-five safety redirects, and `--fail-on-fallback` for checkpoint validation.
- Tournament-style comparisons between candidate, fallback, and optional baseline models.
- **REINFORCE** fine-tuning in `RiichiEnv` (`mahjong-ai-rl-train`) with rank or score rewards, entropy regularization, and discard shaping bonuses.
- RL opponents: `random`, `fallback`, **Akochan** (default in config), or **Mortal** subprocesses.
- RiichiLabs websocket bot runner with checkpoint loading, safe fallback wrapping, and discard safety filters at serve time.
- **Ranked batch runner** (`scripts/run_ranked_games.py`) for playing N ranked games and logging JSONL summaries.
- Configurable paths, training options, model artifact location, and evaluation defaults through TOML.

## Requirements

- Python 3.11 or newer.
- `riichienv` (>= 0.4.8).
- `websockets`.
- Optional training dependencies: `numpy`, `torch`, and `tqdm` (`pip install -e '.[train]'`).

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

Install training extras if you want to train PyTorch policy networks:

```bash
pip install -e '.[train]'
```

Copy the environment template if you plan to run the RiichiLabs bot:

```bash
cp .env.example .env
```

Then fill in:

```bash
RIICHI_BOT_TOKEN=your_token_here
```

Optional environment overrides:

```bash
MAHJONG_AI_CONFIG=config/default.toml
RIICHI_BOT_ENDPOINT=wss://game.riichi.dev/ws/validate
```

The default configuration is in `config/default.toml`. You can override it with `--config` on CLI commands or by setting `MAHJONG_AI_CONFIG`.

## Data

By default, training and offline evaluation read replay data from:

```text
data
```

Supported replay inputs include files, directories, and ZIP archives containing:

- `.json`
- `.jsonl`
- `.mjson`
- `.mjai`
- `.gz`
- `.zip`

The training code streams examples from MJAI events, so large replay directories can be processed without extracting every archive manually.

## Quick Start

Train a policy network:

```bash
mahjong-ai-train --data data --output models/policy.pt --model-type policy-network
```

Evaluate offline and locally:

```bash
mahjong-ai-evaluate offline --data data --model models/policy.pt
mahjong-ai-evaluate local --model models/policy.pt --games 100
```

Run the RiichiLabs bot:

```bash
mahjong-ai-bot --model models/policy.pt
```

See [COMMANDS.md](COMMANDS.md) for the full command reference, bundled training scripts, and recommended workflows.

## Bundled Training Recipes

**v4 — discard-focused pure IL baseline** (extended conv net, no action-type down-weighting):

```bash
bash scripts/train_allyears_v4.sh models/allyears_v4.pt
```

**v5 — competitive baseline** (6× RIICHI weight, discard head, example weighting):

```bash
bash scripts/train_allyears_v5.sh models/allyears_v5.pt
```

Both scripts train on all years under `data/` with validation on `data/2025`, use `--model-arch conv --extended`, and cap at 250k examples by default. Override the example cap with `MAX_EXAMPLES=100000 bash scripts/train_allyears_v5.sh`.

Each training run writes the checkpoint to the selected output path and a metrics sidecar file with the suffix `.metrics.json`.

Optional training flags (also available in the bundled scripts):

- `--action-type-weight-power 0` — keeps discard examples at full CE weight (default in `config/default.toml` for v4)
- `--example-weighting` — up-weight decisions from seats that finished the replay with higher scores
- `--train-discard-head` — trains a global policy plus a 34-type discard specialist (`policy_with_discard_head` checkpoint)

Offline evaluation reports bad-discard diagnostics (red-five rate, pair breaks, shanten regressions) under `discard_metrics`. Ranked runs log per-game `agent_stats` (fallback rate). Use `--fail-on-fallback` on local eval to catch checkpoint/vocabulary mismatches.

## Reinforcement Learning

After training a supervised `policy_network` checkpoint, warm-start **REINFORCE** rollouts in `RiichiEnv` (terminal reward from rank or final score). Hyper-parameters live under `[rl_training]` in `config/default.toml`.

```bash
mahjong-ai-rl-train --warmstart models/policy.pt --output models/rl_policy.pt
```

Compare the RL checkpoint against the supervised baseline:

```bash
mahjong-ai-evaluate tournament \
  --model models/rl_policy.pt \
  --baseline-model models/policy.pt \
  --games 500
```

Or use the bundled script (writes `results/rl_vs_sl_tournament.json`):

```bash
bash scripts/run_rl_vs_sl_tournament.sh
```

**Strong opponents:** configure `opponent = "akochan"` (default in `config/default.toml`) or `opponent = "mortal"` in config or via CLI flags. Both require a riichienv build where `Observation.new_events()` returns incremental MJAI lines per seat.

```bash
mahjong-ai-rl-train \
  --warmstart models/policy.pt \
  --output models/rl_akochan.pt \
  --opponent akochan \
  --akochan-dir RLModels/akochan
```

```bash
mahjong-ai-rl-train \
  --warmstart models/policy.pt \
  --output models/rl_mortal.pt \
  --opponent mortal \
  --mortal-binary docker \
  --mortal-model-dir /path/to/mortal/model
```

## Summarize Results

The repository includes a helper for paired offline/local result files:

```bash
python scripts/summarize_results.py --results-dir results
```

It prints a compact comparison table using `fallback_local.json` as the reference.

## Run Ranked Games

Play multiple ranked games on RiichiLabs and log per-game summaries (scores, duration, agent stats):

```bash
set -a && source .env && set +a
python scripts/run_ranked_games.py --games 10 \
  --endpoint wss://game.riichi.dev/ws/ranked \
  --model models/policy.pt \
  --output results/ranked_runs.jsonl
```

## Configuration

The default TOML config controls:

- Bot endpoint, token environment variable, and optional `RIICHI_BOT_ENDPOINT` override.
- Raw and processed data directories.
- Model artifact path.
- Training batch size, epochs, learning rate, hidden size, architecture, validation split, seed, workers, early stopping, and action-type weight power.
- Evaluation game count and action timeout.
- RL training: warmstart path, episodes per update, reward mode, opponent type, shaping bonuses, and Mortal/Akochan paths.

Use a custom config file with:

```bash
mahjong-ai-train --config path/to/config.toml
mahjong-ai-evaluate --config path/to/config.toml local
mahjong-ai-bot --config path/to/config.toml
```

## Project Layout

```text
bot.py                           Local bot entry-point wrapper
config/default.toml              Default runtime configuration
COMMANDS.md                      Detailed CLI and workflow reference
scripts/
  summarize_results.py           Offline/local result comparison table
  run_ranked_games.py            Batch ranked games on RiichiLabs websocket
  train_allyears_v4.sh           v4 supervised training recipe
  train_allyears_v5.sh           v5 supervised IL recipe (discard head + RIICHI weight)
  run_rl_vs_sl_tournament.sh     Short RL fine-tune + tournament comparison
src/mahjong_ai/
  agents/                        Agent protocols, checkpoint loading, Mortal/Akochan opponents
  bot_runner.py                  RiichiLabs websocket bot runner
  data.py                        MJAI replay streaming and example extraction
  evaluation/                    Offline, local, tournament evaluation + discard metrics
  features/                      Observation/action encoding
  training/                      Supervised + REINFORCE RL training entry points and models
tests/                           Smoke and unit tests
```
