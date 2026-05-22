# Mahjong AI

Riichi Mahjong AI training and bot built to work with `riichienv`, a local Riichi Mahjong simulator. The project can train supervised baseline policies from MJAI replays, evaluate checkpoints offline or in local games, and play games with the RiichiLabs API.

## Features

- Supervised policy training from MJAI replay data.
- Two baseline model types:
  - `policy-network`: a PyTorch neural policy with legal-action masking.
  - `action-prior`: a lightweight frequency baseline that does not require PyTorch.
- Offline imitation metrics, including top-1/top-k accuracy and illegal-action rate.
- Local `RiichiEnv` evaluation against random or deterministic fallback opponents.
- Tournament-style comparisons between candidate, fallback, and optional baseline models.
- Optional **REINFORCE** fine-tuning in `RiichiEnv` (`mahjong-ai-rl-train`) to compare against supervised-only checkpoints.
- RiichiLabs websocket bot runner with checkpoint loading and deterministic fallback behavior.
- Configurable paths, training options, model artifact location, and evaluation defaults through TOML.

## Requirements

- Python 3.11 or newer.
- `riichienv`.
- `websockets`.
- Optional training dependencies: `numpy`, `torch`, and `tqdm`.

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

The default configuration is in `config/default.toml`. You can override it with `--config` on CLI commands or by setting:

```bash
export MAHJONG_AI_CONFIG=config/default.toml
```

## Data

By default, training and offline evaluation read replay data from:

```text
data/raw
```

Supported replay inputs include files, directories, and ZIP archives containing:

- `.json`
- `.jsonl`
- `.mjson`
- `.mjai`
- `.gz`
- `.zip`

The training code streams examples from MJAI events, so large replay directories can be processed without extracting every archive manually.

## Train a Model

Train using defaults from `config/default.toml`:

```bash
mahjong-ai-train
```

Train a PyTorch policy network from a specific replay directory:

```bash
mahjong-ai-train --data data/raw --output models/policy.pt --model-type policy-network
```

Train the lightweight action-prior baseline:

```bash
mahjong-ai-train --data data/raw --output models/action_prior.json --model-type action-prior
```

Useful training options:

```bash
mahjong-ai-train \
  --data data/raw \
  --validation-ratio 0.1 \
  --epochs 10 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --model-arch mlp \
  --max-examples 100000 \
  --skip-bad-replays
```

Each training run writes the checkpoint to the selected output path and a metrics sidecar file with the suffix `.metrics.json`.

### Reinforcement learning (REINFORCE)

After training a supervised `policy_network` checkpoint, you can warm-start **REINFORCE** rollouts in `RiichiEnv` (terminal reward from rank or final score). Hyper-parameters live under `[rl_training]` in `config/default.toml`.

```bash
mahjong-ai-rl-train --warmstart models/policy.pt --output models/rl_policy.pt
```

Compare the RL checkpoint against the supervised baseline in a local tournament:

```bash
mahjong-ai-evaluate tournament \
  --model models/rl_policy.pt \
  --baseline-model models/policy.pt \
  --games 500
```

Or run the bundled script (writes `results/rl_vs_sl_tournament.json`):

```bash
bash scripts/run_rl_vs_sl_tournament.sh
```

**Mortal opponent (optional):** with Mortal installed (see [Mortal docs](https://mortal.ekyu.moe/)), you can train REINFORCE against three Mortal subprocesses instead of `random`. Requires a riichienv build where `Observation.new_events()` returns incremental MJAI lines per seat.

```bash
mahjong-ai-rl-train \
  --warmstart models/policy.pt \
  --output models/rl_mortal.pt \
  --opponent mortal \
  --mortal-binary docker \
  --mortal-model-dir /path/to/mortal/model
```

Use a native Mortal binary with `--mortal-binary /path/to/mortal` (arguments are `mortal <player_id>` as in the upstream CLI). When `mortal_binary` is `docker`, the image defaults to `mortal:latest` (override with `--mortal-docker-image`).

## Evaluate a Model

Run offline imitation evaluation against replay examples:

```bash
mahjong-ai-evaluate offline --data data/raw --model models/policy.pt
```

Run local games in `RiichiEnv`:

```bash
mahjong-ai-evaluate local --model models/policy.pt --games 100
```

Evaluate the deterministic fallback agent without loading a model:

```bash
mahjong-ai-evaluate local --fallback-only --games 100
```

Compare a model against the fallback baseline during local evaluation:

```bash
mahjong-ai-evaluate local --model models/policy.pt --games 100 --compare-fallback
```

Run a small tournament-style comparison:

```bash
mahjong-ai-evaluate tournament --model models/policy.pt --games 100
```

Evaluation commands print JSON to stdout, so results can be saved:

```bash
mahjong-ai-evaluate local --model models/policy.pt --games 100 > results/eval_local.json
```

## Summarize Results

The repository includes a helper script for result files named like `*_offline.json`, `*_local.json`, and `fallback_local.json`:

```bash
python scripts/summarize_results.py --results-dir results
```

It prints a compact comparison table using fallback local performance as the reference.

## Run the Bot

After setting `RIICHI_BOT_TOKEN`, run the websocket bot with the configured endpoint and model:

```bash
mahjong-ai-bot --model models/policy.pt
```

Run with the deterministic fallback only:

```bash
mahjong-ai-bot --fallback-only
```

Override the endpoint if needed:

```bash
mahjong-ai-bot --endpoint wss://game.riichi.dev/ws/validate --model models/policy.pt
```

For local development, `bot.py` also starts the same bot entry point:

```bash
python bot.py --fallback-only
```

## Configuration

The default TOML config controls:

- Bot endpoint and token environment variable.
- Raw and processed data directories.
- Model artifact path.
- Training batch size, epochs, learning rate, hidden size, architecture, validation split, seed, workers, and early stopping.
- Evaluation game count and action timeout.

Use a custom config file with:

```bash
mahjong-ai-train --config path/to/config.toml
mahjong-ai-evaluate --config path/to/config.toml local
mahjong-ai-bot --config path/to/config.toml
```

## Project Layout

```text
bot.py                         Local bot entry-point wrapper
config/default.toml            Default runtime configuration
scripts/summarize_results.py   Result summary helper
src/mahjong_ai/agents/         Agent protocols, checkpoint loading, Mortal MJAI subprocess opponent
src/mahjong_ai/bot_runner.py   RiichiLabs websocket bot runner
src/mahjong_ai/data.py         MJAI replay streaming and example extraction
src/mahjong_ai/evaluation/     Offline, local, and tournament evaluation
src/mahjong_ai/features/       Observation/action encoding
src/mahjong_ai/training/       Supervised + REINFORCE RL training entry points and models
```
