#!/usr/bin/env python3
"""
Corre N partidas contra el WebSocket de RiichiLabs (p. ej. ranked) y guarda un resumen por partida.

Cada partida usa la misma lógica que mahjong-ai-bot (request_action -> política -> mjai).

Requiere RIICHI_BOT_TOKEN en el entorno. Ejemplo:

  set -a && source .env && set +a
  python scripts/run_ranked_games.py --games 10 \\
    --endpoint wss://game.riichi.dev/ws/ranked \\
    --model models/test170k.pt --device cpu --output results/ranked_runs.jsonl

  python scripts/run_ranked_games.py --games 5 --fallback-only --output results/ranked_fb.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import websockets

from mahjong_ai.bot_runner import _build_serving_agent, _select_response
from mahjong_ai.config import authorization_headers, endpoint_from_env, load_config


async def _play_until_end_game(
    url: str,
    headers: dict[str, str],
    agent: Any,
    *,
    log_all: bool,
    log_sink: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Recibe mensajes hasta end_game; responde request_action."""
    started = time.perf_counter()
    actions = 0
    end_game: dict[str, Any] | None = None

    async with websockets.connect(url, additional_headers=headers) as ws:
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if log_all and log_sink is not None:
                log_sink.append({"ts": time.time(), "type": mtype, "msg": msg})

            if mtype == "end_game":
                end_game = msg
                break

            if mtype != "request_action":
                continue

            actions += 1
            await ws.send(json.dumps(_select_response(msg, agent)))

    return {
        "duration_s": round(time.perf_counter() - started, 3),
        "actions": actions,
        "end_game": end_game,
    }


async def _play_n_on_one_connection(
    url: str,
    headers: dict[str, str],
    agent: Any,
    games: int,
    *,
    log_all: bool,
    log_sink: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Una sola conexión WebSocket; cuenta N mensajes end_game."""
    results: list[dict[str, Any]] = []
    async with websockets.connect(url, additional_headers=headers) as ws:
        game_start = time.perf_counter()
        actions = 0
        while len(results) < games:
            raw = await ws.recv()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if log_all and log_sink is not None:
                log_sink.append({"ts": time.time(), "type": mtype, "msg": msg})

            if mtype == "end_game":
                results.append(
                    {
                        "duration_s": round(time.perf_counter() - game_start, 3),
                        "actions": actions,
                        "end_game": msg,
                    }
                )
                game_start = time.perf_counter()
                actions = 0
                continue

            if mtype != "request_action":
                continue

            actions += 1
            await ws.send(json.dumps(_select_response(msg, agent)))
    return results


async def _async_main(args: argparse.Namespace) -> int:
    token = os.environ.get("RIICHI_BOT_TOKEN", "").strip()
    if not token:
        print(
            "Error: falta RIICHI_BOT_TOKEN. Ej.: set -a && source .env && set +a",
            file=sys.stderr,
        )
        return 1

    config = load_config(args.config)
    url = args.endpoint or endpoint_from_env(config.bot)
    headers = authorization_headers(config.bot)

    model_path = args.model or config.model.artifact_path
    device = args.device or config.training.device
    agent = _build_serving_agent(
        model_path,
        device=device,
        fallback_only=args.fallback_only,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    debug_log: list[dict[str, Any]] = []

    with args.output.open("w", encoding="utf-8") as out:
        if args.reuse_connection:
            rows = await _play_n_on_one_connection(
                url,
                headers,
                agent,
                args.games,
                log_all=args.log_all,
                log_sink=debug_log if args.log_all else None,
            )
            for idx, row in enumerate(rows, start=1):
                record = {"game_index": idx, **row}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                if args.verbose:
                    print(f"partida {idx}/{args.games}: actions={row['actions']} t={row['duration_s']}s", flush=True)
        else:
            for idx in range(1, args.games + 1):
                try:
                    row = await _play_until_end_game(
                        url,
                        headers,
                        agent,
                        log_all=args.log_all,
                        log_sink=debug_log if args.log_all else None,
                    )
                except Exception as exc:
                    row = {
                        "duration_s": None,
                        "actions": None,
                        "end_game": None,
                        "error": repr(exc),
                    }
                record = {"game_index": idx, **row}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                if args.verbose:
                    err = record.get("error")
                    extra = f" err={err}" if err else ""
                    print(
                        f"partida {idx}/{args.games}: actions={record.get('actions')} "
                        f"t={record.get('duration_s')}s{extra}",
                        flush=True,
                    )

        if args.log_all and args.debug_output is not None:
            args.debug_output.parent.mkdir(parents=True, exist_ok=True)
            with args.debug_output.open("w", encoding="utf-8") as dbg:
                for line in debug_log:
                    dbg.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Wrote {args.games} game record(s) to {args.output}", file=sys.stderr)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run N RiichiLabs websocket games (e.g. ranked) and log JSONL summaries.",
    )
    parser.add_argument("--games", type=int, required=True, help="Number of games to play")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ranked_runs.jsonl"),
        help="JSONL path: one object per game (duration_s, actions, end_game, ...)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="WebSocket URL (default: RIICHI_BOT_ENDPOINT or [bot].endpoint in config)",
    )
    parser.add_argument("--config", type=Path, default=None, help="TOML config path")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Checkpoint path (default: [model].artifact_path from config)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (default: [training].device from config)",
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Use deterministic fallback agent (no checkpoint)",
    )
    parser.add_argument(
        "--reuse-connection",
        action="store_true",
        help="Keep one WebSocket open and count N end_game events (if the server supports it)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per game to stderr",
    )
    parser.add_argument(
        "--log-all",
        action="store_true",
        help="Record every WS message (requires --debug-output)",
    )
    parser.add_argument(
        "--debug-output",
        type=Path,
        default=None,
        help="JSONL path for all messages when --log-all is set",
    )
    args = parser.parse_args()

    if args.log_all and args.debug_output is None:
        parser.error("--log-all requires --debug-output")

    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
