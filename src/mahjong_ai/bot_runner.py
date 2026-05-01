"""RiichiLabs websocket bot entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from riichienv import Observation
import websockets

from mahjong_ai.agents import Agent, FallbackAgent
from mahjong_ai.config import authorization_headers, endpoint_from_env, load_config


async def run_bot(url: str, headers: dict[str, str], agent: Agent | None = None) -> None:
    """Connect to RiichiLabs and respond to action requests."""
    selected_agent = agent or FallbackAgent()
    async with websockets.connect(url, additional_headers=headers) as websocket:
        while True:
            message = json.loads(await websocket.recv())
            if message.get("type") == "end_game":
                break

            if message.get("type") != "request_action":
                continue

            response = _select_response(message, selected_agent)
            await websocket.send(json.dumps(response))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Mahjong AI RiichiLabs bot.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML config file. Defaults to MAHJONG_AI_CONFIG or config/default.toml.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Override the configured websocket endpoint.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    url = args.endpoint or endpoint_from_env(config.bot)
    asyncio.run(run_bot(url, authorization_headers(config.bot)))


def _select_response(message: dict[str, Any], agent: Agent) -> dict[str, Any]:
    observation_payload = message.get("observation")
    if not observation_payload:
        return {"type": "none"}

    observation = Observation.deserialize_from_base64(observation_payload)
    action = agent.act(observation)
    return json.loads(action.to_mjai())
