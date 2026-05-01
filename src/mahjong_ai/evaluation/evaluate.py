"""Placeholder local evaluation entry point."""

from __future__ import annotations

from mahjong_ai.config import load_config


def main() -> None:
    config = load_config()
    raise NotImplementedError(
        "Local evaluation is not implemented yet. "
        f"Configured evaluation games: {config.evaluation.games}"
    )


if __name__ == "__main__":
    main()
