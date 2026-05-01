"""Configuration loading for training, evaluation, and bot serving."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Any


DEFAULT_CONFIG_PATH = Path("config/default.toml")


@dataclass(frozen=True)
class BotConfig:
    endpoint: str
    token_env: str = "RIICHI_BOT_TOKEN"


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    processed_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    artifact_path: Path


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    device: str
    learning_rate: float = 1e-3
    hidden_size: int = 512


@dataclass(frozen=True)
class EvaluationConfig:
    games: int
    timeout_seconds: float


@dataclass(frozen=True)
class AppConfig:
    bot: BotConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load TOML config, allowing MAHJONG_AI_CONFIG to select a file."""
    config_path = Path(path or os.environ.get("MAHJONG_AI_CONFIG", DEFAULT_CONFIG_PATH))
    with config_path.open("rb") as config_file:
        data = tomllib.load(config_file)

    return AppConfig(
        bot=BotConfig(**_section(data, "bot")),
        data=DataConfig(
            raw_dir=Path(_section(data, "data")["raw_dir"]),
            processed_dir=Path(_section(data, "data")["processed_dir"]),
        ),
        model=ModelConfig(artifact_path=Path(_section(data, "model")["artifact_path"])),
        training=TrainingConfig(**_section(data, "training")),
        evaluation=EvaluationConfig(**_section(data, "evaluation")),
    )


def get_bot_token(config: BotConfig) -> str:
    """Return the bot token from the configured environment variable."""
    token = os.environ.get(config.token_env, "").strip()
    if not token:
        raise RuntimeError(
            f"Missing RiichiLabs bot token. Set {config.token_env} before starting the bot."
        )
    return token


def authorization_headers(config: BotConfig) -> dict[str, str]:
    return {"Authorization": f"Bearer {get_bot_token(config)}"}


def endpoint_from_env(config: BotConfig) -> str:
    return os.environ.get("RIICHI_BOT_ENDPOINT", config.endpoint)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing [{name}] section in config")
    return section
