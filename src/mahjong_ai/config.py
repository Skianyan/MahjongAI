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
    model_arch: str = "mlp"
    validation_ratio: float = 0.0
    seed: int = 7
    num_workers: int = 0
    early_stopping_patience: int | None = None
    action_type_weight_power: float = 0.0


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
        training=TrainingConfig(
            batch_size=int(_section(data, "training")["batch_size"]),
            epochs=int(_section(data, "training")["epochs"]),
            device=str(_section(data, "training")["device"]),
            learning_rate=float(_section(data, "training").get("learning_rate", 1e-3)),
            hidden_size=int(_section(data, "training").get("hidden_size", 512)),
            model_arch=str(_section(data, "training").get("model_arch", "mlp")),
            validation_ratio=float(_section(data, "training").get("validation_ratio", 0.0)),
            seed=int(_section(data, "training").get("seed", 7)),
            num_workers=int(_section(data, "training").get("num_workers", 0)),
            early_stopping_patience=_optional_int(
                _section(data, "training").get("early_stopping_patience")
            ),
            action_type_weight_power=float(
                _section(data, "training").get("action_type_weight_power", 0.0)
            ),
        ),
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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
