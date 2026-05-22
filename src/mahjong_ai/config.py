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
class RLTrainingConfig:
    """Hyper-parameters for REINFORCE policy-gradient training in RiichiEnv."""

    warmstart_model: Path
    episodes_per_update: int = 16
    rl_epochs: int = 200
    gamma: float = 1.0
    entropy_coef: float = 0.01
    baseline_momentum: float = 0.99
    reward_mode: str = "rank"
    opponent: str = "random"
    learning_rate: float = 1e-4
    game_mode: str = "4p-red-single"
    max_actions_per_game: int = 5000
    controlled_seat: int = 0
    mortal_binary: str | None = None
    mortal_model_dir: Path | None = None
    mortal_docker_image: str = "mortal:latest"
    mortal_timeout: float = 30.0
    akochan_dir: str | None = None
    akochan_tactics: Path | None = None
    akochan_timeout: float = 30.0
    shanten_bonus: float = 0.05
    tenpai_bonus: float = 0.10
    riichi_bonus: float = 0.10


@dataclass(frozen=True)
class AppConfig:
    bot: BotConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    rl_training: RLTrainingConfig


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
        rl_training=_load_rl_training(data, model_artifact=Path(_section(data, "model")["artifact_path"])),
    )


def _load_rl_training(data: dict[str, Any], *, model_artifact: Path) -> RLTrainingConfig:
    section = data.get("rl_training")
    if not isinstance(section, dict):
        return RLTrainingConfig(warmstart_model=model_artifact)
    warmstart = section.get("warmstart_model")
    warmstart_path = Path(warmstart) if warmstart else model_artifact
    _raw_akochan_dir = section.get("akochan_dir")
    akochan_dir = (
        (str(_raw_akochan_dir).strip() or None) if _raw_akochan_dir is not None else None
    )
    return RLTrainingConfig(
        warmstart_model=warmstart_path,
        episodes_per_update=int(section.get("episodes_per_update", 16)),
        rl_epochs=int(section.get("rl_epochs", 200)),
        gamma=float(section.get("gamma", 1.0)),
        entropy_coef=float(section.get("entropy_coef", 0.01)),
        baseline_momentum=float(section.get("baseline_momentum", 0.99)),
        reward_mode=str(section.get("reward_mode", "rank")),
        opponent=str(section.get("opponent", "random")),
        learning_rate=float(section.get("learning_rate", 1e-4)),
        game_mode=str(section.get("game_mode", "4p-red-single")),
        max_actions_per_game=int(section.get("max_actions_per_game", 5000)),
        controlled_seat=int(section.get("controlled_seat", 0)),
        mortal_binary=(
            str(section["mortal_binary"]).strip() if section.get("mortal_binary") else None
        ),
        mortal_model_dir=(
            Path(section["mortal_model_dir"]) if section.get("mortal_model_dir") else None
        ),
        mortal_docker_image=str(section.get("mortal_docker_image", "mortal:latest")),
        mortal_timeout=float(section.get("mortal_timeout", 30.0)),
        akochan_dir=akochan_dir,
        akochan_tactics=(
            Path(section["akochan_tactics"]) if section.get("akochan_tactics") else None
        ),
        akochan_timeout=float(section.get("akochan_timeout", 30.0)),
        shanten_bonus=float(section.get("shanten_bonus", 0.05)),
        tenpai_bonus=float(section.get("tenpai_bonus", 0.10)),
        riichi_bonus=float(section.get("riichi_bonus", 0.10)),
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
