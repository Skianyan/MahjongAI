"""Small supervised policy network used by the baseline trainer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class PolicyModelConfig:
    """Serializable architecture settings for the baseline policy."""

    input_shape: tuple[int, int]
    action_count: int
    model_arch: str = "mlp"
    hidden_size: int = 512

    @property
    def input_size(self) -> int:
        return self.input_shape[0] * self.input_shape[1]

    def to_mapping(self) -> dict[str, int | list[int]]:
        return {
            "input_shape": list(self.input_shape),
            "input_size": self.input_size,
            "action_count": self.action_count,
            "model_arch": self.model_arch,
            "hidden_size": self.hidden_size,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> PolicyModelConfig:
        input_shape = data.get("input_shape")
        if not isinstance(input_shape, Sequence) or len(input_shape) != 2:
            raise ValueError("Policy model config must include a 2D input_shape")

        return cls(
            input_shape=(int(input_shape[0]), int(input_shape[1])),
            action_count=int(data["action_count"]),
            model_arch=str(data.get("model_arch", "mlp")),
            hidden_size=int(data.get("hidden_size", 512)),
        )


class MLPPolicy(nn.Module):
    """Compact MLP over riichienv's encoded observation planes."""

    def __init__(self, config: PolicyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.action_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ConvPolicy(nn.Module):
    """Convolutional policy over observation planes."""

    def __init__(self, config: PolicyModelConfig) -> None:
        super().__init__()
        channel_count, tile_count = config.input_shape
        self.config = config
        self.network = nn.Sequential(
            nn.Conv1d(channel_count, config.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(config.hidden_size * tile_count, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.action_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_policy_model(config: PolicyModelConfig) -> nn.Module:
    """Instantiate the configured policy architecture."""
    if config.model_arch == "conv":
        return ConvPolicy(config)
    if config.model_arch == "mlp":
        return MLPPolicy(config)
    raise ValueError(f"Unsupported model_arch: {config.model_arch}")


def mask_illegal_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Suppress illegal actions before scoring or cross-entropy."""
    if logits.shape != legal_mask.shape:
        raise ValueError(f"Expected legal mask shape {tuple(logits.shape)}, got {tuple(legal_mask.shape)}")
    return logits.masked_fill(~legal_mask, -1e9)
