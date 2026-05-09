"""Agent interfaces, checkpoint loading, and deterministic fallback behavior."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from riichienv import Action, ActionType, Observation

from mahjong_ai.features import ActionVocabulary, encode_observation


class Agent(Protocol):
    def act(self, observation: Observation) -> Action:
        """Choose one legal action for the given observation."""


class ScoringAgent(Agent, Protocol):
    vocabulary: ActionVocabulary

    def scores(self, observation: Observation) -> Sequence[float]:
        """Return unmasked action scores aligned to ``vocabulary``."""


class FallbackAgent:
    """Simple legal-action policy used before a learned model is available."""

    def act(self, observation: Observation) -> Action:
        legal_actions = observation.legal_actions()
        if not legal_actions:
            return Action(ActionType.PASS)

        for action_type in (ActionType.TSUMO, ActionType.RON, ActionType.RIICHI):
            action = _first_of_type(legal_actions, action_type)
            if action is not None:
                return action

        discard = _first_of_type(legal_actions, ActionType.DISCARD)
        if discard is not None:
            return discard

        pass_action = _first_of_type(legal_actions, ActionType.PASS)
        return pass_action or legal_actions[0]


class ActionPriorAgent:
    """Checkpoint-backed action prior policy with legal-action masking."""

    def __init__(
        self,
        vocabulary: ActionVocabulary,
        log_priors: Sequence[float],
        *,
        fallback: Agent | None = None,
    ) -> None:
        if len(vocabulary) != len(log_priors):
            raise ValueError(f"Expected {len(vocabulary)} priors, got {len(log_priors)}")
        self.vocabulary = vocabulary
        self._log_priors = tuple(float(score) for score in log_priors)
        self._fallback = fallback or FallbackAgent()

    def scores(self, observation: Observation) -> Sequence[float]:
        return self._log_priors

    def act(self, observation: Observation) -> Action:
        action = self.vocabulary.select_best_legal_action(observation, self._log_priors)
        return action or self._fallback.act(observation)


class MLPPolicyAgent:
    """PyTorch MLP checkpoint adapter used by evaluation and bot serving."""

    def __init__(
        self,
        model: Any,
        vocabulary: ActionVocabulary,
        torch_module: Any,
        *,
        device: Any,
        extended: bool,
        fallback: Agent | None = None,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self._torch = torch_module
        self._device = device
        self._extended = extended
        self._fallback = fallback or FallbackAgent()
        self.model.eval()

    def scores(self, observation: Observation) -> Sequence[float]:
        features = encode_observation(observation, extended=self._extended)
        tensor = self._torch.frombuffer(bytearray(features.data), dtype=self._torch.float32)
        tensor = tensor.reshape((1, *features.shape)).to(self._device)
        with self._torch.no_grad():
            logits = self.model(tensor)[0].detach().cpu()
        return tuple(float(score) for score in logits)

    def act(self, observation: Observation) -> Action:
        scores = self.scores(observation)
        action = self.vocabulary.select_best_legal_action(observation, scores)
        return action or self._fallback.act(observation)


def load_checkpoint_agent(path: Path, *, device: str = "auto") -> ScoringAgent:
    """Load a saved action-prior or MLP policy checkpoint."""
    checkpoint = _load_checkpoint(path, device=device)
    model_type = checkpoint.get("model_type")
    vocabulary = ActionVocabulary.from_mapping(checkpoint["action_vocabulary"])

    if model_type == "action_prior":
        return ActionPriorAgent(vocabulary, checkpoint["model"]["log_priors"])

    if model_type == "mlp_policy":
        torch = _import_torch()
        from mahjong_ai.training.policy import PolicyModelConfig, build_policy_model

        resolved_device = torch.device(_resolve_device(device, torch))
        model_config = PolicyModelConfig.from_mapping(checkpoint["model_config"])
        model = build_policy_model(model_config).to(resolved_device)
        model.load_state_dict(checkpoint["state_dict"])
        feature_schema = checkpoint.get("feature_schema", {})
        return MLPPolicyAgent(
            model,
            vocabulary,
            torch,
            device=resolved_device,
            extended=bool(feature_schema.get("extended", False)),
        )

    if model_type == "policy_network":
        torch = _import_torch()
        from mahjong_ai.training.policy import PolicyModelConfig, build_policy_model

        resolved_device = torch.device(_resolve_device(device, torch))
        model_config = PolicyModelConfig.from_mapping(checkpoint["model_config"])
        model = build_policy_model(model_config).to(resolved_device)
        model.load_state_dict(checkpoint["state_dict"])
        feature_schema = checkpoint.get("feature_schema", {})
        return MLPPolicyAgent(
            model,
            vocabulary,
            torch,
            device=resolved_device,
            extended=bool(feature_schema.get("extended", False)),
        )

    raise ValueError(f"Unsupported checkpoint model_type: {model_type!r}")


def top_k_action_ids(scores: Sequence[float], k: int) -> tuple[int, ...]:
    """Return action ids ordered by descending score."""
    if k <= 0:
        return ()
    ordered_ids = sorted(
        range(len(scores)),
        key=lambda action_id: scores[action_id],
        reverse=True,
    )
    return tuple(ordered_ids[:k])


def masked_top_action_id(scores: Sequence[float], legal_indices: Sequence[int]) -> int | None:
    """Return the highest-scoring legal action id, or ``None`` when no known legal action exists."""
    if not legal_indices:
        return None
    return max(legal_indices, key=lambda action_id: scores[action_id])


def _first_of_type(actions: list[Action], action_type: ActionType) -> Action | None:
    return next((action for action in actions if action.action_type == action_type), None)


def _load_checkpoint(path: Path, *, device: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as checkpoint_file:
            return json.load(checkpoint_file)
    except UnicodeDecodeError:
        pass
    except json.JSONDecodeError:
        pass

    torch = _import_torch()
    return torch.load(path, map_location=_resolve_device(device, torch))


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to load MLP checkpoints. Install the training extras "
            "with `pip install -e '.[train]'`, or evaluate an action-prior checkpoint."
        ) from exc
    return torch


def _resolve_device(requested: str, torch_module: Any) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"
