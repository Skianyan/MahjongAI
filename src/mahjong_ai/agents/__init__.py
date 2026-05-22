"""Agent interfaces, checkpoint loading, and deterministic fallback behavior."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from riichienv import Action, ActionType, Observation

from mahjong_ai.features import ActionVocabulary, encode_observation
from mahjong_ai.features.actions import ActionSpec
from mahjong_ai.features.tiles import discard_priority_score, tile_id_to_type_index
from mahjong_ai.agents.stats import AgentStats, get_agent_stats


class Agent(Protocol):
    stats: AgentStats | None

    def act(self, observation: Observation) -> Action:
        """Choose one legal action for the given observation."""


class ScoringAgent(Agent, Protocol):
    vocabulary: ActionVocabulary

    def scores(self, observation: Observation) -> Sequence[float]:
        """Return unmasked action scores aligned to ``vocabulary``."""


def is_discard_only_turn(observation: Observation) -> bool:
    """True when every legal action is a discard (no calls/riichi/win)."""
    legal = observation.legal_actions()
    return bool(legal) and all(int(action.action_type) == int(ActionType.DISCARD) for action in legal)


class FallbackAgent:
    """Conservative legal-action policy used when checkpoints fail or cannot decide."""

    def __init__(self, *, stats: AgentStats | None = None) -> None:
        self.stats = stats

    def act(self, observation: Observation) -> Action:
        if self.stats is not None:
            self.stats.record_heuristic_fallback()
        legal_actions = observation.legal_actions()
        if not legal_actions:
            return Action(ActionType.PASS)

        for action_type in (ActionType.TSUMO, ActionType.RON):
            action = _first_of_type(legal_actions, action_type)
            if action is not None:
                return action

        pass_action = _first_of_type(legal_actions, ActionType.PASS)
        if pass_action is not None and not is_discard_only_turn(observation):
            return pass_action

        riichi = _first_of_type(legal_actions, ActionType.RIICHI)
        if riichi is not None and not is_discard_only_turn(observation):
            return riichi

        discards = [action for action in legal_actions if int(action.action_type) == int(ActionType.DISCARD)]
        if discards:
            return min(discards, key=lambda action: discard_priority_score(action.tile))

        return pass_action or legal_actions[0]


class SafeAgent:
    """Use a trained policy when possible, falling back for serving-time errors."""

    def __init__(
        self,
        *,
        primary: Agent,
        fallback: Agent | None = None,
        stats: AgentStats | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback or FallbackAgent(stats=stats)
        self.stats = stats or getattr(primary, "stats", None) or AgentStats()

    def act(self, observation: Observation) -> Action:
        try:
            return self._primary.act(observation)
        except Exception as exc:
            if self.stats is not None:
                self.stats.record_inference_exception_fallback()
            print(
                f"Policy inference failed: {exc}. Using conservative fallback action.",
                file=sys.stderr,
            )
            return self._fallback.act(observation)


class ActionPriorAgent:
    """Checkpoint-backed action prior policy with legal-action masking."""

    def __init__(
        self,
        vocabulary: ActionVocabulary,
        log_priors: Sequence[float],
        *,
        fallback: Agent | None = None,
        stats: AgentStats | None = None,
    ) -> None:
        if len(vocabulary) != len(log_priors):
            raise ValueError(f"Expected {len(vocabulary)} priors, got {len(log_priors)}")
        self.vocabulary = vocabulary
        self._log_priors = tuple(float(score) for score in log_priors)
        self.stats = stats or AgentStats()
        self._fallback = fallback or FallbackAgent(stats=self.stats)

    def scores(self, observation: Observation) -> Sequence[float]:
        return self._log_priors

    def act(self, observation: Observation) -> Action:
        action = self.vocabulary.select_best_legal_action(
            observation,
            self._log_priors,
            type_fallback_for_discards=True,
        )
        if action is not None:
            self.stats.record_primary()
            return action
        self.stats.record_no_known_legal_fallback()
        return self._fallback.act(observation)


class MLPPolicyAgent:
    """PyTorch policy checkpoint adapter used by evaluation and bot serving."""

    def __init__(
        self,
        model: Any,
        vocabulary: ActionVocabulary,
        torch_module: Any,
        *,
        device: Any,
        extended: bool,
        fallback: Agent | None = None,
        stats: AgentStats | None = None,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self._torch = torch_module
        self._device = device
        self._extended = extended
        self.stats = stats or AgentStats()
        self._fallback = fallback or FallbackAgent(stats=self.stats)
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
        action = self.vocabulary.select_best_legal_action(
            observation,
            scores,
            type_fallback_for_discards=True,
        )
        if action is not None:
            self.stats.record_primary()
            return action
        self.stats.record_no_known_legal_fallback()
        return self._fallback.act(observation)


class DiscardPolicyAgent:
    """Discard specialist over 34 tile types; used on discard-only turns."""

    def __init__(
        self,
        model: Any,
        torch_module: Any,
        *,
        device: Any,
        extended: bool,
        fallback: Agent | None = None,
        stats: AgentStats | None = None,
    ) -> None:
        self.model = model
        self._torch = torch_module
        self._device = device
        self._extended = extended
        self.stats = stats or AgentStats()
        self._fallback = fallback or FallbackAgent(stats=self.stats)
        self.model.eval()

    def _type_mask(self, observation: Observation) -> tuple[bool, ...]:
        mask = [False] * 34
        for action in observation.legal_actions():
            if int(action.action_type) != int(ActionType.DISCARD) or action.tile is None:
                continue
            type_index = tile_id_to_type_index(action.tile)
            if 0 <= type_index < 34:
                mask[type_index] = True
        return tuple(mask)

    def act(self, observation: Observation) -> Action:
        features = encode_observation(observation, extended=self._extended)
        tensor = self._torch.frombuffer(bytearray(features.data), dtype=self._torch.float32)
        tensor = tensor.reshape((1, *features.shape)).to(self._device)
        type_mask = self._type_mask(observation)
        with self._torch.no_grad():
            logits = self.model(tensor)[0].detach().cpu()
        masked = [
            float(score) if is_legal else -1e9
            for score, is_legal in zip(logits, type_mask, strict=True)
        ]
        if not any(type_mask):
            self.stats.record_no_known_legal_fallback()
            return self._fallback.act(observation)
        best_type = max(range(34), key=lambda idx: masked[idx])
        for action in observation.legal_actions():
            if (
                int(action.action_type) == int(ActionType.DISCARD)
                and action.tile is not None
                and tile_id_to_type_index(action.tile) == best_type
            ):
                self.stats.record_primary()
                return action
        self.stats.record_no_known_legal_fallback()
        return self._fallback.act(observation)


class CompositePolicyAgent:
    """Route discard-only turns to the discard head; other turns to the global policy."""

    def __init__(
        self,
        global_agent: MLPPolicyAgent,
        discard_agent: DiscardPolicyAgent,
    ) -> None:
        self.global_agent = global_agent
        self.discard_agent = discard_agent
        self.vocabulary = global_agent.vocabulary
        self.stats = global_agent.stats

    def scores(self, observation: Observation) -> Sequence[float]:
        return self.global_agent.scores(observation)

    def act(self, observation: Observation) -> Action:
        if is_discard_only_turn(observation):
            return self.discard_agent.act(observation)
        return self.global_agent.act(observation)


def load_checkpoint_agent(path: Path, *, device: str = "auto") -> ScoringAgent:
    """Load a saved policy checkpoint."""
    checkpoint = _load_checkpoint(path, device=device)
    model_type = checkpoint.get("model_type")
    vocabulary = ActionVocabulary.from_mapping(checkpoint["action_vocabulary"])
    stats = AgentStats()
    fallback = FallbackAgent(stats=stats)

    if model_type == "action_prior":
        return ActionPriorAgent(vocabulary, checkpoint["model"]["log_priors"], stats=stats)

    if model_type in ("mlp_policy", "policy_network"):
        return _load_policy_network_agent(checkpoint, vocabulary, device=device, stats=stats, fallback=fallback)

    if model_type == "policy_with_discard_head":
        global_agent = _load_policy_network_agent(
            checkpoint,
            vocabulary,
            device=device,
            stats=stats,
            fallback=fallback,
            state_key="state_dict",
        )
        discard_agent = _load_discard_head_agent(checkpoint, device=device, stats=stats, fallback=fallback)
        return CompositePolicyAgent(global_agent, discard_agent)

    if model_type == "discard_policy_network":
        return _load_discard_head_agent(checkpoint, device=device, stats=stats, fallback=fallback)

    raise ValueError(f"Unsupported checkpoint model_type: {model_type!r}")


def _load_policy_network_agent(
    checkpoint: dict[str, Any],
    vocabulary: ActionVocabulary,
    *,
    device: str,
    stats: AgentStats,
    fallback: FallbackAgent,
    state_key: str = "state_dict",
) -> MLPPolicyAgent:
    torch = _import_torch()
    from mahjong_ai.training.policy import PolicyModelConfig, build_policy_model

    resolved_device = torch.device(_resolve_device(device, torch))
    model_config = PolicyModelConfig.from_mapping(checkpoint["model_config"])
    model = build_policy_model(model_config).to(resolved_device)
    model.load_state_dict(checkpoint[state_key])
    feature_schema = checkpoint.get("feature_schema", {})
    return MLPPolicyAgent(
        model,
        vocabulary,
        torch,
        device=resolved_device,
        extended=bool(feature_schema.get("extended", False)),
        fallback=fallback,
        stats=stats,
    )


def _load_discard_head_agent(
    checkpoint: dict[str, Any],
    *,
    device: str,
    stats: AgentStats,
    fallback: FallbackAgent,
) -> DiscardPolicyAgent:
    torch = _import_torch()
    from mahjong_ai.training.policy import DiscardPolicyConfig, build_discard_policy

    resolved_device = torch.device(_resolve_device(device, torch))
    discard_config = DiscardPolicyConfig.from_mapping(checkpoint["discard_model_config"])
    model = build_discard_policy(discard_config).to(resolved_device)
    model.load_state_dict(checkpoint["discard_state_dict"])
    feature_schema = checkpoint.get("feature_schema", {})
    return DiscardPolicyAgent(
        model,
        torch,
        device=resolved_device,
        extended=bool(feature_schema.get("extended", False)),
        fallback=fallback,
        stats=stats,
    )


def build_serving_agent(
    model_path: Path,
    *,
    device: str,
    fallback_only: bool,
) -> Agent:
    """Load checkpoint for RiichiLabs serving with stats and safe fallback."""
    stats = AgentStats()
    fallback = FallbackAgent(stats=stats)
    if fallback_only:
        print("Running with conservative fallback agent only.", file=sys.stderr)
        stats.checkpoint_load_fallback = True
        return SafeAgent(primary=fallback, fallback=fallback, stats=stats)

    try:
        trained_agent = load_checkpoint_agent(model_path, device=device)
    except Exception as exc:
        print(
            f"Failed to load checkpoint {model_path}: {exc}. "
            "Running with conservative fallback agent.",
            file=sys.stderr,
        )
        stats.checkpoint_load_fallback = True
        return SafeAgent(primary=fallback, fallback=fallback, stats=stats)

    agent_stats = get_agent_stats(trained_agent) or stats
    print(f"Loaded checkpoint from {model_path}.", file=sys.stderr)
    return SafeAgent(primary=trained_agent, fallback=fallback, stats=agent_stats)


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
    try:
        return torch.load(path, map_location=_resolve_device(device, torch), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=_resolve_device(device, torch))


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to load policy checkpoints. Install training extras "
            "with `pip install -e '.[train]'`."
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


from .akochan import AkochanAgent, build_akochan_argv
from .mortal import MortalAgent, build_mortal_argv

__all__ = [
    "Agent",
    "ScoringAgent",
    "AgentStats",
    "get_agent_stats",
    "FallbackAgent",
    "SafeAgent",
    "ActionPriorAgent",
    "MLPPolicyAgent",
    "DiscardPolicyAgent",
    "CompositePolicyAgent",
    "build_serving_agent",
    "is_discard_only_turn",
    "MortalAgent",
    "build_mortal_argv",
    "AkochanAgent",
    "build_akochan_argv",
    "load_checkpoint_agent",
    "top_k_action_ids",
    "masked_top_action_id",
]
