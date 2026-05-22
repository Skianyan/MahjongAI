"""Action vocabulary and legal-action masking helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

from riichienv import Action, ActionType, Observation


ACTION_TYPE_NAMES: dict[int, str] = {
    int(ActionType.DISCARD): "DISCARD",
    int(ActionType.CHI): "CHI",
    int(ActionType.PON): "PON",
    int(ActionType.DAIMINKAN): "DAIMINKAN",
    int(ActionType.RON): "RON",
    int(ActionType.RIICHI): "RIICHI",
    int(ActionType.TSUMO): "TSUMO",
    int(ActionType.PASS): "PASS",
    int(ActionType.ANKAN): "ANKAN",
    int(ActionType.KAKAN): "KAKAN",
    int(ActionType.KYUSHU_KYUHAI): "KYUSHU_KYUHAI", # 9 terminals rule
    int(ActionType.KITA): "KITA",
}
ACTION_TYPES_BY_ID: dict[int, ActionType] = {
    int(ActionType.DISCARD): ActionType.DISCARD,
    int(ActionType.CHI): ActionType.CHI,
    int(ActionType.PON): ActionType.PON,
    int(ActionType.DAIMINKAN): ActionType.DAIMINKAN,
    int(ActionType.RON): ActionType.RON,
    int(ActionType.RIICHI): ActionType.RIICHI,
    int(ActionType.TSUMO): ActionType.TSUMO,
    int(ActionType.PASS): ActionType.PASS,
    int(ActionType.ANKAN): ActionType.ANKAN,
    int(ActionType.KAKAN): ActionType.KAKAN,
    int(ActionType.KYUSHU_KYUHAI): ActionType.KYUSHU_KYUHAI,
    int(ActionType.KITA): ActionType.KITA,
}


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Serializable, hashable identity for a riichienv action."""

    action_type: int
    tile: int | None
    consume_tiles: tuple[int, ...] = ()

    @classmethod
    def from_action(cls, action: Action) -> ActionSpec:
        return cls(
            action_type=int(action.action_type),
            tile=None if action.tile is None else int(action.tile),
            consume_tiles=tuple(int(tile) for tile in action.consume_tiles),
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ActionSpec:
        consume_tiles = data.get("consume_tiles", ())
        if not isinstance(consume_tiles, Iterable):
            raise ValueError("consume_tiles must be iterable")
        return cls(
            action_type=int(data["action_type"]),
            tile=None if data["tile"] is None else int(data["tile"]),
            consume_tiles=tuple(int(tile) for tile in consume_tiles),
        )

    @property
    def action_type_name(self) -> str:
        return ACTION_TYPE_NAMES.get(self.action_type, str(self.action_type))

    def to_action(self) -> Action:
        action_type = ACTION_TYPES_BY_ID.get(self.action_type)
        if action_type is None:
            raise ValueError(f"Unknown action type id {self.action_type}")
        return Action(action_type, self.tile, list(self.consume_tiles))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "action_type_name": self.action_type_name,
            "tile": self.tile,
            "consume_tiles": list(self.consume_tiles),
        }


@dataclass(frozen=True, slots=True)
class LegalActionMask:
    """Boolean mask aligned to an action vocabulary."""

    mask: tuple[bool, ...]
    legal_indices: tuple[int, ...]
    unknown_actions: tuple[ActionSpec, ...] = ()

    def __len__(self) -> int:
        return len(self.mask)

    def is_legal(self, action_id: int) -> bool:
        return 0 <= action_id < len(self.mask) and self.mask[action_id]

    def apply_to_scores(
        self,
        scores: Sequence[float],
        *,
        illegal_score: float = -math.inf,
    ) -> tuple[float, ...]:
        """Return scores with illegal actions replaced by *illegal_score*."""
        if len(scores) != len(self.mask):
            raise ValueError(f"Expected {len(self.mask)} scores, got {len(scores)}")
        return tuple(score if is_legal else illegal_score for score, is_legal in zip(scores, self.mask))


@dataclass(slots=True)
class ActionVocabulary:
    """Stable mapping between riichienv actions and model action ids."""

    specs: list[ActionSpec] = field(default_factory=list)
    _index: dict[ActionSpec, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        existing_specs = list(self.specs)
        self.specs.clear()
        for spec in existing_specs:
            self.add_spec(spec)

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Iterable[ActionSpec]:
        return iter(self.specs)

    @classmethod
    def from_actions(cls, actions: Iterable[Action]) -> ActionVocabulary:
        vocabulary = cls()
        vocabulary.add_actions(actions)
        return vocabulary

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ActionVocabulary:
        specs_data = data.get("actions", ())
        if not isinstance(specs_data, Iterable):
            raise ValueError("actions must be iterable")
        return cls([ActionSpec.from_mapping(spec_data) for spec_data in specs_data])

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": 1,
            "actions": [spec.to_mapping() for spec in self.specs],
        }

    def add(self, action: Action) -> int:
        return self.add_spec(ActionSpec.from_action(action))

    def add_actions(self, actions: Iterable[Action]) -> tuple[int, ...]:
        return tuple(self.add(action) for action in actions)

    def add_spec(self, spec: ActionSpec) -> int:
        existing = self._index.get(spec)
        if existing is not None:
            return existing

        action_id = len(self.specs)
        self.specs.append(spec)
        self._index[spec] = action_id
        return action_id

    def encode(self, action: Action) -> int:
        spec = ActionSpec.from_action(action)
        try:
            return self._index[spec]
        except KeyError as exc:
            raise UnknownActionError(spec) from exc

    def encode_or_add(self, action: Action) -> int:
        return self.add(action)

    def decode(self, action_id: int) -> ActionSpec:
        try:
            return self.specs[action_id]
        except IndexError as exc:
            raise ValueError(f"Unknown action id {action_id}") from exc

    def decode_action(self, action_id: int) -> Action:
        return self.decode(action_id).to_action()

    def mask_for(self, observation: Observation, *, add_missing: bool = False) -> LegalActionMask:
        """Build a legal mask from ``observation.legal_actions()``."""
        legal_indices: list[int] = []
        unknown_actions: list[ActionSpec] = []

        for action in observation.legal_actions():
            spec = ActionSpec.from_action(action)
            action_id = self._index.get(spec)
            if action_id is None:
                if not add_missing:
                    unknown_actions.append(spec)
                    continue
                action_id = self.add_spec(spec)
            legal_indices.append(action_id)

        mask = [False] * len(self)
        for action_id in legal_indices:
            mask[action_id] = True

        return LegalActionMask(
            mask=tuple(mask),
            legal_indices=tuple(legal_indices),
            unknown_actions=tuple(unknown_actions),
        )

    def coverage_for(self, observation: Observation) -> dict[str, int]:
        """Return known/unknown legal action coverage for one observation."""
        legal_mask = self.mask_for(observation, add_missing=False)
        total_legal = len(legal_mask.legal_indices) + len(legal_mask.unknown_actions)
        return {
            "total_legal_actions": total_legal,
            "known_legal_actions": len(legal_mask.legal_indices),
            "unknown_legal_actions": len(legal_mask.unknown_actions),
        }

    def find_legal_action(self, observation: Observation, action_id: int) -> Action | None:
        """Return the matching legal action for *action_id*, if currently available."""
        spec = self.decode(action_id)
        for action in observation.legal_actions():
            if ActionSpec.from_action(action) == spec:
                return action
        return None

    def select_best_legal_action(
        self,
        observation: Observation,
        scores: Sequence[float],
        *,
        add_missing: bool = False,
        type_fallback_for_discards: bool = False,
    ) -> Action | None:
        """Pick the highest-scoring currently legal action."""
        legal_mask = self.mask_for(observation, add_missing=add_missing)
        if len(scores) != len(legal_mask):
            raise ValueError(f"Expected {len(legal_mask)} scores, got {len(scores)}")
        if legal_mask.legal_indices:
            best_id = max(legal_mask.legal_indices, key=lambda action_id: scores[action_id])
            action = self.find_legal_action(observation, best_id)
            if action is not None:
                return action

        if type_fallback_for_discards:
            return self._select_discard_by_tile_type(observation, scores)
        return None

    def _select_discard_by_tile_type(
        self,
        observation: Observation,
        scores: Sequence[float],
    ) -> Action | None:
        """Map unknown discard tile ids to the best-scoring known tile type."""
        from mahjong_ai.features.tiles import tile_id_to_type_index

        discard_type = int(ActionType.DISCARD)
        discard_actions = [
            action
            for action in observation.legal_actions()
            if int(action.action_type) == discard_type and action.tile is not None
        ]
        if not discard_actions:
            return None

        best_action: Action | None = None
        best_type_score = -math.inf
        for candidate in discard_actions:
            type_index = tile_id_to_type_index(int(candidate.tile))
            type_score = -math.inf
            for action_id, spec in enumerate(self.specs):
                if spec.action_type != discard_type or spec.tile is None:
                    continue
                if tile_id_to_type_index(spec.tile) == type_index:
                    type_score = max(type_score, scores[action_id])
            if type_score > best_type_score:
                best_type_score = type_score
                best_action = candidate
        return best_action


class UnknownActionError(KeyError):
    """Raised when an action is absent from a frozen action vocabulary."""

    def __init__(self, spec: ActionSpec) -> None:
        self.spec = spec
        super().__init__(f"Action is not in vocabulary: {spec.to_mapping()}")
