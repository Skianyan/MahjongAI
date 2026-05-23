"""Offline diagnostics for discard imitation quality."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from riichienv import Action, ActionType, Observation, calculate_shanten

from mahjong_ai.features.actions import ActionSpec, ActionVocabulary
from mahjong_ai.features.tiles import is_honor_pai, is_red_five_tile, is_terminal_pai, tile_id_to_pai


@dataclass(slots=True)
class DiscardMetricCounts:
    discard_examples: int = 0
    model_matches_expert: int = 0
    model_red_five_discards: int = 0
    expert_red_five_discards: int = 0
    model_honor_discards: int = 0
    model_terminal_discards: int = 0
    model_pair_breaks: int = 0
    model_shanten_regressions: int = 0
    expert_shanten_regressions: int = 0

    def record(
        self,
        *,
        observation: Observation,
        expert_action: Action,
        model_action: Action,
        match: bool,
    ) -> None:
        self.discard_examples += 1
        if match:
            self.model_matches_expert += 1

        hand = list(observation.hand)
        for label, action in (("model", model_action), ("expert", expert_action)):
            tile = action.tile
            if tile is None:
                continue
            pai = tile_id_to_pai(tile)
            if label == "model":
                if is_red_five_tile(tile):
                    self.model_red_five_discards += 1
                if is_honor_pai(pai):
                    self.model_honor_discards += 1
                if is_terminal_pai(pai):
                    self.model_terminal_discards += 1
                if hand.count(tile) >= 2:
                    self.model_pair_breaks += 1
                shanten_before = calculate_shanten(hand)
                hand_after = list(hand)
                if tile in hand_after:
                    hand_after.remove(tile)
                    shanten_after = calculate_shanten(hand_after)
                    if shanten_after > shanten_before:
                        self.model_shanten_regressions += 1
            elif is_red_five_tile(tile):
                self.expert_red_five_discards += 1
                shanten_before = calculate_shanten(hand)
                hand_after = list(hand)
                if tile in hand_after:
                    hand_after.remove(tile)
                    shanten_after = calculate_shanten(hand_after)
                    if shanten_after > shanten_before:
                        self.expert_shanten_regressions += 1

    def rates(self) -> dict[str, float | int]:
        total = max(self.discard_examples, 1)
        return {
            "discard_examples": self.discard_examples,
            "discard_match_rate": self.model_matches_expert / total,
            "model_red_five_discard_rate": self.model_red_five_discards / total,
            "expert_red_five_discard_rate": self.expert_red_five_discards / total,
            "model_honor_discard_rate": self.model_honor_discards / total,
            "model_terminal_discard_rate": self.model_terminal_discards / total,
            "model_pair_break_rate": self.model_pair_breaks / total,
            "model_shanten_regression_rate": self.model_shanten_regressions / total,
            "expert_shanten_regression_rate": self.expert_shanten_regressions / total,
        }


def select_model_discard(
    observation: Observation,
    vocabulary: ActionVocabulary,
    scores: list[float] | tuple[float, ...],
) -> Action | None:
    """Pick the model's best legal discard action, if any."""
    legal_mask = vocabulary.mask_for(observation, add_missing=False)
    discard_indices = [
        action_id
        for action_id in legal_mask.legal_indices
        if vocabulary.decode(action_id).action_type == int(ActionType.DISCARD)
    ]
    if not discard_indices:
        return None
    best_id = max(discard_indices, key=lambda action_id: scores[action_id])
    return vocabulary.find_legal_action(observation, best_id)


def is_discard_action(action: Action) -> bool:
    return int(action.action_type) == int(ActionType.DISCARD)


def discard_metrics_notes() -> list[str]:
    return [
        "Bad-discard metrics are diagnostic; high expert red-five rate may be normal in some datasets.",
        "Pair-break counts discards where the tile still appears at least twice in the current hand.",
    ]
