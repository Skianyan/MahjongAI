"""Agent interfaces and deterministic fallback behavior."""

from __future__ import annotations

from typing import Protocol

from riichienv import Action, ActionType, Observation


class Agent(Protocol):
    def act(self, observation: Observation) -> Action:
        """Choose one legal action for the given observation."""


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


def _first_of_type(actions: list[Action], action_type: ActionType) -> Action | None:
    return next((action for action in actions if action.action_type == action_type), None)
