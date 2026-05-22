"""Counters for primary-policy vs fallback decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentStats:
    """Mutable counters shared by serving and evaluation agents."""

    total_decisions: int = 0
    primary_decisions: int = 0
    checkpoint_load_fallback: bool = False
    inference_exception_fallback: int = 0
    no_known_legal_fallback: int = 0
    heuristic_fallback_decisions: int = 0

    def record_primary(self) -> None:
        self.total_decisions += 1
        self.primary_decisions += 1

    def record_inference_exception_fallback(self) -> None:
        self.total_decisions += 1
        self.inference_exception_fallback += 1

    def record_no_known_legal_fallback(self) -> None:
        self.total_decisions += 1
        self.no_known_legal_fallback += 1

    def record_heuristic_fallback(self) -> None:
        self.total_decisions += 1
        self.heuristic_fallback_decisions += 1

    def fallback_decisions(self) -> int:
        return (
            self.inference_exception_fallback
            + self.no_known_legal_fallback
            + self.heuristic_fallback_decisions
        )

    def fallback_rate(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return self.fallback_decisions() / self.total_decisions

    def to_mapping(self) -> dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "primary_decisions": self.primary_decisions,
            "checkpoint_load_fallback": self.checkpoint_load_fallback,
            "inference_exception_fallback": self.inference_exception_fallback,
            "no_known_legal_fallback": self.no_known_legal_fallback,
            "heuristic_fallback_decisions": self.heuristic_fallback_decisions,
            "fallback_decisions": self.fallback_decisions(),
            "fallback_rate": self.fallback_rate(),
        }

    def copy(self) -> AgentStats:
        return AgentStats(
            total_decisions=self.total_decisions,
            primary_decisions=self.primary_decisions,
            checkpoint_load_fallback=self.checkpoint_load_fallback,
            inference_exception_fallback=self.inference_exception_fallback,
            no_known_legal_fallback=self.no_known_legal_fallback,
            heuristic_fallback_decisions=self.heuristic_fallback_decisions,
        )

    def snapshot_delta(self, before: AgentStats) -> dict[str, Any]:
        after = self.to_mapping()
        prior = before.to_mapping()
        return {
            key: after[key] - prior[key]
            if isinstance(after[key], (int, float)) and not isinstance(after[key], bool)
            else after[key]
            for key in after
        }


def get_agent_stats(agent: Any) -> AgentStats | None:
    """Return shared stats from an agent wrapper, if present."""
    stats = getattr(agent, "stats", None)
    if isinstance(stats, AgentStats):
        return stats
    primary = getattr(agent, "_primary", None)
    if primary is not None:
        return get_agent_stats(primary)
    return None
