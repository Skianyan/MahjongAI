"""Fallback usage counters for checkpoint-backed agents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
V4_MODEL = REPO_ROOT / "models" / "allyears_v4_no_discard_downweight.pt"


@pytest.fixture(scope="module")
def v4_model() -> Path:
    if not V4_MODEL.exists():
        pytest.skip(f"missing checkpoint: {V4_MODEL}")
    return V4_MODEL


def _run_local(model_path: Path | None, *, games: int) -> dict:
    from mahjong_ai.evaluation.evaluate import LocalEvaluationOptions, evaluate_local

    result = evaluate_local(
        LocalEvaluationOptions(
            model_path=model_path,
            games=games,
            timeout_seconds=15.0,
            max_actions_per_game=5000,
            controlled_seat=0,
            opponent="random",
            fail_on_fallback=False,
        )
    )
    assert result.agent_stats is not None
    return result.agent_stats


def test_v4_fallback_rate_under_threshold(v4_model: Path) -> None:
    """After type-level discard routing, fallback should be rare in local play."""
    stats = _run_local(v4_model, games=20)
    print("\n[v4] agent_stats:", json.dumps(stats, indent=2))
    assert stats["checkpoint_load_fallback"] is False
    assert stats["inference_exception_fallback"] == 0
    assert stats["fallback_rate"] < 0.02, stats


def test_v4_no_unknown_legal_fallback(v4_model: Path) -> None:
    stats = _run_local(v4_model, games=20)
    assert stats["no_known_legal_fallback"] == 0, (
        "no_known_legal_fallback should be 0 when type-level discard routing is active"
    )


def test_fallback_only_baseline_is_all_heuristic() -> None:
    stats = _run_local(None, games=5)
    print("\n[fallback-only] agent_stats:", json.dumps(stats, indent=2))
    assert stats["primary_decisions"] == 0
    assert stats["heuristic_fallback_decisions"] == stats["total_decisions"]


def test_v4_beats_fallback_only_on_primary_rate(v4_model: Path) -> None:
    v4_stats = _run_local(v4_model, games=10)
    fb_stats = _run_local(None, games=10)
    v4_primary_rate = v4_stats["primary_decisions"] / max(v4_stats["total_decisions"], 1)
    fb_primary_rate = fb_stats["primary_decisions"] / max(fb_stats["total_decisions"], 1)
    assert v4_primary_rate > 0.95
    assert v4_primary_rate > fb_primary_rate
