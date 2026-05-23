"""Integration smoke tests (run: pytest tests/ -q)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
V4_MODEL = REPO_ROOT / "models" / "allyears_v4_no_discard_downweight.pt"
SAMPLE_REPLAY = REPO_ROOT / "data" / "2025" / "2025010100gm-00a9-0000-0c603e8f.mjson"


@pytest.fixture(scope="module")
def v4_model() -> Path:
    if not V4_MODEL.exists():
        pytest.skip(f"missing checkpoint: {V4_MODEL}")
    return V4_MODEL


def test_imports() -> None:
    import mahjong_ai.agents  # noqa: F401
    import mahjong_ai.evaluation.discard_metrics  # noqa: F401
    import mahjong_ai.features.tiles  # noqa: F401


def test_red_five_tile_detection() -> None:
    from mahjong_ai.features.tiles import is_red_five_tile

    assert is_red_five_tile(16)


def test_load_v4_checkpoint(v4_model: Path) -> None:
    from mahjong_ai.agents import load_checkpoint_agent

    agent = load_checkpoint_agent(v4_model, device="cpu")
    assert agent.vocabulary is not None


def test_offline_eval_smoke(v4_model: Path) -> None:
    if not SAMPLE_REPLAY.exists():
        pytest.skip(f"missing replay: {SAMPLE_REPLAY}")
    from mahjong_ai.evaluation.evaluate import OfflineEvaluationOptions, evaluate_offline

    result = evaluate_offline(
        OfflineEvaluationOptions(
            data_path=SAMPLE_REPLAY,
            model_path=v4_model,
            max_examples=100,
        )
    )
    assert result.overall["examples"] > 0
    assert result.discard_metrics is not None


def test_local_eval_low_fallback(v4_model: Path) -> None:
    from mahjong_ai.evaluation.evaluate import LocalEvaluationOptions, evaluate_local

    result = evaluate_local(
        LocalEvaluationOptions(
            model_path=v4_model,
            games=2,
            timeout_seconds=10.0,
            max_actions_per_game=500,
            fail_on_fallback=True,
        )
    )
    assert result.agent_stats is not None
    assert result.agent_stats["fallback_rate"] < 0.05


def test_example_weighting() -> None:
    if not SAMPLE_REPLAY.exists():
        pytest.skip(f"missing replay: {SAMPLE_REPLAY}")
    from mahjong_ai.data import iter_supervised_examples_from_paths

    examples = list(
        iter_supervised_examples_from_paths((SAMPLE_REPLAY,), example_weighting=True)
    )
    assert examples
    assert examples[0].weight > 0
