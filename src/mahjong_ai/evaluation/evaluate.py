"""Offline policy metrics and local RiichiEnv evaluation entry point."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from riichienv import RiichiEnv

from mahjong_ai.agents import (
    Agent,
    FallbackAgent,
    SafeAgent,
    get_agent_stats,
    load_checkpoint_agent,
    masked_top_action_id,
    top_k_action_ids,
)
from mahjong_ai.agents.stats import AgentStats
from mahjong_ai.config import load_config
from mahjong_ai.data import iter_supervised_examples
from mahjong_ai.evaluation.discard_metrics import (
    DiscardMetricCounts,
    discard_metrics_notes,
    is_discard_action,
    select_model_discard,
)
from mahjong_ai.features import UnknownActionError
from mahjong_ai.features.actions import ACTION_TYPE_NAMES, ActionSpec


@dataclass(frozen=True, slots=True)
class OfflineEvaluationOptions:
    data_path: Path
    model_path: Path
    device: str = "auto"
    top_k: int = 5
    strict: bool = True
    action_types: frozenset[str] | None = None
    max_examples: int | None = None


@dataclass(slots=True)
class MetricCounts:
    examples: int = 0
    top1_correct: int = 0
    topk_correct: int = 0
    raw_top1_illegal: int = 0
    unknown_labels: int = 0
    no_known_legal_actions: int = 0
    unknown_legal_actions: int = 0
    examples_with_unknown_legal_actions: int = 0

    def record(
        self,
        *,
        label: int | None,
        raw_top1_legal: bool,
        masked_top_ids: tuple[int, ...],
        unknown_legal_actions: int,
    ) -> None:
        if unknown_legal_actions:
            self.unknown_legal_actions += unknown_legal_actions
            self.examples_with_unknown_legal_actions += 1
        if label is None:
            self.unknown_labels += 1
            return

        self.examples += 1
        if not raw_top1_legal:
            self.raw_top1_illegal += 1
        if not masked_top_ids:
            self.no_known_legal_actions += 1
            return
        if masked_top_ids[0] == label:
            self.top1_correct += 1
        if label in masked_top_ids:
            self.topk_correct += 1

    def rates(self) -> dict[str, float | int]:
        denominator = max(self.examples, 1)
        return {
            "examples": self.examples,
            "top1_accuracy": self.top1_correct / denominator,
            "topk_accuracy": self.topk_correct / denominator,
            "raw_top1_illegal_rate": self.raw_top1_illegal / denominator,
            "unknown_labels": self.unknown_labels,
            "no_known_legal_actions": self.no_known_legal_actions,
            "unknown_legal_actions": self.unknown_legal_actions,
            "examples_with_unknown_legal_actions": self.examples_with_unknown_legal_actions,
            "unknown_legal_action_rate": self.examples_with_unknown_legal_actions / denominator,
        }


@dataclass(frozen=True, slots=True)
class OfflineEvaluationResult:
    model_path: Path
    data_path: Path
    top_k: int
    overall: dict[str, float | int]
    by_action_type: dict[str, dict[str, float | int]]
    discard_metrics: dict[str, float | int] | None = None
    notes: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "model_path": str(self.model_path),
            "data_path": str(self.data_path),
            "top_k": self.top_k,
            "overall": self.overall,
            "by_action_type": self.by_action_type,
            "notes": self.notes,
        }
        if self.discard_metrics is not None:
            payload["discard_metrics"] = self.discard_metrics
        return payload


@dataclass(frozen=True, slots=True)
class LocalEvaluationOptions:
    model_path: Path | None
    games: int
    timeout_seconds: float
    device: str = "auto"
    game_mode: str = "4p-red-single"
    controlled_seat: int | None = 0
    opponent: str = "random"
    max_actions_per_game: int = 5000
    compare_fallback: bool = False
    fail_on_fallback: bool = False


@dataclass(frozen=True, slots=True)
class LocalEvaluationResult:
    games: int
    controlled_seat: int
    game_mode: str
    mean_score: float
    mean_rank: float
    first_place_rate: float
    tie_rate: float
    unfinished_game_rate: float
    action_count: int
    mean_action_seconds: float
    max_action_seconds: float
    timeout_actions: int
    seat_metrics: dict[str, dict[str, float | int]]
    comparison: dict[str, float | int] | None = None
    agent_stats: dict[str, float | int | bool] | None = None
    warnings: list[str] = field(default_factory=list)
    scores: list[list[int]] = field(default_factory=list)
    ranks: list[list[int]] = field(default_factory=list)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TournamentOptions:
    model_path: Path
    games: int
    timeout_seconds: float
    device: str = "auto"
    game_mode: str = "4p-red-single"
    baseline_model_path: Path | None = None
    max_actions_per_game: int = 5000


@dataclass(frozen=True, slots=True)
class TournamentResult:
    game_mode: str
    games: int
    variants: dict[str, dict[str, Any]]

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_offline(options: OfflineEvaluationOptions) -> OfflineEvaluationResult:
    agent = load_checkpoint_agent(options.model_path, device=options.device)
    overall = MetricCounts()
    by_action_type: dict[str, MetricCounts] = {}
    discard_metrics = DiscardMetricCounts()
    for example_index, example in enumerate(
        iter_supervised_examples(
            Path(options.data_path),
            action_types=options.action_types,
            strict=options.strict,
        )
    ):
        if options.max_examples is not None and example_index >= options.max_examples:
            break
        scores = agent.scores(example.observation)
        legal_mask = agent.vocabulary.mask_for(example.observation)
        raw_top1 = top_k_action_ids(scores, 1)
        raw_top1_legal = bool(raw_top1 and legal_mask.is_legal(raw_top1[0]))
        masked_top_ids = _masked_top_k_action_ids(scores, legal_mask.legal_indices, options.top_k)
        try:
            label = agent.vocabulary.encode(example.action)
        except UnknownActionError:
            label = None
        action_type = ACTION_TYPE_NAMES.get(
            int(example.action.action_type),
            str(example.action.action_type).upper(),
        )
        overall.record(
            label=label,
            raw_top1_legal=raw_top1_legal,
            masked_top_ids=masked_top_ids,
            unknown_legal_actions=len(legal_mask.unknown_actions),
        )
        by_action_type.setdefault(action_type, MetricCounts()).record(
            label=label,
            raw_top1_legal=raw_top1_legal,
            masked_top_ids=masked_top_ids,
            unknown_legal_actions=len(legal_mask.unknown_actions),
        )
        if is_discard_action(example.action):
            model_discard = select_model_discard(
                example.observation,
                agent.vocabulary,
                list(scores),
            )
            if model_discard is not None:
                discard_metrics.record(
                    observation=example.observation,
                    expert_action=example.action,
                    model_action=model_discard,
                    match=ActionSpec.from_action(model_discard) == ActionSpec.from_action(example.action),
                )
    notes = ["top1/topk son metricas de imitacion, no de fuerza real."]
    notes.extend(discard_metrics_notes())
    if overall.examples_with_unknown_legal_actions:
        notes.append("se detectaron acciones legales fuera del vocabulario del checkpoint.")
    return OfflineEvaluationResult(
        model_path=options.model_path,
        data_path=options.data_path,
        top_k=options.top_k,
        overall=overall.rates(),
        by_action_type={key: value.rates() for key, value in sorted(by_action_type.items())},
        discard_metrics=discard_metrics.rates() if discard_metrics.discard_examples else None,
        notes=notes,
    )


def evaluate_local(options: LocalEvaluationOptions) -> LocalEvaluationResult:
    if options.model_path is not None:
        primary = load_checkpoint_agent(options.model_path, device=options.device)
        stats = get_agent_stats(primary) or AgentStats()
        fallback = FallbackAgent(stats=stats)
        controlled_agent: Agent = SafeAgent(primary=primary, fallback=fallback, stats=stats)
    else:
        stats = AgentStats()
        controlled_agent = FallbackAgent(stats=stats)
    opponent_agents = _build_opponents(options.opponent)

    all_scores: list[list[int]] = []
    all_ranks: list[list[int]] = []
    action_times: list[float] = []
    timeout_actions = 0
    unfinished_games = 0
    tie_games = 0
    seat_scores: dict[int, list[int]] = {seat: [] for seat in range(4)}
    seat_ranks: dict[int, list[int]] = {seat: [] for seat in range(4)}

    for game_index in range(options.games):
        controlled_seat = options.controlled_seat if options.controlled_seat is not None else game_index % 4
        env = RiichiEnv(game_mode=options.game_mode)
        obs_dict = env.reset()
        action_steps = 0
        while not env.done():
            action_steps += 1
            if action_steps > options.max_actions_per_game:
                unfinished_games += 1
                break
            actions = {}
            for player_id, observation in obs_dict.items():
                if player_id == controlled_seat:
                    start = time.perf_counter()
                    action = controlled_agent.act(observation)
                    elapsed = time.perf_counter() - start
                    action_times.append(elapsed)
                    if elapsed > options.timeout_seconds:
                        timeout_actions += 1
                else:
                    action = opponent_agents[player_id].act(observation)
                actions[player_id] = action
            obs_dict = env.step(actions)

        scores = [int(score) for score in env.scores()]
        ranks = [int(rank) for rank in env.ranks()]
        if len(set(scores)) == 1:
            tie_games += 1
        all_scores.append(scores)
        all_ranks.append(ranks)
        for seat in range(4):
            seat_scores[seat].append(scores[seat])
            seat_ranks[seat].append(ranks[seat])

    selected_seat = options.controlled_seat if options.controlled_seat is not None else 0
    selected_scores = [scores[selected_seat] for scores in all_scores]
    selected_ranks = [ranks[selected_seat] for ranks in all_ranks]
    first_place_games = sum(1 for ranks in all_ranks if ranks[selected_seat] == min(ranks))
    seat_metrics = {
        str(seat): {
            "mean_score": mean(seat_scores[seat]) if seat_scores[seat] else 0.0,
            "mean_rank": mean(seat_ranks[seat]) if seat_ranks[seat] else 0.0,
            "first_place_rate": (
                sum(1 for ranks in all_ranks if ranks[seat] == min(ranks)) / max(len(all_ranks), 1)
            ),
        }
        for seat in range(4)
    }
    warnings: list[str] = []
    if tie_games > 0:
        warnings.append("se detectaron partidas con scores iguales entre todos los asientos.")
    if unfinished_games > 0:
        warnings.append("partidas cortadas por max_actions_per_game.")

    agent_stats_mapping = stats.to_mapping() if stats is not None else None
    if options.fail_on_fallback and stats is not None and stats.fallback_decisions() > 0:
        raise RuntimeError(
            f"Local evaluation used fallback for {stats.fallback_decisions()} decisions "
            f"(rate={stats.fallback_rate():.4f})."
        )

    comparison = None
    if options.compare_fallback:
        baseline = evaluate_local(
            LocalEvaluationOptions(
                model_path=None,
                games=options.games,
                timeout_seconds=options.timeout_seconds,
                device=options.device,
                game_mode=options.game_mode,
                controlled_seat=options.controlled_seat,
                opponent=options.opponent,
                max_actions_per_game=options.max_actions_per_game,
            )
        )
        comparison = {
            "baseline": "fallback",
            "delta_mean_score": (mean(selected_scores) if selected_scores else 0.0) - baseline.mean_score,
            "delta_mean_rank": (mean(selected_ranks) if selected_ranks else 0.0) - baseline.mean_rank,
            "delta_first_place_rate": (
                first_place_games / max(len(selected_ranks), 1)
            )
            - baseline.first_place_rate,
        }

    return LocalEvaluationResult(
        games=options.games,
        controlled_seat=selected_seat,
        game_mode=options.game_mode,
        mean_score=mean(selected_scores) if selected_scores else 0.0,
        mean_rank=mean(selected_ranks) if selected_ranks else 0.0,
        first_place_rate=first_place_games / max(len(selected_ranks), 1),
        tie_rate=tie_games / max(len(all_scores), 1),
        unfinished_game_rate=unfinished_games / max(options.games, 1),
        action_count=len(action_times),
        mean_action_seconds=mean(action_times) if action_times else 0.0,
        max_action_seconds=max(action_times, default=0.0),
        timeout_actions=timeout_actions,
        seat_metrics=seat_metrics,
        comparison=comparison,
        agent_stats=agent_stats_mapping,
        warnings=warnings,
        scores=all_scores,
        ranks=all_ranks,
    )


def evaluate_tournament(options: TournamentOptions) -> TournamentResult:
    variants: dict[str, dict[str, Any]] = {}
    variants["candidate_vs_random"] = evaluate_local(
        LocalEvaluationOptions(
            model_path=options.model_path,
            games=options.games,
            timeout_seconds=options.timeout_seconds,
            device=options.device,
            game_mode=options.game_mode,
            controlled_seat=None,
            opponent="random",
            max_actions_per_game=options.max_actions_per_game,
        )
    ).to_mapping()
    variants["fallback_vs_random"] = evaluate_local(
        LocalEvaluationOptions(
            model_path=None,
            games=options.games,
            timeout_seconds=options.timeout_seconds,
            device=options.device,
            game_mode=options.game_mode,
            controlled_seat=None,
            opponent="random",
            max_actions_per_game=options.max_actions_per_game,
        )
    ).to_mapping()
    if options.baseline_model_path is not None:
        variants["baseline_model_vs_random"] = evaluate_local(
            LocalEvaluationOptions(
                model_path=options.baseline_model_path,
                games=options.games,
                timeout_seconds=options.timeout_seconds,
                device=options.device,
                game_mode=options.game_mode,
                controlled_seat=None,
                opponent="random",
                max_actions_per_game=options.max_actions_per_game,
            )
        ).to_mapping()
    return TournamentResult(game_mode=options.game_mode, games=options.games, variants=variants)


def _masked_top_k_action_ids(
    scores: Sequence[float],
    legal_indices: tuple[int, ...],
    k: int,
) -> tuple[int, ...]:
    if k <= 0 or not legal_indices:
        return ()
    first = masked_top_action_id(scores, legal_indices)
    ordered = tuple(
        sorted(legal_indices, key=lambda action_id: scores[action_id], reverse=True)[:k]
    )
    if first is not None and ordered and ordered[0] != first:
        raise AssertionError("Masked top-k ordering is inconsistent")
    return ordered


def _build_opponents(kind: str) -> dict[int, Agent]:
    if kind == "fallback":
        return {player_id: FallbackAgent() for player_id in range(4)}
    try:
        from riichienv.agents import RandomAgent
    except ModuleNotFoundError as exc:
        raise RuntimeError("riichienv.agents.RandomAgent is required for random opponents") from exc
    return {player_id: RandomAgent() for player_id in range(4)}


def _parse_action_types(values: list[str] | None) -> frozenset[str] | None:
    if not values:
        return None
    return frozenset(value.upper() for value in values)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None, help="TOML config path")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    subparsers = parser.add_subparsers(dest="command", required=True)

    offline = subparsers.add_parser("offline", help="Evaluate checkpoint on MJAI replay examples")
    offline.add_argument("--data", type=Path, default=config.data.raw_dir, help="MJAI replay path")
    offline.add_argument("--model", type=Path, default=config.model.artifact_path, help="Checkpoint path")
    offline.add_argument("--device", default=config.training.device)
    offline.add_argument("--top-k", type=int, default=5)
    offline.add_argument("--max-examples", type=int, default=None)
    offline.add_argument("--action-type", action="append", dest="action_types")
    offline.add_argument("--skip-bad-replays", action="store_true")

    local = subparsers.add_parser("local", help="Run local RiichiEnv games")
    local.add_argument("--model", type=Path, default=config.model.artifact_path, help="Checkpoint path")
    local.add_argument("--fallback-only", action="store_true")
    local.add_argument("--games", type=int, default=config.evaluation.games)
    local.add_argument("--timeout-seconds", type=float, default=config.evaluation.timeout_seconds)
    local.add_argument("--device", default=config.training.device)
    local.add_argument("--game-mode", default="4p-red-single")
    local.add_argument("--controlled-seat", type=int, default=0)
    local.add_argument("--all-seats", action="store_true")
    local.add_argument("--opponent", choices=("random", "fallback"), default="random")
    local.add_argument("--max-actions-per-game", type=int, default=5000)
    local.add_argument("--compare-fallback", action="store_true")
    local.add_argument(
        "--fail-on-fallback",
        action="store_true",
        help="Abort local evaluation if any decision used fallback",
    )

    tournament = subparsers.add_parser("tournament", help="Run local baseline tournament")
    tournament.add_argument("--model", type=Path, default=config.model.artifact_path)
    tournament.add_argument("--baseline-model", type=Path, default=None)
    tournament.add_argument("--games", type=int, default=config.evaluation.games)
    tournament.add_argument("--timeout-seconds", type=float, default=config.evaluation.timeout_seconds)
    tournament.add_argument("--device", default=config.training.device)
    tournament.add_argument("--game-mode", default="4p-red-single")
    tournament.add_argument("--max-actions-per-game", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "offline":
        result = evaluate_offline(
            OfflineEvaluationOptions(
                data_path=args.data,
                model_path=args.model,
                device=args.device,
                top_k=args.top_k,
                strict=not args.skip_bad_replays,
                action_types=_parse_action_types(args.action_types),
                max_examples=args.max_examples,
            )
        )
    elif args.command == "local":
        result = evaluate_local(
            LocalEvaluationOptions(
                model_path=None if args.fallback_only else args.model,
                games=args.games,
                timeout_seconds=args.timeout_seconds,
                device=args.device,
                game_mode=args.game_mode,
                controlled_seat=None if args.all_seats else args.controlled_seat,
                opponent=args.opponent,
                max_actions_per_game=args.max_actions_per_game,
                compare_fallback=args.compare_fallback,
                fail_on_fallback=args.fail_on_fallback,
            )
        )
    else:
        result = evaluate_tournament(
            TournamentOptions(
                model_path=args.model,
                baseline_model_path=args.baseline_model,
                games=args.games,
                timeout_seconds=args.timeout_seconds,
                device=args.device,
                game_mode=args.game_mode,
                max_actions_per_game=args.max_actions_per_game,
            )
        )

    print(json.dumps(result.to_mapping(), indent=2))


if __name__ == "__main__":
    main()
