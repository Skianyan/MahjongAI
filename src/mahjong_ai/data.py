"""Streaming MJAI replay loading for supervised training and evaluation."""

from __future__ import annotations

import gzip
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from riichienv import Action, Observation, RiichiEnv

from mahjong_ai.features import ActionVocabulary, EncodedDecision, encode_decision
from mahjong_ai.features.actions import ACTION_TYPE_NAMES


REPLAY_SUFFIXES = {".json", ".jsonl", ".mjson", ".mjai", ".gz", ".zip"}


@dataclass(frozen=True, slots=True)
class SupervisedExample:
    """One expert decision recovered from an MJAI replay."""

    observation: Observation
    action: Action
    event: dict[str, Any]
    source: Path
    weight: float = 1.0

    def encoded_decision(
        self,
        vocabulary: ActionVocabulary,
        *,
        extended: bool = False,
        add_missing_actions: bool = False,
    ) -> EncodedDecision:
        return encode_decision(
            self.observation,
            vocabulary,
            action=self.action,
            extended=extended,
            add_missing_actions=add_missing_actions,
        )


def iter_supervised_examples(
    path: Path,
    *,
    action_types: frozenset[str] | None = None,
    strict: bool = True,
    example_weighting: bool = False,
) -> Iterator[SupervisedExample]:
    """Yield expert decisions from one MJAI file, archive, or directory."""
    for replay_path, events in iter_mjai_replays(path, strict=strict):
        event_list = list(events)
        actor_weights = (
            compute_replay_actor_weights(event_list) if example_weighting else None
        )
        yield from _iter_replay_examples(
            replay_path,
            event_list,
            action_types=action_types,
            strict=strict,
            actor_weights=actor_weights,
        )


def iter_supervised_examples_from_paths(
    paths: Iterable[Path],
    *,
    action_types: frozenset[str] | None = None,
    strict: bool = True,
    example_weighting: bool = False,
) -> Iterator[SupervisedExample]:
    """Yield examples from an explicit list of replay paths."""
    for replay_path in paths:
        yield from iter_supervised_examples(
            replay_path,
            action_types=action_types,
            strict=strict,
            example_weighting=example_weighting,
        )


def list_replay_paths(path: Path) -> tuple[Path, ...]:
    """Return replay files under *path* in stable sorted order."""
    if path.is_dir():
        return tuple(sorted(_iter_replay_files(path)))
    return (path,)


def split_replay_paths(
    path: Path,
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Split replay files into train/validation subsets deterministically."""
    replay_paths = list(list_replay_paths(path))
    if not replay_paths:
        return (), ()
    if validation_ratio <= 0:
        return tuple(replay_paths), ()
    if validation_ratio >= 1:
        raise ValueError("validation_ratio must be in [0, 1)")

    rng = random.Random(seed)
    rng.shuffle(replay_paths)

    if len(replay_paths) == 1:
        return tuple(replay_paths), ()

    validation_count = max(1, int(len(replay_paths) * validation_ratio))
    validation_paths = tuple(sorted(replay_paths[:validation_count]))
    train_paths = tuple(sorted(replay_paths[validation_count:]))
    if not train_paths:
        train_paths = validation_paths
        validation_paths = ()
    return train_paths, validation_paths


def iter_mjai_replays(
    path: Path,
    *,
    strict: bool = True,
) -> Iterator[tuple[Path, Iterable[dict[str, Any]]]]:
    """Yield event iterables for MJAI logs below *path* without extracting archives."""
    if path.is_dir():
        for replay_path in sorted(_iter_replay_files(path)):
            yield from iter_mjai_replays(replay_path, strict=strict)
        return

    if path.suffix == ".zip":
        try:
            with ZipFile(path) as archive:
                for name in archive.namelist():
                    member_path = Path(name)
                    if member_path.suffix not in REPLAY_SUFFIXES:
                        continue
                    with archive.open(name) as member:
                        text = (line.decode("utf-8") for line in member)
                        yield path / name, _parse_mjai_events(text)
        except Exception:
            if strict:
                raise
        return

    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as replay_file:
                yield path, _parse_mjai_events(replay_file)
        else:
            with path.open("r", encoding="utf-8") as replay_file:
                yield path, _parse_mjai_events(replay_file)
    except Exception:
        if strict:
            raise


def _iter_replay_files(path: Path) -> Iterator[Path]:
    for replay_path in path.rglob("*"):
        if replay_path.is_file() and replay_path.suffix in REPLAY_SUFFIXES:
            yield replay_path


def _parse_mjai_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    buffer: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        buffer.append(stripped)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        yield from _events_from_json(parsed)
        buffer.clear()

    if buffer:
        parsed = json.loads("\n".join(buffer))
        yield from _events_from_json(parsed)


def _events_from_json(parsed: Any) -> Iterator[dict[str, Any]]:
    if isinstance(parsed, dict):
        if isinstance(parsed.get("log"), list):
            yield from _events_from_json(parsed["log"])
        else:
            yield parsed
        return

    if isinstance(parsed, list):
        for item in parsed:
            yield from _events_from_json(item)


def compute_replay_actor_weights(events: Iterable[dict[str, Any]]) -> dict[int, float]:
    """Higher weight for seats that finished the replay with better final scores."""
    final_scores: list[int] | None = None
    for event in events:
        if event.get("type") == "end_game":
            scores = event.get("scores")
            if isinstance(scores, list) and len(scores) == 4:
                final_scores = [int(score) for score in scores]
    if final_scores is None:
        return {seat: 1.0 for seat in range(4)}

    ranked = sorted(range(4), key=lambda seat: final_scores[seat], reverse=True)
    weights: dict[int, float] = {}
    for rank, seat in enumerate(ranked):
        weights[seat] = 1.0 + 0.25 * (3 - rank)
    return weights


def _iter_replay_examples(
    source: Path,
    events: Iterable[dict[str, Any]],
    *,
    action_types: frozenset[str] | None,
    strict: bool,
    actor_weights: dict[int, float] | None = None,
) -> Iterator[SupervisedExample]:
    env = RiichiEnv()

    for event in events:
        actor = event.get("actor")
        if isinstance(actor, int):
            try:
                observation = env.get_observation(actor)
                if observation is not None and observation.legal_actions():
                    action = observation.select_action_from_mjai(event)
                    if action is not None and _include_action(action, action_types):
                        weight = 1.0
                        if actor_weights is not None:
                            weight = actor_weights.get(actor, 1.0)
                        yield SupervisedExample(
                            observation=observation,
                            action=action,
                            event=event,
                            source=source,
                            weight=weight,
                        )
            except Exception:
                if strict:
                    raise

        try:
            env.apply_event(event)
        except Exception:
            if strict:
                raise
            return


def _include_action(action: Action, action_types: frozenset[str] | None) -> bool:
    if action_types is None:
        return True
    return _action_type_name(action) in action_types


def _action_type_name(action: Action) -> str:
    return ACTION_TYPE_NAMES.get(int(action.action_type), str(action.action_type).upper())
