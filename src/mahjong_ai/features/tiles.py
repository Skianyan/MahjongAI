"""Tile string helpers for discard heuristics and offline metrics."""

from __future__ import annotations

import json
from functools import lru_cache

from riichienv import Action, ActionType

# First-copy tile ids for red fives in riichienv (pai ends with 'r').
RED_FIVE_TILE_IDS: frozenset[int] = frozenset({16, 52, 88})
HONOR_PAI_SUFFIXES: frozenset[str] = frozenset({"E", "S", "W", "N", "P", "F", "C"})
TERMINAL_PAI: frozenset[str] = frozenset(
    {"1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"}
)

N_TILE_TYPES = 34


@lru_cache(maxsize=1)
def pai_base_to_type_index() -> dict[str, int]:
    """Map MJAI pai base (``5m``, ``5mr`` -> ``5m``) to 0..33 type index."""
    mapping: dict[str, int] = {}
    for tile_id in range(136):
        try:
            pai = tile_id_to_pai(tile_id)
        except (ValueError, json.JSONDecodeError):
            continue
        base = pai[:-1] if pai.endswith("r") else pai
        mapping.setdefault(base, len(mapping))
    if len(mapping) != N_TILE_TYPES:
        raise RuntimeError(f"Expected {N_TILE_TYPES} tile types, got {len(mapping)}")
    return mapping


def tile_id_to_type_index(tile_id: int) -> int:
    pai = tile_id_to_pai(tile_id)
    base = pai[:-1] if pai.endswith("r") else pai
    return pai_base_to_type_index()[base]


@lru_cache(maxsize=136)
def tile_id_to_pai(tile_id: int) -> str:
    """Map a riichienv tile id to MJAI ``pai`` (e.g. ``5mr``, ``3p``)."""
    payload = json.loads(Action(ActionType.DISCARD, tile_id).to_mjai())
    pai = payload.get("pai")
    if not isinstance(pai, str):
        raise ValueError(f"Discard action for tile_id={tile_id} has no pai field")
    return pai


def is_red_five_tile(tile_id: int | None) -> bool:
    if tile_id is None:
        return False
    if tile_id in RED_FIVE_TILE_IDS:
        return True
    pai = tile_id_to_pai(tile_id)
    return "5" in pai and pai.endswith("r")


def is_honor_pai(pai: str) -> bool:
    return len(pai) == 1 and pai in HONOR_PAI_SUFFIXES


def is_terminal_pai(pai: str) -> bool:
    return pai in TERMINAL_PAI


def discard_priority_score(tile_id: int | None) -> float:
    """Lower score means safer to discard in fallback heuristic."""
    if tile_id is None:
        return 0.0
    if is_red_five_tile(tile_id):
        return 1000.0
    pai = tile_id_to_pai(tile_id)
    if is_honor_pai(pai):
        return 1.0
    if is_terminal_pai(pai):
        return 2.0
    if pai.startswith("5"):
        return 50.0
    return 10.0
