"""Observation and decision feature encoding wrappers."""

from __future__ import annotations

from dataclasses import dataclass

from riichienv import Action, Observation

from mahjong_ai.features.actions import ActionVocabulary, LegalActionMask


FLOAT32_DTYPE = "float32"
FLOAT32_NBYTES = 4
BASE_OBSERVATION_SHAPE = (74, 34)
EXTENDED_OBSERVATION_SHAPE = (215, 34)


@dataclass(frozen=True, slots=True)
class ObservationFeatures:
    """Raw float32 observation tensor plus shape metadata."""

    data: bytes
    shape: tuple[int, int]
    dtype: str = FLOAT32_DTYPE

    def __post_init__(self) -> None:
        expected_nbytes = self.shape[0] * self.shape[1] * FLOAT32_NBYTES
        if len(self.data) != expected_nbytes:
            raise ValueError(f"Expected {expected_nbytes} feature bytes, got {len(self.data)}")

    @property
    def nbytes(self) -> int:
        return len(self.data)

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]


@dataclass(frozen=True, slots=True)
class EncodedDecision:
    """Feature tensor, legal action mask, and optional supervised label."""

    features: ObservationFeatures
    legal_actions: LegalActionMask
    label: int | None = None

    def require_label(self) -> int:
        if self.label is None:
            raise ValueError("Encoded decision does not include a supervised label")
        return self.label


def encode_observation_bytes(observation: Observation) -> bytes:
    """Return riichienv's compact float32 observation tensor bytes."""
    return observation.encode()


def encode_observation(
    observation: Observation,
    *,
    extended: bool = False,
) -> ObservationFeatures:
    """Encode an observation as a shape-aware float32 tensor payload."""
    if extended:
        return ObservationFeatures(
            data=observation.encode_extended(),
            shape=EXTENDED_OBSERVATION_SHAPE,
        )
    return ObservationFeatures(
        data=encode_observation_bytes(observation),
        shape=BASE_OBSERVATION_SHAPE,
    )


def encode_decision(
    observation: Observation,
    vocabulary: ActionVocabulary,
    *,
    action: Action | None = None,
    extended: bool = False,
    add_missing_actions: bool = False,
) -> EncodedDecision:
    """Encode model inputs and, when supplied, the expert action label."""
    legal_actions = vocabulary.mask_for(observation, add_missing=add_missing_actions)
    label = None
    if action is not None:
        label = vocabulary.encode_or_add(action) if add_missing_actions else vocabulary.encode(action)

    return EncodedDecision(
        features=encode_observation(observation, extended=extended),
        legal_actions=legal_actions,
        label=label,
    )
