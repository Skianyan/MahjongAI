"""Feature encoding utilities."""

from mahjong_ai.features.actions import (
    ActionSpec,
    ActionVocabulary,
    LegalActionMask,
    UnknownActionError,
)
from mahjong_ai.features.encoding import (
    BASE_OBSERVATION_SHAPE,
    EXTENDED_OBSERVATION_SHAPE,
    EncodedDecision,
    ObservationFeatures,
    encode_decision,
    encode_observation,
    encode_observation_bytes,
)

__all__ = [
    "BASE_OBSERVATION_SHAPE",
    "EXTENDED_OBSERVATION_SHAPE",
    "ActionSpec",
    "ActionVocabulary",
    "EncodedDecision",
    "LegalActionMask",
    "ObservationFeatures",
    "UnknownActionError",
    "encode_decision",
    "encode_observation",
    "encode_observation_bytes",
]
