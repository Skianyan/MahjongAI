"""Supervised baseline policy training entry point."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mahjong_ai.config import load_config
from mahjong_ai.data import iter_supervised_examples
from mahjong_ai.features import ActionVocabulary
from mahjong_ai.features.encoding import BASE_OBSERVATION_SHAPE, EXTENDED_OBSERVATION_SHAPE


@dataclass(frozen=True, slots=True)
class TrainOptions:
    """Runtime options for one supervised baseline training run."""

    data_path: Path
    output_path: Path
    batch_size: int
    epochs: int
    device: str
    learning_rate: float
    hidden_size: int
    model_type: str = "auto"
    extended: bool = False
    strict: bool = True
    action_types: frozenset[str] | None = None
    max_examples: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Summary returned after writing the checkpoint."""

    output_path: Path
    examples: int
    action_count: int
    final_loss: float


def train_baseline(options: TrainOptions) -> TrainingResult:
    """Train the requested supervised baseline policy and save a checkpoint."""
    model_type = _resolve_model_type(options.model_type)
    if model_type == "action-prior":
        return train_action_prior_baseline(options)
    return train_mlp_baseline(options)


def train_action_prior_baseline(options: TrainOptions) -> TrainingResult:
    """Train a simple supervised action-prior policy without optional ML dependencies."""
    vocabulary = ActionVocabulary()
    action_counts: list[int] = []
    example_count = 0

    for example_index, example in enumerate(_iter_examples(options)):
        if options.max_examples is not None and example_index >= options.max_examples:
            break

        for action in example.observation.legal_actions():
            _ensure_count_slot(action_counts, vocabulary.add(action))
        label = vocabulary.add(example.action)
        _ensure_count_slot(action_counts, label)
        action_counts[label] += 1
        example_count += 1

    if example_count == 0:
        raise RuntimeError(f"No supervised examples found under {options.data_path}")
    if len(vocabulary) == 0:
        raise RuntimeError(f"No legal actions found while scanning {options.data_path}")

    total_actions = sum(action_counts)
    action_count = len(vocabulary)
    # Laplace smoothing keeps every vocabulary action selectable when masked legal.
    log_priors = [
        math.log((count + 1) / (total_actions + action_count)) for count in action_counts
    ]
    final_loss = _negative_log_likelihood(action_counts, log_priors, total_actions)
    input_shape = EXTENDED_OBSERVATION_SHAPE if options.extended else BASE_OBSERVATION_SHAPE
    checkpoint = {
        "format_version": 1,
        "model_type": "action_prior",
        "action_vocabulary": vocabulary.to_mapping(),
        "feature_schema": {
            "dtype": "float32",
            "extended": options.extended,
            "shape": list(input_shape),
        },
        "model": {
            "action_counts": action_counts,
            "log_priors": log_priors,
        },
        "training": _training_metadata(options, "cpu", example_count, [{"loss": final_loss}]),
    }

    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    with options.output_path.open("w", encoding="utf-8") as checkpoint_file:
        json.dump(checkpoint, checkpoint_file, indent=2)
        checkpoint_file.write("\n")

    return TrainingResult(
        output_path=options.output_path,
        examples=example_count,
        action_count=action_count,
        final_loss=final_loss,
    )


def train_mlp_baseline(options: TrainOptions) -> TrainingResult:
    """Train a compact masked PyTorch MLP policy and save a checkpoint."""
    torch, nn, dataloader_cls, iterable_dataset_cls = _import_torch_training()
    from mahjong_ai.training.policy import MLPPolicy, PolicyModelConfig, mask_illegal_logits

    class SupervisedDecisionDataset(iterable_dataset_cls):
        """Stream encoded MJAI decisions without materializing the replay corpus."""

        def __init__(self, dataset_options: TrainOptions, vocabulary: ActionVocabulary) -> None:
            super().__init__()
            self.options = dataset_options
            self.vocabulary = vocabulary

        def __iter__(self):
            for example_index, example in enumerate(_iter_examples(self.options)):
                if self.options.max_examples is not None and example_index >= self.options.max_examples:
                    break

                encoded = example.encoded_decision(self.vocabulary, extended=self.options.extended)
                features = torch.frombuffer(encoded.features.data, dtype=torch.float32).clone()
                features = features.reshape(encoded.features.shape)
                legal_mask = torch.tensor(encoded.legal_actions.mask, dtype=torch.bool)
                label = torch.tensor(encoded.require_label(), dtype=torch.long)
                yield features, legal_mask, label

    def train_one_epoch(
        model: Any,
        optimizer: Any,
        criterion: Any,
        dataset: Any,
        batch_size: int,
        device: Any,
    ) -> tuple[float, int]:
        model.train()
        total_loss = 0.0
        total_examples = 0
        loader = dataloader_cls(dataset, batch_size=batch_size)

        for features, legal_mask, labels in loader:
            features = features.to(device)
            legal_mask = legal_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = mask_illegal_logits(model(features), legal_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_examples = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_examples
            total_examples += batch_examples

        return total_loss / max(total_examples, 1), total_examples

    vocabulary, example_count = build_action_vocabulary(options)
    if example_count == 0:
        raise RuntimeError(f"No supervised examples found under {options.data_path}")
    if len(vocabulary) == 0:
        raise RuntimeError(f"No legal actions found while scanning {options.data_path}")

    device = torch.device(_resolve_device(options.device, torch))
    input_shape = EXTENDED_OBSERVATION_SHAPE if options.extended else BASE_OBSERVATION_SHAPE
    model_config = PolicyModelConfig(
        input_shape=input_shape,
        action_count=len(vocabulary),
        hidden_size=options.hidden_size,
    )
    model = MLPPolicy(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float | int]] = []
    for epoch in range(1, options.epochs + 1):
        average_loss, trained_examples = train_one_epoch(
            model,
            optimizer,
            criterion,
            SupervisedDecisionDataset(options, vocabulary),
            options.batch_size,
            device,
        )
        if trained_examples == 0:
            raise RuntimeError("The dataset produced no examples during training")
        history.append({"epoch": epoch, "loss": average_loss, "examples": trained_examples})
        print(f"epoch {epoch}/{options.epochs}: loss={average_loss:.4f} examples={trained_examples}")

    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_type": "mlp_policy",
        "model_config": model_config.to_mapping(),
        "state_dict": model.state_dict(),
        "action_vocabulary": vocabulary.to_mapping(),
        "feature_schema": {
            "dtype": "float32",
            "extended": options.extended,
            "shape": list(input_shape),
        },
        "training": _training_metadata(options, str(device), example_count, history),
    }
    torch.save(checkpoint, options.output_path)

    return TrainingResult(
        output_path=options.output_path,
        examples=example_count,
        action_count=len(vocabulary),
        final_loss=float(history[-1]["loss"]),
    )


def build_action_vocabulary(options: TrainOptions) -> tuple[ActionVocabulary, int]:
    """Scan examples once to create the stable output vocabulary."""
    vocabulary = ActionVocabulary()
    example_count = 0
    for example_index, example in enumerate(_iter_examples(options)):
        if options.max_examples is not None and example_index >= options.max_examples:
            break

        vocabulary.add_actions(example.observation.legal_actions())
        vocabulary.add(example.action)
        example_count += 1

    return vocabulary, example_count


def _iter_examples(options: TrainOptions):
    return iter_supervised_examples(
        options.data_path,
        action_types=options.action_types,
        strict=options.strict,
    )


def _ensure_count_slot(counts: list[int], action_id: int) -> None:
    while len(counts) <= action_id:
        counts.append(0)


def _negative_log_likelihood(counts: list[int], log_priors: list[float], total: int) -> float:
    if total == 0:
        return 0.0
    return -sum(count * log_priors[action_id] for action_id, count in enumerate(counts)) / total


def _training_metadata(
    options: TrainOptions,
    device: str,
    examples: int,
    history: list[dict[str, float | int]],
) -> dict[str, Any]:
    return {
        "data_path": str(options.data_path),
        "batch_size": options.batch_size,
        "epochs": options.epochs,
        "learning_rate": options.learning_rate,
        "device": device,
        "strict": options.strict,
        "action_types": sorted(options.action_types) if options.action_types else None,
        "max_examples": options.max_examples,
        "examples": examples,
        "history": history,
    }


def _resolve_model_type(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        _import_torch_training()
    except RuntimeError:
        print("torch is not available; training action-prior baseline instead")
        return "action-prior"
    return "mlp"


def _import_torch_training():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, IterableDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for --model-type mlp. Install the training extras "
            "with `pip install -e '.[train]'`, or use --model-type action-prior."
        ) from exc

    return torch, nn, DataLoader, IterableDataset


def _resolve_device(requested: str, torch_module: Any) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    parser.add_argument("--data", type=Path, default=config.data.raw_dir, help="MJAI replay path")
    parser.add_argument("--output", type=Path, default=config.model.artifact_path, help="Checkpoint path")
    parser.add_argument("--batch-size", type=int, default=config.training.batch_size)
    parser.add_argument("--epochs", type=int, default=config.training.epochs)
    parser.add_argument("--device", default=config.training.device)
    parser.add_argument("--learning-rate", type=float, default=config.training.learning_rate)
    parser.add_argument("--hidden-size", type=int, default=config.training.hidden_size)
    parser.add_argument(
        "--model-type",
        choices=("auto", "mlp", "action-prior"),
        default="auto",
        help="Baseline model to train. auto uses the MLP when PyTorch is installed.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--action-type",
        action="append",
        dest="action_types",
        help="Filter examples by action type name, e.g. DISCARD. Repeatable.",
    )
    parser.add_argument("--extended", action="store_true", help="Use extended riichienv features")
    parser.add_argument(
        "--skip-bad-replays",
        action="store_true",
        help="Skip replay files that riichienv cannot parse",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_baseline(
        TrainOptions(
            data_path=args.data,
            output_path=args.output,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            model_type=args.model_type,
            extended=args.extended,
            strict=not args.skip_bad_replays,
            action_types=_parse_action_types(args.action_types),
            max_examples=args.max_examples,
        )
    )
    print(
        f"saved {result.action_count}-action policy to {result.output_path} "
        f"from {result.examples} examples; final_loss={result.final_loss:.4f}"
    )


if __name__ == "__main__":
    main()
