"""Supervised baseline policy training entry point."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mahjong_ai.config import load_config
from mahjong_ai.data import (
    iter_supervised_examples_from_paths,
    list_replay_paths,
    split_replay_paths,
)
from mahjong_ai.features import ActionVocabulary, UnknownActionError
from mahjong_ai.features.encoding import (
    BASE_OBSERVATION_SHAPE,
    EXTENDED_OBSERVATION_SHAPE,
    encode_observation,
)
from riichienv import ActionType


@dataclass(frozen=True, slots=True)
class TrainOptions:
    """Runtime options for one supervised baseline training run."""

    train_data_path: Path
    output_path: Path
    batch_size: int
    epochs: int
    device: str
    learning_rate: float
    hidden_size: int
    model_type: str = "auto"
    model_arch: str = "mlp"
    extended: bool = False
    strict: bool = True
    action_types: frozenset[str] | None = None
    max_examples: int | None = None
    validation_data_path: Path | None = None
    validation_ratio: float = 0.0
    seed: int = 7
    num_workers: int = 0
    early_stopping_patience: int | None = None
    action_type_weight_power: float = 0.0
    example_weighting: bool = False
    train_discard_head: bool = False


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Summary returned after writing the checkpoint."""

    output_path: Path
    examples: int
    action_count: int
    final_loss: float
    best_metric: float


def train_baseline(options: TrainOptions) -> TrainingResult:
    """Train the requested supervised baseline policy and save a checkpoint."""
    model_type = _resolve_model_type(options.model_type)
    if model_type == "action-prior":
        return train_action_prior_baseline(options)
    if options.train_discard_head:
        return train_policy_with_discard_head(options)
    return train_policy_network(options)


def train_action_prior_baseline(options: TrainOptions) -> TrainingResult:
    """Train a simple supervised action-prior policy without optional ML dependencies."""
    train_paths, _, dataset_meta = _resolve_dataset_paths(options)
    vocabulary = ActionVocabulary()
    action_counts: list[int] = []
    example_count = 0

    for example_index, example in enumerate(_iter_examples(train_paths, options)):
        if options.max_examples is not None and example_index >= options.max_examples:
            break

        for action in example.observation.legal_actions():
            _ensure_count_slot(action_counts, vocabulary.add(action))
        label = vocabulary.add(example.action)
        _ensure_count_slot(action_counts, label)
        action_counts[label] += 1
        example_count += 1

    if example_count == 0:
        raise RuntimeError(f"No supervised examples found under {options.train_data_path}")
    if len(vocabulary) == 0:
        raise RuntimeError(f"No legal actions found while scanning {options.train_data_path}")

    total_actions = sum(action_counts)
    action_count = len(vocabulary)
    log_priors = [
        math.log((count + 1) / (total_actions + action_count)) for count in action_counts
    ]
    final_loss = _negative_log_likelihood(action_counts, log_priors, total_actions)
    input_shape = EXTENDED_OBSERVATION_SHAPE if options.extended else BASE_OBSERVATION_SHAPE
    checkpoint = {
        "format_version": 2,
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
        "training": _training_metadata(
            options,
            "cpu",
            example_count,
            [{"epoch": 1, "train_loss": final_loss, "validation_loss": None}],
            dataset_meta,
        ),
    }

    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    with options.output_path.open("w", encoding="utf-8") as checkpoint_file:
        json.dump(checkpoint, checkpoint_file, indent=2)
        checkpoint_file.write("\n")

    _write_metrics_report(options.output_path, checkpoint["training"]["history"])
    return TrainingResult(
        output_path=options.output_path,
        examples=example_count,
        action_count=action_count,
        final_loss=final_loss,
        best_metric=final_loss,
    )


def train_policy_network(options: TrainOptions) -> TrainingResult:
    """Train a masked PyTorch policy network and save the best checkpoint."""
    torch, nn, dataloader_cls, iterable_dataset_cls = _import_torch_training()
    from mahjong_ai.training.policy import PolicyModelConfig, build_policy_model, mask_illegal_logits

    train_paths, validation_paths, dataset_meta = _resolve_dataset_paths(options)
    # Shuffle training paths so the vocabulary scan and every training epoch draw
    # from all year-subdirectories uniformly rather than only the first sorted year.
    train_paths = _shuffle_paths(train_paths, options.seed)
    vocabulary, example_count = build_action_vocabulary(options, train_paths)
    # Skip vocab extension when training scan already covered >= _VOCAB_SCAN_MAX_EXAMPLES
    # examples — the action space is fixed in Mahjong and is complete by that point.
    vocab_was_capped = (
        options.max_examples is not None and options.max_examples < _VOCAB_SCAN_MAX_EXAMPLES
    )
    if validation_paths and vocab_was_capped:
        extend_action_vocabulary_from_paths(options, vocabulary, validation_paths)
    elif validation_paths:
        print(f"skipping vocabulary extension (training scan of {example_count:,} examples is sufficient)", flush=True)
    if example_count == 0:
        raise RuntimeError(f"No supervised examples found under {options.train_data_path}")
    if len(vocabulary) == 0:
        raise RuntimeError(f"No legal actions found while scanning {options.train_data_path}")

    _seed_everything(options.seed, torch)
    device = torch.device(_resolve_device(options.device, torch))
    input_shape = EXTENDED_OBSERVATION_SHAPE if options.extended else BASE_OBSERVATION_SHAPE
    model_config = PolicyModelConfig(
        input_shape=input_shape,
        action_count=len(vocabulary),
        model_arch=options.model_arch,
        hidden_size=options.hidden_size,
    )
    model = build_policy_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate)

    class SupervisedDecisionDataset(iterable_dataset_cls):
        def __init__(
            self,
            replay_paths: tuple[Path, ...],
            dataset_options: TrainOptions,
            output_vocabulary: ActionVocabulary,
        ) -> None:
            super().__init__()
            self.replay_paths = replay_paths
            self.options = dataset_options
            self.vocabulary = output_vocabulary

        def __iter__(self):
            # Reshuffle replay files every epoch so each epoch draws from all
            # year-subdirectories, not just the first files in sorted order.
            epoch_paths = _shuffle_paths(self.replay_paths, seed=random.randint(0, 2**31))
            for example_index, example in enumerate(_iter_examples(epoch_paths, self.options)):
                if self.options.max_examples is not None and example_index >= self.options.max_examples:
                    break
                try:
                    encoded = example.encoded_decision(self.vocabulary, extended=self.options.extended)
                except UnknownActionError:
                    continue
                features = torch.frombuffer(bytearray(encoded.features.data), dtype=torch.float32)
                features = features.reshape(encoded.features.shape)
                legal_mask = torch.tensor(encoded.legal_actions.mask, dtype=torch.bool)
                label = torch.tensor(encoded.require_label(), dtype=torch.long)
                weight = torch.tensor(example.weight, dtype=torch.float32)
                yield features, legal_mask, label, weight

    train_dataset = SupervisedDecisionDataset(train_paths, options, vocabulary)
    validation_dataset = (
        SupervisedDecisionDataset(validation_paths, options, vocabulary) if validation_paths else None
    )

    class_weights = _build_action_type_weights(
        options, vocabulary, train_paths, torch, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    def run_epoch(dataset: Any, *, training: bool, epoch: int) -> tuple[float, int]:
        loader = dataloader_cls(
            dataset,
            batch_size=options.batch_size,
            num_workers=options.num_workers,
        )
        if training:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        total_examples = 0
        phase = "train" if training else "val"
        max_ex = options.max_examples
        report_every = max(options.batch_size, (max_ex // 20) if max_ex else 50_000)
        next_report = report_every
        for batch in loader:
            features, legal_mask, labels, sample_weights = batch
            features = features.to(device)
            legal_mask = legal_mask.to(device)
            labels = labels.to(device)
            sample_weights = sample_weights.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                logits = mask_illegal_logits(model(features), legal_mask)
                per_example = criterion(logits, labels)
                loss = (per_example * sample_weights).mean()
            if training:
                loss.backward()
                optimizer.step()
            batch_examples = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_examples
            total_examples += batch_examples
            if total_examples >= next_report:
                running_loss = total_loss / max(total_examples, 1)
                if max_ex:
                    pct = min(100.0, 100.0 * total_examples / max_ex)
                    print(
                        f"  epoch {epoch} [{phase}] {pct:5.1f}%  "
                        f"examples={total_examples}  loss={running_loss:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  epoch {epoch} [{phase}] examples={total_examples}  "
                        f"loss={running_loss:.4f}",
                        flush=True,
                    )
                next_report += report_every
        return total_loss / max(total_examples, 1), total_examples

    history: list[dict[str, float | int | None]] = []
    best_metric = float("inf")
    best_epoch = 0
    best_state_dict = None
    stale_epochs = 0

    for epoch in range(1, options.epochs + 1):
        train_loss, trained_examples = run_epoch(train_dataset, training=True, epoch=epoch)
        if trained_examples == 0:
            raise RuntimeError("The training dataset produced no examples")
        validation_loss = None
        validation_examples = 0
        if validation_dataset is not None:
            validation_loss, validation_examples = run_epoch(validation_dataset, training=False, epoch=epoch)

        metric = validation_loss if validation_loss is not None else train_loss
        improved = metric < best_metric
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_examples": trained_examples,
                "validation_loss": validation_loss,
                "validation_examples": validation_examples if validation_loss is not None else None,
            }
        )
        print(
            f"epoch {epoch}/{options.epochs}: train_loss={train_loss:.4f} "
            f"val_loss={validation_loss if validation_loss is not None else float('nan'):.4f} "
            f"examples={trained_examples}"
        )
        if options.early_stopping_patience is not None and stale_epochs >= options.early_stopping_patience:
            print(
                f"early stopping at epoch {epoch}: no improvement in {stale_epochs} epochs "
                f"(best epoch {best_epoch})"
            )
            break

    if best_state_dict is None:
        raise RuntimeError("Training ended without a valid checkpoint state")

    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 2,
        "model_type": "policy_network",
        "model_config": model_config.to_mapping(),
        "state_dict": best_state_dict,
        "action_vocabulary": vocabulary.to_mapping(),
        "feature_schema": {
            "dtype": "float32",
            "extended": options.extended,
            "shape": list(input_shape),
        },
        "training": _training_metadata(
            options,
            str(device),
            example_count,
            history,
            dataset_meta | {"validation_examples": _count_examples(validation_paths, options)},
        ),
        "selection": {
            "metric": "validation_loss" if validation_paths else "train_loss",
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        },
    }
    torch.save(checkpoint, options.output_path)
    _write_metrics_report(options.output_path, history)
    return TrainingResult(
        output_path=options.output_path,
        examples=example_count,
        action_count=len(vocabulary),
        final_loss=float(history[-1]["train_loss"]),
        best_metric=float(best_metric),
    )

# Mahjong's action vocabulary is finite and small; it converges well within this cap.
_VOCAB_SCAN_MAX_EXAMPLES = 200_000


def build_action_vocabulary(options: TrainOptions, replay_paths: tuple[Path, ...]) -> tuple[ActionVocabulary, int]:
    """Scan examples to create the stable output vocabulary.

    Caps at *options.max_examples* when set, otherwise at ``_VOCAB_SCAN_MAX_EXAMPLES``
    so the vocab scan stays fast regardless of dataset size.
    """
    vocab_cap = options.max_examples if options.max_examples is not None else _VOCAB_SCAN_MAX_EXAMPLES
    vocabulary = ActionVocabulary()
    example_count = 0
    print(f"building vocabulary (scanning up to {vocab_cap:,} examples)...", flush=True)
    for example_index, example in enumerate(_iter_examples(replay_paths, options)):
        if example_index >= vocab_cap:
            break
        vocabulary.add_actions(example.observation.legal_actions())
        vocabulary.add(example.action)
        example_count += 1
        if example_count % 50_000 == 0:
            print(f"  vocab scan: {example_count:,} examples, {len(vocabulary)} actions so far", flush=True)
    print(f"vocabulary built: {len(vocabulary)} actions from {example_count:,} examples", flush=True)
    return vocabulary, example_count


# Bound validation-only vocabulary scans so large replay splits stay tractable.
_VOCAB_EXTENSION_MAX_EXAMPLES = 100_000


def extend_action_vocabulary_from_paths(
    options: TrainOptions,
    vocabulary: ActionVocabulary,
    replay_paths: tuple[Path, ...],
) -> None:
    """Add any actions from *replay_paths* so validation labels stay encodable."""
    if not replay_paths:
        return
    cap = min(
        options.max_examples if options.max_examples is not None else _VOCAB_EXTENSION_MAX_EXAMPLES,
        _VOCAB_EXTENSION_MAX_EXAMPLES,
    )
    print(f"extending vocabulary from validation paths (cap={cap} examples)...", flush=True)
    val_options = replace(options, max_examples=cap)
    for example in _iter_examples(replay_paths, val_options):
        vocabulary.add_actions(example.observation.legal_actions())
        vocabulary.add(example.action)


def _iter_examples(replay_paths: tuple[Path, ...], options: TrainOptions) -> Iterator[Any]:
    return iter_supervised_examples_from_paths(
        replay_paths,
        action_types=options.action_types,
        strict=options.strict,
        example_weighting=options.example_weighting,
    )


def train_policy_with_discard_head(options: TrainOptions) -> TrainingResult:
    """Train global policy, then discard specialist, and save a combined checkpoint."""
    global_options = replace(options, train_discard_head=False)
    global_result = train_policy_network(global_options)
    discard_options = replace(
        options,
        action_types=frozenset({"DISCARD"}),
        train_discard_head=False,
    )
    discard_result = train_discard_policy_network(discard_options, global_checkpoint=global_options.output_path)
    _merge_policy_and_discard_checkpoint(
        global_path=global_options.output_path,
        discard_path=discard_result.output_path,
        output_path=options.output_path,
    )
    return TrainingResult(
        output_path=options.output_path,
        examples=global_result.examples,
        action_count=global_result.action_count,
        final_loss=global_result.final_loss,
        best_metric=global_result.best_metric,
    )


def train_discard_policy_network(
    options: TrainOptions,
    *,
    global_checkpoint: Path | None = None,
) -> TrainingResult:
    """Train a 34-type discard head; optionally inherit feature schema from *global_checkpoint*."""
    torch, nn, dataloader_cls, iterable_dataset_cls = _import_torch_training()
    from mahjong_ai.features.tiles import tile_id_to_type_index
    from mahjong_ai.training.policy import DiscardPolicyConfig, build_discard_policy, mask_illegal_logits

    train_paths, validation_paths, dataset_meta = _resolve_dataset_paths(options)
    train_paths = _shuffle_paths(train_paths, options.seed)
    input_shape = EXTENDED_OBSERVATION_SHAPE if options.extended else BASE_OBSERVATION_SHAPE
    discard_output = options.output_path
    if global_checkpoint is not None:
        discard_output = options.output_path.with_name(
            options.output_path.stem + "_discard_head" + options.output_path.suffix
        )

    _seed_everything(options.seed, torch)
    device = torch.device(_resolve_device(options.device, torch))
    discard_config = DiscardPolicyConfig(
        input_shape=input_shape,
        model_arch=options.model_arch,
        hidden_size=options.hidden_size,
    )
    model = build_discard_policy(discard_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction="none")
    discard_type = int(ActionType.DISCARD)

    class DiscardDataset(iterable_dataset_cls):
        def __init__(self, paths: tuple[Path, ...], opts: TrainOptions) -> None:
            super().__init__()
            self.paths = paths
            self.opts = opts

        def __iter__(self):
            epoch_paths = _shuffle_paths(self.paths, seed=random.randint(0, 2**31))
            for example_index, example in enumerate(_iter_examples(epoch_paths, self.opts)):
                if self.opts.max_examples is not None and example_index >= self.opts.max_examples:
                    break
                if int(example.action.action_type) != discard_type:
                    continue
                if example.action.tile is None:
                    continue
                features = encode_observation(example.observation, extended=self.opts.extended)
                tensor = torch.frombuffer(bytearray(features.data), dtype=torch.float32)
                tensor = tensor.reshape(features.shape)
                type_mask = [False] * 34
                for action in example.observation.legal_actions():
                    if int(action.action_type) == discard_type and action.tile is not None:
                        type_mask[tile_id_to_type_index(action.tile)] = True
                label = tile_id_to_type_index(example.action.tile)
                yield (
                    tensor,
                    torch.tensor(type_mask, dtype=torch.bool),
                    torch.tensor(label, dtype=torch.long),
                    torch.tensor(example.weight, dtype=torch.float32),
                )

    train_dataset = DiscardDataset(train_paths, options)
    validation_dataset = DiscardDataset(validation_paths, options) if validation_paths else None

    def run_epoch(dataset: Any, *, training: bool, epoch: int) -> tuple[float, int]:
        loader = dataloader_cls(dataset, batch_size=options.batch_size, num_workers=options.num_workers)
        model.train(training)
        total_loss = 0.0
        total_examples = 0
        for features, type_mask, labels, sample_weights in loader:
            features = features.to(device)
            type_mask = type_mask.to(device)
            labels = labels.to(device)
            sample_weights = sample_weights.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                logits = mask_illegal_logits(model(features), type_mask)
                per_example = criterion(logits, labels)
                loss = (per_example * sample_weights).mean()
            if training:
                loss.backward()
                optimizer.step()
            batch_examples = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_examples
            total_examples += batch_examples
        return total_loss / max(total_examples, 1), total_examples

    history: list[dict[str, float | int | None]] = []
    best_metric = float("inf")
    best_state_dict = None
    for epoch in range(1, options.epochs + 1):
        train_loss, trained = run_epoch(train_dataset, training=True, epoch=epoch)
        val_loss = None
        if validation_dataset is not None:
            val_loss, _ = run_epoch(validation_dataset, training=False, epoch=epoch)
        metric = val_loss if val_loss is not None else train_loss
        if metric < best_metric:
            best_metric = metric
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})

    if best_state_dict is None:
        raise RuntimeError("Discard-head training produced no checkpoint state")

    discard_output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 2,
        "model_type": "discard_policy_network",
        "discard_model_config": discard_config.to_mapping(),
        "discard_state_dict": best_state_dict,
        "feature_schema": {
            "dtype": "float32",
            "extended": options.extended,
            "shape": list(input_shape),
        },
        "training": _training_metadata(
            options,
            str(device),
            _count_examples(train_paths, options),
            history,
            dataset_meta,
        ),
    }
    torch.save(checkpoint, discard_output)
    return TrainingResult(
        output_path=discard_output,
        examples=_count_examples(train_paths, options),
        action_count=34,
        final_loss=float(history[-1]["train_loss"]),
        best_metric=float(best_metric),
    )


def _merge_policy_and_discard_checkpoint(
    *,
    global_path: Path,
    discard_path: Path,
    output_path: Path,
) -> None:
    torch, _, _, _ = _import_torch_training()
    try:
        global_ckpt = torch.load(global_path, map_location="cpu", weights_only=False)
    except TypeError:
        global_ckpt = torch.load(global_path, map_location="cpu")
    try:
        discard_ckpt = torch.load(discard_path, map_location="cpu", weights_only=False)
    except TypeError:
        discard_ckpt = torch.load(discard_path, map_location="cpu")

    merged = dict(global_ckpt)
    merged["model_type"] = "policy_with_discard_head"
    merged["discard_model_config"] = discard_ckpt["discard_model_config"]
    merged["discard_state_dict"] = discard_ckpt["discard_state_dict"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)


def _resolve_dataset_paths(
    options: TrainOptions,
) -> tuple[tuple[Path, ...], tuple[Path, ...], dict[str, Any]]:
    print(f"listing replay files under {options.train_data_path} ...", flush=True)
    if options.validation_data_path is not None:
        train_paths = list_replay_paths(options.train_data_path)
        validation_paths = list_replay_paths(options.validation_data_path)
    else:
        train_paths, validation_paths = split_replay_paths(
            options.train_data_path,
            validation_ratio=options.validation_ratio,
            seed=options.seed,
        )
    print(f"found {len(train_paths)} train + {len(validation_paths)} validation replay files", flush=True)
    return train_paths, validation_paths, {
        "train_replays": len(train_paths),
        "validation_replays": len(validation_paths),
        "train_paths": [str(path) for path in train_paths],
        "validation_paths": [str(path) for path in validation_paths],
    }


def _count_examples(replay_paths: tuple[Path, ...], options: TrainOptions) -> int:
    total = 0
    for example_index, _ in enumerate(_iter_examples(replay_paths, options)):
        if options.max_examples is not None and example_index >= options.max_examples:
            break
        total += 1
    return total


def _build_action_type_weights(
    options: TrainOptions,
    vocabulary: ActionVocabulary,
    replay_paths: tuple[Path, ...],
    torch_module: Any,
    device: Any,
):
    if options.action_type_weight_power <= 0:
        return None
    vocab_cap = options.max_examples if options.max_examples is not None else _VOCAB_SCAN_MAX_EXAMPLES
    type_counts: Counter[int] = Counter()
    label_counts: Counter[int] = Counter()
    for example_index, example in enumerate(_iter_examples(replay_paths, options)):
        if example_index >= vocab_cap:
            break
        try:
            label = vocabulary.encode(example.action)
        except UnknownActionError:
            continue
        label_counts[label] += 1
        type_counts[int(example.action.action_type)] += 1
    if not label_counts:
        return None
    weights = []
    for label in range(len(vocabulary)):
        spec = vocabulary.decode(label)
        type_count = type_counts.get(spec.action_type, 1)
        value = (1.0 / type_count) ** options.action_type_weight_power
        weights.append(value)
    mean_weight = sum(weights) / max(len(weights), 1)
    normalized = [weight / max(mean_weight, 1e-12) for weight in weights]
    return torch_module.tensor(normalized, dtype=torch_module.float32).to(device)


def _shuffle_paths(paths: tuple[Path, ...], seed: int) -> tuple[Path, ...]:
    """Return *paths* in a deterministically shuffled order for a given *seed*."""
    lst = list(paths)
    random.Random(seed).shuffle(lst)
    return tuple(lst)


def _seed_everything(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


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
    history: list[dict[str, float | int | None]],
    dataset_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "train_data_path": str(options.train_data_path),
        "validation_data_path": (
            str(options.validation_data_path) if options.validation_data_path is not None else None
        ),
        "validation_ratio": options.validation_ratio,
        "batch_size": options.batch_size,
        "epochs": options.epochs,
        "learning_rate": options.learning_rate,
        "device": device,
        "strict": options.strict,
        "action_types": sorted(options.action_types) if options.action_types else None,
        "max_examples": options.max_examples,
        "examples": examples,
        "seed": options.seed,
        "num_workers": options.num_workers,
        "model_arch": options.model_arch,
        "early_stopping_patience": options.early_stopping_patience,
        "action_type_weight_power": options.action_type_weight_power,
        "example_weighting": options.example_weighting,
        "train_discard_head": options.train_discard_head,
        "dataset": dataset_metadata,
        "history": history,
    }


def _write_metrics_report(output_path: Path, history: list[dict[str, float | int | None]]) -> None:
    metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump({"history": history}, metrics_file, indent=2)
        metrics_file.write("\n")


def _resolve_model_type(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        _import_torch_training()
    except RuntimeError:
        print("torch is not available; training action-prior baseline instead")
        return "action-prior"
    return "policy-network"


def _import_torch_training():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, IterableDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for --model-type policy-network. Install training extras "
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
    parser.add_argument("--train-data", type=Path, default=None, help="Training replay path")
    parser.add_argument("--validation-data", type=Path, default=None, help="Validation replay path")
    parser.add_argument("--validation-ratio", type=float, default=config.training.validation_ratio)
    parser.add_argument("--output", type=Path, default=config.model.artifact_path, help="Checkpoint path")
    parser.add_argument("--batch-size", type=int, default=config.training.batch_size)
    parser.add_argument("--epochs", type=int, default=config.training.epochs)
    parser.add_argument("--device", default=config.training.device)
    parser.add_argument("--learning-rate", type=float, default=config.training.learning_rate)
    parser.add_argument("--hidden-size", type=int, default=config.training.hidden_size)
    parser.add_argument("--seed", type=int, default=config.training.seed)
    parser.add_argument("--num-workers", type=int, default=config.training.num_workers)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=config.training.early_stopping_patience,
        help="Stop after N epochs without improvement. Disabled when omitted.",
    )
    parser.add_argument(
        "--action-type-weight-power",
        type=float,
        default=config.training.action_type_weight_power,
        help="Reweight rare action types with inverse-frequency**power. 0 disables.",
    )
    parser.add_argument(
        "--model-type",
        choices=("auto", "policy-network", "action-prior"),
        default="auto",
        help="Baseline model family to train.",
    )
    parser.add_argument(
        "--model-arch",
        choices=("mlp", "conv"),
        default=config.training.model_arch,
        help="Network architecture when --model-type policy-network.",
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
    parser.add_argument(
        "--example-weighting",
        action="store_true",
        help="Weight supervised examples by final replay score rank per seat",
    )
    parser.add_argument(
        "--train-discard-head",
        action="store_true",
        help="Train global policy plus a discard specialist and save a combined checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_data = args.train_data or args.data
    result = train_baseline(
        TrainOptions(
            train_data_path=train_data,
            validation_data_path=args.validation_data,
            validation_ratio=args.validation_ratio,
            output_path=args.output,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            model_type=args.model_type,
            model_arch=args.model_arch,
            extended=args.extended,
            strict=not args.skip_bad_replays,
            action_types=_parse_action_types(args.action_types),
            max_examples=args.max_examples,
            seed=args.seed,
            num_workers=args.num_workers,
            early_stopping_patience=args.early_stopping_patience,
            action_type_weight_power=args.action_type_weight_power,
            example_weighting=args.example_weighting,
            train_discard_head=args.train_discard_head,
        )
    )
    print(
        f"saved {result.action_count}-action policy to {result.output_path} "
        f"from {result.examples} examples; final_loss={result.final_loss:.4f} "
        f"best_metric={result.best_metric:.4f}"
    )


if __name__ == "__main__":
    main()
