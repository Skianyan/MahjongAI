"""REINFORCE policy-gradient training in RiichiEnv (warm-started from a supervised checkpoint)."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from riichienv import ActionType, Observation, RiichiEnv, calculate_shanten

from mahjong_ai.agents import (
    Agent,
    AkochanAgent,
    FallbackAgent,
    MortalAgent,
    build_akochan_argv,
    build_mortal_argv,
)
from mahjong_ai.config import load_config
from mahjong_ai.features import ActionVocabulary, encode_observation
from mahjong_ai.training.policy import PolicyModelConfig, build_policy_model, mask_illegal_logits


@dataclass(frozen=True, slots=True)
class RolloutStep:
    """One policy decision with differentiable log-probability."""

    log_prob: torch.Tensor
    entropy: torch.Tensor
    shaping_reward: float = 0.0


@dataclass(frozen=True, slots=True)
class RLTrainOptions:
    warmstart_path: Path
    output_path: Path
    device: str
    learning_rate: float
    episodes_per_update: int
    rl_epochs: int
    gamma: float
    entropy_coef: float
    baseline_momentum: float
    reward_mode: str
    opponent: str
    game_mode: str
    max_actions_per_game: int
    controlled_seat: int
    seed: int
    mortal_binary: str | None
    mortal_model_dir: Path | None
    mortal_docker_image: str
    mortal_timeout: float
    akochan_dir: str | None
    akochan_tactics: Path | None
    akochan_timeout: float
    shanten_bonus: float
    tenpai_bonus: float
    riichi_bonus: float


@dataclass(frozen=True, slots=True)
class RLTrainingResult:
    output_path: Path
    rl_epochs: int
    final_baseline: float
    history: list[dict[str, float]]


def _build_opponents(
    kind: str,
    *,
    controlled_seat: int,
    mortal_binary: str | None = None,
    mortal_model_dir: Path | None = None,
    mortal_docker_image: str = "mortal:latest",
    mortal_timeout: float = 30.0,
    akochan_dir: str | None = None,
    akochan_tactics: Path | None = None,
    akochan_timeout: float = 30.0,
) -> dict[int, Agent]:
    if kind == "fallback":
        return {player_id: FallbackAgent() for player_id in range(4)}
    if kind == "mortal":
        if not mortal_binary:
            raise RuntimeError(
                "opponent=mortal requires mortal_binary (set [rl_training].mortal_binary "
                "or pass --mortal-binary)."
            )
        opponents: dict[int, Agent] = {}
        for seat in range(4):
            if seat == controlled_seat:
                continue
            argv = build_mortal_argv(
                player_id=seat,
                mortal_binary=mortal_binary,
                model_dir=mortal_model_dir,
                docker_image=mortal_docker_image,
            )
            agent = MortalAgent(seat, argv, timeout=mortal_timeout)
            agent.start()
            opponents[seat] = agent
        return opponents
    if kind == "akochan":
        if not akochan_dir:
            raise RuntimeError(
                "opponent=akochan requires akochan_dir (set [rl_training].akochan_dir "
                "or pass --akochan-dir)."
            )
        opponents_ako: dict[int, Agent] = {}
        for seat in range(4):
            if seat == controlled_seat:
                continue
            argv, extra_env = build_akochan_argv(
                player_id=seat,
                akochan_dir=akochan_dir,
                tactics_path=akochan_tactics,
            )
            agent = AkochanAgent(seat, argv, timeout=akochan_timeout, extra_env=extra_env)
            agent.start()
            opponents_ako[seat] = agent
        return opponents_ako
    try:
        from riichienv.agents import RandomAgent
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "riichienv.agents.RandomAgent is required for opponent='random'."
        ) from exc
    return {player_id: RandomAgent() for player_id in range(4)}


def _observations_for_all_seats(env: RiichiEnv, obs_dict: dict[int, Any]) -> dict[int, Any]:
    getter = getattr(env, "get_observations", None)
    if callable(getter):
        return getter([0, 1, 2, 3])
    return obs_dict


def _terminal_reward(
    scores: list[int],
    ranks: list[int],
    controlled_seat: int,
    *,
    reward_mode: str,
    truncated: bool,
) -> float:
    if truncated:
        return -0.5
    if reward_mode == "score":
        return (float(scores[controlled_seat]) - 25000.0) / 25000.0
    ordered = sorted(range(4), key=lambda s: (ranks[s], s))
    position = ordered.index(controlled_seat)
    table = (1.0, 0.5, -0.5, -1.0)
    return table[position]


def _compute_discard_shaping_reward(
    observation: Observation,
    chosen: Any,
    *,
    shanten_bonus: float,
    tenpai_bonus: float,
    riichi_bonus: float,
) -> float:
    """Shaping for discard/riichi steps: shanten reduction, tenpai, riichi declaration."""
    action_type = chosen.action_type
    if action_type not in (ActionType.DISCARD, ActionType.RIICHI):
        return 0.0

    shaping = 0.0
    if action_type == ActionType.RIICHI:
        shaping += riichi_bonus

    discard_tile = chosen.tile
    if discard_tile is None:
        return shaping

    hand_before = list(observation.hand)
    if discard_tile not in hand_before:
        return shaping

    hand_after = list(hand_before)
    hand_after.remove(discard_tile)

    shanten_before = calculate_shanten(hand_before)
    shanten_after = calculate_shanten(hand_after)

    delta = shanten_before - shanten_after
    if delta > 0:
        shaping += shanten_bonus * delta
    if shanten_after == 0:
        shaping += tenpai_bonus
    return shaping


def _load_warmstart(path: Path, device: torch.device) -> tuple[Any, ActionVocabulary, PolicyModelConfig, bool]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model_type = checkpoint.get("model_type")
    if model_type not in ("policy_network", "mlp_policy"):
        raise ValueError(
            "RL warm-start requires a PyTorch policy checkpoint "
            f"(model_type policy_network or mlp_policy), got {model_type!r}"
        )
    vocabulary = ActionVocabulary.from_mapping(checkpoint["action_vocabulary"])
    model_config = PolicyModelConfig.from_mapping(checkpoint["model_config"])
    feature_schema = checkpoint.get("feature_schema", {})
    extended = bool(feature_schema.get("extended", False))
    model = build_policy_model(model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, vocabulary, model_config, extended


def train_rl_policy(options: RLTrainOptions) -> RLTrainingResult:
    """REINFORCE with moving-average baseline and optional entropy regularization."""
    torch_device = torch.device(_resolve_device(options.device, torch))
    random.seed(options.seed)
    torch.manual_seed(options.seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)

    model, vocabulary, model_config, extended = _load_warmstart(options.warmstart_path, torch_device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate)

    baseline = 0.0
    history: list[dict[str, float]] = []

    persistent_opponents = _build_opponents(
        options.opponent,
        controlled_seat=options.controlled_seat,
        mortal_binary=options.mortal_binary,
        mortal_model_dir=options.mortal_model_dir,
        mortal_docker_image=options.mortal_docker_image,
        mortal_timeout=options.mortal_timeout,
        akochan_dir=options.akochan_dir,
        akochan_tactics=options.akochan_tactics,
        akochan_timeout=options.akochan_timeout,
    )
    try:
        for epoch in range(1, options.rl_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            policy_term = torch.zeros((), device=torch_device)
            entropy_term = torch.zeros((), device=torch_device)
            total_steps = 0
            mean_r = 0.0

            for _ in range(options.episodes_per_update):
                traj, R, _trunc = collect_episode(
                    model,
                    vocabulary,
                    torch_device,
                    extended=extended,
                    controlled_seat=options.controlled_seat,
                    opponent=options.opponent,
                    game_mode=options.game_mode,
                    max_actions_per_game=options.max_actions_per_game,
                    reward_mode=options.reward_mode,
                    mortal_binary=options.mortal_binary,
                    mortal_model_dir=options.mortal_model_dir,
                    mortal_docker_image=options.mortal_docker_image,
                    mortal_timeout=options.mortal_timeout,
                    akochan_dir=options.akochan_dir,
                    akochan_tactics=options.akochan_tactics,
                    akochan_timeout=options.akochan_timeout,
                    opponents=persistent_opponents,
                    shanten_bonus=options.shanten_bonus,
                    tenpai_bonus=options.tenpai_bonus,
                    riichi_bonus=options.riichi_bonus,
                )
                for ag in persistent_opponents.values():
                    if isinstance(ag, (MortalAgent, AkochanAgent)):
                        ag.reset_between_games()
                mean_r += R
                T = len(traj)
                for t, step in enumerate(traj):
                    discounted = R * (options.gamma ** (T - 1 - t))
                    advantage = torch.as_tensor(
                        discounted + step.shaping_reward - baseline,
                        device=torch_device,
                        dtype=torch.float32,
                    )
                    policy_term = policy_term - step.log_prob * advantage
                    entropy_term = entropy_term - step.entropy
                total_steps += max(T, 1)
                baseline = options.baseline_momentum * baseline + (1.0 - options.baseline_momentum) * R

            mean_r /= max(options.episodes_per_update, 1)
            if total_steps == 0:
                total_steps = 1
            loss = policy_term / total_steps + options.entropy_coef * entropy_term / total_steps
            loss.backward()
            optimizer.step()

            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach().cpu()),
                    "mean_episode_return": mean_r,
                    "baseline": baseline,
                }
            )
            print(
                f"rl epoch {epoch}/{options.rl_epochs}: loss={float(loss.detach().cpu()):.4f} "
                f"mean_R={mean_r:.4f} baseline={baseline:.4f}"
            )
    finally:
        for ag in persistent_opponents.values():
            if isinstance(ag, (MortalAgent, AkochanAgent)):
                ag.close()

    model.eval()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    input_shape = model_config.input_shape
    checkpoint = {
        "format_version": 2,
        "model_type": "policy_network",
        "model_config": model_config.to_mapping(),
        "state_dict": best_state,
        "action_vocabulary": vocabulary.to_mapping(),
        "feature_schema": {
            "dtype": "float32",
            "extended": extended,
            "shape": list(input_shape),
        },
        "training": {
            "algorithm": "reinforce",
            "warmstart_path": str(options.warmstart_path),
            "rl_epochs": options.rl_epochs,
            "episodes_per_update": options.episodes_per_update,
            "learning_rate": options.learning_rate,
            "gamma": options.gamma,
            "entropy_coef": options.entropy_coef,
            "baseline_momentum": options.baseline_momentum,
            "reward_mode": options.reward_mode,
            "opponent": options.opponent,
            "game_mode": options.game_mode,
            "max_actions_per_game": options.max_actions_per_game,
            "controlled_seat": options.controlled_seat,
            "seed": options.seed,
            "device": str(torch_device),
            "mortal_binary": options.mortal_binary,
            "mortal_model_dir": str(options.mortal_model_dir) if options.mortal_model_dir else None,
            "mortal_docker_image": options.mortal_docker_image,
            "mortal_timeout": options.mortal_timeout,
            "akochan_dir": options.akochan_dir,
            "akochan_tactics": str(options.akochan_tactics) if options.akochan_tactics else None,
            "akochan_timeout": options.akochan_timeout,
            "shanten_bonus": options.shanten_bonus,
            "tenpai_bonus": options.tenpai_bonus,
            "riichi_bonus": options.riichi_bonus,
            "history": history,
        },
    }
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, options.output_path)
    metrics_path = options.output_path.with_suffix(options.output_path.suffix + ".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"history": history}, f, indent=2)
        f.write("\n")

    return RLTrainingResult(
        output_path=options.output_path,
        rl_epochs=options.rl_epochs,
        final_baseline=baseline,
        history=history,
    )


def collect_episode(
    model: torch.nn.Module,
    vocabulary: ActionVocabulary,
    device: torch.device,
    *,
    extended: bool,
    controlled_seat: int,
    opponent: str,
    game_mode: str,
    max_actions_per_game: int,
    reward_mode: str,
    mortal_binary: str | None = None,
    mortal_model_dir: Path | None = None,
    mortal_docker_image: str = "mortal:latest",
    mortal_timeout: float = 30.0,
    akochan_dir: str | None = None,
    akochan_tactics: Path | None = None,
    akochan_timeout: float = 30.0,
    opponents: dict[int, Agent] | None = None,
    shanten_bonus: float = 0.05,
    tenpai_bonus: float = 0.10,
    riichi_bonus: float = 0.10,
) -> tuple[list[RolloutStep], float, bool]:
    """Play one game; return trajectory, terminal reward, truncated."""
    env = RiichiEnv(game_mode=game_mode)
    owns_opponents = opponents is None
    if opponents is None:
        opponents = _build_opponents(
            opponent,
            controlled_seat=controlled_seat,
            mortal_binary=mortal_binary,
            mortal_model_dir=mortal_model_dir,
            mortal_docker_image=mortal_docker_image,
            mortal_timeout=mortal_timeout,
            akochan_dir=akochan_dir,
            akochan_tactics=akochan_tactics,
            akochan_timeout=akochan_timeout,
        )
    fallback = FallbackAgent()
    obs_dict = env.reset()
    trajectory: list[RolloutStep] = []
    truncated = False
    action_steps = 0

    try:
        while not env.done():
            action_steps += 1
            if action_steps > max_actions_per_game:
                truncated = True
                break
            actions: dict[int, Any] = {}
            full_obs = _observations_for_all_seats(env, obs_dict)
            for seat in range(4):
                if seat == controlled_seat:
                    continue
                ag = opponents.get(seat)
                if isinstance(ag, (MortalAgent, AkochanAgent)) and seat not in obs_dict:
                    ag.observe(full_obs[seat])

            for player_id, observation in obs_dict.items():
                if player_id != controlled_seat:
                    ag = opponents[player_id]
                    obs_for_agent = full_obs[player_id]
                    if isinstance(ag, (MortalAgent, AkochanAgent)):
                        actions[player_id] = ag.act(obs_for_agent)
                    else:
                        actions[player_id] = ag.act(observation)
                    continue

                legal_mask = vocabulary.mask_for(observation)
                if not legal_mask.legal_indices:
                    actions[player_id] = fallback.act(observation)
                    continue

                features = encode_observation(observation, extended=extended)
                tensor = torch.frombuffer(bytearray(features.data), dtype=torch.float32)
                tensor = tensor.reshape(1, *features.shape).to(device)
                logits = model(tensor)[0]
                legal_t = torch.tensor(legal_mask.mask, dtype=torch.bool, device=device)
                masked = mask_illegal_logits(logits.unsqueeze(0), legal_t.unsqueeze(0))[0]
                dist = torch.distributions.Categorical(logits=masked)
                action_id = dist.sample()
                chosen = vocabulary.find_legal_action(observation, int(action_id.item()))
                if chosen is None:
                    actions[player_id] = fallback.act(observation)
                    continue

                shaping_r = _compute_discard_shaping_reward(
                    observation,
                    chosen,
                    shanten_bonus=shanten_bonus,
                    tenpai_bonus=tenpai_bonus,
                    riichi_bonus=riichi_bonus,
                )
                trajectory.append(
                    RolloutStep(
                        log_prob=dist.log_prob(action_id),
                        entropy=dist.entropy(),
                        shaping_reward=shaping_r,
                    )
                )
                actions[player_id] = chosen

            obs_dict = env.step(actions)
    finally:
        if owns_opponents:
            for ag in opponents.values():
                if isinstance(ag, (MortalAgent, AkochanAgent)):
                    ag.close()

    scores = [int(s) for s in env.scores()]
    ranks = [int(r) for r in env.ranks()]
    reward = _terminal_reward(
        scores,
        ranks,
        controlled_seat,
        reward_mode=reward_mode,
        truncated=truncated,
    )
    return trajectory, reward, truncated


def _resolve_device(requested: str, torch_module: Any) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None, help="TOML config path")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)
    rl = config.rl_training

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument(
        "--warmstart",
        type=Path,
        default=rl.warmstart_model,
        help="Supervised PyTorch checkpoint (policy_network or legacy mlp_policy)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/rl_policy.pt"),
        help="Output checkpoint path",
    )
    parser.add_argument("--device", default=config.training.device)
    parser.add_argument("--learning-rate", type=float, default=rl.learning_rate)
    parser.add_argument("--episodes-per-update", type=int, default=rl.episodes_per_update)
    parser.add_argument("--rl-epochs", type=int, default=rl.rl_epochs)
    parser.add_argument("--gamma", type=float, default=rl.gamma)
    parser.add_argument("--entropy-coef", type=float, default=rl.entropy_coef)
    parser.add_argument("--baseline-momentum", type=float, default=rl.baseline_momentum)
    parser.add_argument("--reward-mode", choices=("rank", "score"), default=rl.reward_mode)
    parser.add_argument(
        "--opponent",
        choices=("random", "fallback", "mortal", "akochan"),
        default=rl.opponent,
    )
    parser.add_argument(
        "--mortal-binary",
        default=rl.mortal_binary,
        help="Mortal executable path, or 'docker' to run via docker (see Mortal docs)",
    )
    parser.add_argument(
        "--mortal-model-dir",
        type=Path,
        default=rl.mortal_model_dir,
        help="Host directory mounted as /mnt when mortal_binary=docker",
    )
    parser.add_argument(
        "--mortal-docker-image",
        default=rl.mortal_docker_image,
        help="Docker image tag when mortal_binary=docker",
    )
    parser.add_argument("--mortal-timeout", type=float, default=rl.mortal_timeout)
    parser.add_argument(
        "--akochan-dir",
        default=rl.akochan_dir,
        help="Directory containing system.exe, libai.so, and tactics.json (or akochan_pipe.sh)",
    )
    parser.add_argument(
        "--akochan-tactics",
        type=Path,
        default=rl.akochan_tactics,
        help="Path to tactics.json (default: <akochan-dir>/tactics.json when not using akochan_pipe.sh)",
    )
    parser.add_argument("--akochan-timeout", type=float, default=rl.akochan_timeout)
    parser.add_argument("--game-mode", default=rl.game_mode)
    parser.add_argument("--max-actions-per-game", type=int, default=rl.max_actions_per_game)
    parser.add_argument("--controlled-seat", type=int, default=rl.controlled_seat)
    parser.add_argument("--seed", type=int, default=config.training.seed)
    parser.add_argument("--shanten-bonus", type=float, default=rl.shanten_bonus)
    parser.add_argument("--tenpai-bonus", type=float, default=rl.tenpai_bonus)
    parser.add_argument("--riichi-bonus", type=float, default=rl.riichi_bonus)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mortal_bin = args.mortal_binary
    if isinstance(mortal_bin, str) and mortal_bin.strip() == "":
        mortal_bin = None
    akochan_dir = args.akochan_dir
    if isinstance(akochan_dir, str) and akochan_dir.strip() == "":
        akochan_dir = None
    result = train_rl_policy(
        RLTrainOptions(
            warmstart_path=args.warmstart,
            output_path=args.output,
            device=args.device,
            learning_rate=args.learning_rate,
            episodes_per_update=args.episodes_per_update,
            rl_epochs=args.rl_epochs,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            baseline_momentum=args.baseline_momentum,
            reward_mode=args.reward_mode,
            opponent=args.opponent,
            game_mode=args.game_mode,
            max_actions_per_game=args.max_actions_per_game,
            controlled_seat=args.controlled_seat,
            seed=args.seed,
            mortal_binary=mortal_bin,
            mortal_model_dir=args.mortal_model_dir,
            mortal_docker_image=args.mortal_docker_image,
            mortal_timeout=args.mortal_timeout,
            akochan_dir=akochan_dir,
            akochan_tactics=args.akochan_tactics,
            akochan_timeout=args.akochan_timeout,
            shanten_bonus=args.shanten_bonus,
            tenpai_bonus=args.tenpai_bonus,
            riichi_bonus=args.riichi_bonus,
        )
    )
    print(
        f"saved RL policy to {result.output_path} "
        f"epochs={result.rl_epochs} final_baseline={result.final_baseline:.4f}"
    )


if __name__ == "__main__":
    main()
