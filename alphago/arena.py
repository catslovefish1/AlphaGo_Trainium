"""Checkpoint-backed MCTS arena and evaluation helpers."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .go import BLACK, WHITE, Rules
from .mcts import Search, encode, run_round
from .model import build_model


def load_checkpoint_model(checkpoint, size, channels, blocks, device):
    """Load a policy-value network without initializing distributed training."""
    import torch

    model = build_model(size, channels, blocks).to(device)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_states(model, device, states, size):
    import torch

    inputs = np.stack([encode(state) for state in states]).astype(np.float32)
    with torch.no_grad():
        logits, values = model(torch.from_numpy(inputs).to(device))
    outputs = logits.detach().cpu().numpy()
    scores = values.detach().cpu().numpy()
    if getattr(device, "type", str(device)) != "cpu":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    return outputs, scores


def choose_mcts_move(model, device, state, cfg, rng):
    search = Search(state, cfg, rng, add_noise=False)
    while not search.done():
        run_round(
            [search],
            lambda states: evaluate_states(model, device, states, state.rules.size),
        )
    return search.choose_move(0.0)


def play_game(
    rules,
    model_black,
    model_white,
    device,
    cfg,
    rng,
    random_black=False,
    random_white=False,
):
    state = rules.initial()
    moves = []
    while not state.is_terminal():
        if state.to_play == BLACK:
            model = model_black
            use_random = random_black
        else:
            model = model_white
            use_random = random_white
        if use_random:
            move = rng.choice(state.legal_moves())
        else:
            move = choose_mcts_move(model, device, state, cfg, rng)
        moves.append(move)
        state = state.play(move)
    return state.winner(), moves


def run_arena(
    rules,
    model_black,
    model_white,
    device,
    cfg,
    games,
    seed,
    random_black=False,
    random_white=False,
):
    rng = random.Random(seed)
    wins = {BLACK: 0, WHITE: 0, 0: 0}
    lengths = []
    for _ in range(games):
        winner, moves = play_game(
            rules,
            model_black,
            model_white,
            device,
            cfg,
            rng,
            random_black=random_black,
            random_white=random_white,
        )
        wins[winner] += 1
        lengths.append(len(moves))
    return {
        "games": games,
        "black_wins": wins[BLACK],
        "white_wins": wins[WHITE],
        "draws": wins[0],
        "mean_game_length": sum(lengths) / len(lengths) if lengths else 0.0,
    }


def _load_settings(config_path):
    cfg = json.loads(Path(config_path).read_text())
    cfg.setdefault("sims", 32)
    cfg.setdefault("cpuct", 1.5)
    cfg.setdefault("min_moves_before_pass", 30)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "xla"))
    parser.add_argument("--model-color", choices=("black", "white"), default="black")
    parser.add_argument("--opponent", choices=("random", "checkpoint"), default="random")
    parser.add_argument("--opponent-checkpoint")
    args = parser.parse_args()

    import torch

    cfg = _load_settings(args.config)
    device_name = args.device or cfg.get("device", "cpu")
    device = torch.device("cpu")
    if device_name == "xla":
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
    model = load_checkpoint_model(
        args.checkpoint, cfg["size"], cfg["channels"], cfg["blocks"], device
    )
    opponent = model
    if args.opponent == "checkpoint":
        if not args.opponent_checkpoint:
            raise ValueError("--opponent-checkpoint is required for checkpoint matches")
        opponent = load_checkpoint_model(
            args.opponent_checkpoint,
            cfg["size"],
            cfg["channels"],
            cfg["blocks"],
            device,
        )
    rules = Rules(size=cfg["size"], komi=cfg["komi"], max_moves=cfg["max_moves"])
    if args.model_color == "black":
        result = run_arena(
            rules,
            model,
            opponent,
            device,
            cfg,
            args.games,
            args.seed,
            random_white=args.opponent == "random",
        )
    else:
        result = run_arena(
            rules,
            opponent,
            model,
            device,
            cfg,
            args.games,
            args.seed,
            random_black=args.opponent == "random",
        )
    result.update(
        {
            "checkpoint": str(args.checkpoint),
            "opponent": args.opponent,
            "model_color": args.model_color,
            "sims": cfg["sims"],
            "size": cfg["size"],
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
