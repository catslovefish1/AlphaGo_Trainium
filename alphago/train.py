"""Short distributed AlphaZero-style self-play training."""

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np

from .go import Rules
from .mcts import Search, encode, run_round
from .model import build_model


def load_config(path):
    return json.loads(Path(path).read_text())


def augment(observation, policy, size, transform):
    grid = policy[:-1].reshape(size, size)
    if transform >= 4:
        observation, grid = observation[:, :, ::-1], grid[:, ::-1]
    rotation = transform % 4
    observation = np.rot90(observation, rotation, axes=(1, 2))
    grid = np.rot90(grid, rotation)
    return observation.copy(), np.concatenate([grid.reshape(-1), policy[-1:]])


def evaluate_model(model, runtime, states, size, batch_size):
    import torch

    inputs = np.zeros((batch_size, 4, size, size), dtype=np.float32)
    for index, state in enumerate(states):
        inputs[index] = encode(state)
    with torch.no_grad():
        outputs = model(torch.from_numpy(inputs).to(runtime.device()))
    runtime.mark_step()
    return outputs[0].detach().cpu().numpy(), outputs[1].detach().cpu().numpy()


def collect_games(model, runtime, rules, cfg, rng, games_wanted, deadline):
    slots = []
    for _ in range(cfg["slots"]):
        state = rules.initial()
        slots.append({"state": state, "history": [], "search": Search(state, cfg, rng, True)})
    finished = []
    while len(finished) < games_wanted and time.time() < deadline:
        run_round(
            [slot["search"] for slot in slots],
            lambda states: evaluate_model(model, runtime, states, rules.size, cfg["slots"]),
        )
        for index, slot in enumerate(slots):
            search = slot["search"]
            if not search.done():
                continue
            state = slot["state"]
            policy = search.policy_target(rules.num_actions)
            slot["history"].append((encode(state), policy, state.to_play))
            temperature = 1.0 if state.moves < cfg["temp_moves"] else 0.0
            next_state = state.play(search.choose_move(temperature))
            if next_state.is_terminal():
                winner = next_state.winner()
                samples = []
                for observation, target_policy, color in slot["history"]:
                    value = 0.0 if winner == 0 else 1.0 if winner == color else -1.0
                    samples.append((observation, target_policy, value))
                finished.append((samples, next_state.moves, winner))
                state = rules.initial()
                slots[index] = {"state": state, "history": [], "search": Search(state, cfg, rng, True)}
            else:
                slot["state"] = next_state
                slot["search"] = Search(next_state, cfg, rng, True)
    return finished


def train_steps(model, optimizer, runtime, replay, cfg, size, rng):
    import torch
    import torch.nn.functional as functional

    device = runtime.device()
    batch_size = cfg["train_batch"]
    observations = np.zeros((batch_size, 4, size, size), dtype=np.float32)
    policies = np.zeros((batch_size, size * size + 1), dtype=np.float32)
    values = np.zeros(batch_size, dtype=np.float32)
    policy_losses = []
    value_losses = []
    for _ in range(cfg["train_steps_per_iter"]):
        for row, index in enumerate(rng.integers(0, len(replay), batch_size)):
            observation, policy, value = replay[int(index)]
            observations[row], policies[row] = augment(observation, policy, size, int(rng.integers(8)))
            values[row] = value
        x = torch.from_numpy(observations).to(device)
        target_policy = torch.from_numpy(policies).to(device)
        target_value = torch.from_numpy(values).to(device)
        optimizer.zero_grad()
        logits, prediction = model(x)
        policy_loss = -(target_policy * functional.log_softmax(logits, dim=1)).sum(1).mean()
        value_loss = functional.mse_loss(prediction, target_value)
        (policy_loss + value_loss).backward()
        runtime.optimizer_step(optimizer)
        policy_losses.append(float(policy_loss.detach().cpu()))
        value_losses.append(float(value_loss.detach().cpu()))
    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def eval_random(model, runtime, rules, cfg, seed, games):
    import numpy as np

    rng = random.Random(seed)
    wins = 0
    for _ in range(games):
        state = rules.initial()
        while not state.is_terminal():
            if state.to_play == 1:
                logits, _ = evaluate_model(model, runtime, [state], rules.size, cfg["slots"])
                legal = state.legal_moves()
                masked = np.full(len(logits[0]), -np.inf, dtype=np.float32)
                masked[legal] = logits[0, legal]
                move = int(masked.argmax())
            else:
                legal = state.legal_moves()
                move = legal[rng.randrange(len(legal))]
            state = state.play(move)
        wins += int(state.winner() == 1)
    return wins, games


def reduce_sum(runtime, values):
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(values, dtype=torch.float32, device=runtime.device())
    dist.all_reduce(tensor)
    runtime.mark_step()
    return tensor.cpu().tolist()


def _to_cpu(value):
    """Copy checkpoint values off the accelerator before serializing them."""
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _move_optimizer_state(optimizer, device):
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(model, optimizer, replay, rng, path, iteration, rank, include_replay):
    import torch

    if not include_replay and rank != 0:
        return
    payload = {
        "model_state": _to_cpu(model.state_dict()),
        "optimizer_state": _to_cpu(optimizer.state_dict()),
        "iteration": iteration,
    }
    if include_replay:
        payload.update(
            {
                "replay": list(replay),
                "numpy_rng_state": rng.bit_generator.state,
                "python_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
            }
        )
        rank_path = path.with_name(f"{path.stem}.rank{rank:03d}{path.suffix}")
        torch.save(payload, rank_path)
        if rank == 0:
            torch.save(payload, path)
    elif rank == 0:
        torch.save(payload, path)


def load_checkpoint(model, optimizer, replay, rng, path, rank, device):
    import torch

    rank_path = path.with_name(f"{path.stem}.rank{rank:03d}{path.suffix}")
    load_path = rank_path if rank_path.exists() else path
    try:
        payload = torch.load(load_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(load_path, map_location="cpu")
    if "model_state" not in payload:
        payload = {"model_state": payload, "iteration": 0, "replay": []}
    model.load_state_dict(payload["model_state"])
    if payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
        _move_optimizer_state(optimizer, device)
    replay.clear()
    replay.extend(payload.get("replay", []))
    if payload.get("numpy_rng_state"):
        rng.bit_generator.state = payload["numpy_rng_state"]
    if payload.get("python_rng_state"):
        random.setstate(payload["python_rng_state"])
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"])
    return int(payload.get("iteration", 0))


def resolve_resume_path(value, output, rank):
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_dir():
        checkpoints = sorted(candidate.glob("checkpoint_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoints found in {candidate}")
        candidate = checkpoints[-1]
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="checkpoint file or run directory to resume from",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    from . import runtime

    rank, world = runtime.init_distributed()
    if world != 64:
        raise RuntimeError(f"expected WORLD_SIZE=64, got {world}")
    import torch

    torch.manual_seed(int(cfg["seed"]))
    seed = int(cfg["seed"]) + rank
    random.seed(seed)
    rng = np.random.default_rng(seed)
    rules = Rules(size=cfg["size"], komi=cfg["komi"], max_moves=cfg["max_moves"])
    device = runtime.device()
    model = build_model(cfg["size"], cfg["channels"], cfg["blocks"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    replay = deque(maxlen=cfg["buffer_size"])
    output = Path(args.out)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        output.joinpath("config.json").write_text(json.dumps(cfg, indent=2))
    runtime.rendezvous("train-start")
    resume_path = resolve_resume_path(args.resume, output, rank)
    start_iteration = 0
    if resume_path:
        start_iteration = load_checkpoint(
            model, optimizer, replay, rng, resume_path, rank, device
        )
        runtime.rendezvous("checkpoint-loaded")
    started = time.time()
    deadline = started + cfg["budget_s"]
    records = []
    if rank == 0 and resume_path:
        metrics_path = output / "metrics.jsonl"
        if not metrics_path.exists():
            metrics_path = resume_path.parent / "metrics.jsonl"
        if metrics_path.exists():
            records = [
                json.loads(line)
                for line in metrics_path.read_text().splitlines()
                if line.strip()
            ]
    checkpoint_interval = max(1, int(cfg.get("checkpoint_interval", 1)))
    checkpoint_replay = bool(cfg.get("checkpoint_replay", True))
    for iteration in range(start_iteration, cfg["iterations"]):
        if time.time() >= deadline:
            break
        games = collect_games(model, runtime, rules, cfg, rng, cfg["games_per_iter"], deadline)
        for samples, _, _ in games:
            replay.extend(samples)
        ready_count = int(
            reduce_sum(runtime, [int(len(replay) >= cfg["min_buffer"])])[0]
        )
        if ready_count < world:
            runtime.rendezvous(f"iteration-{iteration + 1}-buffer")
            continue
        policy_loss, value_loss = train_steps(model, optimizer, runtime, replay, cfg, cfg["size"], rng)
        wins, total = eval_random(model, runtime, rules, cfg, seed + iteration, cfg["eval_games"])
        aggregate = reduce_sum(runtime, [len(games), len(replay), policy_loss, value_loss, wins, total])
        record = {
            "iteration": iteration + 1,
            "world": world,
            "games_global": int(aggregate[0]),
            "replay_global": int(aggregate[1]),
            "policy_loss": aggregate[2] / world,
            "value_loss": aggregate[3] / world,
            "random_wins_as_black": int(aggregate[4]),
            "random_games": int(aggregate[5]),
            "elapsed_s": time.time() - started,
        }
        if rank == 0:
            records.append(record)
            with output.joinpath("metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        should_checkpoint = (
            (iteration + 1) % checkpoint_interval == 0
            or iteration + 1 == cfg["iterations"]
        )
        if should_checkpoint:
            save_checkpoint(
                model,
                optimizer,
                replay,
                rng,
                output / f"checkpoint_{iteration + 1:03d}.pt",
                iteration + 1,
                rank,
                checkpoint_replay,
            )
        if rank == 0:
            print(json.dumps(record), flush=True)
        runtime.rendezvous(f"iteration-{iteration + 1}")
    runtime.rendezvous("train-complete")
    if rank == 0:
        output.joinpath("summary.json").write_text(json.dumps({"world": world, "records": records}, indent=2))


if __name__ == "__main__":
    main()
