"""Run a reproducible GTP match against an external Go engine."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .go import BLACK, WHITE, Rules
from .gtp import format_vertex, parse_vertex


class GtpClient:
    def __init__(self, command, env=None, cwd=None, label="engine"):
        self.label = label
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
            cwd=cwd,
        )

    def command(self, line):
        if self.process.poll() is not None:
            raise RuntimeError(f"{self.label} exited with {self.process.returncode}")
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        response = []
        while True:
            text = self.process.stdout.readline()
            if text == "":
                raise RuntimeError(f"{self.label} closed stdout while handling {line!r}")
            if not text.strip():
                break
            response.append(text.rstrip("\n"))
        if not response:
            return ""
        first = response[0]
        if first.startswith("?"):
            raise RuntimeError(f"{self.label} rejected {line!r}: {first}")
        if first.startswith("="):
            first = first[1:].lstrip()
        return "\n".join([first, *response[1:]]).strip()

    def close(self):
        if self.process.poll() is None:
            try:
                self.command("quit")
            except (BrokenPipeError, RuntimeError):
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)


def _move_from_response(response, size):
    fields = response.strip().lstrip("= ").split()
    if not fields:
        raise ValueError(f"empty GTP move response: {response!r}")
    token = fields[0]
    if token.lower() in {"resign", "pass"}:
        return token.lower()
    try:
        return parse_vertex(token, size)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid GTP move response: {response!r}") from exc


def _setup(client, size, komi):
    client.command(f"boardsize {size}")
    client.command(f"komi {komi}")
    client.command("clear_board")


def play_game(ours, opponent, rules, our_color):
    _setup(ours, rules.size, rules.komi)
    _setup(opponent, rules.size, rules.komi)
    state = rules.initial()
    while not state.is_terminal():
        client = ours if state.to_play == our_color else opponent
        other_client = opponent if client is ours else ours
        color = "b" if state.to_play == BLACK else "w"
        response = client.command(f"genmove {color}")
        move = _move_from_response(response, rules.size)
        if move == "resign":
            winner = WHITE if state.to_play == BLACK else BLACK
            return winner, state.moves
        actual_move = rules.pass_move if move == "pass" else move
        state = state.play(actual_move)
        other_client.command(f"play {color} {format_vertex(actual_move, rules.size)}")
    return state.winner(), state.moves


def run_match(our_command, opponent_command, rules, games, our_env=None, opponent_env=None):
    ours = GtpClient(our_command, env=our_env, label="our engine")
    opponent = GtpClient(opponent_command, env=opponent_env, label="opponent")
    result = {
        "games": games,
        "our_wins": 0,
        "opponent_wins": 0,
        "draws": 0,
        "game_lengths": [],
    }
    try:
        for index in range(games):
            our_color = BLACK if index % 2 == 0 else WHITE
            winner, length = play_game(ours, opponent, rules, our_color)
            result["game_lengths"].append(length)
            if winner == 0:
                result["draws"] += 1
            elif winner == our_color:
                result["our_wins"] += 1
            else:
                result["opponent_wins"] += 1
    finally:
        ours.close()
        opponent.close()
    result["mean_game_length"] = (
        sum(result["game_lengths"]) / len(result["game_lengths"])
        if result["game_lengths"]
        else 0.0
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--opponent-cmd", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "xla"), default="xla")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    rules = Rules(
        size=config["size"],
        komi=config["komi"],
        max_moves=config["max_moves"],
    )
    our_command = [
        sys.executable,
        "-m",
        "alphago.gtp",
        "--checkpoint",
        args.checkpoint,
        "--config",
        args.config,
        "--device",
        args.device,
    ]
    result = run_match(
        our_command,
        shlex.split(args.opponent_cmd),
        rules,
        args.games,
        our_env=os.environ.copy(),
        opponent_env=os.environ.copy(),
    )
    result.update(
        {
            "checkpoint": args.checkpoint,
            "opponent_command": args.opponent_cmd,
            "board_size": rules.size,
            "komi": rules.komi,
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
