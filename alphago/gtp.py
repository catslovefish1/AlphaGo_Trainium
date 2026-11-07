"""GTP adapter with optional checkpoint-backed MCTS play."""

import argparse
import json
import sys
from pathlib import Path

from .go import BLACK, WHITE, Rules

LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def parse_vertex(token, size):
    token = token.strip().upper()
    if token == "PASS":
        return size * size
    col = LETTERS.index(token[0])
    row = int(token[1:])
    if not 0 <= col < size or not 1 <= row <= size:
        raise ValueError("vertex out of range")
    return (size - row) * size + col


def format_vertex(move, size):
    if move == size * size:
        return "pass"
    row, col = divmod(move, size)
    return f"{LETTERS[col]}{size - row}"


class GtpSession:
    def __init__(self, size=9, komi=7.0, model=None, device=None, search_cfg=None):
        self.rules = Rules(size=size, komi=komi)
        self.state = self.rules.initial()
        self.history = []
        self.model = model
        self.device = device
        self.search_cfg = search_cfg or {"sims": 32, "cpuct": 1.5}

    def command(self, line):
        parts = line.strip().split()
        if not parts:
            return ""
        name, args = parts[0].lower(), parts[1:]
        if name == "protocol_version":
            return "2"
        if name == "name":
            return "AlphaGo_Trainium"
        if name == "version":
            return "0.1-smoke"
        if name == "list_commands":
            return "protocol_version name version list_commands boardsize clear_board komi play genmove undo final_score quit"
        if name == "boardsize":
            self.rules = Rules(size=int(args[0]), komi=self.rules.komi)
            self.state = self.rules.initial()
            self.history = []
            return ""
        if name == "clear_board":
            self.state = self.rules.initial()
            self.history = []
            return ""
        if name == "komi":
            self.rules.komi = float(args[0])
            return ""
        if name == "play":
            color = BLACK if args[0].lower().startswith("b") else WHITE
            if color != self.state.to_play:
                raise ValueError("wrong color")
            self.history.append(self.state)
            self.state = self.state.play(parse_vertex(args[1], self.rules.size))
            return ""
        if name == "genmove":
            if self.model is None:
                move = self.state.legal_moves()[0]
            else:
                from .arena import choose_mcts_move

                import random

                move = choose_mcts_move(
                    self.model,
                    self.device,
                    self.state,
                    self.search_cfg,
                    random.Random(self.state.moves),
                )
            self.history.append(self.state)
            self.state = self.state.play(move)
            return format_vertex(move, self.rules.size)
        if name == "undo":
            if not self.history:
                raise ValueError("cannot undo")
            self.state = self.history.pop()
            return ""
        if name == "final_score":
            score = self.state.score()
            return f"B+{score:.1f}" if score > 0 else f"W+{-score:.1f}"
        if name == "quit":
            raise EOFError
        raise ValueError(f"unknown command {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--device", choices=("cpu", "xla"), default="cpu")
    args = parser.parse_args()
    model = None
    device = None
    search_cfg = None
    if args.checkpoint:
        if not args.config:
            raise ValueError("--config is required with --checkpoint")
        import torch

        from .arena import _load_settings, load_checkpoint_model

        config = _load_settings(args.config)
        device = torch.device("cpu")
        if args.device == "xla":
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
        model = load_checkpoint_model(
            args.checkpoint,
            config["size"],
            config["channels"],
            config["blocks"],
            device,
        )
        search_cfg = config
    session = GtpSession(model=model, device=device, search_cfg=search_cfg)
    for line in sys.stdin:
        try:
            response = session.command(line)
            print(f"= {response}\n", flush=True)
        except EOFError:
            print("= \n", flush=True)
            return
        except Exception as exc:
            print(f"? {exc}\n", flush=True)


if __name__ == "__main__":
    main()
