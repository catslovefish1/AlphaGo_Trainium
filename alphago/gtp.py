"""Minimal GTP adapter for smoke-test protocol checks."""

import sys

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
    def __init__(self, size=9, komi=7.0):
        self.rules = Rules(size=size, komi=komi)
        self.state = self.rules.initial()
        self.history = []

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
            move = self.state.legal_moves()[0]
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
    session = GtpSession()
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
