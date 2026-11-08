"""Standard-library checks for the external GTP match harness."""

import sys
import tempfile
from pathlib import Path

from .go import Rules
from .gtp import format_vertex, parse_vertex
from .gtp_match import run_match


STUB = """import sys
for line in sys.stdin:
    command = line.strip().lower()
    if command == 'quit':
        print('=')
        print()
        break
    if command.startswith('genmove'):
        print('= pass')
    else:
        print('=')
    print()
    sys.stdout.flush()
"""


def run():
    for size in (9, 19):
        for move in (0, size - 1, size * (size - 1), size * size - 1, size * size):
            token = format_vertex(move, size)
            assert parse_vertex(token, size) == move

    with tempfile.TemporaryDirectory() as directory:
        stub = Path(directory) / "gtp_stub.py"
        stub.write_text(STUB)
        command = [sys.executable, str(stub)]
        result = run_match(command, command, Rules(size=3, komi=7.0, max_moves=9), 2)

    assert result["games"] == 2
    assert result["our_wins"] == 1
    assert result["opponent_wins"] == 1
    assert result["draws"] == 0
    assert all(length == 2 for length in result["game_lengths"])
    print("gtp protocol tests passed")


if __name__ == "__main__":
    run()
