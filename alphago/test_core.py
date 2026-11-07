"""Standard-library tests for board rules."""

from .go import BLACK, EMPTY, Rules


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def run():
    state = Rules(size=5, komi=2.5).initial()
    check(len(state.legal_moves()) == 26, "initial 5x5 move count")
    check(state.to_play == BLACK, "black opens")
    check(state.score() == -2.5, "empty-board score")
    rules = Rules(size=5)
    passed = rules.initial().play(rules.pass_move).play(rules.pass_move)
    check(passed.is_terminal(), "two passes terminate")
    check(not rules.initial().is_legal(-1), "negative move is illegal")
    state = rules.initial().play(0)
    check(state.board[0] == BLACK, "legal placement")
    print("core tests passed")


if __name__ == "__main__":
    run()
