"""Standard-library tests for board rules."""

from .go import BLACK, EMPTY, WHITE, Rules, State


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

    capture_board = [EMPTY] * rules.n
    capture_board[6] = WHITE
    for point in (1, 5, 7):
        capture_board[point] = BLACK
    capture_state = State(rules, capture_board, BLACK, None, 0, 0)
    captured = capture_state.play(11)
    check(captured.board[6] == EMPTY, "single-stone capture")

    suicide_board = [EMPTY] * rules.n
    for point in (1, 5, 7, 11):
        suicide_board[point] = WHITE
    suicide_state = State(rules, suicide_board, BLACK, None, 0, 0)
    check(not suicide_state.is_legal(6), "suicide is illegal")

    ko_rules = Rules(size=3)
    ko_board = [EMPTY, EMPTY, EMPTY, EMPTY, BLACK, WHITE, BLACK, WHITE, EMPTY]
    ko_state = State(ko_rules, ko_board, BLACK, None, 0, 0)
    ko_after_capture = ko_state.play(8)
    check(ko_after_capture.ko == 7, "ko point is tracked")
    check(not ko_after_capture.is_legal(7), "ko recapture is illegal")
    print("core tests passed")


if __name__ == "__main__":
    run()
