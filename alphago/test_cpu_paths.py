"""CPU-only checks for rules, MCTS expansion, and complete self-play games."""

import json
import random
from pathlib import Path

from .go import Rules
from .mcts import Search, run_round

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def _search_move(state, rng, simulations=4):
    config = {
        "sims": simulations,
        "cpuct": 1.5,
        "min_moves_before_pass": 20,
        "dirichlet_alpha": 0.3,
        "dirichlet_eps": 0.25,
    }
    search = Search(state, config, rng, add_noise=False)

    def evaluate(states):
        logits = np.zeros((len(states), state.rules.num_actions), dtype=np.float32)
        values = np.zeros(len(states), dtype=np.float32)
        return logits, values

    while not search.done():
        run_round([search], evaluate)
    move = search.choose_move(0.0)
    if move not in state.legal_moves():
        raise AssertionError(f"MCTS selected illegal move {move}")
    return move


def run():
    if np is None:
        print('{"result": "skipped", "reason": "NumPy is unavailable"}')
        return
    rng = np.random.default_rng(7)
    games = []
    for game_index in range(2):
        rules = Rules(size=9, komi=7.0, max_moves=162)
        state = rules.initial()
        moves = 0
        while not state.is_terminal():
            move = _search_move(state, rng)
            state = state.play(move)
            moves += 1
        if moves == 0 or moves > rules.max_moves:
            raise AssertionError(f"invalid game length {moves}")
        games.append({"moves": moves, "winner": state.winner()})

    rules19 = Rules(size=19, komi=7.5, max_moves=722)
    initial19 = rules19.initial()
    if len(initial19.legal_moves()) != 362:
        raise AssertionError("19x19 action shape is incorrect")

    result = {
        "rules": {"9x9": True, "19x19": True},
        "mcts": {"simulations": 4, "games": games},
        "result": "passed",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    random.seed(7)
    run()
