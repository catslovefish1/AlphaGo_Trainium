"""Small batched PUCT search."""

import math


def encode(state):
    import numpy as np

    from .go import BLACK, WHITE

    size = state.rules.size
    opponent = WHITE if state.to_play == BLACK else BLACK
    board = np.asarray(state.board, dtype=np.int8).reshape(size, size)
    planes = np.zeros((4, size, size), dtype=np.float32)
    planes[0] = board == state.to_play
    planes[1] = board == opponent
    if state.ko is not None:
        planes[2, state.ko // size, state.ko % size] = 1.0
    if state.to_play == BLACK:
        planes[3] = 1.0
    return planes


class Node:
    __slots__ = ("state", "moves", "prior", "visits", "value_sum", "children")

    def __init__(self, state):
        self.state = state
        self.moves = None
        self.prior = None
        self.visits = None
        self.value_sum = None
        self.children = None


def _expand(node, logits, min_moves_before_pass):
    import numpy as np

    moves = node.state.legal_moves()
    if node.state.moves < min_moves_before_pass and len(moves) > 1:
        moves = moves[:-1]
    masked = np.asarray(logits)[moves]
    masked = masked - masked.max()
    prior = np.exp(masked)
    node.moves = moves
    node.prior = prior / prior.sum()
    node.visits = np.zeros(len(moves), dtype=np.float32)
    node.value_sum = np.zeros(len(moves), dtype=np.float32)
    node.children = [None] * len(moves)


def _terminal_value(state):
    winner = state.winner()
    if winner == 0:
        return 0.0
    return 1.0 if winner == state.to_play else -1.0


class Search:
    def __init__(self, state, cfg, rng, add_noise):
        self.root = Node(state)
        self.cfg = cfg
        self.rng = rng
        self.add_noise = add_noise
        self.sims_done = 0
        self.path = None
        self.leaf = None

    def done(self):
        return self.sims_done >= self.cfg["sims"]

    def select_leaf(self):
        node = self.root
        path = []
        while node.moves is not None and not node.state.is_terminal():
            total = node.visits.sum()
            q = node.value_sum / (node.visits + 1e-8)
            ucb = q + self.cfg["cpuct"] * node.prior * math.sqrt(total + 1.0) / (1.0 + node.visits)
            index = int(ucb.argmax())
            path.append((node, index))
            child = node.children[index]
            if child is None:
                child = Node(node.state.play(node.moves[index]))
                node.children[index] = child
            node = child
        self.path = path
        self.leaf = node
        if node.state.is_terminal():
            self._backup(_terminal_value(node.state))
            return None
        return node.state

    def backup(self, logits, value):
        _expand(self.leaf, logits, self.cfg.get("min_moves_before_pass", 0))
        if self.leaf is self.root and self.add_noise:
            noise = self.rng.dirichlet([self.cfg["dirichlet_alpha"]] * len(self.root.moves))
            epsilon = self.cfg["dirichlet_eps"]
            self.root.prior = (1.0 - epsilon) * self.root.prior + epsilon * noise
        self._backup(float(value))

    def _backup(self, value):
        for node, index in reversed(self.path):
            value = -value
            node.visits[index] += 1.0
            node.value_sum[index] += value
        self.sims_done += 1
        self.path = None
        self.leaf = None

    def policy_target(self, num_actions):
        import numpy as np

        policy = np.zeros(num_actions, dtype=np.float32)
        policy[self.root.moves] = self.root.visits
        total = policy.sum()
        if total:
            policy /= total
        else:
            policy[self.root.moves] = self.root.prior
        return policy

    def choose_move(self, temperature):
        import numpy as np

        visits = self.root.visits
        if temperature <= 1e-6 or visits.sum() == 0:
            values = visits if visits.sum() else self.root.prior
            return self.root.moves[int(values.argmax())]
        probabilities = visits ** (1.0 / temperature)
        probabilities /= probabilities.sum()
        return self.root.moves[int(self.rng.choice(len(self.root.moves), p=probabilities))]


def run_round(searches, evaluate):
    pending = []
    states = []
    for search in searches:
        if search.done():
            continue
        leaf = search.select_leaf()
        if leaf is not None:
            pending.append(search)
            states.append(leaf)
    if states:
        logits, values = evaluate(states)
        for index, search in enumerate(pending):
            search.backup(logits[index], values[index])
