"""Pure-Python Go rules used by every backend."""

EMPTY, BLACK, WHITE = 0, 1, 2


def other(color):
    return 3 - color


class Rules:
    def __init__(self, size=9, komi=7.0, max_moves=None):
        self.size = size
        self.n = size * size
        self.pass_move = self.n
        self.num_actions = self.n + 1
        self.komi = komi
        self.max_moves = max_moves or 2 * self.n
        self.neighbors = []
        for point in range(self.n):
            row, col = divmod(point, size)
            neighbors = []
            if row:
                neighbors.append(point - size)
            if row + 1 < size:
                neighbors.append(point + size)
            if col:
                neighbors.append(point - 1)
            if col + 1 < size:
                neighbors.append(point + 1)
            self.neighbors.append(tuple(neighbors))

    def initial(self):
        return State(self, [EMPTY] * self.n, BLACK, None, 0, 0)


class State:
    __slots__ = ("rules", "board", "to_play", "ko", "passes", "moves")

    def __init__(self, rules, board, to_play, ko, passes, moves):
        self.rules = rules
        self.board = board
        self.to_play = to_play
        self.ko = ko
        self.passes = passes
        self.moves = moves

    def is_terminal(self):
        return self.passes >= 2 or self.moves >= self.rules.max_moves

    def _group(self, board, start):
        color = board[start]
        stones = {start}
        stack = [start]
        has_liberty = False
        while stack:
            point = stack.pop()
            for neighbor in self.rules.neighbors[point]:
                value = board[neighbor]
                if value == EMPTY:
                    has_liberty = True
                elif value == color and neighbor not in stones:
                    stones.add(neighbor)
                    stack.append(neighbor)
        return stones, has_liberty

    def is_legal(self, move):
        if move == self.rules.pass_move:
            return True
        if move < 0 or move >= self.rules.n or self.board[move] != EMPTY or move == self.ko:
            return False
        board = self.board[:]
        color = self.to_play
        opponent = other(color)
        board[move] = color
        for neighbor in self.rules.neighbors[move]:
            if board[neighbor] == opponent and not self._group(board, neighbor)[1]:
                return True
        return self._group(board, move)[1]

    def legal_moves(self):
        moves = [point for point in range(self.rules.n) if self.is_legal(point)]
        moves.append(self.rules.pass_move)
        return moves

    def play(self, move):
        if not self.is_legal(move):
            raise ValueError(f"illegal move {move}")
        if move == self.rules.pass_move:
            return State(self.rules, self.board, other(self.to_play), None, self.passes + 1, self.moves + 1)
        board = self.board[:]
        color = self.to_play
        opponent = other(color)
        board[move] = color
        captured = []
        for neighbor in self.rules.neighbors[move]:
            if board[neighbor] == opponent:
                stones, has_liberty = self._group(board, neighbor)
                if not has_liberty:
                    for stone in stones:
                        board[stone] = EMPTY
                    captured.extend(stones)
        own, has_liberty = self._group(board, move)
        if not has_liberty:
            raise ValueError(f"suicide at {move}")
        ko = None
        if len(captured) == 1 and len(own) == 1:
            liberties = [neighbor for neighbor in self.rules.neighbors[move] if board[neighbor] == EMPTY]
            if len(liberties) == 1 and liberties[0] == captured[0]:
                ko = captured[0]
        return State(self.rules, board, opponent, ko, 0, self.moves + 1)

    def score(self):
        black = sum(value == BLACK for value in self.board)
        white = sum(value == WHITE for value in self.board)
        seen = set()
        for point in range(self.rules.n):
            if self.board[point] != EMPTY or point in seen:
                continue
            region = {point}
            stack = [point]
            touches_black = False
            touches_white = False
            while stack:
                current = stack.pop()
                for neighbor in self.rules.neighbors[current]:
                    value = self.board[neighbor]
                    if value == EMPTY and neighbor not in region:
                        region.add(neighbor)
                        stack.append(neighbor)
                    elif value == BLACK:
                        touches_black = True
                    elif value == WHITE:
                        touches_white = True
            seen.update(region)
            if touches_black and not touches_white:
                black += len(region)
            elif touches_white and not touches_black:
                white += len(region)
        return black - white - self.rules.komi

    def winner(self):
        score = self.score()
        return BLACK if score > 0 else WHITE if score < 0 else EMPTY
