from enum import Enum


class SearchSolver:
    def __init__(self, board, player_pos, box_positions, goal_positions):
        self.board = board
        self.player_pos = player_pos
        self.box_positions = set(box_positions)
        self.goal_positions = set(goal_positions)
        self.visited = set()

    def is_solved(self):
        return self.box_positions == self.goal_positions

    def get_possible_moves(self, player_pos):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        possible_moves = []

        for move in directions:
            new_pos = (player_pos[0] + move[0], player_pos[1] + move[1])
            if self.is_valid_move(player_pos, new_pos):
                possible_moves.append(new_pos)

        return possible_moves

    def is_valid_move(self, player_pos, new_pos):
        x, y = new_pos
        if self.board[x][y] == '#':
            return False
        if new_pos in self.box_positions:
            next_pos = (x + (x - player_pos[0]), y + (y - player_pos[1]))
            if next_pos in self.box_positions or self.board[next_pos[0]][next_pos[1]] == '#':
                return False
        return True

    def move(self, player_pos, new_pos):
        new_box_positions = set(self.box_positions)
        if new_pos in new_box_positions:
            next_pos = (new_pos[0] + (new_pos[0] - player_pos[0]), new_pos[1] + (new_pos[1] - player_pos[1]))
            new_box_positions.remove(new_pos)
            new_box_positions.add(next_pos)

        return self.__class__(self.board, new_pos, new_box_positions, self.goal_positions)


class Directions(Enum):
    DOWN = (-1, 0)
    UP = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
