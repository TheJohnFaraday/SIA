from enum import Enum
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Coordinates:
    x: int
    y: int

    def as_tuple(self):
        return self.x, self.y

    @staticmethod
    def from_tuple(t: tuple[int, int]):
        return Coordinates(x=t[0], y=t[1])

    def __iter__(self):
        return iter((self.x, self.y))


class Directions(Enum):
    DOWN = Coordinates(y=1, x=0)
    UP = Coordinates(y=-1, x=0)
    LEFT = Coordinates(y=0, x=-1)
    RIGHT = Coordinates(y=0, x=1)


class SearchSolver:
    def __init__(
        self,
        board,
        player_pos: Coordinates,
        box_positions: list[Coordinates] | set[Coordinates],
        goal_positions: list[Coordinates] | set[Coordinates],
        max_states_repeated: int = 20,
    ):
        self.board = board
        self.player_pos = player_pos
        self.box_positions = set(box_positions)
        self.goal_positions = set(goal_positions)
        self.visited = set()

        self.max_states_repeated = max_states_repeated

    def __eq__(self, other: "SearchSolver"):
        return (
            self.player_pos == other.player_pos
            and self.box_positions == other.box_positions
        )

    def is_solved(self):
        return self.box_positions == self.goal_positions

    def get_possible_moves(self, player_pos: Coordinates):
        directions = [Directions.DOWN, Directions.UP, Directions.LEFT, Directions.RIGHT]
        possible_moves = []

        for move in directions:
            new_pos = Coordinates(
                x=player_pos.x + move.value.x, y=player_pos.y + move.value.y
            )
            if self.is_valid_move(player_pos, new_pos):
                possible_moves.append(new_pos)

        return possible_moves

    def is_valid_move(self, player_pos: Coordinates, new_pos: Coordinates):
        x, y = new_pos
        if self.board[y][x] == "#":
            return False
        if new_pos in self.box_positions:
            next_pos = Coordinates(x=x + (x - player_pos.x), y=y + (y - player_pos.y))
            if (
                next_pos in self.box_positions
                or self.board[next_pos.y][next_pos.x] == "#"
            ):
                return False
        return True

    def move(self, player_pos: Coordinates, new_pos: Coordinates):
        new_box_positions = set(self.box_positions)
        if new_pos in new_box_positions:
            next_pos = Coordinates(
                x=new_pos.x + (new_pos.x - player_pos.x),
                y=new_pos.y + (new_pos.y - player_pos.y),
            )
            new_box_positions.remove(new_pos)
            new_box_positions.add(next_pos)

        return self.__class__(
            self.board,
            new_pos,
            new_box_positions,
            self.goal_positions,
            max_states_repeated=self.max_states_repeated,
        )
