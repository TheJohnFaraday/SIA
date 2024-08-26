from enum import Enum
from dataclasses import dataclass

from .Board import Board, Coordinates


@dataclass(frozen=True, eq=True)
class State:
    player_pos: Coordinates
    box_positions: set[Coordinates]

    def as_tuple(self):
        return self.player_pos, self.box_positions

    @staticmethod
    def from_tuple(t: tuple[Coordinates, set[Coordinates]]):
        return State(player_pos=t[0], box_positions=t[1])

    def __iter__(self):
        return iter((self.player_pos, self.box_positions))


class Directions(Enum):
    DOWN = Coordinates(y=1, x=0)
    UP = Coordinates(y=-1, x=0)
    LEFT = Coordinates(y=0, x=-1)
    RIGHT = Coordinates(y=0, x=1)


class SearchSolver:
    def __init__(
        self, board: Board, max_states_repeated: int = 20, states: list = None, steps: int = 0, frontier_nodes: int = 0
    ):
        self.board = board
        self.max_states_repeated = max_states_repeated
        self.states = states if states else []
        self._steps = steps
        self._frontier_nodes = frontier_nodes

    def __eq__(self, other: "SearchSolver"):
        return (
            self.board.player == other.board.player
            and self.board.boxes == other.board.boxes
        )

    @property
    def steps(self):
        return self._steps

    def step(self):
        self._steps += 1

    @property
    def frontier_nodes(self):
        return self._frontier_nodes

    def increment_frontier_nodes(self):
        self._frontier_nodes += 1

    def is_solved(self):
        return self.board.boxes == self.board.goals

    def get_possible_moves(self, player_pos: Coordinates):
        directions = [Directions.DOWN, Directions.UP,
                      Directions.LEFT, Directions.RIGHT]
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
        if (
            self.board.cell(new_pos) == Board.Cell.WALL
            or self.board.cell(new_pos) == Board.Cell.WALL
        ):
            return False

        if new_pos in self.board.boxes:
            next_pos = Coordinates(x=x + (x - player_pos.x),
                                   y=y + (y - player_pos.y))
            if (
                next_pos in self.board.boxes
                or self.board.cell(next_pos) == Board.Cell.WALL
                or self.board.cell(next_pos) == Board.Cell.WALL
            ):
                return False
        return True

    def is_deadlock(self):
        board = self.board
        print('Deadlock :((((')
        for box in self.board.boxes:
            # Check if the box is in a corner

            right = Coordinates(y=box.y, x=box.x + 1)
            above = Coordinates(y=box.y - 1, x=box.x)
            left = Coordinates(y=box.y, x=box.x - 1)
            below = Coordinates(y=box.y + 1, x=box.x)

            if (Coordinates(y=box.y, x=box.x) not in board.goals
                and
                ((
                    board.cell(right) == Board.Cell.WALL
                    and board.cell(below) == Board.Cell.WALL
                )
                or (
                    board.cell(left) == Board.Cell.WALL
                    and board.cell(below) == Board.Cell.WALL
                )
                or (
                    board.cell(right) == Board.Cell.WALL
                    and board.cell(above) == Board.Cell.WALL
                )
                or (
                    board.cell(left) == Board.Cell.WALL
                    and board.cell(above) == Board.Cell.WALL
                    ))):
                return True

        return False

    def move(self,
             player_position: Coordinates,
             new_player_position: Coordinates):
        new_board = self.board
        if new_player_position in self.board.boxes:
            next_box_position = Coordinates(
                x=new_player_position.x + (new_player_position.x
                                           - player_position.x),
                y=new_player_position.y + (new_player_position.y
                                           - player_position.y),
            )
            new_board = self.board.move_box(
                box=new_player_position, new_position=next_box_position
            )

        new_board = new_board.move_player(new_player_position)
        return self.__class__(
            new_board,
            max_states_repeated=self.max_states_repeated,
            states=self.states,
            steps=self._steps
        )
