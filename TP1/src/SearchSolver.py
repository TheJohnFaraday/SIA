from enum import Enum

from Board import Board, Coordinates


class Directions(Enum):
    DOWN = Coordinates(y=1, x=0)
    UP = Coordinates(y=-1, x=0)
    LEFT = Coordinates(y=0, x=-1)
    RIGHT = Coordinates(y=0, x=1)


class SearchSolver:
    def __init__(
        self,
        board: Board,
        max_states_repeated: int = 20,
        states: list = None
    ):
        self.board = board
        self.max_states_repeated = max_states_repeated
        self.states = states if states else []
        self.execution_time = 0.0

    def __eq__(self, other: "SearchSolver"):
        return (
            self.board.player == other.board.player
            and self.board.boxes == other.board.boxes
        )

    def is_solved(self):
        return self.board.boxes == self.board.goals

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
        if self.board.cell(new_pos) == Board.Cell.WALL or self.board.cell(new_pos) == Board.Cell.BLOCK:
            return False

        if new_pos in self.board.boxes:
            next_pos = Coordinates(x=x + (x - player_pos.x), y=y + (y - player_pos.y))
            if (
                next_pos in self.board.boxes
                or self.board.cell(next_pos) == Board.Cell.WALL or self.board.cell(next_pos) == Board.Cell.BLOCK
            ):
                return False
        return True

    def move(self, player_position: Coordinates, new_player_position: Coordinates):
        new_board = self.board
        if new_player_position in self.board.boxes:
            next_box_position = Coordinates(
                x=new_player_position.x + (new_player_position.x - player_position.x),
                y=new_player_position.y + (new_player_position.y - player_position.y),
            )
            new_board = self.board.move_box(box=new_player_position, new_position=next_box_position)

        new_board = new_board.move_player(new_player_position)
        return self.__class__(
            new_board,
            max_states_repeated=self.max_states_repeated,
            states=self.states
        )
