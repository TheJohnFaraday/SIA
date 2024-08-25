import logging
from dataclasses import dataclass

import numpy as np
from enum import Enum


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


class Board:
    class Cell(Enum):
        WALL = "#"
        PLAYER = "*"
        BOX = "@"
        GOAL = "X"
        EMPTY = " "

    def __init__(
        self,
        player: Coordinates,
        boxes: set[Coordinates],
        goals: set[Coordinates],
        board: list[list[Cell]] = None,
        n_rows: int = None,
        n_cols: int = None,
        blocks: list[Coordinates] = None,
    ):
        """Initialize a new Board

        Parameters:
            player: Player Coordinates.
            boxes: A set of boxes, it must have at least one box.
            goals: A set of goals, it must the same size as `boxes`.
            board: Provide a predefined, custom board.
                At least one of (board) or (n_rows, n_cols, blocks) must be provided.
                Mutually exclusive with the three other properties: n_rows, n_cols, blocks.
            n_rows: Number of rows a rectangular board must have, limit walls included.
                Ignored if `board` is provided.
            n_cols: Number of columns a rectangular board must have, limit walls included.
                Ignored if `board` is provided.
            blocks: Optional list of blocks that the board might have.
                Ignored if `board` is provided.
        """

        self._player = player
        self._boxes = boxes
        self._goals = goals

        if board:
            self._n_rows: int = len(board)
            self._n_cols: int = np.max([len(row) for row in board])
            self._board = board
        elif n_rows and n_cols:
            self._n_rows = n_rows
            self._n_cols = n_cols
            self._board = []
            for row_idx in range(self._n_rows):
                if row_idx == 0 or row_idx == self._n_rows - 1:
                    self._board.append(
                        [self.Cell.WALL for _ in range(0, self._n_cols)]
                    )
                    continue

                row = []
                for col_idx in range(self._n_cols):
                    if col_idx == 0 or col_idx == self._n_cols - 1:
                        row.append(self.Cell.WALL)
                        continue

                    if blocks and Coordinates(x=col_idx, y=row_idx) in blocks:
                        row.append(self.Cell.WALL)
                    else:
                        row.append(self.Cell.EMPTY)

                self._board.append(row)
        else:
            logging.error(
                "Provide a whole board or the required parameters to generate a board:"
                " rows, cols, and optionally a list of blocks"
            )

        if not self.is_valid():
            logging.error("Initialized with an invalid board")
        self.update_board()

    def __str__(self):
        board_str = ""
        for row_idx, row in enumerate(self._board):
            for col_idx, col in enumerate(row):
                point = Coordinates(x=col_idx, y=row_idx)
                if point == self.player:
                    if point in self.boxes:
                        logging.warning(
                            f"Invalid state: player is in the same position as a box! Coordinates: {point}"
                        )

                    board_str += self.Cell.PLAYER.value
                elif point in self.boxes:
                    board_str += self.Cell.BOX.value
                elif point in self.goals:
                    board_str += self.Cell.GOAL.value
                else:
                    board_str += col.value

            board_str += "\n"

        return board_str[:-1]

    @property
    def player(self):
        return self._player

    @property
    def boxes(self):
        return self._boxes

    @property
    def goals(self):
        return self._goals

    def cell(self, coordinates: Coordinates):
        return self._board[coordinates.y][coordinates.x]

    def is_valid(self):
        for row_idx, row in enumerate(self._board):
            for col_idx, col in enumerate(row):
                point = Coordinates(x=col_idx, y=row_idx)

                if col == self.Cell.WALL:
                    if point == self.player:
                        logging.warning(
                            f"Invalid state: player is in the same position as a block! Coordinates: {point}"
                        )
                        return False
                    elif point in self.boxes:
                        logging.warning(
                            f"Invalid state: box is in the same position as a block! Coordinates: {point}"
                        )
                        return False
                    elif point in self.goals:
                        logging.warning(
                            f"Invalid state: box is in the same position as a block! Coordinates: {point}"
                        )
                        return False
                elif col == self.Cell.WALL:
                    if point == self.player:
                        logging.warning(
                            f"Invalid state: player is in the same position as a wall! Coordinates: {point}"
                        )
                        return False
                    elif point in self.boxes:
                        logging.warning(
                            f"Invalid state: box is in the same position as a wall! Coordinates: {point}"
                        )
                        return False
                    elif point in self.goals:
                        logging.warning(
                            f"Invalid state: box is in the same position as a wall! Coordinates: {point}"
                        )
                        return False

        return True

    def move_player(self, new_position: Coordinates):
        return self.__class__(
            player=new_position,
            boxes=self._boxes,
            goals=self._goals,
            board=self._board,
        )

    def move_box(self, box: Coordinates, new_position: Coordinates):
        if box not in self.boxes:
            return None

        new_boxes = set(self._boxes)
        new_boxes.remove(box)
        new_boxes.add(new_position)

        return self.__class__(
            player=new_position,
            boxes=new_boxes,
            goals=self._goals,
            board=self._board,
        )

    def update_board(self):
        for i in range(self._n_rows):
            for j in range(self._n_cols):
                if (self._board[i][j] == Board.Cell.PLAYER
                        or self._board[i][j] == Board.Cell.BOX):
                    self._board[i][j] = Board.Cell.EMPTY
        for goal in self._goals:
            self._board[goal.y][goal.x] = Board.Cell.GOAL
        for box in self._boxes:
            self._board[box.y][box.x] = Board.Cell.BOX
        self._board[self._player.y][self._player.x] = Board.Cell.PLAYER
