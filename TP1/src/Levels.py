from .Board import Coordinates, Board
from .LevelGenerator import LevelGenerator


def random(seed: int = 3, level: int = 1):
    generator = LevelGenerator(seed)
    board, player, boxes, goals = generator.generate(level)
    return Board(
        player=player,
        boxes=boxes,
        goals=goals,
        board=board
    )


def simple():
    return Board(
        player=Coordinates(x=3, y=3),
        boxes={Coordinates(y=3, x=4), Coordinates(y=3, x=2)},
        goals={Coordinates(y=3, x=1), Coordinates(y=3, x=5)},
        n_rows=5,
        n_cols=7,
        blocks=[Coordinates(y=2, x=2), Coordinates(y=2, x=4)],
    )


def narrow():
    return Board(
        player=Coordinates(y=6, x=4),
        boxes={Coordinates(y=3, x=3), Coordinates(y=4, x=3)},
        goals={Coordinates(y=1, x=2), Coordinates(y=3, x=2)},
        board=[
            [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY],
            [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL],
            [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
            [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
            [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
            [Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
            [Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
            [Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
        ]
    )


def unsolvable():
    return Board(
        player=Coordinates(y=2, x=2),
        boxes={
            Coordinates(y=2, x=3),
            Coordinates(y=3, x=4),
            Coordinates(y=4, x=4),
            Coordinates(y=6, x=1),
            Coordinates(y=6, x=3),
            Coordinates(y=6, x=4),
            Coordinates(y=6, x=5),
        },
        goals={
            Coordinates(y=2, x=1),
            Coordinates(y=3, x=5),
            Coordinates(y=4, x=1),
            Coordinates(y=5, x=4),
            Coordinates(y=6, x=6),
            Coordinates(y=7, x=4),
        },
        board=[
            [
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
            ],
            [
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.WALL,
                Board.Cell.WALL,
            ],
            [
                Board.Cell.WALL,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.WALL,
                Board.Cell.WALL,
            ],
            [
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.WALL,
                Board.Cell.WALL,
            ],
            [
                Board.Cell.WALL,
                Board.Cell.EMPTY,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.WALL,
                Board.Cell.WALL,
            ],
            [
                Board.Cell.WALL,
                Board.Cell.EMPTY,
                Board.Cell.WALL,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.WALL,
                Board.Cell.WALL,
            ],
            [
                Board.Cell.WALL,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.WALL,
            ],
            [
                Board.Cell.WALL,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.EMPTY,
                Board.Cell.WALL,
            ],
            [
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
                Board.Cell.WALL,
            ],
        ],
    )


def juani():
   return Board(
       player=Coordinates(y=3, x=2),
       boxes={Coordinates(y=1, x=7), Coordinates(y=2, x=2), Coordinates(y=3, x=8)},
       goals={Coordinates(y=1, x=1), Coordinates(y=2, x=8), Coordinates(y=5, x=8)},
       board = [
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
       ]
   )

def wiki():
   return Board(
       player=Coordinates(y=2, x=2),
       boxes={Coordinates(y=2, x=3), Coordinates(y=3, x=4), Coordinates(y=4, x=4), Coordinates(y=6, x=1),
               Coordinates(y=6,x=3),Coordinates(y=6,x=4), Coordinates(y=6,x=5)},
       goals={Coordinates(y=2, x=1), Coordinates(y=3, x=5), Coordinates(y=4, x=1), Coordinates(y=5, x=4),
               Coordinates(y=6,x=6),Coordinates(y=7,x=4), Coordinates(y=6,x=3)},
       board = [
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
       ]
   )


def level10():
   #Optimal solution: 23
   return Board(
       player=Coordinates(y=5, x=4),
       boxes={Coordinates(y=3, x=3), Coordinates(y=4, x=3)},
       goals={Coordinates(y=1, x=2), Coordinates(y=3, x=3)},
       board=[
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL,Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL]
       ]
   )


def level14():
   #Optimal solution: 29
   return Board(
       player=Coordinates(y=2, x=4),
       boxes={Coordinates(y=2, x=3), Coordinates(y=3, x=2)},
       goals={Coordinates(y=3, x=4), Coordinates(y=4, x=4)},
       board=[
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL]
       ]
   )


def level16():
   #Optimal solution: 22
   return Board(
       player=Coordinates(y=1, x=2),
       boxes={Coordinates(y=2, x=2), Coordinates(y=2, x=3)},
       goals={Coordinates(y=1, x=3), Coordinates(y=1, x=4)},
       board=[
           [Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY,Board.Cell.EMPTY,Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL]
       ]
   )


def level20():
   #Optimal solution: 33
   return Board(
       player=Coordinates(y=5, x=2),
       boxes={Coordinates(y=3, x=1), Coordinates(y=4, x=3)},
       goals={Coordinates(y=3, x=1), Coordinates(y=1, x=2)},
       board=[
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY]
       ]
   )

def level40():
   #Optimal solution: 23
   return Board(
       player=Coordinates(y=4, x=1),
       boxes={Coordinates(y=3, x=1), Coordinates(y=3, x=2), Coordinates(y=3, x=3), Coordinates(y=3, x=4)},
       goals={Coordinates(y=3, x=1), Coordinates(y=3, x=2), Coordinates(y=3, x=4),Coordinates(y=4, x=4)},
       board=[
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL]
       ]
   )

def level53():
   #Optimal solution: 19
   return Board(
       player=Coordinates(y=1, x=4),
       boxes={Coordinates(y=3, x=2), Coordinates(y=3, x=3), Coordinates(y=3, x=4)},
       goals={Coordinates(y=2, x=1), Coordinates(y=4, x=3), Coordinates(y=4, x=4)},
       board=[
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL]
       ]
   )

def level57():
   #Optimal solution: 22
   return Board(
       player=Coordinates(y=1, x=3),
       boxes={Coordinates(y=1, x=2), Coordinates(y=3, x=2), Coordinates(y=2, x=3)},
       goals={Coordinates(y=1, x=2), Coordinates(y=2, x=2), Coordinates(y=1, x=4)},
       board=[
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL]
       ]
   )

def level62():
   #Optimal solution: 24
   return Board(
       player=Coordinates(y=1, x=4),
       boxes={Coordinates(y=1, x=3), Coordinates(y=2, x=3), Coordinates(y=3, x=3), Coordinates(y=4, x=3)},
       goals={Coordinates(y=1, x=4), Coordinates(y=2, x=4), Coordinates(y=4, x=2),Coordinates(y=4, x=4)},
       board=[
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL]
       ]
   )

def level65():
   #Optimal solution: 25
   return Board(
       player=Coordinates(y=2, x=4),
       boxes={Coordinates(y=2, x=3), Coordinates(y=3, x=2), Coordinates(y=4, x=3)},
       goals={Coordinates(y=1, x=3), Coordinates(y=3, x=1), Coordinates(y=3, x=4)},
       board=[
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
           [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL]
       ]
   )

def level67():
   #Optimal solution: 25
   return Board(
       player=Coordinates(y=4, x=1),
       boxes={Coordinates(y=3, x=1), Coordinates(y=3, x=2), Coordinates(y=3, x=3), Coordinates(y=3, x=4)},
       goals={Coordinates(y=3, x=1), Coordinates(y=3, x=2), Coordinates(y=3, x=4),Coordinates(y=4, x=4)},
       board=[
          [Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY],
          [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL],
          [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
          [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
          [Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL],
          [Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL],
          [Board.Cell.EMPTY, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.WALL, Board.Cell.EMPTY],
        ]
   )


