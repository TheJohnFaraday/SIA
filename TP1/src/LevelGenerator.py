import random as rnd

from .Board import Coordinates, Board


class LevelGenerator:
    def __init__(self, seed: int = 3):
        self.seed = seed

    def generate(self, difficulty_level: int):
        rnd.seed(self.seed)
        box_number = 0
        level_size = 0
        match difficulty_level:
            case 1:
                level_size = rnd.randint(7, 9)
                box_number = rnd.randint(2, 4)
            case 2:
                level_size = rnd.randint(8, 12)
                box_number = rnd.randint(3, 5)
            case 3:
                level_size = rnd.randint(9, 13)
                box_number = rnd.randint(4, 5)
            case 4:
                level_size = rnd.randint(10, 14)
                box_number = rnd.randint(5, 7)
            case _:
                level_size = rnd.randint(11, 16)
                box_number = rnd.randint(8, 10)

        # Grid generation. Everything will be empty except for the outer walls
        grid = []
        for i in range(0, level_size):
            row = []
            for j in range(0, level_size):
                if (
                    (i == 0)
                    or (i == level_size - 1)
                    or (j == 0)
                    or (j == level_size - 1)
                ):
                    row.append(Board.Cell.WALL)
                else:
                    row.append(Board.Cell.EMPTY)
            grid.append(row)

        goals = box_number

        goals_location = set()
        # Randomly place the solution buttons
        while goals > 0:
            i = rnd.randint(1, level_size - 2)
            j = rnd.randint(1, level_size - 2)
            if (Coordinates(y=i, x=j) not in goals_location
                    and grid[i][j] == Board.Cell.EMPTY):
                goals -= 1
                goals_location.add(Coordinates(y=i, x=j))

        boxes_location = set()
        # Randomly place the boxes (outside the buttons)
        while box_number > 0:
            i = rnd.randint(2, level_size - 3)
            j = rnd.randint(2, level_size - 3)
            if (Coordinates(y=i, x=j) not in goals_location
                and Coordinates(y=i, x=j) not in boxes_location
                    and grid[i][j] == Board.Cell.EMPTY):
                box_number -= 1
                boxes_location.add(Coordinates(y=i, x=j))

        player_position = None
        # Randomly place the player somewhere in the board
        flag = True
        while flag:
            i = rnd.randint(2, level_size - 3)
            j = rnd.randint(2, level_size - 3)
            if (Coordinates(y=i, x=j) not in goals_location
                and Coordinates(y=i, x=j) not in boxes_location
                    and grid[i][j] == Board.Cell.EMPTY):
                grid[i][j] = Board.Cell.PLAYER
                flag = False
                player_position = Coordinates(y=i, x=j)

        # Randomply place walls on the game
        random_spaces = level_size * 2 - 10
        while random_spaces > 0:
            i = rnd.randint(1, level_size - 2)
            j = rnd.randint(1, level_size - 2)
            if (Coordinates(y=i, x=j) not in goals_location
                and Coordinates(y=i, x=j) not in boxes_location
                and Coordinates(y=j, x=j) != player_position
                    and grid[i][j] == Board.Cell.EMPTY):
                grid[i][j] = Board.Cell.WALL
                random_spaces -= 1

        return (grid, player_position, boxes_location, goals_location)


# "#######"  <- # == wall
# "#     #"  <-   == empty space
# "# # # #"  <- X == button
# "#x@*@x#"  <- @ == box
# "#######"  <- * == player

if __name__ == "__main__":
    level_generator = LevelGenerator(3)
    grid, player_position, boxes_location, buttons_location = level_generator.generate(
        1
    )
    print(
        f"LEVEL ONE: PP - {player_position} | BOXL - {boxes_location} | BUTL - {buttons_location}"
    )
    print("##### MAP RESULT #####")
    print(grid)
    grid, player_position, boxes_location, buttons_location = level_generator.generate(
        2
    )
    print(
        f"LEVEL TWO: PP - {player_position} | BOXL - {boxes_location} | BUTL - {buttons_location}"
    )
    print("##### MAP RESULT #####")
    print(grid)
    grid, player_position, boxes_location, buttons_location = level_generator.generate(
        3
    )
    print(
        f"LEVEL THREE: PP - {player_position} | BOXL - {boxes_location} | BUTL - {buttons_location}"
    )
    print("##### MAP RESULT #####")
    print(grid)
    grid, player_position, boxes_location, buttons_location = level_generator.generate(
        4
    )
    print(
        f"LEVEL FOUR: PP - {player_position} | BOXL - {boxes_location} | BUTL - {buttons_location}"
    )
    print("##### MAP RESULT #####")
    print(grid)
    grid, player_position, boxes_location, buttons_location = level_generator.generate(
        5
    )
    print(
        f"LEVEL FIVE: PP - {player_position} | BOXL - {boxes_location} | BUTL - {buttons_location}"
    )
    print("##### MAP RESULT #####")
    print(grid)
