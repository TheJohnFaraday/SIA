# DFS visits one entire branch doing backtracking once arrived at the deepest node
from enum import Enum
from SearchSolver import SearchSolver

# Sokoban board

#   #######            # -> wall
#   #     #            * -> player
#   # # # #            @ -> box
#   #X *@ #            X -> goal
#   #######

# player position is represented by a tuple. For example, (2, 1) means that the player is in the string at position 2
# and character at position 1 as shown in the picture above (bah is a matrix). The same for box and goal positions

# The state is defined by the player and boxes current position. States are unique! I think so xd

# How we can represent a movement? Our player only can move in four directions: up, down, left and right
# We can think a movement like the increment or decrement of a position :
# (-1, 0) going up; (1, 0) down; (0, -1) left; (0, 1) right

# How we can represent the action of moving a box? If we do a movement and our new position is coincident with
# a box position... well it means that the box has to be moved in the direction that the player moved. Anyway
# we must check if it is possible to move the box

class Directions(Enum):
    DOWN = (-1, 0)
    UP = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class Dfs(SearchSolver):

    def is_solved(self):
        return self.box_positions == self.goal_positions

    def get_possible_moves(self, player_pos):
        possible_moves = []

        for move in Directions:
            new_pos = (player_pos[0] + move.value[0], player_pos[1] + move.value[1])
            if self.is_valid_move(player_pos, new_pos):
                possible_moves.append(new_pos)

        return possible_moves

    def is_valid_move(self, player_pos, new_pos):
        x, y = new_pos
        if self.board[x][y] == '#':
            # Me la di en la pera
            return False
        if new_pos in self.box_positions:
            # Can we push the box? Imagine right now that we are the box
            next_pos = (x + (x - player_pos[0]), y + (y - player_pos[1]))
            if next_pos in self.box_positions or self.board[next_pos[0]][next_pos[1]] == '#':
                # There is another box or the wall
                return False
        return True

    def move(self, player_pos, new_pos):
        # We make the movement effective
        new_box_positions = set(self.box_positions)
        if new_pos in new_box_positions:
            # Moving the box
            next_pos = (new_pos[0] + (new_pos[0] - player_pos[0]), new_pos[1] + (new_pos[1] - player_pos[1]))
            new_box_positions.remove(new_pos)
            new_box_positions.add(next_pos)

        return Dfs(self.board, new_pos, new_box_positions, self.goal_positions)

    def solve(self):
        # The stack is going to persist our frontier states
        stack = [(self.player_pos, self.box_positions)]
        self.visited.add((self.player_pos, frozenset(self.box_positions)))

        while stack:
            # Next movement
            self.player_pos, self.box_positions = stack.pop()
            print(self.player_pos, self.box_positions)

            if self.is_solved():
                return True

            possible_moves = self.get_possible_moves(self.player_pos)
            for move in possible_moves:
                new_state = self.move(self.player_pos, move)
                if (new_state.player_pos, frozenset(new_state.box_positions)) not in self.visited:
                    stack.append((new_state.player_pos, new_state.box_positions))
                    self.visited.add((self.player_pos, frozenset(self.box_positions)))

        return False


if __name__ == '__main__':
    board = [
        "#######",
        "#     #",
        "# # # #",
        "#X@*@X#",
        "#######"
    ]

    player_pos = (3, 3)
    box_positions = [(3, 4), (3, 2)]
    goal_positions = [(3, 1), (3, 5)]

    game = Dfs(board, player_pos, box_positions, goal_positions)
    if game.solve():
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")