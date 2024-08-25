# DFS visits one entire branch doing backtracking once arrived at the deepest node
from enum import Enum
from SearchSolver import SearchSolver, Coordinates


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


class Dfs(SearchSolver):

    def solve(self):
        # The stack is going to persist our frontier states
        stack = [(self.player_pos, self.box_positions)]
        self.visited.add((self.player_pos, frozenset(self.box_positions)))
        pasos = 0

        while stack:
            # Next movement
            self.player_pos, self.box_positions = stack.pop()
            # print(self.player_pos, self.box_positions)

            if self.is_solved():
                print(f'#### PASOS: {pasos}')
                return True

            possible_moves = self.get_possible_moves(self.player_pos)
            for move in possible_moves:
                new_state = self.move(self.player_pos, move)
                if (
                    new_state.player_pos,
                    frozenset(new_state.box_positions),
                ) not in self.visited:
                    stack.append((new_state.player_pos, new_state.box_positions))
                    self.visited.add((self.player_pos, frozenset(self.box_positions)))
            pasos += 1
        print(f'CANTIDAD DE PASOS: {pasos}')
        return False


if __name__ == "__main__":
    '''
    board = ["#######", "#     #", "# # # #", "#X@*@X#", "#######"]

    player_pos = Coordinates(y=3, x=3)
    box_positions = [Coordinates(y=3, x=4), Coordinates(y=3, x=2)]
    goal_positions = [Coordinates(y=3, x=1), Coordinates(y=3, x=5)]
    '''

    board = [
        '##### ',
        '# X ##',
        '#    #',
        '# X@ #',
        '###@ #',
        '  #  #',
        '  # *#',
        '  ####'
    ]

    player_pos = Coordinates(y=6, x=4)
    box_positions = [Coordinates(y=3, x=3), Coordinates(y=4, x=3)]
    goal_positions = [Coordinates(y=1, x=2), Coordinates(y=3, x=2)]

    game = Dfs(board, player_pos, box_positions, goal_positions)
    if game.solve():
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")
