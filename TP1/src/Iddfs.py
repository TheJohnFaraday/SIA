# IDDFS explores nodes incrementally by depth, restarting from the root at each depth limit.
import time

from .Levels import simple
from .SearchSolver import SearchSolver
from .utils import measure_exec_time
from .Dfs import Dfs


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


class Iddfs(SearchSolver):
    def reconstruct_path(self):
        return self._latest_node.path

    @measure_exec_time
    def solve(self, max_depth: int = 2_000):
        for depth in range(max_depth + 1):
            dfs = Dfs(self.board)
            result = dfs.solve(max_depth=depth)

            self.step()

            if result is not None and result.has_solution:
                return result


if __name__ == "__main__":
    board = simple()
    print(board)

    game = Iddfs(board)
    solution = game.solve()
    if solution.has_solution:
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")

    '''
    for state in game.states:
        print(state.board)
    '''

    print(f"Took: {solution.execution_time_ns} ns")
