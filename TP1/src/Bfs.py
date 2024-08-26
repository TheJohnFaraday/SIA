# BFS visits all nodes by level
from collections import deque

from .Levels import simple, narrow
from .SearchSolver import SearchSolver, Coordinates, Board
from .SearchSolverResult import SearchSolverResult
from .utils import measure_exec_time

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


class Bfs(SearchSolver):
    @measure_exec_time
    def solve(self):
        self.states = []

        initial_state = self

        queue: deque[SearchSolver] = deque([self])
        visited = {(initial_state.board.player, frozenset(initial_state.board.boxes))}

        while queue:
            current_state = queue.popleft()

            self.states.append(current_state)

            if current_state.is_solved():
                return SearchSolverResult(
                    has_solution=True,
                    nodes_visited=len(visited),
                    border_nodes=len(queue),
                )

            player_pos = current_state.board.player
            box_positions = current_state.board.boxes

            possible_moves = current_state.get_possible_moves(player_pos)
            for move in possible_moves:
                new_state = current_state.move(player_pos, move)
                if (
                    new_state.board.player,
                    frozenset(new_state.board.boxes),
                ) not in visited:
                    visited.add(
                        # TODO: Revisar cuál es correcto
                        # (new_state.board.player, frozenset(new_state.board.boxes))
                        (player_pos, frozenset(box_positions))
                    )
                    queue.append(new_state)
            self.step()

        return SearchSolverResult(
            has_solution=False,
            nodes_visited=len(visited),
            border_nodes=len(queue),
        )


if __name__ == "__main__":
    # board = Levels.random(seed=5, level=1)
    board = simple()
    print(board)

    game = Bfs(board)
    solution, exec_time = game.solve()
    if solution:
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")
    print(f"Steps: {game.steps}")

    for state in game.states:
        print(state.board)

    print(f"Took: {exec_time} ns")
