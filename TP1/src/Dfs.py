# DFS visits one entire branch doing backtracking once arrived at the deepest node

from dataclasses import dataclass

from .Levels import simple
from .SearchSolver import SearchSolver, State, Node
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


class Dfs(SearchSolver):
    def reconstruct_path(self):
        return self._latest_node.path

    @measure_exec_time
    def solve(self):
        initial_state = self

        stack: list[Node] = [Node(initial_state, [])]
        visited = set()

        while stack:
            # Next movement
            current_node = stack.pop()

            if current_node.state.is_solved():
                self._latest_node = current_node
                return SearchSolverResult(
                    has_solution=True,
                    nodes_visited=len(visited),
                    path_len=len(current_node.path),
                    border_nodes=len(stack),
                )

            current_state = State(
                current_node.state.board.player,
                frozenset(current_node.state.board.boxes),
            )
            if current_state in visited:
                continue

            visited.add(current_state)

            current_player_position = current_node.state.board.player
            for move in current_node.state.get_possible_moves(current_player_position):
                new_state = current_node.state.move(current_player_position, move)

                if (
                    State(new_state.board.player, frozenset(new_state.board.boxes))
                    not in visited
                ):
                    stack.append(Node(new_state, current_node.path + [new_state]))

            self.step()

        return SearchSolverResult(
            has_solution=False,
            nodes_visited=len(visited),
            path_len=0,
            border_nodes=len(stack),
        )


if __name__ == "__main__":
    board = simple()

    game = Dfs(board)
    solution, exec_time = game.solve()
    if solution:
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")

    print(f"Took: {exec_time} ns")
