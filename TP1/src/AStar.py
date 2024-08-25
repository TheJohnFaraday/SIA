import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Union

import Levels

from Heuristics import euclidean, manhattan
from SearchSolver import SearchSolver, Coordinates, Board


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


@dataclass(frozen=True, eq=True)
class AStarNode:
    state: SearchSolver = field(compare=False)
    g_value: int
    h_value: int
    parent: Union[None, "AStarNode"]

    def priority(self):
        return self.g_value + self.h_value

    def __lt__(self, other):
        return self.priority() < other.priority()


class AStar(SearchSolver):
    def __init__(
        self, board: Board, max_states_repeated: int = 20, states: list = None
    ):
        super().__init__(board, max_states_repeated=max_states_repeated, states=states)

        self.came_from: list[AStarNode] = []
        self.latest_node: AStarNode | None = None

    def reconstruct_path(self):
        path = []
        for node in self.came_from:
            path.append(node.state.board)

        return path

    @staticmethod
    def move_cost(current: Coordinates, neighbor: Coordinates):
        # TODO: Deadlock must return infinite cost
        return 1

    def solve(self, heuristic: callable(SearchSolver)):
        timestamp = time.perf_counter_ns()

        initial_node = AStarNode(state=self, g_value=0, h_value=0, parent=None)

        open_set: list[tuple[int, AStarNode]] = []
        heapq.heapify(open_set)
        heapq.heappush(
            open_set,
            (heuristic(initial_node.state), initial_node),
        )

        nodes_visited: dict[Coordinates, AStarNode] = {
            initial_node.state.board.player: initial_node
        }
        self.came_from.append(initial_node)

        previous_node: AStarNode | None = None
        repeated_states = 0

        while open_set:
            current_node: AStarNode
            _, current_node = heapq.heappop(open_set)

            self.latest_node = current_node
            self.states.append(current_node)

            if current_node.state.is_solved():
                self.execution_time = time.perf_counter_ns() - timestamp
                return True

            # Avoid loop
            if current_node == previous_node:
                repeated_states += 1
            else:
                repeated_states = 0

            if repeated_states > self.max_states_repeated:
                self.execution_time = time.perf_counter_ns() - timestamp
                return False

            previous_node = current_node
            # == END == Avoid loop

            current_player_pos = current_node.state.board.player

            for neighbor in current_node.state.get_possible_moves(current_player_pos):
                new_state = current_node.state.move(
                    player_position=current_player_pos,
                    new_player_position=neighbor,
                )

                tentative_g_score = current_node.g_value + AStar.move_cost(
                    current_player_pos, neighbor
                )

                neighbor_came = nodes_visited.get(neighbor, None)

                if (
                    not neighbor_came
                    or neighbor_came.state != new_state
                    or tentative_g_score < neighbor_came.g_value
                ):
                    next_node = AStarNode(
                        state=new_state,
                        g_value=tentative_g_score,
                        h_value=heuristic(new_state),
                        parent=current_node,
                    )

                    nodes_visited[neighbor] = next_node
                    heapq.heappush(open_set, (next_node.priority(), next_node))
                    self.came_from.append(next_node)

        self.execution_time = time.perf_counter_ns() - timestamp
        return False


if __name__ == "__main__":
    board = Levels.simple()

    game = AStar(board)

    print("Euclidean")
    if game.solve(euclidean):
        print("¡Solución encontrada!")

        path = game.reconstruct_path()
        print("Path:")
        for p in path:
            print(p)
    else:
        print("No se encontró solución.")

    print(f"Took: {game.execution_time} ns")

    print("Manhattan")
    if game.solve(manhattan):
        print("¡Solución encontrada!")

        path = game.reconstruct_path()
        print("Path:")
        for p in path:
            print(p)
    else:
        print("No se encontró solución.")

    print(f"Took: {game.execution_time} ns")
