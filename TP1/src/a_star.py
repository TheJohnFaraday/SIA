import heapq
import math
import time

from dataclasses import dataclass, field
from collections import deque
from SearchSolver import SearchSolver, Coordinates
from typing import Callable, Union

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
        self,
        board,
        player_pos: tuple[int, int],
        box_positions: list[tuple[int, int]] | set[tuple[int, int]],
        goal_positions: list[tuple[int, int]] | set[tuple[int, int]],
    ):
        super().__init__(
            board,
            player_pos=player_pos,
            box_positions=box_positions,
            goal_positions=goal_positions,
        )

        self.came_from: list[AStarNode] = []
        self.latest_node: AStarNode | None = None
        self.execution_time = 0.0

    def get_possible_moves(self, player_pos: Coordinates):
        possible_moves = [
            Coordinates.from_tuple(move)
            for move in super().get_possible_moves(player_pos.as_tuple())
        ]

        return possible_moves

    def move(
        self,
        player_pos: Coordinates | tuple[int, int],
        new_pos: Coordinates | tuple[int, int],
    ):
        if type(player_pos) is Coordinates:
            player_pos = player_pos.as_tuple()

        if type(new_pos) is Coordinates:
            new_pos = new_pos.as_tuple()

        return super().move(player_pos, new_pos)

    def reconstruct_path(self):
        return [
            Coordinates.from_tuple(node.state.player_pos) for node in self.came_from
        ]

    @staticmethod
    def move_cost(current: Coordinates, neighbor: Coordinates):
        # TODO: Deadlock must return infinite cost
        return 1

    def solve(self, heuristic: callable(SearchSolver)):
        timestamp = time.perf_counter_ns()

        start = AStarNode(state=self, g_value=0, h_value=0, parent=None)

        open_set: list[tuple[int, AStarNode]] = []
        heapq.heapify(open_set)
        heapq.heappush(
            open_set,
            (heuristic(start.state), start),
        )

        nodes_visited: dict[Coordinates, AStarNode] = {
            Coordinates.from_tuple(self.player_pos): start
        }
        self.came_from.append(start)

        while open_set:
            _, current_node = heapq.heappop(open_set)

            self.latest_node = current_node

            print(
                f"Player: {current_node.state.player_pos} | Boxes: {current_node.state.box_positions}"
            )

            if current_node.state.is_solved():
                self.execution_time = time.perf_counter_ns() - timestamp
                return True

            current_player_pos = Coordinates.from_tuple(current_node.state.player_pos)

            for neighbor in current_node.state.get_possible_moves(current_player_pos):
                new_state = current_node.state.move(
                    player_pos=current_player_pos,
                    new_pos=neighbor,
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
    board = ["#######", "#     #", "# # # #", "#X@*@X#", "#######"]

    player_pos = (3, 3)
    box_positions = [(3, 4), (3, 2)]
    goal_positions = [(3, 1), (3, 5)]

    game = AStar(board, player_pos, box_positions, goal_positions)

    def h(state: SearchSolver) -> int:
        heuristic = 0
        player_x, player_y = state.player_pos

        for box_x, box_y in state.box_positions:
            if (box_x, box_y) in state.goal_positions:
                continue

            # Player to Box distance
            heuristic += math.sqrt((player_x - box_x) ** 2 + (player_y - box_y) ** 2)

            # Box to goal distance
            heuristic = sum(
                (
                    math.sqrt((box_x - goal_x) ** 2 + abs(box_y - goal_y) ** 2)
                    for (goal_x, goal_y) in state.goal_positions
                ),
                heuristic,
            )
        return heuristic

    if game.solve(h):
        print("¡Solución encontrada!")

        path = game.reconstruct_path()
        print("Path:")
        print(path)
    else:
        print("No se encontró solución.")

    print(f"Took: {game.execution_time} ns")
