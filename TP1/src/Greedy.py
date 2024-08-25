# Greedy chooses based on an heuristic which node to visit
import heapq as pq
import math
import time

from dataclasses import dataclass, field
from SearchSolver import SearchSolver, Coordinates, Board
from Heuristics import euclidean, manhattan, minimum_matching_lower_bound


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


@dataclass(order=True)
class StatePriority:
    priority: int | float
    state: SearchSolver = field(compare=False)

    def __eq__(self, other: "StatePriority"):
        if not other:
            return False

        if type(other) is not StatePriority:
            return False

        if type(self.priority) is int and type(other.priority) is int:
            eq_priority = self.priority == other.priority
        else:
            eq_priority = math.isclose(self.priority, other.priority, rel_tol=1e-5)

        return eq_priority and self.state == other.state


class Greedy(SearchSolver):

    def solve(self, heuristic: callable(SearchSolver)):
        timestamp = time.perf_counter_ns()
        self.states: list[StatePriority] = []

        initial_state = StatePriority(heuristic(self), self)

        queue: list[StatePriority] = []
        path = []

        pq.heapify(queue)
        pq.heappush(queue, initial_state)

        visited = {(initial_state.state.board.player, frozenset(initial_state.state.board.boxes))}

        previous_state: StatePriority | None = None
        repeated_states = 0

        while queue:
            # Next movement
            current_state: StatePriority = pq.heappop(queue)

            self.states.append(current_state)

            if current_state.state.is_solved():
                self.execution_time = time.perf_counter_ns() - timestamp
                return True

            # Avoid loop
            if current_state == previous_state:
                repeated_states += 1
            else:
                repeated_states = 0

            if repeated_states > self.max_states_repeated:
                self.execution_time = time.perf_counter_ns() - timestamp
                return False

            previous_state = current_state
            # == END == Avoid loop

            player_pos = current_state.state.board.player
            box_positions = current_state.state.board.boxes

            visited.add((player_pos, frozenset(box_positions)))

            possible_moves = current_state.state.get_possible_moves(player_pos)
            possible_states: [StatePriority] = []
            for move in possible_moves:
                possible_move = current_state.state.move(player_pos, move)
                if (
                    possible_move.board.player,
                    frozenset(possible_move.board.boxes),
                ) not in visited:
                    possible_states.append(
                        StatePriority(heuristic(possible_move), possible_move)
                    )

            final_state = None
            for current_state in possible_states:
                if final_state is not None and final_state.priority < current_state.priority:
                    continue
                else:
                    final_state = current_state

            if final_state is None:
                final_state = path.pop()
            pq.heappush(queue, final_state)

            path.append(final_state)

        self.execution_time = time.perf_counter_ns() - timestamp
        return False


if __name__ == "__main__":
    board = Board(
        player=Coordinates(x=3, y=3),
        boxes={Coordinates(y=3, x=4), Coordinates(y=3, x=2)},
        goals={Coordinates(y=3, x=1), Coordinates(y=3, x=5)},
        n_rows=5,
        n_cols=7,
        blocks=[Coordinates(y=2, x=2), Coordinates(y=2, x=4)],
    )

    game = Greedy(board)
    print("Euclidean")
    if game.solve(euclidean):
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")

    for state in game.states:
        # print(state.state.board)
        print(euclidean(state.state), state.state.board.player, state.state.board.boxes)

    game = Greedy(board)
    print("Manhattan")
    if game.solve(manhattan):
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")

    for state in game.states:
        # print(state.state.board)
        print(manhattan(state.state), state.state.board.player, state.state.board.boxes)

    game = Greedy(board)
    print("MMLB")
    if game.solve(minimum_matching_lower_bound):
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")

    for state in game.states:
        # print(state.state.board)
        print(minimum_matching_lower_bound(state.state), state.state.board.player, state.state.board.boxes)