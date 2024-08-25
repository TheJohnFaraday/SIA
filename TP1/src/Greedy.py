# Greedy chooses based on an heuristic which node to visit
import heapq as pq
import math

from dataclasses import dataclass, field
from SearchSolver import SearchSolver, Coordinates, State
from Heuristics import (euclidean, manhattan,
                        minimum_matching_lower_bound, trivial)


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
            eq_priority = math.isclose(self.priority,
                                       other.priority,
                                       rel_tol=1e-5)

        return eq_priority and self.state == other.state


class Greedy(SearchSolver):

    def solve(self, heuristic: callable(SearchSolver)):
        # The stack is going to persist our frontier states
        queue = []
        path = []
        pq.heapify(queue)
        pq.heappush(queue, StatePriority(heuristic(self), self))
        self.visited.add(State(self.player_pos, frozenset(self.box_positions)))

        previous_state: StatePriority | None = None
        repeated_states = 0

        pasos = 0

        while queue:
            # Next movement
            # print(f'\n#### PATH: {path}\n')
            state = pq.heappop(queue)
            self.player_pos: Coordinates = state.state.player_pos
            self.box_positions: list[Coordinates] | set[Coordinates] = (
                state.state.box_positions
            )
            self.visited.add(State(self.player_pos,
                                   frozenset(self.box_positions)))
            # print(heuristic(self), self.player_pos, self.box_positions)

            if self.is_solved():
                print(f'### PASOS: {pasos}')
                return True

            if state == previous_state:
                repeated_states += 1
            else:
                repeated_states = 0

            if repeated_states > self.max_states_repeated:
                print(f'### PASOS: {pasos}')
                return False

            previous_state = state

            possible_moves = self.get_possible_moves(self.player_pos)
            possible_states: [StatePriority] = []
            for move in possible_moves:
                possible_move = self.move(self.player_pos, move)
                if (
                    State(possible_move.player_pos,
                          frozenset(possible_move.box_positions))
                ) not in self.visited:
                    possible_states.append(
                        StatePriority(heuristic(possible_move), possible_move)
                    )
            final_state: StatePriority | None = None
            # print(f'##### POSSIBLE STATES: {possible_states}')
            for state in possible_states:
                if (
                    final_state is not None
                    and final_state.priority < state.priority
                        ):
                    continue
                else:
                    final_state = state
            if final_state is None:
                if len(path) < 1:
                    print(f'### PASOS: {pasos}')
                    return False
                else:
                    final_state = path.pop()
                # print('Popee un estado viejo owo')
            else:
                path.append(final_state)
            # print(f'##### FINAL STATE: {final_state}')
            pq.heappush(queue, final_state)
            pasos += 1

        print(f'### PASOS: {pasos}')
        return False


if __name__ == '__main__':
    '''
    board = [
        "#######",
        "#     #",
        "# # # #",
        "#X@*@X#",
        "#######"
    ]
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

    game = Greedy(board, player_pos, box_positions, goal_positions)
    print("Euclidean")
    if game.solve(euclidean):
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")

    game = Greedy(board, player_pos, box_positions, goal_positions)
    print("Manhattan")
    if game.solve(manhattan):
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")

    game = Greedy(board, player_pos, box_positions, goal_positions)
    print("MMLB")
    if game.solve(minimum_matching_lower_bound):
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")

    game = Greedy(board, player_pos, box_positions, goal_positions)
    print("Trivial")
    if game.solve(trivial):
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")
