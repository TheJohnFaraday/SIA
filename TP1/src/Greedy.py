# Greedy chooses based on an heuristic which node to visit
from dataclasses import dataclass, field
from SearchSolver import SearchSolver, Coordinates
from Heuristics import euclidean, manhattan
import heapq as pq


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
    priority: int
    state: SearchSolver = field(compare=False)


class Greedy(SearchSolver):

    def solve(self, heuristic: callable(SearchSolver)):
        # The stack is going to persist our frontier states
        queue = []
        path = []
        pq.heapify(queue)
        pq.heappush(queue, StatePriority(heuristic(self), self))
        self.visited.add((self.player_pos, frozenset(self.box_positions)))

        while queue:
            # Next movement
            state = pq.heappop(queue)
            self.player_pos: Coordinates = state.state.player_pos
            self.box_positions: list[Coordinates] | set[Coordinates] = state.state.box_positions
            self.visited.add((self.player_pos, frozenset(self.box_positions)))
            print(heuristic(self), self.player_pos, self.box_positions)

            if self.is_solved():
                return True

            possible_moves = self.get_possible_moves(self.player_pos)
            possible_states: [StatePriority] = []
            for move in possible_moves:
                possible_move = self.move(self.player_pos, move)
                if (possible_move.player_pos, frozenset(possible_move.box_positions)) not in self.visited:
                    possible_states.append(StatePriority(heuristic(possible_move), possible_move))
            final_state = None
            for state in possible_states:
                if (final_state is not None
                        and final_state.priority < state.priority):
                    continue
                else:
                    final_state = state
            if final_state is None:
                final_state = path.pop()
                # print('Popee un estado viejo owo')
            pq.heappush(queue, final_state)

            path.append(final_state)

        return False


if __name__ == '__main__':
    board = [
        "#######",
        "#     #",
        "# # # #",
        "#X@*@X#",
        "#######"
    ]

    player_pos = Coordinates(y=3, x=3)
    box_positions = [Coordinates(y=3, x=4), Coordinates(y=3, x=2)]
    goal_positions = [Coordinates(y=3, x=1), Coordinates(y=3, x=5)]

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
