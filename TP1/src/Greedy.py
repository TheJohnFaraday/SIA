# Greedy chooses based on an heuristic which node to visit
import math
from dataclasses import dataclass, field
from SearchSolver import SearchSolver
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
            self.player_pos = state.state.player_pos
            self.box_positions = state.state.box_positions
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

    #def euclidean(self) -> int:
        px, py = self.player_pos

        player_to_boxes = 0
        for (bx, by) in self.box_positions:
            if (bx, by) in self.goal_positions:
                continue
            else:
                player_to_boxes += sqrt((px - bx)**2 + (py - by)**2)

        boxes_to_goal = 0
        for (bx, by) in self.box_positions:
            # If the box is in a goal position, it shouldn't be counted in the calculation
            if (bx, by) in self.goal_positions:
                continue
            for (gx, gy) in self.goal_positions:
                boxes_to_goal += sqrt((bx - gx)**2 + (by - gy)**2)

        return int(player_to_boxes + boxes_to_goal)
    


def euclidean(state: SearchSolver) -> int:
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

def manhattan(state: SearchSolver) -> int:
    heuristic = 0
    player_x, player_y = state.player_pos

    for box_x, box_y in state.box_positions:
        if (box_x, box_y) in state.goal_positions:
            continue

        # Player to Box distance 
        heuristic += abs(player_x - box_x) + abs(player_y - box_y)

        # Box to goal distance 
        heuristic += min(
            abs(box_x - goal_x) + abs(box_y - goal_y)
            for (goal_x, goal_y) in state.goal_positions
        )
    
    return heuristic

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
    
