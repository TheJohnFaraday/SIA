# BFS visits all nodes by level
import time

from collections import deque
from SearchSolver import SearchSolver, Coordinates, Board
import Levels


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
    def solve(self):
        timestamp = time.perf_counter_ns()
        self.states = []
        pasos = 0

        initial_state = self

        queue: deque[SearchSolver] = deque([self])
        visited = {(initial_state.board.player, frozenset(initial_state.board.boxes))}

        while queue:
            current_state = queue.popleft()

            self.states.append(current_state)

            if current_state.is_solved():
                self.execution_time = time.perf_counter_ns() - timestamp
                print(f'#### PASOS: {pasos}')
                return True

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
            pasos += 1

        self.execution_time = time.perf_counter_ns() - timestamp
        print(f'#### PASOS: {pasos}')
        return False


if __name__ == "__main__":
    # board = Levels.random(seed=5, level=1)
    board = Levels.simple()
    print(board)

    game = Bfs(board)
    if game.solve():
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")

    for state in game.states:
        print(state.board)

    print(f"Took: {game.execution_time} ns")

    '''
    game = Bfs(board)
    if game.solve():
        print("¡Solución encontrada!")
    else:
        print("No se encontró solución.")

    for state in game.states:
        print(state.board)

    print(f"Took: {game.execution_time} ns")
    '''
