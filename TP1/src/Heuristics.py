import math
from SearchSolver import SearchSolver


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
