import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from SearchSolver import SearchSolver, Coordinates


def euclidean(state: SearchSolver) -> int:
    heuristic = 0
    player_x, player_y = state.player_pos

    for box_x, box_y in state.box_positions:
        if Coordinates(box_x, box_y) in state.goal_positions:
            continue

        # Player to Box distance
        heuristic += math.sqrt((player_x - box_x) ** 2 + (player_y - box_y) ** 2)

        # Box to Goal distance
        heuristic = sum(
            (
                math.sqrt((box_x - goal_x) ** 2 + abs(box_y - goal_y) ** 2)
                for goal_x, goal_y in state.goal_positions
            ),
            heuristic,
        )
    return heuristic


def manhattan(state: SearchSolver) -> int:
    heuristic = 0
    player_x, player_y = state.player_pos

    for box_x, box_y in state.box_positions:
        if Coordinates(box_x, box_y) in state.goal_positions:
            continue

        # Player to Box distance
        heuristic += abs(player_x - box_x) + abs(player_y - box_y)

        # Box to Goal distance
        heuristic += min(
            abs(box_x - goal_x) + abs(box_y - goal_y)
            for (goal_x, goal_y) in state.goal_positions
        )

    return heuristic


def minimum_matching_lower_bound(state: SearchSolver) -> int:
    num_boxes = len(state.box_positions)
    cost_matrix = np.zeros((num_boxes, num_boxes))

    # cost matrix boxes X goals
    for i, (box_x, box_y) in enumerate(state.box_positions):
        for j, (goal_x, goal_y) in enumerate(state.goal_positions):
            # Manhattan
            cost_matrix[i, j] = abs(box_x - goal_x) + abs(box_y - goal_y)

    # Apply Hungarian method
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_indices, col_indices].sum()

    return int(total_cost)
