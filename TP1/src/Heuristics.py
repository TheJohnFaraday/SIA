import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from SearchSolver import SearchSolver, Coordinates


def euclidean(state: SearchSolver) -> int:
    heuristic = 0
    player = state.board.player

    for box in state.board.boxes:
        if box in state.board.goals:
            continue

        # Player to Box distance
        heuristic += math.sqrt((player.x - box.x) ** 2 + (player.y - box.y) ** 2)

        # Box to Goal distance
        heuristic = sum(
            (
                math.sqrt((box.x - goal.x) ** 2 + abs(box.y - goal.y) ** 2)
                for goal in state.board.goals
            ),
            heuristic,
        )
    return heuristic


def manhattan(state: SearchSolver) -> int:
    heuristic = 0
    player = state.board.player

    for box in state.board.boxes:
        if box in state.board.goals:
            continue

        # Player to Box distance
        heuristic += abs(player.x - box.x) + abs(player.y - box.y)

        # Box to Goal distance
        heuristic += min(
            abs(box.x - goal.x) + abs(box.y - goal.y)
            for goal in state.board.goals
        )

    return heuristic


def minimum_matching_lower_bound(state: SearchSolver) -> int:
    num_boxes = len(state.board.boxes)
    cost_matrix = np.zeros((num_boxes, num_boxes))

    # cost matrix boxes X goals
    for i, box in enumerate(state.board.boxes):
        for j, goal in enumerate(state.board.goals):
            # Manhattan
            cost_matrix[i, j] = abs(box.x - goal.x) + abs(box.y - goal.y)

    # Apply Hungarian method
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_indices, col_indices].sum()

    return int(total_cost)
