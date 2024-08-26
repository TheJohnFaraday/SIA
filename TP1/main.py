import altair as alt
import pandas as pd
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import src.Levels as Levels

from src.AStar import AStar
from src.Bfs import Bfs
from src.Board import Board
from src.Dfs import Dfs
from src.Greedy import Greedy
from src.SearchSolverResult import SearchSolverResult
import src.Heuristics as heuristics


def solve_dfs(level: Callable[[], Board]):
    dfs = Dfs(level())
    result = dfs.solve()
    return result


def solve_bfs(level: Callable[[], Board]):
    bfs = Bfs(level())
    result = bfs.solve()
    return result


def solve_greedy(level: Callable[[], Board], heuristic: Callable):
    greedy = Greedy(level())
    result = greedy.solve(heuristic)
    return result


def solve_a_star(level: Callable[[], Board], heuristic: Callable):
    a_star = AStar(level())
    result = a_star.solve(heuristic)
    return result


def solve(level: Callable[[], Board], times: int):
    initial_timestamp = time.perf_counter_ns()
    results = []

    def run_solver(
        solver_name: str,
        solver: (
            Callable[[Callable[[], Board]], SearchSolverResult]
            | Callable[[Callable[[], Board], Callable], SearchSolverResult]
        ),
        heuristic_name: str = "",
        heuristic: Callable = None,
    ):
        for _ in range(times):
            if heuristic:
                result = solver(level, heuristic)
            else:
                result = solver(level)
            results.append(
                {"method": solver_name, "heuristic": heuristic_name, **result.__dict__}
            )

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_solver, "DFS", solve_dfs, "", None),
            executor.submit(run_solver, "BFS", solve_bfs, "", None),
            executor.submit(
                run_solver, "A*", solve_a_star, "Euclidean", heuristics.euclidean
            ),
            executor.submit(
                run_solver, "A*", solve_a_star, "Manhattan", heuristics.manhattan
            ),
            executor.submit(
                run_solver, "Greedy", solve_greedy, "Euclidean", heuristics.euclidean
            ),
            executor.submit(
                run_solver, "Greedy", solve_greedy, "Manhattan", heuristics.manhattan
            ),
        ]

        for future in as_completed(futures):
            future.result()

    df = pd.DataFrame(results)

    total_execution_time = time.perf_counter_ns() - initial_timestamp

    print(df)
    print(f"Total execution time: {total_execution_time}")


if __name__ == "__main__":
    solve(Levels.simple, times=10)
