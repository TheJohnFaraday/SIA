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
from src.Iddfs import Iddfs
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


def solve_iddfs(level: Callable[[], Board]):
    iddfs = Iddfs(level())
    result = iddfs.solve()
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
            executor.submit(run_solver, "IDDFS", solve_iddfs, "", None),
            executor.submit(
                run_solver, "A*", solve_a_star, "Euclidean", heuristics.euclidean
            ),
            executor.submit(
                run_solver, "A*", solve_a_star, "Manhattan", heuristics.manhattan
            ),
            executor.submit(
                run_solver, "A*", solve_a_star, "MMLB", heuristics.minimum_matching_lower_bound
            ),
            executor.submit(
                run_solver, "A*", solve_a_star, "Deadlock", heuristics.deadlock
            ),
            executor.submit(
                run_solver, "A*", solve_a_star, "Euclidean + Deadlock", heuristics.euclidean_plus_deadlock
            ),
            executor.submit(
                run_solver, "A*", solve_a_star, "Euclidean MMLB", heuristics.euclidean_minimum_matching_lower_bound
            ),
            executor.submit(
                run_solver, "Greedy", solve_greedy, "Euclidean", heuristics.euclidean
            ),
            executor.submit(
                run_solver, "Greedy", solve_greedy, "Manhattan", heuristics.manhattan
            ),
            executor.submit(
                run_solver, "Greedy", solve_greedy, "MMLB", heuristics.minimum_matching_lower_bound
            ),
            executor.submit(
                run_solver, "Greedy", solve_greedy, "Deadlock", heuristics.deadlock
            ),
            executor.submit(
                run_solver, "Greedy", solve_greedy, "Euclidean + Deadlock", heuristics.euclidean_plus_deadlock
            ),
            executor.submit(
                run_solver, "Greedy", solve_greedy, "Euclidean MMLB", heuristics.euclidean_minimum_matching_lower_bound
            )
        ]

        for future in as_completed(futures):
            future.result()

    df = pd.DataFrame(results)

    total_execution_time = time.perf_counter_ns() - initial_timestamp

    print(df)
    print(f"Total execution time: {total_execution_time}")
    '''
    average_times_by_method_heuristic = (
        df.groupby(["method", "heuristic", "has_solution", "border_nodes", "path_len"])
        .agg({"execution_time_ns": ["mean", "std"]})
        .reset_index()
    )
    print(average_times_by_method_heuristic)
    '''
    return df


def graph_nodes(df):
    df_to_graph = df[["method", 'heuristic', "border_nodes"]].copy()
    df_to_graph['heuristic'].fillna('No aplica')
    df_to_graph['method_heuristic'] = df_to_graph['method'] + ' (' + df_to_graph['heuristic'] + ')'

    # print(df_to_graph)

    chart = alt.Chart(df_to_graph).mark_bar().encode(
            x=alt.X("method_heuristic:N", title="Algoritmo", sort="x"),
            y=alt.Y("border_nodes:Q", title="Cantidad de nodos expandidos"),
            color=alt.Color("heuristic:N", title="Heurística"),
            tooltip=['method', 'heuristic', 'border_nodes']
        ).properties(width=800, height=400).resolve_scale(y="independent")
    chart.mark_bar()
    chart.mark_text(align="center", baseline="bottom", dx=2)

    chart.show()


def graph_exec_time(df):
    df_to_graph = df[["method", "heuristic", "execution_time_ns"]].copy()
    df_to_graph['heuristic'].fillna('No aplica')
    df_to_graph['method_heuristic'] = df_to_graph['method'] + ' (' + df_to_graph['heuristic'] + ')'
    print(df_to_graph)

    base = alt.Chart(df_to_graph).mark_point().encode(
        x=alt.X("method_heuristic:N", title="Algoritmo", sort="x"),
        y=alt.Y("mean(execution_time_ns):Q", title="Tiempo de ejecución (ns)"),
        color="method_heuristic:N",
    )

    errorbars = alt.Chart(df_to_graph).mark_errorbar(extent="ci").encode(
        x='method_heuristic',
        y=alt.Y("execution_time_ns", title=''),
    )
    chart = base + errorbars
    chart.show()


if __name__ == "__main__":
    df = solve(Levels.level53, times=3)

    alt.renderers.enable("browser")

    graph_nodes(df)
    graph_exec_time(df.copy())
