import altair as alt
import pandas as pd
import time
import sys
import json

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from pandas import DataFrame

import src.Levels as Levels

from src.Board import Coordinates, Board

from src.AStar import AStar
from src.Bfs import Bfs
from src.Dfs import Dfs
from src.Greedy import Greedy
from src.Iddfs import Iddfs
from src.SearchSolverResult import SearchSolverResult
import src.Heuristics as heuristics

is_random: bool = False
levels: int = 1
seed: int = 9
times: int = 5

is_custom: bool = False
player: Coordinates | None = None
boxes: set[Coordinates] | None = None
goals: set[Coordinates] | None = None
n_rows: int = 0
n_cols: int = 0
blocks: set[Coordinates] | None = None

def graph_theme():
    return {
        'config': {
            'background': '#1b1b1b',  # Color de fondo
            'title': {
                'color': '#FFFFFF'  # Color del título
            },
            'axis': {
                'labelColor': '#FFFFFF',  # Color de las etiquetas del eje
                'titleColor': '#FFFFFF'  # Color del título del eje
            },
            'text': {
                'color': '#FFFFFF'  # Color del texto
            },
            'legend': {
                'labelColor': '#FFFFFF',  # Color de las etiquetas de la leyenda
                'titleColor': '#FFFFFF'  # Color del título de la leyenda
            }
        }
    }

alt.themes.register('custom_theme', graph_theme)
alt.themes.enable('custom_theme')


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
                run_solver,
                "A*", solve_a_star,
                "Euclidean", heuristics.euclidean
            ),
            executor.submit(
                run_solver,
                "A*", solve_a_star,
                "Manhattan", heuristics.manhattan
            ),
            executor.submit(
                run_solver,
                "A*", solve_a_star,
                "MMLB", heuristics.minimum_matching_lower_bound
            ),
            executor.submit(
                run_solver,
                "A*", solve_a_star,
                "Deadlock", heuristics.deadlock
            ),
            executor.submit(
                run_solver,
                "A*", solve_a_star,
                "Euclidean + Deadlock", heuristics.euclidean_plus_deadlock
            ),
            executor.submit(
                run_solver,
                "A*", solve_a_star,
                "Euclidean MMLB",
                heuristics.euclidean_minimum_matching_lower_bound
            ),
            executor.submit(
                run_solver,
                "Greedy", solve_greedy,
                "Euclidean", heuristics.euclidean
            ),
            executor.submit(
                run_solver,
                "Greedy", solve_greedy,
                "Manhattan", heuristics.manhattan
            ),
            executor.submit(
                run_solver,
                "Greedy", solve_greedy,
                "MMLB", heuristics.minimum_matching_lower_bound
            ),
            executor.submit(
                run_solver,
                "Greedy", solve_greedy,
                "Deadlock", heuristics.deadlock
            ),
            executor.submit(
                run_solver,
                "Greedy", solve_greedy,
                "Euclidean + Deadlock", heuristics.euclidean_plus_deadlock
            ),
            executor.submit(
                run_solver,
                "Greedy", solve_greedy,
                "Euclidean MMLB",
                heuristics.euclidean_minimum_matching_lower_bound
            )
        ]

        for future in as_completed(futures):
            future.result()

    df = pd.DataFrame(results)

    # total_execution_time = time.perf_counter_ns() - initial_timestamp

    # print(df)
    # print(f"Total execution time: {total_execution_time}")
    '''
    average_times_by_method_heuristic = (
        df.groupby(["method", "heuristic",
                    "has_solution", "border_nodes", "path_len"])
        .agg({"execution_time_ns": ["mean", "std"]})
        .reset_index()
    )
    print(average_times_by_method_heuristic)
    '''
    return df


def graph_border_nodes(df):
    df_to_graph = df[["method", 'heuristic', "border_nodes"]].copy()
    df_to_graph['method_heuristic'] = df_to_graph.apply(
        lambda row: f"{row['method']}" if row['heuristic'] == '' else f"{row['method']} ({row['heuristic']})",
        axis=1
    )

    chart = alt.Chart(df_to_graph).mark_bar().encode(
            x=alt.X("method_heuristic:N", title="Algoritmo", sort="x"),
            y=alt.Y("mean(border_nodes):Q", title="Cantidad de nodos frontera"),
            color=alt.Color("heuristic:N", title="Heurística"),
            tooltip=['method', 'heuristic', 'border_nodes']
        ).properties(width=800, height=400).resolve_scale(y="independent")
    chart.mark_bar()
    chart.mark_text(align="center", baseline="bottom", dx=2)

    chart.show()


def graph_visited_nodes(df):
    df_to_graph = df[["method", 'heuristic', "nodes_visited"]].copy()
    df_to_graph['method_heuristic'] = df_to_graph.apply(
        lambda row: f"{row['method']}" if row['heuristic'] == '' else f"{row['method']} ({row['heuristic']})",
        axis=1
    )


    # print(df_to_graph)

    chart = alt.Chart(df_to_graph).mark_bar().encode(
            x=alt.X("method_heuristic:N", title="Algoritmo", sort="x"),
            y=alt.Y("mean(nodes_visited):Q",
                    title="Cantidad de nodos expandidos"),
            color=alt.Color("heuristic:N", title="Heurística"),
            tooltip=['method', 'heuristic', 'nodes_visited']
        ).properties(width=800, height=400).resolve_scale(y="independent")
    chart.mark_bar()
    chart.mark_text(align="center", baseline="bottom", dx=2)

    chart.show()


def graph_exec_time(df):
    df_to_graph = df[["method", "heuristic", "execution_time_ns"]].copy()
    df_to_graph['method_heuristic'] = df_to_graph.apply(
        lambda row: f"{row['method']}" if row['heuristic'] == '' else f"{row['method']} ({row['heuristic']})",
        axis=1
    )


    base = alt.Chart(df_to_graph).mark_point().encode(
        x=alt.X("method_heuristic:N", title="Algoritmo", sort="x"),
        y=alt.Y("mean(execution_time_ns):Q", title="Tiempo de ejecución (ns)"),
        color=alt.Color("method_heuristic:N", title="Heurística"),
    )

    errorbars = alt.Chart(df_to_graph).mark_errorbar(color='white', extent="ci").encode(
        x='method_heuristic',
        y=alt.Y("execution_time_ns", title=''),
    )
    chart = base + errorbars
    chart.show()


def graph_path(df):
    df_to_graph = df[["method", 'heuristic', "path_len"]].copy()
    df_to_graph['heuristic'].fillna('No aplica')
    df_to_graph['method_heuristic'] = (df_to_graph['method']
                                       + ' (' + df_to_graph['heuristic'] + ')')

    chart = alt.Chart(df_to_graph).mark_bar().encode(
            x=alt.X("method_heuristic:N", title="Algoritmo", sort="x"),
            y=alt.Y("mean(path_len):Q", title="Cantidad de pasos"),
            color=alt.Color("heuristic:N", title="Heurística"),
            tooltip=['method', 'heuristic', 'path_len']
        ).properties(width=800, height=400).resolve_scale(y="independent")
    chart.mark_bar()
    chart.mark_text(align="center", baseline="bottom", dx=2)

    chart.show()


def custom_level():
    if is_random:
        return Levels.random(seed, level)
    elif is_custom:
        return Board(player=player,
                     boxes=boxes,
                     goals=goals,
                     blocks=blocks,
                     n_rows=n_rows,
                     n_cols=n_cols)
    else:
        return Levels.level53()


if __name__ == "__main__":
    df: DataFrame | None = None
    if len(sys.argv) < 2:
        df = solve(Levels.level53, times=times)
    else:
        with open(f"{sys.argv[1]}", "r") as f:
            config = json.load(f)
            try:
                times = config["times"]
            except KeyError:
                times = 10
            if times is not int or times < 1:
                times = 10
            try:
                is_random = config["random"]["isRandom"]
            except KeyError:
                is_random = False
            try:
                is_custom = config["custom"]["isCustom"]
            except KeyError:
                is_custom = False
            if is_random:
                try:
                    seed = config["random"]["seed"]
                except KeyError:
                    seed = 9
                if seed is not int or seed < 0:
                    seed = 9
                try:
                    level = config["random"]["level"]
                except KeyError:
                    level = 1
                if level is not int or level < 1:
                    level = 1
                print(f'Random!: Seed: {seed}, Level: {level}')
                df = solve(custom_level, times=times)
            elif is_custom:
                try:
                    player = Coordinates(y=config["custom"]["player"].y,
                                         x=config["custom"]["player"].x)
                    for box in config["custom"]["boxes"]:
                        boxes.add(Coordinates(y=box.y,
                                              x=box.x))
                    for goal in config["custom"]["goals"]:
                        goals.add(Coordinates(y=goal.y,
                                              x=goal.x))
                    for block in config["custom"]["blocks"]:
                        block.add(Coordinates(y=block.y,
                                              x=block.x))
                    n_rows = config["custom"]["n_rows"]
                    n_cols = config["custom"]["n_cols"]
                    df = solve(custom_level, times=times)
                except KeyError:
                    df = solve(Levels.level53, times=times)
            else:
                df = solve(Levels.level53, times=times)

    alt.renderers.enable("browser")

    graph_expanded_nodes(df)
    graph_visited_nodes(df)
    graph_exec_time(df)
    graph_path(df)
