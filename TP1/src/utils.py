import time
import functools

from .SearchSolverResult import SearchSolverResult


def measure_exec_time(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        initial_timestamp = time.perf_counter_ns()
        r: SearchSolverResult = f(*args, **kwargs)
        execution_time = time.perf_counter_ns() - initial_timestamp

        return SearchSolverResult(
            has_solution=r.has_solution,
            nodes_visited=r.nodes_visited,
            border_nodes=r.nodes_visited,
            execution_time_ns=execution_time,
        )

    return wrapper
