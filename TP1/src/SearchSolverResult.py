from dataclasses import dataclass


@dataclass(frozen=True)
class SearchSolverResult:
    has_solution: bool
    nodes_visited: int
    border_nodes: int
    execution_time_ns: int = 0
