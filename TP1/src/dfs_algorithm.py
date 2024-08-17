# Graph must be initialized with vertices amount. Next, edges has to be added.
# DFS visits one entire branch doing backtracking once arrived at the deepest node

class Graph:
    def __init__(self, vertices: int):
        self.vertices = vertices
        self.adj = [[] for _ in range(self.vertices)]

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)

    def dfs_recursive(self, u, visited):
        visited[u] = True

        for neighbour in self.adj[u]:
            if not visited[neighbour]:
                self.dfs_recursive(neighbour, visited)

    def dfs(self, start):
        visited = [False] * self.vertices
        self.dfs_recursive(start, visited)
