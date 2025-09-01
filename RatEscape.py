from collections import deque
import heapq

class PipeNetwork:
    def __init__(self, n):
        self.graph = {i: [] for i in range(n)}
        self.n = n

    def add_pipe(self, u, v, cost):
        self.graph[u].append((v, cost))
        self.graph[v].append((u, cost))


# Breadth First Search
def bfs(network, start, goal):
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for neighbor, _ in network.graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


# Depth First Search
def dfs(network, start, goal, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(start)
    path.append(start)

    if start == goal:
        return path

    for neighbor, _ in network.graph[start]:
        if neighbor not in visited:
            result = dfs(network, neighbor, goal, visited, path)
            if result is not None:
                return result

    path.pop()
    return None


# Depth-Limited Search
def depth_limited_search(network, start, goal, limit, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(start)
    path.append(start)

    if start == goal:
        return path

    if limit <= 0:
        path.pop()
        visited.remove(start)
        return None

    for neighbor, _ in network.graph[start]:
        if neighbor not in visited:
            result = depth_limited_search(network, neighbor, goal, limit - 1, visited, path)
            if result is not None:
                return result

    path.pop()
    visited.remove(start)
    return None


# Iterative Deepening Search 
def iterative_deepening_search(network, start, goal, max_depth):
    for depth in range(max_depth + 1):
        path = depth_limited_search(network, start, goal, depth)
        if path is not None:
            return path
    return None


# Uniform Cost Search 
def uniform_cost_search(network, start, goal):
    heap = [(0, start, [start])]  # (cost, node, path)
    visited = {}

    while heap:
        cost, node, path = heapq.heappop(heap)
        if node == goal:
            return path, cost
        if node in visited and visited[node] <= cost:
            continue
        visited[node] = cost
        for neighbor, ncost in network.graph[node]:
            new_cost = cost + ncost
            if neighbor not in visited or new_cost < visited.get(neighbor, float('inf')):
                heapq.heappush(heap, (new_cost, neighbor, path + [neighbor]))
    return None, float('inf')


def main():
    # n = number of junctions
    # m = number of pipes
    n, m = map(int, input("Enter number of junctions and pipes: ").split())
    network = PipeNetwork(n)

    print("Enter pipe connections as: u v cost (0 indexed junctions)")
    for _ in range(m):
        u, v, cost = map(int, input().split())
        network.add_pipe(u, v, cost)

    start = int(input("Enter starting junction: "))
    goal = int(input("Enter goal junction: "))

    print("\nBFS Path (ignores cost):", bfs(network, start, goal))

    print("\nDFS Path (ignores cost):", dfs(network, start, goal))

    limit = int(input("\nEnter depth limit for Depth-Limited Search: "))
    print("Depth-Limited Search Path:", depth_limited_search(network, start, goal, limit))

    max_depth = int(input("\nEnter max depth for Iterative Deepening Search: "))
    print("Iterative Deepening Search Path:", iterative_deepening_search(network, start, goal, max_depth))

    ucs_path, ucs_cost = uniform_cost_search(network, start, goal)
    print(f"\nUniform Cost Search Path: {ucs_path} with total cost {ucs_cost}")


if __name__ == "__main__":
    main()
