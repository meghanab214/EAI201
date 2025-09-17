import heapq
import math

# --- Grid Utility Functions ---

def parse_grid(grid_str):
    grid = []
    start = goal = None
    for i, row in enumerate(grid_str.strip().split("/")):
        grid_row = []
        for j, cell in enumerate(row.strip().split()):
            grid_row.append(cell)
            if cell == "S":
                start = (i, j)
            elif cell == "G":
                goal = (i, j)
        grid.append(grid_row)
    return grid, start, goal

def print_grid(grid, path=None):
    grid_copy = [row.copy() for row in grid]
    if path:
        for r, c in path[1:-1]:
            grid_copy[r][c] = '*'
    for row in grid_copy:
        print(" ".join(row))

# --- Heuristic Functions ---

def heuristic(a, b, method="manhattan"):
    if method == "manhattan":
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    if method == "euclidean":
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    if method == "diagonal":
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return max(dx, dy)
    return 0

# --- Neighbor Search ---

def neighbors(pos, grid, diagonal=False):
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    if diagonal:
        directions += [(-1,-1), (-1,1), (1,-1), (1,1)]
    res = []
    for dr, dc in directions:
        nr, nc = pos[0]+dr, pos[1]+dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != '1':
            res.append((nr, nc))
    return res

# --- Node Class ---

class Node:
    def __init__(self, pos, g=0, h=0, parent=None):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
    def __lt__(self, other):
        return self.f < other.f

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.pos)
        node = node.parent
    return path[::-1]

# --- Greedy Best-First Search ---

def greedy_bfs(grid, start, goal, heuristic_type="manhattan", diagonal=False):
    heap = []
    start_node = Node(start, g=0, h=heuristic(start, goal, heuristic_type))
    heapq.heappush(heap, (start_node.h, start_node))
    visited = set()
    nodes_explored = 0
    while heap:
        _, current = heapq.heappop(heap)
        nodes_explored += 1
        if current.pos == goal:
            return reconstruct_path(current), nodes_explored
        visited.add(current.pos)
        for npos in neighbors(current.pos, grid, diagonal):
            if npos not in visited:
                n = Node(npos, g=0, h=heuristic(npos, goal, heuristic_type), parent=current)
                heapq.heappush(heap, (n.h, n))
                visited.add(npos)
    return None, nodes_explored

# --- A* Search Algorithm ---

def a_star(grid, start, goal, heuristic_type="manhattan", diagonal=False):
    heap = []
    start_node = Node(start, g=0, h=heuristic(start, goal, heuristic_type))
    heapq.heappush(heap, (start_node.f, start_node))
    visited = set()
    cost_so_far = {start: 0}
    nodes_explored = 0
    while heap:
        _, current = heapq.heappop(heap)
        nodes_explored += 1
        if current.pos == goal:
            return reconstruct_path(current), nodes_explored
        visited.add(current.pos)
        for npos in neighbors(current.pos, grid, diagonal):
            new_cost = cost_so_far[current.pos] + 1
            if npos not in cost_so_far or new_cost < cost_so_far[npos]:
                cost_so_far[npos] = new_cost
                n = Node(npos, g=new_cost, h=heuristic(npos, goal, heuristic_type), parent=current)
                heapq.heappush(heap, (n.f, n))
    return None, nodes_explored

# --- Main Demo with Example Grid ---

example = "S 0 0 1 0 / 1 1 0 1 G / 0 0 0 1 0 / 1 1 0 1 1 / 0 0 0 0 0"
grid, start, goal = parse_grid(example)

print("Original Grid:")
print_grid(grid)
print("\n--- Greedy Best-First Search ---")
for h in ["manhattan", "euclidean", "diagonal"]:
    path, nodes = greedy_bfs(grid, start, goal, heuristic_type=h)
    print(f"\nHeuristic: {h}")
    print("Path:", path)
    print("Path Length:", len(path) if path else "No path")
    print("Nodes Explored:", nodes)
    if path:
        print_grid(grid, path)
print("\n--- A* Search ---")
for h in ["manhattan", "euclidean", "diagonal"]:
    path, nodes = a_star(grid, start, goal, heuristic_type=h)
    print(f"\nHeuristic: {h}")
    print("Path:", path)
    print("Path Length:", len(path) if path else "No path")
    print("Nodes Explored:", nodes)
    if path:
        print_grid(grid, path)
