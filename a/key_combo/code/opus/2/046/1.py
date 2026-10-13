import random
from collections import Counter, deque
import numpy as np


Color = {
    "BLACK": 0,
    "BLUE": 1,
    "RED": 2,
    "GREEN": 3,
    "YELLOW": 4,
    "GREY": 5,
}


def _neigh4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_blocks(grid: np.ndarray):
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    blocks = []
    for r in range(h):
        for c in range(w):
            col = int(grid[r, c])
            if col == Color["BLACK"] or visited[r, c]:
                continue
            comp = []
            q = deque([(r, c)])
            visited[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neigh4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == col:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if len(comp) != 4:
                continue
            rs = [x[0] for x in comp]
            cs = [x[1] for x in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if (r1 - r0 + 1) != 2 or (c1 - c0 + 1) != 2:
                continue
            blocks.append({"color": col, "coords": comp, "r0": r0, "c0": c0, "r1": r1, "c1": c1})
    return blocks


def _mark_block_cells(blocks):
    cell_to_block = {}
    for i, b in enumerate(blocks):
        for r in range(b["r0"], b["r1"] + 1):
            for c in range(b["c0"], b["c1"] + 1):
                cell_to_block[(r, c)] = i
    return cell_to_block


def _block_markers(block):
    r0, c0 = block["r0"], block["c0"]
    return [
        (r0 - 1, c0 - 1, -1, -1),
        (r0 - 1, c0 + 2, -1, 1),
        (r0 + 2, c0 - 1, 1, -1),
        (r0 + 2, c0 + 2, 1, 1),
    ]


def _analyze(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    blocks = _find_blocks(g)
    if len(blocks) < 2:
        return None

    cell_to_block = _mark_block_cells(blocks)
    rays = []
    for sid, b in enumerate(blocks):
        mark_color = int(b["color"])
        for mr, mc, dr, dc in _block_markers(b):
            if 0 <= mr < g.shape[0] and 0 <= mc < g.shape[1] and g[mr, mc] == mark_color:
                path = []
                r, c = mr + dr, mc + dc
                target = None
                while 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                    if (r, c) in cell_to_block:
                        target = int(cell_to_block[(r, c)])
                        break
                    if g[r, c] != Color["BLACK"]:
                        break
                    path.append((int(r), int(c)))
                    r += dr
                    c += dc
                rays.append((sid, target, path))

    terminals = [i for i, b in enumerate(blocks) if int(b["color"]) == Color["RED"]]
    if len(terminals) != 2:
        return None

    adj = {i: [] for i in range(len(blocks))}
    for rid, (s, t, path) in enumerate(rays):
        if t is None:
            continue
        if t == s:
            continue
        if not path:
            continue
        adj[s].append((t, rid))
        adj[t].append((s, rid))

    start, goal = terminals[0], terminals[1]
    q = deque([start])
    prev = {start: None}
    prev_edge = {start: None}
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v, e in adj.get(u, []):
            if v in prev:
                continue
            prev[v] = u
            prev_edge[v] = e
            q.append(v)

    if goal not in prev:
        return None

    path_nodes = []
    path_edges = []
    cur = goal
    while cur is not None:
        path_nodes.append(cur)
        e = prev_edge.get(cur)
        if e is not None:
            path_edges.append(e)
        cur = prev[cur]
    path_nodes = path_nodes[::-1]
    path_edges = path_edges[::-1]

    # count shortest paths to enforce uniqueness
    q = deque([start])
    dist = {start: 0}
    ways = {start: 1}
    while q:
        u = q.popleft()
        for v, _ in adj.get(u, []):
            nd = dist[u] + 1
            if v not in dist:
                dist[v] = nd
                ways[v] = ways[u]
                q.append(v)
            elif nd == dist[v]:
                ways[v] += ways[u]
    if ways.get(goal, 0) != 1:
        return None

    return blocks, rays, path_nodes, set(path_edges), (start, goal)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a graph of 2x2 anchor squares connected by diagonal rays, keep only the shortest path.

    Input structure:
    - Black background.
    - Multiple 2x2 solid colored squares (blue and red), each may have green?
    - Each square has 0-3 diagonal marker pixels in its four corner-adjacent diagonal positions.
    - Rays are cast from markers along the marker's diagonal direction.

    Rule:
    - From each marker, extend a black ray in its 45° away-from-square direction.
    - If a ray reaches any other square before leaving the grid, connect those two squares with a graph edge; otherwise the ray is discarded.
    - Find the unique shortest edge-count path between the two red squares.
    - Recolor non-red path squares to green and path edges to yellow.
    - Erase all non-path rays to black; keep all non-path squares unchanged.

    Key insight:
    - Ray connectivity defines graph topology; only rays that actually hit another square are meaningful edges.
    - The answer is the path induced by shortest-path search, not local ray geometry alone.
    """
    g = np.asarray(grid, dtype=int)
    blocks, rays, path_nodes, path_edge_ids, _ = _analyze(g)

    out = g.copy()
    path_nodes_set = set(path_nodes)

    for i, b in enumerate(blocks):
        if i in path_nodes_set and int(b["color"]) == Color["BLUE"]:
            for r in range(b["r0"], b["r1"] + 1):
                for c in range(b["c0"], b["c1"] + 1):
                    out[r, c] = Color["GREEN"]

    for eid, (_, t, path) in enumerate(rays):
        color = Color["YELLOW"] if eid in path_edge_ids else Color["BLACK"]
        if t is None:
            color = Color["BLACK"]
        if t is not None and eid not in path_edge_ids:
            continue
        for r, c in path:
            out[r, c] = color

    return out


def _random_block_placement(h: int, w: int, n_blocks: int):
    cells = []
    attempts = 0
    while len(cells) < n_blocks and attempts < 2000:
        attempts += 1
        r = random.randint(1, h - 4)
        c = random.randint(1, w - 4)
        if any(abs(r - rr) <= 4 and abs(c - cc) <= 4 for rr, cc in cells):
            continue
        cells.append((r, c))
    return cells if len(cells) == n_blocks else None


def generate_input() -> np.ndarray:
    for _ in range(250):
        h = random.randint(13, 18)
        w = random.randint(13, 18)

        # A simple guaranteed-valid construction:
        # red -> blue -> red with diagonal rays between consecutive blocks.
        g = np.zeros((h, w), dtype=int)

        base_r = random.randint(1, 3)
        base_c = random.randint(1, 3)
        gap1 = random.randint(0, 2)
        gap2 = random.randint(0, 2)
        a = (base_r, base_c)
        b = (base_r + 3 + gap1, base_c + 3 + gap1)
        c = (base_r + 6 + gap1 + gap2, base_c + 6 + gap1 + gap2)

        if c[0] + 1 >= h or c[1] + 1 >= w:
            continue

        cells = [a, b, c]
        colors = [Color["RED"], Color["BLUE"], Color["RED"]]
        for (r, c0), col in zip(cells, colors):
            g[r : r + 2, c0 : c0 + 2] = col

        # Put one diagonal marker on each source of each path edge.
        for (r, c0), col in ((a, Color["RED"]), (b, Color["BLUE"])):
            mr, mc = r + 2, c0 + 2
            if 0 <= mr < h and 0 <= mc < w and g[mr, mc] == Color["BLACK"]:
                g[mr, mc] = col

        if _analyze(g) is None:
            continue

        return g

    # Fallback randomized construction with separation checks.
    for _ in range(200):
        h = random.randint(12, 18)
        w = random.randint(12, 18)
        n_blocks = random.randint(5, 7)
        positions = _random_block_placement(h, w, n_blocks)
        if positions is None:
            continue

        blocks = []
        reds = set(random.sample(range(n_blocks), 2))
        for i, (r, c) in enumerate(positions):
            color = Color["RED"] if i in reds else Color["BLUE"]
            blocks.append((r, c, color))

        g = np.zeros((h, w), dtype=int)
        for r, c, color in blocks:
            g[r : r + 2, c : c + 2] = color

        # add diagonal markers
        for r, c, color in blocks:
            marker_positions = [
                (r - 1, c - 1, -1, -1),
                (r - 1, c + 2, -1, 1),
                (r + 2, c - 1, 1, -1),
                (r + 2, c + 2, 1, 1),
            ]
            k = random.randint(0, 3)
            random.shuffle(marker_positions)
            for mr, mc, dr, dc in marker_positions[:k]:
                if 0 <= mr < h and 0 <= mc < w and g[mr, mc] == Color["BLACK"]:
                    g[mr, mc] = color

        analysis = _analyze(g)
        if analysis is None:
            continue
        blocks_d, rays, path_nodes, path_edges, _ = analysis
        if len(blocks_d) != n_blocks:
            continue
        if not path_edges:
            continue
        return g

    raise RuntimeError("generate_input: failed to sample a valid case for puzzle 1")


if __name__ == "__main__":
    for _ in range(3):
        x = generate_input()
        y = main(x)
        print(x.shape, x.mean(), y.shape)
