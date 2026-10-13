from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

NODE_COLORS = {BLUE, RED}
GRID_CELLS = [(r, c) for r in range(3) for c in range(3)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ORTHO_DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < 3 and 0 <= cc < 3:
            out.append((rr, cc))
    return out


GRID_ADJ = {cell: _neighbors4(cell) for cell in GRID_CELLS}


def _find_color_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            comps.append(
                {
                    "color": color,
                    "pts": pts,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _extract_layout(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    comps = _find_color_components(g, NODE_COLORS)
    nodes = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if len(comp["pts"]) != 4 or r1 - r0 != 1 or c1 - c0 != 1:
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(patch == patch[0, 0]):
            nodes.append({"color": int(patch[0, 0]), "top": int(r0), "left": int(c0)})
    if len(nodes) != 9:
        raise ValueError(f"expected 9 nodes, got {len(nodes)}")

    row_positions = sorted({node["top"] for node in nodes})
    col_positions = sorted({node["left"] for node in nodes})
    if len(row_positions) != 3 or len(col_positions) != 3:
        raise ValueError("nodes do not form a 3x3 grid")
    if len({row_positions[i + 1] - row_positions[i] for i in range(2)}) != 1:
        raise ValueError("row spacing is not regular")
    if len({col_positions[i + 1] - col_positions[i] for i in range(2)}) != 1:
        raise ValueError("col spacing is not regular")

    pos_to_id: dict[tuple[int, int], int] = {}
    id_to_pos: dict[int, tuple[int, int]] = {}
    colors: dict[int, int] = {}
    for node in nodes:
        rr = row_positions.index(node["top"])
        cc = col_positions.index(node["left"])
        nid = rr * 3 + cc
        pos_to_id[(rr, cc)] = nid
        id_to_pos[nid] = (rr, cc)
        colors[nid] = int(node["color"])

    terminals = [nid for nid, color in colors.items() if color == RED]
    if len(terminals) != 2:
        raise ValueError("expected exactly two terminals")

    adj = {nid: set() for nid in range(9)}
    edges = set()
    for rr in range(3):
        for cc in range(3):
            nid = pos_to_id[(rr, cc)]
            top = row_positions[rr]
            left = col_positions[cc]
            if cc < 2:
                right_left = col_positions[cc + 1]
                corridor = g[top : top + 2, left + 2 : right_left]
                if np.any(corridor == YELLOW):
                    other = pos_to_id[(rr, cc + 1)]
                    adj[nid].add(other)
                    adj[other].add(nid)
                    edges.add(tuple(sorted((nid, other))))
            if rr < 2:
                down_top = row_positions[rr + 1]
                corridor = g[top + 2 : down_top, left : left + 2]
                if np.any(corridor == YELLOW):
                    other = pos_to_id[(rr + 1, cc)]
                    adj[nid].add(other)
                    adj[other].add(nid)
                    edges.add(tuple(sorted((nid, other))))

    return {
        "grid": g,
        "row_positions": row_positions,
        "col_positions": col_positions,
        "adj": {nid: sorted(vals) for nid, vals in adj.items()},
        "edges": edges,
        "terminals": sorted(terminals),
        "id_to_pos": id_to_pos,
    }


def _shortest_path(adj: dict[int, list[int]], start: int, goal: int) -> tuple[list[int], int]:
    q = deque([start])
    dist = {start: 0}
    count = {start: 1}
    prev = {start: None}
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            nd = dist[cur] + 1
            if nxt not in dist:
                dist[nxt] = nd
                count[nxt] = count[cur]
                prev[nxt] = cur
                q.append(nxt)
            elif nd == dist[nxt]:
                count[nxt] += count[cur]
    if goal not in dist:
        raise ValueError("terminals are disconnected")
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, int(count[goal])


def _mask_from_path(path: list[int], id_to_pos: dict[int, tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=int)
    for nid in path:
        rr, cc = id_to_pos[nid]
        mask[rr, cc] = 1
    return mask


def _render_output(mask: np.ndarray) -> np.ndarray:
    micro = np.where(mask == 1, GREEN, BG).astype(int)
    expanded = np.zeros((9, 9), dtype=int)
    for rr in range(3):
        for cc in range(3):
            if int(mask[rr, cc]) == 1:
                block = micro
            else:
                block = np.full((3, 3), GREY, dtype=int)
            expanded[rr * 3 : (rr + 1) * 3, cc * 3 : (cc + 1) * 3] = block
    out = np.zeros((11, 11), dtype=int)
    out[1:10, 1:10] = expanded
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the unique shortest terminal-to-terminal route in a node graph into a self-similar mask fractal.

    Input structure:
    - The background is black.
    - Nine 2x2 node blocks sit on a regular 3x3 lattice; exactly two are red terminals and the rest are blue stations.
    - Yellow single-pixel tracks may appear only in the horizontal or vertical corridors between adjacent lattice nodes.

    Rule:
    - Treat the 2x2 blocks as graph vertices and the yellow corridor segments as graph edges.
    - Use BFS to find the shortest path between the two red terminals, then mark the path's node positions inside a 3x3 binary mask.
    - Perform one recursive expansion step: each foreground mask cell becomes a green copy of the full 3x3 mask, while each background mask cell becomes a solid 3x3 grey block.
    - Center the resulting 9x9 pattern inside an 11x11 black output with a one-pixel border.

    Key insight: The graph path is only an intermediate latent object; its node footprint becomes a binary seed that must be expanded again at a larger scale.
    """
    layout = _extract_layout(np.array(grid, dtype=int))
    start, goal = layout["terminals"]
    path, _ = _shortest_path(layout["adj"], start, goal)
    mask = _mask_from_path(path, layout["id_to_pos"])
    return _render_output(mask)


def _enumerate_path_shapes() -> list[tuple[tuple[int, int], ...]]:
    seen: dict[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = {}

    def dfs(path: list[tuple[int, int]]) -> None:
        if 4 <= len(path) <= 8:
            key = tuple(sorted(path))
            seen.setdefault(key, tuple(path))
        if len(path) == 8:
            return
        for nxt in GRID_ADJ[path[-1]]:
            if nxt not in path:
                dfs(path + [nxt])

    for start in GRID_CELLS:
        dfs([start])
    return list(seen.values())


ALL_PATH_SHAPES = _enumerate_path_shapes()
PATH_QUEUE: list[tuple[tuple[int, int], ...]] = []


def _edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))


def _try_add_detour(
    path: tuple[tuple[int, int], ...],
    path_cells: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    rng: random.Random,
) -> set[tuple[int, int]]:
    candidates = []
    for a, b in zip(path, path[1:]):
        if a[0] == b[0]:
            row = a[0]
            c0, c1 = sorted((a[1], b[1]))
            for side in (-1, 1):
                rr = row + side
                x = (rr, c0)
                y = (rr, c1)
                if 0 <= rr < 3 and x not in path_cells and y not in path_cells:
                    candidates.append((a, b, x, y))
        else:
            col = a[1]
            r0, r1 = sorted((a[0], b[0]))
            for side in (-1, 1):
                cc = col + side
                x = (r0, cc)
                y = (r1, cc)
                if 0 <= cc < 3 and x not in path_cells and y not in path_cells:
                    candidates.append((a, b, x, y))
    rng.shuffle(candidates)
    for a, b, x, y in candidates:
        if rng.random() < 0.45:
            continue
        edges.add(_edge(a, x))
        edges.add(_edge(x, y))
        edges.add(_edge(y, b))
        return {x, y}
    return set()


def _add_branch_edges(
    path_cells: set[tuple[int, int]],
    used: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    rng: random.Random,
) -> bool:
    attached = set(path_cells) | set(used)
    remaining = [cell for cell in GRID_CELLS if cell not in attached]
    rng.shuffle(remaining)
    added = False
    made_progress = True
    while remaining and made_progress:
        made_progress = False
        next_remaining = []
        for cell in remaining:
            parents = [nbr for nbr in GRID_ADJ[cell] if nbr in attached]
            if not parents:
                next_remaining.append(cell)
                continue
            if not added or rng.random() < 0.8:
                parent = rng.choice(parents)
                edges.add(_edge(cell, parent))
                attached.add(cell)
                added = True
                made_progress = True
            else:
                next_remaining.append(cell)
        remaining = next_remaining
    return added


def _build_graph_for_path(
    path: tuple[tuple[int, int], ...], rng: random.Random
) -> set[tuple[tuple[int, int], tuple[int, int]]] | None:
    edges = {_edge(a, b) for a, b in zip(path, path[1:])}
    path_cells = set(path)
    detour_cells = _try_add_detour(path, path_cells, edges, rng)
    if not _add_branch_edges(path_cells, detour_cells, edges, rng):
        return None
    return edges


def _paint_edge(
    grid: np.ndarray,
    origin_r: int,
    origin_c: int,
    a: tuple[int, int],
    b: tuple[int, int],
    rng: random.Random,
) -> None:
    ar = origin_r + a[0] * 6
    ac = origin_c + a[1] * 6
    br = origin_r + b[0] * 6
    bc = origin_c + b[1] * 6
    if ar == br:
        lane = rng.randint(0, 1)
        row = ar + lane
        c0, c1 = sorted((ac, bc))
        grid[row, c0 + 2 : c1] = YELLOW
    else:
        lane = rng.randint(0, 1)
        col = ac + lane
        r0, r1 = sorted((ar, br))
        grid[r0 + 2 : r1, col] = YELLOW


def _semantic_check(grid: np.ndarray) -> np.ndarray:
    layout = _extract_layout(grid)
    path, shortest_count = _shortest_path(layout["adj"], layout["terminals"][0], layout["terminals"][1])
    if shortest_count != 1:
        raise ValueError("shortest path is not unique")
    if len(path) - 1 < 3:
        raise ValueError("shortest path is too short")
    if len(layout["edges"]) <= len(path) - 1:
        raise ValueError("graph has no distracting extra edge")
    return _render_output(_mask_from_path(path, layout["id_to_pos"]))


def _next_path_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    global PATH_QUEUE
    if not PATH_QUEUE:
        PATH_QUEUE = list(ALL_PATH_SHAPES)
        rng.shuffle(PATH_QUEUE)
    return PATH_QUEUE.pop()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    path = _next_path_shape(rng)
    edges = _build_graph_for_path(path, rng)
    if edges is None:
        return None

    h = rng.randint(18, 25)
    w = rng.randint(18, 25)
    origin_r = rng.randint(1, h - 15)
    origin_c = rng.randint(1, w - 15)

    grid = np.zeros((h, w), dtype=int)
    path_cells = set(path)
    for rr, cc in GRID_CELLS:
        top = origin_r + rr * 6
        left = origin_c + cc * 6
        color = RED if (rr, cc) in {path[0], path[-1]} else BLUE
        grid[top : top + 2, left : left + 2] = color

    for a, b in sorted(edges):
        _paint_edge(grid, origin_r, origin_c, a, b, rng)

    try:
        expected = _semantic_check(grid)
    except Exception:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.count_nonzero(grid == YELLOW) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to construct a valid sample")
