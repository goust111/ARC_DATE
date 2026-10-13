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
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

DEPTH_TO_COLOR = {1: GREEN, 2: YELLOW, 3: ORANGE}
INNER_COLORS = [YELLOW, GREY, TEAL, MAROON]
PITCH = 7
SLOT = 5
CANVAS = 30


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    pts = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                pts.append((r, c))
    return pts


def _shell_colors(square: np.ndarray) -> list[int]:
    n = square.shape[0]
    colors = []
    for d in range((n + 1) // 2):
        vals = []
        for r in range(n):
            for c in range(n):
                if min(r, c, n - 1 - r, n - 1 - c) == d:
                    vals.append(int(square[r, c]))
        uniq = sorted(set(vals))
        assert len(uniq) == 1, "node shells must be monochrome"
        colors.append(uniq[0])
    return colors


def _nesting_depth(square: np.ndarray) -> int:
    compact = []
    for color in _shell_colors(square):
        if not compact or compact[-1] != color:
            compact.append(color)
    return len(compact)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    node_comps = _components((g != BG) & (g != PINK))
    assert len(node_comps) >= 5

    nodes = []
    pix_to_node: dict[tuple[int, int], int] = {}
    for idx, coords in enumerate(node_comps):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and 3 <= h <= 5
        square = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(square != BG) and np.all(square != PINK)
        border = _shell_colors(square)[0]
        kind = "terminal" if border == RED else "station"
        depth = 1 if kind == "terminal" else _nesting_depth(square)
        outer = _border_coords(box)
        nodes.append(
            {
                "id": idx,
                "bbox": box,
                "kind": kind,
                "depth": depth,
                "outer": outer,
            }
        )
        for r, c in coords:
            pix_to_node[(int(r), int(c))] = idx

    terminals = [node["id"] for node in nodes if node["kind"] == "terminal"]
    assert len(terminals) == 2

    tracks = []
    adjacency = {node["id"]: [] for node in nodes}
    for edge_id, coords in enumerate(_components(g == PINK)):
        touched = set()
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                node_id = pix_to_node.get((nr, nc))
                if node_id is not None:
                    touched.add(node_id)
        assert len(touched) == 2
        a, b = sorted(touched)
        pts = [(int(r), int(c)) for r, c in coords]
        tracks.append({"id": edge_id, "nodes": (a, b), "pixels": pts})
        adjacency[a].append((b, edge_id))
        adjacency[b].append((a, edge_id))

    start, goal = terminals
    q = deque([start])
    dist = {start: 0}
    count = {start: 1}
    prev_node = {start: None}
    prev_edge = {start: None}

    while q:
        cur = q.popleft()
        for nxt, edge_id in adjacency[cur]:
            nd = dist[cur] + 1
            if nxt not in dist:
                dist[nxt] = nd
                count[nxt] = count[cur]
                prev_node[nxt] = cur
                prev_edge[nxt] = edge_id
                q.append(nxt)
            elif dist[nxt] == nd:
                count[nxt] += count[cur]
    assert goal in dist and count[goal] == 1

    node_path = []
    edge_path = []
    cur = goal
    while cur is not None:
        node_path.append(cur)
        edge = prev_edge[cur]
        if edge is not None:
            edge_path.append(edge)
        cur = prev_node[cur]
    node_path.reverse()
    edge_path.reverse()

    return {
        "nodes": nodes,
        "tracks": tracks,
        "adjacency": adjacency,
        "terminals": terminals,
        "node_path": node_path,
        "edge_path": edge_path,
        "shortest_length": dist[goal],
        "shortest_count": count[goal],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique shortest route in a square-node rail graph, using each path station's nesting depth as its output color.

    Input structure:
    - The grid has a black background, red solid square terminals, blue square station nodes, and pink 1-cell-wide horizontal/vertical track segments.
    - Every station is a filled square whose outer shell is blue; some stations are solid blue, some contain one inner colored square, and some contain two nested inner colored squares.
    - Track segments touch node borders and together form a connected node-track graph with exactly two red terminals.

    Rule:
    - Parse each non-track square as a graph node and each pink track component as a graph edge between the two nodes it touches.
    - Find the unique shortest path, by edge count, between the two red terminals.
    - Recolor every pink track segment on that path to grey.
    - For each blue station on that path, compute its nesting depth from outer shell to center: depth 1 becomes green, depth 2 becomes yellow, depth 3 becomes orange; recolor only the station's outer blue border and leave inner colors unchanged.

    Key insight:
    - The selected objects come from graph shortest-path search, but the replacement color of each selected station is determined independently by the station's own nested square structure.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    out = g.copy()

    node_by_id = {node["id"]: node for node in info["nodes"]}
    track_by_id = {track["id"]: track for track in info["tracks"]}
    terminal_set = set(info["terminals"])

    for node_id in info["node_path"]:
        if node_id in terminal_set:
            continue
        node = node_by_id[node_id]
        color = DEPTH_TO_COLOR[node["depth"]]
        for r, c in node["outer"]:
            if out[r, c] == BLUE:
                out[r, c] = color

    for edge_id in info["edge_path"]:
        for r, c in track_by_id[edge_id]["pixels"]:
            out[r, c] = GREY

    return out


def _graph_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _random_backbone(rng: random.Random, rows: int, cols: int, length: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        start = (rng.randrange(rows), rng.randrange(cols))
        path = [start]
        while len(path) < length + 1:
            options = [nbr for nbr in _graph_neighbors(path[-1], rows, cols) if nbr not in path]
            if not options:
                break
            path.append(rng.choice(options))
        if len(path) == length + 1:
            return path
    return None


def _detour_candidates(path: list[tuple[int, int]], rows: int, cols: int) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    occupied = set(path)
    out = []
    for i, (a, b) in enumerate(zip(path, path[1:], strict=False)):
        ar, ac = a
        br, bc = b
        if ar == br:
            for dr in (-1, 1):
                x = (ar + dr, ac)
                y = (br + dr, bc)
                if 0 <= x[0] < rows and 0 <= y[0] < rows and x not in occupied and y not in occupied:
                    out.append((i, x, y))
        else:
            for dc in (-1, 1):
                x = (ar, ac + dc)
                y = (br, bc + dc)
                if 0 <= x[1] < cols and 0 <= y[1] < cols and x not in occupied and y not in occupied:
                    out.append((i, x, y))
    return out


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a < b else (b, a)


def _shortest_stats(
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[int, int]:
    q = deque([start])
    dist = {start: 0}
    count = {start: 1}
    while q:
        cur = q.popleft()
        for nxt in adjacency[cur]:
            nd = dist[cur] + 1
            if nxt not in dist:
                dist[nxt] = nd
                count[nxt] = count[cur]
                q.append(nxt)
            elif dist[nxt] == nd:
                count[nxt] += count[cur]
    return dist[goal], count[goal]


def _count_simple_paths(
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
    goal: tuple[int, int],
    limit: int = 3,
) -> int:
    total = 0

    def dfs(cur: tuple[int, int], seen: set[tuple[int, int]]):
        nonlocal total
        if total >= limit:
            return
        if cur == goal:
            total += 1
            return
        for nxt in adjacency[cur]:
            if nxt in seen:
                continue
            seen.add(nxt)
            dfs(nxt, seen)
            seen.remove(nxt)

    dfs(start, {start})
    return total


def _choose_size(depth: int, rng: random.Random) -> int:
    if depth == 3:
        return 5
    if depth == 2:
        return rng.choice([4, 5])
    return rng.choice([3, 4, 5])


def _station_sprite(depth: int, size: int, rng: random.Random) -> np.ndarray:
    sprite = np.full((size, size), BLUE, dtype=int)
    if depth == 1:
        return sprite
    inner1 = rng.choice(INNER_COLORS)
    sprite[1:-1, 1:-1] = inner1
    if depth == 2:
        return sprite
    inner2_choices = [color for color in INNER_COLORS if color != inner1]
    sprite[2:-2, 2:-2] = rng.choice(inner2_choices)
    return sprite


def _terminal_sprite(size: int) -> np.ndarray:
    return np.full((size, size), RED, dtype=int)


def _place_node(
    grid: np.ndarray,
    cell: tuple[int, int],
    sprite: np.ndarray,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    size = sprite.shape[0]
    base_r = 2 + cell[0] * PITCH
    base_c = 2 + cell[1] * PITCH
    if size == 3:
        off_r = off_c = 1
    elif size == 4:
        off_r = rng.choice([0, 1])
        off_c = rng.choice([0, 1])
    else:
        off_r = off_c = 0
    r0 = base_r + off_r
    c0 = base_c + off_c
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG)
    grid[r0 : r1 + 1, c0 : c1 + 1] = sprite
    return r0, c0, r1, c1


def _route_edge(
    grid: np.ndarray,
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    pixels = []
    if ar0 <= br1 and br0 <= ar1 and ac1 < bc0:
        row = (max(ar0, br0) + min(ar1, br1)) // 2
        for c in range(ac1 + 1, bc0):
            assert grid[row, c] == BG
            grid[row, c] = PINK
            pixels.append((row, c))
    elif ar0 <= br1 and br0 <= ar1 and bc1 < ac0:
        row = (max(ar0, br0) + min(ar1, br1)) // 2
        for c in range(bc1 + 1, ac0):
            assert grid[row, c] == BG
            grid[row, c] = PINK
            pixels.append((row, c))
    elif ac0 <= bc1 and bc0 <= ac1 and ar1 < br0:
        col = (max(ac0, bc0) + min(ac1, bc1)) // 2
        for r in range(ar1 + 1, br0):
            assert grid[r, col] == BG
            grid[r, col] = PINK
            pixels.append((r, col))
    else:
        assert ac0 <= bc1 and bc0 <= ac1 and br1 < ar0
        col = (max(ac0, bc0) + min(ac1, bc1)) // 2
        for r in range(br1 + 1, ar0):
            assert grid[r, col] == BG
            grid[r, col] = PINK
            pixels.append((r, col))
    assert pixels
    return pixels


def _crop_nonzero(inp: np.ndarray, out: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = max(0, int(r0) - margin)
    c0 = max(0, int(c0) - margin)
    r1 = min(inp.shape[0] - 1, int(r1) + margin)
    c1 = min(inp.shape[1] - 1, int(c1) + margin)
    return inp[r0 : r1 + 1, c0 : c1 + 1].copy(), out[r0 : r1 + 1, c0 : c1 + 1].copy()


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    rows = cols = 4
    backbone = _random_backbone(rng, rows, cols, length=4)
    if backbone is None:
        return None

    detours = _detour_candidates(backbone, rows, cols)
    if not detours:
        return None
    detour_index, x, y = rng.choice(detours)

    occupied = set(backbone)
    occupied.update([x, y])
    edges = {_edge_key(a, b) for a, b in zip(backbone, backbone[1:], strict=False)}
    edges.update({_edge_key(backbone[detour_index], x), _edge_key(x, y), _edge_key(y, backbone[detour_index + 1])})

    extra_leaf_count = rng.choice([0, 1, 2])
    for _ in range(extra_leaf_count):
        candidates = []
        for r in range(rows):
            for c in range(cols):
                cell = (r, c)
                if cell in occupied:
                    continue
                nbrs = [nbr for nbr in _graph_neighbors(cell, rows, cols) if nbr in occupied]
                if len(nbrs) == 1:
                    candidates.append((cell, nbrs[0]))
        if not candidates:
            break
        cell, nbr = rng.choice(candidates)
        occupied.add(cell)
        edges.add(_edge_key(cell, nbr))

    adjacency = {cell: [] for cell in occupied}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    dist, count = _shortest_stats(adjacency, backbone[0], backbone[-1])
    if dist != 4 or count != 1:
        return None
    if _count_simple_paths(adjacency, backbone[0], backbone[-1]) < 2:
        return None

    path_depths = [1, 2, 3]
    rng.shuffle(path_depths)
    node_specs = {}
    for i, cell in enumerate(backbone):
        if i in (0, len(backbone) - 1):
            node_specs[cell] = {"terminal": True, "depth": 1, "size": rng.choice([3, 4, 5])}
        else:
            depth = path_depths[i - 1]
            node_specs[cell] = {"terminal": False, "depth": depth, "size": _choose_size(depth, rng)}
    for cell in occupied:
        if cell in node_specs:
            continue
        depth = rng.choice([1, 2, 3])
        node_specs[cell] = {"terminal": False, "depth": depth, "size": _choose_size(depth, rng)}

    inp = np.zeros((CANVAS, CANVAS), dtype=int)
    expected = np.zeros((CANVAS, CANVAS), dtype=int)
    boxes = {}
    for cell in sorted(occupied):
        spec = node_specs[cell]
        sprite = _terminal_sprite(spec["size"]) if spec["terminal"] else _station_sprite(spec["depth"], spec["size"], rng)
        box = _place_node(inp, cell, sprite, rng)
        boxes[cell] = box
        expected[box[0] : box[2] + 1, box[1] : box[3] + 1] = sprite

    edge_pixels = {}
    for edge in sorted(edges):
        pts = _route_edge(inp, boxes[edge[0]], boxes[edge[1]])
        edge_pixels[edge] = pts
        for r, c in pts:
            expected[r, c] = PINK

    for cell in backbone[1:-1]:
        spec = node_specs[cell]
        color = DEPTH_TO_COLOR[spec["depth"]]
        for r, c in _border_coords(boxes[cell]):
            if expected[r, c] == BLUE:
                expected[r, c] = color
    for edge in zip(backbone, backbone[1:], strict=False):
        for r, c in edge_pixels[_edge_key(*edge)]:
            expected[r, c] = GREY

    margin = rng.choice([1, 2])
    inp, expected = _crop_nonzero(inp, expected, margin)
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None

    info = _analyze_grid(inp)
    path_nodes = [info["nodes"][node_id] for node_id in info["node_path"][1:-1]]
    if len(path_nodes) != 3:
        return None
    if sorted(node["depth"] for node in path_nodes) != [1, 2, 3]:
        return None
    if info["shortest_length"] != 4 or info["shortest_count"] != 1:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _build_case(random.Random())
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
