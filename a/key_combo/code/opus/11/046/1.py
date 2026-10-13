from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
TERMINAL = 1
GREEN = 3
TRACK = 5
SEPARATOR = 7
NODE_SIZE = 3
STATION_PALETTE = (2, 4, 6, 8, 9)
GEN_ATTEMPTS = 400


def _neighbors(connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    steps = _neighbors(connectivity)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _find_nodes(top: np.ndarray) -> list[dict]:
    nodes: list[dict] = []
    for color in sorted(int(c) for c in np.unique(top) if int(c) not in (BG, TRACK, SEPARATOR)):
        for coords in _components(top == color, connectivity=8):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if (r1 - r0 + 1, c1 - c0 + 1) != (NODE_SIZE, NODE_SIZE):
                continue
            if int(coords.shape[0]) != NODE_SIZE * NODE_SIZE:
                continue
            center = (int(r0 + 1), int(c0 + 1))
            nodes.append(
                {
                    "id": len(nodes),
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "center": center,
                }
            )
    return nodes


def _analyze_graph(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r] == SEPARATOR)]
    assert len(sep_rows) == 1, "expected one full separator row"
    sep = int(sep_rows[0])

    seed_coords = np.argwhere(g[sep + 1 :] == GREEN)
    assert len(seed_coords) == 1, "expected one lower green seed"
    seed = (int(seed_coords[0, 0] + sep + 1), int(seed_coords[0, 1]))

    top = g[:sep]
    nodes = _find_nodes(top)
    terminals = [node for node in nodes if node["color"] == TERMINAL]
    stations = [node for node in nodes if node["color"] != TERMINAL]
    assert len(terminals) == 2, "expected two blue terminal squares"
    assert 3 <= len(stations) <= 6, "expected 3-6 station squares"

    pix2node: dict[tuple[int, int], int] = {}
    node_by_id: dict[int, dict] = {}
    for node in nodes:
        node_by_id[int(node["id"])] = node
        for r, c in node["coords"]:
            pix2node[(int(r), int(c))] = int(node["id"])

    edges: list[dict] = []
    adjacency: dict[int, list[tuple[int, int]]] = {int(node["id"]): [] for node in nodes}
    for comp in _components(top == TRACK, connectivity=8):
        touched: set[int] = set()
        for r, c in comp:
            for dr, dc in _neighbors(8):
                nr = int(r + dr)
                nc = int(c + dc)
                if (nr, nc) in pix2node:
                    touched.add(pix2node[(nr, nc)])
        assert len(touched) == 2, "each track segment must touch exactly two nodes"
        a, b = sorted(touched)
        edge_id = len(edges)
        edges.append({"id": edge_id, "nodes": (a, b), "coords": comp})
        adjacency[a].append((b, edge_id))
        adjacency[b].append((a, edge_id))

    ordered_terms = sorted(terminals, key=lambda node: (node["center"][0], node["center"][1]))
    start = int(ordered_terms[0]["id"])
    goal = int(ordered_terms[1]["id"])

    dist = {start: 0}
    path_count = {start: 1}
    parent: dict[int, int | None] = {start: None}
    parent_edge: dict[int, int | None] = {start: None}
    q = deque([start])
    while q:
        node = q.popleft()
        for nxt, edge_id in adjacency[node]:
            nd = dist[node] + 1
            if nxt not in dist:
                dist[nxt] = nd
                path_count[nxt] = path_count[node]
                parent[nxt] = node
                parent_edge[nxt] = edge_id
                q.append(nxt)
            elif nd == dist[nxt]:
                path_count[nxt] += path_count[node]

    assert goal in dist, "terminals must be connected"
    assert path_count[goal] == 1, "shortest path must be unique"

    node_path: list[int] = []
    edge_path: list[int] = []
    cur: int | None = goal
    while cur is not None:
        node_path.append(int(cur))
        edge_id = parent_edge[cur]
        if edge_id is not None:
            edge_path.append(int(edge_id))
        cur = parent[cur]
    node_path.reverse()
    edge_path.reverse()

    station_colors = [int(node_by_id[node_id]["color"]) for node_id in node_path[1:-1]]
    return {
        "separator_row": sep,
        "seed": seed,
        "nodes": nodes,
        "stations": stations,
        "terminals": terminals,
        "edges": edges,
        "adjacency": adjacency,
        "node_path": node_path,
        "edge_path": edge_path,
        "station_colors": station_colors,
        "distance": int(dist[goal]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the unique shortest terminal-to-terminal route into a Chebyshev ring palette around the lower seed.

    Input structure:
    - The grid uses black background, a full orange separator row, and one green seed pixel below that separator.
    - Above the separator is a square-and-track graph: exactly two blue 3x3 solid terminal squares, 3-6 other 3x3 solid station squares, and grey 1-cell track segments joining neighboring squares.
    - Station squares may use several visible colors; those colors are the palette candidates consumed by the lower drawing step.

    Rule:
    - Detect the graph in the upper region and choose the row-major earlier blue terminal as the start terminal so the path orientation is unique.
    - Find the unique shortest path, by edge count, from that start terminal to the other blue terminal.
    - Read the intermediate station-square colors along that ordered path, excluding both terminals.
    - Keep the center seed green and paint Chebyshev rings around it: ring d=1..k gets the (d-1)-th path-station color, where k is the number of intermediate stations.
    - Keep the entire upper graph and the separator row unchanged.

    Key insight:
    - The shortest path is not itself repainted; instead its ordered station colors become a one-dimensional code that drives the lower concentric square rings.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_graph(g)
    sep = int(info["separator_row"])
    seed_r, seed_c = info["seed"]
    palette = [GREEN] + list(info["station_colors"])

    out = g.copy()
    rows, cols = np.indices(g.shape)
    distance = np.maximum(np.abs(rows - seed_r), np.abs(cols - seed_c))
    for ring, color in enumerate(palette):
        out[distance == ring] = int(color)
    out[: sep + 1] = g[: sep + 1]
    return out


TEMPLATES = [
    {
        "name": "l2_short",
        "positions": {"A": (0, 0), "P1": (0, 1), "B": (0, 2), "X": (1, 1), "Y": (1, 2)},
        "edges": [("A", "P1"), ("P1", "B"), ("A", "X"), ("X", "Y"), ("Y", "B")],
        "path": ["A", "P1", "B"],
    },
    {
        "name": "l2_long",
        "positions": {
            "A": (0, 0),
            "P1": (0, 1),
            "B": (0, 2),
            "X": (1, 0),
            "Y": (1, 1),
            "Z": (1, 2),
        },
        "edges": [("A", "P1"), ("P1", "B"), ("A", "X"), ("X", "Y"), ("Y", "Z"), ("Z", "B")],
        "path": ["A", "P1", "B"],
    },
    {
        "name": "l3",
        "positions": {
            "A": (0, 0),
            "P1": (0, 1),
            "P2": (0, 2),
            "B": (0, 3),
            "X": (1, 1),
            "Y": (1, 2),
            "Z": (1, 3),
        },
        "edges": [
            ("A", "P1"),
            ("P1", "P2"),
            ("P2", "B"),
            ("A", "X"),
            ("X", "Y"),
            ("Y", "Z"),
            ("Z", "B"),
        ],
        "path": ["A", "P1", "P2", "B"],
    },
    {
        "name": "l4",
        "positions": {
            "A": (0, 0),
            "P1": (0, 1),
            "P2": (0, 2),
            "P3": (0, 3),
            "B": (0, 4),
            "X": (1, 1),
            "Y": (1, 2),
        },
        "edges": [
            ("A", "P1"),
            ("P1", "P2"),
            ("P2", "P3"),
            ("P3", "B"),
            ("P1", "X"),
            ("X", "Y"),
            ("Y", "P3"),
        ],
        "path": ["A", "P1", "P2", "P3", "B"],
    },
    {
        "name": "l5",
        "positions": {
            "A": (0, 0),
            "P1": (0, 1),
            "P2": (0, 2),
            "P3": (0, 3),
            "P4": (0, 4),
            "B": (0, 5),
            "X": (1, 1),
            "Y": (1, 2),
        },
        "edges": [
            ("A", "P1"),
            ("P1", "P2"),
            ("P2", "P3"),
            ("P3", "P4"),
            ("P4", "B"),
            ("P1", "X"),
            ("X", "Y"),
            ("Y", "P3"),
        ],
        "path": ["A", "P1", "P2", "P3", "P4", "B"],
    },
]


def _transform_positions(
    positions: dict[str, tuple[int, int]], mirror_h: bool, mirror_v: bool
) -> dict[str, tuple[int, int]]:
    max_r = max(r for r, _ in positions.values())
    max_c = max(c for _, c in positions.values())
    out: dict[str, tuple[int, int]] = {}
    for name, (r, c) in positions.items():
        rr = max_r - r if mirror_v else r
        cc = max_c - c if mirror_h else c
        out[name] = (int(rr), int(cc))
    return out


def _edge_pixels(a: tuple[int, int], b: tuple[int, int], spacing: int) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dr = 0 if br == ar else (1 if br > ar else -1)
    dc = 0 if bc == ac else (1 if bc > ac else -1)
    start_r = ar + 2 * dr
    start_c = ac + 2 * dc
    end_r = br - 2 * dr
    end_c = bc - 2 * dc
    pixels: list[tuple[int, int]] = []
    rr, cc = start_r, start_c
    while True:
        pixels.append((int(rr), int(cc)))
        if (rr, cc) == (end_r, end_c):
            break
        rr += dr
        cc += dc
    return pixels


def _render_graph(
    template: dict,
    spacing: int,
    left: int,
    top: int,
    mirror_h: bool,
    mirror_v: bool,
    station_colors: dict[str, int],
) -> tuple[np.ndarray, dict[str, tuple[int, int]], list[tuple[int, int]], int, int]:
    positions = _transform_positions(template["positions"], mirror_h=mirror_h, mirror_v=mirror_v)
    max_r = max(r for r, _ in positions.values())
    max_c = max(c for _, c in positions.values())
    graph_h = NODE_SIZE + max_r * spacing
    graph_w = NODE_SIZE + max_c * spacing

    centers: dict[str, tuple[int, int]] = {}
    for name, (cell_r, cell_c) in positions.items():
        centers[name] = (int(top + 1 + cell_r * spacing), int(left + 1 + cell_c * spacing))

    graph = np.full((graph_h, graph_w), BG, dtype=int)
    for name, (r, c) in centers.items():
        color = TERMINAL if name in ("A", "B") else int(station_colors[name])
        graph[r - top - 1 : r - top + 2, c - left - 1 : c - left + 2] = color

    track_pixels: list[tuple[int, int]] = []
    for u, v in template["edges"]:
        pixels = _edge_pixels(centers[u], centers[v], spacing)
        for r, c in pixels:
            graph[r - top, c - left] = TRACK
            track_pixels.append((int(r), int(c)))

    return graph, centers, track_pixels, graph_h, graph_w


def _station_assignment(template: dict, rng: random.Random) -> dict[str, int]:
    station_names = [name for name in template["positions"] if name not in ("A", "B")]
    shortest_names = template["path"][1:-1]
    while True:
        colors = {name: int(rng.choice(STATION_PALETTE)) for name in station_names}
        seq = [colors[name] for name in shortest_names]
        if len(seq) > 1 and len(set(seq)) == 1:
            continue
        if len(seq) > 1 and seq == list(reversed(seq)):
            continue
        return colors


def _simple_path_count(adjacency: dict[str, list[str]], start: str, goal: str) -> int:
    count = 0
    stack = [(start, {start})]
    while stack:
        node, seen = stack.pop()
        if node == goal:
            count += 1
            continue
        for nxt in adjacency[node]:
            if nxt not in seen:
                stack.append((nxt, seen | {nxt}))
    return count


def _pick_seed(
    h: int, w: int, sep: int, radius: int, rng: random.Random
) -> tuple[tuple[int, int], str]:
    clip_modes = ["none", "left", "right", "bottom", "corner"]
    rng.shuffle(clip_modes)
    for mode in clip_modes:
        if mode == "none":
            row_lo = sep + radius + 2
            row_hi = h - radius - 1
            col_lo = radius
            col_hi = w - radius - 1
            if row_lo <= row_hi and col_lo <= col_hi:
                return (rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi)), mode
        elif mode == "left":
            row_lo = sep + radius + 2
            row_hi = h - radius - 1
            if row_lo <= row_hi:
                return (rng.randint(row_lo, row_hi), rng.randint(0, radius - 1)), mode
        elif mode == "right":
            row_lo = sep + radius + 2
            row_hi = h - radius - 1
            if row_lo <= row_hi:
                return (rng.randint(row_lo, row_hi), rng.randint(w - radius, w - 1)), mode
        elif mode == "bottom":
            col_lo = radius
            col_hi = w - radius - 1
            row_lo = max(sep + radius + 2, h - radius)
            row_hi = h - 1
            if row_lo <= row_hi and col_lo <= col_hi:
                return (rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi)), mode
        else:
            row_lo = max(sep + radius + 2, h - radius)
            row_hi = h - 1
            if row_lo <= row_hi:
                side_left = rng.choice([True, False])
                col = rng.randint(0, radius - 1) if side_left else rng.randint(w - radius, w - 1)
                return (rng.randint(row_lo, row_hi), col), mode
    raise RuntimeError("failed to place seed")


def _validate_generated(inp: np.ndarray, template: dict) -> None:
    info = _analyze_graph(inp)
    out = main(inp)
    sep = int(info["separator_row"])
    seed_r, seed_c = info["seed"]
    path_station_count = len(info["station_colors"])
    assert info["distance"] == path_station_count + 1
    assert 2 <= info["distance"] <= 5
    assert out.shape == inp.shape
    assert np.array_equal(out[: sep + 1], inp[: sep + 1])
    assert not np.array_equal(out, inp)

    rows, cols = np.indices(inp.shape)
    cheb = np.maximum(np.abs(rows - seed_r), np.abs(cols - seed_c))
    assert int(out[seed_r, seed_c]) == GREEN
    for ring, color in enumerate(info["station_colors"], start=1):
        ring_mask = cheb == ring
        assert np.any(ring_mask)
        assert np.all(out[ring_mask] == color)
    assert np.all(out[(cheb > path_station_count) & (rows > sep)] == inp[(cheb > path_station_count) & (rows > sep)])

    adjacency: dict[str, list[str]] = {name: [] for name in template["positions"]}
    for u, v in template["edges"]:
        adjacency[u].append(v)
        adjacency[v].append(u)
    assert _simple_path_count(adjacency, "A", "B") >= 2

    seq = info["station_colors"]
    if len(seq) > 1:
        assert len(set(seq)) >= 2
        assert seq != list(reversed(seq))


def _generate_one(rng: random.Random) -> np.ndarray:
    template = rng.choice(TEMPLATES)
    station_colors = _station_assignment(template, rng)
    max_col = max(c for _, c in template["positions"].values())
    spacing_choices = [4, 5] if max_col <= 4 else [4, 5]
    spacing = rng.choice(spacing_choices)
    mirror_h = bool(rng.randint(0, 1))
    mirror_v = bool(rng.randint(0, 1))

    local_positions = _transform_positions(template["positions"], mirror_h=mirror_h, mirror_v=mirror_v)
    graph_h = NODE_SIZE + max(r for r, _ in local_positions.values()) * spacing
    graph_w = NODE_SIZE + max(c for _, c in local_positions.values()) * spacing

    sep_lo = max(graph_h + 1, 8)
    sep_hi = min(13, 29)
    if sep_lo > sep_hi:
        raise RuntimeError("graph too tall")
    sep = rng.randint(sep_lo, sep_hi)

    radius = len(template["path"]) - 2
    h_min = sep + 1 + max(8, 2 * radius + 3)
    h = rng.randint(h_min, 30)
    w_min = max(graph_w + 1, 18)
    w_max = 30
    if w_min > w_max:
        raise RuntimeError("graph too wide")
    w = rng.randint(w_min, w_max)

    top = rng.randint(0, sep - graph_h - 1)
    left = rng.randint(0, w - graph_w)
    grid = np.full((h, w), BG, dtype=int)

    graph, centers, track_pixels, _, _ = _render_graph(
        template,
        spacing=spacing,
        left=left,
        top=top,
        mirror_h=mirror_h,
        mirror_v=mirror_v,
        station_colors=station_colors,
    )
    grid[top : top + graph.shape[0], left : left + graph.shape[1]] = graph
    grid[sep, :] = SEPARATOR

    seed, _clip_mode = _pick_seed(h, w, sep, radius, rng)
    grid[seed] = GREEN

    _validate_generated(grid, template)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            return _generate_one(rng)
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
