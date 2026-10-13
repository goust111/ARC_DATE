from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DRIVER_COLOR = 2
AXIS_COLOR = 9

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
DIR_ORDER = tuple(DIRS)

GEAR_MASK = np.array(
    [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ],
    dtype=bool,
)

GEN_BODY_COLORS = (1, 3, 4, 5, 6, 7, 8)
GEN_MARGIN_RANGE = (1, 4)
GEN_COARSE_SIDE_RANGE = (3, 6)
GEN_DRIVER_SIZE_RANGE = (3, 7)
GEN_EXTRA_COMPONENT_WEIGHTS = ((0, 2), (1, 3), (2, 1))
GEN_EXTRA_SIZE_RANGE = (1, 4)
GEN_EXTRA_EDGE_PROB = 0.3
GEN_ATTEMPTS = 400


def _is_gear_center(g: np.ndarray, r: int, c: int) -> bool:
    if g[r, c] == BG:
        return False
    patch = g[r - 1 : r + 2, c - 1 : c + 2]
    return np.array_equal(patch != BG, GEAR_MASK)


def _find_gears(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    gears: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not _is_gear_center(g, r, c):
                continue

            body = int(g[r, c])
            arm_colors = {name: int(g[r + dr, c + dc]) for name, (dr, dc) in DIRS.items()}
            pointer_dirs = [name for name, color in arm_colors.items() if color != body]
            assert len(pointer_dirs) == 1, "each gear must have exactly one pointer arm"

            pointer_dir = pointer_dirs[0]
            for name, color in arm_colors.items():
                if name == pointer_dir:
                    assert color != body
                else:
                    assert color == body

            gears.append(
                {
                    "center": (r, c),
                    "body": body,
                    "pointer_dir": pointer_dir,
                    "pointer_color": arm_colors[pointer_dir],
                }
            )

    assert gears, "expected at least one gear"
    return gears


def _build_graph(g: np.ndarray, gears: list[dict]) -> list[set[int]]:
    center_to_idx = {gear["center"]: i for i, gear in enumerate(gears)}
    adj = [set() for _ in gears]

    for i, gear in enumerate(gears):
        r, c = gear["center"]
        for dr, dc in DIRS.values():
            other = (r + 5 * dr, c + 5 * dc)
            if other not in center_to_idx:
                continue
            if int(g[r + 2 * dr, c + 2 * dc]) != AXIS_COLOR:
                continue
            if int(g[r + 3 * dr, c + 3 * dc]) != AXIS_COLOR:
                continue
            j = center_to_idx[other]
            adj[i].add(j)
            adj[j].add(i)

    return adj


def _driver_distances(gears: list[dict], adj: list[set[int]]) -> dict[int, int]:
    driver_ids = [i for i, gear in enumerate(gears) if gear["body"] == DRIVER_COLOR]
    assert len(driver_ids) == 1, "expected exactly one red driving gear"

    start = driver_ids[0]
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in adj[cur]:
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            queue.append(nxt)
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the maroon transmission chain from the unique red gear.
    Non-driver gears flip their pointer by 180 degrees iff their graph distance
    from the driver is odd; even-distance gears keep their orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    gears = _find_gears(g)
    adj = _build_graph(g, gears)
    dist = _driver_distances(gears, adj)

    out = g.copy()
    for i, gear in enumerate(gears):
        if i not in dist:
            continue

        new_dir = gear["pointer_dir"]
        if dist[i] % 2 == 1:
            new_dir = OPPOSITE[new_dir]

        r, c = gear["center"]
        body = int(gear["body"])
        pointer_color = int(gear["pointer_color"])

        for dr, dc in DIRS.values():
            out[r + dr, c + dc] = body
        dr, dc = DIRS[new_dir]
        out[r + dr, c + dc] = pointer_color

    return out


def _neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _reachable(
    start: tuple[int, int], available: set[tuple[int, int]], rows: int, cols: int
) -> set[tuple[int, int]]:
    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in _neighbors(cur, rows, cols):
            if nxt not in available or nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def _build_component(
    available: set[tuple[int, int]],
    rows: int,
    cols: int,
    min_size: int,
    max_size: int,
) -> tuple[set[tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]]] | None:
    if not available:
        return None

    start = random.choice(tuple(available))
    reach = _reachable(start, available, rows, cols)
    if len(reach) < min_size:
        return None

    target = random.randint(min_size, min(max_size, len(reach)))
    nodes = {start}
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while len(nodes) < target:
        frontier = []
        for cur in nodes:
            for nxt in _neighbors(cur, rows, cols):
                if nxt in reach and nxt not in nodes:
                    frontier.append((cur, nxt))
        if not frontier:
            break
        cur, nxt = random.choice(frontier)
        nodes.add(nxt)
        edges.add(_edge(cur, nxt))

    if len(nodes) < min_size:
        return None

    all_adjacent = {
        _edge(a, b)
        for a in nodes
        for b in _neighbors(a, rows, cols)
        if b in nodes and a != b
    }
    extras = sorted(all_adjacent - edges)
    for extra in extras:
        if random.random() < GEN_EXTRA_EDGE_PROB:
            edges.add(extra)

    return nodes, edges


def _component_distances(
    nodes: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    start: tuple[int, int],
) -> dict[tuple[int, int], int]:
    adj = {node: set() for node in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    dist = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in adj[cur]:
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            queue.append(nxt)

    assert len(dist) == len(nodes)
    return dist


def _pick_driver(
    nodes: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[tuple[int, int], dict[tuple[int, int], int]] | None:
    candidates = []
    for node in nodes:
        dist = _component_distances(nodes, edges, node)
        has_odd = any(value % 2 == 1 for value in dist.values())
        has_even_nonzero = any(value > 0 and value % 2 == 0 for value in dist.values())
        if has_odd and has_even_nonzero:
            candidates.append((node, dist))
    if not candidates:
        return None
    return random.choice(candidates)


def _center_pixel(
    cell: tuple[int, int], top: int, left: int
) -> tuple[int, int]:
    return top + 1 + 5 * cell[0], left + 1 + 5 * cell[1]


def _render(
    coarse_rows: int,
    coarse_cols: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
    gears: dict[tuple[int, int], dict],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    height = top + 3 + 5 * (coarse_rows - 1) + bottom
    width = left + 3 + 5 * (coarse_cols - 1) + right
    grid = np.full((height, width), BG, dtype=int)

    for cell, gear in gears.items():
        r, c = _center_pixel(cell, top, left)
        body = int(gear["body"])
        grid[r, c] = body
        for dr, dc in DIRS.values():
            grid[r + dr, c + dc] = body
        dr, dc = DIRS[gear["pointer_dir"]]
        grid[r + dr, c + dc] = int(gear["pointer_color"])

    for a, b in edges:
        ar, ac = _center_pixel(a, top, left)
        br, bc = _center_pixel(b, top, left)
        dr = 0 if ar == br else (1 if br > ar else -1)
        dc = 0 if ac == bc else (1 if bc > ac else -1)
        for step in (2, 3):
            rr = ar + dr * step
            cc = ac + dc * step
            assert grid[rr, cc] == BG
            grid[rr, cc] = AXIS_COLOR

    return grid


def _sample_scene() -> tuple[np.ndarray, np.ndarray] | None:
    coarse_rows = random.randint(*GEN_COARSE_SIDE_RANGE)
    coarse_cols = random.randint(*GEN_COARSE_SIDE_RANGE)
    cells = {(r, c) for r in range(coarse_rows) for c in range(coarse_cols)}

    driver_component = _build_component(
        cells,
        coarse_rows,
        coarse_cols,
        GEN_DRIVER_SIZE_RANGE[0],
        min(GEN_DRIVER_SIZE_RANGE[1], len(cells)),
    )
    if driver_component is None:
        return None
    driver_nodes, driver_edges = driver_component

    picked = _pick_driver(driver_nodes, driver_edges)
    if picked is None:
        return None
    driver_cell, driver_dist = picked

    all_edges = set(driver_edges)
    all_nodes = set(driver_nodes)
    remaining = cells - all_nodes

    extra_choices, extra_weights = zip(*GEN_EXTRA_COMPONENT_WEIGHTS, strict=False)
    extra_target = random.choices(extra_choices, weights=extra_weights, k=1)[0]
    extra_count = 0
    while extra_count < extra_target and remaining:
        extra_component = _build_component(
            remaining,
            coarse_rows,
            coarse_cols,
            GEN_EXTRA_SIZE_RANGE[0],
            min(GEN_EXTRA_SIZE_RANGE[1], len(remaining)),
        )
        if extra_component is None:
            break
        nodes, edges = extra_component
        all_nodes |= nodes
        all_edges |= edges
        remaining -= nodes
        extra_count += 1

    if len(all_nodes) > 10:
        return None

    gears: dict[tuple[int, int], dict] = {}
    for cell in all_nodes:
        body = DRIVER_COLOR if cell == driver_cell else random.choice(GEN_BODY_COLORS)
        pointer_dir = random.choice(DIR_ORDER)
        pointer_candidates = [color for color in GEN_BODY_COLORS if color != body]
        pointer_color = random.choice(pointer_candidates)
        gears[cell] = {
            "body": body,
            "pointer_dir": pointer_dir,
            "pointer_color": pointer_color,
        }

    expected_gears = {
        cell: {
            "body": gear["body"],
            "pointer_dir": gear["pointer_dir"],
            "pointer_color": gear["pointer_color"],
        }
        for cell, gear in gears.items()
    }
    for cell, dist in driver_dist.items():
        if dist % 2 == 1:
            expected_gears[cell]["pointer_dir"] = OPPOSITE[expected_gears[cell]["pointer_dir"]]

    top = random.randint(*GEN_MARGIN_RANGE)
    bottom = random.randint(*GEN_MARGIN_RANGE)
    left = random.randint(*GEN_MARGIN_RANGE)
    right = random.randint(*GEN_MARGIN_RANGE)

    inp = _render(coarse_rows, coarse_cols, top, bottom, left, right, gears, all_edges)
    expected = _render(
        coarse_rows,
        coarse_cols,
        top,
        bottom,
        left,
        right,
        expected_gears,
        all_edges,
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _sample_scene()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid gear-transmission input")
