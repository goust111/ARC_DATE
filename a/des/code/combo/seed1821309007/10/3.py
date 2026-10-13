import random
from collections import Counter

import numpy as np


AZURE = 8
MAROON = 9
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_BY_STEP = {(-1, 0): "N", (1, 0): "S", (0, -1): "W", (0, 1): "E"}
DELTA_BY_DIR = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}
OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}

GRID_MIN = 10
GRID_MAX = 18
OBJECT_COUNT_RANGE = (3, 6)
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components_of_color(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_objects(grid: np.ndarray, ignore_cells: set[tuple[int, int]]) -> list[dict]:
    objects = []
    for color in np.unique(grid):
        color = int(color)
        if color in (AZURE, MAROON):
            continue
        for coords in _components_of_color(grid == color):
            kept = [(int(r), int(c)) for r, c in coords if (int(r), int(c)) not in ignore_cells]
            if not kept:
                continue
            arr = np.array(kept, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "coord_set": set(kept),
                }
            )
    return objects


def _marker_and_start(grid: np.ndarray, objects: list[dict]) -> tuple[tuple[int, int], int]:
    h, w = grid.shape
    cell_to_obj = {}
    for idx, obj in enumerate(objects):
        for cell in obj["coord_set"]:
            cell_to_obj[cell] = idx

    candidates = []
    for coords in _components_of_color(grid == MAROON):
        if len(coords) != 1:
            continue
        r, c = map(int, coords[0])
        if r not in (0, h - 1) and c not in (0, w - 1):
            continue
        hits = {cell_to_obj[(nr, nc)] for nr, nc in _neighbors4(r, c) if (nr, nc) in cell_to_obj}
        if len(hits) == 1:
            candidates.append(((r, c), next(iter(hits))))
    assert len(candidates) == 1, "expected a unique border maroon marker"
    return candidates[0]


def _build_adjacency(objects: list[dict]) -> list[set[int]]:
    cell_to_obj = {}
    for idx, obj in enumerate(objects):
        for cell in obj["coord_set"]:
            cell_to_obj[cell] = idx

    adjacency = [set() for _ in objects]
    for idx, obj in enumerate(objects):
        for r, c in obj["coord_set"]:
            for nr, nc in _neighbors4(r, c):
                other = cell_to_obj.get((nr, nc))
                if other is not None and other != idx:
                    adjacency[idx].add(other)
    return adjacency


def _traverse_chain(adjacency: list[set[int]], start: int) -> list[int]:
    assert len(adjacency) >= 2
    for nbrs in adjacency:
        assert 1 <= len(nbrs) <= 2

    order = []
    seen = set()
    prev = None
    cur = start
    while True:
        order.append(cur)
        seen.add(cur)
        next_nodes = [n for n in adjacency[cur] if n != prev]
        if not next_nodes:
            break
        assert len(next_nodes) == 1, "chain must be unbranched"
        prev, cur = cur, next_nodes[0]
        assert cur not in seen, "chain must not loop"

    assert len(seen) == len(adjacency), "all objects must belong to one chain"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _find_objects(g, ignore_cells=set())
    marker_cell, start = _marker_and_start(g, objects)
    adjacency = _build_adjacency(objects)
    order = _traverse_chain(adjacency, start)

    original_colors = [objects[idx]["color"] for idx in order]
    out = g.copy()
    out[marker_cell] = AZURE

    for pos, idx in enumerate(order):
        new_color = MAROON if pos == 0 else original_colors[pos - 1]
        coords = objects[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = new_color

    return out


def _candidate_extras(shape: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = []
    for r in (-1, 0, 1):
        for c in (-1, 0, 1):
            if (r, c) in shape or (r, c) == (0, 0):
                continue
            if any((r + dr, c + dc) in shape for dr, dc in DIRS4):
                pts.append((r, c))
    return pts


def _make_shape(required_dirs: list[str], rng: random.Random) -> set[tuple[int, int]]:
    shape = {(0, 0)}
    for d in required_dirs:
        dr, dc = DELTA_BY_DIR[d]
        shape.add((dr, dc))

    max_extras = rng.randint(0, 3)
    for _ in range(max_extras):
        choices = _candidate_extras(shape)
        if not choices:
            break
        shape.add(rng.choice(choices))
    return shape


def _paint_shape(grid: np.ndarray, center: tuple[int, int], shape: set[tuple[int, int]], color: int):
    cr, cc = center
    for dr, dc in shape:
        grid[cr + dr, cc + dc] = color


def _shape_cells(center: tuple[int, int], shape: set[tuple[int, int]]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr, dc in shape}


def _walk_neighbors(node: tuple[int, int], valid: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = node
    out = []
    for dr, dc in DIRS4:
        nxt = (r + 3 * dr, c + 3 * dc)
        if nxt in valid:
            out.append(nxt)
    return out


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    rows = list(range(2, h - 1, 3))
    cols = list(range(2, w - 1, 3))
    if len(rows) < 2 or len(cols) < 2:
        return None

    border_side = rng.choice(["N", "S", "W", "E"])
    if border_side == "N":
        starts = [(rows[0], c) for c in cols]
    elif border_side == "S":
        starts = [(rows[-1], c) for c in cols]
    elif border_side == "W":
        starts = [(r, cols[0]) for r in rows]
    else:
        starts = [(r, cols[-1]) for r in rows]

    valid = {(r, c) for r in rows for c in cols}
    rng.shuffle(starts)
    n_objects = rng.randint(*OBJECT_COUNT_RANGE)

    for start in starts:
        path = [start]
        visited = {start}
        while len(path) < n_objects:
            cur = path[-1]
            candidates = [p for p in _walk_neighbors(cur, valid) if p not in visited]
            rng.shuffle(candidates)
            if not candidates:
                break
            path.append(candidates[0])
            visited.add(candidates[0])
        if len(path) == n_objects:
            break
    else:
        return None

    grid = np.full((h, w), AZURE, dtype=int)
    colors = list(range(8))
    rng.shuffle(colors)
    obj_colors = colors[:n_objects]

    shapes = []
    for i, center in enumerate(path):
        dirs = []
        if i > 0:
            step = ((path[i - 1][0] - center[0]) // 3, (path[i - 1][1] - center[1]) // 3)
            dirs.append(DIR_BY_STEP[step])
        if i + 1 < len(path):
            step = ((path[i + 1][0] - center[0]) // 3, (path[i + 1][1] - center[1]) // 3)
            dirs.append(DIR_BY_STEP[step])
        shape = _make_shape(dirs, rng)
        shapes.append(shape)
        _paint_shape(grid, center, shape, obj_colors[i])

    first_center = path[0]
    marker_r, marker_c = first_center
    if border_side == "N":
        marker_r = 0
    elif border_side == "S":
        marker_r = h - 1
    elif border_side == "W":
        marker_c = 0
    else:
        marker_c = w - 1
    grid[marker_r, marker_c] = MAROON

    objects = _find_objects(grid, ignore_cells=set())
    try:
        marker_cell, start_idx = _marker_and_start(grid, objects)
    except AssertionError:
        return None
    if marker_cell != (marker_r, marker_c) or start_idx != 0:
        return None

    adjacency = _build_adjacency(objects)
    if any(len(nbrs) == 0 or len(nbrs) > 2 for nbrs in adjacency):
        return None

    for a in range(len(objects)):
        for b in range(a + 1, len(objects)):
            should_touch = abs(a - b) == 1
            touches = b in adjacency[a]
            if touches != should_touch:
                return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-color-shift input")
