import random
from collections import Counter, deque

import numpy as np


ORANGE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 8, 9]
MAX_GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _monochrome_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _component_cells(grid == color):
            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return objects


def _ordered_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    coord_set = set(coords)
    neighbors = {
        cell: [
            (cell[0] + dr, cell[1] + dc)
            for dr, dc in DIR4
            if (cell[0] + dr, cell[1] + dc) in coord_set
        ]
        for cell in coords
    }
    degrees = {cell: len(nbrs) for cell, nbrs in neighbors.items()}
    assert all(deg <= 2 for deg in degrees.values()), "object must be branch-free"
    endpoints = sorted([cell for cell, deg in degrees.items() if deg == 1])
    assert len(endpoints) == 2, "object must have exactly two endpoints"

    start = endpoints[0]
    ordered = [start]
    prev = None
    cur = start
    while len(ordered) < len(coords):
        nxts = [cell for cell in neighbors[cur] if cell != prev]
        assert len(nxts) == 1, "path ordering must be unique"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse each foreground object as a direction program, then redraw that program
    from the object's bounding-box bottom-right corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    for obj in _monochrome_objects(g, bg):
        path = _ordered_path(obj["coords"])
        directions = [
            (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            for i in range(len(path) - 1)
        ]
        r, c = obj["bbox"][2], obj["bbox"][3]
        for dr, dc in directions:
            r += dr
            c += dc
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                out[r, c] = obj["color"]
    return out


def _generate_chain(length: int) -> list[tuple[int, int]] | None:
    path = [(0, 0)]
    for _ in range(length - 1):
        cur = path[-1]
        options = []
        used = set(path)
        for dr, dc in DIR4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in used:
                continue
            ok = True
            for adr, adc in DIR4:
                adj = (nxt[0] + adr, nxt[1] + adc)
                if adj in used and adj != cur:
                    ok = False
                    break
            if ok:
                options.append(nxt)
        if not options:
            return None
        path.append(random.choice(options))
    return path


def _expanded(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _projected_cells(path: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    bbox = (min(rows), min(cols), max(rows), max(cols))
    ordered = _ordered_path(path)
    directions = [
        (ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1])
        for i in range(len(ordered) - 1)
    ]
    r, c = bbox[2], bbox[3]
    trace = []
    for dr, dc in directions:
        r += dr
        c += dc
        trace.append((r, c))
    return trace, bbox


def _can_place(
    path: list[tuple[int, int]],
    top: int,
    left: int,
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    blocked_with_margin: set[tuple[int, int]],
    trace_blocked: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    abs_path = [(r + top, c + left) for r, c in path]
    path_set = set(abs_path)
    if any(not (0 <= r < h and 0 <= c < w) for r, c in abs_path):
        return None
    if path_set & blocked:
        return None
    if path_set & blocked_with_margin:
        return None

    trace_rel, _bbox = _projected_cells(abs_path)
    trace_set = set(trace_rel)
    if any(not (0 <= r < h and 0 <= c < w) for r, c in trace_set):
        return None
    if trace_set & path_set:
        return None
    if trace_set & blocked:
        return None
    if trace_set & trace_blocked:
        return None
    return path_set, trace_set


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 20)
    w = random.randint(12, 20)
    grid = np.full((h, w), ORANGE, dtype=int)

    target_count = random.choice([2, 3, 4])
    colors = random.sample(PALETTE, target_count)

    blocked: set[tuple[int, int]] = set()
    blocked_with_margin: set[tuple[int, int]] = set()
    trace_blocked: set[tuple[int, int]] = set()

    for color in colors:
        placed = False
        for _ in range(120):
            length = random.randint(3, 6)
            chain = _generate_chain(length)
            if chain is None:
                continue
            rows = [r for r, _ in chain]
            cols = [c for _, c in chain]
            span_h = max(rows) - min(rows) + 1
            span_w = max(cols) - min(cols) + 1
            norm = [(r - min(rows), c - min(cols)) for r, c in chain]

            max_top = h - span_h
            max_left = w - span_w
            if max_top < 0 or max_left < 0:
                continue
            top = random.randint(0, max_top)
            left = random.randint(0, max_left)

            placed_sets = _can_place(
                norm, top, left, h, w, blocked, blocked_with_margin, trace_blocked
            )
            if placed_sets is None:
                continue
            path_set, trace_set = placed_sets
            for r, c in path_set:
                grid[r, c] = color
            blocked |= path_set
            blocked_with_margin |= _expanded(path_set, h, w, radius=1)
            trace_blocked |= trace_set
            placed = True
            break
        if not placed:
            return None

    objs = _monochrome_objects(grid, ORANGE)
    if len(objs) != target_count:
        return None
    for obj in objs:
        if not (3 <= len(obj["coords"]) <= 6):
            return None
        try:
            _ordered_path(obj["coords"])
        except AssertionError:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
