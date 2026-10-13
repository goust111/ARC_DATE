import random
from collections import Counter, deque

import numpy as np


BG = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_RETRIES = 400
GRID_SIDE_MIN = 14
GRID_SIDE_MAX = 26
OBJECT_COUNT_OPTIONS = (1, 2, 3, 4)
MAX_INTERIOR_AREA = 6


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _touches_edge(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _boundary_of_region(region: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    boundary: set[tuple[int, int]] = set()
    for r, c in region:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region:
                boundary.add((nr, nc))
    return boundary


def _local_boundary(region: set[tuple[int, int]]) -> set[tuple[int, int]]:
    boundary: set[tuple[int, int]] = set()
    for r, c in region:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nbr = (r + dr, c + dc)
            if nbr not in region:
                boundary.add(nbr)
    return boundary


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each enclosed region by a same-color cross whose arm length is the region area."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg_mask = g == BG
    enclosed = []
    for comp in _components(bg_mask):
        if _touches_edge(comp, h, w):
            continue
        region = set(comp)
        boundary = _boundary_of_region(region, h, w)
        colors = {int(g[r, c]) for r, c in boundary if int(g[r, c]) != BG}
        assert len(colors) == 1, "each enclosed region must have a single boundary color"
        color = next(iter(colors))
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        center_r = (min(rows) + max(rows)) // 2
        center_c = (min(cols) + max(cols)) // 2
        enclosed.append(
            {
                "color": color,
                "area": len(comp),
                "center": (center_r, center_c),
                "boundary": boundary,
            }
        )

    assert enclosed, "expected at least one enclosed region"

    out = g.copy()
    for item in enclosed:
        for r, c in item["boundary"]:
            if out[r, c] == item["color"]:
                out[r, c] = BG

    for item in enclosed:
        color = item["color"]
        arm = item["area"]
        cr, cc = item["center"]
        for d in range(-arm, arm + 1):
            rr = cr + d
            cc2 = cc + d
            if 0 <= rr < h:
                out[rr, cc] = color
            if 0 <= cc2 < w:
                out[cr, cc2] = color

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _make_rect_interior(rng: random.Random) -> set[tuple[int, int]]:
    options = []
    for ih in range(1, 5):
        for iw in range(1, 5):
            area = ih * iw
            if 1 <= area <= MAX_INTERIOR_AREA:
                options.append((ih, iw))
    ih, iw = rng.choice(options)
    return {(r, c) for r in range(ih) for c in range(iw)}


def _make_irregular_interior(rng: random.Random) -> set[tuple[int, int]]:
    target = rng.randint(2, MAX_INTERIOR_AREA)
    cells = {(0, 0)}
    while len(cells) < target:
        r, c = rng.choice(tuple(cells))
        nbrs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        rng.shuffle(nbrs)
        placed = False
        for nr, nc in nbrs:
            cand = set(cells)
            cand.add((nr, nc))
            norm = _normalize(cand)
            height = max(r for r, _ in norm) + 1
            width = max(c for _, c in norm) + 1
            if height <= 4 and width <= 4:
                cells = norm
                placed = True
                break
        if not placed:
            break
    if len(cells) < 2:
        return _make_rect_interior(rng)
    return _normalize(cells)


def _dilate4(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        out.add((r + 1, c))
        out.add((r - 1, c))
        out.add((r, c + 1))
        out.add((r, c - 1))
    return out


def _stampable(
    occupancy: np.ndarray, top: int, left: int, interior: set[tuple[int, int]], boundary: set[tuple[int, int]]
) -> bool:
    h, w = occupancy.shape
    cells = interior | boundary
    placed = {(top + r, left + c) for r, c in cells}
    if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in placed):
        return False
    guard = _dilate4(placed)
    return all(0 <= r < h and 0 <= c < w and not occupancy[r, c] for r, c in guard)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w = rng.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    grid = np.full((h, w), BG, dtype=int)
    occupancy = np.zeros((h, w), dtype=bool)

    object_count = rng.choice(OBJECT_COUNT_OPTIONS)
    colors = rng.sample(PALETTE, object_count)

    for color in colors:
        shape_type = rng.choice(("rect", "irregular"))
        interior = _make_rect_interior(rng) if shape_type == "rect" else _make_irregular_interior(rng)
        boundary = _local_boundary(interior)
        min_r = min(r for r, _ in boundary)
        min_c = min(c for _, c in boundary)
        max_r = max(r for r, _ in boundary)
        max_c = max(c for _, c in boundary)
        interior = {(r - min_r, c - min_c) for r, c in interior}
        boundary = {(r - min_r, c - min_c) for r, c in boundary}
        box_h = max(r for r, _ in boundary) + 1
        box_w = max(c for _, c in boundary) + 1

        placed = False
        if box_h > h - 2 or box_w > w - 2:
            return None

        for _ in range(80):
            top = rng.randint(1, h - box_h - 1)
            left = rng.randint(1, w - box_w - 1)
            if not _stampable(occupancy, top, left, interior, boundary):
                continue
            for r, c in boundary:
                grid[top + r, left + c] = color
            for r, c in _dilate4({(top + r, left + c) for r, c in interior | boundary}):
                if 0 <= r < h and 0 <= c < w:
                    occupancy[r, c] = True
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
