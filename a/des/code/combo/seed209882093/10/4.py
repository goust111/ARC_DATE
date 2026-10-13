from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

# Asymmetric, 4-connected shapes with a well-defined center pixel at (0, 0).
BASE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (-1, 0), (0, 1)),
    ((0, 0), (-1, 0), (0, 1), (1, 0)),
    ((0, 0), (-1, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (-1, 0), (0, 1), (1, 0), (-1, -1)),
    ((0, 0), (0, -1), (-1, 0), (1, 0), (1, 1)),
    ((0, 0), (-1, 0), (0, 1), (0, -1), (1, 1)),
)

PATH_TEMPLATES = {
    "straight": 0,
    "left_one": 1,
    "right_one": 3,
    "left_twice": 2,
    "right_twice": 2,
    "left_three": 1,
    "right_three": 3,
}

GEN_ATTEMPTS = 400
GEN_OBJECT_COUNT_RANGE = (2, 4)
GEN_HEIGHT_RANGE = (13, 23)
GEN_LEFT_WIDTH_RANGE = (7, 12)
GEN_RIGHT_WIDTH_RANGE = (7, 13)


def _background_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components_of_color(panel: np.ndarray, color: int, col_offset: int = 0) -> list[list[tuple[int, int]]]:
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(panel[r, c]) != color:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc + col_offset)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(panel[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            comps.append(comp)
    return comps


def _find_divider(g: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = g.shape
    candidates = []
    for c in range(w):
        if int(g[0, c]) == bg:
            continue
        if np.all(g[:, c] == g[0, c]):
            candidates.append(c)
    assert len(candidates) == 1, "expected exactly one solid divider column"
    divider_col = int(candidates[0])
    return divider_col, int(g[0, divider_col])


def _ordered_path(coords: list[tuple[int, int]], divider_col: int) -> list[tuple[int, int]]:
    coord_set = set(coords)
    h = max(r for r, _c in coords) + 1
    w = divider_col
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        adjacency[(r, c)] = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if (nr, nc) in coord_set]

    degrees = {pt: len(neis) for pt, neis in adjacency.items()}
    assert max(degrees.values()) <= 2, "path must not branch"
    endpoints = [pt for pt, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    start_candidates = [pt for pt in endpoints if pt[1] == 0]
    end_candidates = [pt for pt in endpoints if pt[1] == divider_col - 1]
    assert len(start_candidates) == 1, "path must start on the left edge"
    assert len(end_candidates) == 1, "path must end next to the divider"

    start = start_candidates[0]
    end = end_candidates[0]
    ordered = [start]
    prev = None
    cur = start
    while cur != end:
        nxts = [pt for pt in adjacency[cur] if pt != prev]
        assert len(nxts) == 1, "path traversal must be unambiguous"
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    assert len(ordered) == len(coords), "path must be a single chain"
    return ordered


def _path_turns(coords: list[tuple[int, int]], divider_col: int) -> list[int]:
    ordered = _ordered_path(coords, divider_col)
    dirs = []
    for (r0, c0), (r1, c1) in zip(ordered, ordered[1:]):
        dr = int(r1 - r0)
        dc = int(c1 - c0)
        assert abs(dr) + abs(dc) == 1, "path must move orthogonally"
        dirs.append((dr, dc))

    turns: list[int] = []
    for d0, d1 in zip(dirs, dirs[1:]):
        if d0 == d1:
            continue
        cross = d0[0] * d1[1] - d0[1] * d1[0]
        assert cross in (-1, 1), "path must turn by 90 degrees"
        turns.append(1 if cross > 0 else -1)
    return turns


def _object_center(coords: list[tuple[int, int]]) -> tuple[int, int]:
    rows = sorted(r for r, _c in coords)
    cols = sorted(c for _r, c in coords)
    row_lo = int(rows[(len(rows) - 1) // 2])
    row_hi = int(rows[len(rows) // 2])
    col_lo = int(cols[(len(cols) - 1) // 2])
    col_hi = int(cols[len(cols) // 2])
    assert row_lo == row_hi and col_lo == col_hi, "object must have a discrete center"
    pivot = (row_lo, col_lo)
    assert pivot in set(coords), "object center must be a filled pixel"
    return pivot


def _rotate_ccw(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _rotate_offsets(offsets: list[tuple[int, int]], quarter_turns_ccw: int) -> list[tuple[int, int]]:
    rotated = list(offsets)
    for _ in range(quarter_turns_ccw % 4):
        rotated = [_rotate_ccw(dr, dc) for dr, dc in rotated]
    return rotated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clear the left panel and rotate each right-panel object according to the
    left/right turn sequence encoded by its matching colored path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    divider_col, divider_color = _find_divider(g, bg)
    assert 0 < divider_col < g.shape[1] - 1

    left = g[:, :divider_col]
    right = g[:, divider_col + 1 :]

    left_colors = {int(v) for v in np.unique(left) if int(v) != bg}
    right_colors = {int(v) for v in np.unique(right) if int(v) != bg}
    assert left_colors == right_colors and left_colors, "path colors and object colors must match"

    out = np.full_like(g, bg)
    out[:, divider_col] = divider_color

    for color in sorted(left_colors):
        path_comps = _components_of_color(left, color)
        obj_comps = _components_of_color(right, color, col_offset=divider_col + 1)
        assert len(path_comps) == 1, "expected one path per color"
        assert len(obj_comps) == 1, "expected one object per color"

        quarter_turns = sum(_path_turns(path_comps[0], divider_col)) % 4
        obj = obj_comps[0]
        pivot = _object_center(obj)
        offsets = [(r - pivot[0], c - pivot[1]) for r, c in obj]
        rotated_offsets = _rotate_offsets(offsets, quarter_turns)

        for dr, dc in rotated_offsets:
            rr = pivot[0] + dr
            cc = pivot[1] + dc
            assert 0 <= rr < g.shape[0] and divider_col + 1 <= cc < g.shape[1]
            assert int(out[rr, cc]) == bg, "rotated objects must stay separate"
            out[rr, cc] = color

    return out


def _draw_polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        assert r0 == r1 or c0 == c1
        segment: list[tuple[int, int]] = []
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            for c in range(c0, c1 + step, step):
                segment.append((r0, c))
        else:
            step = 1 if r1 >= r0 else -1
            for r in range(r0, r1 + step, step):
                segment.append((r, c0))
        if coords:
            segment = segment[1:]
        coords.extend(segment)
    return coords


def _path_points(kind: str, center: int, left_width: int) -> list[tuple[int, int]]:
    end_col = left_width - 1
    if kind == "straight":
        return [(center, 0), (center, end_col)]
    if kind == "left_one":
        return [(center, 0), (center, end_col), (center - 1, end_col)]
    if kind == "right_one":
        return [(center, 0), (center, end_col), (center + 1, end_col)]
    if kind == "left_twice":
        return [(center - 1, 0), (center, 0), (center, end_col), (center - 1, end_col)]
    if kind == "right_twice":
        return [(center + 1, 0), (center, 0), (center, end_col), (center + 1, end_col)]
    turn_col = random.randint(2, left_width - 3)
    if kind == "left_three":
        return [(center, 0), (center, turn_col), (center + 1, turn_col), (center + 1, end_col), (center, end_col)]
    if kind == "right_three":
        return [(center, 0), (center, turn_col), (center - 1, turn_col), (center - 1, end_col), (center, end_col)]
    raise AssertionError(kind)


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        assert int(grid[r, c]) == BLACK
        grid[r, c] = color


def _rotated_shape(shape: tuple[tuple[int, int], ...], quarter_turns_ccw: int) -> list[tuple[int, int]]:
    return _rotate_offsets(list(shape), quarter_turns_ccw)


def _candidate_grid() -> np.ndarray:
    n_objects = random.randint(*GEN_OBJECT_COUNT_RANGE)
    min_h = max(GEN_HEIGHT_RANGE[0], 4 * n_objects + 1)
    h = random.randint(min_h, GEN_HEIGHT_RANGE[1])
    left_width = random.randint(*GEN_LEFT_WIDTH_RANGE)
    right_width = random.randint(*GEN_RIGHT_WIDTH_RANGE)
    w = left_width + 1 + right_width

    grid = np.full((h, w), BLACK, dtype=int)
    grid[:, left_width] = GREY

    colors = random.sample(COLOR_POOL, n_objects)

    band_shift = random.randint(0, h - 4 * n_objects)
    centers = [2 + band_shift + 4 * i for i in range(n_objects)]
    path_kinds = [random.choice(list(PATH_TEMPLATES)) for _ in range(n_objects)]
    if all(PATH_TEMPLATES[k] == 0 for k in path_kinds):
        path_kinds[random.randrange(n_objects)] = random.choice([k for k, q in PATH_TEMPLATES.items() if q != 0])

    for color, center, kind in zip(colors, centers, path_kinds, strict=False):
        _paint(grid, _draw_polyline(_path_points(kind, center, left_width)), color)

    candidate_rows = [r for r in range(2, h - 1, 4)]
    candidate_cols = [c for c in range(left_width + 2, w - 1, 4)]
    slots = [(r, c) for r in candidate_rows for c in candidate_cols]
    assert len(slots) >= n_objects
    pivots = random.sample(slots, n_objects)

    for color, pivot in zip(colors, pivots, strict=False):
        shape = random.choice(BASE_SHAPES)
        rot = random.randrange(4)
        coords = [(pivot[0] + dr, pivot[1] + dc) for dr, dc in _rotated_shape(shape, rot)]
        _paint(grid, coords, color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _candidate_grid()
        try:
            main(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
