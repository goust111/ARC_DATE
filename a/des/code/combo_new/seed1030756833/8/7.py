import random
from collections import Counter

import numpy as np


GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _trace_grey_path(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = {(int(r), int(c)) for r, c in np.argwhere(grid == GREY)}
    assert path, "expected a grey path"

    def path_neighbors(p: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = p
        return [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in path]

    degrees = {p: len(path_neighbors(p)) for p in path}
    assert all(d <= 2 for d in degrees.values()), "path must be unbranched"
    endpoints = [p for p, d in degrees.items() if d == 1]
    assert len(endpoints) == 2, "path must have two endpoints"
    assert all(r in (0, h - 1) or c in (0, w - 1) for r, c in endpoints), "endpoints must be on the edge"

    start = min(endpoints)
    walk: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    prev = None
    cur = start
    while True:
        walk.append(cur)
        seen.add(cur)
        nxt = [p for p in path_neighbors(cur) if p != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path must not branch"
        prev, cur = cur, nxt[0]
        assert cur not in seen, "path must not loop"

    assert len(walk) == len(path), "all grey pixels must belong to one path"
    return walk


def _turns_from_walk(walk: list[tuple[int, int]]) -> list[tuple[int, int]]:
    turns: list[tuple[int, int]] = []
    for a, b, c in zip(walk, walk[1:], walk[2:]):
        d1 = (b[0] - a[0], b[1] - a[1])
        d2 = (c[0] - b[0], c[1] - b[1])
        if d1 != d2:
            turns.append(b)
    return turns


def _marker_pixels(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    pts: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((grid != bg) & (grid != GREY)):
        pts.append((int(r), int(c), int(grid[int(r), int(c)])))
    return pts


def _nearest_marker(turn: tuple[int, int], markers: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    tr, tc = turn
    return min(markers, key=lambda p: (abs(p[0] - tr) + abs(p[1] - tc), p[0], p[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark every turn of a grey polyline with a nearest-marker-colored cross whose arm length is the turn index.

    Input structure:
    - The background is the dominant color.
    - A single-pixel-wide Grey path is 4-connected, unbranched, and runs from one grid edge to another.
    - The path endpoints are its only degree-one Grey pixels; the start is the endpoint with smaller row, then smaller column.
    - Several non-background, non-Grey singleton marker pixels sit near the path but not on it.

    Rule:
    - Walk the Grey path from the start endpoint and collect pixels where the walking direction changes by 90 degrees.
    - Number those turn pixels in walking order starting at 1.
    - For each turn, choose the nearest colored marker by Manhattan distance, breaking ties by smaller row then smaller column.
    - Draw a four-direction cross centered on the turn in that marker color, with arm length equal to the turn number.
    - Cross pixels are clipped at the grid boundary and are painted only on current background cells, so Grey path and markers remain visible.

    Key insight:
    - The cross size is not locally visible at a turn: it comes from tracing the whole path order, while the color comes from a separate nearest-marker relation.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    walk = _trace_grey_path(g)
    turns = _turns_from_walk(walk)
    assert turns, "expected at least one path turn"

    markers = _marker_pixels(g, bg)
    assert markers, "expected colored marker pixels"

    out = g.copy()
    for k, turn in enumerate(turns, start=1):
        _, _, color = _nearest_marker(turn, markers)
        r, c = turn
        for dr, dc in DIR4:
            for step in range(1, k + 1):
                rr = r + dr * step
                cc = c + dc * step
                if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == int(bg):
                    out[rr, cc] = int(color)
    return out


# ---------------- generator ----------------

GEN_ATTEMPTS = 400
GRID_RANGE = (13, 30)
MARKER_COUNT_RANGE = (2, 4)
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _append_axis_segment(points: list[tuple[int, int]], target: tuple[int, int]) -> None:
    r, c = points[-1]
    tr, tc = target
    dr = 0 if tr == r else (1 if tr > r else -1)
    dc = 0 if tc == c else (1 if tc > c else -1)
    assert (dr == 0) != (dc == 0)
    while (r, c) != (tr, tc):
        r += dr
        c += dc
        points.append((r, c))


def _choose_spaced(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    vals = list(range(lo, hi + 1))
    if len(vals) < count:
        return None
    for _ in range(200):
        picked = sorted(random.sample(vals, count))
        if all(b - a >= gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _next_axis_value(lo: int, hi: int, prev: int) -> int:
    choices = [v for v in range(lo, hi + 1) if abs(v - prev) >= 2]
    return random.choice(choices)


def _build_vertical_path(h: int, w: int) -> list[tuple[int, int]] | None:
    max_pairs = min(4, (h - 5) // 2 + 1)
    if max_pairs < 2 or w < 7:
        return None
    pairs = random.randint(2, max_pairs)
    rows = _choose_spaced(2, h - 3, pairs, 2)
    if rows is None:
        return None

    c = random.randint(2, w - 3)
    points = [(0, c)]
    for r in rows:
        _append_axis_segment(points, (r, c))
        c = _next_axis_value(2, w - 3, c)
        _append_axis_segment(points, (r, c))
    _append_axis_segment(points, (h - 1, c))
    return points


def _build_horizontal_path(h: int, w: int) -> list[tuple[int, int]] | None:
    max_pairs = min(4, (w - 5) // 2 + 1)
    if max_pairs < 2 or h < 7:
        return None
    pairs = random.randint(2, max_pairs)
    cols = _choose_spaced(2, w - 3, pairs, 2)
    if cols is None:
        return None

    r = random.randint(2, h - 3)
    points = [(r, 0)]
    for c in cols:
        _append_axis_segment(points, (r, c))
        r = _next_axis_value(2, h - 3, r)
        _append_axis_segment(points, (r, c))
    _append_axis_segment(points, (r, w - 1))
    return points


def _path_is_clean(points: list[tuple[int, int]], h: int, w: int) -> bool:
    if len(points) != len(set(points)):
        return False
    pts = set(points)
    for r, c in pts:
        if not (0 <= r < h and 0 <= c < w):
            return False
        deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        if deg > 2:
            return False
    return sum(sum((nr, nc) in pts for nr, nc in _neighbors4(r, c)) == 1 for r, c in pts) == 2


def _nearest_assignments(
    turns: list[tuple[int, int]], markers: list[tuple[int, int, int]]
) -> list[int] | None:
    assignments: list[int] = []
    for tr, tc in turns:
        dists = [abs(r - tr) + abs(c - tc) for r, c, _ in markers]
        best = min(dists)
        if dists.count(best) != 1:
            return None
        assignments.append(dists.index(best))
    return assignments


def _path_distance(cell: tuple[int, int], path: set[tuple[int, int]]) -> int:
    r, c = cell
    return min(abs(r - pr) + abs(c - pc) for pr, pc in path)


def _place_markers(
    h: int,
    w: int,
    bg: int,
    path: list[tuple[int, int]],
    turns: list[tuple[int, int]],
) -> list[tuple[int, int, int]] | None:
    color_pool = [c for c in MARKER_COLORS if c != bg]
    marker_count = random.randint(MARKER_COUNT_RANGE[0], min(MARKER_COUNT_RANGE[1], len(turns), len(color_pool)))
    path_set = set(path)

    for _ in range(300):
        selected_turn_ids = random.sample(range(len(turns)), marker_count)
        colors = random.sample(color_pool, marker_count)
        markers: list[tuple[int, int, int]] = []
        used: set[tuple[int, int]] = set()

        for turn_id, color in zip(selected_turn_ids, colors):
            tr, tc = turns[turn_id]
            candidates = []
            for r in range(max(0, tr - 5), min(h, tr + 6)):
                for c in range(max(0, tc - 5), min(w, tc + 6)):
                    cell = (r, c)
                    if cell in path_set or cell in used:
                        continue
                    dt = abs(r - tr) + abs(c - tc)
                    if not (2 <= dt <= 5):
                        continue
                    if _path_distance(cell, path_set) > 5:
                        continue
                    if any(max(abs(r - mr), abs(c - mc)) <= 1 for mr, mc, _ in markers):
                        continue
                    candidates.append(cell)
            if not candidates:
                break
            r, c = random.choice(candidates)
            used.add((r, c))
            markers.append((r, c, color))
        else:
            assignments = _nearest_assignments(turns, markers)
            if assignments is None:
                continue
            if len(set(assignments)) < min(2, marker_count):
                continue
            if not set(range(marker_count)).issubset(set(assignments)):
                continue
            return markers
    return None


def _valid_generated_grid(grid: np.ndarray, bg: int) -> bool:
    if max(grid.shape) > 30 or _background_color(grid) != bg:
        return False
    try:
        walk = _trace_grey_path(grid)
        turns = _turns_from_walk(walk)
        markers = _marker_pixels(grid, bg)
        assignments = _nearest_assignments(turns, markers)
        out = main(grid)
    except Exception:
        return False
    if len(turns) < 4 or len(markers) < 2 or assignments is None:
        return False
    if len(set(assignments)) < 2:
        return False
    return bool(np.any(out != grid))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    bg = random.choice([0, 0, 0, 1, 2, 3, 4, 6, 7, 8, 9])

    builders = [_build_vertical_path, _build_horizontal_path]
    random.shuffle(builders)
    path = None
    for builder in builders:
        path = builder(h, w)
        if path is not None and _path_is_clean(path, h, w):
            break
        path = None
    if path is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for r, c in path:
        grid[r, c] = GREY

    try:
        turns = _turns_from_walk(_trace_grey_path(grid))
    except Exception:
        return None
    if len(turns) < 4:
        return None

    markers = _place_markers(h, w, bg, path, turns)
    if markers is None:
        return None
    for r, c, color in markers:
        grid[r, c] = color

    if not _valid_generated_grid(grid, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid turn-cross path input")
